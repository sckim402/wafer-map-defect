"""D-024 §5의 판정 절차를 실행한다 — 수정판이 결론을 바꾸는가.

실행:
    ./.venv/Scripts/python.exe src/w6_impact.py

순서가 중요하다 (§3-3):
    [A] 수정안이 **귀무 검사(이면군 불변)를 통과하는지 먼저** 본다.
        통과 못 하는 수정안의 성능은 볼 필요가 없다.
    [B] 통과한 것만 전수 재계산 → macro-F1을 seed 0/1/2로 비교
    [C] 중복 제거의 영향 (D-024 §6 뒤집을 조건)

사전 등록한 판정선 (**실행 전에 적는다**):
    macro-F1 변화가 **seed 간 흔들림 ±0.01**(`w5_order.md`)을 넘지 않으면
    「수정해도 결론이 같다」로 기록하고 수정판을 채택한다. 넘으면 그 자체가 결과다.
"""
import sys

import numpy as np
from scipy import ndimage
from sklearn.metrics import f1_score

import config
from edge_band import edge_band, N_WEIGHT_BINS, MIN_FAIL
from shape2 import shape2, MAX_FAIL, DISK
from model8_final import oof, SETS
from w6_checks import load_ordered, six_feats, DIHEDRAL, FEAT_NAMES, K, TOL, N_PER_CLASS

EIG_TOL = 1e-9          # 고유값 동점 판정 (상대)
JITTER = 0.01          # w5_order.md의 seed 간 흔들림 — 사전 등록 판정선


# ══════════════════════════════════════════════════════════
# 수정안
# ══════════════════════════════════════════════════════════
def cv_variant(m, k=K, mode="base", n_bins=N_WEIGHT_BINS, min_fail=MIN_FAIL):
    """circular variance. mode로 bin 부여 방식만 바꾼다 — 나머지는 원본과 동일.

    base  : 원본. edges = linspace(-pi, pi). **theta=+pi와 -pi가 다른 bin으로 간다**
    mod   : theta를 [0, 2pi)로 접는다. +pi와 -pi가 같아져 wrap 동점이 사라진다
    half  : mod + bin 격자를 **반 칸** 돌린다. 축(0, +-90, 180)이 경계에서 벗어난다
    *+c0  : 위에 더해 **중심에 정확히 놓인 점(r=0)을 뺀다.**
            arctan2(0,0)=0이라 그 점만 방위각이 가짜로 0deg가 되고
            **모든 회전에서 혼자 안 움직인다.** 중심에서 방위각은 정의되지 않는다.
    """
    a = np.asarray(m)
    valid = a != config.VAL_OUTSIDE
    if not valid.any():
        return np.nan
    b = edge_band(a, k)
    ii, jj = np.nonzero(b)
    if ii.size == 0:
        return np.nan
    rows, cols = np.nonzero(valid)
    cy, cx = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
    hy = max((rows.max() - rows.min()) / 2, 1e-9)
    hx = max((cols.max() - cols.min()) / 2, 1e-9)
    uu, vv = (ii - cy) / hy, (jj - cx) / hx
    if mode.endswith("+c0"):                 # 중심점은 방위각이 없다 -> 뺀다
        keep = np.hypot(uu, vv) > 0.0
        ii, jj, uu, vv = ii[keep], jj[keep], uu[keep], vv[keep]
        if ii.size == 0:
            return np.nan
    theta = np.arctan2(uu, vv)
    is_fail = a[ii, jj] == config.VAL_FAIL
    if is_fail.sum() < min_fail:
        return np.nan

    two_pi = 2.0 * np.pi
    if mode == "base":
        edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        t_bin = theta
    else:
        shift = (two_pi / n_bins) / 2.0 if mode.startswith("half") else 0.0
        t_bin = np.mod(theta + shift, two_pi)
        edges = np.linspace(0.0, two_pi, n_bins + 1)
    idx = np.clip(np.digitize(t_bin, edges) - 1, 0, n_bins - 1)
    n_per_bin = np.bincount(idx, minlength=n_bins).astype(float)

    f_idx, f_th = idx[is_fail], theta[is_fail]     # 각도는 언제나 원래 theta를 쓴다
    w = 1.0 / n_per_bin[f_idx]
    ws = w.sum()
    if ws <= 0:
        return np.nan
    rbar = np.hypot((w * np.cos(f_th)).sum(), (w * np.sin(f_th)).sum()) / ws
    return 1.0 - rbar


def cc_variant(m, mode="base"):
    """cc_compact. mode='nan'이면 최대 연결성분이 동점일 때 NaN을 반환한다.

    원본은 argmax가 동점을 **래스터 스캔 순서**로 깬다 — 회전과 무관하게 임의값이다.
    """
    if mode == "base":
        return shape2(m)["cc_compact"]
    a = np.asarray(m)
    fail = a == config.VAL_FAIL
    tot = int(fail.sum())
    if tot < 3 or tot > MAX_FAIL:
        return np.nan
    lab, k = ndimage.label(fail, structure=np.ones((3, 3), dtype=int))
    if k == 0:
        return np.nan
    sizes = np.bincount(lab.ravel())[1:]
    n = int(sizes.max())
    if (sizes == n).sum() > 1:          # 동점 -> 정의되지 않는다
        return np.nan
    if n < 3:
        return np.nan
    big = int(np.argmax(sizes)) + 1
    yy, xx = np.nonzero(lab == big)
    P = np.column_stack([xx.astype(float), yy.astype(float)])
    P -= P.mean(0)
    ev, V = np.linalg.eigh(np.cov(P.T))
    # 고유값이 같으면 주축이 유일하지 않다 -> 값이 정의되지 않는다.
    # 1e-12*I를 더해도 두 고유값에 같은 값이 실려 동점은 그대로다.
    # (2026-09-11 외부 검토 반례: 7x9 십자 맵이 90도에서 28.57% 변한다)
    if abs(ev[1] - ev[0]) <= EIG_TOL * max(abs(ev[1]), 1e-300):
        return np.nan
    proj = P @ V[:, 1]
    L = float(proj.max() - proj.min()) + 1.0
    return (n / L) / (DISK * np.sqrt(n))


# ══════════════════════════════════════════════════════════
# [A] 귀무 검사 — 수정안이 이면군 불변을 통과하는가
# ══════════════════════════════════════════════════════════
def invariance_of(fn, maps, pick, label):
    base = np.array([fn(maps[i]) for i in pick])
    worst, nan_mm, n_bad, n_cmp = 0.0, 0, 0, 0
    where = ""
    for nm, tf in DIHEDRAL[1:]:
        cur = np.array([fn(tf(np.asarray(maps[i]))) for i in pick])
        nb, nc = np.isnan(base), np.isnan(cur)
        nan_mm += int((nb != nc).sum())
        ok = ~nb & ~nc
        if not ok.any():
            continue
        rel = np.abs(cur[ok] - base[ok]) / np.maximum(np.abs(base[ok]), 1e-12)
        n_cmp += int(ok.sum())
        n_bad += int((rel > TOL).sum())
        if rel.max() > worst:
            worst, where = float(rel.max()), nm
    # 유한 비교가 한 번도 없었으면 「합격」이 아니라 「판정 불가」다.
    # worst의 초기값 0이 그대로 합격선을 통과해, 전부 NaN을 돌려주는
    # 수정안이 gate를 우회할 수 있었다 (2026-09-12 외부 검토 #5).
    if n_cmp == 0:
        print(f"    {label:<28}{'판정 불가':>13}{'':>10}{'':>9}{nan_mm:>10,}  ⚠ 유한 비교 0건")
        return False
    ok = worst <= TOL and nan_mm == 0
    print(f"    {label:<28}{worst:>13.3e}{where:>10}{n_bad:>9,}{nan_mm:>10,}  "
          f"{'✅' if ok else '❌'}")
    return ok


def stage_a(maps, cls):
    print("=" * 78)
    print("[A] 수정안이 귀무 검사를 통과하는가 — 성능보다 이게 먼저다 (§3-3)")
    print("=" * 78)
    rng = np.random.default_rng(config.SEED)
    pick = np.concatenate([
        rng.choice(np.nonzero(cls == c)[0],
                   min(N_PER_CLASS, int((cls == c).sum())), replace=False)
        for c in config.PATTERN_CLASSES])
    print(f"  표본 {len(pick)}장 x 8 이면군 변환. 합격선 상대오차 < {TOL:g}\n")
    print(f"    {'후보':<28}{'최대 상대오차':>13}{'변환':>10}{'불합격':>9}{'NaN불일치':>10}  판정")
    print("    " + "-" * 72)
    res = {}
    for mode in ("base", "mod", "half", "mod+c0", "half+c0"):
        res[f"cv:{mode}"] = invariance_of(
            lambda m, _m=mode: cv_variant(m, mode=_m), maps, pick, f"circ_var  [{mode}]")
    for mode in ("base", "nan"):
        res[f"cc:{mode}"] = invariance_of(
            lambda m, _m=mode: cc_variant(m, mode=_m), maps, pick, f"cc_compact [{mode}]")
    return res


# ══════════════════════════════════════════════════════════
# [B] 성능 영향
# ══════════════════════════════════════════════════════════
def macro_by_seed(d, keys, y, folds, L):
    out = []
    X = np.column_stack([d[k] for k in keys])
    for s in range(folds.shape[0]):
        p = oof(X, y, folds[s])
        out.append(f1_score(y, p, labels=L, average=None, zero_division=0).mean())
    return np.array(out)


def stage_b(maps, cls, folds, cv_mode, cc_mode):
    print("\n" + "=" * 78)
    print("[B] 성능 영향 — 전수 재계산 후 macro-F1 (seed 0/1/2)")
    print("=" * 78)
    print(f"  사전 등록 판정선: |Δmacro-F1| <= {JITTER} 이면 「결론이 같다」\n")
    d = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        for k in ("cov", "ctr", "cv"):
            d[k] = z[k]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        d["rc"], d["mp"] = z["radial_contrast"], z["mid_peak"]
    with np.load(config.DATA_PROCESSED / "shape2.npz", allow_pickle=True) as z:
        d["cc"] = z["cc_compact"]

    print(f"  재계산 중: circ_var[{cv_mode}] · cc_compact[{cc_mode}] "
          f"({len(maps):,}장)", flush=True)
    cv_new = (np.array([cv_variant(m, mode=cv_mode) for m in maps])
              if cv_mode else d["cv"])
    cc_new = np.array([cc_variant(m, mode=cc_mode) for m in maps])
    print(f"    circ_var  NaN {np.isnan(d['cv']).sum():,} -> {np.isnan(cv_new).sum():,}")
    print(f"    cc_compact NaN {np.isnan(d['cc']).sum():,} -> {np.isnan(cc_new).sum():,}")
    if cv_mode:
        fin = ~np.isnan(d["cv"]) & ~np.isnan(cv_new)
        print(f"    circ_var 값 변화: 최대 |Δ| {np.abs(cv_new[fin] - d['cv'][fin]).max():.4e}"
              f" / 중앙 {np.median(np.abs(cv_new[fin] - d['cv'][fin])):.4e}")

    L = config.PATTERN_CLASSES
    variants = {"base (현행)": dict(d)}
    if cv_mode:
        variants[f"circ_var만 [{cv_mode}]"] = {**d, "cv": cv_new}
    variants[f"cc_compact만 [{cc_mode}]"] = {**d, "cc": cc_new}
    if cv_mode:
        variants["둘 다"] = {**d, "cv": cv_new, "cc": cc_new}
    keys = SETS["6종 (+cc_compact)"]
    print(f"\n  {'변이':<24}{'seed0':>8}{'seed1':>8}{'seed2':>8}{'평균':>9}{'Δ':>9}  판정")
    print("  " + "-" * 74)
    base_mean = None
    rows = {}
    for nm, dd in variants.items():
        v = macro_by_seed(dd, keys, cls, folds, L)
        rows[nm] = v
        if base_mean is None:
            base_mean = v.mean()
            print(f"  {nm:<24}" + "".join(f"{x:>8.3f}" for x in v)
                  + f"{v.mean():>9.3f}{'—':>9}  기준")
        else:
            dlt = v.mean() - base_mean
            print(f"  {nm:<24}" + "".join(f"{x:>8.3f}" for x in v)
                  + f"{v.mean():>9.3f}{dlt:>+9.3f}  "
                  + ("✅ 결론 동일" if abs(dlt) <= JITTER else "❌ 판정선 초과"))
    return rows, cv_new, cc_new


# ══════════════════════════════════════════════════════════
# [C] 중복 제거의 영향
# ══════════════════════════════════════════════════════════
def stage_c(maps, cls, folds):
    import hashlib
    from collections import defaultdict
    print("\n" + "=" * 78)
    print("[C] 중복 제거의 영향 — D-024 §6 뒤집을 조건")
    print("=" * 78)
    buckets = defaultdict(list)
    for i, m in enumerate(maps):
        a = np.ascontiguousarray(m)
        h = hashlib.blake2b(a.tobytes(), digest_size=16)
        h.update(str(a.shape).encode())
        buckets[h.digest()].append(i)
    keep = np.zeros(len(maps), bool)
    for g in buckets.values():
        keep[g[0]] = True                      # 그룹당 첫 장만 남긴다
    print(f"  {len(maps):,}장 -> {keep.sum():,}장 (제거 {(~keep).sum():,}장)")
    print("  ⚠ 표본이 달라지므로 base도 **같은 부분집합에서** 다시 잰다\n")

    d = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        for k in ("cov", "ctr", "cv"):
            d[k] = z[k]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        d["rc"], d["mp"] = z["radial_contrast"], z["mid_peak"]
    with np.load(config.DATA_PROCESSED / "shape2.npz", allow_pickle=True) as z:
        d["cc"] = z["cc_compact"]
    keys = SETS["6종 (+cc_compact)"]
    X = np.column_stack([d[k] for k in keys])
    L = config.PATTERN_CLASSES

    full, dedup = [], []
    for s in range(folds.shape[0]):
        p_full = oof(X, cls, folds[s])                       # 중복 포함 학습
        full.append(f1_score(cls[keep], p_full[keep], labels=L,
                             average=None, zero_division=0).mean())
        p_ded = oof(X[keep], cls[keep], folds[s][keep])      # 중복 제거 학습
        dedup.append(f1_score(cls[keep], p_ded, labels=L,
                              average=None, zero_division=0).mean())
    full, dedup = np.array(full), np.array(dedup)
    print(f"  {'조건':<32}{'seed0':>8}{'seed1':>8}{'seed2':>8}{'평균':>9}")
    print("  " + "-" * 65)
    print(f"  {'중복 포함 학습 (현행)':<32}" + "".join(f"{x:>8.3f}" for x in full)
          + f"{full.mean():>9.3f}")
    print(f"  {'중복 제거 학습':<32}" + "".join(f"{x:>8.3f}" for x in dedup)
          + f"{dedup.mean():>9.3f}")
    dlt = dedup.mean() - full.mean()
    print(f"\n  Δ = {dlt:+.4f}   판정선 ±{JITTER}  ->  "
          + ("✅ 중복 누수는 결론을 바꾸지 않는다" if abs(dlt) <= JITTER
             else "❌ 기존 수치에 중복 보정을 건다"))
    print("  (둘 다 **같은 25,425장**에서 평가했다 — 표본 차이가 아니라 학습 차이다)")
    return full, dedup


def main():
    maps, cls, lot, folds, seeds = load_ordered()
    a = stage_a(maps, cls)
    cv_mode = next((m for m in ("mod+c0", "half+c0", "mod", "half")
                    if a.get(f"cv:{m}")), None)
    cc_mode = "nan" if a.get("cc:nan") else None
    print()
    if cv_mode is None:
        print("  ⚠ circ_var 수정안이 **둘 다 귀무 검사를 통과하지 못했다.**")
        print("    성능을 재는 것은 의미가 없다 — 원인을 다시 판다.")
    if cc_mode is None:
        print("  ⚠ cc_compact 수정안이 귀무 검사를 통과하지 못했다.")
    if cc_mode:
        stage_b(maps, cls, folds, cv_mode, cc_mode)
    stage_c(maps, cls, folds)


if __name__ == "__main__":
    sys.exit(main())
