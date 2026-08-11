"""F1 등면적 분할 — §3-9를 충족시키기 위한 재정의 + 전이 실험.

실행:
    python -u src/radial_equal.py

## 왜 다시 정하나

`docs/pattern_process_mapping.md` §4-1에서 두 가지가 걸렸다.

**걸림 ②**: `R1=1/3, R2=2/3`(반경 3등분)은 **면적이 11% / 33% / 56%로 불균등**하다.
안쪽 밴드의 die 수가 훨씬 적어 **추정 분산이 크고 Laplace 평활에 취약**하다.
그리고 R1·R2 스윕이 단조라 **값 선택이 결과를 바꾼다** — 성능으로 고르면 D-013에서
기각한 것과 같은 짓이 된다.

**걸림 ①**: 그래서 F1a도 크기 4분위별로 **1.67~1.80배** 움직였다
(`edge_contrast`의 2.65배와 같은 기전).

## 등면적 분할 — 성능이 아니라 추정량의 성질로 정한다

단위원을 **면적 3등분**하는 반경:

    r1 = sqrt(1/3) ≈ 0.5774
    r2 = sqrt(2/3) ≈ 0.8165

세 밴드의 die 수가 같아져 **추정 분산이 균등**해지고 평활 편향이 줄어든다.
**이것은 성능 근거가 아니라 구성 타당도 근거다** (§3-9).

**단, 대비는 약해질 수 있다** — 안쪽 밴드가 넓어져 중간 영역이 섞이므로
Center의 중심 집중이 희석된다. **얻는 것과 잃는 것을 둘 다 잰다.**

## 이 스크립트가 판정하는 것

    [1] 등면적 vs 3등분 — 밴드별 die 수가 실제로 균등해지는가 (설계 확인)
    [2] 판별력 — 대비 희석의 대가가 얼마인가
    [3] **크기 의존** — 1.80배가 줄어드는가 ★ 걸림 ①의 해소 여부
    [4] **전이 실험** — 크기 층 간 결정경계가 옮겨가는가 (D-003 분할 위)
"""
import numpy as np

import config
from azimuth import auc
from transfer_size import fit_threshold, balanced_acc

CACHE = config.DATA_PROCESSED / "radial_eq.npz"
R1_EQ, R2_EQ = np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0)
ALPHA = 1.0
N_STRATA = 4


def feats(m, r1, r2, alpha=ALPHA):
    """(radial_contrast, mid_peak, 밴드별 die 수)."""
    a = np.asarray(m)
    valid = a != config.VAL_OUTSIDE
    if not valid.any():
        return np.nan, np.nan, (0, 0, 0)
    rr, cc = np.nonzero(valid)
    cy, cx = (rr.min() + rr.max()) / 2, (cc.min() + cc.max()) / 2
    hy = max((rr.max() - rr.min()) / 2, 1e-9)
    hx = max((cc.max() - cc.min()) / 2, 1e-9)
    r = np.hypot((rr - cy) / hy, (cc - cx) / hx)
    f = a[rr, cc] == config.VAL_FAIL
    masks = (r < r1, (r >= r1) & (r < r2), r >= r2)
    ns = tuple(int(mk.sum()) for mk in masks)
    if min(ns) == 0:
        return np.nan, np.nan, ns
    rate = [(int(f[mk].sum()) + alpha) / (n + 2 * alpha)
            for mk, n in zip(masks, ns)]
    return rate[0] / rate[2], rate[1] / max(rate[0], rate[2]), ns


def build(cls_order, idx_order):
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=True) as z:
            return {k: z[k] for k in z.keys()}
    store = {}
    for c in config.PATTERN_CLASSES:
        with np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True) as z:
            store[c] = z["wafer_maps"]
    n = len(cls_order)
    out = {k: np.full(n, np.nan) for k in
           ("eq_rc", "eq_mp", "n_in", "n_mid", "n_out")}
    for i, (c, j) in enumerate(zip(cls_order, idx_order)):
        rc, mp, ns = feats(store[c][j], R1_EQ, R2_EQ)
        out["eq_rc"][i], out["eq_mp"][i] = rc, mp
        out["n_in"][i], out["n_mid"][i], out["n_out"][i] = ns
        if (i + 1) % 6000 == 0:
            print(f"  {i+1:,}/{n:,}", flush=True)
    np.savez_compressed(CACHE, **out)
    print(f"  캐시 저장: {CACHE}")
    return out


def main():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        cls, idx, F, seeds = z["cls"].astype(str), z["idx_in_cls"], z["folds"], z["seeds"]
    eq = build(cls, idx)
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        old = {k: z[k] for k in z.keys()}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        base = {k: z[k] for k in z.keys()}

    # ── [1] 설계 확인 — 밴드 die 수가 균등해졌는가 ────────
    print("\n" + "=" * 78)
    print("[1] 설계 확인 — 등면적 분할이 실제로 밴드 die 수를 균등하게 하는가")
    print("=" * 78)
    tot = eq["n_in"] + eq["n_mid"] + eq["n_out"]
    ok = tot > 0
    fr = np.column_stack([eq["n_in"][ok] / tot[ok], eq["n_mid"][ok] / tot[ok],
                          eq["n_out"][ok] / tot[ok]])
    print(f"  등면적(r={R1_EQ:.4f}, {R2_EQ:.4f}) 실측 면적 비율 (중앙값):")
    print(f"    안 {np.median(fr[:,0]):.3f} / 중간 {np.median(fr[:,1]):.3f} "
          f"/ 밖 {np.median(fr[:,2]):.3f}   (이상값 0.333씩)")
    print("  3등분(r=0.333, 0.667) 이론 면적 비율: 안 0.111 / 중간 0.333 / 밖 0.556")
    print("  -> 등면적이 0.333에 가까우면 설계대로 작동한 것이다.")

    # ── [2] 판별력 — 대비 희석의 대가 ─────────────────────
    print("\n" + "=" * 78)
    print("[2] 판별력 비교 — 등면적의 대가는 얼마인가 |AUC-0.5|")
    print("=" * 78)
    pairs = [("Center", "Loc"), ("Center", "Edge-Loc"), ("Donut", "Loc"),
             ("Center", "Scratch"), ("Loc", "Scratch")]
    print(f"  {'쌍':<24}{'3등분 F1a':>11}{'등면적 F1a':>12}"
          f"{'3등분 F1b':>11}{'등면적 F1b':>12}")
    for p, n in pairs:
        mp_, mn = cls == p, cls == n
        f = lambda arr: abs(auc(arr[mp_], arr[mn]) - .5)
        print(f"  {p+' ↔ '+n:<24}{f(old['radial_contrast']):>11.3f}"
              f"{f(eq['eq_rc']):>12.3f}{f(old['mid_peak']):>11.3f}"
              f"{f(eq['eq_mp']):>12.3f}")

    # ── [3] 크기 의존 — 걸림 ①이 해소되는가 ★ ────────────
    print("\n" + "=" * 78)
    print("[3] 크기 의존 — 걸림 ①(1.67~1.80배)이 줄어드는가")
    print("=" * 78)
    size = base["size"]
    e = np.quantile(size, np.linspace(0, 1, N_STRATA + 1)); e[0], e[-1] = -np.inf, np.inf
    print(f"  {'클래스':<12}{'3등분 최대/최소':>18}{'등면적 최대/최소':>18}")
    for c in ("Center", "Loc", "Edge-Ring", "Scratch", "Donut"):
        m = cls == c
        out = []
        for arr in (old["radial_contrast"], eq["eq_rc"]):
            v = [np.nanmedian(arr[m & (size >= e[s]) & (size < e[s + 1])])
                 for s in range(N_STRATA)]
            v = np.array([x for x in v if np.isfinite(x)])
            out.append(v.max() / max(v.min(), 1e-9) if len(v) > 1 else np.nan)
        print(f"  {c:<12}{out[0]:>18.2f}{out[1]:>18.2f}")
    print("\n  1에 가까울수록 크기에 무관하다. **줄지 않으면 평활이 원인이 아니다.**")

    # ── [4] 전이 실험 (D-003 분할 위) ─────────────────────
    print("\n" + "=" * 78)
    print("[4] 크기 층 간 전이 — Center↔Loc, 임계값은 train에서만")
    print("=" * 78)
    y = (cls == "Center").astype(int)
    sel = (cls == "Center") | (cls == "Loc")
    cands = {"F1a 3등분": old["radial_contrast"], "F1a 등면적": eq["eq_rc"],
             "coverage(참고)": base["cov"]}
    print(f"  {'특징':<18}{'대각(상한)':>12}{'비대각':>10}{'최악':>9}{'전이 손실':>16}")
    for nm, arr in cands.items():
        per = []
        for si in range(len(seeds)):
            for fo in range(5):
                te = (F[si] == fo) & sel; tr = (F[si] != fo) & sel
                ed = np.quantile(size[tr], np.linspace(0, 1, N_STRATA + 1))
                ed[0], ed[-1] = -np.inf, np.inf
                st = lambda m_, s: m_ & (size >= ed[s]) & (size < ed[s + 1])
                A = np.full((N_STRATA, N_STRATA), np.nan)
                for a_ in range(N_STRATA):
                    m_ = st(tr, a_)
                    if m_.sum() < 50 or len(np.unique(y[m_])) < 2:
                        continue
                    t, sg = fit_threshold(arr[m_], y[m_])
                    for b_ in range(N_STRATA):
                        mt = st(te, b_)
                        if mt.sum() < 20 or len(np.unique(y[mt])) < 2:
                            continue
                        A[a_, b_] = balanced_acc(arr[mt], y[mt], t, sg)
                dg = np.nanmean([A[i, i] for i in range(N_STRATA)])
                of = np.nanmean([A[i, j] for i in range(N_STRATA)
                                 for j in range(N_STRATA) if i != j])
                per.append((dg, of, np.nanmin(A)))
        P = np.array(per)
        print(f"  {nm:<18}{P[:,0].mean():>12.3f}{P[:,1].mean():>10.3f}"
              f"{np.nanmin(P[:,2]):>9.3f}"
              f"{f'{(P[:,0]-P[:,1]).mean():+.3f} ± {(P[:,0]-P[:,1]).std():.3f}':>16}")
    print("\n  `coverage`의 Ring↔Loc 전이 손실은 0.036이었다 (다른 쌍이므로 참고만).")
    print("  **등면적이 3등분보다 손실이 작으면 §3-9와 전이가 함께 해결된다.**")


if __name__ == "__main__":
    main()
