"""외부 분할 재현 — **D-016의 뒤집을 조건을 판정한다.**

실행:
    python -u src/external_radial.py

## 무엇을 묻나

D-016은 *"병목이 두 번 이동했다"*를 초록의 결론으로 확정했다.
**그런데 그 이동은 주 분할(SGK) 하나에서만 본 값이다.**

전례가 있다 — `w3_external.md`에서 `Edge-Loc↔Edge-Ring`이
주 분할 **0.040(8위)** → 외부 분할 **0.126(5위)**로 3.2배 튀었다.
**같은 일이 Center 계열에도 일어날 수 있다.**

> **D-016 뒤집을 조건**: *"외부 분할에서 이 이동이 재현되지 않으면 서술을 약화한다."*

## 판정 항목

    [1] macro-F1과 per-class — 절대값이 아니라 **재현율**로 본다
        (사전확률이 다르면 정밀도가 따라 움직인다 — `w3_external.md` §1)
    [2] **혼동 쌍 순위** — `Loc↔Scratch`가 외부에서도 1위인가
    [3] **Center 계열이 낮게 유지되는가** — 0.063 / 0.013이 튀지 않는가
    [4] Spearman 순위 상관 (기준선: 같은 분할 seed 간 0.997)
    [5] D-015 증분(F1a·F1b)의 방향이 외부에서도 같은가
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

import config

SETS = {"base": ("cov", "ctr", "cv"),
        "base+F1a": ("cov", "ctr", "cv", "rc"),
        "base+F1a+F1b": ("cov", "ctr", "cv", "rc", "mp")}


def rf(**kw):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                  random_state=0, class_weight="balanced", **kw)


def pair_rates(y, p, L):
    M = confusion_matrix(y, p, labels=L)
    return {(L[i], L[j]): (M[i, j] + M[j, i]) / (M[i].sum() + M[j].sum())
            for i in range(len(L)) for j in range(i + 1, len(L))}


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                             np.argsort(np.argsort(b)))[0, 1])


def main():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        y, orig, F = z["cls"].astype(str), z["split_orig"].astype(str), z["folds"]
    d = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        for k in ("cov", "ctr", "cv"):
            d[k] = z[k]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        d["rc"], d["mp"] = z["radial_contrast"], z["mid_peak"]
    L = config.PATTERN_CLASSES
    tr, te = orig == "Training", orig == "Test"
    print(f"외부 분할 train {tr.sum():,} / test {te.sum():,} ({te.mean()*100:.1f}%)")
    print("고정 분할이라 반복 불가 — 주 분할은 seed 0의 out-of-fold와 비교\n")

    ext, main_ = {}, {}
    for nm, keys in SETS.items():
        X = np.column_stack([d[k] for k in keys])
        ext[nm] = rf().fit(X[tr], y[tr]).predict(X[te])
        p = np.empty(len(y), dtype=object)
        for f in np.unique(F[0]):
            m = F[0] == f
            p[m] = rf().fit(X[~m], y[~m]).predict(X[m])
        main_[nm] = p.astype(str)
        print(f"  {nm:<16} 주 {f1_score(y, main_[nm], labels=L, average='macro', zero_division=0):.3f}"
              f"  /  외부 {f1_score(y[te], ext[nm], labels=L, average='macro', zero_division=0):.3f}",
              flush=True)

    FULL = "base+F1a+F1b"
    # ── [1] per-class ────────────────────────────────────
    print("\n" + "=" * 84)
    print("[1] per-class — **재현율**로 본다 (정밀도는 사전확률에 딸려 움직인다)")
    print("=" * 84)
    print(f"  {'클래스':<12}{'주 재현':>9}{'외부 재현':>11}{'차이':>8}{'주 F1':>9}{'외부 F1':>9}")
    for c in L:
        rs_ = ((main_[FULL] == c) & (y == c)).sum() / max((y == c).sum(), 1)
        re_ = ((ext[FULL] == c) & (y[te] == c)).sum() / max((y[te] == c).sum(), 1)
        f_s = f1_score(y, main_[FULL], labels=[c], average=None, zero_division=0)[0]
        f_e = f1_score(y[te], ext[FULL], labels=[c], average=None, zero_division=0)[0]
        print(f"  {c:<12}{rs_:>9.3f}{re_:>11.3f}{re_-rs_:>+8.3f}{f_s:>9.3f}{f_e:>9.3f}")

    # ── [2][3] 혼동 쌍 ───────────────────────────────────
    print("\n" + "=" * 84)
    print("[2] 혼동 쌍 — 병목 이동이 외부에서도 재현되는가 ★")
    print("=" * 84)
    rm = pair_rates(y, main_[FULL], L)
    re2 = pair_rates(y[te], ext[FULL], L)
    rb = pair_rates(y, main_["base"], L)          # 이동 전(가장자리 3종)
    om = sorted(rm, key=lambda k: -rm[k]); oe = sorted(re2, key=lambda k: -re2[k])
    print(f"  {'쌍':<26}{'base(주)':>10}{'5종(주)':>10}{'5종(외부)':>11}"
          f"{'주순위':>8}{'외부순위':>9}")
    for k in sorted(re2, key=lambda k: -re2[k])[:8]:
        print(f"  {k[0]+' ↔ '+k[1]:<26}{rb[k]:>10.3f}{rm[k]:>10.3f}{re2[k]:>11.3f}"
              f"{om.index(k)+1:>8}{oe.index(k)+1:>9}")
    keys = list(rm)
    print(f"\n  28쌍 Spearman(주 vs 외부) = **{spearman([rm[k] for k in keys], [re2[k] for k in keys]):.3f}**")
    print("  기준선: 같은 분할 seed 간 0.997 / 가장자리 3종일 때 주-외부 0.942")

    print("\n  ── D-016 핵심 3쌍 ──")
    for a, b in (("Center", "Loc"), ("Center", "Edge-Loc"), ("Loc", "Scratch")):
        k = (a, b) if (a, b) in rm else (b, a)
        print(f"    {a+' ↔ '+b:<22} base {rb[k]:.3f} → 5종 주 {rm[k]:.3f} "
              f"→ 5종 외부 **{re2[k]:.3f}**  (외부 {oe.index(k)+1}위)")

    # ── [5] 증분 방향 ────────────────────────────────────
    print("\n" + "=" * 84)
    print("[5] D-015 증분이 외부에서도 같은 방향인가")
    print("=" * 84)
    g = lambda p, yy: f1_score(yy, p, labels=L, average=None, zero_division=0)
    e0, e1, e2 = g(ext["base"], y[te]), g(ext["base+F1a"], y[te]), g(ext[FULL], y[te])
    m0, m1, m2 = g(main_["base"], y), g(main_["base+F1a"], y), g(main_[FULL], y)
    print(f"  {'클래스':<12}{'F1a(주)':>10}{'F1a(외부)':>11}{'F1b(주)':>10}{'F1b(외부)':>11}")
    for i, c in enumerate(L):
        print(f"  {c:<12}{m1[i]-m0[i]:>+10.3f}{e1[i]-e0[i]:>+11.3f}"
              f"{m2[i]-m1[i]:>+10.3f}{e2[i]-e1[i]:>+11.3f}")
    print(f"  {'(macro)':<12}{(m1-m0).mean():>+10.3f}{(e1-e0).mean():>+11.3f}"
          f"{(m2-m1).mean():>+10.3f}{(e2-e1).mean():>+11.3f}")


if __name__ == "__main__":
    main()
