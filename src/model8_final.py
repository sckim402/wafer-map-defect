"""D-017 뒤집을 조건 판정 — `cc_compact`가 8종 다변량에서 자리를 버는가.

실행:
    python -u src/model8_final.py

**사전 등록한 판정 기준 (D-017)**

> *"8종 다변량 증분이 **+0.01 미만**이면 채택하지 않는다.
>  쌍별 판별력만으로는 근거가 부족하다 (D-012에서 배운 것)."*

`Loc↔Scratch` 쌍별 |AUC−0.5|는 0.152 → 0.461로 3배였지만,
**D-012에서 확인했듯 쌍별 판별력이 다변량 기여를 보장하지 않는다.**
(CV는 Ring↔Loc 단독 0.915인데 다변량 증분 0.000이었다 — 천장 효과)

    5종 = cov + ctr + cv + F1a + F1b          (D-015 확정)
    6종 = 5종 + cc_compact

**증분을 남은 여유와 함께 읽는다** (§3-12의 천장 효과).
5종의 Scratch F1이 0.358이므로 여유가 크다 — 천장 효과는 없을 것이다.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

import config

SETS = {"5종 (D-015)": ("cov", "ctr", "cv", "rc", "mp"),
        "6종 (+cc_compact)": ("cov", "ctr", "cv", "rc", "mp", "cc")}


def oof(X, y, folds):
    p = np.empty(len(y), dtype=object)
    for f in np.unique(folds):
        te = folds == f
        p[te] = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0,
            class_weight="balanced").fit(X[~te], y[~te]).predict(X[te])
    return p.astype(str)


def pair_rates(y, p, L):
    M = confusion_matrix(y, p, labels=L)
    return {(L[i], L[j]): (M[i, j] + M[j, i]) / (M[i].sum() + M[j].sum())
            for i in range(len(L)) for j in range(i + 1, len(L))}


def main():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        y, orig, F, seeds = (z["cls"].astype(str), z["split_orig"].astype(str),
                             z["folds"], z["seeds"])
    d = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        for k in ("cov", "ctr", "cv"):
            d[k] = z[k]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        d["rc"], d["mp"] = z["radial_contrast"], z["mid_peak"]
    with np.load(config.DATA_PROCESSED / "shape2.npz", allow_pickle=True) as z:
        d["cc"] = z["cc_compact"]
    L = config.PATTERN_CLASSES

    res, preds = {}, {}
    for nm, keys in SETS.items():
        X = np.column_stack([d[k] for k in keys])
        ps = [oof(X, y, F[s]) for s in range(len(seeds))]
        preds[nm] = ps
        res[nm] = np.array([f1_score(y, p, labels=L, average=None, zero_division=0)
                            for p in ps])
        print(f"  {nm:<20} macro {res[nm].mean(axis=0).mean():.3f} "
              f"± {res[nm].mean(axis=1).std():.3f}", flush=True)

    a = res["5종 (D-015)"].mean(axis=0)
    b = res["6종 (+cc_compact)"].mean(axis=0)
    print("\n" + "=" * 78)
    print("[1] per-class F1 과 증분 — 남은 여유와 함께 읽는다")
    print("=" * 78)
    print(f"  {'클래스':<12}{'5종':>9}{'6종':>9}{'증분':>9}{'남은 여유':>11}{'여유대비':>10}")
    for i, c in enumerate(L):
        head = 1.0 - a[i]
        print(f"  {c:<12}{a[i]:>9.3f}{b[i]:>9.3f}{b[i]-a[i]:>+9.3f}{head:>11.3f}"
              f"{(b[i]-a[i])/head if head > 1e-6 else np.nan:>10.2f}")
    inc = b.mean() - a.mean()
    print(f"  {'(macro)':<12}{a.mean():>9.3f}{b.mean():>9.3f}{inc:>+9.3f}")

    print("\n" + "=" * 78)
    print("[2] ★ D-017 판정")
    print("=" * 78)
    print(f"  macro 증분 = {inc:+.3f}   (사전 등록 기준: +0.01)")
    print(f"  Scratch 증분 = {b[L.index('Scratch')] - a[L.index('Scratch')]:+.3f}"
          f"  /  Loc 증분 = {b[L.index('Loc')] - a[L.index('Loc')]:+.3f}")
    print()
    print("  → +0.01 이상이면 **채택.** 미만이면 쌍별 판별력에도 불구하고 기각한다.")

    print("\n" + "=" * 78)
    print("[3] 혼동 쌍 — `Loc↔Scratch`가 실제로 내려갔는가")
    print("=" * 78)
    r5 = [pair_rates(y, p, L) for p in preds["5종 (D-015)"]]
    r6 = [pair_rates(y, p, L) for p in preds["6종 (+cc_compact)"]]
    keys = sorted(r5[0], key=lambda k: -np.mean([r[k] for r in r5]))
    print(f"  {'쌍':<26}{'5종':>9}{'6종':>9}{'변화':>9}")
    for k in keys[:6]:
        v5 = np.mean([r[k] for r in r5]); v6 = np.mean([r[k] for r in r6])
        print(f"  {k[0]+' ↔ '+k[1]:<26}{v5:>9.3f}{v6:>9.3f}{v6-v5:>+9.3f}")

    print("\n" + "=" * 78)
    print("[4] 외부 분할 (`split_orig`) — D-016의 병기 원칙 적용")
    print("=" * 78)
    tr, te = orig == "Training", orig == "Test"
    for nm, keys in SETS.items():
        X = np.column_stack([d[k] for k in keys])
        pe = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                    random_state=0, class_weight="balanced"
                                    ).fit(X[tr], y[tr]).predict(X[te])
        m = f1_score(y[te], pe, labels=L, average='macro', zero_division=0)
        rr = pair_rates(y[te], pe, L)
        k = ("Loc", "Scratch") if ("Loc", "Scratch") in rr else ("Scratch", "Loc")
        print(f"  {nm:<20} 외부 macro {m:.3f}   Loc↔Scratch {rr[k]:.3f}")
    print("\n  **주 분할과 외부 분할 값을 병기해 기록한다** (D-016)")


if __name__ == "__main__":
    main()
