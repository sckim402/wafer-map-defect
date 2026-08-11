"""8종 다변량 증분 — F1 계열이 실제로 per-class F1을 올리는가.

실행:
    python -u src/model8_radial.py

## 무엇을 판정하나

지금까지 F1a(`radial_contrast`) · F1b(`mid_peak`) · F1c(`r_shift`)의 근거는
**전부 쌍별 AUC**다. §3-1 5항 중 5번(범위 전체)이 미충족이라 **"첫 신호"** 상태다.
`w3_model8.md`가 확립한 프로토콜로 8종 전체에서 다시 잰다.

**★ D-014의 뒤집을 조건이 여기서 판정된다:**

> *"8종 다변량에서 `r_shift`가 F1a를 완전히 흡수하면(F1a 증분 < +0.01)
> F1a를 빼고 파라미터를 0개로 만든다. 그쪽이 더 낫다."*

그래서 **순서를 뒤집은 두 경로**를 모두 낸다.

    base                = cov + ctr + cv            (현행 주 모델)
    base + F1a          → F1a 증분
    base + F1a + F1b    → F1b 증분
    base + r_shift      → r_shift 증분 (파라미터 0개 추가)
    base + F1a+F1b+r_shift
                        → **(전부) − (base+r_shift) = r_shift 위에서의 F1a·F1b 증분**
                          이 값이 +0.01 미만이면 **F1a·F1b를 뺀다**

설계는 `w3_model8.md`와 동일 — RandomForest(`class_weight="balanced"`),
D-003 SGK(5) × seed 0/1/2, out-of-fold 전수, **accuracy 미보고.**
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

import config

SETS = {
    "base (cov+ctr+cv)":        ("cov", "ctr", "cv"),
    "base + F1a":               ("cov", "ctr", "cv", "rc"),
    "base + F1a + F1b":         ("cov", "ctr", "cv", "rc", "mp"),
    "base + r_shift":           ("cov", "ctr", "cv", "rs"),
    "base + F1a+F1b+r_shift":   ("cov", "ctr", "cv", "rc", "mp", "rs"),
}


def oof(X, y, folds):
    p = np.empty(len(y), dtype=object)
    for f in np.unique(folds):
        te = folds == f
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                   random_state=0, class_weight="balanced")
        p[te] = m.fit(X[~te], y[~te]).predict(X[te])
    return p.astype(str)


def pair_rates(y, p, labels):
    M = confusion_matrix(y, p, labels=labels)
    out = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            out[(labels[i], labels[j])] = (M[i, j] + M[j, i]) / (M[i].sum() + M[j].sum())
    return out


def main():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        y, F, seeds = z["cls"].astype(str), z["folds"], z["seeds"]
    d = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        for k in ("cov", "ctr", "cv"):
            d[k] = z[k]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        d["rc"], d["mp"] = z["radial_contrast"], z["mid_peak"]
    with np.load(config.DATA_PROCESSED / "rshift.npz", allow_pickle=True) as z:
        d["rs"] = z["rs"]
    L = config.PATTERN_CLASSES

    print(f"대상 {len(y):,}장 / 8종 / SGK(5) × seed {tuple(int(s) for s in seeds)}")
    print("RandomForest, class_weight=balanced, out-of-fold. **accuracy 미보고**\n")

    res, preds = {}, {}
    for nm, keys in SETS.items():
        X = np.column_stack([d[k] for k in keys])
        ps = [oof(X, y, F[s]) for s in range(len(seeds))]
        preds[nm] = ps
        res[nm] = np.array([f1_score(y, p, labels=L, average=None, zero_division=0)
                            for p in ps])
        m = res[nm].mean(axis=0)
        print(f"  {nm:<26} macro {m.mean():.3f} ± {res[nm].mean(axis=1).std():.3f}",
              flush=True)

    # ── [1] per-class F1 ─────────────────────────────────
    print("\n" + "=" * 92)
    print("[1] per-class F1 (seed 3개 평균)")
    print("=" * 92)
    print(f"  {'특징 집합':<26}{'macro':>8}" + "".join(f"{c[:7]:>8}" for c in L))
    for nm in SETS:
        m = res[nm].mean(axis=0)
        print(f"  {nm:<26}{m.mean():>8.3f}" + "".join(f"{v:>8.3f}" for v in m))

    # ── [2] 증분 ─────────────────────────────────────────
    print("\n" + "=" * 92)
    print("[2] 증분 — 어느 클래스가 오르는가")
    print("=" * 92)
    b = res["base (cov+ctr+cv)"].mean(axis=0)
    a1 = res["base + F1a"].mean(axis=0)
    a2 = res["base + F1a + F1b"].mean(axis=0)
    rs = res["base + r_shift"].mean(axis=0)
    al = res["base + F1a+F1b+r_shift"].mean(axis=0)
    cols = [("F1a (base 위)", a1 - b), ("F1b (+F1a 위)", a2 - a1),
            ("r_shift (base 위)", rs - b), ("★F1a+F1b (r_shift 위)", al - rs),
            ("r_shift (F1a+F1b 위)", al - a2)]
    print(f"  {'클래스':<12}" + "".join(f"{n:>22}" for n, _ in cols))
    for i, c in enumerate(L):
        print(f"  {c:<12}" + "".join(f"{v[i]:>+22.3f}" for _, v in cols))
    print(f"  {'(macro)':<12}" + "".join(f"{v.mean():>+22.3f}" for _, v in cols))

    # ── [3] D-014 뒤집을 조건 판정 ───────────────────────
    print("\n" + "=" * 92)
    print("[3] ★ D-014 판정 — r_shift가 F1a·F1b를 흡수하는가")
    print("=" * 92)
    inc = (al - rs).mean()
    print(f"  `base + r_shift` 위에 F1a·F1b를 더한 macro 증분 = {inc:+.3f}")
    print(f"  (역방향: `base + F1a + F1b` 위에 r_shift = {(al - a2).mean():+.3f})")
    print()
    if inc < 0.01:
        print("  → **+0.01 미만. F1a·F1b를 뺀다.** `base + r_shift`로 확정하면")
        print("     추가 파라미터가 0개가 되고 §3-9 논쟁이 사라진다.")
    else:
        print("  → **+0.01 이상. F1a·F1b가 흡수되지 않는다.** 셋 다 유지한다.")
        print("     대신 R1·R2가 임의값이라는 한계를 명시하고 스윕을 병기한다.")

    # ── [4] 혼동 쌍이 실제로 내려갔는가 ─────────────────
    print("\n" + "=" * 92)
    print("[4] 상위 혼동 쌍 — 병목이 내려갔는가 (seed 평균)")
    print("=" * 92)
    r0 = [pair_rates(y, p, L) for p in preds["base (cov+ctr+cv)"]]
    r1 = [pair_rates(y, p, L) for p in preds["base + F1a+F1b+r_shift"]]
    keys = sorted(r0[0], key=lambda k: -np.mean([r[k] for r in r0]))
    print(f"  {'쌍':<26}{'base':>9}{'전부':>9}{'변화':>9}")
    for k in keys[:8]:
        v0 = np.mean([r[k] for r in r0]); v1 = np.mean([r[k] for r in r1])
        print(f"  {k[0]+' ↔ '+k[1]:<26}{v0:>9.3f}{v1:>9.3f}{v1-v0:>+9.3f}")
    print("\n  **`Loc↔Scratch`가 안 내려가면 예측대로다** — 반경 특징으로는")
    print("  형상(선형성)을 잴 수 없다. F2 재설계가 남은 이유다.")


if __name__ == "__main__":
    main()
