"""8종 다중 클래스 — out-of-fold per-class F1과 **혼동 행렬 하나**.

실행:
    python src/model8.py        # 특징 캐시가 없으면 먼저 계산 (수 분)

이 스크립트가 답하는 것:
    1. 세 특징으로 8종을 나누면 **어느 패턴이 어느 패턴과 헷갈리는가**
       -> 이 프로젝트의 최종 산출물. accuracy는 보고하지 않는다 (D-003 평가지표)
    2. **D-011의 뒤집을 조건**: 8종에서 contrast 추가가 per-class F1을 개선하는가
    3. **D-012의 뒤집을 조건**: Edge-Loc 계열 증분이 +0.01 미만이면 CV를 강등한다

설계:
    - 분할: **D-003 SGK(5) × seed 0/1/2.** 전 웨이퍼 out-of-fold 예측 → 혼동 행렬 하나
    - 모델: RandomForest. **NaN을 그대로 넣는다** (sklearn ≥ 1.4)
    - `class_weight="balanced"`: **근거** — 목적이 성능이 아니라 혼동 구조 관찰이다.
      가중치가 없으면 Near-full(149장)·Donut(555장)이 아예 예측되지 않아
      **혼동 행렬에서 그 행이 통째로 비고, 정작 보고 싶은 것이 안 보인다.**
      가중치 유무를 §4 절제표에 함께 실어 선택을 기록한다.
    - **맵 크기(`size`)는 특징에 넣지 않는다.** 알려진 교란이고(단독 AUC 0.780),
      `w3_transfer.md`에서 **크기 축으로 전이되지 않음**을 확인했다.
      넣은 변이를 §4에 대조로 싣되 주 모델에서는 뺀다.

한계:
    라벨 노이즈가 Edge-Ring 10.5% / Edge-Loc 14.1%다 (`w2_features.md`).
    **이 두 클래스의 F1 상한이 0.9 근처라는 뜻이다.** 그보다 높게 나오면
    오히려 의심해야 한다.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

import config
from edge_band import coverage, edge_band, band_circular_variance
from edge_contrast import edge_contrast

CACHE = config.DATA_PROCESSED / "feats8.npz"
K = 1
FEATS = ("cov", "ctr", "cv")          # 3종 세트 (D-010/D-011/D-012)


def build_features(cls_order, idx_order):
    """split_folds의 행 순서에 정확히 맞춰 특징을 만든다."""
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=True) as z:
            return {k: z[k] for k in z.keys()}
    print("특징 계산 중 (최초 1회, 수 분)...")
    store = {}
    for c in config.PATTERN_CLASSES:
        with np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True) as z:
            store[c] = z["wafer_maps"]
        print(f"  {c:<12} {len(store[c]):,}장")
    n = len(cls_order)
    out = {k: np.full(n, np.nan) for k in ("cov", "ctr", "cv", "nfail", "size")}
    for i, (c, j) in enumerate(zip(cls_order, idx_order)):
        a = np.asarray(store[c][j])
        b = edge_band(a, K)
        out["cov"][i] = coverage(a, K)
        out["ctr"][i] = edge_contrast(a)
        out["cv"][i] = band_circular_variance(a, K)
        out["nfail"][i] = float(((a == config.VAL_FAIL) & b).sum())
        out["size"][i] = float((a != config.VAL_OUTSIDE).sum())
    np.savez_compressed(CACHE, **out)
    print(f"  캐시 저장: {CACHE}")
    return out


def oof_predict(X, y, folds, **kw):
    """out-of-fold 예측. 모든 웨이퍼가 정확히 한 번 예측된다 (커버리지 100%)."""
    pred = np.empty(len(y), dtype=object)
    for f in np.unique(folds):
        te = folds == f
        mdl = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1,
            random_state=0, **kw).fit(X[~te], y[~te])
        pred[te] = mdl.predict(X[te])
    return pred.astype(str)


def main():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        cls, idx, F, seeds = z["cls"].astype(str), z["idx_in_cls"], z["folds"], z["seeds"]
    d = build_features(cls, idx)
    labels = config.PATTERN_CLASSES
    y = cls

    print(f"\n대상 {len(y):,}장 / 8종 / 분할 SGK(5) × seed {tuple(int(s) for s in seeds)}")
    print("**accuracy는 보고하지 않는다** (D-003 평가지표: per-class F1 + 혼동 쌍)\n")

    X3 = np.column_stack([d[k] for k in FEATS])

    # ── [1] per-class F1 (seed 간 평균±편차) ──────────────
    print("=" * 78)
    print("[1] out-of-fold per-class F1 — 3종 세트 (cov + ctr + CV)")
    print("=" * 78)
    P = [oof_predict(X3, y, F[s], class_weight="balanced") for s in range(len(seeds))]
    f1 = np.array([f1_score(y, p, labels=labels, average=None, zero_division=0)
                   for p in P])
    print(f"  {'클래스':<12}{'n':>8}{'F1 (평균±편차)':>20}{'재현율':>9}{'정밀도':>9}")
    for i, c in enumerate(labels):
        rec = np.mean([( (p == c) & (y == c)).sum() / max((y == c).sum(), 1) for p in P])
        pre = np.mean([((p == c) & (y == c)).sum() / max((p == c).sum(), 1) for p in P])
        print(f"  {c:<12}{int((y==c).sum()):>8,}"
              f"{f'{f1[:,i].mean():.3f} ± {f1[:,i].std():.3f}':>20}{rec:>9.3f}{pre:>9.3f}")
    macro = f1.mean(axis=1)
    print(f"\n  macro-F1 = {macro.mean():.3f} ± {macro.std():.3f}")
    print("  라벨 노이즈 Edge-Ring 10.5% / Edge-Loc 14.1% → 두 클래스의 상한은 0.9 근처다")

    # ── [2] 혼동 행렬 (seed 0, out-of-fold 전수) ──────────
    print("\n" + "=" * 78)
    print("[2] 혼동 행렬 — out-of-fold 전수 25,519장 (seed 0). 행=실제, 열=예측")
    print("=" * 78)
    M = confusion_matrix(y, P[0], labels=labels)
    short = [c[:9] for c in labels]
    hdr = "실제 \\ 예측"
    print(f"  {hdr:<12}" + "".join(f"{s:>10}" for s in short))
    for i, c in enumerate(labels):
        row = "".join(f"{M[i,j]:>10,}" for j in range(len(labels)))
        print(f"  {c:<12}{row}")

    # ── [3] 혼동 쌍 상호 오분류율 ★ 이 프로젝트의 산출물 ──
    print("\n" + "=" * 78)
    print("[3] 혼동 쌍 상호 오분류율 — (i→j + j→i) / (n_i + n_j), seed 평균")
    print("=" * 78)
    Ms = [confusion_matrix(y, p, labels=labels) for p in P]
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            r = [(m[i, j] + m[j, i]) / (m[i].sum() + m[j].sum()) for m in Ms]
            pairs.append((np.mean(r), np.std(r), labels[i], labels[j],
                          np.mean([m[i, j] for m in Ms]),
                          np.mean([m[j, i] for m in Ms])))
    pairs.sort(reverse=True)
    print(f"  {'쌍':<26}{'상호 오분류율':>14}{'i→j':>9}{'j→i':>9}{'비대칭':>9}")
    for r, sd, a, b, ij, ji in pairs[:10]:
        asym = max(ij, ji) / max(min(ij, ji), 1)
        print(f"  {a+' ↔ '+b:<26}{f'{r:.3f} ± {sd:.3f}':>14}"
              f"{ij:>9.0f}{ji:>9.0f}{asym:>8.1f}배")
    print("\n  **비대칭이 크면 한쪽으로만 흡수된다는 뜻이다** — 원인 해석이 달라진다.")

    # ── [4] 특징 절제 — D-011·D-012의 뒤집을 조건 ─────────
    print("\n" + "=" * 78)
    print("[4] 특징 절제 — D-011·D-012의 뒤집을 조건을 여기서 판정한다")
    print("=" * 78)
    sets = {
        "cov 단독": ("cov",),
        "cov + ctr": ("cov", "ctr"),
        "cov + ctr + CV  ★주": ("cov", "ctr", "cv"),
        "(대조) n_fail 단독": ("nfail",),
        "(대조) 3종 + size": ("cov", "ctr", "cv", "size"),
    }
    res = {}
    for nm, keys in sets.items():
        Xs = np.column_stack([d[k] for k in keys])
        ps = [oof_predict(Xs, y, F[s], class_weight="balanced")
              for s in range(len(seeds))]
        res[nm] = np.array([f1_score(y, p, labels=labels, average=None,
                                     zero_division=0) for p in ps])
    # 가중치 없는 변이 (선택 근거 기록용)
    ps = [oof_predict(X3, y, F[s]) for s in range(len(seeds))]
    res["3종 · 가중치 없음"] = np.array([f1_score(y, p, labels=labels, average=None,
                                             zero_division=0) for p in ps])

    print(f"  {'특징 집합':<22}{'macro':>8}" + "".join(f"{c[:7]:>8}" for c in labels))
    for nm, arr in res.items():
        m = arr.mean(axis=0)
        print(f"  {nm:<22}{m.mean():>8.3f}" + "".join(f"{v:>8.3f}" for v in m))

    base2 = res["cov + ctr"].mean(axis=0)
    base1 = res["cov 단독"].mean(axis=0)
    base3 = res["cov + ctr + CV  ★주"].mean(axis=0)
    print("\n  ── 증분 (per-class F1) ──")
    print(f"  {'클래스':<12}{'ctr 증분':>12}{'CV 증분':>12}")
    for i, c in enumerate(labels):
        print(f"  {c:<12}{base2[i]-base1[i]:>+12.3f}{base3[i]-base2[i]:>+12.3f}")
    print(f"  {'(macro)':<12}{base2.mean()-base1.mean():>+12.3f}"
          f"{base3.mean()-base2.mean():>+12.3f}")
    print("\n  D-011 뒤집을 조건: contrast 추가가 per-class F1을 개선 못 하면 재검토")
    print("  D-012 뒤집을 조건: Edge-Loc 계열 CV 증분이 +0.01 미만이면 보조로 강등")


if __name__ == "__main__":
    main()
