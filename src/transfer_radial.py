"""`r_shift` 전이 실험 — D-014의 미해결 항목을 닫는다.

실행:
    python -u src/transfer_radial.py

## 무엇을 묻나

D-014에서 경계 없는 `r_shift`를 도입했지만 **전이 실험을 안 했다.**
`w3_transfer.md`가 확립한 원칙은 명확하다 —
**같은 층 안에서 잘 되는 것과 층을 넘어 전이되는 것은 다른 문제다.**

    coverage    전이 손실 0.036 ± 0.010   (정규화 O)
    n_fail      전이 손실 0.176 ± 0.016   (정규화 X, S3→S0에서 0.500 붕괴)
    edge_contrast            0.096 ± 0.010

`r_shift`는 **웨이퍼 자신의 평균 반경을 빼는 차분**이므로 구조적으로
크기에 둔감할 것으로 예상된다. **예상은 근거가 아니다 (§3-3).**

## 설계 — `w3_transfer.md`와 동일 프로토콜

- 대상 쌍: `Center↔Loc`(최대 병목) 및 `Donut↔Loc`(r_shift가 가장 센 자리)
- 분할: D-003 SGK(5) × seed 0/1/2. **임계값·층 경계 모두 train에서만**
- 분류기: 단일 특징 임계값(Youden J) — **임계값 전이성만 재기 위해**
- 지표: balanced accuracy (층마다 클래스 비율이 다르므로)
- 비교군: `F1a(3등분)` · `r_shift` · `coverage`(참고 기준선)

## 판정 기준 (사전 등록)

`r_shift`의 전이 손실이 **F1a보다 작으면** 파라미터 0개라는 이점에 더해
이식성까지 확보된다 → **D-014의 "F1a 흡수" 조건 검토로 넘어간다.**
**크면** 파라미터가 없다는 것만으로는 채택 근거가 부족하다.
"""
import numpy as np

import config
from radial import r_shift
from transfer_size import fit_threshold, balanced_acc

CACHE = config.DATA_PROCESSED / "rshift.npz"
N_STRATA = 4
PAIRS = [("Center", "Loc"), ("Donut", "Loc"), ("Center", "Edge-Loc")]


def build(cls_order, idx_order):
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=True) as z:
            return z["rs"]
    store = {}
    for c in config.PATTERN_CLASSES:
        with np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True) as z:
            store[c] = z["wafer_maps"]
    rs = np.full(len(cls_order), np.nan)
    for i, (c, j) in enumerate(zip(cls_order, idx_order)):
        rs[i] = r_shift(store[c][j])
        if (i + 1) % 6000 == 0:
            print(f"  {i+1:,}/{len(cls_order):,}", flush=True)
    np.savez_compressed(CACHE, rs=rs)
    print(f"  캐시 저장: {CACHE}")
    return rs


def transfer(arr, y, sel, size, folds, seeds):
    """전이 행렬을 fold·seed마다 내고 (대각, 비대각, 최악) 목록을 돌려준다."""
    per = []
    for si in range(len(seeds)):
        for fo in range(5):
            te = (folds[si] == fo) & sel
            tr = (folds[si] != fo) & sel
            ed = np.quantile(size[tr], np.linspace(0, 1, N_STRATA + 1))
            ed[0], ed[-1] = -np.inf, np.inf
            st = lambda m, s: m & (size >= ed[s]) & (size < ed[s + 1])
            A = np.full((N_STRATA, N_STRATA), np.nan)
            for a in range(N_STRATA):
                m = st(tr, a)
                if m.sum() < 50 or len(np.unique(y[m])) < 2:
                    continue
                t, sg = fit_threshold(arr[m], y[m])
                for b in range(N_STRATA):
                    mt = st(te, b)
                    if mt.sum() < 20 or len(np.unique(y[mt])) < 2:
                        continue
                    A[a, b] = balanced_acc(arr[mt], y[mt], t, sg)
            dg = np.nanmean([A[i, i] for i in range(N_STRATA)])
            of = np.nanmean([A[i, j] for i in range(N_STRATA)
                             for j in range(N_STRATA) if i != j])
            per.append((dg, of, np.nanmin(A)))
    return np.array(per)


def main():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        cls, idx, F, seeds = z["cls"].astype(str), z["idx_in_cls"], z["folds"], z["seeds"]
    rs = build(cls, idx)
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        base = {k: z[k] for k in z.keys()}
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        rad = {k: z[k] for k in z.keys()}
    size = base["size"]

    print(f"\n대상 {len(cls):,}장 / SGK(5) × seed {tuple(int(s) for s in seeds)}")
    print("임계값·층 경계 모두 train에서만 결정. 지표는 balanced accuracy\n")

    cands = {"F1a (3등분, 파라미터 2)": rad["radial_contrast"],
             "r_shift (파라미터 0)": rs,
             "coverage (참고)": base["cov"]}

    for p, n in PAIRS:
        sel = (cls == p) | (cls == n)
        y = (cls == p).astype(int)
        print("=" * 78)
        print(f"[{p} ↔ {n}]")
        print("=" * 78)
        print(f"  {'특징':<24}{'대각(상한)':>12}{'비대각':>10}{'최악':>9}{'전이 손실':>16}")
        for nm, arr in cands.items():
            P = transfer(arr, y, sel, size, F, seeds)
            loss = P[:, 0] - P[:, 1]
            print(f"  {nm:<24}{P[:,0].mean():>12.3f}{P[:,1].mean():>10.3f}"
                  f"{np.nanmin(P[:,2]):>9.3f}"
                  f"{f'{loss.mean():+.3f} ± {loss.std():.3f}':>16}")
        print()

    print("=" * 78)
    print("판정 기준 (사전 등록)")
    print("=" * 78)
    print("  참고: w3_transfer.md의 Edge-Ring↔Edge-Loc 전이 손실")
    print("    coverage 0.036 ± 0.010 / edge_contrast 0.096 / n_fail 0.176")
    print("  **r_shift < F1a 이면** 파라미터 0개 + 이식성 → D-014의 F1a 흡수 검토로")
    print("  **r_shift > F1a 이면** 파라미터 없음만으로는 채택 근거가 부족하다")


if __name__ == "__main__":
    main()
