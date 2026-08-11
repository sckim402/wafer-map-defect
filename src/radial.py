"""F1 방사형 특징 — **대조비로 재정의한다.** 형상 특징 1차 시도의 정정.

실행:
    python -u src/radial.py

## 왜 다시 쓰는가 — `shape_feats.py`의 4종이 실패했다

1차 시도(`src/shape_feats.py`)는 **불량 die의 절대 통계**를 썼다:
`r_med`(반경 중앙값) · `elong`(PCA 이심률) · `cluster`(최대 연결성분 비율).
**기대 4개 중 3개가 어긋났다.**

| 특징 | 기대 | 실측 | |
|---|---|---|---|
| Center `r_med` | 낮아야 함 | **0.723** (Donut 0.525보다 높다) | ✗ |
| Scratch `elong` | 커야 함 | **1.16** (Center 1.08과 차이 없음) | ✗ |
| Loc `cluster` | Random보다 높아야 함 | **0.227 < Random 0.299** | ✗ |

### 원인 — 배경 산발이 지배한다

**면적은 `r²`로 커진다.** 유효 die의 대부분이 바깥쪽에 있으므로,
*"불량 die의 median r"*은 **패턴이 아니라 배경 산발의 분포**를 잰다.
Center 웨이퍼도 중심 클러스터보다 배경 불량 die 수가 많으면 median이 밖으로 끌린다.
`elong`·`cluster`도 같은 이유로 배경에 희석된다.

### 검증 — 패턴은 실재한다. 특징 정의가 틀린 것이다

반경 구간별 **면적 정규화 불량률**로 다시 재니:

| 클래스 | r<0.33 | 0.33~0.66 | r>0.66 | 안/밖 비 |
|---|---|---|---|---|
| **Center** | **0.432** | 0.157 | 0.133 | **3.24** |
| **Donut** | 0.276 | **0.476** | 0.181 | 1.53 (중간 피크) |
| Loc | 0.111 | 0.101 | 0.099 | 1.12 (평탄) |
| Scratch | 0.058 | 0.048 | 0.070 | 0.83 |
| Edge-Ring | 0.052 | 0.040 | **0.155** | 0.34 |
| none | 0.084 | 0.065 | 0.100 | 0.84 |

**Center는 중심이 바깥의 3.24배다. 패턴은 명확히 있다.**

### 교훈 — `edge_contrast`가 통한 이유와 같다

`edge_contrast`는 **내부 층을 기준으로 한 비**였기 때문에 배경 밀도에 불변이었다
(밀도 0.02→0.35에서 1.13→0.98). **형상 특징에도 같은 설계가 필요하다** —
절대 통계가 아니라 **웨이퍼 내부 기준선 대비 대조비**로 정의한다.

## 이 파일의 특징 2종 (회전 불변)

    F1a  radial_contrast = (r<R1 불량률) / (r>R2 불량률)     ← Center 겨냥
    F1b  mid_peak        = (R1~R2 불량률) / max(안, 밖)      ← Donut 겨냥

Laplace 평활 α=1 (`edge_contrast`와 동일). **경계는 물리로 정한다** —
R1=1/3, R2=2/3은 유효 반경의 3등분이며 성능을 보고 고르지 않았다 (§3-9).
"""
import numpy as np

import config
from azimuth import auc

CACHE = config.DATA_PROCESSED / "radial_feats.npz"
R1, R2 = 1.0 / 3.0, 2.0 / 3.0
ALPHA = 1.0


def radial_feats(m, r1=R1, r2=R2, alpha=ALPHA):
    a = np.asarray(m)
    valid = a != config.VAL_OUTSIDE
    if not valid.any():
        return np.nan, np.nan
    rr, cc = np.nonzero(valid)
    cy, cx = (rr.min() + rr.max()) / 2, (cc.min() + cc.max()) / 2
    hy = max((rr.max() - rr.min()) / 2, 1e-9)
    hx = max((cc.max() - cc.min()) / 2, 1e-9)
    r = np.hypot((rr - cy) / hy, (cc - cx) / hx)
    f = a[rr, cc] == config.VAL_FAIL

    def rate(mask):
        n = int(mask.sum())
        if n == 0:
            return np.nan
        return (int(f[mask].sum()) + alpha) / (n + 2 * alpha)

    inn, mid, out = rate(r < r1), rate((r >= r1) & (r < r2)), rate(r >= r2)
    if not np.isfinite(inn) or not np.isfinite(out) or not np.isfinite(mid):
        return np.nan, np.nan
    return inn / out, mid / max(inn, out)


def build(cls_order, idx_order):
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=True) as z:
            return {k: z[k] for k in z.keys()}
    store = {}
    for c in config.PATTERN_CLASSES:
        with np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True) as z:
            store[c] = z["wafer_maps"]
    rc = np.full(len(cls_order), np.nan)
    mp = np.full(len(cls_order), np.nan)
    for i, (c, j) in enumerate(zip(cls_order, idx_order)):
        rc[i], mp[i] = radial_feats(store[c][j])
        if (i + 1) % 5000 == 0:
            print(f"  {i+1:,}/{len(cls_order):,}", flush=True)
    d = dict(radial_contrast=rc, mid_peak=mp)
    np.savez_compressed(CACHE, **d)
    print(f"  캐시 저장: {CACHE}")
    return d


def main():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        cls, idx = z["cls"].astype(str), z["idx_in_cls"]
    d = build(cls, idx)
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        old = {k: z[k] for k in z.keys()}

    print("\n" + "=" * 78)
    print("[1] 클래스별 분포 — 대조비로 다시 재면 패턴이 보이는가")
    print("=" * 78)
    print(f"  {'클래스':<12}{'radial_contrast':>18}{'mid_peak':>11}"
          f"{'(기존) r_med':>14}")
    for c in config.PATTERN_CLASSES:
        m = cls == c
        print(f"  {c:<12}{np.nanmedian(d['radial_contrast'][m]):>18.2f}"
              f"{np.nanmedian(d['mid_peak'][m]):>11.2f}"
              f"{'—':>14}")
    print("\n  Center가 크고 Edge 계열이 1 미만이면 F1a가 작동하는 것이다.")
    print("  Donut의 mid_peak이 1을 넘으면 F1b가 '중간 링'을 잡는 것이다.")

    # ── [2] 이동한 병목에 실제로 듣는가 ───────────────────
    print("\n" + "=" * 78)
    print("[2] 이동한 병목 쌍에 듣는가 — |AUC-0.5| (기존 3종과 같은 조건)")
    print("=" * 78)
    pairs = [("Center", "Loc"), ("Loc", "Scratch"), ("Center", "Edge-Loc"),
             ("Donut", "Loc"), ("Center", "Scratch"), ("Edge-Loc", "Loc")]
    print(f"  {'쌍':<24}{'cov':>7}{'ctr':>7}{'CV':>7}"
          f"{'F1a':>8}{'F1b':>8}{'최강':>10}")
    for p, n in pairs:
        mp_, mn = cls == p, cls == n
        vals = {}
        for k, arr in (("cov", old["cov"]), ("ctr", old["ctr"]), ("cv", old["cv"]),
                       ("F1a", d["radial_contrast"]), ("F1b", d["mid_peak"])):
            vals[k] = abs(auc(arr[mp_], arr[mn]) - .5)
        best = max(vals, key=vals.get)
        print(f"  {p+' ↔ '+n:<24}" + "".join(f"{vals[k]:>7.3f}" if k in ("cov","ctr","cv")
              else f"{vals[k]:>8.3f}" for k in ("cov","ctr","cv","F1a","F1b"))
              + f"{best:>10}")
    print("\n  **기존 3종이 최약이던 쌍에서 F1a·F1b가 이기면 도입 근거가 된다.**")
    print("  (여기서는 쌍별 판별력만 본다 — 8종 다중분류 증분은 별도 확인)")


if __name__ == "__main__":
    main()
