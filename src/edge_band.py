"""가장자리 밴드를 die 층 수로 정의한다 (D-007 재정의).

실행:
    python src/edge_band.py

배경 (docs/w2_confound.md):
    기존 `R_CUT=0.8`(정규화 반경 비율)은 맵 크기에 따라 완전히 다른 것을 쟀다.
    링 두께는 die 1~2개로 고정인데, r>=0.8 밴드는 25x27에서 172 die,
    100x105에서 2,988 die다. die 수가 600배 차이 나는 데이터에서
    반경 *비율* 기준은 성립하지 않는다.

    -> 밴드를 **유효 영역의 최외곽 k개 die 층**으로 정의한다.
       합성 검증에서 맵 크기 의존성이 3.4배 -> 6%로 줄었다.

이 스크립트가 하는 일:
    [1] Edge-Ring의 층별 불량률 프로파일을 재서 **링이 실제로 몇 층인지** 측정
        -> EDGE_LAYERS를 성능이 아니라 물리로 정한다 (D-007)
    [2] k를 바꿔가며 AUC 곡선 전량 보고 (민감도 확인용, 승자만 쓰지 않는다)
    [3] 재정의 후에도 맵 크기 의존성이 남는지 확인
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from azimuth import auc

MAX_LAYERS = 8          # 프로파일을 몇 층까지 벗겨볼지
MIN_FAIL = 12           # D-008
SUBSAMPLE_N = 20        # D-008
N_WEIGHT_BINS = 36      # 가중치 정규화용 (대표 지표 CV는 bin 비의존)


def _erode(mask):
    """8-이웃 침식. 이웃이 모두 True인 셀만 남긴다 (scipy 없이)."""
    pad = np.pad(mask, 1, constant_values=False)
    out = np.ones_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= pad[1 + dy:1 + dy + mask.shape[0],
                       1 + dx:1 + dx + mask.shape[1]]
    return out


def layer_masks(m, k):
    """최외곽부터 k개 층의 마스크를 순서대로 돌려준다."""
    valid = np.asarray(m) != config.VAL_OUTSIDE
    cur, out = valid.copy(), []
    for _ in range(k):
        if not cur.any():
            out.append(np.zeros_like(valid)); continue
        er = _erode(cur)
        out.append(cur & ~er)
        cur = er
    return out


def edge_band(m, k):
    """최외곽 k개 die 층을 합친 밴드 마스크."""
    ls = layer_masks(m, k)
    band = np.zeros_like(ls[0])
    for l in ls:
        band |= l
    return band


def layer_profile(m, max_k=MAX_LAYERS):
    """층별 (불량 die 수, 전체 die 수). 링 두께 측정용."""
    a = np.asarray(m)
    fail = a == config.VAL_FAIL
    return [(int((fail & l).sum()), int(l.sum())) for l in layer_masks(a, max_k)]


def coverage(m, k):
    """밴드 내 불량 die 비율. '둘레를 얼마나 덮었나' (D-010)."""
    a = np.asarray(m)
    b = edge_band(a, k)
    n = int(b.sum())
    if n == 0:
        return np.nan
    return float(((a == config.VAL_FAIL) & b).sum()) / n


def band_circular_variance(m, k, min_fail=MIN_FAIL, n_bins=N_WEIGHT_BINS,
                           subsample=None, rng=None):
    """밴드 위에서 계산한 가중 circular variance. '어떻게 흩어졌나'.

    가중치는 1/(그 방위각 구간의 밴드 die 수) — 격자 기하 위장을 상쇄한다.
    회전 불변량이므로 notch 방향 정보가 없어도 유효하다.
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
    theta = np.arctan2((ii - cy) / hy, (jj - cx) / hx)
    is_fail = a[ii, jj] == config.VAL_FAIL
    if is_fail.sum() < min_fail:
        return np.nan

    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(theta, edges) - 1, 0, n_bins - 1)
    n_per_bin = np.bincount(idx, minlength=n_bins).astype(float)

    f_idx, f_th = idx[is_fail], theta[is_fail]
    if subsample is not None:
        if f_th.size < subsample:
            return np.nan
        rng = rng or np.random.default_rng(config.SEED)
        pick = rng.choice(f_th.size, subsample, replace=False)
        f_idx, f_th = f_idx[pick], f_th[pick]

    w = 1.0 / n_per_bin[f_idx]
    ws = w.sum()
    if ws <= 0:
        return np.nan
    rbar = np.hypot((w * np.cos(f_th)).sum(), (w * np.sin(f_th)).sum()) / ws
    return 1.0 - rbar


# ─────────────────────────────────────────────────────────
def load(cls, cap=None, rng=None):
    with np.load(config.DATA_PROCESSED / f"{cls}.npz", allow_pickle=True) as z:
        maps = z["wafer_maps"]
    if cap and len(maps) > cap:
        maps = maps[(rng or np.random.default_rng(config.SEED))
                    .choice(len(maps), cap, replace=False)]
    return maps


def main():
    a, b = config.TARGET_PAIR                 # Edge-Loc, Edge-Ring
    rng = np.random.default_rng(config.SEED)
    M = {a: load(a, 3000, rng), b: load(b, 3000, rng),
         config.NONE_CLASS: load(config.NONE_CLASS, 3000, rng)}
    for c, v in M.items():
        print(f"{c:<12} {len(v):>6,}장")

    # ── [1] 층별 불량률 프로파일 → 링 두께 ────────────────
    print("\n" + "=" * 70)
    print("[1] 층별 불량률 — 링이 실제로 몇 층인가 (D-007을 물리로 정한다)")
    print("=" * 70)
    prof = {}
    for c in (b, a, config.NONE_CLASS):
        P = np.array([layer_profile(m) for m in M[c]], dtype=float)  # (n, L, 2)
        rate = P[:, :, 0].sum(0) / np.maximum(P[:, :, 1].sum(0), 1)
        prof[c] = rate
    base = prof[config.NONE_CLASS]
    print(f"{'층':>4}" + "".join(f"{c:>12}" for c in (b, a, config.NONE_CLASS))
          + f"{'Ring 초과분':>13}")
    for i in range(MAX_LAYERS):
        exc = prof[b][i] - base[i]
        print(f"{i+1:>4}" + "".join(f"{prof[c][i]:>12.3f}"
              for c in (b, a, config.NONE_CLASS)) + f"{exc:>13.3f}")
    exc = prof[b] - base
    k_phys = int(np.sum(exc > exc[0] * 0.25))       # 1층 초과분의 25% 이상 유지되는 층수
    print(f"\n  1층 초과분의 25% 이상을 유지하는 층 수 = {k_phys}")
    print(f"  --> EDGE_LAYERS 물리 기반 후보: k = {k_phys}")
    print("      (성능이 아니라 '링이 실제로 차지하는 두께'로 정한 값이다)")

    # ── [2] k 민감도 — 곡선 전량 보고 ─────────────────────
    print("\n" + "=" * 70)
    print("[2] k 민감도 — 승자만 쓰지 않고 곡선 전체를 본다")
    print("=" * 70)
    ks = [1, 2, 3, 4, 5, 6]
    print(f"{'k':>3}{'밴드die 중앙값(Ring/Loc)':>26}{'AUC cov':>10}{'AUC CV':>9}"
          f"{'AUC CV(n고정)':>15}{'none CV':>10}")
    res = {}
    for k in ks:
        cov = {c: np.array([coverage(m, k) for m in M[c]]) for c in M}
        cv = {c: np.array([band_circular_variance(m, k) for m in M[c]]) for c in M}
        cvf = {c: np.array([band_circular_variance(m, k, subsample=SUBSAMPLE_N,
               rng=np.random.default_rng(config.SEED)) for m in M[c]]) for c in M}
        nb = {c: np.median([edge_band(m, k).sum() for m in M[c]]) for c in (b, a)}
        res[k] = (cov, cv, cvf)
        print(f"{k:>3}{f'{nb[b]:.0f} / {nb[a]:.0f}':>26}"
              f"{auc(cov[b],cov[a]):>10.3f}{auc(cv[b],cv[a]):>9.3f}"
              f"{auc(cvf[b],cvf[a]):>15.3f}"
              f"{np.nanmedian(cv[config.NONE_CLASS]):>10.3f}")
    print("\n  곡선이 평탄하면 결과가 k 선택의 산물이 아니라는 강한 증거다.")
    print("  none CV가 1에 가까워지면 sanity check 개선 (현재 r>=0.8 기준 0.855)")

    # ── [3] 재정의 후 맵 크기 의존성 ──────────────────────
    k = k_phys
    print("\n" + "=" * 70)
    print(f"[3] k={k}에서 맵 크기 의존성이 남는가")
    print("=" * 70)
    sizes = {c: np.array([int((np.asarray(m) != 0).sum()) for m in M[c]]) for c in (b, a)}
    cov, cv, _ = res[k]
    edges = np.percentile(np.concatenate([sizes[a], sizes[b]]), [0, 25, 50, 75, 100])
    print(f"{'유효die 범위':>20}{'n_Ring':>9}{'n_Loc':>8}{'cov Ring':>10}"
          f"{'cov Loc':>9}{'AUC cov':>9}{'AUC CV':>9}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = {c: (sizes[c] >= lo) & (sizes[c] < hi) for c in (b, a)}
        if sel[b].sum() < 30 or sel[a].sum() < 30:
            continue
        print(f"{f'{int(lo):,}~{int(hi):,}':>20}{sel[b].sum():>9,}{sel[a].sum():>8,}"
              f"{np.nanmedian(cov[b][sel[b]]):>10.3f}{np.nanmedian(cov[a][sel[a]]):>9.3f}"
              f"{auc(cov[b][sel[b]],cov[a][sel[a]]):>9.3f}"
              f"{auc(cv[b][sel[b]],cv[a][sel[a]]):>9.3f}")
    print("\n  층별 coverage 값이 크기에 따라 평탄하면 교란이 제거된 것이다.")
    print("  (r>=0.8 기준에서는 Ring coverage가 0.855 -> 0.248로 3.4배 변동했다)")

    # ── 그림 ──────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for c, col in ((b, "#1976d2"), (a, "#d32f2f"), (config.NONE_CLASS, "#888")):
        ax[0].plot(range(1, MAX_LAYERS + 1), prof[c], "o-", color=col, label=c)
    ax[0].set_xlabel("layer index (1 = outermost)")
    ax[0].set_ylabel("failure rate"); ax[0].set_title("Layer-wise failure rate")
    ax[0].legend(fontsize=8); ax[0].axvline(k_phys + .5, ls="--", c="k", lw=.8)

    ax[1].plot(ks, [auc(res[t][0][b], res[t][0][a]) for t in ks], "o-", label="coverage")
    ax[1].plot(ks, [auc(res[t][1][b], res[t][1][a]) for t in ks], "s-", label="CV")
    ax[1].plot(ks, [auc(res[t][2][b], res[t][2][a]) for t in ks], "^-", label="CV (n fixed)")
    ax[1].axhline(.5, ls="--", c="gray"); ax[1].set_xlabel("EDGE_LAYERS k")
    ax[1].set_ylabel("AUC"); ax[1].set_title("Sensitivity to k"); ax[1].legend(fontsize=8)

    for c, col in ((a, "#d32f2f"), (b, "#1976d2")):
        v = cov[c][~np.isnan(cov[c])]
        ax[2].hist(v, bins=np.linspace(0, 1, 41), alpha=.55, density=True,
                   color=col, label=c)
    ax[2].set_xlabel(f"coverage (k={k})"); ax[2].set_title("Coverage separation")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    out = config.FIGURES / "edge_band.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
