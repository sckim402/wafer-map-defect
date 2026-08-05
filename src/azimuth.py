"""방위각 균일성 지표(F3)의 판별력을 전수 데이터로 확인한다.

실행:
    python src/azimuth.py

배경 (docs/w1_first_look.md):
    육안 표본 8장으로는 Edge-Loc과 Edge-Ring을 판정할 수 없었다 (seed 간 불일치).
    대신 전체 웨이퍼에 지표를 계산해 분포를 비교한다.

핵심 질문:
    가중 circular variance가 두 클래스를 분리하는가?
    그리고 계산 범위를 가장자리로 제한하면 분리가 좋아지는가?

두 가지 함정을 처리한다:
  (1) 격자 기하 위장
      직사각 die grid는 각도 구간마다 die 개수가 다르다. 그냥 세면 격자 구조가
      "각도 편중"처럼 보인다. -> 구간별 유효 die 수로 가중한다.
  (2) 좌표계 기준점 부재
      notch 방향 정보가 없어 절대 각도는 의미를 갖지 못한다.
      -> 회전 불변량인 circular variance만 쓴다.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

import config

# 방위각을 몇 등분해 가중치를 계산할지.
# 이 값은 "해상도"가 아니라 "정규화용 구간"이다 — circular variance 자체는
# bin에 의존하지 않는다. (docs/decisions.md D-005)
N_WEIGHT_BINS = 36  # 10도 간격

# 가장자리 영역 정의. 값은 D-007에서 확정한다.
R_CUT = 0.80

# 가장자리 영역에 이만큼도 die가 없으면 계산하지 않는다 (D-006과 연동)
MIN_DIES = 30

# 불량 die가 이보다 적으면 circular variance를 신뢰할 수 없다.
# 표본 수가 작으면 평균 결과 길이 R_bar가 과대추정되어 CV가 0쪽으로 편향된다
# (합성 데이터 검증: 불량 3개 -> CV 0.52, 69개 -> CV 0.92, 이론값은 1).
MIN_FAIL = 12

# 편향 제거용 — 모든 웨이퍼에서 동일한 개수의 불량 die만 뽑아 계산한다.
# "방위각 대칭성"과 "불량 die 개수"를 분리하기 위한 장치다.
SUBSAMPLE_N = 20


def polar_coords(wafer_map):
    """유효 영역을 단위원에 매핑한 (반경, 방위각)을 돌려준다.

    웨이퍼는 원형이지만 배열은 직사각이고 행/열 수도 다르다((25,27) 등).
    유효 영역의 행/열 범위로 각각 정규화하면 타원이 원으로 펴져
    종횡비 보정이 자동으로 된다.
    """
    m = np.asarray(wafer_map)
    valid = m != config.VAL_OUTSIDE
    if not valid.any():
        return None, None, None

    rows, cols = np.nonzero(valid)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    cy, cx = (r0 + r1) / 2.0, (c0 + c1) / 2.0
    hy = max((r1 - r0) / 2.0, 1e-9)
    hx = max((c1 - c0) / 2.0, 1e-9)

    ii, jj = np.nonzero(valid)
    y = (ii - cy) / hy
    x = (jj - cx) / hx

    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)          # -pi ~ pi
    is_fail = m[ii, jj] == config.VAL_FAIL
    return radius, theta, is_fail


def weighted_circular_variance(wafer_map, r_cut=None, n_bins=N_WEIGHT_BINS,
                               min_dies=MIN_DIES, min_fail=MIN_FAIL,
                               subsample=None, rng=None):
    """가중 circular variance. 0에 가까울수록 한 방향에 몰림, 1에 가까울수록 고름.

    가중치는 1 / (그 방위각 구간의 유효 die 수). 격자 기하가 만드는
    각도별 die 수 편차를 상쇄한다.

    subsample: 정수를 주면 불량 die를 그 개수만큼 무작위로 뽑아 계산한다.
               표본 수 편향을 제거해 "대칭성"과 "개수"를 분리한다.
    """
    radius, theta, is_fail = polar_coords(wafer_map)
    if radius is None:
        return np.nan

    if r_cut is not None:
        keep = radius >= r_cut
        radius, theta, is_fail = radius[keep], theta[keep], is_fail[keep]

    if radius.size < min_dies or is_fail.sum() < min_fail:
        return np.nan

    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(theta, edges) - 1, 0, n_bins - 1)
    n_valid_per_bin = np.bincount(idx, minlength=n_bins).astype(float)

    f_idx = idx[is_fail]
    f_theta = theta[is_fail]

    if subsample is not None:
        if f_theta.size < subsample:
            return np.nan
        rng = rng or np.random.default_rng(config.SEED)
        pick = rng.choice(f_theta.size, subsample, replace=False)
        f_idx, f_theta = f_idx[pick], f_theta[pick]

    w = 1.0 / n_valid_per_bin[f_idx]
    wsum = w.sum()
    if wsum <= 0:
        return np.nan
    rbar = np.hypot((w * np.cos(f_theta)).sum(),
                    (w * np.sin(f_theta)).sum()) / wsum
    return 1.0 - rbar


def n_fail_edge(wafer_map, r_cut=R_CUT):
    """가장자리 영역의 불량 die 수. F3의 경쟁 가설을 검증하기 위한 대조 특징.

    F3의 판별력이 사실 이 개수에서 오는 것이라면, 물리적 해석(대칭성)이
    아니라 단순 개수를 보고 있는 것이다. 반드시 따로 확인해야 한다.
    """
    radius, theta, is_fail = polar_coords(wafer_map)
    if radius is None:
        return np.nan
    keep = radius >= r_cut
    return float(is_fail[keep].sum())


def load_maps(cls):
    path = config.DATA_PROCESSED / f"{cls}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        return z["wafer_maps"]


def auc(pos, neg):
    """단일 특징의 분리도. 0.5=구분 못함, 1.0=완전 분리.

    Mann-Whitney U 통계량 기반 — 순위만 쓰므로 분포 가정이 없다.
    """
    pos = pos[~np.isnan(pos)]
    neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def main():
    a, b = config.TARGET_PAIR          # ("Edge-Loc", "Edge-Ring")
    targets = [a, b, config.NONE_CLASS]
    rng = np.random.default_rng(config.SEED)

    data = {}
    for cls in targets:
        maps = load_maps(cls)
        if maps is None:
            raise SystemExit(f"[중단] {cls}.npz 없음. load_data.py 먼저 실행.")
        data[cls] = maps
        print(f"{cls:<12} {len(maps):>7,}장")

    if len(data[config.NONE_CLASS]) > 5000:      # sanity check용이라 전수는 과하다
        pick = rng.choice(len(data[config.NONE_CLASS]), 5000, replace=False)
        data[config.NONE_CLASS] = data[config.NONE_CLASS][pick]

    # ── 조건 3가지 ────────────────────────────────────────
    #   (1) 전체        : 반경 컷의 효과를 재기 위한 기준선
    #   (2) 가장자리    : 본 설계
    #   (3) 개수 고정   : 대칭성과 개수를 분리한 결정적 조건
    conds = [
        ("전체",            dict(r_cut=None,  subsample=None)),
        (f"r>={R_CUT}",     dict(r_cut=R_CUT, subsample=None)),
        (f"r>={R_CUT}·n고정", dict(r_cut=R_CUT, subsample=SUBSAMPLE_N)),
    ]

    results = {}
    for label, kw in conds:
        print(f"\n--- {label} ---")
        results[label] = {}
        for cls in targets:
            vals = np.array([
                weighted_circular_variance(m, rng=rng, **kw) for m in data[cls]
            ])
            results[label][cls] = vals
            n_ok = int(np.sum(~np.isnan(vals)))
            print(f"  {cls:<12} 계산가능 {n_ok:>6,}/{len(vals):,}  "
                  f"중앙값 {np.nanmedian(vals):.3f}")
        print(f"  >> AUC({b} > {a}) = {auc(results[label][b], results[label][a]):.3f}")

    # ── 경쟁 가설: 판별력이 사실 "불량 die 개수"에서 오는 것 아닌가 ──
    print("\n--- 대조: 가장자리 불량 die 수만으로 ---")
    counts = {cls: np.array([n_fail_edge(m) for m in data[cls]]) for cls in (a, b)}
    for cls in (a, b):
        print(f"  {cls:<12} 중앙값 {np.nanmedian(counts[cls]):.0f}개")
    auc_count = auc(counts[b], counts[a])
    print(f"  >> AUC({b} > {a}) = {auc_count:.3f}")

    auc_fixed = auc(results[conds[2][0]][b], results[conds[2][0]][a])
    print("\n--- 해석 ---")
    print(f"  개수만으로       AUC = {auc_count:.3f}")
    print(f"  개수 고정 후 CV  AUC = {auc_fixed:.3f}")
    if auc_fixed >= 0.65:
        print("  -> 개수를 통제해도 CV가 분리한다. 대칭성이 실제 신호다. 가설 유지")
    else:
        print("  -> 개수를 통제하면 CV의 분리력이 사라진다.")
        print("     F3는 '방위각 대칭성'이 아니라 '불량 die 개수'의 대리변수일 수 있다.")
        print("     이 경우 물리적 해석을 수정해야 한다 — 중요한 결과이므로 반드시 기록.")

    # ── sanity check ──────────────────────────────────────
    print("\n--- sanity check (none은 1에 가까워야 정상) ---")
    for label in results:
        print(f"  {label:<14} none 중앙값 "
              f"{np.nanmedian(results[label][config.NONE_CLASS]):.3f}")

    # ── 그림 ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, (label, _) in zip(axes, conds):
        for cls, color in ((a, "#d32f2f"), (b, "#1976d2")):
            v = results[label][cls]
            v = v[~np.isnan(v)]
            ax.hist(v, bins=bins, alpha=0.55, density=True,
                    label=f"{cls} (n={len(v):,})", color=color)
        ax.set_title(f"{label}   AUC={auc(results[label][b], results[label][a]):.3f}")
        ax.set_xlabel("weighted circular variance\n(0=몰림, 1=고름)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("방위각 균일성 지표의 판별력 — 계산 범위 및 표본 수 통제", fontsize=12)
    fig.tight_layout()
    out = config.FIGURES / "azimuth_separation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
