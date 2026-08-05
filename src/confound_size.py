"""W2 — 맵 크기 교란 검증.

실행:
    python src/confound_size.py

배경 (docs/w2_azimuth.md, eda_size.py 결과):
    Edge-Ring의 맵이 Edge-Loc보다 2배 이상 크다 (가장자리 die 중앙값 798 vs 366).
    가장자리 die가 많으면 방위각 해상도가 높아 circular variance가 1에 가깝게 나온다
    (함정 3: 표본 수 편향). 따라서 기존 AUC 0.927 / 0.743이 **대칭성이 아니라
    맵 크기 차이**의 반영일 수 있다.

    개수 통제(SUBSAMPLE_N)로는 이것이 잡히지 않는다. 불량 die 개수는 맞췄지만
    가중치 계산에 쓰는 '구간별 전체 die 수'는 여전히 2배 차이이기 때문이다.

확인할 것:
    [1] 웨이퍼별 coverage 비율이 정말 두 클래스에서 같은가
    [2] 가장자리 die 수만으로 클래스가 얼마나 예측되는가 (= 교란의 크기)
    [3] 크기를 층화하면 CV의 AUC가 유지되는가  <- 이 프로젝트의 최대 관문
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from azimuth import (polar_coords, weighted_circular_variance,
                     R_CUT, MIN_FAIL, MIN_DIES, SUBSAMPLE_N, auc)


def edge_stats(m):
    """(가장자리 die 수, 가장자리 불량 die 수, coverage 비율)"""
    r, th, f = polar_coords(m)
    if r is None:
        return 0, 0, np.nan
    k = r >= R_CUT
    n, nf = int(k.sum()), int(f[k].sum())
    return n, nf, (nf / n if n > 0 else np.nan)


def strat_auc(pos, neg, pos_key, neg_key, edges):
    """층화 AUC. 각 층에서 AUC를 구하고 층 크기로 가중 평균한다.

    층 안에서는 교란변수(가장자리 die 수)가 거의 고정되므로,
    여기서 살아남는 분리력이 교란과 독립적인 신호다.
    """
    rows, wsum, asum = [], 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        p = pos[(pos_key >= lo) & (pos_key < hi)]
        n = neg[(neg_key >= lo) & (neg_key < hi)]
        p = p[~np.isnan(p)]; n = n[~np.isnan(n)]
        if len(p) < 30 or len(n) < 30:
            rows.append((lo, hi, len(p), len(n), np.nan)); continue
        a = auc(p, n)
        w = min(len(p), len(n))          # 작은 쪽이 유효 표본
        rows.append((lo, hi, len(p), len(n), a))
        wsum += w; asum += a * w
    return rows, (asum / wsum if wsum > 0 else np.nan)


def main():
    a, b = config.TARGET_PAIR                     # Edge-Loc, Edge-Ring
    rng = np.random.default_rng(config.SEED)
    D = {}
    for cls in (a, b):
        with np.load(config.DATA_PROCESSED / f"{cls}.npz", allow_pickle=True) as z:
            maps = z["wafer_maps"]
        st = np.array([edge_stats(m) for m in maps])
        D[cls] = dict(
            maps=maps,
            edge_n=st[:, 0], edge_fail=st[:, 1], coverage=st[:, 2],
            cv=np.array([weighted_circular_variance(m, r_cut=R_CUT) for m in maps]),
            cv_fix=np.array([weighted_circular_variance(
                m, r_cut=R_CUT, subsample=SUBSAMPLE_N, rng=rng) for m in maps]),
        )
        print(f"{cls:<12} {len(maps):>6,}장 계산 완료")

    # ── [1] coverage 비율 ─────────────────────────────────
    print("\n" + "=" * 74)
    print("[1] 웨이퍼별 coverage = 가장자리 불량 die / 가장자리 전체 die")
    print("=" * 74)
    for cls in (a, b):
        c = D[cls]["coverage"]; c = c[~np.isnan(c)]
        print(f"  {cls:<12} 중앙값 {np.median(c):.3f}   "
              f"Q1 {np.percentile(c,25):.3f}  Q3 {np.percentile(c,75):.3f}")
    auc_cov = auc(D[b]["coverage"], D[a]["coverage"])
    print(f"  >> AUC({b} > {a}) = {auc_cov:.3f}")
    print("     0.5에 가까우면 coverage는 두 클래스를 구분하지 못한다는 뜻 —")
    print("     즉 '개수 차이'의 정체가 맵 크기라는 근거가 된다")

    # ── [2] 교란의 크기 ───────────────────────────────────
    print("\n" + "=" * 74)
    print("[2] 교란 크기 — 가장자리 die 수(맵 크기)만으로 분류하면")
    print("=" * 74)
    auc_size = auc(D[b]["edge_n"], D[a]["edge_n"])
    auc_cnt = auc(D[b]["edge_fail"], D[a]["edge_fail"])
    print(f"  가장자리 die 수만으로   AUC = {auc_size:.3f}   <- 교란의 크기")
    print(f"  가장자리 불량 수만으로  AUC = {auc_cnt:.3f}")
    print(f"  CV (r>={R_CUT})          AUC = {auc(D[b]['cv'], D[a]['cv']):.3f}")
    print(f"  CV (개수 고정)          AUC = {auc(D[b]['cv_fix'], D[a]['cv_fix']):.3f}")

    # ── [3] 크기 층화 ★ ───────────────────────────────────
    pooled = np.concatenate([D[a]["edge_n"], D[b]["edge_n"]])
    edges = np.unique(np.percentile(pooled, [0, 20, 40, 60, 80, 100])).astype(int)
    print("\n" + "=" * 74)
    print("[3] ★ 가장자리 die 수 층화 후 AUC — 층 안에서는 맵 크기가 거의 고정")
    print("=" * 74)
    for name, key in (("CV (r>=cut)", "cv"), ("CV (개수 고정)", "cv_fix"),
                      ("coverage", "coverage")):
        rows, pooled_auc = strat_auc(
            D[b][key], D[a][key], D[b]["edge_n"], D[a]["edge_n"], edges)
        print(f"\n  --- {name} ---")
        print(f"  {'edge die 범위':>18} {'n_'+b:>10} {'n_'+a:>10} {'AUC':>8}")
        for lo, hi, nb, na, av in rows:
            sa = f"{av:.3f}" if not np.isnan(av) else "표본부족"
            print(f"  {str(lo)+'~'+str(hi):>18} {nb:>10,} {na:>10,} {sa:>8}")
        print(f"  {'>> 층화 통합 AUC':>18} {'':>10} {'':>10} {pooled_auc:>8.3f}")

    print("\n" + "=" * 74)
    print("[판정]")
    print("=" * 74)
    _, sa_cv = strat_auc(D[b]["cv"], D[a]["cv"],
                         D[b]["edge_n"], D[a]["edge_n"], edges)
    raw = auc(D[b]["cv"], D[a]["cv"])
    print(f"  층화 전 AUC {raw:.3f}  ->  층화 후 {sa_cv:.3f}   (교란 AUC {auc_size:.3f})")
    if not np.isnan(sa_cv) and sa_cv >= 0.70:
        print("  ==> 맵 크기를 통제해도 분리가 유지된다. 대칭성이 실제 신호다.")
        print("      §3 주장 유지. 초록에 '맵 크기 층화 조건'을 병기한다.")
    elif not np.isnan(sa_cv) and sa_cv >= 0.60:
        print("  ==> 분리가 상당히 약해졌다. 겉보기 판별력의 큰 부분이 맵 크기였다.")
        print("      §3 주장을 '부분적'으로 수정하고 층화 수치를 헤드라인으로 쓴다.")
    else:
        print("  ==> 맵 크기를 통제하면 분리가 사라진다.")
        print("      F3의 판별력은 대칭성이 아니라 맵 크기의 대리였다는 뜻이다.")
        print("      §3 주장을 다시 쓰고 플랜 B(coverage + 보정 런 지표)로 전환한다.")

    # ── 그림 ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(D[a]["edge_n"], D[a]["cv"], s=3, alpha=.25, c="#d32f2f", label=a)
    axes[0].scatter(D[b]["edge_n"], D[b]["cv"], s=3, alpha=.25, c="#1976d2", label=b)
    axes[0].set_xscale("log"); axes[0].set_xlabel("edge dies (log)")
    axes[0].set_ylabel("weighted circular variance")
    axes[0].set_title("CV vs map size"); axes[0].legend(fontsize=8, markerscale=3)

    bins = np.linspace(0, 1, 41)
    for cls, c in ((a, "#d32f2f"), (b, "#1976d2")):
        v = D[cls]["coverage"]; v = v[~np.isnan(v)]
        axes[1].hist(v, bins=bins, alpha=.55, density=True, color=c, label=cls)
    axes[1].set_xlabel("coverage (edge fail / edge dies)")
    axes[1].set_title(f"Coverage  AUC={auc_cov:.3f}"); axes[1].legend(fontsize=8)

    rows, _ = strat_auc(D[b]["cv"], D[a]["cv"], D[b]["edge_n"], D[a]["edge_n"], edges)
    xs = [f"{lo}~{hi}" for lo, hi, _, _, _ in rows]
    ys = [r[4] for r in rows]
    axes[2].bar(range(len(ys)), ys, color="#455a64")
    axes[2].axhline(0.5, ls="--", c="gray"); axes[2].axhline(raw, ls=":", c="#1976d2")
    axes[2].set_xticks(range(len(xs))); axes[2].set_xticklabels(xs, rotation=30, fontsize=7)
    axes[2].set_ylim(0.4, 1.0); axes[2].set_ylabel("AUC within stratum")
    axes[2].set_title("CV AUC by size stratum")
    fig.tight_layout()
    out = config.FIGURES / "confound_size.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
