"""W2 ⓪ — 클래스 × 맵 크기 교차표.

실행:
    python src/eda_size.py

**분할 구현보다 먼저 돌린다.** 한 번에 세 질문에 답한다.

  Q1. Edge-Loc과 Edge-Ring의 맵 크기 분포가 유사한가?
      -> 유사하면 "비원형도 편향이 클래스별로 다르다"는 교란 경로가 대부분 배제된다.
         다르면 층화가 필요하고 분할 설계 자체를 바꿔야 한다.
  Q2. 작은 맵에 특정 클래스가 몰려 있는가?
      -> D-006 딜레마("하한을 올리면 특정 클래스가 통째로 썰린다")의 실재 여부.
  Q3. F3를 계산할 수 있는 웨이퍼가 클래스별로 몇 %인가?
      -> 가장자리 영역(r>=R_CUT)의 불량 die 수가 MIN_FAIL 이상이어야 계산 가능.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from azimuth import polar_coords, R_CUT, MIN_FAIL, MIN_DIES


def stats_for(maps):
    """웨이퍼별 (배열 셀 수, 유효 die 수, 가장자리 die 수, 가장자리 불량 die 수)"""
    cells, valid, edge, edge_fail = [], [], [], []
    for m in maps:
        a = np.asarray(m)
        cells.append(a.size)
        v = a != config.VAL_OUTSIDE
        valid.append(int(v.sum()))
        r, th, f = polar_coords(a)
        if r is None:
            edge.append(0); edge_fail.append(0); continue
        k = r >= R_CUT
        edge.append(int(k.sum()))
        edge_fail.append(int(f[k].sum()))
    return (np.array(cells), np.array(valid),
            np.array(edge), np.array(edge_fail))


def q(x):
    return np.percentile(x, [0, 25, 50, 75, 100]).astype(int)


def mannwhitney_auc(a, b):
    """분포 동일성의 실용적 척도. 0.5면 동일, 0/1이면 완전 분리."""
    allv = np.concatenate([a, b])
    ranks = allv.argsort().argsort() + 1
    u = ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return u / (len(a) * len(b))


def main():
    data = {}
    print("클래스별 통계 계산 중...\n")
    for cls in config.ALL_CLASSES:
        path = config.DATA_PROCESSED / f"{cls}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as z:
            maps = z["wafer_maps"]
        # none은 표본으로 충분 (14만 장 전수는 낭비)
        if cls == config.NONE_CLASS and len(maps) > 8000:
            rng = np.random.default_rng(config.SEED)
            maps = maps[rng.choice(len(maps), 8000, replace=False)]
        data[cls] = stats_for(maps)
        print(f"  {cls:<12} {len(maps):>7,}장 완료")

    # ── 표 1: 유효 die 수 분포 ────────────────────────────
    print("\n" + "=" * 78)
    print("[표 1] 클래스별 유효 die 수 분포")
    print("=" * 78)
    print(f"{'클래스':<12}{'n':>8}{'min':>8}{'Q1':>8}{'중앙값':>9}{'Q3':>8}{'max':>8}")
    for cls, (c, v, e, ef) in data.items():
        s = q(v)
        print(f"{cls:<12}{len(v):>8,}{s[0]:>8,}{s[1]:>8,}{s[2]:>9,}{s[3]:>8,}{s[4]:>8,}")

    # ── Q1: 타깃 쌍의 크기 분포 비교 ──────────────────────
    a, b = config.TARGET_PAIR
    print("\n" + "=" * 78)
    print(f"[Q1] {a} vs {b} — 맵 크기 분포가 유사한가")
    print("=" * 78)
    va, vb = data[a][1], data[b][1]
    auc = mannwhitney_auc(vb, va)
    print(f"  중앙값     {a}: {np.median(va):>7,.0f}   {b}: {np.median(vb):>7,.0f}")
    print(f"  분포 AUC({b} > {a}) = {auc:.3f}")
    if abs(auc - 0.5) < 0.10:
        print("  ==> 두 클래스의 맵 크기 분포가 유사하다.")
        print("      비원형도 편향이 클래스별로 다르게 작용할 경로는 대부분 배제된다.")
        print("      (§6-A 정정 ⓑ의 확인 조건 충족)")
    else:
        print("  ==> 두 클래스의 맵 크기 분포가 다르다. **층화가 필요하다.**")
        print("      분할 설계를 lot + 맵 크기 층화로 바꾸는 것을 검토한다.")

    # ── Q2: 작은 맵의 클래스 구성 ─────────────────────────
    print("\n" + "=" * 78)
    print("[Q2] die 수 구간별 클래스 구성비 (%) — 작은 맵에 특정 클래스가 몰리는가")
    print("=" * 78)
    bins = [0, 300, 700, 1500, 10**9]
    names = ["<300", "300-700", "700-1500", ">=1500"]
    print(f"{'클래스':<12}" + "".join(f"{n:>12}" for n in names))
    for cls, (c, v, e, ef) in data.items():
        idx = np.digitize(v, bins[1:-1])
        pct = [100 * np.mean(idx == i) for i in range(4)]
        print(f"{cls:<12}" + "".join(f"{p:>11.1f}%" for p in pct))
    print("\n  각 행은 그 클래스 내부의 분포다. 특정 클래스가 '<300'에 몰려 있으면")
    print("  하한선을 올릴 때 그 클래스가 통째로 빠진다 (D-006 딜레마).")

    # ── Q3: F3 계산 가능 비율 ─────────────────────────────
    print("\n" + "=" * 78)
    print(f"[Q3] F3 계산 가능 비율 (r>={R_CUT} 영역 die>={MIN_DIES} 이고 불량>={MIN_FAIL})")
    print("=" * 78)
    print(f"{'클래스':<12}{'가장자리die 중앙값':>20}{'가장자리불량 중앙값':>22}{'계산가능':>10}")
    for cls, (c, v, e, ef) in data.items():
        ok = np.mean((e >= MIN_DIES) & (ef >= MIN_FAIL)) * 100
        print(f"{cls:<12}{np.median(e):>20,.0f}{np.median(ef):>22,.0f}{ok:>9.1f}%")
    print("\n  계산 불가 웨이퍼는 버리지 않는다. F3=NaN + F3_available=0 (D-006)")

    # ── 그림 ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = [c for c in config.ALL_CLASSES if c in data]
    axes[0].boxplot([data[c][1] for c in order], labels=order, showfliers=False)
    axes[0].set_yscale("log"); axes[0].set_ylabel("valid dies per wafer (log)")
    axes[0].set_title("Wafer size distribution by class")
    axes[0].tick_params(axis="x", rotation=45)

    for cls, color in ((a, "#d32f2f"), (b, "#1976d2")):
        axes[1].hist(data[cls][1], bins=np.logspace(1.5, 4.5, 40), alpha=0.55,
                     density=True, label=f"{cls} (n={len(data[cls][1]):,})", color=color)
    axes[1].set_xscale("log"); axes[1].set_xlabel("valid dies per wafer (log)")
    axes[1].set_title(f"Target pair size overlap  (AUC={auc:.3f})")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    out = config.FIGURES / "size_by_class.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
