"""coverage(k=1)가 라벨 정의의 동어반복인지 검증한다.

실행:
    python src/verify_coverage.py

우려:
    Edge-Ring의 라벨 정의가 "최외곽에 링"인데, coverage는 "최외곽 1층의 불량 비율"이다.
    AUC 0.971은 물리적 발견이 아니라 **라벨 정의를 그대로 되읽은 것**일 수 있다.

검증 전략 — 라벨을 쓰지 않는 예측을 세워 확인한다:

  [1] lot 내 재현성  ★ 가장 결정적
      Edge-Ring의 원인이 챔버 조건 편차(gas flow, plasma density, 척 온도 구배)라면
      **같은 lot의 웨이퍼들이 비슷한 coverage를 가져야 한다.** lot 정보는 라벨과
      완전히 독립이므로, 여기서 재현성이 나오면 coverage는 물리를 잡고 있는 것이다.
      Edge-Loc(국부·확률적 원인)은 재현성이 낮아야 한다.

  [2] 8종 전체 순서
      coverage가 '가장자리 불량 비율'을 정확히 재고 있다면 클래스 순서가
      물리적 예상과 맞아야 한다. **특히 Donut** — 링이지만 안쪽이므로
      최외곽 coverage는 낮아야 한다. Edge-Ring과 구분되면 지표가
      '링'이 아니라 '가장자리'를 정확히 잡고 있다는 뜻이다.

  [3] 라벨 불일치 케이스
      동어반복이라면 지표와 라벨이 어긋나는 웨이퍼가 거의 없어야 한다.
      **어긋나는 케이스가 존재하고 그것이 라벨 노이즈로 설명되면**,
      지표는 라벨과 독립적인 정보를 갖는 것이다.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import config
from edge_band import coverage
from azimuth import auc

K = 1
CMAP = ListedColormap(["#ffffff", "#dcdcdc", "#d32f2f"])


def load(cls):
    with np.load(config.DATA_PROCESSED / f"{cls}.npz", allow_pickle=True) as z:
        return z["wafer_maps"], z["lot_name"].astype(str)


def main():
    a, b = config.TARGET_PAIR
    rng = np.random.default_rng(config.SEED)

    # ── [1] lot 내 재현성 ─────────────────────────────────
    print("=" * 72)
    print("[1] lot 내 재현성 — coverage가 lot 단위로 뭉치는가 (라벨 독립 검증)")
    print("=" * 72)
    print("  Edge-Ring 원인이 챔버 조건이면 같은 lot 웨이퍼끼리 비슷해야 한다.\n")
    print(f"{'클래스':<12}{'lot(2장+)':>10}{'같은lot 차이':>14}{'무작위 차이':>13}{'비율':>8}")
    lot_res = {}
    for cls in (b, a):
        maps, lots = load(cls)
        cov = np.array([coverage(m, K) for m in maps])
        ok = ~np.isnan(cov)
        cov, lots = cov[ok], lots[ok]

        same, n_lot = [], 0
        for lt in np.unique(lots):
            v = cov[lots == lt]
            if len(v) < 2:
                continue
            n_lot += 1
            idx = rng.choice(len(v), size=(min(20, len(v) * (len(v) - 1) // 2), 2))
            idx = idx[idx[:, 0] != idx[:, 1]]
            same.extend(np.abs(v[idx[:, 0]] - v[idx[:, 1]]))
        pairs = rng.choice(len(cov), size=(4000, 2))
        rand = np.abs(cov[pairs[:, 0]] - cov[pairs[:, 1]])
        ms, mr = np.median(same), np.median(rand)
        lot_res[cls] = (ms, mr, ms / mr)
        print(f"{cls:<12}{n_lot:>10,}{ms:>14.4f}{mr:>13.4f}{ms/mr:>8.3f}")

    print("\n  비율 < 1 이면 같은 lot 웨이퍼가 서로 더 비슷하다 = lot 단위 재현성")
    rr, rl = lot_res[b][2], lot_res[a][2]
    if rr < 0.85 and rr < rl:
        print(f"  ==> Edge-Ring({rr:.3f})이 Edge-Loc({rl:.3f})보다 lot 재현성이 높다.")
        print("      챔버 조건(lot 단위) 원인 가설과 일치. **라벨 정의로는 설명 안 되는")
        print("      구조를 coverage가 잡고 있다** -> 동어반복 아님")
    else:
        print(f"  ==> Edge-Ring {rr:.3f} / Edge-Loc {rl:.3f}.")
        print("      기대한 대비가 나오지 않았다. 해석을 보류하고 원인을 따로 본다.")

    # ── [2] 8종 전체 순서 ─────────────────────────────────
    print("\n" + "=" * 72)
    print("[2] 8종 전체 coverage 순서 — 물리적 예상과 맞는가")
    print("=" * 72)
    allcov = {}
    for cls in config.ALL_CLASSES:
        maps, _ = load(cls)
        if len(maps) > 2500:
            maps = maps[rng.choice(len(maps), 2500, replace=False)]
        v = np.array([coverage(m, K) for m in maps])
        allcov[cls] = v[~np.isnan(v)]
    order = sorted(allcov, key=lambda c: -np.median(allcov[c]))
    print(f"{'순위':>4}{'클래스':<14}{'coverage 중앙값':>16}{'n':>8}")
    for i, c in enumerate(order, 1):
        print(f"{i:>4}{c:<14}{np.median(allcov[c]):>16.3f}{len(allcov[c]):>8,}")

    print("\n  ★ 핵심 확인: Donut은 '링'이지만 안쪽이므로 최외곽 coverage가 낮아야 한다.")
    d_med = np.median(allcov.get("Donut", [np.nan]))
    r_med = np.median(allcov[b])
    print(f"     Donut {d_med:.3f}  vs  Edge-Ring {r_med:.3f}   "
          f"AUC(Ring>Donut) = {auc(allcov[b], allcov.get('Donut', np.array([]))):.3f}")
    if d_med < r_med * 0.6:
        print("     ==> Donut이 확실히 낮다. 지표는 '링'이 아니라 '가장자리'를 잡고 있다.")
        print("         라벨에 '링'이 들어간 두 클래스를 구분하므로 동어반복이 아니다.")
    else:
        print("     ==> Donut과 구분이 약하다. 지표가 '가장자리'를 정확히 잡는지 재검토.")

    # ── [3] 라벨 불일치 케이스 ────────────────────────────
    print("\n" + "=" * 72)
    print("[3] 라벨과 지표가 어긋나는 웨이퍼 — 동어반복이면 거의 없어야 한다")
    print("=" * 72)
    cov_b, cov_a = allcov[b], allcov[a]
    thr_lo = np.percentile(cov_b, 5)
    thr_hi = np.percentile(cov_a, 95)
    n_b_low = np.mean(cov_b < thr_hi) * 100
    n_a_high = np.mean(cov_a > thr_lo) * 100
    print(f"  Edge-Loc 상위 5% 경계 = {thr_hi:.3f} / Edge-Ring 하위 5% 경계 = {thr_lo:.3f}")
    print(f"  Edge-Ring 인데 Edge-Loc 상위 5%보다 낮은 웨이퍼: {n_b_low:5.1f}%")
    print(f"  Edge-Loc  인데 Edge-Ring 하위 5%보다 높은 웨이퍼: {n_a_high:5.1f}%")
    print("\n  이 케이스들이 존재하면 지표는 라벨을 그대로 읽는 것이 아니다.")
    print("  그림으로 뽑아 라벨 노이즈인지 지표 실패인지 눈으로 확인한다.")

    # 불일치 사례 그림
    maps_b, _ = load(b); maps_a, _ = load(a)
    cb = np.array([coverage(m, K) for m in maps_b])
    ca = np.array([coverage(m, K) for m in maps_a])
    pick_b = np.argsort(np.where(np.isnan(cb), 9, cb))[:8]          # Ring인데 coverage 최저
    pick_a = np.argsort(-np.where(np.isnan(ca), -9, ca))[:8]        # Loc인데 coverage 최고
    fig, axes = plt.subplots(2, 8, figsize=(15, 4.2))
    for j, i in enumerate(pick_b):
        axes[0, j].imshow(maps_b[i], cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
        axes[0, j].set_title(f"{cb[i]:.2f}", fontsize=8)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
    for j, i in enumerate(pick_a):
        axes[1, j].imshow(maps_a[i], cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
        axes[1, j].set_title(f"{ca[i]:.2f}", fontsize=8)
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
    axes[0, 0].set_ylabel(f"{b}\nlowest cov", fontsize=8)
    axes[1, 0].set_ylabel(f"{a}\nhighest cov", fontsize=8)
    fig.suptitle("Label vs metric disagreement (label noise or metric failure?)", fontsize=11)
    fig.tight_layout()
    out = config.FIGURES / "coverage_disagreement.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n저장: {out}")

    # 8종 분포 그림
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([allcov[c] for c in order], tick_labels=order, showfliers=False)
    ax.set_ylabel(f"coverage (outermost {K} die layer)")
    ax.set_title("Edge coverage by class")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out2 = config.FIGURES / "coverage_by_class.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"저장: {out2}")


if __name__ == "__main__":
    main()
