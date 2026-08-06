"""세 가장자리 지표를 **같은 밴드(EDGE_LAYERS=1)**에서 8종 전체로 비교한다.

실행:
    python src/feature_table.py

배경:
    지금까지 세 지표는 서로 다른 밴드에서 계산됐다.
      - coverage / edge_contrast : EDGE_LAYERS=1 (die 층 기준, D-007)
      - circular variance        : r>=0.8 (반경 비율 기준, D-007에서 폐기된 정의)
    밴드가 다르면 "어느 지표가 어디서 실패하는가"를 비교할 수 없다.
    -> 셋을 k=1에서 다시 재고 한 표에 넣는다.

    주의: `edge_band.band_circular_variance`는 이미 k 기반으로 구현돼 있고
    edge_band.py [2]에서 Ring/Loc/none 3종에 대해 k=1~6이 계산됐다.
    **없는 것은 "k=1 CV" 자체가 아니라 "8종 전체의 CV"다.**

이 스크립트가 새로 확인하는 것:
    [1] 8종 CV 분포 — 지금까지 3종(Ring/Loc/none)만 봤다 (작업지침 §3-8)
    [2] **CV의 계산가능 비율** — MIN_FAIL=12 미만은 NaN이다.
        NaN 비율이 클래스마다 다르면 CV의 AUC는 편향된 부분표본 위의 값이다.
        이건 지금까지 어느 문서에도 없다.
    [3] **coverage <-> CV 중복성** (D-010 뒤집을 조건: |r|>0.9)
        k=1에서 두 지표는 **같은 1개 층**을 공유한다. coverage는 그 층이 얼마나
        불량인지, CV는 그 불량이 방위각으로 어떻게 흩어졌는지를 잰다.
        층 전체가 불량이면 coverage=1이고 CV도 자동으로 1에 가깝다 —
        구조적으로 얽혀 있다. cov<->contrast(최대 0.447)보다 훨씬 위험하다.
        **이 검사를 통과하지 못하면 "3종 세트" 주장은 2종으로 축소된다.**
    [4] 쌍별 |AUC-0.5| 3지표 동시 — 세 지표가 각각 어디서 실패하는지
    [5] 헤드라인 쌍에 개수 통제 · 크기 층화 병기

이 표의 수치는 **단일 계산이다**. D-003 분할 구현 전이므로 최종 성능이 아니라
관찰이다 (작업지침 §3-1의 3항·4항). 표본 조건(cap/seed)을 함께 기록한다.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from edge_band import coverage, band_circular_variance, edge_band
from edge_contrast import edge_contrast
from azimuth import auc

# azimuth.py가 전역 rcParams에 한글 폰트를 걸어둔다. 이 스크립트의 그림은
# ASCII 라벨만 쓰므로 되돌린다 (환경에 따라 폰트가 없으면 경고가 쏟아진다).
plt.rcParams["font.family"] = "DejaVu Sans"

K = 1                  # EDGE_LAYERS (D-007, 물리 기반 확정)
CAP = 2500             # 클래스당 표본 상한 — edge_contrast.py와 동일하게 맞춘다
SUBSAMPLE_N = 20       # 개수 통제용 (D-008)

# 헤드라인 + 세 지표가 각각 약한 자리를 포함하도록 고른 쌍
PAIRS = [
    ("Edge-Ring", "Edge-Loc"),    # 헤드라인
    ("Edge-Ring", "Random"),      # coverage의 밀도 문제
    ("Edge-Ring", "Near-full"),   # coverage 최약점 (0.081)
    ("Edge-Ring", "Donut"),       # 동어반복 대조
    ("Edge-Loc", "Random"),
    ("Edge-Loc", "none"),         # contrast 최약점 (0.048)
    ("Edge-Ring", "Scratch"),
    ("Edge-Loc", "Loc"),
]


def load(cls, cap=CAP, rng=None):
    with np.load(config.DATA_PROCESSED / f"{cls}.npz", allow_pickle=True) as z:
        maps = z["wafer_maps"]
    if len(maps) > cap:
        maps = maps[(rng or np.random.default_rng(config.SEED))
                    .choice(len(maps), cap, replace=False)]
    return maps


def feats(maps, k=K):
    """세 지표를 같은 맵 집합에 계산한다. 순서가 보존되므로 상관 계산이 가능하다."""
    return dict(
        cov=np.array([coverage(m, k) for m in maps]),
        ctr=np.array([edge_contrast(m) for m in maps]),
        cv=np.array([band_circular_variance(m, k) for m in maps]),
        cvn=np.array([band_circular_variance(
            m, k, subsample=SUBSAMPLE_N,
            rng=np.random.default_rng(config.SEED)) for m in maps]),
        size=np.array([int((np.asarray(m) != config.VAL_OUTSIDE).sum())
                       for m in maps], dtype=float),
        nband=np.array([int(edge_band(m, k).sum()) for m in maps], dtype=float),
        # 대조 특징 — coverage의 분자. 판정용이 아니라 병기용이다 (§3-2).
        nfb=np.array([float(((np.asarray(m) == config.VAL_FAIL)
                             & edge_band(m, k)).sum()) for m in maps]),
    )


def main():
    rng = np.random.default_rng(config.SEED)
    print(f"표본 조건: cap={CAP}/클래스, seed={config.SEED}, EDGE_LAYERS={K}")
    print("이 수치는 D-003 분할 이전의 **단일 계산 관찰**이다. 최종 성능이 아니다.\n")

    F = {}
    for c in config.ALL_CLASSES:
        maps = load(c, rng=rng)
        F[c] = feats(maps)
        print(f"  {c:<12} {len(maps):>6,}장")

    # ── [1] 8종 × 3지표 + 계산가능 비율 ───────────────────
    print("\n" + "=" * 86)
    print(f"[1] 8종 전체 — 세 지표를 같은 밴드(k={K})에서. CV는 처음 8종으로 확장된다")
    print("=" * 86)
    order = sorted(F, key=lambda c: -np.nanmedian(F[c]["cov"]))
    print(f"{'클래스':<13}{'coverage':>10}{'contrast':>10}{'CV':>9}"
          f"{'CV계산가능':>12}{'밴드die':>9}{'유효die':>9}")
    for c in order:
        ok = float(np.mean(np.isfinite(F[c]["cv"])))
        print(f"{c:<13}{np.nanmedian(F[c]['cov']):>10.3f}"
              f"{np.nanmedian(F[c]['ctr']):>10.2f}"
              f"{np.nanmedian(F[c]['cv']):>9.3f}{ok*100:>11.1f}%"
              f"{np.nanmedian(F[c]['nband']):>9.0f}"
              f"{np.nanmedian(F[c]['size']):>9.0f}")
    print("\n  ★ CV계산가능 비율이 클래스마다 크게 다르면, CV의 AUC는 서로 다른")
    print("    부분표본을 비교한 값이다. 낮은 클래스에서는 CV를 인용할 수 없다.")
    print(f"    (MIN_FAIL=12 — 밴드 내 불량 die가 12개 미만이면 CV는 NaN)")

    # ── [2] 중복성 ★ D-010 뒤집을 조건 ────────────────────
    print("\n" + "=" * 86)
    print("[2] 중복성 — k=1에서 coverage와 CV는 같은 층을 공유한다 (|r|>0.9면 축소)")
    print("=" * 86)
    print(f"{'클래스':<13}{'cov~CV':>10}{'cov~lnctr':>12}{'CV~lnctr':>11}{'n':>8}{'판정':>12}")
    for c in ("Edge-Ring", "Edge-Loc", "Random", "Near-full", "Donut",
              "Scratch", "Loc", "Center", config.NONE_CLASS):
        if c not in F:
            continue
        cov, cv = F[c]["cov"], F[c]["cv"]
        lct = np.log(np.clip(F[c]["ctr"], 1e-3, None))
        ok = np.isfinite(cov) & np.isfinite(cv) & np.isfinite(lct)
        if ok.sum() < 30:
            print(f"{c:<13}{'표본부족':>10}{'':>12}{'':>11}{ok.sum():>8}")
            continue
        r1 = np.corrcoef(cov[ok], cv[ok])[0, 1]
        r2 = np.corrcoef(cov[ok], lct[ok])[0, 1]
        r3 = np.corrcoef(cv[ok], lct[ok])[0, 1]
        tag = "중복 위험" if abs(r1) > 0.9 else ""
        print(f"{c:<13}{r1:>10.3f}{r2:>12.3f}{r3:>11.3f}{ok.sum():>8,}{tag:>12}")
    print("\n  cov~CV는 **CV가 계산 가능한 웨이퍼에서만** 잰 값이다.")
    print("  NaN이 많은 클래스의 r은 신뢰 구간이 넓다 — n을 함께 본다.")

    # ── [3] 쌍별 |AUC-0.5| 3지표 ──────────────────────────
    print("\n" + "=" * 86)
    print("[3] 쌍별 판별력 |AUC-0.5| — 0.5 미만은 역방향 신호이지 실패가 아니다 (§3-7)")
    print("=" * 86)
    print("  `n_fail`(밴드 불량 die 수)은 **대조 특징**이다. coverage의 분자이므로")
    print("  판정을 내리지 않고 숫자만 병기한다 — D-010 보류 항목 참조.\n")
    print(f"{'쌍':<26}{'cov':>8}{'ctr':>8}{'CV':>8}{'n_fail':>9}"
          f"{'|cov|':>8}{'|ctr|':>8}{'|CV|':>8}{'최약':>10}")
    weak = {}
    for p, n in PAIRS:
        if p not in F or n not in F:
            continue
        a_ = {k_: auc(F[p][k_], F[n][k_]) for k_ in ("cov", "ctr", "cv", "nfb")}
        d_ = {k_: abs(v - .5) for k_, v in a_.items() if k_ != "nfb"}
        lo = min(d_, key=d_.get)
        weak[(p, n)] = d_
        print(f"{p+' vs '+n:<26}{a_['cov']:>8.3f}{a_['ctr']:>8.3f}{a_['cv']:>8.3f}"
              f"{a_['nfb']:>9.3f}"
              f"{d_['cov']:>8.3f}{d_['ctr']:>8.3f}{d_['cv']:>8.3f}{lo:>10}")
    print("\n  세 지표 중 **어느 쌍에서도 셋이 동시에 약하지 않으면** 세트가 성립한다.")
    for (p, n), d_ in weak.items():
        if max(d_.values()) < 0.15:
            print(f"  ⚠ {p} vs {n}: 세 지표 전부 |AUC-0.5|<0.15 — 세트로도 못 가른다")

    # ── [4] 헤드라인 쌍의 통제 조건 ───────────────────────
    a, b = "Edge-Loc", "Edge-Ring"
    print("\n" + "=" * 86)
    print(f"[4] 헤드라인 {b} vs {a} — 통제 조건 병기 (§3-2)")
    print("=" * 86)
    print(f"{'조건':<28}{'cov':>9}{'ctr':>9}{'CV':>9}{'n_Ring':>9}{'n_Loc':>8}")
    print(f"{'통제 없음':<28}{auc(F[b]['cov'],F[a]['cov']):>9.3f}"
          f"{auc(F[b]['ctr'],F[a]['ctr']):>9.3f}"
          f"{auc(F[b]['cv'],F[a]['cv']):>9.3f}"
          f"{np.isfinite(F[b]['cov']).sum():>9,}{np.isfinite(F[a]['cov']).sum():>8,}")
    print(f"{'불량 die 개수 고정(n=20)':<28}{'-':>9}{'-':>9}"
          f"{auc(F[b]['cvn'],F[a]['cvn']):>9.3f}"
          f"{np.isfinite(F[b]['cvn']).sum():>9,}{np.isfinite(F[a]['cvn']).sum():>8,}")

    # 대조 특징: 밴드 불량 die 수 단독
    nf = {c: np.array([float(((np.asarray(m) == config.VAL_FAIL) &
                              edge_band(m, K)).sum())
                       for m in load(c, rng=np.random.default_rng(config.SEED))])
          for c in (a, b)}
    print(f"\n  대조 — 밴드 불량 die 수 **단독** AUC = {auc(nf[b], nf[a]):.3f}")
    print("  (이 값이 높으면 세 지표의 분리 중 일부는 '개수' 효과다)")

    print("\n  맵 크기 층화 (유효 die 4분위):")
    edges = np.percentile(np.concatenate([F[a]["size"], F[b]["size"]]),
                          [0, 25, 50, 75, 100])
    print(f"{'유효die 범위':>20}{'n_Ring':>8}{'n_Loc':>7}{'cov':>9}{'ctr':>9}{'CV':>9}")
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        sel = {c: (F[c]["size"] >= lo_) & (F[c]["size"] < hi_) for c in (a, b)}
        if sel[b].sum() < 30 or sel[a].sum() < 30:
            continue
        print(f"{f'{int(lo_):,}~{int(hi_):,}':>20}{sel[b].sum():>8,}{sel[a].sum():>7,}"
              f"{auc(F[b]['cov'][sel[b]],F[a]['cov'][sel[a]]):>9.3f}"
              f"{auc(F[b]['ctr'][sel[b]],F[a]['ctr'][sel[a]]):>9.3f}"
              f"{auc(F[b]['cv'][sel[b]],F[a]['cv'][sel[a]]):>9.3f}")
    print("\n  층마다 값이 평탄하면 크기 교란이 제거된 것이다.")

    # 개수 단독이 크기 층화 후에도 살아남는가 — 0.967이 크기 교란인지 확인
    print("\n  ★ 밴드 불량 die 수 **단독**을 같은 층에서 재본다:")
    print(f"{'유효die 범위':>20}{'개수단독':>10}{'cov':>9}{'차이':>9}")
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        sel = {c: (F[c]["size"] >= lo_) & (F[c]["size"] < hi_) for c in (a, b)}
        if sel[b].sum() < 30 or sel[a].sum() < 30:
            continue
        an = auc(nf[b][sel[b]], nf[a][sel[a]])
        ac = auc(F[b]["cov"][sel[b]], F[a]["cov"][sel[a]])
        print(f"{f'{int(lo_):,}~{int(hi_):,}':>20}{an:>10.3f}{ac:>9.3f}{ac-an:>9.3f}")
    print("\n  ⚠ **이 표로는 판정하지 않는다.** k=1 밴드에서 `n_band`는 같은 크기 층")
    print("    안에서 거의 상수라 coverage ≈ 개수/상수다. 단조 관계면 AUC는 같게")
    print("    나올 수밖에 없다 — **AUC는 threshold-free라서 순위만 본다.**")
    print("    개수의 진짜 약점은 '맵 크기가 바뀌면 임계값이 바뀐다'인데,")
    print("    순위 기반 지표로는 그게 보이지 않는다.")
    print("    -> 필요한 것은 **크기 층 간 전이 실험**이다: 한 층에서 학습한 결정경계를")
    print("       다른 층에 그대로 적용해 성능이 유지되는가. **D-003 분할이 전제다.**")
    print("    -> D-010 보류 항목으로 남긴다. §3-2 미충족 상태를 명시한다.")

    # ── [6] 증분 기여 ★ '3종 세트'가 성립하는지의 결정적 검사 ──
    print("\n" + "=" * 86)
    print("[6] CV가 세트의 자리를 버는가 — 증분 기여 (2-fold, 동일 표본)")
    print("=" * 86)
    from edge_contrast import two_fold_auc
    print("  ⚠ **증분은 반드시 베이스라인의 남은 여유와 함께 읽는다.**")
    print("    베이스라인이 0.99면 더할 자리가 0.01뿐이다. 그 조건에서 증분 0은")
    print("    'CV가 약하다'가 아니라 **'cov+ctr가 이미 다 해서 CV가 중복이다'**다.")
    print("    **redundant와 weak는 다른 말이고, 구분은 CV 단독 열에서 읽는다.**\n")
    print("  세 조합은 같은 행에서만 비교한다 — NaN 처리로 표본이 달라지면")
    print("  비교 자체가 성립하지 않는다.\n")
    print(f"{'쌍':<26}{'CV단독':>9}{'cov+ctr':>10}{'여유':>8}"
          f"{'+CV':>9}{'CV 증분':>10}{'여유대비':>10}{'n':>8}")
    for p, n in PAIRS:
        if p not in F or n not in F:
            continue
        ok = {c: np.isfinite(F[c]["cov"]) & np.isfinite(F[c]["ctr"])
              & np.isfinite(F[c]["cv"]) for c in (p, n)}
        if ok[p].sum() < 60 or ok[n].sum() < 60:
            continue
        cols = lambda c, ks: np.c_[tuple(F[c][k_][ok[c]] for k_ in ks)]
        a1 = auc(F[p]["cv"][ok[p]], F[n]["cv"][ok[n]])       # CV 단독
        a2 = two_fold_auc(cols(p, ("cov", "ctr")), cols(n, ("cov", "ctr")))
        a3 = two_fold_auc(cols(p, ("cov", "ctr", "cv")),
                          cols(n, ("cov", "ctr", "cv")))
        b2, b3 = max(a2, 1 - a2), max(a3, 1 - a3)
        head = 1.0 - b2                                       # 남은 여유
        inc = b3 - b2
        # 증분을 남은 여유로 나눈 값 — 천장 효과를 보정한 상대 기여
        rel = inc / head if head > 1e-6 else np.nan
        print(f"{p+' vs '+n:<26}{abs(a1-.5)+.5:>9.3f}{b2:>10.3f}{head:>8.3f}"
              f"{b3:>9.3f}{inc:>10.3f}{rel:>10.2f}{ok[p].sum()+ok[n].sum():>8,}")
    print("\n  읽는 법:")
    print("   · CV단독이 높은데 증분이 0  -> **중복(redundant)**. CV는 그 쌍을 가를 수")
    print("     있지만 cov+ctr가 이미 같은 일을 한다. '약하다'가 아니다.")
    print("   · CV단독이 낮고 증분도 0    -> 그 쌍에서 무력하다.")
    print("   · 증분 > 0                  -> 두 지표가 못 하는 것을 CV가 한다.")
    print("  '여유대비'는 증분/(1-베이스라인) — 천장 효과를 보정한 상대 기여다.")

    # ── [5] sanity check ──────────────────────────────────
    print("\n" + "=" * 86)
    print("[5] sanity check — none의 CV는 1에 가까워야 한다")
    print("=" * 86)
    nn = config.NONE_CLASS
    print(f"  none CV 중앙값 (k={K}) = {np.nanmedian(F[nn]['cv']):.3f}")
    print(f"  none CV 계산가능      = {np.mean(np.isfinite(F[nn]['cv']))*100:.1f}%")
    print("  §3-5: 미통과라도 '절대값 해석 불가 / 상대 비교는 조건부 유효'다.")
    print("  확인할 것은 'none이 1인가'가 아니라 **편향이 클래스별로 다른가**다.")

    # ── [7] 관찰: 밴드가 지표마다 달라야 할 수 있다 ────────
    print("\n" + "=" * 86)
    print("[7] 관찰 — coverage와 CV의 최적 밴드가 다르다 (판정 아님)")
    print("=" * 86)
    print("  edge_band.py [2]의 k 스윕 (Ring↔Loc, cap=3000):")
    print(f"    {'k':>3}{'AUC cov':>10}{'AUC CV':>9}{'CV(n고정)':>12}{'none CV':>10}")
    for k_, c_, v_, vn_, nc_ in ((1, .971, .912, .730, .828), (2, .893, .921, .778, .849),
                                 (3, .801, .926, .781, .863), (4, .732, .925, .790, .869),
                                 (5, .684, .920, .772, .876), (6, .647, .916, .770, .879)):
        print(f"    {k_:>3}{c_:>10.3f}{v_:>9.3f}{vn_:>12.3f}{nc_:>10.3f}")
    print("\n  **coverage는 k=1에서 최고, CV는 k=3~4에서 최고다.** 그리고 세 지표가")
    print("  같은 방향을 가리킨다:")
    print("   · CV(n고정)도 k=4에서 최고 (0.790)")
    print("   · none sanity check도 k와 함께 단조 개선 (0.828 -> 0.879)")
    print(f"   · 위 [4]에서 CV는 최소 크기 층에서 붕괴 (k=1, AUC 0.551)")
    print("\n  물리적으로도 어긋나지 않는다 — coverage는 **링 두께**(1층)를 재고,")
    print("  CV는 **배치**를 재는데 배치는 표본 수가 필요하다. k=1은 밴드 die가")
    print("  100~200개뿐이라 방위각 분포를 논하기엔 얇다.")
    print("\n  -> **검토 결과: 기각. 세 지표 모두 k=1로 통일한다 (D-013).**")
    print("     위 네 관찰 중 성능이 아닌 것은 none sanity 하나뿐이고, 나머지 AUC는")
    print("     전부 Ring<->Loc에서 잰 값이다 — [6]에서 CV 증분이 0.000인,")
    print("     **CV가 쓰이지 않는 쌍**이다.")
    print("     그리고 3~4층에서 Edge-Ring은 배경 수준(0.070)인데 Edge-Loc은 배경의")
    print("     2배(0.169)다. 밴드를 넓히면 '방위각을 잘 쟀다'와 'Edge-Loc의 전면")
    print("     오염을 쟀다'가 구분되지 않는다 — §3-2 미충족.")
    print("     none sanity 개선은 '표본이 늘면 편향이 준다'는 뜻이므로 밴드가 아니라")
    print("     **D-008 개수 통제**로 다룬다.")
    print("     얻는 것 +0.014의 대가로 체리피킹 공격면을 연다. 감수한 비용은")
    print("     CV 소형 맵 붕괴(0.551)와 none sanity 0.828 — 알면서 남긴 것이다.")

    # ── 그림 ──────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    names = [c for c in order]
    for i, (key, lab, logy) in enumerate(
            (("cov", "coverage", False), ("ctr", "edge_contrast", True),
             ("cv", "circular variance", False))):
        d = [F[c][key][np.isfinite(F[c][key])] for c in names]
        ax[i].boxplot(d, tick_labels=names, showfliers=False)
        if logy:
            ax[i].set_yscale("log")
        ax[i].set_title(f"{lab}  (k={K})")
        ax[i].tick_params(axis="x", rotation=60, labelsize=8)
    fig.tight_layout()
    out = config.FIGURES / "feature_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
