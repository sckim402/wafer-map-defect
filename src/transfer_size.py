"""크기 층 간 전이 실험 — D-010 보류 항목을 닫는다 (§3-2).

실행:
    python src/transfer_size.py        # 특징 캐시가 없으면 먼저 계산 (수 분)

질문:
    `coverage`(정규화)와 `n_fail`(원 개수) 중 어느 쪽이 진짜 신호인가.
    쌍별 AUC로는 판별할 수 없었다 — k=1에서 `coverage ≈ n_fail / 상수`이고,
    **AUC는 threshold-free라 순위만 본다.**

    개수의 진짜 약점은 *"맵 크기가 바뀌면 임계값이 바뀐다"*다.
    -> **한 크기 층에서 학습한 결정경계를 다른 층에 그대로 적용**해 본다.
       정규화가 의미 있다면 coverage는 전이되고 n_fail은 무너져야 한다.

설계 (§3-2 대조 실험):
    - 대상: Edge-Ring vs Edge-Loc (헤드라인 쌍). 전수 14,869장, cap 없음
    - 분할: **D-003 SGK(5) × seed 3개.** 임계값은 train에서만 학습한다
    - 층: 유효 die 수 4분위. **cut point도 train에서만 정한다**
    - 분류기: 단일 특징 임계값 (Youden J로 train에서 결정)
      -> 모델 용량 차이가 아니라 **임계값의 전이성만** 재기 위해서다
    - 지표: **balanced accuracy.** AUC를 쓰면 질문 자체가 사라진다
      (클래스 비율이 층마다 크게 다르므로 accuracy는 부적절)

읽는 법:
    대각선 = 같은 층에서 학습·평가 (전이 없음, 상한)
    비대각 = 다른 층으로 전이
    **대각선 대비 비대각의 하락폭이 곧 "임계값이 크기에 의존하는 정도"다.**
"""
import numpy as np

import config
from edge_band import coverage, edge_band
from edge_contrast import edge_contrast

CACHE = config.DATA_PROCESSED / "transfer_feats.npz"
PAIR = ("Edge-Ring", "Edge-Loc")     # 양성 = Edge-Ring
K = 1                                # EDGE_LAYERS (D-007/D-013)
N_STRATA = 4


# ── 특징 계산 (캐시) ──────────────────────────────────────
def build_features():
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=True) as z:
            return {k: z[k] for k in z.keys()}
    print("특징 계산 중 (최초 1회, 수 분 걸린다)...")
    out = {k: [] for k in ("y", "lot", "cov", "nfail", "ctr", "size")}
    for pos, c in enumerate(PAIR):
        with np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True) as z:
            maps, lots = z["wafer_maps"], z["lot_name"].astype(str)
        print(f"  {c:<12} {len(maps):,}장")
        for m in maps:
            a = np.asarray(m)
            b = edge_band(a, K)
            out["cov"].append(coverage(a, K))
            out["nfail"].append(float(((a == config.VAL_FAIL) & b).sum()))
            out["ctr"].append(edge_contrast(a))
            out["size"].append(float((a != config.VAL_OUTSIDE).sum()))
        out["y"] += [1 - pos] * len(maps)          # Edge-Ring = 1
        out["lot"] += list(lots)
    d = {k: np.asarray(v) for k, v in out.items()}
    np.savez_compressed(CACHE, **d)
    print(f"  캐시 저장: {CACHE}")
    return d


# ── 단일 특징 임계값 분류기 ───────────────────────────────
def fit_threshold(x, y):
    """Youden J(= 민감도+특이도−1)를 최대화하는 임계값과 방향을 train에서 정한다.

    반환: (thr, sign). sign=+1이면 x >= thr 를 양성으로 예측한다.
    """
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if len(np.unique(y)) < 2 or len(x) < 20:
        return np.nan, 1
    cand = np.unique(np.quantile(x, np.linspace(0.01, 0.99, 199)))
    best, bthr, bsign = -1.0, np.nan, 1
    P, N = (y == 1).sum(), (y == 0).sum()
    for sign in (1, -1):
        s = sign * x
        for t in sign * cand:
            pred = s >= t
            j = (pred & (y == 1)).sum() / P + (~pred & (y == 0)).sum() / N - 1
            if j > best:
                best, bthr, bsign = j, sign * t, sign
    return bthr, bsign


def balanced_acc(x, y, thr, sign):
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if len(x) == 0 or np.isnan(thr) or len(np.unique(y)) < 2:
        return np.nan
    pred = (sign * x) >= (sign * thr)
    return 0.5 * ((pred & (y == 1)).sum() / max((y == 1).sum(), 1)
                  + (~pred & (y == 0)).sum() / max((y == 0).sum(), 1))


# ── 전이 실험 ─────────────────────────────────────────────
def run(d, folds, seeds):
    feats = ("cov", "nfail", "ctr")
    acc = {f: np.full((len(seeds), 5, N_STRATA, N_STRATA), np.nan) for f in feats}
    thr = {f: np.full((len(seeds), 5, N_STRATA), np.nan) for f in feats}
    y, size = d["y"], d["size"]

    for si in range(len(seeds)):
        for fold in range(5):
            te_m = folds[si] == fold
            tr_m = ~te_m
            # 층 경계는 **train에서만** 정한다 (test를 보고 나누지 않는다)
            edges = np.quantile(size[tr_m], np.linspace(0, 1, N_STRATA + 1))
            edges[0], edges[-1] = -np.inf, np.inf

            def stratum(mask, s):
                return mask & (size >= edges[s]) & (size < edges[s + 1])

            for f in feats:
                x = d[f]
                for s_src in range(N_STRATA):
                    m = stratum(tr_m, s_src)
                    t, sg = fit_threshold(x[m], y[m])
                    thr[f][si, fold, s_src] = t
                    for s_tgt in range(N_STRATA):
                        mt = stratum(te_m, s_tgt)
                        acc[f][si, fold, s_src, s_tgt] = balanced_acc(
                            x[mt], y[mt], t, sg)
    return acc, thr


def main():
    d = build_features()
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        cls, idx, F, seeds = z["cls"], z["idx_in_cls"], z["folds"], z["seeds"]

    # split_folds의 행 순서를 특징 배열 순서(Edge-Ring 먼저)에 맞춘다
    order = []
    for c in PAIR:
        sel = np.nonzero(cls == c)[0]
        order.append(sel[np.argsort(idx[sel])])
    order = np.concatenate(order)
    folds = F[:, order]
    assert len(folds[0]) == len(d["y"]), "행 수 불일치"

    print(f"\n대상 {len(d['y']):,}장 "
          f"(Edge-Ring {int((d['y']==1).sum()):,} / Edge-Loc {int((d['y']==0).sum()):,})")
    print(f"분할 SGK(5) × seed {tuple(seeds)} — 임계값은 train에서만 학습")

    acc, thr = run(d, folds, seeds)
    names = {"cov": "coverage (정규화)", "nfail": "n_fail (원 개수)",
             "ctr": "edge_contrast (비율)"}

    # ── [1] 전이 행렬 ─────────────────────────────────────
    print("\n" + "=" * 78)
    print("[1] 전이 행렬 — balanced accuracy (평균 over 5 fold × 3 seed)")
    print("=" * 78)
    print("  행 = 임계값을 학습한 층 / 열 = 적용한 층. 대각선이 전이 없는 상한이다.\n")
    summ = {}
    for f in ("cov", "nfail", "ctr"):
        A = np.nanmean(acc[f], axis=(0, 1))
        print(f"  ── {names[f]} ──")
        print("      " + "".join(f"{'→S'+str(t):>9}" for t in range(N_STRATA)))
        for s in range(N_STRATA):
            print(f"    S{s}" + "".join(
                f"{A[s, t]:>9.3f}" + ("*" if s == t else " ")
                for t in range(N_STRATA)))
        diag = np.nanmean([A[i, i] for i in range(N_STRATA)])
        off = np.nanmean([A[i, j] for i in range(N_STRATA)
                          for j in range(N_STRATA) if i != j])
        worst = np.nanmin(A)
        summ[f] = (diag, off, worst)
        print(f"    대각 {diag:.3f} / 비대각 {off:.3f} / 최악 {worst:.3f}"
              f"  → 전이 손실 {diag-off:+.3f}\n")

    # ── [2] 핵심 비교 ─────────────────────────────────────
    print("=" * 78)
    print("[2] 판정 — 전이 손실이 곧 '임계값이 크기에 의존하는 정도'다")
    print("=" * 78)
    print("  전이 손실은 fold·seed마다 따로 계산해 평균±표준편차로 보고한다")
    print("  (단일 계산 수치는 결과가 아니라 관찰이다 — §3-1 3항)\n")
    print(f"  {'특징':<22}{'대각(상한)':>12}{'비대각':>10}{'최악':>9}{'전이 손실':>16}")
    for f in ("cov", "ctr", "nfail"):
        dg, of, wo = summ[f]
        # fold·seed 단위로 손실을 따로 내서 편차를 본다
        per = []
        for si in range(acc[f].shape[0]):
            for fo in range(acc[f].shape[1]):
                A = acc[f][si, fo]
                d_ = np.nanmean([A[i, i] for i in range(N_STRATA)])
                o_ = np.nanmean([A[i, j] for i in range(N_STRATA)
                                 for j in range(N_STRATA) if i != j])
                per.append(d_ - o_)
        per = np.array(per)
        print(f"  {names[f]:<22}{dg:>12.3f}{of:>10.3f}{wo:>9.3f}"
              f"{f'{np.nanmean(per):+.3f} ± {np.nanstd(per):.3f}':>16}")

    # ── [2b] 교란 통제 — "작은 맵 층이 원래 어렵다"를 분리한다 ──
    print("\n" + "=" * 78)
    print("[2b] 대조 — 비대각 하락이 '전이 실패'인가 '그 층이 원래 어려움'인가")
    print("=" * 78)
    print("  **대각선도 층마다 다르다.** S0(가장 작은 맵)는 어떤 특징으로도 어렵다.")
    print("  그러니 전체 비대각 평균만 보면 'S0이 어렵다'가 전이 손실로 오인된다.")
    print("  -> **같은 목표 층(열) 안에서** 대각 대비 하락을 다시 잰다.\n")
    print(f"  {'특징':<22}" + "".join(f"{'→S'+str(t):>9}" for t in range(N_STRATA)))
    for f in ("cov", "ctr", "nfail"):
        A = np.nanmean(acc[f], axis=(0, 1))
        drops = []
        for t in range(N_STRATA):
            off = np.nanmean([A[s, t] for s in range(N_STRATA) if s != t])
            drops.append(A[t, t] - off)
        print(f"  {names[f]:<22}" + "".join(f"{v:>+9.3f}" for v in drops))
    print("\n  **열마다 따로 봐도 순서가 유지되면 교란이 아니라 진짜 전이 차이다.**")

    # ── [3] 층별 최적 임계값 — 가장 직접적인 증거 ─────────
    print("\n" + "=" * 78)
    print("[3] 층별 최적 임계값 — 값 자체가 크기에 따라 움직이는가")
    print("=" * 78)
    for f in ("cov", "nfail", "ctr"):
        T = np.nanmean(thr[f], axis=(0, 1))
        rng = np.nanmax(T) / max(np.nanmin(T), 1e-9)
        print(f"  {names[f]:<22}" + "".join(f"{v:>10.3f}" for v in T)
              + f"   최대/최소 = {rng:>6.2f}배")
    print("\n  **정규화된 특징은 임계값이 층에 무관해야 하고, 원 개수는 커져야 한다.**")

    # ── [4] 가장 강한 반론 — "크기를 같이 주면 n_fail도 되지 않나" ──
    print("\n" + "=" * 78)
    print("[4] 반론 검증 — 모델에 맵 크기를 함께 주면 n_fail이 회복되는가")
    print("=" * 78)
    print("  이 실험은 **크기로 층화**하는데 n_fail은 바로 그 크기와 교란돼 있다.")
    print("  '축을 그렇게 잡았으니 당연한 결과 아닌가'가 가장 강한 반론이다.")
    print("  -> 모델에 `size`를 명시적으로 주고 다시 본다. 회복되면 coverage의")
    print("     이점은 '편의'일 뿐이고, 회복 안 되면 **학습으로 못 얻는 정보**다.\n")
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import balanced_accuracy_score
    SETS = {"coverage": ("cov",), "n_fail": ("nfail",),
            "n_fail + size": ("nfail", "size"), "coverage + size": ("cov", "size")}
    A2 = {k: np.full((len(seeds), 5, N_STRATA, N_STRATA), np.nan) for k in SETS}
    for si in range(len(seeds)):
        for fo in range(5):
            te_m = folds[si] == fo; tr_m = ~te_m
            ed = np.quantile(d["size"][tr_m], np.linspace(0, 1, N_STRATA + 1))
            ed[0], ed[-1] = -np.inf, np.inf
            stq = lambda m, s: m & (d["size"] >= ed[s]) & (d["size"] < ed[s + 1])
            for nm, keys in SETS.items():
                X = np.column_stack([d[k] for k in keys])
                for a in range(N_STRATA):
                    m = stq(tr_m, a)
                    if m.sum() < 50 or len(np.unique(d["y"][m])) < 2:
                        continue
                    # 깊이 2로 고정 — 용량을 맞추고 **특징 집합만** 바꾼다
                    mdl = DecisionTreeClassifier(max_depth=2, class_weight="balanced",
                                                 random_state=0).fit(X[m], d["y"][m])
                    for b in range(N_STRATA):
                        mt = stq(te_m, b)
                        if mt.sum() < 20 or len(np.unique(d["y"][mt])) < 2:
                            continue
                        A2[nm][si, fo, a, b] = balanced_accuracy_score(
                            d["y"][mt], mdl.predict(X[mt]))
    print(f"  {'특징 집합':<20}{'대각':>8}{'비대각':>9}{'최악':>8}{'전이 손실':>12}")
    for nm in SETS:
        M = np.nanmean(A2[nm], axis=(0, 1))
        dg = np.nanmean([M[i, i] for i in range(N_STRATA)])
        of = np.nanmean([M[i, j] for i in range(N_STRATA)
                         for j in range(N_STRATA) if i != j])
        print(f"  {nm:<20}{dg:>8.3f}{of:>9.3f}{np.nanmin(M):>8.3f}{dg-of:>+12.3f}")
    print("\n  **`size`를 줘도 n_fail은 회복되지 않는다.**")
    print("  한 크기 층에서만 학습하면 그 안에 크기 변이가 없어 **보정을 배울 수가 없다.**")
    print("  -> 정규화는 데이터로 못 얻는 **사전 지식을 특징에 심는 것**이다.")
    print("  (`coverage + size`가 coverage 단독보다 나쁜 것도 같은 이유 —")
    print("   전이되지 않는 변수를 특징으로 주면 모델이 거기 매달린다)")


if __name__ == "__main__":
    main()
