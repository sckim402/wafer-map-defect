"""W5 — 특징 투입 순서 의존성 + leave-one-out (§3-14 첫 적용).

실행:
    python -u src/order_dep.py

**판정 기준은 docs/w5_order.md §1 에 실행 전 등록돼 있다** (2026-09-09, 커밋 ec1afa0).
요약: *"각 계열이 「가장 크게 푸는 쌍」이 투입 위치와 무관하게 동일한가."*
3계열 전부 동일하면 ⓑ(데이터 성질), 하나라도 갈리면 ⓐ(경로 성질).
동률 여유 0.005 미만이면 **보수적으로 갈린 것으로 본다.**
"""
import itertools

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

import config

FAM = {"E": ("cov", "ctr", "cv"), "R": ("rc", "mp"), "S": ("cc",)}
TIE = 0.005          # 동률 여유 — 이보다 작으면 「구분 불가」 → 갈린 것으로 간주
LOO_CUT = 0.01       # leave-one-out 하락이 이 미만이면 「대체 가능」으로 기록


def rf(seed=0):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                  random_state=seed, class_weight="balanced")


def oof(X, y, folds):
    p = np.empty(len(y), dtype=object)
    for f in np.unique(folds):
        te = folds == f
        p[te] = rf().fit(X[~te], y[~te]).predict(X[te])
    return p.astype(str)


def pair_rates(y, p, L):
    """쌍별 상호 오분류율 — 초록과 같은 지표."""
    M = confusion_matrix(y, p, labels=L)
    return {(L[i], L[j]): (M[i, j] + M[j, i]) / (M[i].sum() + M[j].sum())
            for i in range(len(L)) for j in range(i + 1, len(L))}


def load():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        y, orig, F, seeds = (z["cls"].astype(str), z["split_orig"].astype(str),
                             z["folds"], z["seeds"])
    d = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        for k in ("cov", "ctr", "cv"):
            d[k] = z[k]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        d["rc"], d["mp"] = z["radial_contrast"], z["mid_peak"]
    with np.load(config.DATA_PROCESSED / "shape2.npz", allow_pickle=True) as z:
        d["cc"] = z["cc_compact"]
    return y, orig, F, len(seeds), d


def evaluate(keys, y, F, n_seed, d, L):
    """주 분할 out-of-fold — macro-F1 과 쌍별 오분류율 (seed 평균)."""
    X = np.column_stack([d[k] for k in keys])
    ps = [oof(X, y, F[s]) for s in range(n_seed)]
    macro = float(np.mean([f1_score(y, p, labels=L, average="macro",
                                    zero_division=0) for p in ps]))
    rs = [pair_rates(y, p, L) for p in ps]
    pr = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    return macro, pr


def main():
    y, orig, F, n_seed, d = load()
    L = config.PATTERN_CLASSES
    fams = list(FAM)

    # ── 계열 부분집합 7개 (∅ 제외) ────────────────────────────
    subsets = [c for n in (1, 2, 3) for c in itertools.combinations(fams, n)]
    res = {}
    print("[*] 계열 부분집합 평가", flush=True)
    for sub in subsets:
        keys = tuple(k for f in sub for k in FAM[f])
        res[frozenset(sub)] = evaluate(keys, y, F, n_seed, d, L)
        print(f"    {'+'.join(sub):<8} macro {res[frozenset(sub)][0]:.3f}", flush=True)

    def top(pr, n=1):
        o = sorted(pr.items(), key=lambda kv: -kv[1])
        return o[:n]

    print("\n" + "=" * 78)
    print("[나] 6경로 × 단계별 최대 병목 — 전량 보고 (§3-9)")
    print("=" * 78)
    print(f"  {'경로':<12}{'1단계':<30}{'2단계':<30}{'3단계'}")
    for order in itertools.permutations(fams):
        row = [f"{'→'.join(order):<12}"]
        for n in (1, 2, 3):
            m, pr = res[frozenset(order[:n])]
            (a, b), v = top(pr)[0]
            row.append(f"{a}↔{b} {v:.3f} (M{m:.3f})".ljust(30))
        print("  " + "".join(row))

    print("\n" + "=" * 78)
    print("[가] ★ 각 계열이 「가장 크게 푸는 쌍」 — 선행 집합 4가지에서 동일한가")
    print("=" * 78)
    verdict = {}
    for X in fams:
        others = [f for f in fams if f != X]
        # ∅ 는 제외한다 — "푸는 쌍"은 이전 상태가 있어야 정의된다 (w5_order.md §1-B 정정 ⓑ)
        Ps = [(others[0],), (others[1],), tuple(others)]
        print(f"\n  [{X}] {'+'.join(FAM[X])}")
        print(f"    {'선행':<10}{'가장 크게 푸는 쌍':<28}{'감소':>9}{'2위와 차':>10}")
        winners = []
        for P in Ps:
            after = frozenset(P) | {X}
            pr_a, pr_b = res[after][1], res[frozenset(P)][1]
            dec = {k: pr_b[k] - pr_a[k] for k in pr_a}
            o = sorted(dec.items(), key=lambda kv: -kv[1])
            (a, b), v1 = o[0]
            gap = v1 - o[1][1]
            winners.append(((a, b), gap))
            lab = "+".join(P)
            print(f"    {lab:<10}{a+'↔'+b:<28}{v1:>+9.3f}{gap:>10.3f}")
        pairs = {w[0] for w in winners}
        tie = [w for w in winners if w[1] < TIE]
        ok = len(pairs) == 1 and not tie
        verdict[X] = ok
        why = ("일치" if len(pairs) == 1 else f"불일치 {len(pairs)}종") + \
              (f" · 동률 {len(tie)}건(<{TIE})" if tie else "")
        print(f"    → {'✅ 안정' if ok else '❌ 갈림'} ({why})")

    print("\n" + "=" * 78)
    print("[다] leave-one-out — 6종에서 하나씩 뺀다")
    print("=" * 78)
    full = tuple(k for f in fams for k in FAM[f])
    m_full, _ = res[frozenset(fams)]
    print(f"    {'제거':<10}{'macro':>9}{'하락':>9}   판정")
    for k in full:
        keys = tuple(x for x in full if x != k)
        m, _ = evaluate(keys, y, F, n_seed, d, L)
        drop = m_full - m
        tag = "대체 가능 (제외 아님)" if drop < LOO_CUT else "대체 불가"
        print(f"    {k:<10}{m:>9.3f}{drop:>+9.3f}   {tag}")

    print("\n" + "=" * 78)
    print("[외부] split_orig 병기 (D-016) — 6경로 1·2단계 macro")
    print("=" * 78)
    tr, te = orig == "Training", orig == "Test"
    for sub in subsets:
        keys = tuple(k for f in sub for k in FAM[f])
        X = np.column_stack([d[k] for k in keys])
        pe = rf().fit(X[tr], y[tr]).predict(X[te])
        pr = pair_rates(y[te], pe, L)
        (a, b), v = sorted(pr.items(), key=lambda kv: -kv[1])[0]
        m = f1_score(y[te], pe, labels=L, average="macro", zero_division=0)
        print(f"    {'+'.join(sub):<8} macro {m:.3f}   최대 병목 {a}↔{b} {v:.3f}")

    print("\n" + "=" * 78)
    print("★ 사전 등록 판정 (docs/w5_order.md §1-C)")
    print("=" * 78)
    for X in fams:
        print(f"    {X}: {'✅ 안정' if verdict[X] else '❌ 갈림'}")
    if all(verdict.values()):
        print("\n  → ⓑ 성립. 계열↔병목 대응이 순서와 무관하다. **초록 무수정**")
    else:
        print("\n  → ⓐ. 대응이 경로에 의존한다.")
        print("     **초록에 「우리가 밟은 진단 경로에서」를 삽입한다**")


if __name__ == "__main__":
    main()
