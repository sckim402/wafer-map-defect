# -*- coding: utf-8 -*-
"""11월 목록 #8 — 「`R`의 승자에는 항상 `Center`가 있다」 확증 (D-029 §5 → D-046).

⛔ **예측과 반증 조건은 `docs/w9_center_prereg.md`에 있고, 이 스크립트보다 먼저 커밋됐다.**
   여기서 기준을 새로 만들지 않는다.

`order_boot.py`의 기계를 그대로 쓴다(`FAM`·`rates`·`winners`·`lot_conf`·`oof`).
다른 것은 **집계**뿐이다 — 승자 쌍의 인덱스가 아니라 **`Center` 포함 여부**를 센다.

돌리기:
    ./.venv/Scripts/python.exe   -u src/w9_center_boot.py [B]
    ./.venv19/Scripts/python.exe -u src/w9_center_boot.py [B]
"""
import itertools
import sys

import numpy as np

sys.path.insert(0, "src")
import config
from order_boot import FAM, L, PAIRS, PNAME, lot_conf, rates, winners, oof

B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEED = 20260916          # 원래 관찰(20260912)과 다른 draw — 사전 등록 §3
CENTER = L.index("Center")
HAS_CENTER = np.array([CENTER in (i, j) for i, j in PAIRS])


def demo():
    """§3-3 — 집계가 실제로 `Center`를 세는가, 그리고 승자 없음을 안 세는가."""
    assert HAS_CENTER.sum() == len(L) - 1, f"Center가 걸린 쌍이 7개가 아니다: {HAS_CENTER.sum()}"
    base = HAS_CENTER.mean()
    assert abs(base - 7 / 28) < 1e-12, "우연 기준선이 25%가 아니다"

    k = int(np.flatnonzero(HAS_CENTER)[0])       # Center가 걸린 쌍
    m = int(np.flatnonzero(~HAS_CENTER)[0])      # 안 걸린 쌍
    dec = np.zeros((100, len(PAIRS))); dec[:, k] = 0.1
    w = winners(dec)
    assert tally(w) == (100, 100), f"Center 승자를 못 센다: {tally(w)}"
    dec = np.zeros((100, len(PAIRS))); dec[:, m] = 0.1
    assert tally(winners(dec)) == (100, 0), "Center 아닌 승자를 Center로 센다"
    # 승자 없음은 분모에서 빠져야 한다
    assert tally(winners(np.zeros((100, len(PAIRS))))) == (0, 0), "승자 없음을 분모에 넣는다"
    print(f"demo ok — 우연 기준선 {base:.1%} · 집계가 승자 없음을 분모에서 뺀다")


def tally(top):
    """(확정 draw 수, 그중 Center 포함 수)."""
    ok = top >= 0
    return int(ok.sum()), int(HAS_CENTER[top[ok]].sum()) if ok.any() else 0


def main():
    import sklearn
    print("=" * 86)
    print(f"  환경 sklearn {sklearn.__version__} · numpy {np.__version__} "
          f"· B={B:,} · 재추출 SEED={SEED}")
    print(f"  우연 기준선: Center가 걸린 쌍은 28쌍 중 {HAS_CENTER.sum()}개 = {HAS_CENTER.mean():.1%}")
    print("=" * 86)

    rng = np.random.default_rng(SEED)
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        y, lot, folds, seeds = (z["cls"].astype(str), z["lot"].astype(str),
                                z["folds"], z["seeds"])
    F = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        for k in ("cov", "ctr", "cv"):
            F[k] = z[k]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        F["rc"], F["mp"] = z["radial_contrast"], z["mid_peak"]
    with np.load(config.DATA_PROCESSED / "shape2.npz", allow_pickle=True) as z:
        F["cc"] = z["cc_compact"]

    idx = {c: k for k, c in enumerate(L)}
    y_i = np.array([idx[c] for c in y])
    lots, lot_id = np.unique(lot, return_inverse=True)
    n_lots = len(lots)
    W = rng.multinomial(n_lots, np.full(n_lots, 1.0 / n_lots), size=B).astype(np.float64)
    print(f"  웨이퍼 {len(y):,}장 · lot {n_lots:,}개 · seed {len(seeds)}개\n")

    LM = {}
    for r in (1, 2, 3):
        for c in itertools.combinations("ERS", r):
            S_ = frozenset(c)
            X = np.column_stack([F[k] for f in "ERS" if f in S_ for k in FAM[f]])
            LM[S_] = [lot_conf(y_i, np.array([idx[q] for q in oof(X, y, folds[s])]),
                               lot_id, n_lots) for s in range(len(seeds))]
            print(f"  적합 완료 {'+'.join(sorted(S_)):<6}", flush=True)

    print("\n" + "=" * 86)
    print(f"{'계열':<5}{'선행':<8}{'seed':<6}{'승자(최빈)':<26}{'확정draw':>10}{'Center 포함':>13}")
    print("-" * 86)
    summary = {}
    for X in "ERS":
        others = [f for f in "ERS" if f != X]
        Ps = [(others[0],), (others[1],), tuple(others)]
        tot, cen = 0, 0
        for P in Ps:
            a, b = LM[frozenset(P) | {X}], LM[frozenset(P)]
            for s in range(len(seeds)):
                dec = rates(W @ b[s]) - rates(W @ a[s])
                top = winners(dec)
                n_ok, n_cen = tally(top)
                tot += n_ok; cen += n_cen
                cnt = np.bincount(top[top >= 0], minlength=len(PAIRS))
                best = PNAME[int(cnt.argmax())] if n_ok else "—"
                print(f"{X:<5}{'+'.join(P):<8}{seeds[s]:<6}{best:<26}"
                      f"{n_ok:>10,}{n_cen/n_ok if n_ok else np.nan:>12.2%}")
        summary[X] = (tot, cen)
        print(f"{'':<5}{'(계)':<8}{'':<6}{'':<26}{tot:>10,}{cen/tot if tot else np.nan:>12.2%}")
        print("-" * 86)

    print("\n=== 사전 등록 대조 (docs/w9_center_prereg.md §2-B) ===")
    for X, pred in (("R", "100%"), ("S", "0%"), ("E", "혼합")):
        tot, cen = summary[X]
        print(f"  {X}: 예측 {pred:<6} → 실측 {cen/tot if tot else np.nan:.2%}  ({cen:,}/{tot:,})")


if __name__ == "__main__":
    demo()
    main()
