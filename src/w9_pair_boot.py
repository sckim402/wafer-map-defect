# -*- coding: utf-8 -*-
"""11월 목록 #6 — D-037의 `a`/`b`/`c`에 **lot 블록 부트스트랩**을 건다 (D-046).

**왜 필요한가.** D-037은 **점추정뿐**이다. 초록은 그 위에 인과 문장을 싣고 있고
(*"앞의 두 병목은 특징 부재가 원인이었고 설계하자 완화됐다"*),
**「86%면 지배적」이라는 읽기는 내 판단**이다 — `CLAUDE.md` §7에 그렇게 적혀 있다.
**D-026이 미리 정한 것은 방법이지 합격선이 아니다.**

**⛔ 그래서 이 스크립트는 새 합격선을 만들지 않는다** (§3-13이 막는 것이 그것이다).
**사전 등록하는 것은 「순서 관계」 하나뿐이고, 임계값이 아니다**:

    ① `a > 0`                     — 해소가 실제로 일어났는가
    ② `a > |b|`                   — 해소가 이동보다 큰가  ← **이 순서가 재추출에서 유지되는가**
    ③ `c > 0`                     — 두 클래스의 정답이 실제로 늘었는가
    셋의 **재추출 유지율**과 `share = a/(a+|b|)`의 **백분위 구간**을 보고한다.
    **「몇 % 이상이면 지배적」은 정하지 않는다** — 구간을 주는 것이 답이지 새 임계값이 아니다.

**블록이 lot인 이유** (D-029와 같다): 같은 lot은 공정 조건을 공유한다(D-003).
OOF 예측을 고정한 채 **lot만 다시 뽑는다.**

돌리기 (두 환경 모두 — D-047 이후로는 이게 기본이다):
    ./.venv/Scripts/python.exe   -u src/w9_pair_boot.py
    ./.venv19/Scripts/python.exe -u src/w9_pair_boot.py
"""
import sys

import numpy as np

sys.path.insert(0, "src")
import config
from abstract_numbers import STAGES, load, mutual
from pair_resolve import oof, decompose

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEED = 20260916
TRANS = [(0, 1, "Center", "Loc", "가장자리 3종 → +반경 2종"),
         (1, 2, "Loc", "Scratch", "+반경 2종 → +형상 1종")]


def fold_lots(y, p_old, p_new, i, j, lot_id, n_lots):
    """lot마다 **전이 3×3 표**를 접어 둔다 → 재추출이 행렬곱 한 번이 된다.

    상태 0=정답 · 1=쌍 안 오류 · 2=제3자 오류. `decompose`와 같은 정의다.
    """
    m = (y == i) | (y == j)
    yy, lo = y[m], lot_id[m]
    other = np.where(yy == i, j, i)

    def st(pred):
        q = pred[m]
        return np.where(q == yy, 0, np.where(q == other, 1, 2))

    s0, s1 = st(p_old), st(p_new)
    M = np.zeros((n_lots, 9))
    np.add.at(M, (lo, s0 * 3 + s1), 1.0)
    return M


def abc(cnt):
    """접힌 9칸에서 a·b·c를 낸다. cnt[x*3+z] = n(x→z). 전부 선형이다."""
    n = lambda x, z: cnt[..., x * 3 + z]
    a = n(1, 0) - n(0, 1)
    b = n(1, 2) - n(2, 1)
    c = (n(0, 0) + n(1, 0) + n(2, 0)) - (n(0, 0) + n(0, 1) + n(0, 2))
    return a, b, c


def share_of(a, b):
    d = a + np.abs(b)
    return np.where(d != 0, a / np.where(d != 0, d, 1), np.nan)


def demo():
    """§3-3 — 실패 조건에서 먼저 돌린다. **D-026의 반례를 부트스트랩도 잡는가.**"""
    y = np.array(["A"] * 6 + ["B"] * 6)
    lot_id = np.arange(12) // 3          # lot 4개
    old = np.array(["B"] * 6 + ["A"] * 6)
    fix = np.array(["A"] * 6 + ["B"] * 6)
    mov = np.array(["C"] * 12)

    M = fold_lots(y, old, fix, "A", "B", lot_id, 4)
    a, b, c = abc(M.sum(0))
    assert (a, b, c) == (12, 0, 12), f"해소를 못 잡는다: {(a, b, c)}"
    assert np.allclose(M.sum(), 12), "접기에서 표본이 새어 나갔다"

    M = fold_lots(y, old, mov, "A", "B", lot_id, 4)
    a, b, c = abc(M.sum(0))
    assert (a, b, c) == (0, 12, 0), f"오류 이동을 해소로 센다: {(a, b, c)}"

    # 접기 결과가 원본 decompose와 같은가 — **다른 경로로 같은 값을 내는지 본다**
    mix = np.array(["A"] * 3 + ["C"] * 3 + ["B"] * 3 + ["C"] * 3)
    r = decompose(y, old, mix, "A", "B")
    a, b, c = abc(fold_lots(y, old, mix, "A", "B", lot_id, 4).sum(0))
    assert (a, b, c) == (r["a"], r["b"], r["c"]), f"접기가 decompose와 다르다: {(a,b,c)} vs {r}"

    # 재추출이 실제로 흔들리는가 — 전부 같은 값이면 부트스트랩이 아니다
    rng = np.random.default_rng(0)
    W = rng.multinomial(4, np.full(4, .25), size=200).astype(float)
    M = fold_lots(y, old, mix, "A", "B", lot_id, 4)
    aa, bb, _ = abc(W @ M)
    assert aa.std() > 0, "재추출이 안 흔들린다 — 가중이 안 먹었다"
    print(f"demo ok — 해소/이동/혼합을 가르고 decompose와 일치. "
          f"재추출 a 범위 {aa.min():.0f}~{aa.max():.0f}")


def main():
    import sklearn
    print("=" * 84)
    print(f"  환경 sklearn {sklearn.__version__} · numpy {np.__version__} · 재추출 B={B:,}")
    print("=" * 84)
    d, F = load()
    y, folds, seeds = d["cls"], d["folds"], d["seeds"]
    lots, lot_id = np.unique(d["lot"], return_inverse=True)
    n_lots = len(lots)

    preds = []
    for name, ks in STAGES:
        X = np.column_stack([F[k] for k in ks])
        preds.append([oof(X, y, folds[s]) for s in range(len(seeds))])
        print(f"  적합 완료 {name}", flush=True)

    rng = np.random.default_rng(SEED)
    W = rng.multinomial(n_lots, np.full(n_lots, 1.0 / n_lots), size=B).astype(float)
    print(f"\n  lot {n_lots:,}개 · 웨이퍼 {len(y):,}장 · seed {len(seeds)}개\n")

    for old, new, ci, cj, label in TRANS:
        print("=" * 84)
        print(f"[{label}]   쌍 {ci}↔{cj}")
        print("=" * 84)
        print(f"  {'seed':<6}{'상호율':>14}{'a(해소)':>9}{'b(이동)':>9}{'c(정답)':>9}"
              f"{'share':>8}{'  share 95% 구간':>18}{'a>0':>6}{'a>|b|':>7}{'c>0':>6}")
        pool = []
        for s in range(len(seeds)):
            M = fold_lots(y, preds[old][s], preds[new][s], ci, cj, lot_id, n_lots)
            a0, b0, c0 = abc(M.sum(0))
            A, Bv, C = abc(W @ M)
            sh = share_of(A, Bv)
            lo, hi = np.nanpercentile(sh, [2.5, 97.5])
            pool.append((sh, A, Bv, C))
            print(f"  {seeds[s]:<6}"
                  f"{mutual(y, preds[old][s], ci, cj):>6.3f}→{mutual(y, preds[new][s], ci, cj):.3f}"
                  f"{a0:>9.0f}{b0:>+9.0f}{c0:>+9.0f}{share_of(a0, b0):>8.1%}"
                  f"{'  [' + f'{lo:.1%}, {hi:.1%}' + ']':>18}"
                  f"{np.mean(A > 0):>6.0%}{np.mean(A > np.abs(Bv)):>7.0%}{np.mean(C > 0):>6.0%}")
        sh = np.concatenate([p[0] for p in pool])
        A = np.concatenate([p[1] for p in pool])
        Bv = np.concatenate([p[2] for p in pool])
        C = np.concatenate([p[3] for p in pool])
        lo, hi = np.nanpercentile(sh, [2.5, 97.5])
        print(f"  {'(합)':<6}{'':>14}{'':>9}{'':>9}{'':>9}{np.nanmedian(sh):>8.1%}"
              f"{'  [' + f'{lo:.1%}, {hi:.1%}' + ']':>18}"
              f"{np.mean(A > 0):>6.0%}{np.mean(A > np.abs(Bv)):>7.0%}{np.mean(C > 0):>6.0%}")
        print(f"\n  → 사전 등록 ①a>0 {np.mean(A > 0):.1%} · ②a>|b| "
              f"{np.mean(A > np.abs(Bv)):.1%} · ③c>0 {np.mean(C > 0):.1%}\n")


if __name__ == "__main__":
    demo()
    main()
