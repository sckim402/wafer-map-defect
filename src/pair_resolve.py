# -*- coding: utf-8 -*-
"""초록의 인과 문장을 판정한다 — *"앞의 두 병목은 특징 부재가 원인이었고
특징을 설계하자 해소됐다"* (D-037).

**왜 필요한가.** D-026이 ***"상호 오분류율이 줄면 그 쌍이 풀린 것"***을 금지했다 —
**A·B가 둘 다 C로 틀려도 줄어든다.** 그래서 `a`·`b`·정답 순증을 **함께** 봐야 한다고
규칙을 적어 뒀는데, **실제 두 전이(3종→5종, 5종→6종)에 적용한 기록이 0건이다.**
규칙만 있고 적용이 없었다. 초록은 그 사이에 **인과 문장**을 싣고 있다.

**분해** (쌍 `P = {i, j}`, 나머지를 `O`라 할 때):
  a = (쌍 오류 -> 정답) − (정답 -> 쌍 오류)      ... 진짜 해소
  b = (쌍 오류 -> O 오류) − (O 오류 -> 쌍 오류)  ... 오류의 이동
  c = 두 클래스의 정답 순증
**`a`가 감소분의 대부분이어야 「해소」다.** `b`가 크면 **오류가 제3자에게 넘어간 것**이다.

돌리기: ./.venv/Scripts/python.exe src/pair_resolve.py
"""
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, "src")
import config
from abstract_numbers import STAGES, RF, load, mutual


def decompose(y, p_old, p_new, i, j):
    """쌍 (i, j)의 상호 오분류 감소를 a·b·c로 가른다."""
    m = (y == i) | (y == j)                      # 두 클래스의 참 라벨만 본다
    yy, a_, b_ = y[m], p_old[m], p_new[m]

    def cls(pred):
        """각 웨이퍼의 상태: 0=정답, 1=쌍 안 오류, 2=O 오류."""
        other = np.where(yy == i, j, i)
        return np.where(pred == yy, 0, np.where(pred == other, 1, 2))

    s0, s1 = cls(a_), cls(b_)
    n = lambda x, z: int(((s0 == x) & (s1 == z)).sum())
    a = n(1, 0) - n(0, 1)
    b = n(1, 2) - n(2, 1)
    c = int((s1 == 0).sum()) - int((s0 == 0).sum())
    return dict(a=a, b=b, c=c, pair_old=int((s0 == 1).sum()), pair_new=int((s1 == 1).sum()),
                other_old=int((s0 == 2).sum()), other_new=int((s1 == 2).sum()), n=int(m.sum()))


def oof(X, y, folds):
    p = np.empty(len(y), dtype=object)
    for f in np.unique(folds):
        te = folds == f
        p[te] = RandomForestClassifier(**RF).fit(X[~te], y[~te]).predict(X[te])
    return p.astype(str)


def main():
    d, F = load()
    y = d["cls"]
    preds = []
    for name, ks in STAGES:
        X = np.column_stack([F[k] for k in ks])
        preds.append([oof(X, y, d["folds"][s]) for s in range(len(d["seeds"]))])
        print(f"  적합 완료 {name}", flush=True)

    TRANS = [(0, 1, "Center", "Loc", "가장자리 3종 -> +반경 2종"),
             (1, 2, "Loc", "Scratch", "+반경 2종 -> +형상 1종")]
    print("")
    print("=== 초록의 「특징 부재가 원인이었고 설계하자 해소됐다」 판정 ===")
    print("  a = 쌍 오류->정답 순증 (해소)   b = 쌍 오류->제3자 순증 (이동)   c = 정답 순증")
    for old, new, ci, cj, label in TRANS:
        print("")
        print(f"[{label}]  쌍 {ci}↔{cj}")
        print(f"  {'seed':<6}{'상호율':>10}{'a(해소)':>10}{'b(이동)':>10}{'c(정답)':>10}{'a/(a+|b|)':>12}")
        for s in range(len(d["seeds"])):
            r0 = mutual(y, preds[old][s], ci, cj)
            r1 = mutual(y, preds[new][s], ci, cj)
            r = decompose(y, preds[old][s], preds[new][s], ci, cj)
            share = r["a"] / (r["a"] + abs(r["b"])) if (r["a"] + abs(r["b"])) else np.nan
            print(f"  {d['seeds'][s]:<6}{r0:>5.3f}->{r1:.3f}"
                  f"{r['a']:>10}{r['b']:>+10}{r['c']:>+10}{share:>12.1%}")
    return preds


def demo():
    """자기 검사 — **D-026이 든 반례를 실제로 잡는가.**

    ⓐ 쌍 오류가 전부 정답이 되면 `a`만 커야 한다
    ⓑ **쌍 오류가 전부 제3자 오류로 옮겨가면** 상호율은 줄지만 `a=0 · b>0 · c=0`이어야 한다
       — 이것이 *"둘 다 C로 틀려도 줄어든다"*의 형태다
    """
    y = np.array(["A"] * 6 + ["B"] * 6)
    old = np.array(["B"] * 6 + ["A"] * 6)                  # 전부 쌍 안 오류
    fix = np.array(["A"] * 6 + ["B"] * 6)                  # 전부 정답
    mov = np.array(["C"] * 6 + ["C"] * 6)                  # 전부 제3자 오류

    r = decompose(y, old, fix, "A", "B")
    assert (r["a"], r["b"], r["c"]) == (12, 0, 12), f"해소를 못 잡는다: {r}"
    r = decompose(y, old, mov, "A", "B")
    assert r["a"] == 0 and r["b"] == 12 and r["c"] == 0, f"오류 이동을 해소로 센다: {r}"
    assert mutual(y, mov, "A", "B") == 0.0, "상호율이 0으로 안 떨어진다 — 반례가 아니다"
    # ⓒ 섞인 경우: 절반 정답 · 절반 이동
    mix = np.array(["A"] * 3 + ["C"] * 3 + ["B"] * 3 + ["C"] * 3)
    r = decompose(y, old, mix, "A", "B")
    assert (r["a"], r["b"], r["c"]) == (6, 6, 6), f"섞인 경우를 못 가른다: {r}"
    print("demo ok — 해소(12,0,12) / 이동(0,12,0) / 절반(6,6,6)을 가른다")


if __name__ == "__main__":
    demo()
    main()
