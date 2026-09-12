# -*- coding: utf-8 -*-
"""5차 키트 채점 — 재현값을 정답키와 대조한다.

**채점 방식은 검토요청.md §A-4에 미리 공개했다.** 맞히기 게임이 아니라 문서 검사다.

불일치를 세 종류로 가른다:
  ① 정밀도  — rtol을 낮추면 맞는다
  ② 스케일  — 값은 다른데 **상관이 높다** (상수/정규화를 다르게 읽었다)
  ③ 정의    — 상관도 낮다 (다른 양을 계산했다)

**중간량 4종(n_valid·n_band·f_band·cc_n)이 맞고 최종값만 틀리면 갈라진 곳은 마지막 식이다.**

돌리기: ./.venv/Scripts/python.exe src/grade_kit5.py <재현값.csv>
"""
import sys

import numpy as np

import config

KEY = config.DATA_PROCESSED / "kit5_answer_key.npz"
COLS = {"coverage": "cov", "edge_contrast": "ctr", "circ_var": "cv",
        "radial_contrast": "rc", "mid_peak": "mp", "cc_compact": "cc"}
MID = ["n_valid", "n_band", "f_band", "cc_n"]
TOLS = [1e-9, 1e-6, 1e-3]

# ⚠ np.isclose의 기본 atol은 1e-8이다. coverage는 0~1, mid_peak는 최소 0.004라
# 기본값을 쓰면 **작은 값에서 틀린 구현이 조용히 통과한다** (가짜 제출에 +1e-8을
# 심었더니 100% 일치로 나왔다). 전부 atol=0.0으로 못박는다.


def read_csv(path):
    raw = open(path, encoding="utf-8-sig").read().replace("\r\n", "\n").strip().split("\n")
    head = [h.strip() for h in raw[0].split(",")]
    body = np.array([r.split(",") for r in raw[1:]], dtype=object)
    out = {}
    for j, h in enumerate(head):
        col = np.array([str(v).strip() for v in body[:, j]])
        out[h] = np.array([np.nan if v.lower() in ("", "nan", "na", "none") else float(v)
                           for v in col])
    return out


def classify(a, b):
    """불일치의 종류를 가른다."""
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() < 2:
        return "판정 불가(유한 비교 부족)"
    for t in TOLS:
        if np.isclose(a[both], b[both], rtol=t, atol=0.0).mean() >= 0.99:
            return f"① 정밀도 (rtol {t:g}에서 일치)"
    r = np.corrcoef(a[both], b[both])[0, 1]
    if abs(r) >= 0.99:
        return f"② 스케일/상수 (상관 {r:+.4f})"
    return f"③ 정의 다름 (상관 {r:+.4f})"


def grade(path):
    sub = read_csv(path)
    key = dict(np.load(KEY, allow_pickle=True).items())
    assert "idx" in sub, "재현값.csv에 idx 열이 없다"

    order = {int(v): i for i, v in enumerate(sub["idx"])}
    miss = [int(v) for v in key["idx"] if int(v) not in order]
    take = np.array([order[int(v)] for v in key["idx"] if int(v) in order])
    have = np.array([i for i, v in enumerate(key["idx"]) if int(v) in order])
    print(f"제출 {len(sub['idx']):,}행 / 정답키 {len(key['idx']):,}행 · 누락 {len(miss):,}행\n")

    print("=== 중간량 (여기가 맞아야 최종값 불일치를 마지막 식으로 좁힐 수 있다) ===")
    mid_ok = {}
    for c in MID:
        if c not in sub:
            print(f"  {c:<10} 제출 없음"); mid_ok[c] = None; continue
        a, b = sub[c][take], key[c][have].astype(float)
        eq = np.isclose(a, b, rtol=0, atol=0, equal_nan=True)
        mid_ok[c] = eq.mean()
        print(f"  {c:<10} 일치 {eq.mean():>7.2%}" +
              ("" if eq.all() else f"   {classify(a, b)}"))

    print("\n=== 특징 6종 ===")
    print(f"  {'특징':<17}{'엄밀 일치':>10}{'NaN 위치':>10}   진단")
    print("  " + "-" * 68)
    for name, k in COLS.items():
        if name not in sub:
            print(f"  {name:<17}{'미제출':>10}"); continue
        a, b = sub[name][take], key[k][have].astype(float)
        eq = np.isclose(a, b, rtol=1e-9, atol=0.0, equal_nan=True)
        nan_eq = (np.isnan(a) == np.isnan(b)).mean()
        diag = "✅" if eq.all() else classify(a, b)
        print(f"  {name:<17}{eq.mean():>10.2%}{nan_eq:>10.2%}   {diag}")

    print("\n  ⚠ 불일치가 곧 문서 결함은 아니다. 해당 정의식과 내 코드를 직접 대조한다.")
    if any(v is not None and v < 1 for v in mid_ok.values()):
        print("  ⚠ 중간량부터 갈렸다 — 최종 식을 보기 전에 밴드/연결성분 정의부터 본다.")


def demo():
    """자기 검사 — 세 종류의 불일치를 실제로 가르는가."""
    key = dict(np.load(KEY, allow_pickle=True).items())
    t = key["cov"].astype(float)
    assert classify(t, t).startswith("①"), "같은 값을 못 맞춘다"
    eps = t + 1e-8                      # atol 기본값(1e-8)에 먹히면 안 된다
    assert not np.isclose(eps, t, rtol=1e-9, atol=0.0, equal_nan=True).all(),         "atol이 살아 있어 작은 오차를 통과시킨다"
    assert classify(t * 3.0 + 1.0, t).startswith("②"), "선형 변환을 스케일로 못 가른다"
    rng = np.random.default_rng(0)
    assert classify(rng.permutation(t), t).startswith("③"), "무작위를 정의 차이로 못 가른다"
    print("demo ok — 동일/선형변환/무작위를 ①②③으로 가른다")


if __name__ == "__main__":
    demo()
    if len(sys.argv) > 1:
        print()
        grade(sys.argv[1])
    else:
        print("\n사용법: src/grade_kit5.py <재현값.csv>")
