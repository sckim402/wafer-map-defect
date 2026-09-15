"""11월 목록 #4 — `cc_compact` 동점→NaN 수정판을 **정식 경로에 적용**하고 판정한다 (D-046).

실행 (두 환경 모두 돌린다):
    ./.venv/Scripts/python.exe   -u src/w9_cc_apply.py     # sklearn 1.7.2 (제출본 판)
    ./.venv19/Scripts/python.exe -u src/w9_cc_apply.py     # sklearn 1.9.0

**사전 등록한 판정선 — D-024 §5에서 정한 것을 그대로 쓴다. 새로 정하지 않는다.**
    ① 귀무 검사(이면군 8변환)를 **먼저** 통과해야 한다 (§3-3). 못 하면 성능은 볼 필요가 없다
    ② macro-F1 변화가 **seed 간 흔들림 ±0.01**을 넘지 않으면 「수정해도 결론이 같다」
    ③ **최대 병목 쌍이 바뀌면** 그 자체가 결과다 (초록의 결론이 병목이기 때문)

**기대값** (D-025 (나), 1.7.2에서 측정됨):
    귀무 3.877e-15 · 전수 NaN 138장(0.54%) · macro-F1 0.8368 → 0.8353 (Δ −0.0015) · 병목 불변
⚠ **기대값은 판정선이 아니다.** 일치하면 재현, 어긋나면 그 자체가 관찰이다.
"""
import sys

import numpy as np
from sklearn.metrics import f1_score

import config
from shape2 import shape2, build
from w6_checks import load_ordered, DIHEDRAL, TOL, N_PER_CLASS
from w6_impact import invariance_of
from model8_final import oof, pair_rates, SETS

KEYS6 = ("cov", "ctr", "cv", "rc", "mp", "cc")


def env():
    import sklearn, scipy
    return (f"sklearn {sklearn.__version__} / numpy {np.__version__} "
            f"/ scipy {scipy.__version__}")


def main():
    print("=" * 80)
    print(f"  환경: {env()}")
    print("=" * 80)
    maps, cls, lot, folds, seeds = load_ordered()
    L = config.PATTERN_CLASSES

    # ── [A] 귀무 검사가 먼저다 (§3-3) ──────────────────────────
    print("\n[A] 귀무 검사 — 이면군 8변환 불변. 통과 못 하면 성능은 볼 필요가 없다")
    rng = np.random.default_rng(config.SEED)
    pick = np.concatenate([
        rng.choice(np.nonzero(cls == c)[0],
                   min(N_PER_CLASS, int((cls == c).sum())), replace=False)
        for c in L])
    print(f"    표본 {len(pick)}장 · 합격선 상대오차 < {TOL:g}\n")
    print(f"    {'판':<28}{'최대 상대오차':>13}{'변환':>10}{'불합격':>9}{'NaN불일치':>10}  판정")
    print("    " + "-" * 72)
    old_ok = invariance_of(lambda m: shape2(m, tie_nan=False)["cc_compact"],
                           maps, pick, "cc_compact [옛 판]")
    new_ok = invariance_of(lambda m: shape2(m, tie_nan=True)["cc_compact"],
                           maps, pick, "cc_compact [동점→NaN]")
    if not new_ok:
        print("\n  ❌ 수정판이 귀무 검사를 통과하지 못했다. 성능은 보지 않는다.")
        return 1

    # ── [B] 전수 재계산 — NaN이 얼마나 늘었나 ──────────────────
    print("\n[B] 전수 재계산 — 결측이 얼마나 늘었나")
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        y, idx = z["cls"].astype(str), z["idx_in_cls"]
    old = np.load(config.DATA_PROCESSED / "shape2.npz", allow_pickle=True)["cc_compact"]
    new = build(y, idx, tie_nan=True)["cc_compact"]
    n = len(y)
    no, nn = int(np.isnan(old).sum()), int(np.isnan(new).sum())
    print(f"    옛 판 NaN {no:,}장 ({no/n*100:.2f}%)  →  수정판 NaN {nn:,}장 ({nn/n*100:.2f}%)"
          f"   추가 {nn-no:,}장")
    both = np.isfinite(old) & np.isfinite(new)
    print(f"    둘 다 유한한 {int(both.sum()):,}장에서 최대 상대차 "
          f"{np.max(np.abs(new[both]-old[both])/np.maximum(np.abs(old[both]),1e-12)):.3e}"
          f"   ← 가드는 값을 바꾸지 않고 **지우기만** 해야 한다")
    print(f"    {'클래스':<12}{'추가 NaN':>10}{'비율':>9}")
    for c in L:
        m = y == c
        add = int((np.isnan(new) & ~np.isnan(old))[m].sum())
        print(f"    {c:<12}{add:>10,}{add/max(int(m.sum()),1)*100:>8.2f}%")

    # ── [C] 성능·병목 ──────────────────────────────────────────
    print("\n[C] 6종 macro-F1 과 혼동 쌍 — seed 0/1/2")
    d = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        for k in ("cov", "ctr", "cv"):
            d[k] = z[k]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        d["rc"], d["mp"] = z["radial_contrast"], z["mid_peak"]

    res = {}
    for tag, cc in (("옛 판", old), ("동점→NaN", new)):
        d["cc"] = cc
        X = np.column_stack([d[k] for k in KEYS6])
        ps = [oof(X, y, folds[s]) for s in range(len(seeds))]
        f1 = np.array([f1_score(y, p, labels=L, average=None, zero_division=0) for p in ps])
        rates = [pair_rates(y, p, L) for p in ps]
        mean = {k: float(np.mean([r[k] for r in rates])) for k in rates[0]}
        top = max(mean, key=mean.get)
        res[tag] = dict(macro=f1.mean(axis=1), per=f1.mean(axis=0), top=top, mean=mean)
        print(f"    {tag:<10} macro {f1.mean():.4f}  (seed별 "
              f"{', '.join(f'{v:.4f}' for v in f1.mean(axis=1))})   최대 병목 "
              f"{top[0]}↔{top[1]} {mean[top]:.3f}")

    a, b = res["옛 판"], res["동점→NaN"]
    delta = b["macro"].mean() - a["macro"].mean()
    print(f"\n    Δ macro-F1 = {delta:+.4f}   (사전 등록선 ±{0.01:g})")
    print(f"    {'클래스':<12}{'옛 판':>9}{'동점→NaN':>11}{'차':>9}")
    for i, c in enumerate(L):
        print(f"    {c:<12}{a['per'][i]:>9.3f}{b['per'][i]:>11.3f}{b['per'][i]-a['per'][i]:>+9.3f}")
    print(f"\n    {'쌍':<26}{'옛 판':>9}{'동점→NaN':>11}{'차':>9}")
    for k in sorted(a["mean"], key=lambda k: -a["mean"][k])[:5]:
        print(f"    {k[0]+' ↔ '+k[1]:<26}{a['mean'][k]:>9.3f}{b['mean'][k]:>11.3f}"
              f"{b['mean'][k]-a['mean'][k]:>+9.3f}")

    # ── 판정 ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    within = abs(delta) <= 0.01
    same_top = a["top"] == b["top"]
    print(f"  ① 귀무 검사      {'✅ 통과' if new_ok else '❌'}   (옛 판은 {'통과' if old_ok else '불합격'})")
    print(f"  ② |Δ macro| ≤ 0.01  {'✅' if within else '❌'}  ({abs(delta):.4f})")
    print(f"  ③ 최대 병목 불변    {'✅' if same_top else '❌'}  "
          f"{a['top'][0]}↔{a['top'][1]} → {b['top'][0]}↔{b['top'][1]}")
    print(f"\n  → {'채택 (수정해도 결론이 같다)' if (within and same_top) else '재검토 — 그 자체가 결과다'}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
