"""잔여 ① — `size`의 +0.111은 **반경 정보의 대리**였는가.

실행:
    python -u src/model8_size.py

## 질문

`model8.py`의 대조 세트에서 **3종 + `size`가 macro 0.582 → 0.693 (+0.111)**이었다.
그때는 반경 특징(F1a·F1b)이 없었다. **`size`가 무엇을 대신하고 있었는지 모른 채
"교란이니 넣지 않는다"로 끝냈다.**

D-015에서 F1a·F1b가 들어오며 macro가 0.768이 됐다. 그렇다면 —

> **`size`가 벌던 +0.111은 반경 구조 정보였는가?**
> 그렇다면 반경 특징이 있는 자리에서 `size`는 **더 이상 벌 것이 없어야 한다.**

## ⚠ 이것은 채택 검토가 아니다

**`size`는 어떤 결과가 나와도 최종 특징에 넣지 않는다.** 크기 층 간 전이가
안 되는 것이 이미 측정됐다(`w3_transfer.md`). **이 실험은 진단이다** —
*"우리가 만든 특징이 무엇을 흡수했는가"*를 확인하는 것이고,
답이 어느 쪽이든 **D-015의 근거가 갱신된다.**

## 사전 등록한 판정 기준 (돌리기 전에 고정한다 — §3-9)

기준선은 **D-017에서 쓴 +0.01**을 그대로 쓴다. 새로 만들지 않는다.

| 5종 + `size` 증분 | 판정 |
|---|---|
| **< +0.01** | **대리였다.** F1a·F1b가 `size`가 벌던 것을 흡수했다 |
| **≥ +0.01** | **대리가 아니었다.** 반경으로 설명 안 되는 축이 남아 있다 |

**흡수율 = 1 − (5종+size 증분) / (3종+size 증분)** 로 정량화한다.

**증분은 남은 여유와 함께 읽는다** (§3-12의 천장 효과).
5종 macro가 이미 0.768이라 여유가 0.232뿐이다 — **증분이 작게 나오는 것이
"흡수"인지 "천장"인지 구분해야 하므로, 여유 대비 비율을 같이 낸다.**

**그리고 6종 + `size`도 같이 돈다** (§3-8 — 한 지점에서 되면 나머지도 확인한다).
5종에서 흡수됐는데 6종에서 되살아나면 해석이 달라진다.
"""
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

import config

# Windows 콘솔은 cp949다. 출력을 파일로 넘기면 한글 print가 죽는다
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE3 = ("cov", "ctr", "cv")
BASE5 = BASE3 + ("rc", "mp")
BASE6 = BASE5 + ("cc",)

SETS = {
    "3종":              BASE3,
    "3종 + size":       BASE3 + ("size",),
    "5종 (D-015)":      BASE5,
    "5종 + size":       BASE5 + ("size",),      # ★ 이 실험의 표적
    "6종 (D-017 확정)": BASE6,
    "6종 + size":       BASE6 + ("size",),
}
PAIRS = [("3종", "3종 + size"), ("5종 (D-015)", "5종 + size"),
         ("6종 (D-017 확정)", "6종 + size")]
THRESH = 0.01


def oof(X, y, folds):
    p = np.empty(len(y), dtype=object)
    for f in np.unique(folds):
        te = folds == f
        p[te] = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0,
            class_weight="balanced").fit(X[~te], y[~te]).predict(X[te])
    return p.astype(str)


def pair_rates(y, p, L):
    M = confusion_matrix(y, p, labels=L)
    return {(L[i], L[j]): (M[i, j] + M[j, i]) / (M[i].sum() + M[j].sum())
            for i in range(len(L)) for j in range(i + 1, len(L))}


def load():
    # allow_pickle: 이 프로젝트가 직접 만든 캐시만 읽는다 (src/*.py 공통 관례)
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        y, orig, F, seeds = (z["cls"].astype(str), z["split_orig"].astype(str),
                             z["folds"], z["seeds"])
    d = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        for k in ("cov", "ctr", "cv", "size"):
            d[k] = z[k]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        d["rc"], d["mp"] = z["radial_contrast"], z["mid_peak"]
    with np.load(config.DATA_PROCESSED / "shape2.npz", allow_pickle=True) as z:
        d["cc"] = z["cc_compact"]
    return y, orig, F, seeds, d


def main():
    y, orig, F, seeds, d = load()
    L = config.PATTERN_CLASSES
    print(f"n = {len(y)},  seed {len(seeds)}개,  클래스 {len(L)}종\n")

    res, preds = {}, {}
    for nm, keys in SETS.items():
        X = np.column_stack([d[k] for k in keys])
        ps = [oof(X, y, F[s]) for s in range(len(seeds))]
        preds[nm] = ps
        res[nm] = np.array([f1_score(y, p, labels=L, average=None, zero_division=0)
                            for p in ps])
        print(f"  {nm:<18} macro {res[nm].mean(axis=0).mean():.3f} "
              f"± {res[nm].mean(axis=1).std():.3f}", flush=True)

    m = {k: v.mean(axis=0) for k, v in res.items()}

    print("\n" + "=" * 78)
    print("[1] ★ 판정 — `size` 증분이 반경 특징 도입 후 어떻게 되는가")
    print("=" * 78)
    print(f"  {'기반':<18}{'기반 macro':>12}{'+size':>9}{'증분':>9}"
          f"{'남은 여유':>11}{'여유대비':>10}")
    incs = {}
    for base, plus in PAIRS:
        a, b = m[base].mean(), m[plus].mean()
        head = 1.0 - a
        incs[base] = b - a
        print(f"  {base:<18}{a:>12.3f}{b:>9.3f}{b-a:>+9.3f}{head:>11.3f}"
              f"{(b-a)/head:>10.3f}")

    i3, i5 = incs["3종"], incs["5종 (D-015)"]
    absorb = 1 - i5 / i3 if abs(i3) > 1e-9 else float("nan")
    print(f"\n  흡수율 = 1 − ({i5:+.3f}) / ({i3:+.3f}) = {absorb:.1%}")
    print(f"  사전 등록 기준: 5종+size 증분 {THRESH:+.3f} 미만이면 '대리였다'")
    print(f"  → 실측 {i5:+.3f}  ⇒  **{'대리였다' if i5 < THRESH else '대리가 아니었다'}**")

    print("\n" + "=" * 78)
    print("[2] per-class — 어느 클래스에서 흡수됐는가 (§3-8)")
    print("=" * 78)
    print(f"  {'클래스':<12}{'3종':>8}{'+size':>8}{'증분':>8}   |"
          f"{'5종':>8}{'+size':>8}{'증분':>8}")
    for i, c in enumerate(L):
        print(f"  {c:<12}{m['3종'][i]:>8.3f}{m['3종 + size'][i]:>8.3f}"
              f"{m['3종 + size'][i]-m['3종'][i]:>+8.3f}   |"
              f"{m['5종 (D-015)'][i]:>8.3f}{m['5종 + size'][i]:>8.3f}"
              f"{m['5종 + size'][i]-m['5종 (D-015)'][i]:>+8.3f}")

    print("\n" + "=" * 78)
    print("[3] 혼동 쌍 — 3종에서 size가 풀던 쌍을 반경이 풀었는가")
    print("=" * 78)
    r = {k: [pair_rates(y, p, L) for p in preds[k]] for k in SETS}
    keys = sorted(r["3종"][0], key=lambda k: -np.mean([x[k] for x in r["3종"]]))
    print(f"  {'쌍':<24}{'3종':>8}{'3+sz':>8}{'5종':>8}{'5+sz':>8}"
          f"{'size가 풀던 몫':>15}{'반경이 푼 몫':>14}")
    for k in keys[:6]:
        v = {nm: np.mean([x[k] for x in r[nm]]) for nm in SETS}
        by_size = v["3종"] - v["3종 + size"]
        by_rad = v["3종"] - v["5종 (D-015)"]
        print(f"  {k[0]+' ↔ '+k[1]:<24}{v['3종']:>8.3f}{v['3종 + size']:>8.3f}"
              f"{v['5종 (D-015)']:>8.3f}{v['5종 + size']:>8.3f}"
              f"{by_size:>+15.3f}{by_rad:>+14.3f}")

    print("\n" + "=" * 78)
    print("[4] 기제 — `size`와 각 특징의 순위 상관 (Spearman)")
    print("=" * 78)
    print("  **단 상관은 기제의 방향을 못 정한다** — [1]~[3]의 보조 증거로만 읽는다 (§3-11)")
    print()
    print("  ⚠ **두 열은 다른 양이다. 섞어 인용하면 D-015와 모순처럼 보인다.**")
    print("    · 전체(pooled): 클래스 차이까지 섞여 들어간다 → **신호 자체를 재는 쪽**")
    print("    · 클래스별 |max|: 한 클래스 안에서 size를 따라가는가 → **교란을 재는 쪽**")
    print("    D-015가 *\"F1a는 size 대리가 아니다\"*의 근거로 쓴 것은 **클래스별**이고,")
    print("    그 목적에는 클래스별이 맞다. 아래에서 그 값이 그대로 재현된다.")
    print()
    print(f"    {'특징':<8}{'전체 Spearman':>15}{'클래스별 |max| Pearson':>24}{'그 클래스':>12}")
    for k in ("cov", "ctr", "cv", "rc", "mp", "cc"):
        ok = np.isfinite(d[k]) & np.isfinite(d["size"])
        rho = spearmanr(d["size"][ok], d[k][ok]).statistic
        best = (0.0, "-")
        for c in L:
            msk = ok & (y == c)
            if msk.sum() > 30:
                r = pearsonr(d["size"][msk], d[k][msk]).statistic
                if abs(r) > abs(best[0]):
                    best = (r, c)
        print(f"    {k:<8}{rho:>15.3f}{best[0]:>24.3f}{best[1]:>12}")
    print("\n    → 클래스별 열의 cov −0.457 / rc(F1a) +0.264 / mp(F1b) −0.259는")
    print("      **D-015 표의 값과 일치한다.** D-015는 정정할 것이 없다.")

    print("\n" + "=" * 78)
    print("[5] 외부 분할 (`split_orig`) — 병기 원칙 (D-016)")
    print("=" * 78)
    # 외부는 분할이 하나뿐이라 fold 반복이 불가능하다. 대신 **RF seed를 3개** 돌려
    # 편차를 붙인다 — 단일 fit 하나로 부호를 읽는 것은 §8 위반이다.
    tr, te = orig == "Training", orig == "Test"
    print(f"  외부 train {tr.sum():,}장 / test {te.sum():,}장 · RF seed 3개")
    ext = {}
    for nm, keys in SETS.items():
        X = np.column_stack([d[k] for k in keys])
        ext[nm] = np.array([
            f1_score(y[te],
                     RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                            n_jobs=-1, random_state=rs,
                                            class_weight="balanced"
                                            ).fit(X[tr], y[tr]).predict(X[te]),
                     labels=L, average="macro", zero_division=0)
            for rs in range(3)])
    print(f"\n  {'기반':<18}{'외부':>17}{'+size':>17}{'증분':>10}")
    for base, plus in PAIRS:
        a, b = ext[base], ext[plus]
        print(f"  {base:<18}{a.mean():>11.3f} ±{a.std():.3f}"
              f"{b.mean():>11.3f} ±{b.std():.3f}{b.mean()-a.mean():>+10.3f}")
    print("\n  **주 분할과 외부 분할을 병기한다.** 부호가 갈리면 그것도 결과다")


if __name__ == "__main__":
    main()
