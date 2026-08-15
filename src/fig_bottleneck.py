"""Figure 1 — 병목 이동 그림. 초록의 주장 전체가 이 한 장이다.

실행:
    python -u src/fig_bottleneck.py

## 왜 이 그림인가

로드맵 §4의 기존 후보(`azimuth_separation.png` 3패널)는 **v3.3 시점 것이라
맞지 않는다.** 그때는 circular variance가 헤드라인이었으나 D-012에서 역할이
정정됐고, 초록의 주장도 *"병목이 세 번 이동했다"*(D-018)로 바뀌었다.

**이 그림은 그 주장 자체다.**
세 번의 이동이 보이고, **마지막 선 하나만 평평한 것이 곧 결론**이다.
8×8 혼동 행렬보다 읽기 쉽고 1페이지 제약에도 맞는다.

## 설계

- x축: 특징 집합 (가장자리 3종 → +반경 2종 → +형상 1종)
- y축: 혼동 쌍 상호 오분류율 `(i→j + j→i) / (n_i + n_j)`
- **범례 대신 선 끝에 직접 라벨** — 인쇄물에서 훨씬 읽기 쉽다
- **`Edge-Loc↔Loc`만 강조**, 나머지는 흐리게. 그 하나가 결론이므로
- **축 라벨은 영문** (로드맵 §4)

## 재현성

수치를 하드코딩하지 않고 **캐시된 특징 + D-003 분할에서 매번 다시 계산한다.**
seed 3개 평균이며, 같은 값이 `docs/abstract_draft.md`와 D-018에 인용돼 있다.
→ 그림과 본문 숫자가 갈라질 수 없다.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

import config

plt.rcParams["font.family"] = "DejaVu Sans"      # 영문 라벨만 쓴다

STAGES = [
    ("Edge (3)",      ("cov", "ctr", "cv")),
    ("+ Radial (5)",  ("cov", "ctr", "cv", "rc", "mp")),
    ("+ Shape (6)",   ("cov", "ctr", "cv", "rc", "mp", "cc")),
]
# 그릴 쌍 — 3종 시점 상위 4개. 마지막 하나가 강조 대상이다
SHOW = [("Center", "Loc"), ("Loc", "Scratch"),
        ("Center", "Edge-Loc"), ("Edge-Loc", "Loc")]
HILITE = ("Edge-Loc", "Loc")

OUT_DIR = config.ROOT / "figures" / "keep"


def oof(X, y, folds):
    p = np.empty(len(y), dtype=object)
    for f in np.unique(folds):
        te = folds == f
        p[te] = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0,
            class_weight="balanced").fit(X[~te], y[~te]).predict(X[te])
    return p.astype(str)


def pair_rate(y, p, L, a, b):
    M = confusion_matrix(y, p, labels=L)
    i, j = L.index(a), L.index(b)
    return (M[i, j] + M[j, i]) / (M[i].sum() + M[j].sum())


def main():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        y, F, seeds = z["cls"].astype(str), z["folds"], z["seeds"]
    d = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        for k in ("cov", "ctr", "cv"):
            d[k] = z[k]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        d["rc"], d["mp"] = z["radial_contrast"], z["mid_peak"]
    with np.load(config.DATA_PROCESSED / "shape2.npz", allow_pickle=True) as z:
        d["cc"] = z["cc_compact"]
    L = config.PATTERN_CLASSES

    rates = {pr: [] for pr in SHOW}
    macros = []
    for nm, keys in STAGES:
        X = np.column_stack([d[k] for k in keys])
        ps = [oof(X, y, F[s]) for s in range(len(seeds))]
        macros.append(np.mean([f1_score(y, p, labels=L, average="macro",
                                        zero_division=0) for p in ps]))
        for pr in SHOW:
            rates[pr].append(np.mean([pair_rate(y, p, L, *pr) for p in ps]))
        print(f"  {nm:<14} macro {macros[-1]:.3f}", flush=True)

    # ── 수치를 텍스트로도 남긴다 (본문·D-018과 대조용) ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    txt = OUT_DIR / "bottleneck_shift.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("혼동 쌍 상호 오분류율 — seed 3개 평균, D-003 분할 out-of-fold\n")
        f.write(f"{'pair':<24}" + "".join(f"{n:>15}" for n, _ in STAGES) + "\n")
        for pr in SHOW:
            f.write(f"{pr[0]+' <-> '+pr[1]:<24}"
                    + "".join(f"{v:>15.3f}" for v in rates[pr]) + "\n")
        f.write(f"\n{'macro-F1':<24}" + "".join(f"{m:>15.3f}" for m in macros) + "\n")
    print(f"\n수치 저장: {txt}")

    # ── 그림 ──────────────────────────────────────────────
    # 세로를 줄였다 — 초록은 텍스트가 대부분이라 그림이 납작해야 한다
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    xs = np.arange(len(STAGES))
    ymax = max(max(v) for v in rates.values())

    for pr in SHOW:
        hot = pr == HILITE
        ax.plot(xs, rates[pr], "o-",
                lw=2.8 if hot else 1.4, ms=7 if hot else 5,
                color="#c62828" if hot else "#9e9e9e",
                zorder=3 if hot else 2)

    # 끝점 라벨 — 겹치면 세로로 밀어낸다.
    # 구분자는 '↔'다. 클래스명에 하이픈이 있어 '–'를 쓰면
    # `Center–Edge-Loc`처럼 읽혀 쌍인지 이름인지 구분되지 않는다.
    MIN_GAP = ymax * 0.075
    placed, y_lab = [], {}
    for pr in sorted(SHOW, key=lambda p: -rates[p][-1]):   # 위에서 아래로
        yv = rates[pr][-1]
        while any(abs(yv - py) < MIN_GAP for py in placed):
            yv -= MIN_GAP * 0.5
        placed.append(yv)
        y_lab[pr] = yv

    for pr in SHOW:
        hot = pr == HILITE
        ax.annotate(f"{pr[0]} ↔ {pr[1]}", (xs[-1], y_lab[pr]),
                    xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9, annotation_clip=False,
                    color="#c62828" if hot else "#616161",
                    fontweight="bold" if hot else "normal")

    # macro-F1을 x축 눈금 둘째 줄로 — 아래 여백을 먹지 않는다
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\n(macro-F1 {m:.3f})" for (n, _), m in zip(STAGES, macros)],
                       fontsize=10)
    ax.set_xlim(-0.15, len(STAGES) - 1 + 1.05)
    ax.set_ylim(0, ymax * 1.10)
    ax.set_xlabel("Feature set", fontsize=11, labelpad=8)
    ax.set_ylabel("Mutual misclassification rate", fontsize=11)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "bottleneck_shift.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"저장: {out}")
    print("\n  읽는 법: 세 선은 내려가고 **강조된 한 선만 평평하다.**")
    print("  그 하나가 D-018의 결론 — 남은 병목은 특징 부재가 아니라")
    print("  두 패턴 사이에 경계가 없어서 남는다.")
    print("\n  ※ `figures/*`는 .gitignore 대상이나 `figures/keep/`는 예외다.")
    print("     이 파일은 커밋된다.")


if __name__ == "__main__":
    main()
