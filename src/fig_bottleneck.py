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

## 크기 — 최종 인쇄 치수로 그린다 (2026-08-20 개정)

**이전 판은 `figsize=(7.0, 3.6)`으로 그려 DOC에 9.6 cm(=3.78 in) 폭으로 넣었다.**
축소율이 **0.548**이라 코드의 pt가 그대로 인쇄되지 않았다 —
끝점 라벨 9 pt → **실제 4.9 pt**, 눈금 10 pt → 5.5 pt.
로드맵 §4가 *"그림 눈금 5pt 가독"*을 확인 항목으로 남긴 것이 이 지점이다.

**해법은 폰트를 키우는 게 아니라 캔버스를 최종 치수로 잡는 것이다.**
`figsize = (9.6cm, 4.87cm)`로 그리면 축소율이 1이 되고,
**지면 점유는 그대로인 채 코드의 pt = 인쇄 pt**가 된다.
같은 9.6 cm 안에서 끝점 라벨이 4.9 → 6.5 pt, 눈금이 5.5 → 7 pt로 커진다.

두 가지가 여기에 딸려 온다.

1. **`bbox_inches="tight"`를 쓰지 않는다.** 내용에 맞춰 잘라내면 저장된 폭이
   `figsize`와 달라져 *"100%로 넣는다"*는 전제가 깨진다. `tight_layout`만 쓰고
   그대로 저장해 **출력 픽셀 = figsize × dpi**를 보장한다.
   끝점 라벨은 `xlim` 오른쪽 여백 안에 들어가므로 잘리지 않는다.
2. **선 굵기·마커도 절대 pt다.** 축소율이 1이 되면 `lw=2.8`은 그대로 2.8 pt라
   작은 그림에서 과하다. 굵기와 마커를 같이 내린다.

**dpi는 600 — 단 이유는 "PDF가 600 dpi가 된다"가 아니다. (2026-08-20 정정)**
처음에 *"작은 글자는 300 dpi에서 획이 뭉친다"*고 적었는데 **기제를 잘못 봤다.**
DOC → PDF 내보내기에서 **Word가 이미지를 200 dpi로 리샘플한다** — 실측 결과
제출 PDF 안의 이미지는 **754 × 383 px**이고, 이전 제출본(v2)도 같았다.
**source dpi를 아무리 올려도 PDF는 200 dpi다.**

그래도 600을 유지하는 이유는 다르다: **200 dpi로 내려갈 때 3:1 정수배
축소(supersampling)가 되어**, 300 dpi에서 1.5:1로 줄이는 것보다 글자 획이 깨끗하다.
**해상도를 얻으려는 게 아니라 리샘플 품질을 얻으려는 것이다.**

인코딩은 **FlateDecode(무손실)**로 확인됐다 — JPEG 재인코딩이었다면 선화 글자
주변에 링잉이 생겼을 것이다. 그건 일어나지 않는다.

→ **가독성의 실질 이득은 dpi가 아니라 글자 크기에서 나온다.**
200 dpi 래스터에서 끝점 라벨의 cap height가 **9.5 px → 12.6 px**가 된다.

## 재현성

수치를 하드코딩하지 않고 **캐시된 특징 + D-003 분할에서 매번 다시 계산한다.**
seed 3개 평균이며, 같은 값이 `docs/abstract_draft.md`와 D-018에 인용돼 있다.
→ 그림과 본문 숫자가 갈라질 수 없다.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

import config

plt.rcParams["font.family"] = "DejaVu Sans"      # 영문 라벨만 쓴다
# Windows 콘솔은 cp949다. 출력을 파일로 넘기면 아래 한글 print가
# UnicodeEncodeError로 죽는다 — 그림은 이미 저장된 뒤라 더 헷갈린다
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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

# ── 인쇄 치수 — 여기가 이 그림의 유일한 크기 근거다 ──────────
# 9.6 cm는 D-020에서 1페이지 실측으로 확정된 값이다. 바꾸지 않는다.
# 높이는 이전 판의 지면 점유(9.6 × 1050/2070)를 그대로 유지한다.
CM = 1 / 2.54
PRINT_W_CM, PRINT_H_CM = 9.6, 4.87
DPI = 600

# 전부 인쇄 pt다 — 축소율이 1이므로 코드 값 = 종이 위 값
FS_TICK, FS_AXIS, FS_LABEL, FS_XTICK = 7.0, 7.5, 6.5, 6.8


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
    fig, ax = plt.subplots(figsize=(PRINT_W_CM * CM, PRINT_H_CM * CM))
    xs = np.arange(len(STAGES))
    ymax = max(max(v) for v in rates.values())

    # 회색을 이전 `#9e9e9e`보다 어둡게 잡는다 — 선이 얇아지면 같은 명도라도
    # 종이에서 옅어 보인다. 강조 대비는 굵기가 이미 충분히 만든다
    for pr in SHOW:
        hot = pr == HILITE
        ax.plot(xs, rates[pr], "o-",
                lw=1.6 if hot else 0.9, ms=3.6 if hot else 2.6,
                color="#c62828" if hot else "#757575",
                zorder=3 if hot else 2)

    # 값을 점 위에 찍어보고 **뺐다.** 강조 라벨 바로 위에 놓여 둘이 한 덩어리로
    # 읽혔다. 좁은 그림에서 요소를 더하는 것은 가독성을 깎는다 —
    # 잔여 병목의 크기는 y축 0.10 격자선이 이미 말해주고, 정확한 값은 본문에 있다.
    # **이 그림의 주장은 숫자가 아니라 "한 선만 평평하다"다.**

    # macro-F1을 x축 눈금 둘째 줄로 — 아래 여백을 먹지 않는다
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\n(macro-F1 {m:.3f})" for (n, _), m in zip(STAGES, macros)],
                       fontsize=FS_XTICK)
    # y 눈금을 9개에서 5개로 — 작은 그림에서 눈금 글자끼리 붙으면
    # 개별 숫자가 커져도 덩어리로 읽힌다. 간격이 곧 가독성이다
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.tick_params(labelsize=FS_TICK, length=2, pad=2)
    ax.set_xlim(-0.12, len(STAGES) - 1 + 1.10)
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xlabel("Feature set", fontsize=FS_AXIS, labelpad=4)
    ax.set_ylabel("Mutual misclassification rate", fontsize=FS_AXIS, labelpad=3)
    ax.grid(axis="y", ls=":", lw=0.4, alpha=0.55)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)

    fig.tight_layout(pad=0.3)

    # ── 끝점 라벨 — 레이아웃이 확정된 뒤에 놓는다 ───────────────
    # 겹칠 때 밀어낼 간격은 **글자 크기에서 나와야 한다.**
    # 이전 판은 `MIN_GAP = ymax * 0.075`라는 데이터 비율이었다. 캔버스를
    # 인쇄 치수로 줄이자 같은 비율이 6.9 pt가 되어 `Center↔Loc`과
    # `Loc↔Scratch`가 붙었다 — **비율은 크기가 바뀌면 따라오지 않는다.**
    # 그려진 축에서 pt를 데이터 단위로 환산한다. 치수를 다시 바꿔도 유지된다.
    fig.canvas.draw()
    inv = ax.transData.inverted()
    gap_px = FS_LABEL * 1.45 * fig.dpi / 72          # 행간 1.45배
    MIN_GAP = abs(inv.transform((0, gap_px))[1] - inv.transform((0, 0))[1])

    # 구분자는 '↔'다. 클래스명에 하이픈이 있어 '–'를 쓰면
    # `Center–Edge-Loc`처럼 읽혀 쌍인지 이름인지 구분되지 않는다.
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
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=FS_LABEL, annotation_clip=False,
                    color="#c62828" if hot else "#424242",
                    fontweight="bold" if hot else "normal")

    out = OUT_DIR / "bottleneck_shift.png"
    # bbox_inches="tight"를 쓰지 않는다 — 저장 폭 = figsize × dpi여야
    # DOC에 9.6 cm로 넣었을 때 축소율이 정확히 1이 된다
    fig.savefig(out, dpi=DPI)
    print(f"저장: {out}  ({PRINT_W_CM} × {PRINT_H_CM} cm @ {DPI} dpi, 축소율 1.0)")
    print("\n  읽는 법: 세 선은 내려가고 **강조된 한 선만 평평하다.**")
    print("  그 하나가 D-018의 결론 — 남은 병목은 특징 부재가 아니라")
    print("  두 패턴 사이에 경계가 없어서 남는다.")
    print("\n  ※ `figures/*`는 .gitignore 대상이나 `figures/keep/`는 예외다.")
    print("     이 파일은 커밋된다.")


if __name__ == "__main__":
    main()
