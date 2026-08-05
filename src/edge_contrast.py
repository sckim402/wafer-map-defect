"""edge_contrast 검증 — coverage의 Random 문제를 해결하는가.

실행:
    python src/edge_contrast.py

배경 (docs/w2_confound.md, verify_coverage.py):
    coverage(최외곽 1층 불량 비율)는 Edge-Ring↔Edge-Loc을 AUC 0.971로 가르지만,
    8종 전체에서 **Random이 3위(0.583)**로 Edge-Loc(0.379)보다 높게 나왔다.
    coverage는 '가장자리에 *특이적으로* 몰렸는가'가 아니라 '가장자리 불량 비율'을
    재기 때문에, 전체가 지저분한 웨이퍼는 자동으로 높아진다.

후보 지표:
    edge_contrast = (최외곽 1층 불량률) / (내부 층 불량률)
    -> 균일하게 지저분한 Random은 1에 가깝고, 가장자리에만 몰린 Edge-Ring은 커야 한다.

**채택 전 필수 3가지 (작업지침 §3-3)**:
    ① 알려진 실패 조건에서 먼저 돌린다  ② sanity check  ③ 기존 지표와 같은 조건 비교
    호 길이 비율은 이 절차를 건너뛰었다가 비원형에서 AUC 0.000이었다.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from edge_band import layer_masks, coverage
from azimuth import auc

INNER_FROM, INNER_TO = 3, 6        # 내부 기준층 (2층은 링의 번짐이 섞인다)
ALPHA = 1.0                        # Laplace 평활 — 0으로 나누기 방지


def edge_contrast(m, inner=(INNER_FROM, INNER_TO), alpha=ALPHA, ratio=True):
    """(최외곽 1층 불량률) / (내부 층 불량률). 평활을 넣어 0분모를 막는다."""
    a = np.asarray(m)
    fail = a == config.VAL_FAIL
    ls = layer_masks(a, inner[1])
    f1, n1 = int((fail & ls[0]).sum()), int(ls[0].sum())
    fi = ni = 0
    for l in ls[inner[0] - 1:inner[1]]:
        fi += int((fail & l).sum()); ni += int(l.sum())
    if n1 == 0 or ni == 0:
        return np.nan
    r1 = (f1 + alpha) / (n1 + 2 * alpha)
    ri = (fi + alpha) / (ni + 2 * alpha)
    return r1 / ri if ratio else r1 - ri


def two_fold_auc(Xp, Xn):
    """두 특징을 합쳤을 때의 AUC. 2-fold로 재서 과적합 인상을 피한다."""
    from sklearn.linear_model import LogisticRegression
    X = np.vstack([Xp, Xn]); y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    ok = np.isfinite(X).all(1); X, y = X[ok], y[ok]
    rng = np.random.default_rng(config.SEED)
    fold = rng.integers(0, 2, len(y))
    sc = np.empty(len(y))
    for f in (0, 1):
        tr, te = fold != f, fold == f
        mdl = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        sc[te] = mdl.decision_function(X[te])
    return auc(sc[y == 1], sc[y == 0])


def load(cls, cap=2500, rng=None):
    with np.load(config.DATA_PROCESSED / f"{cls}.npz", allow_pickle=True) as z:
        maps = z["wafer_maps"]
    if len(maps) > cap:
        maps = maps[(rng or np.random.default_rng(config.SEED))
                    .choice(len(maps), cap, replace=False)]
    return maps


def main():
    rng = np.random.default_rng(config.SEED)
    M = {c: load(c, rng=rng) for c in config.ALL_CLASSES}
    F = {}
    for c, maps in M.items():
        F[c] = dict(cov=np.array([coverage(m, 1) for m in maps]),
                    ctr=np.array([edge_contrast(m) for m in maps]),
                    dif=np.array([edge_contrast(m, ratio=False) for m in maps]))
        print(f"{c:<12} {len(maps):>6,}장")

    # ── [1] 8종 분포 ──────────────────────────────────────
    print("\n" + "=" * 74)
    print("[1] 8종 edge_contrast 중앙값 — Random이 1 근처로 내려가는가")
    print("=" * 74)
    order = sorted(F, key=lambda c: -np.nanmedian(F[c]["ctr"]))
    print(f"{'클래스':<14}{'contrast':>11}{'coverage':>11}{'차이(1층-내부)':>16}")
    for c in order:
        print(f"{c:<14}{np.nanmedian(F[c]['ctr']):>11.2f}"
              f"{np.nanmedian(F[c]['cov']):>11.3f}{np.nanmedian(F[c]['dif']):>16.3f}")
    print("\n  Random이 1 근처면 '균일하게 지저분한 것'을 걸러낸다는 뜻.")
    print("  none의 값이 baseline이다 — 1이 아니어도 된다(실제 웨이퍼 가장자리는")
    print("  edge exclusion·핸들링 때문에 원래 불량률이 높다). 상대 비교가 기준이다.")

    # ── [2] 문제 쌍별 비교 ────────────────────────────────
    print("\n" + "=" * 74)
    print("[2] 문제 쌍별 AUC — coverage 단독 vs contrast 단독 vs 둘 다")
    print("=" * 74)
    pairs = [("Edge-Ring", "Random"), ("Edge-Ring", "Near-full"),
             ("Edge-Ring", "Edge-Loc"), ("Edge-Loc", "Random"),
             ("Edge-Loc", "none"), ("Edge-Ring", "Donut")]
    print(f"{'쌍':<26}{'coverage':>10}{'contrast':>10}{'둘 다':>9}{'판정':>10}")
    for p, n in pairs:
        if p not in F or n not in F:
            continue
        ac = auc(F[p]["cov"], F[n]["cov"])
        at = auc(F[p]["ctr"], F[n]["ctr"])
        ab = two_fold_auc(np.c_[F[p]["cov"], F[p]["ctr"]],
                          np.c_[F[n]["cov"], F[n]["ctr"]])
        best = max(ac, at, ab)
        tag = "개선" if ab > max(ac, at) + 0.01 else ("contrast" if at > ac + 0.01 else "-")
        print(f"{p+' vs '+n:<26}{ac:>10.3f}{at:>10.3f}{ab:>9.3f}{tag:>10}")
    print("\n  ★ Edge-Ring vs Random 이 핵심이다. coverage 0.7대에서 얼마나 오르는가.")

    # ── [3] 실패 조건 (합성) ★ 채택 전 필수 ───────────────
    print("\n" + "=" * 74)
    print("[3] 실패 조건 테스트 — 채택 전 필수 (작업지침 §3-3)")
    print("=" * 74)
    srng = np.random.default_rng(0)

    def synth(kind, h=25, w=27, bg=0.05):
        ii, jj = np.mgrid[0:h, 0:w]
        cy, cx = (h - 1) / 2, (w - 1) / 2
        r = np.hypot((ii - cy) / cy, (jj - cx) / cx)
        m = np.where(r <= 1.0, 1, 0).astype(np.uint8)
        v = m == 1
        m[(srng.random(m.shape) < bg) & v] = 2
        if kind == "ring":
            ls = layer_masks(m, 1)
            m[ls[0]] = 2
        return m

    print("\n  (a) 내부가 완전히 깨끗한 링 — 분모가 0이 되는 조건")
    for bg in (0.0, 0.01, 0.05):
        vals = [edge_contrast(synth("ring", bg=bg)) for _ in range(200)]
        print(f"      배경률 {bg:.2f} -> contrast 중앙값 {np.nanmedian(vals):>7.2f}  "
              f"NaN {np.isnan(vals).sum():>3}  최대 {np.nanmax(vals):>7.2f}")
    print("      -> 평활(alpha=1) 덕에 발산하지 않아야 한다")

    print("\n  (b) 균일 무작위 — 밀도를 바꿔도 1 근처를 유지하는가")
    for bg in (0.02, 0.05, 0.15, 0.35):
        vals = [edge_contrast(synth("none", bg=bg)) for _ in range(300)]
        print(f"      배경률 {bg:.2f} -> contrast 중앙값 {np.nanmedian(vals):>7.2f}")
    print("      -> 밀도와 무관하게 평탄해야 '균일함'을 제대로 재는 것이다")

    print("\n  (c) 작은 맵 — 층당 die가 적을 때 안정한가")
    for h, w in ((15, 16), (20, 22), (25, 27), (40, 42)):
        vr = [edge_contrast(synth("ring", h, w)) for _ in range(200)]
        vn = [edge_contrast(synth("none", h, w)) for _ in range(200)]
        print(f"      {h}x{w:<3} -> ring {np.nanmedian(vr):>6.2f}   "
              f"none {np.nanmedian(vn):>5.2f}   비 {np.nanmedian(vr)/np.nanmedian(vn):>5.2f}")
    print("      -> 비가 맵 크기에 따라 무너지면 크기 의존성이 다시 들어온 것이다")

    # ── [4] 중복성과 표본 신뢰도 (v2 추가) ────────────────
    print("\n" + "=" * 74)
    print("[4] 두 지표가 정말 다른 것을 재는가 + 작은 표본의 신뢰구간")
    print("=" * 74)
    print("\n  (a) coverage-contrast 상관 (D-010 뒤집을 조건: |r|>0.9면 하나로 줄인다)")
    for c in ("Edge-Ring", "Edge-Loc", "Random", config.NONE_CLASS):
        if c not in F: continue
        x, y = F[c]["cov"], np.log(np.clip(F[c]["ctr"], 1e-3, None))
        ok = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        print(f"      {c:<12} r = {r:>6.3f}  {'중복 위험' if abs(r) > 0.9 else ''}")

    print("\n  (b) 판별력은 |AUC-0.5| 로 읽는다 — 0.5 미만은 역방향 신호이지 실패가 아니다")
    print(f"      {'쌍':<24}{'|cov-.5|':>10}{'|ctr-.5|':>10}{'우세':>10}")
    for p_, n_ in pairs:
        if p_ not in F or n_ not in F: continue
        dc = abs(auc(F[p_]["cov"], F[n_]["cov"]) - .5)
        dt = abs(auc(F[p_]["ctr"], F[n_]["ctr"]) - .5)
        print(f"      {p_+' vs '+n_:<24}{dc:>10.3f}{dt:>10.3f}"
              f"{('contrast' if dt>dc else 'coverage'):>10}")

    print("\n  (c) Near-full(n=149) 부트스트랩 95% CI — 표본이 작다")
    if "Near-full" in F:
        rb = np.random.default_rng(config.SEED)
        for name, key in (("coverage", "cov"), ("contrast", "ctr")):
            xs = F["Edge-Ring"][key]; ys = F["Near-full"][key]
            xs = xs[np.isfinite(xs)]; ys = ys[np.isfinite(ys)]
            bs = [auc(rb.choice(xs, len(xs)), rb.choice(ys, len(ys))) for _ in range(400)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"      {name:<10} AUC {auc(xs, ys):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    # ── 그림 ──────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    d = [F[c]["ctr"][np.isfinite(F[c]["ctr"])] for c in order]
    ax[0].boxplot(d, tick_labels=order, showfliers=False)
    ax[0].axhline(1.0, ls="--", c="gray"); ax[0].set_yscale("log")
    ax[0].set_ylabel("edge_contrast (log)"); ax[0].set_title("Edge contrast by class")
    ax[0].tick_params(axis="x", rotation=45)
    for c, col in (("Edge-Ring", "#1976d2"), ("Random", "#f9a825"),
                   ("Edge-Loc", "#d32f2f")):
        if c in F:
            ax[1].scatter(F[c]["cov"], F[c]["ctr"], s=4, alpha=.3, c=col, label=c)
    ax[1].set_yscale("log"); ax[1].set_xlabel("coverage"); ax[1].set_ylabel("edge_contrast")
    ax[1].set_title("Do the two features separate Random?")
    ax[1].legend(fontsize=8, markerscale=3)
    fig.tight_layout()
    out = config.FIGURES / "edge_contrast.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
