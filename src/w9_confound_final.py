"""W9 — 최종 6종의 개수·크기 교란 재측정 (사전 등록: docs/w9_confound_final.md).

    python src/w9_confound_final.py --demo    # 합성 실패 조건만 (§3-3)
    python src/w9_confound_final.py --count   # 교란 변수 기술 통계만 (설계 정보, 판정 없음)
    python src/w9_confound_final.py           # demo → M1 → M2 → 판정

교란 변수: N = 전체 불량 die 수, S = 유효 die 수, N×S = 결합 층.
판정은 구간·순서 관계로만 한다 — 합격선 없음 (§3-13).
"""
import sys
import zlib

import numpy as np
from scipy.stats import rankdata

import config

SEED = 20260918
B_BOOT = 1000
# 층은 **정확값**이다. N·S가 정수 die 수라 가능하고, 층 안에서 교란이 상수라 잔여 교란이 0이다.
# 1차 설계(10분위·5×5)는 demo에서 걸렸다 — 교란의 순수 함수인 특징이 0.500 중 0.461을 남겼다
# (분위 안에서도 교란이 클래스를 가른다. 층 4의 교란 자기 AUC 0.966). docs §2-A.
UNDEF_MAX = 0.05  # 추정 가능성 가드 (효과의 합격선이 아니다)

FEATS = ("cov", "ctr", "cv", "rc", "mp", "cc")
DESIGN = {"cov": ("Edge-Loc", "Edge-Ring"), "ctr": ("Edge-Loc", "Edge-Ring"),
          "cv": ("Edge-Loc", "Edge-Ring"), "rc": ("Center", "Loc"),
          "mp": ("Donut", "Loc"), "cc": ("Loc", "Scratch")}
CONF = ("N", "S", "NxS")


# ── 통계 ─────────────────────────────────────────────────────────
def auc(x, pos):
    """평균 순위 AUC. 한 클래스가 없으면 NaN."""
    n1 = int(pos.sum()); n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(x)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def layer_ids(conf, key):
    """정확값 층. NxS는 (N, S) 쌍이 같은 웨이퍼끼리."""
    if key == "NxS":
        v = np.stack([conf["N"], conf["S"]], axis=1)
        return np.unique(v, axis=0, return_inverse=True)[1].ravel()
    return np.unique(conf[key], return_inverse=True)[1]


def strat_auc(x, pos, g):
    """층화 AUC와 비교 쌍 보존 비율 κ. Σw=0이면 (NaN, 0)."""
    num = den = 0.0
    for s in np.unique(g):
        m = g == s
        n1 = int(pos[m].sum()); n0 = int(m.sum()) - n1
        if n1 == 0 or n0 == 0:
            continue
        w = n1 * n0
        num += w * auc(x[m], pos[m]); den += w
    tot = pos.sum() * (len(pos) - pos.sum())
    return (num / den if den > 0 else np.nan), den / tot


def rng_for(*keys):
    h = zlib.crc32("|".join(map(str, keys)).encode())
    return np.random.default_rng([SEED, h])


# ── 자료 ─────────────────────────────────────────────────────────
def load():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        y, lot, idx = z["cls"].astype(str), z["lot"].astype(str), z["idx_in_cls"]
        F, seeds = z["folds"], z["seeds"]
    d = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz") as z:
        d["cov"], d["ctr"], d["cv"], S = z["cov"], z["ctr"], z["cv"], z["size"]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz") as z:
        d["rc"], d["mp"] = z["radial_contrast"], z["mid_peak"]
    with np.load(config.DATA_PROCESSED / "shape2.npz") as z:   # 제출본 경로 (tie_nan 아님)
        d["cc"] = z["cc_compact"]
    N = np.full(len(y), np.nan)
    for c in config.PATTERN_CLASSES:
        with np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True) as z:
            maps = z["wafer_maps"]
        rows = np.where(y == c)[0]
        for r in rows:
            N[r] = float((np.asarray(maps[idx[r]]) == config.VAL_FAIL).sum())
    # 행 정렬 검증: S는 feats8에서, 같은 맵에서 다시 세어 대조한다
    chk = np.random.default_rng(0).choice(len(y), 300, replace=False)
    for r in chk:
        with np.load(config.DATA_PROCESSED / f"{y[r]}.npz", allow_pickle=True) as z:
            a = np.asarray(z["wafer_maps"][idx[r]])
        assert float((a != config.VAL_OUTSIDE).sum()) == S[r], "행 정렬이 어긋났다"
    return y, lot, F, seeds, d, {"N": N, "S": S}


# ── M2 셀 하나 ───────────────────────────────────────────────────
def cell(x, y, lot, conf, pair, key, feat):
    a, b = pair
    m = ((y == a) | (y == b)) & np.isfinite(x)
    xs, pos, ls = x[m], (y[m] == b), lot[m]
    cs = {k: v[m] for k, v in conf.items()}
    g = layer_ids(cs, key)
    d_raw = auc(xs, pos) - 0.5
    sa, kappa = strat_auc(xs, pos, g)
    d_conf = {k: auc(cs[k], pos) - 0.5 for k in ("N", "S")}
    # lot 블록 부트스트랩
    ul, inv = np.unique(ls, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(ul))]
    rng = rng_for(feat, a, b, key)
    bs = np.full((B_BOOT, 3), np.nan)          # d_raw, d_strat, rho
    bc = np.full((B_BOOT, 2), np.nan)          # d of N, S
    for t in range(B_BOOT):
        pick = rng.integers(0, len(ul), len(ul))
        ii = np.concatenate([members[p] for p in pick])
        dr = auc(xs[ii], pos[ii]) - 0.5
        sa_b, _ = strat_auc(xs[ii], pos[ii], layer_ids({k: v[ii] for k, v in cs.items()}, key))
        ds = sa_b - 0.5
        bs[t] = dr, ds, (ds / dr if abs(dr) > 1e-12 else np.nan)
        bc[t] = [auc(cs[k][ii], pos[ii]) - 0.5 for k in ("N", "S")]
    return dict(n=int(m.sum()), d_raw=d_raw, d_strat=sa - 0.5, kappa=kappa,
                d_conf=d_conf, boot=bs, boot_conf=bc)


def ci(v):
    v = v[np.isfinite(v)]
    return (np.nan, np.nan) if len(v) == 0 else tuple(np.percentile(v, [2.5, 97.5]))


def judge(r):
    """사전 등록 §3의 규칙 그대로. 반환: (판정, 교란 부재 여부)."""
    undef = np.isnan(r["boot"][:, 1]).mean()
    if undef > UNDEF_MAX or not np.isfinite(r["d_strat"]):
        return "⬜ 판정 불가", False
    lo, hi = ci(r["boot"][:, 1])
    rlo, rhi = ci(r["boot"][:, 2])
    sgn = np.sign(r["d_raw"])
    # 교란 부재: 그 쌍에서 N·S 자기 판별력 구간이 둘 다 0을 포함
    absent = all(ci(r["boot_conf"][:, j])[0] <= 0 <= ci(r["boot_conf"][:, j])[1] for j in (0, 1))
    if lo <= 0 <= hi:
        v = "❌ 소멸"
    elif np.sign(lo) != sgn:
        v = "🔄 역전"
    elif rlo <= 1 <= rhi or rlo > 1:
        v = "✅ 유지"
    else:
        v = "🟡 부분"
    return v, absent


# ── demo: 합성 실패 조건 (§3-3) ─────────────────────────────────
def demo():
    rng = np.random.default_rng(1)
    n = 4000
    y = np.where(rng.random(n) < 0.5, "A", "B")
    pos = y == "B"
    lot = np.array([f"L{i // 4}" for i in range(n)])
    Nc = rng.poisson(np.where(pos, 60, 50)).astype(float)       # 교란: 클래스와 연관, 겹친다
    # (1차 demo는 40 vs 80이라 교란이 클래스를 거의 완전히 갈랐다 — 나쁜 시험이었다)
    S = rng.normal(1000, 100, n)
    conf = {"N": Nc, "S": S}
    # ① 교란의 순수 함수인 특징 → 층화하면 사라져야 한다
    x1 = Nc + rng.normal(0, 1e-3, n)
    # ② 교란과 독립이고 클래스 신호가 있는 특징 → 유지돼야 한다
    x2 = rng.normal(np.where(pos, 1.0, 0.0), 1.0)
    global B_BOOT
    keep, B_BOOT = B_BOOT, 200
    try:
        r1 = cell(x1, y, lot, conf, ("A", "B"), "N", "demo1")
        r2 = cell(x2, y, lot, conf, ("A", "B"), "N", "demo2")
        j1, j2 = judge(r1)[0], judge(r2)[0]
        print(f"  ① 교란 함수:  d_raw {r1['d_raw']:+.3f}  d_strat {r1['d_strat']:+.3f}  → {j1}")
        print(f"  ② 독립 신호:  d_raw {r2['d_raw']:+.3f}  d_strat {r2['d_strat']:+.3f}  → {j2}")
        assert r1["d_raw"] > 0.15 and abs(r1["d_strat"]) < 0.02, "교란 함수가 층화에서 안 사라졌다"
        # 층화가 교란 자체를 지웠는가 — 정확값 층이면 교란의 층화 d는 정확히 0이다
        g1 = layer_ids(conf, "N")
        assert abs(strat_auc(Nc, pos, g1)[0] - 0.5) < 1e-12, "층 안에 교란이 남아 있다"
        assert not j1.startswith("✅"), "교란 함수가 「유지」로 판정됐다"
        assert j2.startswith("✅"), "독립 신호가 「유지」가 아니다"
        # ③ 교란이 클래스를 완전히 가르면 → 판정 불가
        Nsep = np.where(pos, 100.0, 10.0)
        r3 = cell(x2, y, lot, {"N": Nsep, "S": S}, ("A", "B"), "N", "demo3")
        assert judge(r3)[0].startswith("⬜"), "완전 분리 교란이 판정 불가로 안 간다"
        print(f"  ③ 완전 분리:  κ {r3['kappa']:.3f} → {judge(r3)[0]}")
        # ④ 동점만 있는 특징 → AUC 0.5 (argsort 순위 버그 회귀)
        assert abs(auc(np.ones(n), pos) - 0.5) < 1e-12
        # ⑤ 층 안에서 라벨을 섞으면 층화 AUC는 0.5 근처 (귀무가 신호를 실제로 깬다)
        g = layer_ids(conf, "N")
        yp = pos.copy()
        for s in np.unique(g):
            m = np.where(g == s)[0]; yp[m] = rng.permutation(yp[m])
        sa, _ = strat_auc(x2, yp, g)
        assert abs(sa - 0.5) < 0.03, f"층 내 순열인데 층화 AUC {sa:.3f}"
        # ⑥ 결합 층 번호가 서로 다른 (a,b)를 다른 층으로 보낸다
        gg = layer_ids({"N": np.array([1., 1, 2, 2]), "S": np.array([5., 6, 5, 5])}, "NxS")
        assert list(gg) == [0, 1, 2, 2], gg
        # ⑦ 역전: 층 안에서 방향이 뒤집히는 구성 (Simpson)
        x7 = -0.5 * pos + Nc / 10 + rng.normal(0, 0.1, n)
        r7 = cell(x7, y, lot, conf, ("A", "B"), "N", "demo7")
        print(f"  ⑦ 심슨 구성: d_raw {r7['d_raw']:+.3f}  d_strat {r7['d_strat']:+.3f}  → {judge(r7)[0]}")
        assert judge(r7)[0].startswith("🔄"), "층 안 역전을 못 잡는다"
    finally:
        B_BOOT = keep
    print("demo OK")


def count_only():
    y, lot, F, seeds, d, conf = load()
    print(f"{'클래스':<11}{'장수':>7}{'N 중앙':>9}{'S 중앙':>9}")
    for c in config.PATTERN_CLASSES:
        m = y == c
        print(f"{c:<11}{m.sum():>7}{np.median(conf['N'][m]):>9.0f}{np.median(conf['S'][m]):>9.0f}")


# ── 본 실행 ──────────────────────────────────────────────────────
def main():
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    print(f"sklearn {sklearn.__version__} · numpy {np.__version__} · SEED {SEED} · B {B_BOOT}")
    demo()
    y, lot, F, seeds, d, conf = load()
    L = config.PATTERN_CLASSES

    print("\n[M1] 교란 변수만의 모델 (N, S) — 기술, 합격선 없음")
    X = np.column_stack([conf["N"], conf["S"]])
    f1s = []
    for s in range(len(seeds)):
        p = np.empty(len(y), dtype=object)
        for f in np.unique(F[s]):
            te = F[s] == f
            p[te] = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                           random_state=0, class_weight="balanced"
                                           ).fit(X[~te], y[~te]).predict(X[te])
        f1s.append(f1_score(y, p.astype(str), labels=L, average=None, zero_division=0))
    f1s = np.array(f1s)
    for i, c in enumerate(L):
        print(f"  {c:<11} F1 {f1s[:, i].mean():.3f}")
    print(f"  macro {f1s.mean(axis=1).mean():.3f} ± {f1s.mean(axis=1).std():.3f}"
          f"   (참고: 가장자리 3종 0.582 · 6종 0.837)")

    print("\n[M2] 설계 쌍 × 교란 — 사전 등록 판정")
    print(f"  {'특징':<5}{'쌍':<22}{'교란':<5}{'d_raw':>8}{'d_strat [95%]':>24}"
          f"{'ρ [95%]':>22}{'κ':>7}  판정")
    verdicts = {}
    for ft in FEATS:
        for key in CONF:
            r = cell(d[ft], y, lot, conf, DESIGN[ft], key, ft)
            v, absent = judge(r)
            verdicts[(ft, key)] = (v, absent)
            lo, hi = ci(r["boot"][:, 1]); rlo, rhi = ci(r["boot"][:, 2])
            rho = r["d_strat"] / r["d_raw"] if abs(r["d_raw"]) > 1e-12 else np.nan
            tag = " (교란 부재)" if absent and v.startswith("✅") else ""
            print(f"  {ft:<5}{DESIGN[ft][0]+'↔'+DESIGN[ft][1]:<22}{key:<5}{r['d_raw']:>+8.3f}"
                  f"{r['d_strat']:>+9.3f} [{lo:+.3f},{hi:+.3f}]{rho:>7.2f} [{rlo:.2f},{rhi:.2f}]"
                  f"{r['kappa']:>7.3f}  {v}{tag}")
        dc = cell(d[ft], y, lot, conf, DESIGN[ft], "N", ft)["d_conf"] if ft in ("cov", "rc", "mp", "cc") else None
        if dc:
            print(f"        └ 그 쌍에서 교란 자기 판별력: d(N) {dc['N']:+.3f} · d(S) {dc['S']:+.3f}")

    print("\n[판정] 가설 — 수치 조건과 최종 상태를 따로 적는다")
    ok = lambda f: all(verdicts[(f, k)][0].startswith("✅") for k in CONF)
    free = lambda f: any(verdicts[(f, k)][1] for k in CONF)
    print(f"  H1 rc·mp 셋 다 ✅: rc {ok('rc')} · mp {ok('mp')}"
          f"{'  ⚠ 교란 부재 셀 있음 — 강하다의 근거로 안 쓴다' if free('rc') or free('mp') else ''}")
    print(f"  H2 cov S에서 🟡 이하: {verdicts[('cov', 'S')][0]}")
    print(f"  H3 cv N에서 🟡: {verdicts[('cv', 'N')][0]}")
    print(f"  H4 cc (판단 걸지 않음): N {verdicts[('cc', 'N')][0]}")

    print("\n[보조] 28쌍 전체 — 층화 N×S 뒤 d_strat (점추정만, 판정 아님)")
    pairs = [(L[i], L[j]) for i in range(len(L)) for j in range(i + 1, len(L))]
    print("  " + "".join(f"{f:>8}" for f in FEATS))
    for a, b in pairs:
        row = []
        for ft in FEATS:
            x = d[ft]; m = ((y == a) | (y == b)) & np.isfinite(x)
            pos = y[m] == b
            g = layer_ids({k: v[m] for k, v in conf.items()}, "NxS")
            raw = auc(x[m], pos) - 0.5
            sa, _ = strat_auc(x[m], pos, g)
            row.append(f"{raw:+.2f}/{sa - 0.5:+.2f}")
        print(f"  {a[:6]+'↔'+b[:6]:<14}" + " ".join(f"{c:>11}" for c in row))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    elif "--count" in sys.argv:
        count_only()
    else:
        main()
