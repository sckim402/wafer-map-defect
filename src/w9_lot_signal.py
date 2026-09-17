# -*- coding: utf-8 -*-
"""11월 목록 #2 — 로트 단위 분석 + 로드맵 §6 「후보를 좁히는 신호」 열 검증.

⛔ **예측과 반증 조건은 `docs/w9_lot_signal.md` §1~§3에 있고 이 스크립트보다 먼저 커밋됐다.**

세 검사 — 셋 다 귀무는 **순열로 만든다** (§3-13. 직관으로 정한 합격선을 두지 않는다):
  ① lot 내 재현성   : 라벨만 섞는다 (lot 소속 고정)      → lot 크기·클래스 주변분포 보존
  ② waferIndex 상관 : lot 안에서 슬롯을 섞는다           → 클래스 구성·점유 슬롯 보존
  ③ 특징 lot 일관성 : 클래스 안에서 웨이퍼를 lot에 재배치 → lot별 장수 보존

⛔ 난수는 검사·클래스마다 분리한다 (D-034 — 하나를 돌려 쓰면 표본이 앞 소비량에 달린다).

돌리기: ./.venv/Scripts/python.exe -u src/w9_lot_signal.py [B_PERM] [B_BOOT]
"""
import sys
import numpy as np

sys.path.insert(0, "src")
import config

# 🔴 11차 — 플래그를 먼저 걷어내지 않아 `--power` 단독 인수가 int()에서 터졌다 (O-03).
_nums = [a for a in sys.argv[1:] if not a.startswith("--")]
B_PERM = int(_nums[0]) if len(_nums) > 0 else 999
B_BOOT = int(_nums[1]) if len(_nums) > 1 else 2000
SEED = 20260916
N_SLOT = 25
FEATS = ("cov", "ctr", "cv", "rc", "mp", "cc")
CLS9 = config.ALL_CLASSES


def rng_for(part, k=0):
    """검사·클래스마다 독립 난수 (D-034)."""
    return np.random.default_rng([SEED, part, k])


# ────────────────────────────── 공통 ──────────────────────────────
def perm_within(group, rng):
    """그룹(lot) 안에서만 섞는 치환 인덱스를 돌려준다.

    `group + U[0,1)`을 argsort하면 그룹 순서는 유지되고 그룹 안만 무작위가 된다
    (group이 정수라 소수부가 경계를 못 넘는다).
    """
    base = np.argsort(group, kind="stable")
    shuf = np.argsort(group + rng.random(len(group)))
    out = np.empty(len(group), dtype=np.int64)
    out[base] = shuf
    return out


def lot_boot_ratio(num_per_lot, den_per_lot, B, rng):
    """lot 블록 부트스트랩 → 비(比) 통계량의 95% 백분위 구간 (D-050과 같은 기계)."""
    nl = len(num_per_lot)
    W = rng.multinomial(nl, np.full(nl, 1.0 / nl), size=B).astype(np.float64)
    d = W @ den_per_lot
    r = np.where(d > 0, (W @ num_per_lot) / np.maximum(d, 1e-300), np.nan)
    return tuple(np.nanpercentile(r, [2.5, 97.5]))


def pval(obs, null, rtol=1e-9):
    """단측(큰 쪽) 순열 p — +1 보정. 0을 내지 않는다.

    🔴 **10차 외부 검토(H01·H02)가 두 가지를 잡았다. 둘 다 여기서 고친다.**

    ⓐ **수학적 동점을 부동소수 우열로 세면 안 된다.** 층=그룹처럼 귀무가 관측과
       **정확히 같아야 하는** 퇴화 조건에서 합산 순서의 반올림(2e-16)이 우열로 집계돼
       **`p=0.034`라는 거짓 유의**가 나왔다. 상대 허용오차 안이면 **동점으로 센다**.
    ⓑ **정의되지 않은 통계량에 `p`를 내면 안 된다.** `obs`가 NaN이면
       `NaN >= NaN`이 전부 False라 **최소 `p`가 나왔다.** 이제 **NaN을 낸다** —
       「검정 불가」는 「가장 유의」의 정반대다.

    ⓒ 🔴 **11차 외부 검토 — ⓑ를 고친 자리 옆에 같은 결함이 남아 있었다.**
       귀무가 **일부만** NaN이면 `nanmax`로 통과한 뒤 **분모에는 `len(null)`을 그대로** 썼다.
       `pval(1, [1, NaN×998])`이 **0.002**를 냈다 — 유효 귀무가 1개(그것도 동점)인데
       **998개를 「관측보다 작았다」로 세는 것과 같다.** 분모도 **유효한 것만** 센다.
    """
    null = np.asarray(null, dtype=float)
    ok = np.isfinite(null)
    n_ok = int(ok.sum())
    if not np.isfinite(obs) or n_ok == 0:
        return np.nan
    null = null[ok]
    tol = rtol * max(abs(obs), float(np.max(np.abs(null))), 1e-300)
    return (1 + int(np.sum(null >= obs - tol))) / (1 + n_ok)


# ───────────────────────── ① lot 내 재현성 ─────────────────────────
def repeat_terms(lot, cls, n_lot, n_cls):
    """클래스별 (분자, 분모)를 lot별로 낸다. R_c = Σnum / Σden."""
    cnt = np.bincount(lot * n_cls + cls, minlength=n_lot * n_cls).reshape(n_lot, n_cls)
    tot = cnt.sum(axis=1, keepdims=True)
    return cnt * (cnt - 1.0), cnt * (tot - 1.0)


def part1(lot, cls, shape=None, out=True, tag=""):
    """`shape`를 주면 **같은 맵 shape 안에서만** 라벨을 섞는다 (사후 대조)."""
    n_lot, n_cls = lot.max() + 1, len(CLS9)
    num, den = repeat_terms(lot, cls, n_lot, n_cls)
    R = num.sum(axis=0) / np.maximum(den.sum(axis=0), 1e-300)

    rng = rng_for(1 if shape is None else 11)
    null = np.empty((B_PERM, n_cls))
    for b in range(B_PERM):
        p = rng.permutation(len(cls)) if shape is None else perm_within(shape, rng)
        n2, d2 = repeat_terms(lot, cls[p], n_lot, n_cls)
        null[b] = n2.sum(axis=0) / np.maximum(d2.sum(axis=0), 1e-300)
        if out and (b + 1) % 200 == 0:
            print(f"      {tag}순열 {b+1}/{B_PERM}", flush=True)

    rows = []
    for i, c in enumerate(CLS9):
        lo, hi = lot_boot_ratio(num[:, i], den[:, i], B_BOOT, rng_for(1, i + 1))
        rows.append((c, R[i], null[:, i].mean(), R[i] / max(null[:, i].mean(), 1e-300),
                     pval(R[i], null[:, i]), lo, hi))
    return rows


# ──────────────────────── ② waferIndex 상관 ────────────────────────
def chi2_by_class(cls, slot, E, n_cls):
    O = np.bincount(cls * N_SLOT + slot, minlength=n_cls * N_SLOT).reshape(n_cls, N_SLOT)
    return (((O - E) ** 2) / np.maximum(E, 1e-300)).sum(axis=1), O


def part2(lot, cls, slot, out=True):
    n_lot, n_cls = lot.max() + 1, len(CLS9)
    # 해석적 기대도수: lot의 점유 슬롯 위에 그 lot의 클래스 장수를 고르게 편다
    occ = np.zeros((n_lot, N_SLOT))
    occ[lot, slot] = 1.0                                    # 점유 슬롯 집합
    cnt = np.bincount(lot * n_cls + cls, minlength=n_lot * n_cls).reshape(n_lot, n_cls)
    E = (cnt / np.maximum(occ.sum(axis=1, keepdims=True), 1)).T @ occ

    T, O = chi2_by_class(cls, slot, E, n_cls)

    # 유효 표본 — 클래스가 섞인 lot 안의 웨이퍼만 순열이 움직인다 (§2-② 검출력 경고)
    mixed = (cnt > 0).sum(axis=1) > 1
    n_eff = np.array([cnt[mixed, i].sum() for i in range(n_cls)])

    rng = rng_for(2)
    null = np.empty((B_PERM, n_cls))
    for b in range(B_PERM):
        null[b] = chi2_by_class(cls, slot[perm_within(lot, rng)], E, n_cls)[0]
        if out and (b + 1) % 200 == 0:
            print(f"      순열 {b+1}/{B_PERM}", flush=True)

    slots = np.arange(1, N_SLOT + 1)
    rows = []
    for i, c in enumerate(CLS9):
        n = O[i].sum()
        rows.append((c, n, int(n_eff[i]), T[i], null[:, i].mean(), pval(T[i], null[:, i]),
                     (O[i] @ slots) / max(n, 1), (E[i] @ slots) / max(E[i].sum(), 1e-300)))
    return rows


def part2_power(lot, cls, slot, target="Loc", lams=(0.2, 0.4, 0.6, 0.8, 1.2), n_sim=300):
    """②가 **비유의일 때 얼마나 큰 편중까지 놓치는가** (사후. D-036의 교훈).

    ⛔ **「비기각 = 슬롯 무관」이 아니다** — 방위 검정에서 이미 걸린 자리다.
    lot 안에서 슬롯 가중 `w(s) ∝ exp(λ(s−13)/12)`로 라벨을 다시 뽑아
    **본검정과 같은 규칙(순열 `p ≤ 0.05`)으로 기각되는 비율**을 센다 (11차 — 전에는 귀무 95백분위였다).
    귀무 분포는 본 검사의 것을 그대로 쓴다. `tcrit`(95백분위)은 **참고값으로만** 돌려준다.
    """
    n_lot, n_cls = lot.max() + 1, len(CLS9)
    ti = CLS9.index(target)
    occ = np.zeros((n_lot, N_SLOT))
    occ[lot, slot] = 1.0
    cnt = np.bincount(lot * n_cls + cls, minlength=n_lot * n_cls).reshape(n_lot, n_cls)
    E = (cnt / np.maximum(occ.sum(axis=1, keepdims=True), 1)).T @ occ

    rng = rng_for(2)
    null = np.array([chi2_by_class(cls, slot[perm_within(lot, rng)], E, n_cls)[0][ti]
                     for _ in range(B_PERM)])
    tcrit = np.percentile(null, 95)

    is_t = cls == ti
    # 🔴 기준은 **순열 기대**이지 관측 평균이 아니다 (§3-11 — 기준선과 함께 읽는다).
    #    관측 평균을 기준으로 삼으면 λ=0의 모의가 0이 아닌 값을 내고 이동량이 통째로 밀린다.
    slots = np.arange(1, N_SLOT + 1)
    base_mean = float(E[ti] @ slots / E[ti].sum())
    obs_dev = float(slot[is_t].mean() + 1 - base_mean)
    g = np.where(((cnt > 0).sum(axis=1) > 1)[lot])[0]       # 순열이 움직일 수 있는 웨이퍼
    lot_m, slot_m = lot[g], slot[g]
    nm = np.bincount(lot_m, minlength=n_lot)
    start = (nm.cumsum() - nm)[lot_m]                        # lot 정렬시 그 lot의 시작 위치
    k_of = np.bincount(lot_m, weights=is_t[g].astype(float), minlength=n_lot)[lot_m]
    Et = E[ti]
    r2 = rng_for(22)
    rows = []
    for lam in lams:
        logw = lam * (slot_m + 1 - 13) / 12.0
        hit, means = 0, []
        for _ in range(n_sim):
            key = logw - np.log(-np.log(r2.random(len(g))))  # Gumbel top-k = 가중 비복원 추출
            order = np.lexsort((-key, lot_m))
            rank = np.empty(len(g), dtype=np.int64)
            rank[order] = np.arange(len(g))
            new_t = is_t.copy()
            new_t[g] = (rank - start) < k_of                 # lot마다 원래 장수만큼 고른다
            O = np.bincount(slot[new_t], minlength=N_SLOT).astype(float)
            # 🔴 11차 — 전에는 `> tcrit`(귀무 95백분위)였는데 **본검정은 `pval <= 0.05`**다.
            #    두 규칙은 경계에서 갈린다(귀무 0..998·관측 948.5에서 백분위는 기각,
            #    본검정 p는 0.051). **검출력은 「그 검정의」 검출력이라야 한다.**
            hit += pval((((O - Et) ** 2) / np.maximum(Et, 1e-300)).sum(), null) <= 0.05
            means.append((slot[new_t] + 1).mean())
        rows.append((lam, hit / n_sim, float(np.mean(means)) - base_mean))
    return tcrit, base_mean, obs_dev, rows


# ─────────────────── ③ 특징값의 lot 내 일관성 (ICC) ───────────────────
def icc(x, g):
    """자유도 보정한 `1 − SS_within/SS_total`. g는 0..G-1로 압축된 lot 번호.

    🔴 **보정 없는 판은 귀무값이 0이 아니다** — 무작위여도 `E[1−SS_w/SS_t] = (G−1)/(N−1)`이고,
    `N/G`가 클래스마다 크게 다르다(`Edge-Ring` 9.0장/lot vs `Scratch` 1.1장/lot).
    그대로 쓰면 **응집이 아니라 lot당 장수를 비교하게 된다** (§3-11 · §3-13).
    자유도로 나누면 귀무 기대가 0 근처로 오고, 그래도 **판정은 순열 귀무가 한다.**
    """
    n = np.bincount(g).astype(float)
    m = np.bincount(g, weights=x) / n
    ss_w = float(((x - m[g]) ** 2).sum())
    ss_t = float(((x - x.mean()) ** 2).sum())
    n_tot, n_grp = len(x), len(n)
    # 🔴 11차 — `ss_t <= 0`은 상수 입력을 못 막는다. `np.full(24, 0.1)`은 원소가 전부
    #    같은데 평균 반올림으로 양의 미소 `ss_t`(~2e-33)가 생겨 **icc가 1.0을 냈다.**
    #    `demo()`는 `ones`만 시험했고 그건 우연히 정확히 0이 나오는 한 점이다.
    #    「값이 하나뿐인 자료」는 **검정 불가**이지 「완전 일치」가 아니다.
    scale = float(np.max(np.abs(x))) if n_tot else 0.0
    if not np.isfinite(ss_t) or n_tot <= n_grp or ss_t <= 1e-24 * n_tot * max(scale, 1.0) ** 2:
        return np.nan
    return 1.0 - (ss_w / (n_tot - n_grp)) / (ss_t / (n_tot - 1))


def part3_one(x, lot_c, rng, strata=None):
    """lot에 2장 이상 있는 것만 쓴다 — 단장 lot은 SS_within에 0을 넣어 ICC를 부풀린다.

    귀무는 **웨이퍼를 lot에 재배치**한다 (lot별 장수 보존).
    ⛔ **lot 「안에서」 섞으면 안 된다** — 각 lot의 값 집합이 그대로라 ICC가 안 변하고
       귀무가 관측값과 같아진다. 2026-09-16에 실제로 그렇게 짰고 p가 전부 1 근처로 나왔다.
    `strata`를 주면 **그 층 안에서만** 재배치한다 (사후 대조: 맵 shape 고정).
    """
    ok = np.isfinite(x)
    x, lot_c = x[ok], lot_c[ok]
    st = None if strata is None else np.asarray(strata)[ok]
    u, g = np.unique(lot_c, return_inverse=True)
    keep = np.bincount(g)[g] >= 2
    x, g = x[keep], np.unique(g[keep], return_inverse=True)[1]
    if st is not None:
        st = np.unique(st[keep], return_inverse=True)[1]
    if len(x) < 10 or g.max() < 1:
        return np.nan, np.nan, np.nan, len(x), int(ok.sum()), True
    obs = icc(x, g)
    if st is None:
        null = np.array([icc(x[rng.permutation(len(x))], g) for _ in range(B_PERM)])
    else:
        null = np.array([icc(x[perm_within(st, rng)], g) for _ in range(B_PERM)])
    # 퇴화 귀무 — 섞을 것이 없어 귀무 분포가 한 점이면 **검정 자체가 성립하지 않는다**
    # (층이 lot을 사실상 식별하면 일어난다. 10차 H01)
    spread = float(np.nanstd(null)) if np.isfinite(null).any() else np.nan
    scale = max(abs(obs), float(np.nanmax(np.abs(null))) if np.isfinite(null).any() else 0,
                1e-300)
    degen = not np.isfinite(spread) or spread <= 1e-9 * scale
    return obs, null.mean(), pval(obs, null), len(x), int(ok.sum()), degen


def residualize(x, size):
    """클래스 안에서 log(size)에 대한 선형 잔차 — 웨이퍼 크기 교란 통제 (§3-2)."""
    ok = np.isfinite(x) & np.isfinite(size) & (size > 0)
    out = np.full_like(x, np.nan)
    if ok.sum() >= 3:
        t = np.log(size[ok])
        b = np.polyfit(t, x[ok], 1)
        out[ok] = x[ok] - np.polyval(b, t)
    return out


def load_feats():
    d = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        for k in ("cov", "ctr", "cv"):
            d[k] = z[k]
        d["size"] = z["size"]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        d["rc"], d["mp"] = z["radial_contrast"], z["mid_peak"]
    with np.load(config.DATA_PROCESSED / "shape2_tienan.npz", allow_pickle=True) as z:
        d["cc"] = z["cc_compact"]          # 채택판(동점→NaN, D-046)
    return d


def shape_ids():
    """웨이퍼별 맵 shape (클래스별 npz 순서).

    ⛔ **shape은 제품 ID가 아니다** (D-053) — 같은 shape의 다른 제품, 한 제품의 여러 shape,
    shape과 라벨 선별을 함께 정하는 수집 묶음과 **구분되지 않는다.**
    """
    with np.load(config.DATA_PROCESSED / "map_shapes.npz", allow_pickle=True) as z:
        return {c: z[c][:, 0].astype(np.int64) * 1000 + z[c][:, 1] for c in CLS9}


def part3(out=True):
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        y, lot, idx = z["cls"].astype(str), z["lot"].astype(str), z["idx_in_cls"]
    d, sh = load_feats(), shape_ids()
    rows = []
    for ci, c in enumerate(config.PATTERN_CLASSES):
        sel = y == c
        lot_c, size_c = lot[sel], d["size"][sel]
        sh_c = sh[c][idx[sel]]
        r = {"size": part3_one(size_c.copy(), lot_c, rng_for(3, ci * 100))}
        for fi, f in enumerate(FEATS):
            x = d[f][sel]
            r[f] = part3_one(x.copy(), lot_c, rng_for(3, ci * 100 + fi + 1))
            r[f + "_res"] = part3_one(residualize(x, size_c), lot_c,
                                      rng_for(3, ci * 100 + fi + 30))
            r[f + "_shp"] = part3_one(x.copy(), lot_c, rng_for(3, ci * 100 + fi + 60),
                                      strata=sh_c)
        rows.append((c, int(sel.sum()), r))
        if out:
            print(f"      {c:<11} 완료 (맵 shape {len(np.unique(sh_c))}종)", flush=True)
    return rows


# ────────────────────────────── 자체 검사 ──────────────────────────────
def demo():
    """§3-3 — 실패 조건에서 먼저 돌린다. **새 성질과 옛 성질을 둘 다 넣는다 (D-051).**"""
    g = np.repeat(np.arange(50), 4)
    r = np.random.default_rng(0)

    # [새] 그룹 안 순열이 그룹 소속을 안 바꾼다
    for _ in range(20):
        p = perm_within(g, r)
        assert np.array_equal(g[p], g), "perm_within이 lot 경계를 넘었다"
        assert len(np.unique(p)) == len(g), "perm_within이 치환이 아니다"
    assert not np.array_equal(perm_within(g, r), np.arange(len(g))), "안 섞인다"

    # [새] ① 완전 응집 → R=1 · 완전 무작위 → 귀무 근처
    lot = np.repeat(np.arange(200), 5)
    cls_c = np.repeat(r.integers(0, 3, 200), 5)                  # lot마다 한 클래스
    num, den = repeat_terms(lot, cls_c, 200, 3)
    assert abs(num.sum(axis=0) / den.sum(axis=0) - 1).max() < 1e-12, "완전 응집인데 R≠1"
    cls_m = r.integers(0, 3, 1000)
    num, den = repeat_terms(lot, cls_m, 200, 3)
    R = num.sum(axis=0) / den.sum(axis=0)
    assert abs(R - 1 / 3).max() < 0.06, f"무작위인데 R이 주변분포에서 멀다: {R}"

    # [새·강] ② 해석적 기대도수가 순열 평균과 일치하는가 — 이게 ②의 판정 장치다
    n, nl = 600, 60
    lot2 = r.integers(0, nl, n)
    slot2 = np.concatenate([r.permutation(N_SLOT)[: (lot2 == l).sum()] for l in range(nl)])
    order = np.argsort(lot2, kind="stable")
    lot2, cls2 = lot2[order], r.integers(0, 3, n)
    cnt = np.bincount(lot2 * 3 + cls2, minlength=nl * 3).reshape(nl, 3)
    occ = np.zeros((nl, N_SLOT))
    occ[lot2, slot2] = 1.0
    E = (cnt / np.maximum(occ.sum(axis=1, keepdims=True), 1)).T @ occ
    emp = np.mean([np.bincount(cls2 * N_SLOT + slot2[perm_within(lot2, r)],
                               minlength=3 * N_SLOT).reshape(3, N_SLOT)
                   for _ in range(400)], axis=0)
    assert np.abs(emp - E).max() < 0.9, f"기대도수가 순열 평균과 어긋난다: {np.abs(emp-E).max():.2f}"

    # [새] ③ ICC — 같은 lot이면 같은 값 → 1 · 무작위 → 0 근처
    gg = np.repeat(np.arange(100), 5)
    assert abs(icc(np.repeat(r.normal(size=100), 5), gg) - 1) < 1e-9, "완전 일치인데 ICC≠1"
    assert abs(np.mean([icc(r.normal(size=500), gg) for _ in range(200)])) < 0.02, \
        "무작위인데 보정 ICC가 0에서 멀다"
    # [새] 보정이 실제로 필요한가 — 보정 전 판은 lot당 장수에 따라 귀무가 달라진다
    raw = lambda x, g: 1 - ((x - np.bincount(g, weights=x)[g] / np.bincount(g)[g]) ** 2
                            ).sum() / ((x - x.mean()) ** 2).sum()
    g_few, g_many = np.repeat(np.arange(250), 2), np.repeat(np.arange(50), 10)
    b_few = np.mean([raw(r.normal(size=500), g_few) for _ in range(100)])
    b_many = np.mean([raw(r.normal(size=500), g_many) for _ in range(100)])
    assert b_few - b_many > 0.3, "보정 전 판의 귀무 편차가 안 잡힌다 — 검사가 무력하다"
    assert abs(np.mean([icc(r.normal(size=500), g_few) for _ in range(100)])
               - np.mean([icc(r.normal(size=500), g_many) for _ in range(100)])) < 0.05, \
        "보정 후에도 lot당 장수에 따라 귀무가 달라진다"

    # [새·강] ③의 귀무가 **실제로 응집을 깨는가** — 2026-09-16에 여기서 걸렸다.
    #         lot 안에서 섞으면 각 lot의 값 집합이 그대로라 귀무 = 관측이 되고 p가 1로 붙는다.
    g5 = np.repeat(np.arange(60), 5)
    xc = np.repeat(r.normal(size=60), 5) + 0.01 * r.normal(size=300)   # 완전 응집에 가깝다
    o5, n5, p5, *_ = part3_one(xc, g5.astype(str), np.random.default_rng(4))
    assert o5 > 0.9, f"응집 자료인데 ICC가 낮다: {o5:.3f}"
    assert abs(n5) < 0.1, f"귀무가 응집을 안 깬다 — lot 안에서 섞고 있다 (귀무평균 {n5:.3f})"
    assert p5 <= 1.0 / (1 + B_PERM) + 1e-12, f"응집이 뚜렷한데 p가 최소값이 아니다: {p5:.3f}"
    # 층 고정 귀무: 층이 lot과 같으면 귀무가 관측과 같아져 **검출력이 0**이어야 한다
    _, n6, p6, _, _, dg6 = part3_one(xc, g5.astype(str), np.random.default_rng(4), strata=g5)
    assert n6 > 0.9 and dg6, "층=lot이면 귀무 = 관측이고 퇴화로 표시돼야 한다"
    assert p6 > 0.5, f"수학적 동점인데 p가 작다 — 반올림을 우열로 센다 (10차 H01): {p6}"
    # [10차 H01] 반올림이 우열로 세지는 수치 영역에서도 동점이어야 한다
    gh = np.repeat(np.arange(6), 10)
    xh = np.random.default_rng(8).normal(size=60)
    _, _, ph, _, _, dgh = part3_one(xh, gh.astype(str), np.random.default_rng(4), strata=gh)
    assert ph > 0.5 and dgh, f"퇴화 귀무에서 거짓 유의가 난다: p={ph}"
    # [10차 H02] 유한하지만 분산이 0인 입력 -> 통계량 미정의 -> p는 NaN이어야 한다
    oc, _, pc, nu, nf, dgc = part3_one(np.ones(24), np.repeat(np.arange(6), 4).astype(str),
                                       np.random.default_rng(1))
    assert np.isnan(oc) and np.isnan(pc), f"상수 입력인데 p가 나온다: obs={oc} p={pc}"
    assert np.isnan(pval(np.nan, [np.nan] * 9)), "NaN 관측에 p를 낸다"

    # ── [11차] 같은 계열의 다섯 번째·여섯 번째. **고친 자리 옆을 센다** ──
    # ⓐ 귀무가 **일부만** NaN이면 분모도 유효한 것만이라야 한다.
    #    옛 판은 무효 귀무를 「관측보다 작았다」로 세어 p를 1/999까지 떨어뜨렸다.
    mixed = np.r_[10.0, np.full(998, np.nan)]
    assert abs(pval(100.0, mixed) - 0.5) < 1e-12, \
        f"무효 귀무가 분모에 남아 있다 (11차): {pval(100.0, mixed)}"
    assert np.isnan(pval(1.0, np.full(9, np.nan))), "유효 귀무 0개인데 p를 낸다"
    # ⓑ 「값이 하나뿐인 자료」는 **검정 불가**다. `ones`만 시험하면 우연히 통과한다 —
    #    소수 상수는 평균 반올림으로 양의 미소 SS_t가 생겨 옛 판이 ICC=1.0을 냈다.
    for const in (0.1, 3.7, -2.5, 1e6):
        assert np.isnan(icc(np.full(24, const), np.repeat(np.arange(6), 4))), \
            f"상수 {const}인데 ICC가 나온다 (11차)"
    assert np.isfinite(icc(r.normal(size=24), np.repeat(np.arange(6), 4))), \
        "상수 가드가 멀쩡한 자료까지 막는다"
    # ⓒ 판정이 **3상태**인가 — 유효 검정이 0개면 「통과」가 아니라 「판정 불가」다
    import contextlib as _c, io as _io
    dead = (np.nan, np.nan, np.nan, 0, 0, True)
    r3d = [(c, 0, {f + k: dead for f in FEATS for k in ("", "_res", "_shp")})
           for c in CLS9[:-1]]
    r1d = [(c, 0.0, 0.0, 0.0, 1.0, np.nan, np.nan) for c in CLS9]
    r2d = [(c, 0, 0, 0.0, 0.0, 1.0, 0.0, 0.0) for c in CLS9]
    buf = _io.StringIO()
    with _c.redirect_stdout(buf):
        judge(r1d, r1d, r2d, r3d)
    out = buf.getvalue()
    assert "OK 0 근처" not in out, "유효 검정 0개인데 H7이 통과로 인쇄된다 (11차)"
    assert "OK 슬롯 무관" not in out, "비유의를 「무관」으로 인쇄한다 (11차)"
    assert "OK 전멸 신호" not in out, "동시발생 p로 「전멸」을 판정한다 (11차)"
    assert out.count("⬜ 판정 불가") >= 4, f"3상태가 아니다 (11차):\n{out}"

    # [옛] NaN이 조용히 값으로 안 바뀐다 (D-030: 막을 것은 사례가 아니라 계열)
    x = r.normal(size=500)
    x[:7] = np.nan
    o, _, _, n_used, n_fin, _ = part3_one(x, gg.astype(str), np.random.default_rng(1))
    assert n_fin == 493 and np.isfinite(o), "NaN 처리가 깨졌다"
    assert np.isnan(part3_one(np.full(500, np.nan), gg.astype(str),
                              np.random.default_rng(1))[0]), "전부 NaN인데 값이 나온다"
    # [옛] 단장 lot을 섞으면 ICC가 부풀지 않는가 — 2장 미만 배제가 실제로 걸리는지
    x2 = np.r_[r.normal(size=100), r.normal(size=400)]
    g2 = np.r_[np.arange(100), np.repeat(np.arange(100, 180), 5)].astype(str)
    assert part3_one(x2, g2, np.random.default_rng(2))[3] == 400, "단장 lot이 안 걸러진다"

    # [옛] lot 블록 부트스트랩 — 군집이 없으면 폭이 이항과 비슷하다 (D-050과 같은 검사)
    num1 = (r.random(2000) < 0.3).astype(float)
    lo, hi = lot_boot_ratio(num1, np.ones(2000), 4000, np.random.default_rng(3))
    assert 0.02 < (hi - lo) < 0.05, f"군집 없는데 구간 폭이 이상하다: {hi-lo:.3f}"
    lo2, hi2 = lot_boot_ratio(num1.reshape(100, 20).sum(1),
                              np.full(100, 20.0), 4000, np.random.default_rng(3))
    assert (hi2 - lo2) < 1.6 * (hi - lo), "lot 안이 독립인데 폭이 튄다"
    print("demo ok — 순열 3종 · 기대도수↔순열평균 일치 · ICC 양끝 · NaN·단장lot·부트스트랩")


# ────────────────────────────── 본체 ──────────────────────────────
def load_lots():
    sh = shape_ids()
    lot_s, cls_s, slot_s, sh_s = [], [], [], []
    for i, c in enumerate(CLS9):
        with np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True) as z:
            ln, wi = z["lot_name"].astype(str), np.asarray(z["wafer_index"]).ravel()
        assert len(sh[c]) == len(ln), f"{c}: map_shapes와 장수가 다르다"
        lot_s.append(ln)
        cls_s.append(np.full(len(ln), i))
        slot_s.append(wi.astype(int) - 1)
        sh_s.append(sh[c])
    lot = np.concatenate(lot_s)
    cls = np.concatenate(cls_s)
    slot = np.concatenate(slot_s)
    shape = np.unique(np.concatenate(sh_s), return_inverse=True)[1]
    assert slot.min() == 0 and slot.max() == N_SLOT - 1, "waferIndex가 1~25를 벗어난다"
    u, lot_id = np.unique(lot, return_inverse=True)
    order = np.argsort(lot_id, kind="stable")          # ②의 perm_within은 lot 정렬을 쓴다
    return lot_id[order], cls[order], slot[order], shape[order], len(u)


def main():
    lot, cls, slot, shape, n_lot = load_lots()
    print("=" * 96)
    print(f"  로트 신호 3종 · 라벨 {len(cls):,}장 · lot {n_lot:,}개 · 맵 shape "
          f"{shape.max()+1}종 · 순열 B={B_PERM} · lot 재추출 {B_BOOT:,} · SEED={SEED}")
    print("=" * 96)

    print("\n-- (1) lot 내 재현성 R_c = 앞 원소가 c인 lot 내 순서쌍 중 뒤도 c인 비율 "
          "(순서쌍 균등 추출 가중 — 웨이퍼 평균이 아니다. 11차) --")
    r1 = part1(lot, cls)
    r1b = part1(lot, cls, shape=shape, tag="[shape고정] ")       # 사후 대조
    print(f"  {'클래스':<11}{'장수':>8}{'R_c':>9}{'귀무':>9}{'응집배수':>9}"
          f"{'순열 p':>9}{'lot블록 95%':>22}{'shape고정 귀무':>14}{'배수':>7}{'p':>8}")
    print("-" * 96)
    for (c, R, R0, mult, p, lo, hi), (_, _, R0b, multb, pb, _, _) in zip(r1, r1b):
        n = int((cls == CLS9.index(c)).sum())
        print(f"  {c:<11}{n:>8,}{R:>9.4f}{R0:>9.4f}{mult:>9.2f}{p:>9.4f}"
              f"{f'[{lo:.4f}, {hi:.4f}]':>22}{R0b:>14.4f}{multb:>7.2f}{pb:>8.4f}")

    print("\n-- (2) waferIndex(슬롯) 상관 · lot 안 순열 --")
    r2 = part2(lot, cls, slot)
    print(f"  {'클래스':<11}{'장수':>8}{'유효표본':>9}{'T(chi2)':>10}{'귀무평균':>10}"
          f"{'순열 p':>9}{'평균슬롯':>10}{'기대':>8}")
    print("-" * 96)
    for c, n, ne, T, T0, p, ms, es in r2:
        print(f"  {c:<11}{n:>8,}{ne:>9,}{T:>10.2f}{T0:>10.2f}{p:>9.4f}{ms:>10.2f}{es:>8.2f}")

    print("\n-- (3) 특징값의 lot 내 일관성 (ICC) --")
    r3 = part3()
    for key, name in (("", "A 원값 · 귀무=클래스 안 전체 재배치 (사전 등록)"),
                      ("_res", "A 잔차 · log size 회귀 잔차 (사전 등록)"),
                      ("_shp", "B 원값 · 귀무=같은 맵 shape 안에서만 재배치 (사후 대조)")):
        print(f"\n  [{name}]")
        print(f"  {'클래스':<11}{'size ICC':>9} | " + " ".join(f"{f:>12}" for f in FEATS))
        print("-" * 96)
        for c, n, r in r3:
            cells = []
            for f in FEATS:
                o, n0, p, nu, nf, dg = r[f + key]
                mark = "x" if (dg or not np.isfinite(p)) else ("*" if p <= 0.05 else " ")
                cells.append(f"{o:>6.3f}/{n0:>4.2f}{mark}"
                             if np.isfinite(o) else f"{'-- 검정불가':>12}")
            print(f"  {c:<11}{r['size'][0]:>9.3f} | " + " ".join(cells))
        print("   (칸 = ICC / 귀무평균 · * 는 순열 p <= 0.05 · **x 는 검정 불가**"
              " — 귀무가 한 점이거나 통계량 미정의. 10차 H01·H02)")

    judge(r1, r1b, r2, r3)


def judge(r1, r1b, r2, r3):
    """사전 등록 판정 — 경고가 아니라 판정에 연결한다 (§7 폐기 목록).

    🔴 **11차 외부 검토로 통째로 다시 썼다. 이유는 셋이다.**

    ⓐ **판정이 2상태였다.** 「통과/기각」뿐이라 **「검정 불가」가 갈 자리가 없었다.**
       유효 검정이 0개여도 `유의 0개 -> OK 0 근처`가 나왔다 — 10차가 `sig()`로
       **집계에서는** 퇴화를 뺐는데 **최종 한 줄에서는 다시 넣고 있었다.**
    ⓑ **등록문이 말한 대상과 다른 것을 쟀다.**
       `H3`은 *"lot 단위 전멸"*인데 코드는 **동시발생 `p`**로 판정했고,
       `H5`·`H7`은 *"무관"*·*"0 근처"*라는 **크기** 주장인데 **비유의**로 판정했다.
       ⛔ **비유의는 무관이 아니다** — §2-②에 내가 직접 *"이 수가 작으면 비유의여도
       「슬롯 무관」이 아니라 「못 잰다」"*라고 써 놓고 코드는 반대로 인쇄했다.
    ⓒ **`H1`이 `H2`의 거부 조건을 안 받았다.** 등록문이 *"「전부 응집했다」는 H1의
       성공이 아니다"*라고 못박았는데 `OK 응집`과 `① 판정 불가`가 나란히 인쇄됐다.

    → **수치 조건(`[수치]`)과 등록 주장의 최종 상태(`=>`)를 따로 인쇄한다.**
    """
    OK, NO, UNK = "OK", "X", "⬜ 판정 불가"
    g1 = {c: (mult, p) for c, R, R0, mult, p, lo, hi in r1}
    g1b = {c: (mult, p) for c, R, R0, mult, p, lo, hi in r1b}
    g2 = {c: (T, p, ne) for c, n, ne, T, T0, p, ms, es in r2}
    g3 = {c: r for c, n, r in r3}
    print("\n" + "=" * 96)
    print("=== 사전 등록 판정 (docs/w9_lot_signal.md §3) ===")

    h1n = all(g1[c][0] > 1 and g1[c][1] <= 0.05 for c in ("Center", "Edge-Ring"))
    rm, rp = g1["Random"]
    h2 = rp > 0.05
    print(f"  H1 Center {g1['Center'][0]:.2f}배 p={g1['Center'][1]:.4f} · "
          f"Edge-Ring {g1['Edge-Ring'][0]:.2f}배 p={g1['Edge-Ring'][1]:.4f}  "
          f"[수치] {OK if h1n else NO}")
    print(f"     => ①의 신호 판정: {OK if (h1n and h2) else UNK}"
          + ("" if h2 else " — H2가 깨졌다. 등록문: 「전부 응집했다」는 H1의 성공이 아니다"))
    print(f"  H2 음성대조 Random {rm:.2f}배 p={rp:.4f}  "
          f"[수치] {'이 검정에서 비유의' if h2 else NO}")
    print(f"     => {OK if h2 else '[!] Random도 응집 — ①은 클래스 특이성 판정 불가'}")
    nf = g1["Near-full"]
    print(f"  H3 Near-full 동시발생 {nf[0]:.2f}배 p={nf[1]:.4f}  "
          f"[수치] 응집 {OK if nf[1] <= 0.05 else NO}")
    print(f"     => 등록 예측(lot 단위 전멸): {UNK} — ⛔ **이 코드는 전멸을 재지 않는다.**"
          "\n        전멸 lot의 도수·분모·정의가 없고 동시발생 `R`로는 역산되지 않는다 (11차)")

    T, p, ne = g2["Loc"]
    print(f"  H4 Loc 슬롯 T={T:.1f} p={p:.4f} (유효표본 {ne:,})  "
          f"[수치] {'유의' if p <= 0.05 else '비유의'}")
    print(f"     => {OK + ' 슬롯 의존' if p <= 0.05 else UNK} — 비유의는 "
          "「무관」이 아니다. 등록문 §2-②의 「유효 표본이 작으면 못 잰다」는 "
          "기준이 미등록이라 ❌로 닫지 않는다 (11차)")
    h5 = all(g2[c][1] > 0.05 for c in ("Center", "Edge-Ring"))
    print(f"  H5 음성대조 Center p={g2['Center'][1]:.4f}(유효 {g2['Center'][2]:,}) · "
          f"Edge-Ring p={g2['Edge-Ring'][1]:.4f}(유효 {g2['Edge-Ring'][2]:,})  "
          f"[수치] {'둘 다 비유의' if h5 else '유의 있음'}")
    print(f"     => 등록 예측(슬롯 무관): "
          + (f"{UNK} — 반증 조건이 발동하지 않았을 뿐이고 **동등성 범위가 미등록**이다 (11차)"
             if h5 else "[!] 얘들도 유의 — Loc 신호로 못 쓴다"))

    def sig(cell):
        """유효한 검정만 유의로 센다 — 퇴화·미정의는 「검정 불가」다 (10차 H01·H02)."""
        o, n0, p, nu, nf, dg = cell
        return np.isfinite(p) and not dg and p <= 0.05

    def ntest(r, key=""):
        return sum(np.isfinite(r[f + key][2]) and not r[f + key][5] for f in FEATS)

    def valid(cell):
        return np.isfinite(cell[2]) and not cell[5]

    er = g3["Edge-Ring"]
    keys6 = [f + k for f in ("rc", "mp", "ctr") for k in ("", "_res")]
    n_ok6 = sum(valid(er[k]) for k in keys6)
    h6 = n_ok6 == 6 and all(sig(er[k]) for k in keys6)
    print("  H6 Edge-Ring " + " · ".join(
        f"{f} {er[f][0]:.3f}(p={er[f][2]:.3f})/잔차 {er[f+'_res'][0]:.3f}(p={er[f+'_res'][2]:.3f})"
        for f in ("rc", "mp", "ctr")) + f"  [수치] 유효 {n_ok6}/6")
    print(f"     => {OK if h6 else (UNK + ' — 유효 검정이 6개가 아니다 (11차)'
                                    if n_ok6 < 6 else NO + ' 잔차에서 죽음')}"
          "\n        ⚠ 보고 ICC는 **자유도 보정판**이고 등록 식 `1−SSw/SSt`가 아니다 (11차)")
    ra = g3["Random"]
    nsig, nt = sum(sig(ra[f]) for f in FEATS), ntest(ra)
    print(f"  H7 음성대조 Random 유의 {nsig}/{nt}개 (잔차 "
          f"{sum(sig(ra[f+'_res']) for f in FEATS)}/{ntest(ra,'_res')}개)  "
          f"[수치] {'전부 비유의' if (nt and nsig == 0) else ('유효 검정 0개' if not nt else '유의 있음')}")
    print("     => " + ("[!] (3)에 lot 일반 효과가 섞여 있다" if nsig
                        else (UNK + " — 유효 검정이 0개다 (11차)" if not nt else
                              UNK + " — 등록 예측은 「0 근처」라는 **크기** 주장이고 "
                              "비유의는 크기의 보장이 아니다 (11차)")))

    print("\n--- 사후 대조 (사전 등록 아님. 위 판정을 바꾸지 않는다) ---")
    print("  P1 맵 shape 고정 귀무에서도 응집이 남는가 (1) — ⛔ 남든 안 남든 제품·선별 기여는 미식별 (D-053):")
    for c in CLS9:
        m, p = g1b[c]
        print(f"     {c:<11} 배수 {m:>7.2f}  p={p:.4f}  "
              f"{'남는다' if p <= 0.05 and m > 1 else '사라진다'}")
    print("  P2 맵 shape 고정 귀무에서 ICC가 남는가 (3) — 유의/유효검정 (10차 H01 반영):")
    for c, n, r in r3:
        k, kn = sum(sig(r[f + "_shp"]) for f in FEATS), ntest(r, "_shp")
        k0, k0n = sum(sig(r[f]) for f in FEATS), ntest(r)
        print(f"     {c:<11} 전체재배치 {k0}/{k0n} -> shape고정 {k}/{kn}"
              f"{'   [!] 퇴화 ' + str(6-kn) + '칸' if kn < 6 else ''}")


def power_mode():
    """②의 비유의를 「얼마나 큰 편중까지 놓치는가」로 바꾼다 (사후)."""
    lot, cls, slot, shape, n_lot = load_lots()
    print("=" * 96)
    n_sim = 400          # 🔴 11차 — 전에는 머리글에 「300회」가 박혀 있었고 실제는 400이었다
    print(f"  (2) 검출력 — 슬롯 가중 w(s) ∝ exp(λ(s−13)/12) · 귀무 B={B_PERM} · 모의 {n_sim}회"
          "\n      판정 규칙은 **본검정과 같은 순열 p ≤ 0.05**다 (11차. 전에는 95백분위였다)")
    print("=" * 96)
    for target in ("Loc", "Center", "Edge-Ring"):
        tcrit, base, obs, rows = part2_power(lot, cls, slot, target=target,
                                             lams=(0.05, 0.10, 0.15, 0.20, 0.40), n_sim=n_sim)
        print(f"\n  {target} (귀무 T 95백분위 {tcrit:.2f} · 순열 기대 평균슬롯 {base:.3f} · "
              f"실측 이탈 {obs:+.3f}칸)")
        print(f"     {'λ':>5}{'기각률':>9}{'평균슬롯 이동':>14}")
        for lam, rate, dm in rows:
            print(f"     {lam:>5.2f}{rate:>9.1%}{dm:>14.3f}")


if __name__ == "__main__":
    demo()
    if "--power" in sys.argv:
        power_mode()
    else:
        main()
