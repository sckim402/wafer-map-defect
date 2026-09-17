# -*- coding: utf-8 -*-
"""11월 목록 #5 — 방위 기각률의 표본 + lot 군집 불확실성 (D-034 → D-046).

⛔ **예측과 반증 조건은 `docs/w9_azimuth_boot.md` §1~§3에 있고 이 스크립트보다 먼저 커밋됐다.**

⛔ **`one_map`의 RNG 소비 패턴을 바꾸지 않는다** — D-034가 고친 자리다.
   `azimuth_local`에서 그대로 import한다. 벡터화하면 순열이 달라져 비교 불가가 된다.

🔴 **`grid=True`(이산판)로 못박는다 — 10차 외부 검토 H03.**
   D-051이 `one_map`의 기본값을 연속판(`grid=False`)으로 바꿨고, 이 스크립트는
   기본값을 쓰고 있어서 **지금 돌리면 D-050의 이산 수치가 안 나왔다.**
   D-050은 이산 정의의 기록이므로 **정의를 호출부에 명시**한다
   (`w6_checks.py`가 `shape2(m, tie_nan=False)`로 고정된 것과 같은 처리).
   ⛔ **연속판 수치는 `w9_arc_cont.py`가 낸다. 두 스크립트를 섞지 않는다.**

돌리기: ./.venv/Scripts/python.exe -u src/w9_azimuth_boot.py [B_BOOT]
"""
import sys
import numpy as np

sys.path.insert(0, "src")
import config
from azimuth_local import one_map, null_rng, outer_layer, N_PER, B as B_PERM

B_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEED = 20260916
ALPHA = 0.05
CLASSES = ["Edge-Loc", "Loc", "Edge-Ring", "Center"]


def wilson(k, n, z=1.959963984540054):
    """이항 비율의 Wilson 95% 구간 — **독립 가정**. lot 군집을 무시한다."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# 400장 표본의 **실제** 기각 수 / 적격 분모 — `docs/pattern_process_mapping.md` 표(2026-09-12, D-034).
# ⛔ 12차(D-055): 10차·11차에 *"이 기록이 없다"*고 적고 비율에서 분모를 복원했는데 **기록은 처음부터 있었다.**
#    `Loc`·`Edge-Ring`은 제외 뒤 분모가 400이 아니다 (374 · 387).
SAMPLE400 = {"Edge-Loc": (289, 400), "Loc": (71, 374), "Edge-Ring": (97, 387), "Center": (31, 400)}


def h3(full_rate, k, n):
    """등록 H3 (§3): **전수 기각률**이 **400장 표본의 Wilson 구간** 안인가.

    🔴 10차 H3 — 옛 구현은 방향이 반대였다(*400장 값*이 *전수의 lot 구간* 안인가).
    대상과 분산이 둘 다 뒤바뀌었고, 등록 방향으로 재면 판정이 반대다.
    """
    lo, hi = wilson(k, n)
    return lo <= full_rate <= hi, (lo, hi)


def demo():
    """§3-3 — Wilson과 부트스트랩이 **독립 자료에서 일치**하는가 (deff ≈ 1)."""
    # H3의 방향 — `Center`: 등록 방향(전수 9.67% ∈ Wilson(31/400))은 안,
    # 옛 역방향(표본 7.75% ∈ 전수 lot 구간 [8.40, 10.95])은 밖. **둘이 갈리는 실제 사례다.**
    ok, (lo, hi) = h3(0.0967, 31, 400)
    assert ok and abs(lo - 0.0551) < 5e-5 and abs(hi - 0.1079) < 5e-5, f"H3 방향/구간: {lo:.4f},{hi:.4f}"
    assert not (0.0840 <= 31 / 400 <= 0.1095), "역방향 대조가 갈리지 않는다 — 방향 시험이 무의미하다"
    assert not h3(0.20, 31, 400)[0], "구간 밖 전수값을 통과시킨다"
    lo, hi = wilson(50, 100)
    assert abs((lo + hi) / 2 - 0.5) < 1e-9, "Wilson 중심이 안 맞는다"
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61, f"Wilson 폭이 이상하다: {lo:.3f},{hi:.3f}"

    # lot 하나에 웨이퍼 하나 = 군집 없음 -> lot 부트스트랩이 Wilson과 비슷해야 한다
    rng = np.random.default_rng(0)
    rej = (rng.random(2000) < 0.3)
    lot_id = np.arange(2000)
    r = boot(rej, lot_id, 4000, rng)
    w = wilson(int(rej.sum()), len(rej))
    deff = ((r[1] - r[0]) / (w[1] - w[0])) ** 2
    assert 0.8 < deff < 1.25, f"군집이 없는데 deff가 1에서 멀다: {deff:.2f}"

    # 같은 lot 안에서 완전히 같은 값 -> 유효 표본이 lot 수로 줄어 구간이 넓어져야 한다
    lot_id = np.arange(2000) // 20                      # lot 100개
    rej = np.repeat(rng.random(100) < 0.3, 20)
    r2 = boot(rej, lot_id, 4000, rng)
    assert (r2[1] - r2[0]) > 3 * (r[1] - r[0]), "완전 군집인데 구간이 안 넓어진다"
    print(f"demo ok — 군집 없음 deff {deff:.2f} · 완전 군집이면 폭 "
          f"{(r2[1]-r2[0])/(r[1]-r[0]):.1f}배")


def boot(rej, lot_id, B, rng):
    """lot 블록 부트스트랩 → 기각률의 95% 백분위 구간."""
    lots, inv = np.unique(lot_id, return_inverse=True)
    nl = len(lots)
    num = np.bincount(inv, weights=rej.astype(float), minlength=nl)
    den = np.bincount(inv, minlength=nl).astype(float)
    W = rng.multinomial(nl, np.full(nl, 1.0 / nl), size=B).astype(np.float64)
    d = W @ den
    r = np.where(d > 0, (W @ num) / np.maximum(d, 1), np.nan)
    return tuple(np.nanpercentile(r, [2.5, 97.5]))


def main():
    print("=" * 92)
    print(f"  전수 방위 검정 · **이산(격자) 호 grid=True** · 맵 안 순열 B={B_PERM} "
          f"(규약 불변) · lot 재추출 B_BOOT={B_BOOT:,} · SEED={SEED}")
    print("=" * 92)
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        cls, lot, idx_in_cls = (z["cls"].astype(str), z["lot"].astype(str), z["idx_in_cls"])

    rng = np.random.default_rng(SEED)
    rows = []
    for c in CLASSES:
        ci = config.PATTERN_CLASSES.index(c)
        maps = np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True)["wafer_maps"]
        sel = cls == c
        lot_of = dict(zip(idx_in_cls[sel].tolist(), lot[sel].tolist()))

        rej, lots_, n_ex = [], [], 0
        for i in range(len(maps)):
            o, p = one_map(np.asarray(maps[i]), null_rng(ci, i, 0), grid=True)
            if np.isnan(o):
                n_ex += 1
                continue
            rej.append(p <= ALPHA)
            lots_.append(lot_of.get(int(i), f"__missing_{i}"))
        rej = np.array(rej, dtype=bool)
        lots_ = np.array(lots_)
        k, n = int(rej.sum()), len(rej)
        wl, wh = wilson(k, n)
        bl, bh = boot(rej, lots_, B_BOOT, rng)
        deff = ((bh - bl) / (wh - wl)) ** 2 if wh > wl else np.nan
        rows.append((c, n, n_ex, len(np.unique(lots_)), k / n, (wl, wh), (bl, bh), deff))
        print(f"  {c:<11} 전수 {n:>6,}장 (제외 {n_ex:>4,}) · lot {len(np.unique(lots_)):>5,}개 "
              f"· 기각률 {k/n:>6.2%}", flush=True)

    print("\n" + "=" * 92)
    print(f"  {'클래스':<11}{'전수 기각률':>11}{'Wilson 95%(독립)':>24}"
          f"{'lot 블록 95%':>24}{'deff':>8}")
    print("-" * 92)
    for c, n, ex, nl, r, (wl, wh), (bl, bh), deff in rows:
        print(f"  {c:<11}{r:>11.2%}{f'[{wl:.2%}, {wh:.2%}]  ±{(wh-wl)/2:.2%}':>24}"
              f"{f'[{bl:.2%}, {bh:.2%}]  ±{(bh-bl)/2:.2%}':>24}{deff:>8.2f}")

    el, lo_ = rows[0], rows[1]
    print("\n=== 사전 등록 판정 (docs/w9_azimuth_boot.md §3) ===")
    print(f"  H1 deff>1        : {' · '.join(f'{r[0]} {r[7]:.2f}' for r in rows)}")
    sep = el[6][0] > lo_[6][1]
    print(f"  H2 Edge-Loc 하한 {el[6][0]:.2%} {'>' if sep else '≤'} Loc 상한 {lo_[6][1]:.2%}"
          f"   → {'✅ 분리' if sep else '❌ 겹침'}")
    print("  H3 전수 기각률이 400장 표본의 Wilson 구간 안인가 (등록 방향 · 실제 k/n):")
    for c, n, ex, nl, r, w, b, deff in rows:
        k4, n4 = SAMPLE400[c]
        ok, (lo, hi) = h3(r, k4, n4)
        print(f"     {c:<11} 전수 {r:>6.2%}  vs 400장 {k4}/{n4} Wilson [{lo:.2%}, {hi:.2%}]"
              f"  → {'✅ 안' if ok else '❌ 밖'}")
    print("     ⛔ 「안」은 한 번의 구간 포함이다 — 400장 표본의 대표성·무편향을 증명하지 않는다")


if __name__ == "__main__":
    demo()
    main()
