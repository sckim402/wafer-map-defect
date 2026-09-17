# -*- coding: utf-8 -*-
"""11월 목록 #7 — **연속 시작점 호**로 방위 검정을 다시 낸다 (8차 C1 → D-046 → D-051).

⛔ **예측과 반증 조건은 `docs/w9_arc_continuous.md` §1~§3. 이 파일보다 먼저 커밋됐다.**

정의는 `azimuth_local.arc_masks`가 갖는다(`grid=True`가 옛 격자판).
**후보는 중단점 전체 `{θⱼ} ∪ {θⱼ − ARC}`**다(충분조건 — 좌단만으로도 최대는 달성된다).
⛔ 정정 (D-053 → D-055): 옛 주석의 *"좌단만 쓰면 틀린다"*는 거짓이었다. 1차 구현의 실패는
**경계 허용오차를 양쪽에 주지 않은 부동소수 문제**였다 — 근거는 `azimuth_local.arc_masks` 주석.

돌리기: ./.venv/Scripts/python.exe -u src/w9_arc_cont.py [B_BOOT]
"""
import sys
import numpy as np

sys.path.insert(0, "src")
import config
import azimuth_local as az
from azimuth_local import ARC, ARC_EPS, arc_masks, null_rng, outer_layer
from w9_azimuth_boot import wilson, boot, CLASSES, ALPHA, SEED

B_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def grid_max(theta, fail, step):
    """`step`° 격자 완전탐색. **하한이다** — 고립점 최대는 못 찾는다."""
    S = np.arange(0.0, 360.0, step)
    best = 0.0
    for i in range(0, len(S), 20000):
        ch = S[i:i + 20000]
        best = max(best, float(((((theta[None, :] - ch[:, None]) % 360)
                                 <= ARC + ARC_EPS) @ fail).max()))
    return best / fail.sum()


def demo():
    """§3-3. ⚠ **반사를 반드시 본다** — 1차 구현이 여기서 걸렸다(원인은 경계 허용오차 누락. D-053)."""
    g = np.mgrid[-20:21, -20:21]
    v = (g[0] ** 2 + g[1] ** 2) <= 400
    lay = outer_layer(v)
    ys, xs = np.nonzero(lay)

    def stat(mm):
        vv = mm > 0
        l = outer_layer(vv)
        cy, cx = np.array(np.nonzero(vv)).mean(1)
        y, x = np.nonzero(l)
        k = ~((y == cy) & (x == cx))
        return (np.degrees(np.arctan2(y[k] - cy, x[k] - cx)) % 360,
                (mm[l] == 2)[k].astype(float))

    # ⓐ 반사 불변 — 격자 기하에서 300판. **1차 구현(좌단만 · 양쪽 경계 허용오차 없음)은 여기서 25/300 실패했다.**
    rng = np.random.default_rng(11)
    bad = 0
    for _ in range(300):
        m2 = np.where(v, 1, 0).astype(np.uint8)
        m2[ys[rng.choice(len(ys), 20, replace=False)],
           xs[rng.choice(len(ys), 20, replace=False)] * 0] = 1          # noop 자리맞춤
        idx = rng.choice(len(ys), 20, replace=False)
        m2 = np.where(v, 1, 0).astype(np.uint8)
        m2[ys[idx], xs[idx]] = 2
        vals = []
        for mm in (m2, m2[:, ::-1], m2[::-1, :], m2.T):
            th, f = stat(mm)
            vals.append((arc_masks(th) @ f).max() / f.sum())
        bad += (max(vals) - min(vals)) > 1e-12
    assert bad == 0, f"반사에서 통계량이 바뀐다: {bad}/300"

    # ⓑ 연속 ≥ 격자 완전탐색(하한) · 연속 ≥ 이산
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(30):
        n = int(rng.integers(8, 40))
        th = rng.random(n) * 360.0
        f = (rng.random(n) < 0.35).astype(float)
        if f.sum() < 2:
            continue
        c = (arc_masks(th) @ f).max() / f.sum()
        d = (arc_masks(th, grid=True) @ f).max() / f.sum()
        assert c >= d - 1e-12, f"연속이 이산보다 작다: {c} < {d}"
        assert c >= grid_max(th, f, 0.01) - 1e-12, "연속이 0.01° 완전탐색보다 작다"
        worst = max(worst, c - d)

    # ⓒ 8차 C1 반례 + 임의 회전 불변
    th, f = np.array([5.0, 94.0, 185.0]), np.ones(3)
    d = (arc_masks(th, grid=True) @ f).max() / 3
    c = (arc_masks(th) @ f).max() / 3
    assert abs(d - 1 / 3) < 1e-12 and abs(c - 2 / 3) < 1e-12, f"C1 반례 실패: {d}, {c}"
    for off in (37.3, 111.7, 259.01):
        assert abs((arc_masks((th + off) % 360) @ f).max() / 3 - c) < 1e-12, "회전에서 바뀐다"
    print(f"demo ok — 반사 0/300 · 연속≥이산(최대격차 {worst:.3f}) · 연속≥완전탐색 · "
          f"C1 {d:.3f}→{c:.3f} · 회전 3종 불변")


def main():
    print("=" * 92)
    print(f"  연속 시작점 호(중단점 {2}n개) · 전수 · 순열 B={az.B} · lot 재추출 {B_BOOT:,}")
    print("=" * 92)
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        cls, lot, idx_in_cls = (z["cls"].astype(str), z["lot"].astype(str), z["idx_in_cls"])
    rng = np.random.default_rng(SEED)
    rows, n_viol, n_cmp = [], 0, 0
    for c in CLASSES:
        ci = config.PATTERN_CLASSES.index(c)
        maps = np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True)["wafer_maps"]
        sel = cls == c
        lot_of = dict(zip(idx_in_cls[sel].tolist(), lot[sel].tolist()))
        rej, lots_, n_ex = [], [], 0
        for i in range(len(maps)):
            m = np.asarray(maps[i])
            o, p = az.one_map(m, null_rng(ci, i, 0))
            if np.isnan(o):
                n_ex += 1
                continue
            o_d, _ = az.one_map(m, null_rng(ci, i, 0), grid=True)
            n_cmp += 1
            n_viol += int(o < o_d - 1e-12)
            rej.append(p <= ALPHA)
            lots_.append(lot_of.get(int(i), f"__m{i}"))
        rej, lots_ = np.array(rej, bool), np.array(lots_)
        k, n = int(rej.sum()), len(rej)
        bl, bh = boot(rej, lots_, B_BOOT, rng)
        wl, wh = wilson(k, n)
        rows.append((c, n, n_ex, k / n, (bl, bh), ((bh - bl) / (wh - wl)) ** 2))
        print(f"  {c:<11} 전수 {n:>6,} (제외 {n_ex:>4,}) · 기각률 {k/n:>6.2%}", flush=True)

    print(f"\n  [V] 연속 ≥ 이산 : 비교 {n_cmp:,}건 중 위반 **{n_viol}건**")
    OLD = {"Edge-Loc": (.7170, (.7017, .7316)), "Loc": (.2007, (.1862, .2156)),
           "Edge-Ring": (.2495, (.2146, .2850)), "Center": (.0967, (.0840, .1095))}
    print("\n" + "=" * 92)
    print(f"  {'클래스':<11}{'이산(D-050)':>26}{'연속(채택판)':>26}{'구간 겹침':>10}{'deff':>8}")
    print("-" * 92)
    for c, n, ex, r, (bl, bh), deff in rows:
        orate, (ol, oh) = OLD[c]
        ov = not (bh < ol or bl > oh)
        print(f"  {c:<11}{f'{orate:.2%} [{ol:.2%}, {oh:.2%}]':>26}"
              f"{f'{r:.2%} [{bl:.2%}, {bh:.2%}]':>26}{'✅ 겹침' if ov else '❌ 분리':>10}{deff:>8.2f}")
    el, lo_ = rows[0], rows[1]
    print("\n=== 사전 등록 판정 (docs/w9_arc_continuous.md §3) ===")
    print(f"  V  위반 {n_viol}건 → {'✅' if n_viol == 0 else '❌ 구현 오류'}")
    print(f"  H2 Edge-Loc 하한 {el[4][0]:.2%} vs Loc 상한 {lo_[4][1]:.2%} → "
          f"{'✅ 분리' if el[4][0] > lo_[4][1] else '❌ 겹침'}")


if __name__ == "__main__":
    demo()
    main()
