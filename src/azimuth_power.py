# -*- coding: utf-8 -*-
"""방위 국소성 기각률의 대조 — 66.2% vs 21.1%가 국소성인가 검출력인가 (§3-2).

**6차 외부 검토 #6이 제기했다.** 순열 검정의 검출력은 층 안 불량 수 `k`에 붙는다.
같은 기하·같은 방위 가중치에서 `k`만 6/20/50으로 바꾸면 기각률이 3% -> 40% -> 75%다.
**Edge-Loc은 Loc보다 `k`가 크다**(중앙값 51 vs 34). 그러면 기각률 차이는
국소성이 아니라 검출력일 수 있다 — `F3 AUC 0.927`이 개수 효과였던 것과 같은 형태다.

**그래서 `k` 계층을 맞추고 다시 잰다.** 맞춘 뒤에도 차이가 남으면 국소성이다.

돌리기: ./.venv/Scripts/python.exe src/azimuth_power.py [N_PER] [B]
"""
import sys

import numpy as np

sys.path.insert(0, "src")
import config
import azimuth_local as AZ
from azimuth_local import one_map, outer_layer

N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 400
AZ.B = int(sys.argv[2]) if len(sys.argv) > 2 else 999
SEED = 20260912
CLASSES = ("Edge-Loc", "Loc", "Edge-Ring", "Center")
EDGES = [3, 10, 16, 25, 40, 10 ** 9]
MIN_CELL = 10          # 이보다 적은 칸은 비율을 적지 않는다


def probe(c, rng):
    maps = np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True)["wafer_maps"]
    rows = []
    for i in rng.permutation(len(maps))[:N_PER]:
        m = np.asarray(maps[i])
        o, p = one_map(m, rng)
        if np.isnan(o):
            continue
        rows.append((int((m[outer_layer(m > 0)] == 2).sum()), o, p))
    return np.array(rows)


def main():
    rng = np.random.default_rng(SEED)
    R = {c: probe(c, rng) for c in CLASSES}

    print(f"클래스당 {N_PER}장 · 순열 B={AZ.B}")
    print("")
    print(f"{'클래스':<11}{'검정 가능':>10}{'기각률':>9}{'k 중앙':>8}{'k 평균':>9}")
    for c, r in R.items():
        print(f"{c:<11}{len(r):>10}{(r[:, 2] <= .05).mean():>9.1%}"
              f"{np.median(r[:, 0]):>8.0f}{r[:, 0].mean():>9.1f}")

    print("")
    print("=== k 계층을 맞춘 기각률 — 여기서도 차이가 남는가 ===")
    print(f"{'k 구간':<10}" + "".join(f"{c:>20}" for c in CLASSES))
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        lab = f"{a}~{b - 1}" if b < 10 ** 9 else f"{a}+"
        line = f"{lab:<10}"
        for c in CLASSES:
            s = R[c][(R[c][:, 0] >= a) & (R[c][:, 0] < b)]
            cell = (f"{(s[:, 2] <= .05).mean():.1%} (n={len(s)})" if len(s) >= MIN_CELL
                    else f"- (n={len(s)})")
            line += f"{cell:>20}"
        print(line)
    print("")
    print("⚠ 이 표는 **비를 고정하지 못한다** — k 안에서도 분포가 클래스마다 다르다.")
    print("  판정은 「k를 맞춰도 방향이 유지되는가」까지이고, 3.1배라는 배수는 쓰지 않는다.")
    return R


def spread(n_per=200, seeds=(1, 2, 3)):
    """표본을 다시 뽑으면 기각률이 얼마나 움직이는가.

    같은 lot이 공정 조건을 공유하므로(D-003) 이항 오차는 **하한**일 것이다.
    ⚠ 다만 **초과 산포를 실측하지는 못했다** — n=200에서 3 draw 폭 4.5~9.0pp는
    이항 기대 폭(≈2.5σ = 5~8pp) 안이다. **군집 때문이라고 쓰지 않는다**(§3-11:
    기준선을 먼저 계산한다). lot 블록 부트스트랩은 11월 재측정에 건다.
    """
    print("")
    print(f"=== 표본을 다시 뽑았을 때의 폭 (클래스당 {n_per}장 × seed {len(seeds)}개) ===")
    old_B, AZ.B = AZ.B, 399
    for c in CLASSES:
        r = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            x = probe_n(c, rng, n_per)
            r.append((x[:, 2] <= .05).mean())
        print(f"  {c:<11}" + " / ".join(f"{v:.1%}" for v in r) + f"   폭 {max(r) - min(r):.1%}p")
    AZ.B = old_B
    print("  ⚠ 이 폭은 이항 오차로 설명되는 범위 안이다 — **초과 산포는 아직 못 쟀다.**")
    print("     결론은 하나뿐이다: **소수점 첫째 자리를 인용하지 않는다.** 방향만 쓴다.")


def probe_n(c, rng, n):
    maps = np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True)["wafer_maps"]
    rows = []
    for i in rng.permutation(len(maps))[:n]:
        m = np.asarray(maps[i])
        o, p = one_map(m, rng)
        if not np.isnan(o):
            rows.append((int((m[outer_layer(m > 0)] == 2).sum()), o, p))
    return np.array(rows)


def counterexample():
    """이 검정이 **식별하지 않는 것**을 합성으로 보인다 (6차 외부 검토 #3).

    기각을 「한 방위의 단일 덩어리」로 읽으면 안 되는 이유를 수치로 남긴다.
    """
    g = np.mgrid[-20:21, -20:21]
    v = (g[0] ** 2 + g[1] ** 2) <= 20 ** 2
    lay = outer_layer(v)
    ys, xs = np.nonzero(lay)
    cy, cx = np.array(np.nonzero(v)).mean(axis=1)
    th = np.degrees(np.arctan2(ys - cy, xs - cx)) % 360

    def run(sel, rng=None):
        m = np.where(v, 1, 0).astype(np.uint8)
        m[ys[sel], xs[sel]] = 2
        return m, one_map(m, rng or np.random.default_rng(0))

    print("")
    print("=== 이 검정이 식별하지 않는 것 ===")
    print("  ⓐ 서로 반대편 두 덩어리 — 단일 덩어리가 아닌데도 기각되는가")
    print(f"  {'각 덩어리 폭':>12}{'die':>6}{'최선 호 점유':>13}{'p':>9}")
    two = None
    for w in (22.5, 35, 45, 60):
        sel = (th < w) | ((th >= 180) & (th < 180 + w))
        _, (o, pv) = run(sel)
        if w == 35:
            two = sel
        print(f"  {w:>10.1f}°{int(sel.sum()):>6}{o:>13.3f}{pv:>9.4f}"
              + ("   <- 기각" if pv <= .05 else ""))

    m_lo, a = run(two, np.random.default_rng(1))
    m_hi = m_lo.copy()
    m_hi[v & ~lay] = 2                      # 내부만 전부 불량으로
    b = one_map(m_hi, np.random.default_rng(1))
    print(f"  ⓑ 내부 {int((v & ~lay).sum()):,} die를 전부 정상->전부 불량: "
          f"통계량 {a[0]:.6f} vs {b[0]:.6f} · p {a[1]:.4f} vs {b[1]:.4f}")
    assert a == b, "최외곽 층만 읽어야 하는데 내부가 결과를 바꿨다"
    print("  -> **구배는 측정 대상이 아니다.** 기각을 「구배의 부재」로 읽지 않는다.")


def demo():
    """자기 검사 — k만 바꾼 같은 기하에서 기각률이 실제로 움직이는가.

    움직이지 않으면 이 대조 자체가 무의미하다. **대조가 살아 있는지부터 본다.**
    """
    rng = np.random.default_rng(0)
    g = np.mgrid[-20:21, -20:21]
    v = (g[0] ** 2 + g[1] ** 2) <= 20 ** 2
    lay = outer_layer(v)
    ys, xs = np.nonzero(lay)
    cy, cx = np.array(np.nonzero(v)).mean(axis=1)
    th = np.degrees(np.arctan2(ys - cy, xs - cx)) % 360
    w = np.where(th < 90, 3.0, 1.0)                 # 같은 방위 편향, k만 바꾼다
    w /= w.sum()

    old_B, AZ.B = AZ.B, 199
    rates = []
    for k in (6, 20, 50):
        hit = 0
        for _ in range(60):
            m = np.where(v, 1, 0).astype(np.uint8)
            sel = rng.choice(len(ys), k, replace=False, p=w)
            m[ys[sel], xs[sel]] = 2
            hit += one_map(m, rng)[1] <= .05
        rates.append(hit / 60)
    AZ.B = old_B
    assert rates[0] < rates[-1] - 0.3, f"k에 검출력이 안 붙는다 — 대조가 죽었다: {rates}"
    print(f"demo ok — 같은 방위 편향에서 k=6/20/50 기각률 "
          f"{rates[0]:.0%}/{rates[1]:.0%}/{rates[2]:.0%} (검출력이 k에 붙는다)")


if __name__ == "__main__":
    demo()
    main()
    spread()
    counterexample()
