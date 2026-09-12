# -*- coding: utf-8 -*-
"""Edge-Loc의 방위각 국소성 — 층별 평균이 지운 축을 직접 잰다 (D-028).

**질문**: 「전면 오염 + 약한 가장자리 구배」와 「한쪽 가장자리에 몰린 덩어리」는
층별 프로파일이 같을 수 있다. 후자를 여러 장 평균 내면 방위가 상쇄돼 전자처럼 보인다.
`pattern_process_mapping.md`는 **층별 평균만 보고** 전자로 결론냈다.

**통계량**: 최외곽 1층 불량의 **최대 연속 90° 호 점유율**.
**귀무**: 같은 층·같은 불량 수를 **그 층의 유효 die 위에만** 재배치 (기하·표본 수 보존).
4차 외부 검토는 층별 2×K G 통계로 같은 질문을 물었다 — **통계량이 다른 두 경로다**.

돌리기: ./.venv/Scripts/python.exe src/azimuth_local.py [N_PER_CLASS] [B]
"""
import sys
import numpy as np

sys.path.insert(0, "src")
import config

N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 300
B = int(sys.argv[2]) if len(sys.argv) > 2 else 399
SEED = 20260912
ARC = 90.0          # 호 폭(도)
STEP = 10           # 호 시작점 간격(도)
MIN_DIE = 20        # 최외곽 층이 이보다 작으면 해상도가 없다
MIN_FAIL = 3


def outer_layer(v):
    """유효영역의 최외곽 1층 = 8-이웃 중 하나라도 영역 밖인 유효 die."""
    pad = np.pad(v, 1)
    inner = np.ones_like(v)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            inner &= pad[1 + dy:1 + dy + v.shape[0], 1 + dx:1 + dx + v.shape[1]]
    return v & ~inner


def arc_masks(theta):
    """호 시작점마다 '이 die가 호 안인가'를 미리 만들어 둔다."""
    return np.stack([((theta - s) % 360) < ARC for s in range(0, 360, STEP)])


def best_frac(masks, fail):
    return (masks @ fail).max() / fail.sum()


def one_map(m, rng):
    v = m > 0
    lay = outer_layer(v)
    ys, xs = np.nonzero(lay)
    if len(ys) < MIN_DIE:
        return np.nan, np.nan
    fail = (m[lay] == 2)
    k = int(fail.sum())
    if k < MIN_FAIL or k == len(fail):
        return np.nan, np.nan          # 전부 불량이면 호 점유율이 1로 고정된다
    cy, cx = np.array(np.nonzero(v)).mean(axis=1)
    masks = arc_masks(np.degrees(np.arctan2(ys - cy, xs - cx)) % 360)
    obs = best_frac(masks, fail.astype(float))
    n = len(ys)
    null = np.empty(B)
    for b in range(B):
        p = np.zeros(n)
        p[rng.choice(n, k, replace=False)] = 1.0
        null[b] = best_frac(masks, p)
    return obs, (1 + int((null >= obs).sum())) / (B + 1)


def main():
    rng = np.random.default_rng(SEED)
    print(f"최외곽 1층 · 최대 연속 {ARC:.0f}° 호 점유율 · 순열 B={B} · 클래스당 {N_PER}장")
    print(f"\n{'클래스':<11}{'n':>5}{'평균 점유':>11}{'p<=.05':>9}{'비율':>8}{'p 중앙값':>11}")
    print("-" * 56)
    out = {}
    for c in config.PATTERN_CLASSES:
        maps = np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True)["wafer_maps"]
        pick = rng.permutation(len(maps))[:N_PER]
        O, P = [], []
        for i in pick:
            o, p = one_map(maps[i], rng)
            if not np.isnan(o):
                O.append(o); P.append(p)
        O, P = np.array(O), np.array(P)
        out[c] = (len(O), O.mean(), int((P <= .05).sum()), float(np.median(P)))
        n, mo, nr, pm = out[c]
        print(f"{c:<11}{n:>5}{mo:>11.3f}{nr:>9}{nr / n:>8.1%}{pm:>11.4f}")
    return out


def demo():
    """합성 자기 검사 — 몰린 것과 안 몰린 것을 실제로 가르는가."""
    rng = np.random.default_rng(0)
    g = np.mgrid[-20:21, -20:21]
    v = (g[0] ** 2 + g[1] ** 2) <= 20 ** 2
    lay = outer_layer(v)
    ys, xs = np.nonzero(lay)
    th = np.degrees(np.arctan2(ys - 0.0, xs - 0.0)) % 360

    m = np.where(v, 1, 0).astype(np.uint8)      # ⓐ 한 방위에 몰린 덩어리
    m[ys[th < 45], xs[th < 45]] = 2
    _, p_local = one_map(m, rng)

    m2 = np.where(v, 1, 0).astype(np.uint8)     # ⓑ 같은 개수를 층 전체에 흩뿌린 것
    idx = rng.choice(len(ys), int((th < 45).sum()), replace=False)
    m2[ys[idx], xs[idx]] = 2
    _, p_unif = one_map(m2, rng)

    assert p_local <= 0.01, f"몰린 입력을 못 잡는다: p={p_local}"
    assert p_unif > 0.05, f"흩뿌린 입력을 잡아 버린다: p={p_unif}"
    print(f"demo ok — 몰림 p={p_local:.4f} / 흩뿌림 p={p_unif:.4f}")


if __name__ == "__main__":
    demo()
    main()
