# -*- coding: utf-8 -*-
"""Edge-Loc의 방위각 국소성 — 층별 평균이 지운 축을 직접 잰다 (D-028).

**질문**: 「전면 오염 + 약한 가장자리 구배」와 「한쪽 가장자리에 몰린 덩어리」는
층별 프로파일이 같을 수 있다. 후자를 여러 장 평균 내면 방위가 상쇄돼 전자처럼 보인다.
`pattern_process_mapping.md`는 **층별 평균만 보고** 전자로 결론냈다.

**통계량**: 최외곽 1층 불량의 **최대 연속 90° 호 점유율**.
**귀무**: 같은 층·같은 불량 수를 **그 층의 유효 die 위에만** 재배치 (기하·표본 수 보존).
4차 외부 검토는 층별 2×K G 통계로 같은 질문을 물었다 — **통계량이 다른 두 경로다**.

**⛔ 이 검사가 식별하지 않는 것** (2026-09-12 6차 외부 검토 #3, 자체 재현 완료):
  - **「한 방위의 단일 덩어리」가 아니다.** 서로 반대편 두 덩어리도 기각된다
    (각 35° 폭·32 die, 최선 호 점유 0.500, **p=0.015**). 최댓값 하나만 보기 때문이다
  - **「내부 구배의 부재」가 아니다.** 최외곽 층만 읽으므로 내부 1,101 die를
    전부 정상↔전부 불량으로 바꿔도 통계량과 p가 **소수점까지 같다**
  - 말할 수 있는 것은 **「층 내 균등 재배치에 비해 일부 90° 호의 불량 점유가 높다」**뿐이다.
    기각된 Edge-Loc조차 최선 호 점유 중앙값이 **0.474**(같은 맵의 귀무 기대 **0.339**)이고,
    **불량의 절반 이상은 그 호 밖에 있다**

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
    """호 시작점마다 '이 die가 호 안인가'를 미리 만들어 둔다.

    ⚠ **호는 닫힘 `[s, s+ARC]`이다** (2026-09-12 6차 외부 검토 #2 정정).
    반열림 `[s, s+ARC)`으로 두면 **좌우 반사에서 통계량이 바뀐다** —
    유효영역이 대칭이면 중심 `cy`가 정수라 같은 행의 die가 θ=0°·180°에
    **정확히** 얹히고, 반열림 경계가 그 점의 소속을 반사에서 뒤집는다.
    `circ_var`의 bin 경계 문제와 같은 계열이다 (D-024).

    **닫으면 사라진다.** 반사 θ→−θ는 호 `s`를 호 `270−s`로 보내고 둘 다
    10의 배수라 **족(族)이 자기 자신으로 가므로 최댓값이 보존된다.**
    `circ_var`와 달리 **이쪽은 구조적이지 않다** — 실측 1,962장 전수에서
    반열림 16장 변화(최대 |Δ| 0.038) → 닫힘 **0장**.
    """
    return np.stack([((theta - s) % 360) <= ARC for s in range(0, 360, STEP)])


def best_frac(masks, fail):
    return (masks @ fail).max() / fail.sum()


def one_map(m, rng):
    v = m > 0
    lay = outer_layer(v)
    cy, cx = np.array(np.nonzero(v)).mean(axis=1)
    ys, xs = np.nonzero(lay)

    # ⚠ **무게중심에 정확히 놓인 die는 방위각이 없다** (8차 외부 검토 C2 정정).
    # `arctan2(0, 0)`은 예외 없이 **0.0**을 돌려주므로 그 die가 **가짜 0°**를 받고,
    # 좌우 반사에서 소속 호가 달라져 **통계량이 바뀐다**(합성 예: 0.4667 → 0.4333).
    # 내부에 구멍이 있으면 중심 die가 최외곽 층에 들 수 있어 실제로 발생 가능하다.
    # **호에 넣을 수 없는 점이므로 층에서 뺀다** — 귀무의 die 풀에서도 뺀다.
    # ⚠ **현행 25,519장에는 해당 die가 0개다**(검토자 실측: 적격 25,007장 × 반사 3종
    # = 75,021건 전부 값 변화 0). **잠재 입력 결함을 막는 것이고 표를 바꾸지 않는다.**
    keep = ~((ys == cy) & (xs == cx))
    ys, xs = ys[keep], xs[keep]
    if len(ys) < MIN_DIE:
        return np.nan, np.nan
    fail = (m[lay] == 2)[keep]
    k = int(fail.sum())
    if k < MIN_FAIL or k == len(fail):
        # ⚠ 주석이 틀렸었다 (6차 외부 검토 #7): 전부 불량이면 점유율은 1이 아니라
        # **기하가 정하는 최대 호의 die 비율**이다(반경 20 원판이면 0.25).
        # 진짜 이유는 **모든 재배치가 같아 검정이 퇴화**하는 것이다(p=1).
        return np.nan, np.nan
    masks = arc_masks(np.degrees(np.arctan2(ys - cy, xs - cx)) % 360)
    obs = best_frac(masks, fail.astype(float))
    n = len(ys)
    null = np.empty(B)
    for b in range(B):
        p = np.zeros(n)
        p[rng.choice(n, k, replace=False)] = 1.0
        null[b] = best_frac(masks, p)
    return obs, (1 + int((null >= obs).sum())) / (B + 1)


def sample_rng(ci):
    """클래스 `ci`의 **표본 추출용** RNG. 귀무 추출과 분리돼 있다.

    ⚠ **7차 외부 검토 A1 정정 (2026-09-12).** 전에는 `rng` 하나가 표본 추출과
    귀무 재배치를 **번갈아** 소비했다. 그러면 `Edge-Loc`의 표본이 앞 클래스들이
    소비한 난수 개수에 달라진다 — `B`나 `N_PER`를 바꾸면 **표본이 통째로 바뀐다.**
    검토자가 클래스마다 RNG를 새로 만들자 **Edge-Loc 400장 중 370장이 교체**됐고
    기각률이 75.25% → 70.50%로 움직였다. **문서로 고정되지 않는 수치였다.**
    """
    return np.random.default_rng([SEED, ci])


def null_rng(ci, i, nseed=0):
    """맵 하나의 **귀무 재배치용** RNG. 표본 순서와 무관하게 (클래스, 맵)으로 정해진다."""
    return np.random.default_rng([SEED, ci, int(i), nseed])


def table(nseed=0, n_per=None, verbose=True):
    """클래스별 기각률 한 판. `nseed`만 바꾸면 **몬테카를로 잡음**을 볼 수 있다."""
    n_per = n_per or N_PER
    out = {}
    for ci, c in enumerate(config.PATTERN_CLASSES):
        maps = np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True)["wafer_maps"]
        pick = sample_rng(ci).permutation(len(maps))[:n_per]
        O, P, K = [], [], []
        for i in pick:
            m = np.asarray(maps[i])
            o, p = one_map(m, null_rng(ci, i, nseed))
            if not np.isnan(o):
                O.append(o); P.append(p)
                K.append(int((m[outer_layer(m > 0)] == 2).sum()))
        O, P, K = np.array(O), np.array(P), np.array(K)
        out[c] = dict(n=len(O), excluded=len(pick) - len(O), occ=float(O.mean()),
                      rej=int((P <= .05).sum()), rate=float((P <= .05).mean()),
                      p_med=float(np.median(P)), k_med=float(np.median(K)))
        if verbose:
            r = out[c]
            print(f"{c:<11}{r['n']:>5}{r['excluded']:>5}{r['k_med']:>8.0f}"
                  f"{r['occ']:>11.3f}{r['rej']:>9}{r['rate']:>8.1%}{r['p_med']:>11.4f}")
    return out


def main(nseeds=(0, 1, 2)):
    print(f"최외곽 1층 · 최대 연속 {ARC:.0f}° 호 점유율(닫힘) · 순열 B={B} · 클래스당 {N_PER}장")
    print("")
    print("⚠ 아래 '비율'은 **검정 가능한 맵에서의 명목 p<=.05 기각률**이다 (6차 검토 #6).")
    print("  검출력이 층 안 불량 수 k에 강하게 붙는다 — 같은 기하·같은 방위 가중치에서")
    print("  k만 6/20/50으로 바꾸면 기각률이 3% -> 40% -> 75%로 움직인다.")
    print("  **클래스 간 비를 국소성의 크기로 읽지 않는다.** k 중앙값과 제외 장수를 같이 적는다.")
    print("")
    print("⚠ 표본 RNG와 귀무 RNG를 **분리**했다 (7차 검토 A1). 표본은 (SEED, 클래스)로,")
    print("  귀무는 (SEED, 클래스, 맵)으로 정해진다 — B나 N_PER을 바꿔도 표본이 안 바뀐다.")
    print("  제외는 **뽑은 뒤에** 하고 분모는 적격 표본만이다 (7차 검토 A2).")
    print("")
    print(f"{'클래스':<11}{'n':>5}{'제외':>5}{'k 중앙':>8}{'평균 점유':>11}{'p<=.05':>9}{'비율':>8}{'p 중앙값':>11}")
    print("-" * 70)
    out = table(nseeds[0])

    print("")
    print(f"=== 귀무 seed만 바꾼 몬테카를로 잡음 (B={B}, seed {len(nseeds)}개) ===")
    print("  7차 검토 A3: **p가 0.05 근처인 맵의 판정이 흔들린다.** 폭을 같이 적는다.")
    rows = {c: [out[c]["rate"]] for c in config.PATTERN_CLASSES}
    for ns in nseeds[1:]:
        t = table(ns, verbose=False)
        for c in config.PATTERN_CLASSES:
            rows[c].append(t[c]["rate"])
    for c in config.PATTERN_CLASSES:
        v = rows[c]
        print(f"  {c:<11}" + " / ".join(f"{x:.1%}" for x in v)
              + f"   폭 {max(v) - min(v):.1%}p")
    print("")
    print("  → **소수점 첫째 자리를 인용하지 않는다.** 방향만 쓴다.")
    return out


def demo():
    """합성 자기 검사 — 몰린 것과 안 몰린 것을 실제로 가르는가.

    ⚠ **원점을 배열 원점으로 뒀던 것을 고쳤다** (6차 외부 검토 #5).
    원판이 (20,20) 중심인데 `arctan2(ys−0, xs−0)`으로 쟀으므로 θ가 0~90°에만
    깔렸고, `th < 45`는 의도한 45° 쐐기(20 die)가 아니라 **거의 반원(77 die)**을
    골랐다. **출력 p=0.001은 그대로였다** — 반원도 몰린 입력이라 통과했을 뿐이고,
    **검사가 무엇을 재고 있는지는 검사 자신이 보장하지 않는다.**

    ⚠ **반사 회귀를 같이 건다** (6차 #2). 몰림/흩뿌림 두 사례만 막으면
    **같은 계열의 다음 실패를 또 놓친다** (D-030).
    """
    rng = np.random.default_rng(0)
    g = np.mgrid[-20:21, -20:21]
    v = (g[0] ** 2 + g[1] ** 2) <= 20 ** 2
    lay = outer_layer(v)
    ys, xs = np.nonzero(lay)
    cy, cx = np.array(np.nonzero(v)).mean(axis=1)          # 본체와 같은 중심
    th = np.degrees(np.arctan2(ys - cy, xs - cx)) % 360
    wedge = th < 45
    assert wedge.sum() < 0.2 * len(ys),         f"45° 쐐기가 층의 {wedge.sum() / len(ys):.0%}다 — 원점이 또 어긋났다"

    m = np.where(v, 1, 0).astype(np.uint8)      # ⓐ 한 방위에 몰린 덩어리
    m[ys[wedge], xs[wedge]] = 2
    _, p_local = one_map(m, rng)

    m2 = np.where(v, 1, 0).astype(np.uint8)     # ⓑ 같은 개수를 층 전체에 흩뿌린 것
    idx = rng.choice(len(ys), int(wedge.sum()), replace=False)
    m2[ys[idx], xs[idx]] = 2
    _, p_unif = one_map(m2, rng)

    assert p_local <= 0.01, f"몰린 입력을 못 잡는다: p={p_local}"
    assert p_unif > 0.05, f"흩뿌린 입력을 잡아 버린다: p={p_unif}"

    # 반사 회귀 — 호를 닫아 놨으니 값이 정확히 같아야 한다
    for mm in (m, m2):
        for flip in (mm[:, ::-1], mm[::-1, :], mm.T):
            a, b = one_map(mm, np.random.default_rng(1))[0], one_map(flip, np.random.default_rng(1))[0]
            assert abs(a - b) < 1e-12, f"반사에서 통계량이 바뀐다: {a} vs {b}"

    # C2 회귀 (8차) — **중심 die가 층에 드는 기하**에서도 반사 불변인가.
    # 안쪽에 고리 구멍을 뚫으면 무게중심 die가 최외곽 층에 든다.
    r2 = g[0] ** 2 + g[1] ** 2
    v3 = (r2 <= 400) & ~((r2 >= 2) & (r2 <= 9))
    l3 = outer_layer(v3)
    c3 = np.array(np.nonzero(v3)).mean(axis=1)
    y3, x3 = np.nonzero(l3)
    ctr = np.flatnonzero((y3 == c3[0]) & (x3 == c3[1]))
    assert len(ctr) == 1, "이 합성 기하에 중심 die가 층에 들지 않는다 — 회귀가 죽었다"
    m3 = np.where(v3, 1, 0).astype(np.uint8)
    pick = np.unique(np.r_[np.random.default_rng(3).choice(len(y3), 30, replace=False), ctr])
    m3[y3[pick], x3[pick]] = 2
    base = one_map(m3, np.random.default_rng(5))[0]
    for flip in (m3[:, ::-1], m3[::-1, :], m3.T):
        f = one_map(flip, np.random.default_rng(5))[0]
        assert abs(base - f) < 1e-12, f"중심 die 때문에 반사에서 바뀐다: {base} vs {f}"

    print(f"demo ok — 쐐기 {int(wedge.sum())}/{len(ys)} die · 몰림 p={p_local:.4f} / "
          f"흩뿌림 p={p_unif:.4f} · 반사 6종 불변 · 중심 die 기하 반사 3종 불변")


if __name__ == "__main__":
    demo()
    main()
