"""W6 문헌 스캔이 잡아낸 「빠뜨린 대조」 2건을 실행한다.

실행:
    ./.venv/Scripts/python.exe src/w6_checks.py      # (D-022 — base anaconda 아님)

배경 — `docs/w6_litscan.md` §5·§6:

  [1] 중복 웨이퍼맵
      원논문(Wu et al. 2015)은 811,457장 중 **696,599장만 고유**하다고 보고하고
      *"both training and test sets comprised unique wafer maps"*라며 중복을
      제거하고 분할했다. **우리는 표본 중복을 한 번도 세지 않았다** —
      `docs`·`src`의 「중복」은 전부 특징 간 상관(D-010)이다.
      lot 그룹 분할은 **같은 lot 안의** 중복만 막는다. 다른 lot에 있는 동일 맵은
      그대로 fold를 넘는다.

  [2] 회전·플립 불변성
      우리는 5개 파일에서 *"회전 불변"*을 **주장**하고 측정 기록이 0건이다.
      원논문도 같다 — *"The Rμ and Rσ **appear to be** rotation-invariant"*.
      근거가 그림을 눈으로 본 것이다. 8편 중 실측은 Sci Rep 2023 하나뿐이다.

왜 [2]의 합격선이 「계산된 귀무값」인가 (§3-13):
      웨이퍼 맵은 다이 격자다. **90°·180°·270° 회전과 좌우/상하 플립은 격자 정렬**
      이라 리샘플링이 없다 — 보간 오차가 원천적으로 0이다.
      따라서 **6종 특징값은 8개 이면군(dihedral) 변환 전부에서 부동소수점 오차 내로
      정확히 동일해야 한다.** 직관으로 정한 합격선이 아니다.
      어긋나면 **불변성 주장이 틀렸거나 구현에 버그가 있다.**
"""
import hashlib
import sys
from collections import defaultdict

import numpy as np

import config
from edge_band import coverage, band_circular_variance
from edge_contrast import edge_contrast
from radial import radial_feats
from shape2 import shape2

K = 1                    # EDGE_LAYERS (D-007)
N_PER_CLASS = 60         # [2] 클래스당 표본. 8종 x 60 x 8변환 x 6특징
TOL = 1e-9               # 상대 오차 허용 — 귀무값은 "정확히 같다"이다


# ══════════════════════════════════════════════════════════
# 공통 — 정규 순서(split_folds.npz)대로 맵을 꺼낸다
# ══════════════════════════════════════════════════════════
def load_ordered():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        cls, lot = z["cls"].astype(str), z["lot"].astype(str)
        idx, folds = z["idx_in_cls"], z["folds"]
        seeds = z["seeds"]
    per_cls = {}
    for c in config.PATTERN_CLASSES:
        with np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True) as z:
            per_cls[c] = z["wafer_maps"]
    maps = np.empty(len(cls), dtype=object)
    for i, (c, j) in enumerate(zip(cls, idx)):
        maps[i] = np.asarray(per_cls[c][j], dtype=np.uint8)
    return maps, cls, lot, folds, seeds


def six_feats(m):
    """최종 6종 (D-015 + D-017). 순서: cov ctr cv F1a F1b cc_compact."""
    rc, mp = radial_feats(m)
    return np.array([
        coverage(m, K),
        edge_contrast(m),
        band_circular_variance(m, K),   # subsample=None — 난수 개입 없음
        rc, mp,
        shape2(m)["cc_compact"],
    ], dtype=float)


FEAT_NAMES = ["coverage", "edge_contrast", "circ_var", "F1a(rc)", "F1b(mp)", "cc_compact"]


# ══════════════════════════════════════════════════════════
# [1] 중복 웨이퍼맵
# ══════════════════════════════════════════════════════════
def check_duplicates(maps, cls, lot, folds, seeds):
    print("=" * 78)
    print("[1] 중복 웨이퍼맵 — 같은 맵이 두 번 들어있는가")
    print("=" * 78)
    print(f"  대상: 패턴 8종 {len(maps):,}장 (none 제외 — D-002)")
    print("  해시: blake2b(shape + 바이트열). 완전 동일만 중복으로 센다\n")

    buckets = defaultdict(list)
    for i, m in enumerate(maps):
        a = np.ascontiguousarray(m)
        h = hashlib.blake2b(a.tobytes(), digest_size=16)
        h.update(str(a.shape).encode())
        buckets[h.digest()].append(i)

    groups = [g for g in buckets.values() if len(g) > 1]
    n_dup_members = sum(len(g) for g in groups)
    print(f"  고유 맵      {len(buckets):>8,}")
    print(f"  중복 그룹    {len(groups):>8,}")
    print(f"  중복에 속한 장수 {n_dup_members:>5,}  "
          f"({n_dup_members / len(maps) * 100:.2f}%)")
    print(f"  제거 시 남는 장수 {len(buckets):>4,}  "
          f"(-{len(maps) - len(buckets):,})")

    if not groups:
        print("\n  ✅ 중복 0건. 이 축의 누수는 없다.")
        print("     원논문은 원본 811K에서 14.2%를 중복으로 걷어냈으나,")
        print("     **라벨된 패턴 8종 부분집합에는 남아 있지 않다.**")
        return dict(n_groups=0)

    # 1-A. 라벨이 갈리는가 — 같은 맵에 다른 라벨이면 라벨 모순이다
    mixed_lab = [g for g in groups if len({cls[i] for i in g}) > 1]
    # 1-B. lot이 갈리는가 — 갈리면 그룹 분할이 막지 못한다
    cross_lot = [g for g in groups if len({lot[i] for i in g}) > 1]
    print(f"\n  라벨이 갈리는 그룹 {len(mixed_lab):>5,}  <- 같은 맵에 다른 라벨 = 라벨 모순")
    print(f"  lot이 갈리는 그룹  {len(cross_lot):>5,}  <- 그룹 분할이 막지 못하는 쪽")

    # 1-C. fold를 실제로 넘는가 (seed별)
    print("\n  fold를 넘는 중복 그룹 (seed별):")
    for s, sd in enumerate(seeds):
        crossing = [g for g in groups if len({int(folds[s][i]) for i in g}) > 1]
        n_mem = sum(len(g) for g in crossing)
        print(f"    seed {sd}: {len(crossing):>5,}개 그룹 / {n_mem:>5,}장 "
              f"({n_mem / len(maps) * 100:.2f}%)")

    print("\n  중복 그룹 크기 분포:")
    sz = np.bincount([len(g) for g in groups])
    for k in np.nonzero(sz)[0]:
        print(f"    {k}장짜리 {sz[k]:>6,}개")

    print("\n  클래스별 중복 장수:")
    per = defaultdict(int)
    for g in groups:
        for i in g:
            per[cls[i]] += 1
    tot = defaultdict(int)
    for c in cls:
        tot[c] += 1
    for c in config.PATTERN_CLASSES:
        print(f"    {c:<12}{per[c]:>6,} / {tot[c]:>6,}  "
              f"({per[c] / max(tot[c], 1) * 100:5.2f}%)")

    if mixed_lab:
        print("\n  라벨이 갈리는 예시 (최대 5건):")
        for g in mixed_lab[:5]:
            print(f"    {[f'{cls[i]}@{lot[i]}' for i in g]}")
    return dict(n_groups=len(groups), n_members=n_dup_members,
                mixed_label=len(mixed_lab), cross_lot=len(cross_lot))


# ══════════════════════════════════════════════════════════
# [2] 회전·플립 불변성
# ══════════════════════════════════════════════════════════
DIHEDRAL = [
    ("항등", lambda a: a),
    ("rot90", lambda a: np.rot90(a, 1)),
    ("rot180", lambda a: np.rot90(a, 2)),
    ("rot270", lambda a: np.rot90(a, 3)),
    ("좌우flip", lambda a: np.fliplr(a)),
    ("flip+90", lambda a: np.rot90(np.fliplr(a), 1)),
    ("flip+180", lambda a: np.rot90(np.fliplr(a), 2)),
    ("flip+270", lambda a: np.rot90(np.fliplr(a), 3)),
]


def check_invariance(maps, cls):
    print("\n" + "=" * 78)
    print("[2] 회전·플립 불변성 — 주장을 처음으로 측정한다")
    print("=" * 78)
    rng = np.random.default_rng(config.SEED)
    pick = []
    for c in config.PATTERN_CLASSES:
        idx = np.nonzero(cls == c)[0]
        n = min(N_PER_CLASS, len(idx))
        pick.append(rng.choice(idx, n, replace=False))
    pick = np.concatenate(pick)
    print(f"  표본: 클래스당 최대 {N_PER_CLASS}장 -> {len(pick)}장 "
          f"x 8변환 x 6특징 = {len(pick) * 8 * 6:,}회 계산")
    print(f"  합격선: **정확히 동일** (상대오차 < {TOL:g}). 격자 정렬 변환이라")
    print("          보간 오차가 0이므로 이건 계산된 귀무값이다 (§3-13)\n")

    base = np.array([six_feats(maps[i]) for i in pick])          # (N, 6)
    worst = np.zeros(6)
    worst_where = [("", -1)] * 6
    nan_mismatch = np.zeros(6, dtype=int)
    fail_rows = defaultdict(set)

    for name, fn in DIHEDRAL[1:]:
        cur = np.array([six_feats(fn(np.asarray(maps[i]))) for i in pick])
        for f in range(6):
            b, c = base[:, f], cur[:, f]
            nb, nc = np.isnan(b), np.isnan(c)
            nan_mismatch[f] += int((nb != nc).sum())
            ok = ~nb & ~nc
            if not ok.any():
                continue
            rel = np.abs(c[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-12)
            j = int(np.argmax(rel))
            if rel[j] > worst[f]:
                worst[f] = rel[j]
                worst_where[f] = (name, int(pick[np.nonzero(ok)[0][j]]))
            bad = np.nonzero(rel > TOL)[0]
            if bad.size:
                fail_rows[f] |= {int(pick[np.nonzero(ok)[0][k]]) for k in bad}

    print(f"  {'특징':<14}{'최대 상대오차':>16}{'발생 변환':>12}"
          f"{'불합격 맵':>10}{'NaN 불일치':>11}  판정")
    print("  " + "-" * 74)
    n_fail_feat = 0
    for f, nm in enumerate(FEAT_NAMES):
        w = worst[f]
        bad = len(fail_rows[f])
        ok = (w <= TOL) and (nan_mismatch[f] == 0)
        n_fail_feat += (not ok)
        print(f"  {nm:<14}{w:>16.3e}{worst_where[f][0]:>12}"
              f"{bad:>10,}{nan_mismatch[f]:>11,}  {'✅' if ok else '❌'}")

    print()
    if n_fail_feat == 0:
        print("  ✅ 6종 전부 8개 이면군 변환에서 동일하다. **불변성 주장이 측정으로 섰다.**")
    else:
        print(f"  ❌ {n_fail_feat}종이 불합격이다. 주장이 틀렸거나 구현 버그다.")
        print("     §3-5대로 파급 범위를 한정한다 — 어느 특징이 어느 변환에서")
        print("     깨지는지가 위 표에 있다.")
    return dict(worst=worst, n_fail=n_fail_feat, nan_mismatch=nan_mismatch,
                fail_rows={k: sorted(v) for k, v in fail_rows.items()})


def main():
    if not (config.DATA_PROCESSED / "split_folds.npz").exists():
        raise SystemExit("[중단] split_folds.npz 없음. 먼저 python src/split.py")
    maps, cls, lot, folds, seeds = load_ordered()
    r1 = check_duplicates(maps, cls, lot, folds, seeds)
    r2 = check_invariance(maps, cls)
    print("\n" + "=" * 78)
    print("요약")
    print("=" * 78)
    print(f"  [1] 중복 그룹 {r1['n_groups']:,}개")
    print(f"  [2] 불변성 불합격 특징 {r2['n_fail']}/6종")
    print("\n  -> docs/w6_litscan.md §5·§6에 결과를 적는다.")


if __name__ == "__main__":
    sys.exit(main())
