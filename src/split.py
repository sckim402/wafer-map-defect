"""D-003 분할 구현 — lot 기준 그룹 분할. **이후 모든 수치의 전제다.**

실행:
    python src/split.py

왜 lot 기준인가 (D-003):
    같은 lot의 웨이퍼는 공정 조건이 같아 강하게 상관된다. 실측으로
    **Edge-Ring의 lot 내 coverage 차이가 무작위 쌍의 11.3%**다 — 거의 같은 맵이다.
    웨이퍼 단위 random split은 이 형제들을 train과 test에 갈라 넣어
    *"특징을 추가하니 오분류가 줄었다"*는 핵심 수치를 부풀린다.

왜 StratifiedGroupKFold인가 (GroupShuffleSplit 대신):
    이 프로젝트의 산출물은 성능 숫자가 아니라 **혼동 쌍 분석**이다.
    GSS는 seed 5개를 돌려도 웨이퍼의 16.1%가 한 번도 test에 들어가지 않고,
    빠지는 단위가 웨이퍼가 아니라 **lot 통째**다.
    -> 혼동 사례가 그 lot에 있으면 혼동 행렬에 **아예 나타나지 않는다.**
    SGK는 커버리지 100%라 전 웨이퍼에 out-of-fold 예측이 붙고,
    **25,519장 전체에 대한 혼동 행렬 하나**가 나온다.
    층화 덕에 Near-full(149장)도 폴드당 28~34로 안정적이다 (GSS는 38~49).

    **층화는 라벨 `y`만 사용한다. 특징은 쓰지 않고 lot은 통째로 유지되므로
    누수가 아니다.** (폴드 구성의 균형을 맞추는 용도)

산출물:
    data/processed/split_folds.npz
      - cls, lot, idx_in_cls : 웨이퍼 식별자 (클래스 npz의 몇 번째인지)
      - folds  (n_seeds, N)  : 각 seed에서의 fold 번호 0~4
      - split_orig           : 원저자 분할 (외부 검증용)
      - seeds                : 사용한 random_state

    **이 파일은 재생성 가능하다** (`data/`는 .gitignore 대상).
    단 sklearn 버전이 바뀌면 SGK 결과가 달라질 수 있으므로,
    최종 수치를 낼 때의 sklearn 버전을 문서에 남긴다.
"""
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

import config

N_SPLITS = 5
SEEDS = (0, 1, 2)          # 폴드 간 편차 = test 변동 / seed 간 편차 = 분할 변동


def build_index():
    """패턴 8종 전체를 하나의 인덱스로 모은다.

    lot이 여러 클래스에 걸치므로(전체 lot의 39.1%) **클래스별로 따로 분할하면
    같은 lot이 train과 test로 갈린다.** 반드시 한 번에 분할해야 한다.
    """
    cls, lot, idx, orig = [], [], [], []
    for c in config.PATTERN_CLASSES:
        with np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True) as z:
            L = z["lot_name"].astype(str)
            S = z["split_orig"].astype(str)
        cls += [c] * len(L)
        lot += list(L)
        idx += list(range(len(L)))
        orig += list(S)
    return (np.array(cls), np.array(lot), np.array(idx, dtype=np.int64),
            np.array(orig))


def make_folds(y, groups, seeds=SEEDS, n_splits=N_SPLITS):
    """seed마다 독립적인 5-fold 분할을 만든다. 반환 shape = (len(seeds), N)."""
    out = np.full((len(seeds), len(y)), -1, dtype=np.int8)
    for si, s in enumerate(seeds):
        sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=s)
        for f, (_, te) in enumerate(sgk.split(y, y, groups=groups)):
            out[si, te] = f
        assert (out[si] >= 0).all(), f"seed {s}: 배정 안 된 웨이퍼가 있다"
    return out


# ─────────────────────────────────────────────────────────
def check_lot_integrity(folds, groups, seeds=SEEDS):
    """★ 핵심 불변조건 — 한 lot의 웨이퍼는 전부 같은 fold에 있어야 한다."""
    print("\n" + "=" * 78)
    print("[1] lot 무결성 — 한 lot이 두 fold에 걸치면 분할이 무효다")
    print("=" * 78)
    ok = True
    for si, s in enumerate(seeds):
        bad = 0
        for g in np.unique(groups):
            if len(np.unique(folds[si][groups == g])) > 1:
                bad += 1
        ok &= (bad == 0)
        print(f"  seed {s}: 두 개 이상 fold에 걸친 lot = {bad}개 "
              f"{'✓' if bad == 0 else '✗ 분할 무효'}")
    if not ok:
        raise SystemExit("[중단] lot 무결성 실패. 이 분할로는 아무것도 하지 않는다.")
    print("  -> 통과. 같은 lot의 웨이퍼는 항상 같은 쪽에 있다.")


def check_composition(folds, y, seeds=SEEDS, n_splits=N_SPLITS):
    print("\n" + "=" * 78)
    print("[2] 폴드 구성 — 희소 클래스가 어느 폴드에서도 사라지지 않는가")
    print("=" * 78)
    classes = sorted(set(y))
    for si, s in enumerate(seeds):
        print(f"\n  seed {s}")
        print(f"    {'클래스':<12}" + "".join(f"{'f'+str(f):>8}" for f in range(n_splits))
              + f"{'min~max':>12}")
        for c in classes:
            n = [int(((folds[si] == f) & (y == c)).sum()) for f in range(n_splits)]
            flag = "  ← 주의" if min(n) < 20 else ""
            print(f"    {c:<12}" + "".join(f"{v:>8,}" for v in n)
                  + f"{f'{min(n)}~{max(n)}':>12}{flag}")
        sz = [int((folds[si] == f).sum()) for f in range(n_splits)]
        print(f"    {'(합계)':<12}" + "".join(f"{v:>8,}" for v in sz)
              + f"{f'{min(sz)/len(y)*100:.1f}~{max(sz)/len(y)*100:.1f}%':>12}")
    print("\n  test 표본이 20장 미만인 폴드가 있으면 그 클래스의 per-class F1은")
    print("  폴드 단위로 해석하지 않는다 — out-of-fold 전체로만 본다.")


def quantify_leakage(lot, y):
    """★ 누수 규모를 모델 없이 잰다.

    순진한 웨이퍼 단위 random split에서, test 웨이퍼가 **같은 lot 형제를
    train에 갖는 비율**. 이 비율이 높을수록 random split의 성능은 부풀려진다.
    (형제는 거의 같은 맵이다 — lot 내 coverage 차이가 무작위의 11.3%)
    """
    print("\n" + "=" * 78)
    print("[3] 누수 규모 — 만약 웨이퍼 단위 random split을 썼다면")
    print("=" * 78)
    rng = np.random.default_rng(config.SEED)
    N = len(y)
    te = np.zeros(N, dtype=bool)
    te[rng.choice(N, int(N * 0.2), replace=False)] = True
    tr_lots = set(lot[~te])
    print(f"  {'클래스':<12}{'test 장수':>10}{'형제가 train에':>15}{'비율':>9}")
    for c in sorted(set(y)):
        m = te & (y == c)
        if m.sum() == 0:
            continue
        hit = int(np.isin(lot[m], list(tr_lots)).sum())
        print(f"  {c:<12}{int(m.sum()):>10,}{hit:>15,}{hit/m.sum()*100:>8.1f}%")
    m = te
    hit = int(np.isin(lot[m], list(tr_lots)).sum())
    print(f"  {'(전체)':<12}{int(m.sum()):>10,}{hit:>15,}{hit/m.sum()*100:>8.1f}%")
    print("\n  **이 비율이 곧 random split이 부풀리는 몫의 상한이다.**")
    print("  lot당 웨이퍼가 많은 클래스일수록 높다 (Edge-Ring 9.0장 / Scratch 1.1장)")
    print("  -> Edge-Ring이 가장 크게 부풀려진다. 그룹 분할로 가장 크게 떨어질 것이다.")


def compare_split_orig(folds, y, orig, lot, seeds=SEEDS):
    print("\n" + "=" * 78)
    print("[4] 원저자 분할(split_orig)과의 관계 — 외부 검증용")
    print("=" * 78)
    span = sum(1 for g in np.unique(lot)
               if len(np.unique(orig[lot == g])) > 1)
    print(f"  Training/Test에 걸친 lot: {span}개 / {len(np.unique(lot)):,}개")
    print("  -> **원저자 분할도 이미 lot 단위다.** D-003이 상정한 '병행 검증용'이")
    print("     아니라 그 자체로 정당한 그룹 분할이다.")
    print(f"\n  {'클래스':<12}{'Training':>10}{'Test':>9}{'test 비율':>11}")
    for c in sorted(set(y)):
        m = y == c
        tr = int((orig[m] == "Training").sum()); te = int((orig[m] == "Test").sum())
        flag = "  ← 치우침" if not (0.15 <= te / (tr + te) <= 0.5) else ""
        print(f"  {c:<12}{tr:>10,}{te:>9,}{te/(tr+te)*100:>10.1f}%{flag}")
    tr = int((orig == "Training").sum()); te = int((orig == "Test").sum())
    print(f"  {'(전체)':<12}{tr:>10,}{te:>9,}{te/(tr+te)*100:>10.1f}%")
    print("\n  **전체 비율은 멀쩡한데 클래스별로는 크게 다르다.**")
    print("  train은 Edge-Ring이 많고 test는 Edge-Loc이 많다 — 분포 이동이 있다.")
    print("  그래서 주 분할로 쓰지 않고 **결론이 갈리는지 확인하는 용도**로만 쓴다.")


def check_reproducible(y, groups):
    print("\n" + "=" * 78)
    print("[5] 재현성 — 같은 seed에서 같은 분할이 나오는가")
    print("=" * 78)
    a = make_folds(y, groups, seeds=(0,))
    b = make_folds(y, groups, seeds=(0,))
    same = bool((a == b).all())
    print(f"  seed 0을 두 번 생성 -> 동일: {same} {'✓' if same else '✗'}")
    print(f"  sklearn {sklearn.__version__}")
    print("  **버전이 바뀌면 분할이 달라질 수 있다.** 최종 수치를 낼 때의 버전을")
    print("  문서에 남긴다 (현재 요구: sklearn >= 1.6 — shuffle/random_state 지원)")


def main():
    cls, lot, idx, orig = build_index()
    print(f"패턴 8종 {len(cls):,}장 / 고유 lot {len(np.unique(lot)):,}개")
    print(f"분할: StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True), "
          f"seeds={SEEDS}")

    folds = make_folds(cls, lot)

    check_lot_integrity(folds, lot)
    check_composition(folds, cls)
    quantify_leakage(lot, cls)
    compare_split_orig(folds, cls, orig, lot)
    check_reproducible(cls, lot)

    out = config.DATA_PROCESSED / "split_folds.npz"
    np.savez_compressed(out, cls=cls, lot=lot, idx_in_cls=idx,
                        folds=folds, split_orig=orig,
                        seeds=np.array(SEEDS), n_splits=N_SPLITS,
                        sklearn_version=sklearn.__version__)
    print(f"\n저장: {out}")
    print("  이후 모든 스크립트는 이 파일을 읽어 같은 분할을 쓴다.")
    print("  **분할 이전에 계산된 수치는 전부 '관찰'이며 결과가 아니다.**")


if __name__ == "__main__":
    main()
