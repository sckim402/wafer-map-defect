# -*- coding: utf-8 -*-
"""순서 의존성 — lot 블록 부트스트랩으로 승자 불확실성을 직접 잰다 (D-026 예고 → D-029).

**왜 필요한가.** D-026이 임의 임계값 `TIE=0.005`를 없앴는데, D-027이 그 자리에
`MARGIN_MIN=0.01`을 새로 놓았다. 같은 병이다 — *"1·2위 차가 얼마면 갈린 것인가"*를
**내가 정하고 있다.** 부트스트랩은 그 질문을 데이터에 넘긴다:
**"lot을 다시 뽑으면 같은 쌍이 이기는가"**.

**블록이 lot인 이유.** 같은 lot의 웨이퍼는 공정 조건을 공유한다(D-003).
웨이퍼 단위로 재추출하면 그 상관을 무시해 구간이 좁아진다.

**한계 — 먼저 적는다.** 이것은 **평가 표본의 불확실성**이고 **재학습 변동이 아니다.**
OOF 예측을 고정한 채 lot만 다시 뽑는다. 재학습 변동은 seed 3개가 부분적으로만 잡는다.

**모든 조건에 같은 재추출을 쓴다** (2차 외부 검토 권고). 조건 간 차이가
재추출 잡음에서 오지 않게 하려는 것이다.

실행: ./.venv/Scripts/python.exe src/order_boot.py [B]
"""
import itertools
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import config

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEED = 20260912
CONF = 0.95          # 관례적 신뢰수준. 이 프로젝트가 고른 값이 아니다

FAM = {"E": ["cov", "ctr", "cv"], "R": ["rc", "mp"], "S": ["cc"]}
L = list(config.PATTERN_CLASSES)
NL = len(L)
PAIRS = [(i, j) for i in range(NL) for j in range(i + 1, NL)]
PNAME = [f"{L[i]}↔{L[j]}" for i, j in PAIRS]

CRITERION = f"""
=== 사전 등록한 판정 기준 (결과를 보기 전에 쓴다) ===

 1. 한 조건 (X, 선행 P, seed)의 승자는 **재추출의 {CONF:.0%} 이상에서 같은 쌍이
    유일한 argmax**일 때만 확정한다.
      - 최댓값이 둘 이상에서 나오면(동점) 그 draw는 **승자 없음**이다
      - 최댓값이 0 이하이면 그 draw도 **승자 없음**이다 (감소가 없다)
 2. 9개 조건(3 선행 × 3 seed)이 **전부 확정**일 때만 계열 X를 판정한다
 3. 확정된 3개 선행의 승자가 **모두 같으면 ⓑ 순서 무관 · 다르면 ⓐ 순서 의존**
 4. 하나라도 미확정이면 **판정 불가**

 ⚠ {CONF:.0%}는 관례적 신뢰수준이다. `TIE=0.005`·`MARGIN_MIN=0.01`과 다른 점은
    **내가 자료를 보고 고른 값이 아니라는 것**이다. 그래도 임계값이긴 하므로
    **승자 빈도를 표에 그대로 적어** 독자가 다른 선을 그을 수 있게 한다.
"""


def oof(X, y, folds):
    p = np.empty(len(y), dtype=object)
    for f in np.unique(folds):
        te = folds == f
        p[te] = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0,
            class_weight="balanced").fit(X[~te], y[~te]).predict(X[te])
    return p.astype(str)


def lot_conf(y_i, p_i, lot_id, n_lots):
    """lot마다 8x8 혼동 행렬을 미리 접어 둔다 -> 재추출이 행렬곱 한 번이 된다."""
    out = np.zeros((n_lots, NL * NL))
    np.add.at(out, (lot_id, y_i * NL + p_i), 1.0)
    return out


def rates(counts):
    """(B, 64) 혼동 개수 -> (B, 28) 쌍별 상호 오분류율."""
    M = counts.reshape(-1, NL, NL)
    row = M.sum(axis=2)
    out = np.empty((len(M), len(PAIRS)))
    for k, (i, j) in enumerate(PAIRS):
        den = row[:, i] + row[:, j]
        out[:, k] = np.where(den > 0, (M[:, i, j] + M[:, j, i]) / np.maximum(den, 1), np.nan)
    return out


def winners(dec):
    """(B, 28) 감소량 -> 승자 인덱스. 동점·비양수·NaN은 -1(승자 없음).

    ⚠ **NaN을 명시적으로 막아야 한다** (2026-09-12 5차 외부 검토 #1).
    한 쌍이라도 NaN이면 `max`가 NaN이 되는데, `dec == mx`는 전부 False라
    동점수가 0이고 `mx <= 0`도 False다 — **두 배제 조건이 다 빠져나가
    NaN 쌍이 승자가 된다.** D-027이 고친 것과 같은 계열이고, 이 함수의
    `demo()`가 0과 음수만 시험해 놓쳤다. 분모 `row[i]+row[j]`가 0인 draw에서
    실제로 발생할 수 있다(D-029 실행분에서는 0건이었다).
    """
    bad_row = ~np.isfinite(dec).all(axis=1)
    safe = np.where(np.isfinite(dec), dec, -np.inf)
    mx = safe.max(axis=1)
    top = safe.argmax(axis=1)
    n_at_max = (safe == mx[:, None]).sum(axis=1)
    return np.where(bad_row | (n_at_max > 1) | (mx <= 0), -1, top)


def demo():
    """합성 자기 검사 — 신호가 없을 때 확정을 내지 않는가."""
    zero = np.zeros((200, len(PAIRS)))
    assert (winners(zero) >= 0).sum() == 0, "감소량이 전부 0인데 승자를 냈다"
    assert (winners(-np.abs(np.arange(len(PAIRS)) + 1.0) * np.ones((200, 1))) >= 0).sum() == 0, \
        "감소량이 전부 음수인데 승자를 냈다"
    clear = np.zeros((200, len(PAIRS)))
    clear[:, 7] = 0.1
    assert (winners(clear) == 7).all(), "명백한 승자를 못 잡는다"

    # NaN 회귀 — 5차 외부 검토 #1. 0과 음수만 시험하면 이 계열을 놓친다.
    nan_zero = np.zeros((50, len(PAIRS))); nan_zero[:, 5] = np.nan
    assert (winners(nan_zero) < 0).all(), "NaN 쌍을 승자로 통과시킨다"
    nan_neg = np.full((50, len(PAIRS)), -0.5); nan_neg[:, 11] = np.nan
    assert (winners(nan_neg) < 0).all(), "NaN이 섞이면 음수뿐인데 승자를 낸다"
    nan_ok = np.zeros((50, len(PAIRS))); nan_ok[:, 3] = 0.2; nan_ok[:, 9] = np.nan
    assert (winners(nan_ok) < 0).all(), "NaN이 한 쌍이라도 있으면 그 draw는 버려야 한다"

    print("demo ok — 0/음수/NaN 입력에서 승자 0건, 명백한 승자는 200/200")


def main():
    print(CRITERION)
    rng = np.random.default_rng(SEED)

    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        y, lot, folds, seeds = (z["cls"].astype(str), z["lot"].astype(str),
                                z["folds"], z["seeds"])
    F = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        for k in ("cov", "ctr", "cv"):
            F[k] = z[k]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        F["rc"], F["mp"] = z["radial_contrast"], z["mid_peak"]
    with np.load(config.DATA_PROCESSED / "shape2.npz", allow_pickle=True) as z:
        F["cc"] = z["cc_compact"]

    idx = {c: k for k, c in enumerate(L)}
    y_i = np.array([idx[c] for c in y])
    lots, lot_id = np.unique(lot, return_inverse=True)
    n_lots = len(lots)
    print(f"웨이퍼 {len(y):,}장 · lot {n_lots:,}개 · seed {len(seeds)}개 · 재추출 B={B:,}\n")

    # 모든 조건이 공유하는 재추출 하나 (2차 외부 검토 권고)
    W = rng.multinomial(n_lots, np.full(n_lots, 1.0 / n_lots), size=B).astype(np.float64)

    LM = {}
    for r in (1, 2, 3):
        for c in itertools.combinations("ERS", r):
            S_ = frozenset(c)
            X = np.column_stack([F[k] for f in "ERS" if f in S_ for k in FAM[f]])
            LM[S_] = []
            for s in range(len(seeds)):
                p = oof(X, y, folds[s])
                LM[S_].append(lot_conf(y_i, np.array([idx[c] for c in p]), lot_id, n_lots))
            print(f"  적합 완료 {'+'.join(sorted(S_)):<6}", flush=True)

    print("\n=== 결과 ===")
    print(f"{'계열':<4}{'선행':<7}{'seed':<6}{'승자':<26}{'빈도':>8}{'확정':>6}")
    print("-" * 60)
    verdict = {}
    for X in "ERS":
        others = [f for f in "ERS" if f != X]
        Ps = [(others[0],), (others[1],), tuple(others)]
        win_per_P, all_ok = [], True
        for P in Ps:
            a, b = LM[frozenset(P) | {X}], LM[frozenset(P)]
            ws, oks = [], []
            for s in range(len(seeds)):
                dec = rates(W @ b[s]) - rates(W @ a[s])
                top = winners(dec)
                cnt = np.bincount(top[top >= 0], minlength=len(PAIRS))
                k = int(cnt.argmax())
                freq = cnt[k] / B
                ok = freq >= CONF
                ws.append(k)
                oks.append(ok)
                print(f"{X:<4}{'+'.join(P):<7}{seeds[s]:<6}{PNAME[k]:<26}"
                      f"{freq:>7.1%}{'  ✅' if ok else '  ❌':>6}")
            cond_ok = all(oks) and len(set(ws)) == 1
            all_ok &= cond_ok
            win_per_P.append(PNAME[ws[0]] if cond_ok else None)
        verdict[X] = ("판정 불가" if not all_ok else
                      "ⓑ 순서 무관" if len(set(win_per_P)) == 1 else "ⓐ 순서 의존")
        print(f"  -> [{X}] {verdict[X]}   {win_per_P}\n")
    return verdict


if __name__ == "__main__":
    demo()
    main()
