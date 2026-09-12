"""W5 2차 — seed별 판정. 임의 임계값을 **기준선**으로 대체한다 (§3-11).

실행:
    ./.venv/Scripts/python.exe -u src/order_seed.py     # sklearn 1.7.2 고정 (D-022)

**판정 기준은 docs/w5_order.md §3-C 에 실행 전 등록돼 있다.**
어느 (계열, 선행)이라도 3 seed의 argmax 쌍이 갈리면 **그 계열은 판정 불가**다.
"""
import itertools

import numpy as np

# 1·2위 차가 이 아래면 승자 이름을 확정하지 않는다.
# seed 흔들림 대역(±0.01)에서 가져왔다 — 새 임의값이 아니라
# 이미 프로젝트가 쓰던 잡음 대역이다 (D-026).
MARGIN_MIN = 0.01
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import config

FAM = {"E": ("cov", "ctr", "cv"), "R": ("rc", "mp"), "S": ("cc",)}


def oof(X, y, folds):
    p = np.empty(len(y), dtype=object)
    for f in np.unique(folds):
        te = folds == f
        p[te] = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0,
            class_weight="balanced").fit(X[~te], y[~te]).predict(X[te])
    return p.astype(str)


def pair_rates(y, p, L):
    M = confusion_matrix(y, p, labels=L)
    return {(L[i], L[j]): (M[i, j] + M[j, i]) / (M[i].sum() + M[j].sum())
            for i in range(len(L)) for j in range(i + 1, len(L))}


def main():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        y, F, n_seed = z["cls"].astype(str), z["folds"], len(z["seeds"])
    d = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        for k in ("cov", "ctr", "cv"):
            d[k] = z[k]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        d["rc"], d["mp"] = z["radial_contrast"], z["mid_peak"]
    with np.load(config.DATA_PROCESSED / "shape2.npz", allow_pickle=True) as z:
        d["cc"] = z["cc_compact"]
    L, fams = config.PATTERN_CLASSES, list(FAM)

    # ── seed별 쌍 오분류율 (평균하지 않는다) ──────────────────
    pr = {}
    print("[*] 계열 부분집합 × seed 평가", flush=True)
    for n in (1, 2, 3):
        for sub in itertools.combinations(fams, n):
            keys = tuple(k for f in sub for k in FAM[f])
            X = np.column_stack([d[k] for k in keys])
            pr[frozenset(sub)] = [pair_rates(y, oof(X, y, F[s]), L)
                                  for s in range(n_seed)]
            print(f"    {'+'.join(sub)}", flush=True)

    print("\n" + "=" * 86)
    print("★ seed별 「X가 가장 크게 푸는 쌍」 — 같은 선행 안의 흔들림이 기준선이다")
    print("=" * 86)
    verdict = {}
    for X in fams:
        others = [f for f in fams if f != X]
        Ps = [(others[0],), (others[1],), tuple(others)]
        print(f"\n  [{X}] {'+'.join(FAM[X])}")
        print(f"    {'선행':<8}" + "".join(f"seed{s}".ljust(26) for s in range(n_seed))
              + "seed안정")
        stable, per_P = True, []
        for P in Ps:
            a, b = pr[frozenset(P) | {X}], pr[frozenset(P)]
            wins, marg, neg = [], [], False
            for s in range(n_seed):
                dec = {k: b[s][k] - a[s][k] for k in a[s]}
                # 부호와 1·2위 차를 함께 낸다. max()만 쓰면 28쌍이 전부 동점이거나
                # 전부 악화여도 사전식 첫 쌍(Center↔Donut)을 승자로 내놓는다.
                # (2026-09-11 외부 검토 #9가 합성 귀무로 재현했다)
                srt = sorted(dec.items(), key=lambda kv: -kv[1])
                wins.append(srt[0][0])
                marg.append(srt[0][1] - srt[1][1])
                neg |= srt[0][1] <= 0
            # 승자 이름의 안정성과, 그 승자가 유효한지를 **따로** 본다.
            # 경고만 찍고 판정에 반영하지 않으면, 감소량이 0이어도 「순서 무관」이
            # 확정된다 (2026-09-12 외부 검토 #3이 합성 입력 9/9로 재현).
            name_ok = len(set(wins)) == 1
            valid = (not neg) and min(marg) >= MARGIN_MIN
            ok = name_ok and valid
            stable &= ok
            per_P.append(wins[0] if ok else None)
            cells = "".join(f"{w[0]}↔{w[1]}".ljust(26) for w in wins)
            flag = "❌ 감소량 ≤ 0" if neg else (
                "⚠ 마진 < 0.01" if min(marg) < 0.01 else ("✅" if ok else "❌ 갈림"))
            print(f"    {'+'.join(P):<8}{cells}{flag}")
            print(f"    {'':<8}1·2위 차 " + " / ".join(f"{m:.4f}" for m in marg) +
                  ("   <- 잡음 대역 ±0.01 안이다. 승자 이름을 확정하지 않는다"
                   if min(marg) < 0.01 else ""))
        if not stable:
            verdict[X] = "판정 불가"      # 이름이 갈렸거나 승자가 유효하지 않다
        else:
            verdict[X] = "ⓑ 순서 무관" if len(set(per_P)) == 1 else "ⓐ 순서 의존"
        print(f"    → {verdict[X]}")

    print("\n" + "=" * 86)
    print("★ 사전 등록 판정 (docs/w5_order.md §3-C)")
    print("=" * 86)
    for X in fams:
        print(f"    {X}: {verdict[X]}")
    if any(v == "판정 불가" for v in verdict.values()):
        print("\n  → ⚠ 판정 불가가 있다. **§2의 「ⓐ 확정」을 철회하고 계열별로 다시 쓴다**")
    elif all(v == "ⓐ 순서 의존" for v in verdict.values()):
        print("\n  → ⓐ 확정. §2 유지")
    else:
        print("\n  → 계열별로 갈린다. §3-4대로 나누어 기록한다")


if __name__ == "__main__":
    main()
