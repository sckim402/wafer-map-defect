# -*- coding: utf-8 -*-
"""초록에 실린 수치를 **한 곳에서 전부 다시 만든다** (D-033, 7차 검토 정답키).

**왜 필요한가.** 1~6차 외부 검토는 **특징 6종의 값**과 **방위 검정**만 봤다.
초록의 나머지 수치 — 누수 79.5% · macro-F1 3단계 · 혼동 쌍 3개 · 외부 검증 3개 ·
층별 비 1.81~1.07 · |AUC−0.5| 0.313 — 은 **한 번도 외부에서 재계산된 적이 없다.**
9/27 잠금 전에 그것을 시킨다. 이 스크립트는 **그 채점의 정답키**다.

⚠ **이 파일은 키트에 넣지 않는다.**

돌리기: ./.venv/Scripts/python.exe src/abstract_numbers.py
"""
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, "src")
import config
from azimuth import auc
from edge_band import layer_masks

STAGES = [("가장자리 3종", ["cov", "ctr", "cv"]),
          ("+ 반경 2종", ["cov", "ctr", "cv", "rc", "mp"]),
          ("+ 형상 1종", ["cov", "ctr", "cv", "rc", "mp", "cc"])]
FEAT_LABEL = {"cov": "coverage", "ctr": "edge_contrast", "cv": "circ_var",
              "rc": "radial_contrast", "mp": "mid_peak", "cc": "cc_compact"}
RF = dict(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
          random_state=0, n_jobs=-1)
MAX_LAYER = 6


def load():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        d = dict(cls=z["cls"].astype(str), lot=z["lot"].astype(str),
                 idx_in_cls=z["idx_in_cls"], folds=z["folds"],
                 orig=z["split_orig"].astype(str), seeds=z["seeds"])
    F = {}
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        for k in ("cov", "ctr", "cv"):
            F[k] = z[k]
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        F["rc"], F["mp"] = z["radial_contrast"], z["mid_peak"]
    with np.load(config.DATA_PROCESSED / "shape2.npz", allow_pickle=True) as z:
        F["cc"] = z["cc_compact"]
    return d, F


def mutual(y, p, a, b):
    """상호 오분류율 = (M[a,b] + M[b,a]) / (M[a].sum() + M[b].sum())."""
    ab = int(((y == a) & (p == b)).sum()) + int(((y == b) & (p == a)).sum())
    den = int((y == a).sum()) + int((y == b).sum())
    return ab / den if den else np.nan


def pair_table(y_list, p_list, L):
    """**seed 3개의 평균**으로 쌍별 상호 오분류율을 매긴다.

    ⚠ **평균이 규약이다** — 초록의 0.188·0.168·0.102가 전부 3 seed 평균이다.
    seed 하나로 고르면 1단계 승자가 바뀐다(아래 `per_seed`가 그걸 보여준다).
    """
    pairs = [(L[i], L[j]) for i in range(len(L)) for j in range(i + 1, len(L))]
    M = np.array([[mutual(y, p, a, b) for a, b in pairs]
                  for y, p in zip(y_list, p_list)])
    mean = M.mean(axis=0)
    order = np.argsort(-mean)
    per_seed = [pairs[int(np.argmax(row))] for row in M]
    return pairs, mean, order, per_seed


def leakage(cls, lot):
    """웨이퍼 단위 random split이었다면 test의 몇 %가 train에 같은 lot을 갖나."""
    rng = np.random.default_rng(config.SEED)
    n = len(cls)
    te = np.zeros(n, dtype=bool)
    te[rng.choice(n, int(n * 0.2), replace=False)] = True
    tr_lots = set(lot[~te])
    out = {"(전체)": float(np.isin(lot[te], list(tr_lots)).mean())}
    for c in config.PATTERN_CLASSES:
        m = te & (cls == c)
        out[c] = float(np.isin(lot[m], list(tr_lots)).mean()) if m.sum() else np.nan
    return out


def layer_ratio(maps, cls):
    """층별 불량률 비 = Edge-Loc의 층 l 불량률 ÷ Loc의 층 l 불량률.

    **층 불량률은 전수 pooled다** — 클래스의 모든 맵에서 층 `l`의 불량 die 수를
    다 더하고 층 `l`의 die 수를 다 더해 나눈다. 웨이퍼별 비율의 평균이 **아니다**
    (그쪽은 작은 맵에 큰 가중치를 준다 — 6층에서 0.89 vs 1.02로 갈린다).
    부분표본도 쓰지 않는다.

    ⚠ **2026-08-13 표의 `Loc` 열은 이 정의로 재현되지 않는다** (D-033).
    `Edge-Loc` 열은 소수점 셋째 자리까지 맞는데 `Loc`만 계통적으로 낮다.
    **재현되는 쪽을 채택했다.**
    """
    out = []
    acc = {c: np.zeros((MAX_LAYER, 2)) for c in ("Edge-Loc", "Loc")}
    for c in ("Edge-Loc", "Loc"):
        for m in maps[cls == c]:
            a = np.asarray(m)
            for li, lay in enumerate(layer_masks(a, MAX_LAYER)):
                acc[c][li, 0] += int((a[lay] == config.VAL_FAIL).sum())
                acc[c][li, 1] += int(lay.sum())
    for li in range(MAX_LAYER):
        e = acc["Edge-Loc"][li, 0] / acc["Edge-Loc"][li, 1]
        l = acc["Loc"][li, 0] / acc["Loc"][li, 1]
        out.append(e / l)
    return out


def main():
    d, F = load()
    y, lot, cls = d["cls"], d["lot"], d["cls"]
    L = list(config.PATTERN_CLASSES)
    key = {}

    print("=== ① 표본 ===")
    for c in L:
        print(f"  {c:<11}{int((cls == c).sum()):>7,}")
    print(f"  {'합계':<11}{len(cls):>7,} · lot {len(set(lot)):,}개")
    key["n_total"] = len(cls)

    print("")
    print("=== ② 누수 — 웨이퍼 단위 random split이었다면 ===")
    lk = leakage(cls, lot)
    for k, v in lk.items():
        print(f"  {k:<11}{v:>8.1%}")
    key["leak_all"], key["leak_edge_ring"] = lk["(전체)"], lk["Edge-Ring"]

    print("")
    print("=== ③④ 주 분할 macro-F1과 최대 병목 (5-fold × seed 3개) ===")
    print(f"  {'단계':<14}{'macro-F1(평균)':>15}{'seed별':>26}   최대 병목")
    for name, ks in STAGES:
        X = np.column_stack([F[k] for k in ks])
        f1s, preds = [], []
        for s in range(len(d["seeds"])):
            p = np.empty(len(y), dtype=object)
            for f in np.unique(d["folds"][s]):
                te = d["folds"][s] == f
                p[te] = RandomForestClassifier(**RF).fit(X[~te], y[~te]).predict(X[te])
            p = p.astype(str)
            f1s.append(f1_score(y, p, average="macro", labels=L))
            preds.append(p)
        pairs, mean, order, per_seed = pair_table([y] * 3, preds, L)
        a, b = pairs[order[0]]
        a2, b2 = pairs[order[1]]
        flip = len(set(per_seed)) > 1
        print(f"  {name:<14}{np.mean(f1s):>15.4f}"
              f"{'/'.join(f'{x:.4f}' for x in f1s):>26}   "
              f"{a}↔{b} {mean[order[0]]:.3f}  (2위 {a2}↔{b2} {mean[order[1]]:.3f})"
              + ("   ⚠ seed별 승자가 갈린다" if flip else ""))
        if flip:
            print(f"  {'':<14}{'':<15}seed별 승자: "
                  + " / ".join(f"{x}↔{z}" for x, z in per_seed))
        key[f"macro_{len(ks)}"] = float(np.mean(f1s))
        key[f"pair_{len(ks)}"] = f"{a}↔{b} {mean[order[0]]:.3f}"
        key[f"pair2_{len(ks)}"] = f"{a2}↔{b2} {mean[order[1]]:.3f}"

    print("")
    print("=== ⑤ 외부 검증 — 원저자 분할(단일 fit 1회) ===")
    tr, te = d["orig"] == "Training", d["orig"] == "Test"
    for name, ks in STAGES:
        X = np.column_stack([F[k] for k in ks])
        p = RandomForestClassifier(**RF).fit(X[tr], y[tr]).predict(X[te])
        f1 = f1_score(y[te], p, average="macro", labels=L)
        print(f"  {name:<14}{f1:>10.4f}   (train {int(tr.sum()):,} / test {int(te.sum()):,})")
        key[f"ext_{len(ks)}"] = float(f1)

    print("")
    print("=== ⑥ 층별 불량률 비 Edge-Loc ÷ Loc ===")
    maps = np.empty(len(cls), dtype=object)
    for c in L:
        w = np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True)["wafer_maps"]
        m = cls == c
        maps[m] = w[d["idx_in_cls"][m]]
    lr = layer_ratio(maps, cls)
    print("  " + " · ".join(f"{i+1}층 {v:.2f}" for i, v in enumerate(lr)))
    print(f"  ⚠ 최댓값은 {int(np.argmax(lr))+1}층 {max(lr):.2f}다 — 단조 수렴이 아니다")
    key["layer_ratio"] = [float(v) for v in lr]

    print("")
    print("=== ⑦ Edge-Loc↔Loc 단일 특징 |AUC−0.5| ===")
    pos, neg = cls == "Edge-Loc", cls == "Loc"
    rows = sorted(((abs(auc(F[k][pos], F[k][neg]) - 0.5), FEAT_LABEL[k]) for k in FEAT_LABEL),
                  reverse=True)
    for v, n in rows:
        print(f"  {n:<17}{v:>8.3f}")
    key["auc_max"] = float(rows[0][0])
    key["auc_max_feat"] = rows[0][1]

    return key


def bottleneck_stability(stage=0, B=1000, seed=20260912):
    """★ 1단계 최대 병목이 **lot을 다시 뽑아도** 같은 쌍인가 (D-029의 도구를 재사용).

    **왜 이것만 따로 재나.** 3 seed 평균에서 `Center↔Loc` 0.188과
    `Loc↔Scratch` 0.185의 **차가 0.003**이고, **seed 0 단독으로는 순서가 뒤집힌다.**
    seed는 같은 25,519장의 fold 배정만 바꾸므로 **웨이퍼를 다시 뽑는 축은 못 흔든다**
    (D-029에서 배운 것이다). lot 블록으로 직접 흔든다.
    """
    d, F = load()
    y, L = d["cls"], list(config.PATTERN_CLASSES)
    name, ks = STAGES[stage]
    X = np.column_stack([F[k] for k in ks])
    idx = {c: i for i, c in enumerate(L)}
    y_i = np.array([idx[c] for c in y])
    lots, lot_id = np.unique(d["lot"].astype(str), return_inverse=True)
    nl, NL = len(lots), len(L)
    pairs = [(i, j) for i in range(NL) for j in range(i + 1, NL)]

    mats = []
    for s in range(len(d["seeds"])):
        p = np.empty(len(y), dtype=object)
        for f in np.unique(d["folds"][s]):
            te = d["folds"][s] == f
            p[te] = RandomForestClassifier(**RF).fit(X[~te], y[~te]).predict(X[te])
        pi = np.array([idx[c] for c in p.astype(str)])
        m = np.zeros((nl, NL * NL))
        np.add.at(m, (lot_id, y_i * NL + pi), 1.0)
        mats.append(m)

    rng = np.random.default_rng(seed)
    W = rng.multinomial(nl, np.full(nl, 1.0 / nl), size=B).astype(np.float64)
    acc = np.zeros((B, len(pairs)))
    for m in mats:                              # seed 3개 평균이 규약이다
        M = (W @ m).reshape(B, NL, NL)
        row = M.sum(axis=2)
        for k, (i, j) in enumerate(pairs):
            den = row[:, i] + row[:, j]
            acc[:, k] += np.where(den > 0, (M[:, i, j] + M[:, j, i]) / np.maximum(den, 1), np.nan)
    acc /= len(mats)

    bad = ~np.isfinite(acc).all(axis=1)
    safe = np.where(np.isfinite(acc), acc, -np.inf)
    top = safe.argmax(axis=1)
    tie = (safe == safe.max(axis=1)[:, None]).sum(axis=1) > 1
    top = np.where(bad | tie, -1, top)
    cnt = np.bincount(top[top >= 0], minlength=len(pairs))
    print("")
    print(f"=== ⑧ [{name}] 최대 병목의 lot 재추출 안정성 (B={B:,}, 무효 {int((top < 0).sum())}건) ===")
    for k in np.argsort(-cnt)[:3]:
        i, j = pairs[k]
        print(f"  {L[i]}↔{L[j]:<12}{cnt[k] / B:>8.1%}")
    return {f"{L[pairs[k][0]]}↔{L[pairs[k][1]]}": float(cnt[k] / B)
            for k in np.argsort(-cnt)[:3]}


def demo():
    """자기 검사 — 상호 오분류율과 AUC가 정의대로인가."""
    y = np.array(["A", "A", "B", "B"])
    p = np.array(["B", "A", "A", "B"])
    assert abs(mutual(y, p, "A", "B") - 0.5) < 1e-12, "상호 오분류율이 (1+1)/(2+2)가 아니다"
    assert np.isnan(mutual(y, p, "C", "D")), "두 클래스가 다 없는데 분모가 0이 아니다"
    # 분모는 **참 라벨 수**다 — 한쪽만 없으면 0이 아니라 다른 쪽 장수다
    assert mutual(y, p, "A", "C") == 0.0, "분모를 참 라벨 수로 안 세고 있다"
    assert abs(auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) - 0.5) < 1e-12, \
        "완전 동점 AUC가 0.5가 아니다"
    assert abs(auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) - 1.0) < 1e-12, \
        "완전 분리 AUC가 1.0이 아니다"
    print("demo ok — 상호 오분류율·AUC 동점/완전분리/빈 클래스")


if __name__ == "__main__":
    demo()
    key = main()
    for st in range(len(STAGES)):
        for nm, v in bottleneck_stability(stage=st).items():
            key[f"stab_{st}_{nm}"] = v
    np.savez(config.DATA_PROCESSED / "abstract_key.npz",
             **{k: np.array(v) for k, v in key.items()})
    print("")
    print(f"정답키 저장 — {config.DATA_PROCESSED / 'abstract_key.npz'}")
