"""외부 검증 — 원저자 분할(`split_orig`)에서 결론이 갈리는가.

실행:
    python -u src/external_check.py

D-003의 마지막 항목:
    *"뒤집을 조건: 없음. **두 분할에서 결론이 갈리면 그 사실 자체를 보고한다.**"*

왜 이 분할이 검증으로 쓸 만한가 (`docs/w3_split.md` §1-③):
    Training/Test에 걸친 lot이 **0개 / 10,762개**다. 원저자 분할도 lot 누수가 없는
    정당한 그룹 분할이다. 그리고 **외부에서 주어진 고정 분할**이라
    "유리하게 골랐다"는 공격이 원천 차단된다.

왜 주 분할로는 못 쓰는가:
    클래스 비율이 심하게 치우쳐 있다 — Edge-Ring test 11.6% vs Near-full 63.8%.
    **train은 Edge-Ring이 많고 test는 Edge-Loc 계열이 많은 분포 이동**이 있다.
    그리고 고정 분할이라 반복 실험이 불가능하다 (§3-1 3항 미충족).

★ 비교할 때 주의:
    **절대 수치를 직접 비교하지 않는다.** 클래스 사전확률이 다르면 정밀도가
    따라 움직여 per-class F1이 달라진다 — 그것은 결론의 차이가 아니라
    구성의 차이다.
    -> **결론이 갈리는가**를 세 가지로 본다.
       ① 혼동 쌍 **순위**가 유지되는가 (Spearman)
       ② `Edge-Loc↔Edge-Ring`이 여전히 하위인가 (병목 이동 주장)
       ③ D-011·D-012의 **증분 방향**이 유지되는가
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

import config

SEED_BOOT = 400


def fit_eval(X, y, tr, te, **kw):
    mdl = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                 n_jobs=-1, random_state=0, **kw).fit(X[tr], y[tr])
    return mdl.predict(X[te])


def pair_rates(y, p, labels):
    M = confusion_matrix(y, p, labels=labels)
    out = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            n = M[i].sum() + M[j].sum()
            out[(labels[i], labels[j])] = (M[i, j] + M[j, i]) / max(n, 1)
    return out, M


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        y = z["cls"].astype(str); orig = z["split_orig"].astype(str); F = z["folds"]
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        d = {k: z[k] for k in z.keys()}
    labels = config.PATTERN_CLASSES
    X3 = np.column_stack([d[k] for k in ("cov", "ctr", "cv")])

    tr, te = orig == "Training", orig == "Test"
    print(f"원저자 분할: train {tr.sum():,} / test {te.sum():,} "
          f"(test 비율 {te.mean()*100:.1f}%)")
    print("**반복 불가 — 고정 분할이다.** 불확실성은 test 부트스트랩으로 낸다.\n")

    p_ext = fit_eval(X3, y, tr, te, class_weight="balanced")
    # 주 분할(SGK) 재현 — seed 0의 out-of-fold
    p_sgk = np.empty(len(y), dtype=object)
    for f in np.unique(F[0]):
        m = F[0] == f
        p_sgk[m] = fit_eval(X3, y, ~m, m, class_weight="balanced")
    p_sgk = p_sgk.astype(str)

    # ── [1] per-class 비교 — 재현율 위주로 본다 ───────────
    print("=" * 78)
    print("[1] per-class 비교 — **재현율**을 본다 (정밀도는 사전확률에 딸려 움직인다)")
    print("=" * 78)
    print(f"  {'클래스':<12}{'SGK n':>8}{'ext n':>7}"
          f"{'SGK 재현':>10}{'ext 재현':>10}{'차이':>8}{'SGK F1':>9}{'ext F1':>9}")
    f_s = f1_score(y, p_sgk, labels=labels, average=None, zero_division=0)
    f_e = f1_score(y[te], p_ext, labels=labels, average=None, zero_division=0)
    for i, c in enumerate(labels):
        rs = ((p_sgk == c) & (y == c)).sum() / max((y == c).sum(), 1)
        re_ = ((p_ext == c) & (y[te] == c)).sum() / max((y[te] == c).sum(), 1)
        print(f"  {c:<12}{int((y==c).sum()):>8,}{int((y[te]==c).sum()):>7,}"
              f"{rs:>10.3f}{re_:>10.3f}{re_-rs:>+8.3f}{f_s[i]:>9.3f}{f_e[i]:>9.3f}")
    print(f"\n  macro-F1  SGK {f_s.mean():.3f}  /  외부 {f_e.mean():.3f}")

    # ── [2] 혼동 쌍 순위 ★ 결론이 갈리는지의 핵심 ─────────
    print("\n" + "=" * 78)
    print("[2] 혼동 쌍 순위 — 결론이 갈리는가")
    print("=" * 78)
    rs_, _ = pair_rates(y, p_sgk, labels)
    re_, _ = pair_rates(y[te], p_ext, labels)
    keys = list(rs_)
    rho = spearman([rs_[k] for k in keys], [re_[k] for k in keys])
    top = sorted(keys, key=lambda k: -rs_[k])
    print(f"  {'쌍':<26}{'SGK':>9}{'외부':>9}{'SGK 순위':>10}{'외부 순위':>10}")
    ord_e = sorted(keys, key=lambda k: -re_[k])
    for k in top[:8]:
        print(f"  {k[0]+' ↔ '+k[1]:<26}{rs_[k]:>9.3f}{re_[k]:>9.3f}"
              f"{top.index(k)+1:>10}{ord_e.index(k)+1:>10}")
    print(f"\n  **28개 쌍 전체의 Spearman 순위 상관 = {rho:.3f}**")

    # ── [3] 병목 이동 주장이 유지되는가 ───────────────────
    print("\n" + "=" * 78)
    print("[3] 핵심 주장 검증 — '병목이 Edge 계열에서 벗어났다'")
    print("=" * 78)
    tgt = (config.TARGET_PAIR[1], config.TARGET_PAIR[0])
    tgt = tgt if tgt in rs_ else (config.TARGET_PAIR[0], config.TARGET_PAIR[1])
    rng = np.random.default_rng(config.SEED)
    yb, pb = y[te], p_ext
    boots = []
    for _ in range(SEED_BOOT):
        s = rng.choice(len(yb), len(yb), replace=True)
        r, _ = pair_rates(yb[s], pb[s], labels)
        boots.append(r[tgt])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  Edge 계열 쌍 {tgt[0]} ↔ {tgt[1]}")
    print(f"    주 분할(SGK) : {rs_[tgt]:.3f}  (순위 {top.index(tgt)+1}/28)")
    print(f"    외부 분할    : {re_[tgt]:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
          f"  (순위 {ord_e.index(tgt)+1}/28)")
    print(f"\n  상위 3쌍 — 주 분할: {', '.join(a+'↔'+b for a,b in top[:3])}")
    print(f"           외부 분할: {', '.join(a+'↔'+b for a,b in ord_e[:3])}")

    # ── [4] D-011·D-012 증분 방향 ─────────────────────────
    print("\n" + "=" * 78)
    print("[4] D-011·D-012의 증분이 외부 분할에서도 같은 방향인가")
    print("=" * 78)
    sets = {"cov": ("cov",), "cov+ctr": ("cov", "ctr"),
            "cov+ctr+CV": ("cov", "ctr", "cv")}
    ext = {}
    for nm, ks in sets.items():
        Xs = np.column_stack([d[k] for k in ks])
        ext[nm] = f1_score(y[te], fit_eval(Xs, y, tr, te, class_weight="balanced"),
                           labels=labels, average=None, zero_division=0)
    print(f"  {'클래스':<12}{'ctr 증분(외부)':>16}{'CV 증분(외부)':>16}")
    for i, c in enumerate(labels):
        print(f"  {c:<12}{ext['cov+ctr'][i]-ext['cov'][i]:>+16.3f}"
              f"{ext['cov+ctr+CV'][i]-ext['cov+ctr'][i]:>+16.3f}")
    print(f"  {'(macro)':<12}{ext['cov+ctr'].mean()-ext['cov'].mean():>+16.3f}"
          f"{ext['cov+ctr+CV'].mean()-ext['cov+ctr'].mean():>+16.3f}")
    print("\n  주 분할 기준값: ctr macro +0.145 / CV macro +0.023")
    print("  D-012 핵심: Edge-Loc은 크게 양수, Edge-Ring은 ≈0이어야 한다")


if __name__ == "__main__":
    main()
