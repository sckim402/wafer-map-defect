"""잔여 ② — `none` sanity check가 1에 닿지 않는 원인 규명.

실행:
    python -u src/none_sanity.py

## 문제

가중 circular variance는 *"고르게 흩어지면 1"*이므로 산발 불량인 `none`은
1에 가까워야 한다. 그런데 **k=1에서 0.828**이다 (`w2_metric_set.md`).
밴드 재정의(D-007)로도 개선되지 않았다.

`w2_azimuth.md`가 후보 3개를 적어뒀다.

| 후보 | 원문 판정 |
|---|---|
| ① 표본 수 편향 잔존 | *"n 고정에서도 0.804이므로 이것만으로는 설명 부족"* |
| ② 유효 영역이 완전한 원이 아님 | **"가장 유력"** |
| ③ `N_WEIGHT_BINS=36`이 과도하게 촘촘 | D-005와 연동 |

## ★ 사전 등록한 가설 — ①이다. 그리고 ①의 기각 논리가 틀렸다

**`R̄`은 유한 표본에서 0으로 수렴하지 않는다.** θ가 완전히 균일해도
n개를 뽑으면 합 벡터가 우연히 한쪽으로 쏠린다. 2차원 랜덤워크이므로

    E[R̄] ≈ (√π / 2) / √n_eff  ≈  0.886 / √n_eff
    E[CV] = 1 − E[R̄]                    ← **귀무값은 1.0이 아니다**

    (가중이 있으면 n_eff = (Σw)² / Σw² 로 대체된다)

`n=20`을 넣으면 **E[CV] = 1 − 0.886/√20 = 0.802**이고,
실측 *"r≥0.8 · n고정"* 값은 **0.804**다.

> **①을 기각한 논리가 틀렸다.** *"n을 고정했는데도 0.804"*는 편향이 없다는 뜻이
> 아니라 **편향을 n=20 수준에 고정했다**는 뜻이다. 통제와 제거를 혼동했다.
> 이 스크립트의 docstring은 그 편향을 이미 알고 있었다 —
> *"불량 3개 → CV 0.52, 69개 → 0.92, 이론값은 1"* (`azimuth.py`).
> **알고 있던 사실을 sanity check의 기준선에 반영하지 않은 것이다.**

**즉 지표가 고장난 것이 아니라 검사가 잘못 정의됐다.**
비교 대상이 `1.0`이 아니라 `1 − 0.886/√n_eff`여야 한다.

## 검사 설계 — 세 기준선을 나란히 놓는다

| 기준선 | 무엇을 포함하나 | 무엇을 가려내나 |
|---|---|---|
| **이론** `1 − 0.886/√n_eff` | 유한 표본만 | 후보 ① 단독 크기 |
| **순열** (기하·n 고정, 불량 위치만 무작위) | 유한 표본 **+ 실제 기하** | 이론과의 차 = **후보 ②** |
| **실측** | 전부 | 순열과의 차 = **설명 안 되는 나머지** |

**순열 귀무분포가 이 문제의 정답 도구다.** 웨이퍼의 유효 영역·밴드·불량 개수를
그대로 두고 **어느 die가 불량인지만 섞는다.** 격자 비원형성이든 notch든
전부 그대로 보존되므로, **실측이 순열과 같으면 `none`은 균일하게 행동하는 것**이다.

후보 ③은 `n_bins`를 12~72로 쓸어 즉시 판별한다 (D-005의 예고된 검사).

## ⚠ 지표를 바꾸려는 것이 아니다

**CV는 최종 6종에 들어 있고 D-012/D-015/D-017/D-018이 그 위에 서 있다.**
이 실험의 산출은 **원인 규명과 sanity check의 재정의**이지 특징 교체가 아니다.
보정판도 계산하지만 **§3-6에 따라 보정 전후를 모두 내고 어느 클래스가
나빠지는지 본다** — 런 지표에서 이미 한 번 데인 자리다
(*"보정 후 sanity는 통과했으나 Ring과 none이 겹쳤다"*).
"""
import sys

import numpy as np

import config
from edge_band import edge_band

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

K = 1                      # D-007 확정 밴드
MIN_FAIL = 12              # D-008
N_BINS = 36                # 현행값
BIN_SWEEP = (12, 18, 24, 36, 72)
N_PERM = 40                # 웨이퍼당 순열 횟수
N_SAMPLE = 4000            # sanity check용 표본 (전수는 과하다 — azimuth.py 관례)
RAYLEIGH = np.sqrt(np.pi) / 2      # 0.8862


def band_geometry(m, k=K, n_bins=N_BINS):
    """밴드 die의 (방위각, bin 인덱스, bin별 die 수, 불량 여부)를 한 번만 계산한다.

    순열을 돌릴 때마다 침식을 다시 하면 느리다. 기하는 고정이므로 분리한다.
    """
    a = np.asarray(m)
    valid = a != config.VAL_OUTSIDE
    if not valid.any():
        return None
    b = edge_band(a, k)
    ii, jj = np.nonzero(b)
    if ii.size == 0:
        return None
    rows, cols = np.nonzero(valid)
    cy, cx = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
    hy = max((rows.max() - rows.min()) / 2, 1e-9)
    hx = max((cols.max() - cols.min()) / 2, 1e-9)
    theta = np.arctan2((ii - cy) / hy, (jj - cx) / hx)
    is_fail = a[ii, jj] == config.VAL_FAIL
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(theta, edges) - 1, 0, n_bins - 1)
    n_per_bin = np.bincount(idx, minlength=n_bins).astype(float)
    return theta, idx, n_per_bin, is_fail


def cv_from(theta_sel, w_sel):
    ws = w_sel.sum()
    if ws <= 0:
        return np.nan
    rbar = np.hypot((w_sel * np.cos(theta_sel)).sum(),
                    (w_sel * np.sin(theta_sel)).sum()) / ws
    return 1.0 - rbar


def analyse(m, rng, k=K, n_bins=N_BINS, n_perm=N_PERM):
    """실측 CV · 순열 귀무 CV 평균 · 이론 예측 CV · n_eff 를 한 번에."""
    g = band_geometry(m, k, n_bins)
    if g is None:
        return None
    theta, idx, n_per_bin, is_fail = g
    nf = int(is_fail.sum())
    if nf < MIN_FAIL:
        return None

    w_all = 1.0 / n_per_bin[idx]
    obs = cv_from(theta[is_fail], w_all[is_fail])

    # 이론 — 실측과 같은 가중을 쓰되 n_eff는 **선택된 불량**의 가중에서 낸다
    ws = w_all[is_fail]
    n_eff = ws.sum() ** 2 / (ws ** 2).sum()
    theo = 1.0 - RAYLEIGH / np.sqrt(n_eff)

    # 순열 — 기하와 불량 개수는 그대로, **어느 die가 불량인지만** 섞는다
    nb = theta.size
    perm = np.empty(n_perm)
    for t in range(n_perm):
        sel = rng.choice(nb, nf, replace=False)
        perm[t] = cv_from(theta[sel], w_all[sel])
    return obs, perm.mean(), theo, n_eff, nf


def load(cls, rng, n=N_SAMPLE):
    # allow_pickle: 이 프로젝트가 직접 만든 캐시만 읽는다 (src/*.py 공통 관례)
    with np.load(config.DATA_PROCESSED / f"{cls}.npz", allow_pickle=True) as z:
        maps = z["wafer_maps"]
    if len(maps) > n:
        maps = maps[rng.choice(len(maps), n, replace=False)]
    return maps


def main():
    rng = np.random.default_rng(config.SEED)
    classes = [config.NONE_CLASS, "Edge-Loc", "Edge-Ring", "Random", "Center"]

    print(f"밴드 k={K} (D-007) · bin={N_BINS} · 순열 {N_PERM}회/장 · 표본 {N_SAMPLE:,}장\n")
    R = {}
    for c in classes:
        rows = [r for r in (analyse(m, rng) for m in load(c, rng)) if r is not None]
        obs, perm, theo, neff, nf = (np.array(x) for x in zip(*rows))
        R[c] = dict(obs=obs, perm=perm, theo=theo, neff=neff, nf=nf)
        print(f"  {c:<11} 계산가능 {len(obs):>5,}장  중앙 n_fail {np.median(nf):>6.0f}"
              f"  n_eff {np.median(neff):>7.1f}", flush=True)

    print("\n" + "=" * 78)
    print("[1] ★ 세 기준선 — 어디까지가 편향이고 어디부터가 신호인가")
    print("=" * 78)
    print(f"  {'클래스':<11}{'실측':>8}{'순열귀무':>10}{'이론':>8}"
          f"{'실측−순열':>11}{'순열−이론':>11}")
    for c in classes:
        d = R[c]
        o, p, t = np.median(d["obs"]), np.median(d["perm"]), np.median(d["theo"])
        print(f"  {c:<11}{o:>8.3f}{p:>10.3f}{t:>8.3f}{o-p:>+11.3f}{p-t:>+11.3f}")
    print("\n  · 실측−순열 ≈ 0  →  그 클래스는 방위각으로 **균일**하다 (귀무와 구분 안 됨)")
    print("  · 순열−이론      →  **기하(비원형·notch)가 만드는 몫** = 후보 ②의 크기")
    print("  · 1 − 이론       →  **유한 표본이 만드는 몫** = 후보 ①의 크기")

    n0 = R[config.NONE_CLASS]
    gap_finite = 1.0 - np.median(n0["theo"])
    gap_geom = np.median(n0["perm"]) - np.median(n0["theo"])
    gap_resid = np.median(n0["obs"]) - np.median(n0["perm"])
    total = 1.0 - np.median(n0["obs"])
    print(f"\n  `none`의 1.0까지의 간격 {total:.3f}을 분해하면:")
    print(f"    후보 ① 유한 표본  {gap_finite:>+8.3f}  ({gap_finite/total:>6.1%})")
    print(f"    후보 ② 기하       {-gap_geom:>+8.3f}  ({-gap_geom/total:>6.1%})")
    print(f"    설명 안 되는 나머지{-gap_resid:>+8.3f}  ({-gap_resid/total:>6.1%})")

    print("\n  ⚠ 잔차 −0.021을 신호라고 부르려면 **웨이퍼 단위로** 확인해야 한다.")
    print("     중앙값 하나로는 못 쓴다 (§8).")
    print(f"\n  {'클래스':<11}{'실측−순열 평균':>16}{'표준편차':>10}"
          f"{'중앙값 SE':>11}{'obs<perm 비율':>15}")
    for c in classes:
        d = R[c]
        dif = d["obs"] - d["perm"]
        se = 1.2533 * dif.std(ddof=1) / np.sqrt(len(dif))     # 중앙값의 근사 SE
        print(f"  {c:<11}{dif.mean():>+16.3f}{dif.std(ddof=1):>10.3f}"
              f"{se:>11.4f}{(dif < 0).mean():>15.1%}")

    print("\n" + "=" * 78)
    print("[2] 후보 ③ 판별 — bin 수를 쓸어본다 (D-005의 예고된 검사)")
    print("=" * 78)
    maps_none = load(config.NONE_CLASS, np.random.default_rng(config.SEED), 1500)
    print(f"  {'n_bins':>8}{'none 실측':>12}{'순열귀무':>11}{'차':>9}")
    for nb in BIN_SWEEP:
        rr = [r for r in (analyse(m, rng, n_bins=nb, n_perm=10) for m in maps_none)
              if r is not None]
        o = np.median([x[0] for x in rr]); p = np.median([x[1] for x in rr])
        print(f"  {nb:>8}{o:>12.3f}{p:>11.3f}{o-p:>+9.3f}")
    print("\n  bin을 바꿔도 **실측과 순열이 같이 움직이면** bin은 원인이 아니다.")
    print("  (곡선 전체를 보고한다 — 승자만 쓰지 않는다, §3-9)")

    print("\n" + "=" * 78)
    print("[3] 보정하면 무엇을 잃는가 (§3-6) — 보정 전후를 모두 낸다")
    print("=" * 78)
    print("  보정판 CV* = 실측 − 순열귀무  (귀무 대비 초과 비대칭성)")
    print(f"\n  {'클래스':<11}{'보정 전':>10}{'보정 후':>10}{'none과의 거리(전)':>19}"
          f"{'(후)':>9}")
    base = np.median(n0["obs"]); base_c = np.median(n0["obs"] - n0["perm"])
    for c in classes:
        d = R[c]
        v, vc = np.median(d["obs"]), np.median(d["obs"] - d["perm"])
        print(f"  {c:<11}{v:>10.3f}{vc:>10.3f}{v-base:>19.3f}{vc-base_c:>9.3f}")
    print("\n  **`none`과의 거리가 보정 후 줄어드는 클래스가 있으면 그만큼 잃은 것이다.**")
    print("  런 지표에서 이미 겪었다 — 보정 후 sanity는 통과했으나 Ring과 none이 겹쳤다.")


if __name__ == "__main__":
    main()
