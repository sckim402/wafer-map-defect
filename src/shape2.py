"""F2 형상 특징 재설계 — `Loc↔Scratch`(0.168), 마지막 병목.

실행:
    python -u src/shape2.py

## 1차 시도가 왜 실패했나 (`shape_feats.py`)

전체 불량 die에 PCA를 걸어 이심률을 쟀더니 **Scratch 1.16, Center 1.08**로
차이가 없었다. **배경 산발이 주축을 등방으로 만든다** —
`r_med`가 실패한 것과 같은 구조다 (§3-12).

## 재설계 — 두 원칙을 함께 적용

**원칙 ① 배경을 뺀다.** 전체가 아니라 **최대 연결성분**(8-이웃)에서만 잰다.
**원칙 ② 대조비로 만든다.** 절대 통계가 아니라 **"같은 크기의 조밀한 덩어리"
대비 비**로 정의한다. `edge_contrast`·F1a가 통한 이유와 같다.

## 특징 정의

    cc_n      최대 연결성분의 die 수
    cc_frac   cc_n / 전체 불량 die        ← 응집도. 흩어진 Scratch는 낮다
    cc_len    주축 방향 die 범위(길이)
    cc_width  cc_n / cc_len               ← **유효 폭.** 1-die 스크래치면 ≈1
    ★ cc_compact = cc_width / (0.886·√cc_n)   ← **대조비**

`cc_compact`의 기준선: **원판이면 정확히 1이다.**
반지름 `r`인 원판은 `n = πr²`, 길이 `2r`, 폭 `n/(2r) = √(nπ)/2 ≈ 0.886√n`.
→ **1 = 조밀한 덩어리 / 1보다 작을수록 가늘고 길다.**
**파라미터가 없고 물리적 기준선이 이론값 1이다** (§3-9).

## 판정 (사전 등록)

`Loc↔Scratch`의 |AUC−0.5|가 기존 최고(`edge_contrast` 0.152)를 넘으면 채택 후보.
넘지 못하면 **"형상으로도 안 갈린다"가 결론**이고, 그것도 결과다.
"""
import numpy as np
from scipy import ndimage

import config
from azimuth import auc

CACHE = config.DATA_PROCESSED / "shape2.npz"
MAX_FAIL = 4000          # 이보다 불량이 많으면 생략(Near-full 등) → NaN
DISK = np.sqrt(np.pi) / 2.0      # ≈ 0.8862


def shape2(m):
    a = np.asarray(m)
    fail = a == config.VAL_FAIL
    out = dict(cc_n=np.nan, cc_frac=np.nan, cc_len=np.nan,
               cc_width=np.nan, cc_compact=np.nan, cc_aspect=np.nan)
    tot = int(fail.sum())
    if tot < 3 or tot > MAX_FAIL:
        return out
    lab, k = ndimage.label(fail, structure=np.ones((3, 3), dtype=int))  # 8-이웃
    if k == 0:
        return out
    sizes = np.bincount(lab.ravel())[1:]
    big = int(np.argmax(sizes)) + 1
    n = int(sizes[big - 1])
    out["cc_n"], out["cc_frac"] = float(n), n / tot
    if n < 3:
        return out

    yy, xx = np.nonzero(lab == big)
    P = np.column_stack([xx.astype(float), yy.astype(float)])
    P -= P.mean(0)
    ev, V = np.linalg.eigh(np.cov(P.T) + 1e-12 * np.eye(2))
    proj = P @ V[:, 1]                      # 주축(최대 고유값) 투영
    L = float(proj.max() - proj.min()) + 1.0
    out["cc_len"] = L
    out["cc_width"] = n / L
    out["cc_compact"] = (n / L) / (DISK * np.sqrt(n))
    out["cc_aspect"] = float(np.sqrt(max(ev[1], 1e-12) / max(ev[0], 1e-12)))
    return out


def build(cls_order, idx_order):
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=True) as z:
            return {k: z[k] for k in z.keys()}
    store = {}
    for c in config.PATTERN_CLASSES:
        with np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True) as z:
            store[c] = z["wafer_maps"]
    keys = ("cc_n", "cc_frac", "cc_len", "cc_width", "cc_compact", "cc_aspect")
    out = {k: np.full(len(cls_order), np.nan) for k in keys}
    for i, (c, j) in enumerate(zip(cls_order, idx_order)):
        f = shape2(store[c][j])
        for k in keys:
            out[k][i] = f[k]
        if (i + 1) % 6000 == 0:
            print(f"  {i+1:,}/{len(cls_order):,}", flush=True)
    np.savez_compressed(CACHE, **out)
    print(f"  캐시 저장: {CACHE}")
    return out


def main():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        cls, idx = z["cls"].astype(str), z["idx_in_cls"]
    d = build(cls, idx)
    with np.load(config.DATA_PROCESSED / "feats8.npz", allow_pickle=True) as z:
        base = {k: z[k] for k in z.keys()}
    with np.load(config.DATA_PROCESSED / "radial_feats.npz", allow_pickle=True) as z:
        rad = {k: z[k] for k in z.keys()}

    # ── [1] 클래스별 분포 ────────────────────────────────
    print("\n" + "=" * 84)
    print("[1] 클래스별 — `cc_compact`는 원판이면 1이다 (이론 기준선)")
    print("=" * 84)
    print(f"  {'클래스':<12}{'cc_n':>8}{'cc_frac':>9}{'cc_len':>8}"
          f"{'cc_width':>10}{'★cc_compact':>13}{'cc_aspect':>11}{'계산가능':>9}")
    for c in config.PATTERN_CLASSES:
        m = cls == c
        f = lambda k: np.nanmedian(d[k][m])
        print(f"  {c:<12}{f('cc_n'):>8.0f}{f('cc_frac'):>9.3f}{f('cc_len'):>8.1f}"
              f"{f('cc_width'):>10.2f}{f('cc_compact'):>13.3f}{f('cc_aspect'):>11.2f}"
              f"{np.mean(np.isfinite(d['cc_compact'][m]))*100:>8.1f}%")
    print("\n  기대: Scratch가 cc_width·cc_compact 최저 (가늘고 길다),")
    print("        Loc·Center는 1에 가깝다 (조밀한 덩어리)")

    # ── [2] ★ 판정 — Loc↔Scratch ────────────────────────
    print("\n" + "=" * 84)
    print("[2] ★ Loc ↔ Scratch — 기존 최고는 edge_contrast 0.152였다")
    print("=" * 84)
    mp, mn = cls == "Loc", cls == "Scratch"
    cands = [("coverage", base["cov"]), ("edge_contrast", base["ctr"]),
             ("circular var", base["cv"]), ("F1a", rad["radial_contrast"]),
             ("F1b", rad["mid_peak"]),
             ("cc_frac", d["cc_frac"]), ("cc_width", d["cc_width"]),
             ("★ cc_compact", d["cc_compact"]), ("cc_aspect", d["cc_aspect"])]
    print(f"  {'특징':<18}{'AUC':>9}{'|AUC-0.5|':>12}")
    best = 0.0
    for nm, arr in cands:
        a = auc(arr[mp], arr[mn]); v = abs(a - .5)
        best = max(best, v) if "cc_" in nm else best
        print(f"  {nm:<18}{a:>9.3f}{v:>12.3f}")
    print(f"\n  형상 특징 최고 = {best:.3f}  (기존 최고 0.152)")
    print("  → 넘으면 채택 후보. 못 넘으면 **'형상으로도 안 갈린다'가 결론이다.**")

    # ── [3] 다른 쌍도 본다 (§3-8) ────────────────────────
    print("\n" + "=" * 84)
    print("[3] 다른 쌍 — 한 쌍만 보고 판단하지 않는다 (§3-8)")
    print("=" * 84)
    pairs = [("Loc", "Scratch"), ("Edge-Loc", "Scratch"), ("Center", "Scratch"),
             ("Loc", "Random"), ("Center", "Loc"), ("Edge-Ring", "Scratch")]
    print(f"  {'쌍':<24}{'cc_frac':>10}{'cc_width':>10}{'cc_compact':>12}{'cc_aspect':>11}")
    for p, n in pairs:
        a, b = cls == p, cls == n
        print(f"  {p+' ↔ '+n:<24}" + "".join(
            f"{abs(auc(d[k][a], d[k][b]) - .5):>{w}.3f}"
            for k, w in (("cc_frac", 10), ("cc_width", 10),
                         ("cc_compact", 12), ("cc_aspect", 11))))

    # ── [4] 중복성·크기 의존 (§3-2, D-010) ──────────────
    print("\n" + "=" * 84)
    print("[4] 중복성과 크기 의존 — 채택 전 필수")
    print("=" * 84)
    print(f"  {'클래스':<12}{'cc_cmp~cov':>12}{'cc_cmp~F1a':>12}{'cc_cmp~size':>13}")
    for c in ("Loc", "Scratch", "Center", "Edge-Ring"):
        m = cls == c
        x = d["cc_compact"][m]
        ok = np.isfinite(x)
        f = lambda a: float(np.corrcoef(x[ok], a[m][ok])[0, 1])
        print(f"  {c:<12}{f(base['cov']):>12.3f}{f(rad['radial_contrast']):>12.3f}"
              f"{f(base['size']):>13.3f}")
    print("\n  |r|>0.9면 중복(D-010). size와 상관이 높으면 크기 대리변수 의심(§3-2)")


if __name__ == "__main__":
    main()
