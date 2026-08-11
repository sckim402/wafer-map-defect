"""불량 영역의 **형상 특징 4종** — §6 매핑표의 근거이자 F1·F2의 실질 구현.

실행:
    python -u src/shape_feats.py          # 캐시 생성 (수 분)

왜 필요한가:
    기존 세 특징(coverage·edge_contrast·CV)은 **전부 가장자리 특징**이다.
    `w3_model8.md`에서 최대 혼동이 `Center↔Loc`(0.188)·`Loc↔Scratch`(0.186)로
    이동했는데, **이 쌍들은 가장자리와 무관하다** — 즉 지금 특징으로는
    *"왜 헷갈리는가"*에 답할 수 없다. **"우리 지표에 안 보인다"밖에 못 말한다.**

    -> 오분류 사례를 특징화하려면 **불량 영역의 형상**을 재는 특징이 필요하다.
       그리고 그 특징이 곧 로드맵의 F1(방사형)·F2(기하)다.

특징 4종 — 각각 어느 클래스를 겨냥하는가:
    F1 `r_med`   : 불량 die의 정규화 반경 중앙값. **Center(작음) ↔ Loc(중간)**
    F1 `r_iqr`   : 반경 사분위 범위. **Donut(좁음, 링) ↔ Loc(넓음)**
    F2 `elong`   : 불량 좌표 PCA 고유값 비 `sqrt(λ1/λ2)`. **Scratch(큼, 선형)**
    F2 `cluster` : 최대 연결성분 / 전체 불량. **Loc(응집) ↔ Random(산발)**
    (보조) `n_tot` : 전체 불량 die 수. **Center↔Loc 크기 연속성 검증용**

주의 — 회전 불변만 쓴다 (D-007/W1 확정):
    `elong`은 **비**이므로 회전 불변이다. 주축의 *방향*은 notch 정보가 없어
    의미를 갖지 못하므로 **쓰지 않는다.**

주의 — 정규화:
    `r_med`·`r_iqr`은 유효 영역의 행/열 범위로 각각 정규화한 뒤 계산한다
    (`azimuth.polar_coords`와 동일). 맵 크기 600배 차이를 흡수하기 위해서다.
    **`n_tot`은 정규화하지 않는다** — 크기 교란이 남아 있으므로
    §6에서 해석할 때 `size`와 함께 본다.
"""
import numpy as np

import config

CACHE = config.DATA_PROCESSED / "shape_feats.npz"


def _components(fail):
    """4-이웃 연결성분 크기 목록. scipy 없이 반복 팽창으로 센다."""
    lab = np.zeros(fail.shape, dtype=np.int32)
    cur = 0
    sizes = []
    idx = np.argwhere(fail)
    seen = np.zeros(fail.shape, dtype=bool)
    H, W = fail.shape
    for y0, x0 in idx:
        if seen[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]
        seen[y0, x0] = True
        n = 0
        while stack:
            y, x = stack.pop()
            lab[y, x] = cur
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and fail[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        sizes.append(n)
    return sizes


def shape_features(m):
    """형상 특징 4종 + 보조 1종."""
    a = np.asarray(m)
    valid = a != config.VAL_OUTSIDE
    fail = a == config.VAL_FAIL
    out = dict(n_tot=np.nan, r_med=np.nan, r_iqr=np.nan,
               elong=np.nan, cluster=np.nan)
    if not valid.any() or not fail.any():
        return out

    rows, cols = np.nonzero(valid)
    cy, cx = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
    hy = max((rows.max() - rows.min()) / 2, 1e-9)
    hx = max((cols.max() - cols.min()) / 2, 1e-9)

    fy, fx = np.nonzero(fail)
    n = len(fy)
    out["n_tot"] = float(n)

    # ── F1: 방사형 프로파일 ──────────────────────────────
    y = (fy - cy) / hy
    x = (fx - cx) / hx
    r = np.hypot(x, y)
    out["r_med"] = float(np.median(r))
    out["r_iqr"] = float(np.percentile(r, 75) - np.percentile(r, 25))

    # ── F2: 이심률 (회전 불변) ───────────────────────────
    if n >= 3:
        P = np.column_stack([x, y])
        P = P - P.mean(0)
        ev = np.linalg.eigvalsh(np.cov(P.T) + 1e-12 * np.eye(2))
        out["elong"] = float(np.sqrt(max(ev[1], 1e-12) / max(ev[0], 1e-12)))

    # ── F2: 응집도 ───────────────────────────────────────
    if n <= 6000:                     # 매우 큰 맵은 비용이 커 생략(Near-full 등)
        sz = _components(fail)
        out["cluster"] = float(max(sz)) / n if sz else np.nan
    return out


def build(cls_order, idx_order):
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=True) as z:
            return {k: z[k] for k in z.keys()}
    store = {}
    for c in config.PATTERN_CLASSES:
        with np.load(config.DATA_PROCESSED / f"{c}.npz", allow_pickle=True) as z:
            store[c] = z["wafer_maps"]
    keys = ("n_tot", "r_med", "r_iqr", "elong", "cluster")
    out = {k: np.full(len(cls_order), np.nan) for k in keys}
    for i, (c, j) in enumerate(zip(cls_order, idx_order)):
        f = shape_features(store[c][j])
        for k in keys:
            out[k][i] = f[k]
        if (i + 1) % 2500 == 0:
            print(f"  {i+1:,}/{len(cls_order):,}", flush=True)
    np.savez_compressed(CACHE, **out)
    print(f"  캐시 저장: {CACHE}")
    return out


def main():
    with np.load(config.DATA_PROCESSED / "split_folds.npz", allow_pickle=True) as z:
        cls, idx = z["cls"].astype(str), z["idx_in_cls"]
    print(f"형상 특징 계산: {len(cls):,}장")
    d = build(cls, idx)

    print("\n" + "=" * 78)
    print("클래스별 형상 특징 — sanity check")
    print("=" * 78)
    print(f"  {'클래스':<12}{'n_tot':>9}{'r_med':>8}{'r_iqr':>8}"
          f"{'elong':>8}{'cluster':>9}{'계산가능':>10}")
    for c in config.PATTERN_CLASSES:
        m = cls == c
        print(f"  {c:<12}{np.nanmedian(d['n_tot'][m]):>9.0f}"
              f"{np.nanmedian(d['r_med'][m]):>8.3f}"
              f"{np.nanmedian(d['r_iqr'][m]):>8.3f}"
              f"{np.nanmedian(d['elong'][m]):>8.2f}"
              f"{np.nanmedian(d['cluster'][m]):>9.3f}"
              f"{np.mean(np.isfinite(d['cluster'][m]))*100:>9.1f}%")
    print("\n  기대: Center r_med 낮음 / Donut r_iqr 좁음 / Scratch elong 큼 /")
    print("        Loc cluster 높음 · Random cluster 낮음")
    print("  **기대와 어긋나면 그 자체가 결과다 — §3-4에 따라 기록한다.**")


if __name__ == "__main__":
    main()
