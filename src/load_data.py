"""LSWMD.pkl을 읽어 클래스별 npz로 변환한다.

실행:
    python src/load_data.py

하는 일:
  1. data/LSWMD.pkl 로드 (메모리 수 GB 사용 — 다른 프로그램을 닫아두는 편이 좋다)
  2. failureType / trainTestLabel의 중첩 ndarray를 문자열로 펼침
  3. 라벨 보유분만 추출, 클래스별 개수 출력
  4. 웨이퍼 크기(die 수) 분포 출력
  5. data/processed/<클래스명>.npz 로 저장

왜 npz로 변환하나:
    원본 pkl 로드는 매번 수십 초가 걸린다. 이후 세션마다 반복할 이유가 없다.
    (docs/decisions.md D-001)
"""
from collections import Counter

import numpy as np
import pandas as pd

import config


def unwrap_label(value):
    """LSWMD의 failureType/trainTestLabel은 [['Center']] 같은 중첩 ndarray다.

    라벨이 없는 행은 빈 배열([])로 저장돼 있다 — 이걸 그냥 str()로 바꾸면
    '[]'라는 가짜 클래스가 생긴다. 실제로 많이 밟는 함정이므로 명시적으로 처리한다.
    """
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return str(arr.reshape(-1)[0])


def main():
    if not config.DATA_RAW.exists():
        raise SystemExit(
            f"[중단] {config.DATA_RAW} 가 없다.\n"
            "       data/README.md 를 보고 LSWMD.pkl을 먼저 받아라."
        )

    print(f"[1/5] 로드 중: {config.DATA_RAW}")
    df = pd.read_pickle(config.DATA_RAW)
    print(f"      전체 웨이퍼: {len(df):,}장")
    print(f"      컬럼: {list(df.columns)}")

    print("[2/5] 라벨 펼치는 중 (중첩 ndarray -> 문자열)")
    df["label"] = df["failureType"].map(unwrap_label)
    # 원본 WM-811K는 컬럼명에 오타가 있다: 'trianTestLabel' (train -> trian).
    # 미러/재배포본에 따라 정상 표기인 경우도 있어 둘 다 받는다.
    split_col = next(
        (c for c in ("trainTestLabel", "trianTestLabel") if c in df.columns), None
    )
    if split_col is None:
        raise SystemExit(f"[중단] 분할 라벨 컬럼 없음. 실제 컬럼: {list(df.columns)}")
    df["split_orig"] = df[split_col].map(unwrap_label)

    labeled = df[df["label"].notna()].copy()
    print(f"      라벨 보유: {len(labeled):,}장 "
          f"({len(labeled) / len(df) * 100:.1f}%)")

    print("[3/5] 클래스 분포")
    counts = Counter(labeled["label"])
    total = sum(counts.values())
    for cls in config.ALL_CLASSES:
        n = counts.get(cls, 0)
        print(f"      {cls:<12} {n:>7,}  ({n / total * 100:5.2f}%)")
    unknown = set(counts) - set(config.ALL_CLASSES)
    if unknown:
        print(f"      [경고] 예상 밖 라벨: {unknown}")

    n_pattern = sum(counts.get(c, 0) for c in config.PATTERN_CLASSES)
    print(f"      --> none 제외 패턴 합계: {n_pattern:,}장")

    print("[4/5] 웨이퍼 크기(die 수) 분포")
    shapes = labeled["waferMap"].map(lambda m: np.asarray(m).shape)
    sizes = shapes.map(lambda s: s[0] * s[1])
    print(f"      shape 종류: {shapes.nunique():,}가지")
    print(f"      die 수  min={sizes.min():,}  median={int(sizes.median()):,}  "
          f"max={sizes.max():,}")
    print("      가장 흔한 shape 상위 5개:")
    for shp, n in shapes.value_counts().head(5).items():
        print(f"        {shp}  {n:,}장")

    print(f"[5/5] 저장 중: {config.DATA_PROCESSED}")
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for cls in config.ALL_CLASSES:
        sub = labeled[labeled["label"] == cls]
        if len(sub) == 0:
            continue
        maps = np.empty(len(sub), dtype=object)
        for i, m in enumerate(sub["waferMap"].to_numpy()):
            maps[i] = np.asarray(m, dtype=np.uint8)
        out = config.DATA_PROCESSED / f"{cls}.npz"
        np.savez_compressed(
            out,
            wafer_maps=maps,
            lot_name=sub["lotName"].to_numpy().astype(str),
            wafer_index=sub["waferIndex"].to_numpy(),
            split_orig=sub["split_orig"].to_numpy().astype(str),
        )
        print(f"      {out.name:<16} {len(sub):>7,}장")

    print("\n완료. 다음: python src/explore.py")
    print("주의: npz 로드 시 allow_pickle=True 가 필요하다 (가변 크기 배열이라 object dtype).")


if __name__ == "__main__":
    main()
