"""프로젝트 전역 설정.

여기 있는 값을 바꿀 때는 반드시 docs/decisions.md에 이유를 남긴다.
"""
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "LSWMD.pkl"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"

# ── 재현성 ────────────────────────────────────────────────
# 모든 난수는 이 seed에서 파생시킨다.
SEED = 42

# ── 클래스 ────────────────────────────────────────────────
# 분류 대상은 8종. none은 제외한다.
# 이유: 목적이 "불량 검출"이 아니라 "검출된 패턴의 원인 규명"이므로
#       none을 넣으면 오류 대부분이 (패턴 <-> none)이 되어
#       정작 보고 싶은 Edge 계열 혼동이 묻힌다.
#       → docs/decisions.md D-002 참조
PATTERN_CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]
NONE_CLASS = "none"
ALL_CLASSES = PATTERN_CLASSES + [NONE_CLASS]

# 이 프로젝트의 핵심 혼동 쌍 (W6 confusion matrix로 재확인 대상)
TARGET_PAIR = ("Edge-Loc", "Edge-Ring")

# ── 웨이퍼맵 값 ───────────────────────────────────────────
VAL_OUTSIDE = 0   # 웨이퍼 영역 밖
VAL_PASS = 1      # 정상 die
VAL_FAIL = 2      # 불량 die
