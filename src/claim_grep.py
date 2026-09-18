"""정정 전파 점검 — 계열별 핵심어 사전 (D-056, 13차 §5 「S2 ≥ 1」 발동).

12차 뒤 「15계열 grep」을 기억으로 돌렸고 13차에서 목록 밖 현행 잔존 4곳이 나왔다.
놓친 형태는 셋이었다: 조건 줄의 괄호 표기(「제품(맵 shape)」) · 표 머리글(「제품 고정 귀무」) ·
추론형(「크기가 같다 → 그래서 제품 효과와 안 갈린다」). 그래서 사전을 **코드로** 둔다.

판정: **적중한 그 줄에** 정정 표지(⛔ · ~~ · 원판 보존 · 정정 · 철회 · 금지 …)가 없으면
「표지 없는 적중」으로 인쇄한다. ⛔ **앞뒤 몇 줄을 보면 안 된다** — 1차판이 ±3줄을 봤더니
13차 키트의 알려진 잔존 6곳 중 4곳(R1-01·R1-03·K1·K2)이 **옆 줄의 다른 정정** 때문에 숨었다.
**「근처에 정정이 있다」는 「이 주장이 정정됐다」가 아니다** — 잔존은 바로 그런 자리에 남는다. **표지 없는 적중 = 잔존은 아니다** — 사람이 읽고 가른다.
⛔ 역으로 「표지 없는 적중 0」도 전파 완료가 아니다. 사전에 없는 말은 못 찾는다.

    python src/claim_grep.py            # 리포 docs·src + ../CLAUDE.md + ../planning (있으면)
    python src/claim_grep.py --all      # 표지 있는 적중까지 전부
    python src/claim_grep.py <폴더>     # 그 폴더만 (13차 키트로 사전 자체를 검증했다)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 계열 → 정규식. 새 정정을 하면 **옛 주장의 핵심어를 여기 먼저 넣고** 돌린다 (지침 §5-3).
FAMILIES = {
    "C01·C03 shape≠제품": r"제품 ?고정|제품\(맵 ?shape\)|제품 효과|같은 제품|lot ?≈ ?제품|제품 누수|제품 중복|제품고정",
    "C02 deff≠라벨 군집": r"같은 군집|deff.{0,40}(설명한다|뿌리)|같은 구조로",
    "C13~15 좌단 진단": r"좌단만|후보를? 절반|절반만 잡",
    "C17·C18·C32 방위 H3": r"치우쳐 ?있었|분모.{0,15}기록이 없|판정은 \*?\*?보류",
    "C24 검출력 옛 값": r"74\.5 ?%|25\.5 ?%|0\.43칸 이상",
    "C25 호 차이 상한": r"≤ ?0\.02 ?%p",
    "C28 우연 기준선": r"우연 기준선|우연 25",
    "C05~C12 판정어": r"슬롯 (무관|과 무관)|0 근처|전멸 신호|확인된 칸|확인 0개",
    "C16 E 혼합": r"0\.11 ?%.{0,12}몰|반대 조건이 (실제로 )?일어",
    "C33 반경·폭": r"반경·폭 ?(일관)?은 관측|반경·폭이 lot 안에서 일관",
    "C34 H7 과잉": r"H5.{0,6}H7.{0,30}⬜",
    "D-041·042 초록": r"겹침 구간|더 추가해도|더 넣어도|왜 다른 종류",
    "R3-04 방위 현행값": r"현행.{0,25}71\.70",
    "R9-02 수치 불변 총괄": r"한 칸도 안 바뀌|수치 변동 0|세 판 모두",
    "R9-04 발동 추론": r"발동하지 않았다|발동 조건이.{0,12}없",
    "C19 Random 최대": r"Random.{0,15}(최대|제일 컸)",
    # ↓ 1차 사전을 돌리고 pattern_process_mapping §2-1에서 **사전 밖 잔존**을 보고 더했다 (D-056).
    #   13차 축(C01~C34)만 넣었더니 그 이전 계열(D-025·D-028·D-033·D-041)이 통째로 빠져 있었다.
    "D-033 이동 횟수": r"(세|두) 번 이동|세 번의 이동",
    "D-025·026 경계 부재": r"임계값이 (없|존재하지)|경계가 없",
    "D-028 집계=기전": r"Edge-Loc`? ?= ?`?Loc`? ?\+",
    "D-041 풀렸다": r"특징을 만들면 풀렸|두 병목이 (해소|풀렸)|완전히 풀렸",
}
MARK = re.compile(r"⛔|~~|원판 보존|정정|철회|금지|❌|인용하지 않|쓰지 않|옛 |원래 ")
CTX = 0                                              # 같은 줄만. 위 docstring


def targets(paths=None):
    if paths:                                       # 검증용: 임의 폴더 (예: 13차 키트)
        for d in map(Path, paths):
            yield from sorted(x for x in d.rglob("*") if x.suffix in (".md", ".py"))
        return
    for sub in ("docs", "src"):
        yield from sorted((ROOT / sub).rglob("*.md" if sub == "docs" else "*.py"))
    up = ROOT.parent
    if (up / "CLAUDE.md").exists():
        yield up / "CLAUDE.md"
    if (up / "planning").exists():
        yield from sorted((up / "planning").rglob("*.md"))


def scan(show_all=False, paths=None):
    pats = {k: re.compile(v) for k, v in FAMILIES.items()}
    counts = {k: [0, 0] for k in FAMILIES}          # [전체, 표지 없음]
    me = Path(__file__).resolve()
    for f in targets(paths):
        if f.resolve() == me:
            continue
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, ln in enumerate(lines):
            for k, p in pats.items():
                if not p.search(ln):
                    continue
                ctx = "\n".join(lines[max(0, i - CTX): i + CTX + 1])
                marked = bool(MARK.search(ctx))
                counts[k][0] += 1
                counts[k][1] += not marked
                if show_all or not marked:
                    rel = f.relative_to(ROOT.parent) if ROOT.parent in f.parents else f
                    print(f"[{'표지' if marked else '없음'}] {k} | {rel}:{i + 1} | {ln.strip()[:110]}")
    print("\n계열별 적중 (전체 / 표지 없음)")
    for k, (a, b) in counts.items():
        print(f"  {k:<22} {a:>4} / {b}")
    return counts


def demo():
    pats = {k: re.compile(v) for k, v in FAMILIES.items()}
    hit = lambda s: [k for k, p in pats.items() if p.search(s)]
    # 13차에서 놓친 세 형태를 반드시 잡아야 한다
    assert "C01·C03 shape≠제품" in hit("lot 10,762개 · 제품(맵 shape) 346종")
    assert "C01·C03 shape≠제품" in hit("| **제품 고정 귀무** | **제품 고정 배수** |")
    assert "C01·C03 shape≠제품" in hit("그래서 「lot 효과」와 「제품 효과」가 이 자료에서 안 갈린다")
    assert "C02 deff≠라벨 군집" in hit("`deff`=16.20 — §7 누수 97.0%와 같은 군집")
    assert "D-041·042 초록" in hit('정확한 서술은 "특징을 더 넣어도 겹침 구간은 남는다"')
    assert "R3-04 방위 현행값" in hit("현행은 전수값 `71.70 / 20.07 / 24.95 / 9.67%`이고")
    # 사전 밖에서 찾은 잔존 (pattern_process_mapping §2-1, D-056)
    assert "D-033 이동 횟수" in hit("### 결론 — 병목이 세 번 이동했고, 마지막 하나는")
    assert "D-025·026 경계 부재" in hit("**구배가 연속이라 임계값이 없다.**")
    assert "D-028 집계=기전" in hit("즉 **`Edge-Loc` = `Loc` + 가장자리 구배.**")
    assert "D-041 풀렸다" in hit("1~3단계 병목은 전부 (a) 특징 부재였고, 특징을 만들면 풀렸다.")
    # 음성: 정정된 현행 문장은 이 계열에 안 걸려야 한다
    assert not hit("lot 내 동시발생이 맵 shape과 강하게 얽혀 있고")
    # 표지 판정
    assert MARK.search("(⛔ 원래 ~~「제품 효과」~~)") and not MARK.search("제품 효과가 안 갈린다")
    print("demo OK")


if __name__ == "__main__":
    demo()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    scan(show_all="--all" in sys.argv, paths=args or None)
