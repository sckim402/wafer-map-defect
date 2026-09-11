"""외부 AI 검토 결과(JSON)의 인용을 원문과 대조한다.

    ./.venv/Scripts/python.exe src/verify_review.py <결과.json> [--kit ../ai_review_kit]
    ./.venv/Scripts/python.exe src/verify_review.py --demo      # 자체 검사

왜 필요한가 (전역 규칙 「외부 자료」 / `paper-source` 스킬):
    AI는 **읽지 않은 파일에 대해서도 그럴듯한 지적을 만든다.** 그것도 「못 읽었다」는
    신호 없이. 그래서 지적의 내용을 읽기 전에 **그 인용이 실제로 그 파일에 있는지**
    부터 본다.

판정:
    exact  원문에 글자 그대로 있다            -> 검토한다
    norm   공백·줄바꿈만 다르다               -> 검토한다 (표기 차이)
    MISS   어떤 형태로도 없다                 -> **지어낸 것. 그 항목을 버린다**

    한 검토자의 항목이 전부 MISS면 그 검토자는 파일을 안 읽은 것이다 -> 결과 전체 폐기.
"""
import json
import re
import sys
from pathlib import Path

KIT_DEFAULT = Path(__file__).resolve().parents[2] / "ai_review_kit"
TARGET = "docs/method_overview.md"


def norm(s):
    """공백류를 하나로 접는다. 마크다운 강조(**)도 뗀다 — 인용 시 자주 흘린다."""
    return re.sub(r"\s+", " ", s.replace("*", "")).strip()


def find(quote, text):
    if not quote:
        return "MISS(빈 인용)"
    if quote in text:
        return "exact"
    if norm(quote) and norm(quote) in norm(text):
        return "norm"
    return "MISS"


def verify(result_path, kit):
    r = json.loads(Path(result_path).read_text(encoding="utf-8"))
    cache = {}

    def body(rel):
        if rel not in cache:
            p = kit / rel
            cache[rel] = p.read_text(encoding="utf-8") if p.is_file() else None
        return cache[rel]

    who = r.get("reviewer", {})
    print(f"검토자 {who.get('model','?')} / {who.get('date','?')}")
    print(f"연 파일 {len(r.get('files_read', []))}개\n")

    rows, kept = [], 0
    tgt = body(TARGET)
    if tgt is None:
        raise SystemExit(f"[중단] {kit/TARGET} 없음. --kit 경로를 확인하라")

    for it in r.get("overstatements", []):
        q = find(it.get("quote", ""), tgt)
        ef = it.get("evidence_file", "")
        eb = body(ef) if ef else None
        e = "파일없음" if (ef and eb is None) else (
            find(it.get("evidence_quote", ""), eb) if eb is not None else "MISS(미기재)")
        ok = q in ("exact", "norm") and e in ("exact", "norm")
        kept += ok
        rows.append((it.get("id", "?"), it.get("severity", "?"), q, e, ok,
                     (it.get("problem", "") or "")[:44], ef))

    print(f"  {'id':<4}{'등급':<8}{'인용':<7}{'근거':<10}{'판정':<6}지적")
    print("  " + "-" * 76)
    for i, sev, q, e, ok, prob, ef in rows:
        print(f"  {str(i):<4}{sev:<8}{q:<7}{e:<10}{'OK' if ok else 'BUY':<6}{prob}")
        if not ok:
            print(f"       ^ 버린다 ({ef or '근거파일 미기재'})")

    n = len(rows)
    print(f"\n  지적 {n}건 중 인용 검증 통과 {kept}건 / 폐기 {n - kept}건")
    if n and kept == 0:
        print("  ** 전부 MISS — 이 검토자는 파일을 읽지 않았다. 결과 전체를 폐기한다 **")
    print(f"  반박 측정 {len(r.get('refutation_tests', []))}건 / "
          f"재구현 부족 {len(r.get('missing_for_reproduction', []))}건")
    fq = r.get("first_question")
    if fq:
        print(f"  첫 질문: {fq}")
    return kept, n


def demo():
    """자체 검사 — 있는 인용은 통과하고, 지어낸 인용은 MISS여야 한다."""
    txt = "**주 분할 0.837 / 외부 검증 0.704**(항상 병기). 마지막 병목만 성격이 다르다."
    assert find("주 분할 0.837 / 외부 검증 0.704", txt) == "exact"
    assert find("주 분할 0.837 /   외부 검증 0.704", txt) == "norm"   # 공백 차이
    assert find("주 분할 0.837 / 외부 검증 0.704", txt.replace("**", "")) == "exact"
    assert find("macro-F1은 0.92에 도달했다", txt) == "MISS"           # 지어낸 문장
    assert find("", txt).startswith("MISS")
    print("demo 통과 — exact/norm/MISS 판정이 의도대로 갈린다")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--demo":
        demo()
    else:
        kit = Path(a[a.index("--kit") + 1]) if "--kit" in a else KIT_DEFAULT
        verify(a[0], kit)
