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

    # ── 4차 스키마(공정 물리): 인용 + **예측·반증이 서로 달라야** 통과 ──
    if "attributions" in r:
        doc = "docs/pattern_process_mapping.md"
        db = body(doc)
        if db is None:
            raise SystemExit(f"[중단] {kit/doc} 없음")
        rows, kept, mem = [], 0, 0
        for a in r["attributions"]:
            q = find(a.get("doc_quote", ""), db)
            tp = (a.get("testable_prediction") or "").strip()
            fl = (a.get("falsifier") or "").strip()
            pair_ok = bool(tp) and bool(fl) and norm(tp) != norm(fl)
            st = a.get("source_type", "?")
            mem += (st == "기억")
            ok = q in ("exact", "norm") and pair_ok
            kept += ok
            rows.append((a.get("id", "?"), a.get("pattern", "?"), a.get("verdict", "?"),
                         st, q, pair_ok, a.get("computable_now"), ok,
                         (a.get("assessment", "") or "")[:30]))
        print(f"  {'id':<4}{'패턴':<12}{'판정':<7}{'출처':<7}{'인용':<10}"
              f"{'예측≠반증':<10}{'계산가능':<9}{'판정':<6}평가")
        print("  " + "-" * 94)
        for i, pat, vd, st, q, pk, cn, ok, asm in rows:
            print(f"  {str(i):<4}{pat:<12}{vd:<7}{st:<7}{q:<10}"
                  f"{'O' if pk else 'X':<10}{str(cn):<9}{'OK' if ok else 'BUY':<6}{asm}")
            if not ok:
                why = ([] if q in ("exact", "norm") else ["인용 " + q]) +                       ([] if pk else ["예측/반증이 없거나 동일"])
                print(f"       ^ 버린다 ({', '.join(why)})")
        n = len(rows)
        print("")
        print(f"  귀속 {n}건 중 검증 통과 {kept}건 / 폐기 {n - kept}건")
        print(f"  출처 구분: 기억 {mem}건 / 논문·교재 {n - mem}건  "
              f"<- 논문·교재는 DOI·서지를 직접 확인할 것")
        for key, lab in (("missing_causes", "빠진 원인"),
                         ("physically_confusable", "물리적으로 겹치는 쌍"),
                         ("what_the_images_show", "그림 관찰")):
            v = r.get(key, [])
            print(f"  {lab} {len(v)}건")
            for e in v[:4]:
                head = e.get("pattern") or e.get("pair") or "?"
                body_ = e.get("cause") or e.get("why") or e.get("observation") or ""
                print(f"    [{head}] {str(body_)[:64]}")
        cn_ok = sum(1 for a in r["attributions"] if a.get("computable_now"))
        print(f"  지금 우리 데이터로 잴 수 있는 예측 {cn_ok}/{n}건")
        fq = r.get("first_question")
        if fq:
            print(f"  첫 질문: {fq}")
        return kept, n

    # ── 3차 스키마(구현 전수 조사): 인용 + **재현 코드**를 함께 요구한다 ──
    if "degeneracies" in r:
        rows, kept = [], 0
        for it in r["degeneracies"]:
            ef = it.get("evidence_file", "")
            eb = body(ef) if ef else None
            q = "파일없음" if (ef and eb is None) else (
                find(it.get("code_quote", ""), eb) if eb is not None else "MISS(미기재)")
            has_repro = bool((it.get("repro_code") or "").strip())
            has_obs = bool((it.get("observed") or "").strip())
            ok = q in ("exact", "norm") and has_repro and has_obs
            kept += ok
            rows.append((it.get("id", "?"), it.get("kind", "?"), it.get("severity", "?"),
                         q, has_repro, has_obs, ok, ef,
                         (it.get("symptom", "") or "")[:34]))
        print(f"  {'id':<4}{'종류':<11}{'등급':<8}{'인용':<10}{'재현코드':<9}{'관찰값':<8}{'판정':<6}증상")
        print("  " + "-" * 88)
        for i, kind, sev, q, hr, ho, ok, ef, sym in rows:
            print(f"  {str(i):<4}{kind:<11}{sev:<8}{q:<10}{'있음' if hr else '없음':<9}"
                  f"{'있음' if ho else '없음':<8}{'OK' if ok else 'BUY':<6}{sym}")
            if not ok:
                why = []
                if q not in ("exact", "norm"): why.append("인용 " + q)
                if not hr: why.append("재현 코드 없음")
                if not ho: why.append("관찰값 없음")
                print(f"       ^ 버린다 ({', '.join(why)}) {ef}")
        n = len(rows)
        print("")
        print(f"  결함 {n}건 중 검증 통과 {kept}건 / 폐기 {n - kept}건")
        cc = r.get("checked_but_clean", [])
        print(f"  안 걸렸다고 보고한 파일 {len(cc)}개 / 계열 밖 지적 {len(r.get('out_of_class', []))}건")
        if not cc:
            print("  ** checked_but_clean이 비었다 — 커버리지를 알 수 없으므로 "
                  "「결함 0건」을 근거로 쓰지 않는다 **")
        else:
            src_n = len([f for f in (kit / "src").glob("*.py")]) if (kit / "src").is_dir() else 0
            seen = len({c.get("file", "") for c in cc})
            print(f"     -> src 전체 {src_n}개 중 {seen}개에 대해 확인 근거가 있다")
        fq = r.get("first_question")
        if fq:
            print(f"  첫 질문: {fq}")
        return kept, n

    # ── 2차 스키마(설계 검토): design_flaws는 근거 인용만 대조한다 ──
    if "design_flaws" in r and "overstatements" not in r:
        rows, kept = [], 0
        for it in r["design_flaws"]:
            ef = it.get("evidence_file", "")
            eb = body(ef) if ef else None
            e = "파일없음" if (ef and eb is None) else (
                find(it.get("evidence_quote", ""), eb) if eb is not None else "MISS(미기재)")
            ok = e in ("exact", "norm")
            kept += ok
            rows.append((it.get("id", "?"), it.get("item", "?"), it.get("aspect", "?"),
                         it.get("severity", "?"), e, ok, (it.get("problem", "") or "")[:38], ef))
        print(f"  {'id':<4}{'건':<4}{'측면':<12}{'등급':<8}{'근거':<10}{'판정':<6}지적")
        print("  " + "-" * 80)
        for i, item, asp, sev, e, ok, prob, ef in rows:
            print(f"  {str(i):<4}{str(item):<4}{asp:<12}{sev:<8}{e:<10}{'OK' if ok else 'BUY':<6}{prob}")
            if not ok:
                print(f"       ^ 버린다 ({ef or '근거파일 미기재'})")
        n = len(rows)
        print("")
        print(f"  설계 지적 {n}건 중 근거 검증 통과 {kept}건 / 폐기 {n - kept}건")
        if n and kept == 0:
            print("  ** 전부 MISS — 파일을 읽지 않았다. 결과 전체를 폐기한다 **")
        ex = r.get("executed", [])
        print(f"  실제 실행 {len(ex)}건 / 빠진 대조군 {len(r.get('missing_controls', []))}건 "
              f"/ 답 못하는 질문 {len(r.get('unanswerable', []))}건")
        for e_ in ex:
            print(f"    [건{e_.get('item','?')}] {str(e_.get('result',''))[:70]}")
        fq = r.get("first_question")
        if fq:
            print(f"  첫 질문: {fq}")
        return kept, n

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
