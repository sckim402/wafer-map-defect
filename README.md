# 웨이퍼맵 불량 패턴의 공정 원인 규명

WM-811K 웨이퍼맵 데이터셋에서 불량 패턴을 분류하고, **오분류가 발생하는 지점을
공정 물리로 설명**하는 것을 목표로 한다.

**이 프로젝트의 목표는 분류 정확도가 아니다.**

## 주장

> 웨이퍼맵 불량 패턴 분류의 주요 오분류는 원인 공정이 다른 패턴들이 유사한
> 공간 분포를 갖는 데서 발생한다. 원인 공정의 대칭성 차이를 반영한
> **회전 불변 방위각 균일성 지표**가 이 오분류를 줄이는지 검증한다.

Edge-Ring과 Edge-Loc은 둘 다 웨이퍼 가장자리에 나타나지만 원인 공정의 대칭성이
다르다. 챔버 전체의 조건 편차(gas flow, plasma density, 척 온도 구배)는 방위각
방향으로 **고르게** 나타나고, 국부적 접촉·오염은 한 방향에 **몰린다.**

## 방법 요약

| 항목 | 내용 |
|---|---|
| 분류 대상 | 패턴 8종 (`none` 제외) |
| 분할 | `lotName` 기준 GroupShuffleSplit (+ 원저자 분할 병행) |
| 평가지표 | per-class F1, 대상 혼동 쌍의 상호 오분류율 (**accuracy 사용 안 함**) |
| 특징 | F1 방사형 프로파일 / F2 기하 / **F3 방위각 균일성 (회전 불변)** |
| 검증 | 헤드라인 대조 **F1+F3 vs F1**, 반복 실험 평균±편차 |

### 왜 회전 불변인가

WM-811K에는 **notch/flat 방향 정보가 없다.** 웨이퍼맵의 0°가 물리적으로 무엇인지
알 수 없고, 웨이퍼들이 공통 좌표계로 정렬되어 있다는 보장도 없다. 따라서 절대
각도 히스토그램은 검증 불가능한 가정 위에 서게 된다.

대신 "**몰렸는가 퍼졌는가**"만 본다 — 가중 circular variance가 대표 지표다.
좌표계 정렬 여부는 특징이 아니라 **별도 진단**으로 다룬다
(`docs/coordinate_alignment.md`).

## 구조

    src/config.py       전역 설정 (경로, seed, 클래스)
    src/load_data.py    LSWMD.pkl -> 클래스별 npz 변환 + 기초 통계
    src/explore.py      클래스별 대표 샘플 시각화
    docs/decisions.md   설계 판단 기록  <- 이 프로젝트의 핵심 산출물
    data/               원본·전처리 데이터 (git 제외)
    figures/            그림 (keep/ 안의 것만 git 추적)

## 실행

    python -m venv .venv
    .venv\Scripts\activate          # Windows
    pip install -r requirements.txt

    # data/README.md 를 보고 LSWMD.pkl 을 data/ 에 배치한 뒤
    python src/load_data.py
    python src/explore.py

## 데이터

WM-811K (LSWMD) — 실제 fab에서 수집된 웨이퍼맵 811,457장, 그중 172,950장에
9종 불량 패턴 라벨이 붙어 있다. 출처는 `data/README.md` 참조.

## 기록 원칙

**모든 설계 판단은 `docs/decisions.md`에 이유와 함께 남긴다.**
근거 없이 내려진 결정은 3개월 뒤 설명할 수 없고, 설명할 수 없는 결정은
그 사람의 것이 아니다.
