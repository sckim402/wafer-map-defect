# W6 — 문헌 1차 스캔 (2026-09-10)

**목적은 신규성 확인이 아니다.** 로드맵 v4.5 §W6에서 좁힌 대로
**「이 분야가 당연히 요구하는 대조 중 내가 빠뜨린 게 있나」** 하나다.
읽을 것을 ①분할 정책 ②보고 지표 ③교란 통제 **셋으로 못박고** 들어갔다.

> **왜 이 스캔이 필요했나**: D-021로 외부 검토가 0회가 됐다(10/23까지).
> §3-14가 잡는 종류 — *계산 실수가 아니라 안 떠올린 질문* — 을 잡을 수 있는
> **유일한 외부 소스**가 선행 논문의 방법론 절이다. 이것이 D-021 대체 장치 ②다.

---

## 0. 방법 — 요약문이 아니라 원문을 읽었다

첫 시도는 웹 검색 요약이었고 **버렸다.** 검색 엔진 요약이
*"lot-group split을 쓴다"*를 출처 불명으로 서술했는데, 확인해보니
**그 문장은 7편 중 1편에만 있었다.** 요약이 여러 논문을 한 문단에 섞은 것이다.

→ **PDF를 내려받아 PyMuPDF로 전문 추출 후 정규식으로 방법론 절을 뽑았다.**
`split | lot | stratif | leak | k-fold`, `macro-F1 | per-class | balanced accuracy |
accuracy of`, `defect density | number of defect | die count | wafer size | confound`
세 패턴, 히트마다 전후 160자.

**이하의 모든 인용은 원문 추출 텍스트에서 나온 것이다.** 못 읽은 것은 §5에 적었다.

---

## 1. 읽은 것 — 7편

| # | 논문 | 연도 | 성격 |
|---|---|---|---|
| A | Architecture-Aware Explanation Auditing for Industrial Visual Inspection (arXiv 2605.14255) | 2026 | XAI 감사 · preprint |
| B | Wafer Map Defect Classification Using Autoencoder-Based Data Augmentation and CNN (arXiv 2411.11029) | 2024 | 증강+CNN |
| C | Wafer map failure pattern classification using geometric transformation-invariant CNN (Sci Rep 13:8127) | 2023 | 회전·플립 불변 CNN |
| D | A Controlled CV-versus-DV Comparison on Wafer-Map Defect Classification (arXiv 2607.00961) | 2026 | 양자 회로 비교 |
| E | Wafer Map Defect Patterns Semi-Supervised Classification Using Latent Vector (arXiv 2311.12840) | 2023 | 준지도 |
| F | A novel approach for wafer defect pattern classification based on TDA (arXiv 2209.08945) | 2022 | 위상 데이터 분석 |
| G | Iterative Cluster Harvesting for Wafer Map Defect Patterns (arXiv 2404.15436) | 2024 | 비지도 군집 |

---

## 2. ① 분할 정책 — **lot 단위는 표준이 아니다**

| 논문 | 분할 | lot 언급 |
|---|---|---|
| **A** | **lot-group split 70/15/15** | ✅ **명시** |
| B | 랜덤 4:1 | ⚠ **데이터 설명에서만** |
| C | 크기별 train set(최대 6400) | ❌ **분할 정책 서술 자체가 없음** |
| D | stratified split, 증강 전 분리 | ❌ |
| E | (미기재) | ❌ |
| F | 300/100/100 (합성 500장/클래스) | ❌ |
| G | 비지도 — 분할 없음 | ❌ |

**A의 원문**:
> "The dataset is partitioned into training, validation, and test sets in a 70/15/15
> ratio using a **lot-group split**. Under this scheme, all wafers originating from the
> same manufacturing lot are confined to a single partition… Lots are identified using
> the **`lotName` field** in the WM-811K metadata… This choice addresses the
> **information-leakage risk** that arises from the correlation between nearby wafers
> in the same lot: a simple random split would allow the model to exploit lot-specific
> defect patterns shared between training and test samples, **inflating apparent test
> performance**."

**→ D-003과 같은 논거, 같은 필드(`lotName`)다.** 우리 판단이 재발명이 아니라는
외부 확인이 하나 생겼다.

### 2-A. ★ 더 센 발견은 B다

B는 **lot 구조를 알고 있다**:
> "The dataset was collected from **47,543 physical lots** from a FAB, with each lot
> consisting of 25 wafers."

**그러고도 분할은 랜덤 4:1이다.** *"lot이 뭔지 몰라서 안 한 것"*이 아니라
**알면서 분할에 쓰지 않았다.** 헤드라인은 **accuracy 98.56%**다.

> **이것이 이 스캔의 1번 수확이다.** 7편 중 lot 단위 분할은 **1편(A, 2026 preprint)**.
> **lot 단위 분할은 이 분야의 표준이 아니고, 우리는 소수파다.**
> D-003은 성능이 아니라 누수 논거로 정한 것이므로 **바꿀 이유가 없다.**
> 다만 **「표준을 따랐다」고 쓰면 거짓이다** — 정확한 서술은
> **「lot 단위 분할은 소수파이며, 우리는 누수 근거로 그쪽을 택했다」**이다.

### 2-B. C는 분할 정책을 아예 안 적었다

`split|lot|stratif|leak|fold` **히트 0건.** Scientific Reports 게재본에서
train/test 구성 방식이 서술되지 않는다. **재현 가능성의 하한이 이 정도다.**

---

## 3. ② 보고 지표 — **accuracy 헤드라인이 여전히 다수**

| 논문 | 헤드라인 | per-class | accuracy 반대 논거 |
|---|---|---|---|
| A | **balanced acc + macro-F1** | ✅ per-class F1 (radar) | ✅ **명시** |
| B | **accuracy 98.56%** | 혼동행렬 대각만 | ❌ |
| C | **accuracy** | 클래스별 accuracy 차이 | ❌ |
| D | **accuracy 85.0% (macro-F1 0.798)** | ✅ per-class recall | ✅ **명시** |
| E | Precision/Recall/F1/Accuracy 4종 | ❌ | ❌ |
| F | **accuracy 99.0%** | 혼동행렬 | ❌ |
| G | 혼동행렬 (군집) | ✅ | — |

**A의 원문**:
> "Because the majority class alone exceeds four-fifths of the labelled data, **raw
> accuracy is an uninformative summary of model behaviour**; this study therefore
> adopts balanced accuracy and macro-F1 as primary performance metrics."

### 3-A. ★ D는 우리 결론과 같은 문장을 쓴다

> "**per-class recall on morphologically ambiguous classes can matter more for yield
> management than aggregate accuracy.** A classifier with high aggregate accuracy that
> systematically [misroutes a class]…"

그리고 D가 지목한 최대 실패 클래스는 **`Edge-Loc`**이며,
**`Scratch`로 81% 오분류**된다고 보고한다. `Loc`은 *"an intrinsically hard class
that no head solves"*.

> **우리 병목과 대상이 겹친다** — 우리는 `Loc↔Scratch`(2단계) ·
> `Edge-Loc↔Loc`(3단계). **D는 `Edge-Loc→Scratch`.**
> 세 클래스가 같은 삼각형 안에 있다. **다른 방법론(양자 회로)·다른 특징에서도
> 같은 클래스 묶음이 병목으로 나온다** — 우리 경로 의존이 아닐 가능성을 지지하는
> **외부 정황**이다. ⚠ **정황이지 증거가 아니다** — D는 5클래스 구성·다른 전처리다.
> **§3-14대로 「이 투입 순서에서의 관찰」을 떼는 근거로 쓰지 않는다.**

**판정**: per-class 보고 자체는 드물지 않다(7편 중 4편).
**「accuracy를 주 지표로 쓰지 않는다」는 소수(2편)**다.
**우리 방침(§7 accuracy 금지 + per-class F1 + 혼동 쌍)은 유지한다.**

---

## 4. ③ 교란 통제 — **대응물을 못 찾았다**

`defect density | number of defect | defect count | failed die | die count |
wafer size | confound` 패턴:

- **B·C·D·F·G: 히트 0건.**
- E: 1건 — *"N represents the number of defect **types**"* (무관).
- A: 3건 — 전부 **모델 정확도 차이를 교란으로 인정**한다는 서술이고,
  **결함 개수·웨이퍼 크기 교란이 아니다.**

**즉 7편 중 「결함 개수」나 「웨이퍼 크기」를 교란 변수로 놓고 통제한 사례가 0이다.**

| 우리가 한 것 | 선행 대응물 |
|---|---|
| F3 AUC 0.927 → **개수 통제 후 0.743** (§3-2) | **없음** |
| `size` 증분 +0.111 중 **84.3%가 F1a·F1b의 반경 정보** (`w4_size.md`) | **없음** |
| `none` sanity 귀무값 순열 생성 (§3-13) | **없음** |

> ⚠ **이 결론의 한계를 정확히 적는다.** *"선행에 없다"*가 아니라
> **"내가 읽은 7편에 없다"**이다. 7편은 3~5편 상한을 넘긴 표본이지만
> 이 분야 전수가 아니다. **「최초」라고 쓰지 않는다.**
> 쓸 수 있는 것은 **「비교 대상으로 삼은 7편에는 개수·크기 교란 통제가 없었다」**뿐이다.

---

## 5. ★ 실제 수확 — **내가 빠뜨린 대조 1건을 찾았다**

이 스캔의 유일한 목적이 이것이었다. **찾았다.**

### 5-A. 회전·플립 불변성을 **주장만 하고 측정한 적이 없다**

C(Sci Rep 2023)는 회전·플립 불변성을 **설계로 주장하고 끝내지 않는다.**
**회전·플립을 가한 unseen test set을 따로 만들어 성능 유지를 측정한다.**
클래스별로 회전·플립 분산이 다르다는 것까지 보고한다.

**우리 쪽 상태** (`grep` 결과):

| 위치 | 문장 |
|---|---|
| `src/azimuth.py:20` | *"회전 불변량인 circular variance만 쓴다"* |
| `src/edge_band.py:90` | *"회전 불변량이므로 notch 방향 정보가 없어도 유효하다"* |
| `src/radial.py:46` | *"이 파일의 특징 2종 (회전 불변)"* |
| `src/shape_feats.py:22` | *"회전 불변만 쓴다"* |
| `docs/coordinate_alignment.md:48` | *"F3는 이미 회전 불변 지표로 설계했으므로"* |

**5개 파일에서 주장한다. 측정한 기록은 0건이다.**

> **이것이 §3-14가 말한 「안 떠올린 질문」의 실물이다.** 설계상 불변이므로
> 불변일 것이라고 **가정**했고, 그 가정을 **한 번도 실행으로 확인하지 않았다.**
> §3-3(*"새 방법은 실패 조건에서 먼저 돌린다"*)을 특징 6종 전체에는 적용하지 않은 것이다.

**왜 이게 값싼 검사인가**: 웨이퍼 맵은 다이 격자다. **90°·180°·270° 회전과
좌우/상하 플립은 격자 정렬이라 리샘플링이 없다** — 보간 오차가 원천적으로 0이다.
따라서 **귀무값이 정확히 계산된다(§3-13)**:

> **6종 특징값은 8개 이면군(dihedral) 변환 전부에서 부동소수점 오차 내로 동일해야 한다.**
> 하나라도 어긋나면 **불변성 주장이 틀렸거나 구현에 버그가 있다.**
> *"직관으로 정한 합격선"*이 아니라 **계산된 귀무값**이다.

**예상 소요 30분~1시간.** §3-3의 상한(30분~2시간) 안이다.

### 5-B. 등급이 낮은 후보 2건 (기록만)

- **train set 크기별 성능 곡선** (C가 보고) — 우리는 없다. **등급 중.**
  전이 실험·외부 검증이 이미 있어 우선순위가 밀린다
- **부분집합 선택의 자의성** — 검색 요약에 *"902장(라벨 0.5%)만 쓴 연구"* 언급이
  있었으나 **출처가 언론 보도자료라 인용하지 않는다.** 원 논문 미확인

---

## 6. 못 읽은 것 — 명시한다

| 대상 | 상태 |
|---|---|
| **Wu et al. 2015 (IEEE TSM 28(1):1-12)** — WM-811K 원논문 | ❌ **전문 미확보.** IEEE Xplore가 스크립트 접근을 차단(HTTP 응답이 `<script>`), Semantic Scholar API는 429. **분할 정책·지표를 확인하지 못했다** |
| PLOS ONE / MDPI / Nature 웹판 | ❌ 403 또는 인증 리다이렉트 |

> **Wu 2015가 남은 가장 큰 공백이다.** 원논문의 밀도 기반 13특징은
> *"영역별 결함 밀도"*이므로 **면적 정규화를 이미 하고 있을 가능성**이 있고,
> 그렇다면 §4의 판정(개수 교란 통제 0건)이 **부분적으로 달라진다.**
> **§4를 「7편 중 0건」으로만 쓰고 「선행에 없다」로 쓰지 않은 이유가 이것이다.**
> 도서관 프록시로 받을 수 있으면 그때 이 절을 갱신한다.

---

## 7. 판정 요약

| 질문 | 답 | 우리 대응 |
|---|---|---|
| ① lot 단위 분할이 표준인가 | ❌ **7편 중 1편.** 소수파다 | **D-003 유지.** 단 *"표준을 따랐다"*고 쓰지 않는다 |
| ② accuracy가 주 지표인가 | ⚠ **여전히 다수(7편 중 5편).** per-class 병기는 4편 | **§7 accuracy 금지 유지** |
| ③ 개수·크기 교란을 통제하는가 | ❌ **7편 중 0건** (Wu 2015 미확인) | **우리 쪽이 더 촘촘하다.** 단 *"최초"* 금지 |
| ★ 내가 빠뜨린 대조가 있나 | ✅ **있다 — 회전·플립 불변성 실측** | **§5-A. 실행 대상** |

**초록에 미치는 영향: 0.** 로드맵 v4.5 ④가 정정한 대로다 —
참고문헌 추가가 물리적으로 불가하고(1건=2페이지), 초록에 신규성 주장이 없다.
**이 스캔의 결과로 초록의 어떤 문장도 바뀌지 않는다.**

---

*2026-09-10 작성. 소요 약 1시간. 원문 추출 스크립트와 PDF는 스크래치패드에만 두었다
(리포에 넣지 않음 — 저작권).*
