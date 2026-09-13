# W6 — 문헌 1차 스캔 (2026-09-10)

**목적은 신규성 확인이 아니다.** 로드맵 v4.5 §W6에서 좁힌 대로
**「이 분야가 당연히 요구하는 대조 중 내가 빠뜨린 게 있나」** 하나다.
읽을 것을 ①분할 정책 ②보고 지표 ③교란 통제 **셋으로 못박고** 들어갔다.

> **왜 이 스캔이 필요했나**: D-021로 외부 검토가 0회가 됐다(10/23까지).
> §3-14가 잡는 종류 — *계산 실수가 아니라 안 떠올린 질문* — 을 잡을 수 있는
> **유일한 외부 소스**가 선행 논문의 방법론 절이다. 이것이 D-021 대체 장치 ②다.

> **⚠ 개정 이력**
> - **v1 (오전)**: 7편. **Wu et al. 2015 원논문은 미확보**로 남겼다
> - **v2 (오후, 이 문서)**: **원논문 전문 입수 → 8편.**
>   **v1의 §4·§5·§6을 정정한다.** 무엇이 뒤집혔는지는 §5-0과 §6-0에 적었다

---

## 0. 방법 — 요약문이 아니라 원문을 읽었다

첫 시도는 웹 검색 요약이었고 **버렸다.** 검색 엔진 요약이
*"lot-group split을 쓴다"*를 출처 불명으로 서술했는데, 확인해보니
**그 문장은 7편 중 1편에만 있었다.** 요약이 여러 논문을 한 문단에 섞은 것이다.

→ **PDF를 내려받아 PyMuPDF로 전문 추출 후 정규식으로 방법론 절을 뽑았다.**
`split | lot | stratif | leak | k-fold`, `macro-F1 | per-class | balanced accuracy |
accuracy of`, `defect density | number of defect | die count | wafer size | confound`
세 패턴, 히트마다 전후 160자. 히트가 걸린 절은 **원문 문단째로 다시 읽었다.**

**이하의 모든 인용은 원문 추출 텍스트에서 나온 것이다.**

> **PDF는 리포에 넣지 않는다** — IEEE 구독본이다. `.gitignore`에 `docs/*.pdf`를 추가했다.
> **서지 정보와 인용문만 이 문서에 남긴다.**

---

## 1. 읽은 것 — 8편

| # | 논문 | 연도 | 성격 |
|---|---|---|---|
| **W** | **Wu, Jang, Chen — Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets** (IEEE TSM **28**(1):1-12, DOI `10.1109/TSM.2014.2364237`) | **2015** | ★ **WM-811K 원논문.** 피인용 377 |
| A | Architecture-Aware Explanation Auditing for Industrial Visual Inspection (arXiv 2605.14255) | 2026 | XAI 감사 · preprint |
| B | Wafer Map Defect Classification Using Autoencoder-Based Data Augmentation and CNN (arXiv 2411.11029) | 2024 | 증강+CNN |
| C | Wafer map failure pattern classification using geometric transformation-invariant CNN (Sci Rep 13:8127) | 2023 | 회전·플립 불변 CNN |
| D | A Controlled CV-versus-DV Comparison on Wafer-Map Defect Classification (arXiv 2607.00961) | 2026 | 양자 회로 비교 |
| E | Wafer Map Defect Patterns Semi-Supervised Classification Using Latent Vector (arXiv 2311.12840) | 2023 | 준지도 |
| F | A novel approach for wafer defect pattern classification based on TDA (arXiv 2209.08945) | 2022 | 위상 데이터 분석 |
| G | Iterative Cluster Harvesting for Wafer Map Defect Patterns (arXiv 2404.15436) | 2024 | 비지도 군집 |

---

## 1-A. ★ WM-811K의 출처 — **이 논문이 만들었다** (2026-09-13 원문 재확인)

*"초록의 `WM-811K[1]` 각주가 맞나"*를 물어 **원문에서 직접 뽑았다**(전역 규칙: 요약을 인용 근거로 쓰지 않는다).

> **기여 목록 2번**: *"The WM-811K dataset**¹** **developed in this study** is the largest known
> wafer map data set available to the public."*  — 각주 1 = `http://mirlab.org/dataSet/public/`
>
> **초록**: *"the WM-811K dataset **was built** comprising 811 457 wafer maps, in which each wafer map
> was collected from real-world fabrication."*
>
> **§VI-A**: *"comprises 811 457 wafer maps that were collected from **46 293 lots** in real-world fabrication."*

**→ 데이터셋의 생성·명명·공개가 전부 이 논문이다. `WM-811K[1]`은 정확한 1차 출처 인용이다.**
저자 소속에 **TSMC 제조기술센터**가 있고(J.-L. Chen), 본문은 *"TSMC가 이 시스템을 도입했다"*고 적는다.

**딸림 확인 — 「패턴 8종」도 원논문의 용어다.** *"labeled from one of the **nine types** …
Center, Donut, Edge-local, Edge-ring, Local, Near-full, Random, Scratch and Nonpattern
(the **first eight types are regarded as Pattern**)"*. 우리의 **「패턴 8종」(`none` 제외)은 이 구분 그대로**다.

⚠ **단 `25,519장`은 [1]의 수치가 아니라 우리가 공개 파일에서 센 값이다.**
원논문은 라벨분을 **54,356(train) + 118,595(test) = 172,951**로 적는데,
공개된 `LSWMD`에서 세면 **172,950**(= `none` 147,431 + 패턴 8종 **25,519**)으로 **1장 차이**가 난다.
**초록 문장의 `[1]`은 「WM-811K」에 붙지 「25,519」에 붙지 않으므로 서술에 문제는 없다.**
**차이의 원인은 확인하지 않았다 — 추정으로 적지 않는다.**

---

## 2. ① 분할 정책 — **lot 단위는 표준이 아니고, 원논문도 아니다**

| 논문 | 분할 | lot 언급 |
|---|---|---|
| **W (원논문)** | **전문가가 선별한 train / 전문가가 무작위로 고른 test** | ⚠ **데이터 설명에서만** |
| **A** | **lot-group split 70/15/15** | ✅ **명시** |
| B | 랜덤 4:1 | ⚠ **데이터 설명에서만** |
| C | 크기별 train set(최대 6400) | ❌ **분할 정책 서술 자체가 없음** |
| D | stratified split, 증강 전 분리 | ❌ |
| E | (미기재) | ❌ |
| F | 300/100/100 (합성 500장/클래스) | ❌ |
| G | 비지도 — 분할 없음 | ❌ |

### 2-A. ★ 원논문의 분할은 lot 단위가 아니다 — **§7의 서술을 정밀화한다**

**W 원문 (§VI-A)**:
> "The data set was divided into a training set … and a test set … For creating the
> training set, **a diverse range of wafer maps were selected to include each pattern
> type** to ensure that the constructed model would be robust. Conversely, **the test
> set comprised wafer maps that were randomly selected by domain experts.**
> Approximately 20% of the wafer maps were labeled … (**54 356 in the training set and
> 118 595 in the test set**) … In addition, **both training and test sets comprised
> unique wafer maps**."

**lot이라는 단어는 분할 서술에 없다.** 원논문의 분할은
**전문가 선별(purposive sampling)**이고, 통제한 것은 **lot이 아니라 중복**이다.

> **CLAUDE.md §7의 「원저자 분할도 lot 단위이나…」는 취소 대상이 아니다.**
> 그 문장의 근거는 논문이 아니라 **우리 실측**이다 —
> `w3_split.md` §③: **교차 lot 0개 / 10,762개** (8종만 봐도 0 / 8,047).
> **즉 「결과적으로 lot이 교차하지 않는다」는 참이고, 「lot 단위로 나눴다」는 논문 근거가 없다.**
> 정확한 서술로 바꾼다: **「원저자 분할은 전문가 선별이며, 실측 결과 lot 누수는 0이다」.**
> 의도가 아니라 부수 효과일 수 있다(lot이 연속 블록이라 lot 단위로 훑으면 자연히 그렇게 된다).

**부수 확인** — 표본 크기가 비대칭이다. **test가 train의 2.2배**(118,595 vs 54,356)다.
8종 부분집합에서는 반대로 train 17,625 / test 7,894(`w3_split.md`)이므로,
**`none`이 test 쪽에 몰려 있다.** 우리가 관찰한 클래스별 분포 이동과 같은 뿌리다.

### 2-B. B는 lot 구조를 알면서 안 썼다

B는 *"the dataset was collected from **47,543 physical lots**, with each lot consisting
of 25 wafers"*라고 쓴다. **그러고도 분할은 랜덤 4:1이고 헤드라인은 accuracy 98.56%다.**
몰라서 안 한 게 아니라 알면서 분할에 쓰지 않았다.

> ⚠ **B의 숫자는 원논문과 다르다.** 원논문은 **46,293 lots**다(§VI-A).
> B가 어디서 47,543을 가져왔는지 불명이다. **우리는 원논문 값을 쓴다.**

### 2-C. C는 분할 정책을 아예 안 적었다

`split|lot|stratif|leak|fold` **히트 0건.** Scientific Reports 게재본에서
train/test 구성 방식이 서술되지 않는다. **재현 가능성의 하한이 이 정도다.**

**→ 판정: lot 단위 분할은 8편 중 1편(A, 2026 preprint). 우리는 소수파다.**
**D-003은 성능이 아니라 누수 논거로 정한 것이므로 바꿀 이유가 없다.**
**단 「표준을 따랐다」고 쓰면 거짓이다.**

---

## 3. ② 보고 지표 — **accuracy 헤드라인은 원논문부터다**

| 논문 | 헤드라인 | per-class | accuracy 반대 논거 |
|---|---|---|---|
| **W (원논문)** | **accuracy 94.63%** | 혼동행렬 대각 | ❌ |
| A | balanced acc + macro-F1 | ✅ per-class F1 | ✅ **명시** |
| B | accuracy 98.56% | 혼동행렬 대각 | ❌ |
| C | accuracy | 클래스별 accuracy 차이 | ❌ |
| D | accuracy 85.0% (macro-F1 0.798) | ✅ per-class recall | ✅ **명시** |
| E | Precision/Recall/F1/Accuracy 4종 | ❌ | ❌ |
| F | accuracy 99.0% | 혼동행렬 | ❌ |
| G | 혼동행렬 (군집) | ✅ | — |

**W 전문에서 `precision` 0회 · `recall` 0회 · `F1` 0회다.** 표 제목이 그대로
**"TABLE IV — ACCURACY COMPARISON FOR WMFPR"**이다.

### 3-A. 다만 원논문은 불균형을 **손실 쪽에서** 처리한다

> "each failure pattern type is **equally crucial, regardless of the number of samples** …
> the new cost value for each pattern type was **proportional to the inverse of its
> corresponding sample size**."

**동기는 macro-F1과 같다** — 클래스를 동등하게 본다. **적용 지점만 지표가 아니라 비용이다.**
그래서 **모델은 균형을 맞추고 보고는 accuracy로 한다.** 이 조합이
*"어느 클래스가 왜 틀렸나"*를 숫자에서 지운다. 우리가 accuracy를 금지한 이유와 정확히 같다.

### 3-B. ★ 원논문의 혼동 서술이 우리 진단 (c)와 같다

> "The matrix shows that **Local was frequently confused with other failure types.**
> … Although the wafer maps were misclassified, users generally accept the prediction
> because these wafer maps **seem to saddle across the boundary of two types**."

**데이터셋을 만든 사람들이 `Loc` 경계 혼동을 「두 유형의 경계에 걸쳐 있다」로 설명한다.**
우리 3단계 병목 `Edge-Loc↔Loc`의 진단 (c) — *"구배가 연속이라 임계값이 없다"* — 와
**같은 주장이다.** 그들은 육안 근거(Fig. 13(b))이고, 우리는 **층별 비 수렴**(⚠ **9/12 재측정:
1.81→1.07이 아니라 1.79→1.02이고 2층 2.01이 최대라 단조도 아니다** — D-033)이라는
측정치다.

> **⚠ 이것을 「검증됐다」로 쓰지 않는다.** 원논문은 8종 통합 SVM이고 우리는 6특징 구성이다.
> 쓸 수 있는 것은 **「원논문도 같은 클래스에서 같은 성격의 혼동을 보고했다」**는 정황뿐이다.
> D(양자 회로)의 `Edge-Loc→Scratch` 81% 오분류까지 합치면
> **다른 방법론 3개가 같은 클래스 삼각형에서 막힌다.** §3-14대로
> **「이 투입 순서에서의 관찰」을 떼는 근거로는 쓰지 않는다.**

**→ 판정: accuracy 헤드라인이 8편 중 6편. 뿌리가 원논문이다. §7의 accuracy 금지 유지.**

---

## 4. ③ 교란 통제 — **정정. 「0건」이 아니라 「설계상 정규화는 있고 교란 실측은 없다」다**

### 4-0. ⚠ v1의 판정을 정정한다

v1은 *"7편 중 개수·크기 교란 통제 0건"*이라 썼고, **Wu 2015의 밀도 기반 13특징이
면적 정규화를 하고 있을 가능성**을 미확인 위험으로 달아뒀다.

**전제부터 틀렸다. Wu 2015에 밀도 기반 특징은 없다.**
원논문의 특징은 **Radon 기반 40차원 + 기하 기반 18차원**이고,
그 조합에 잡음 제거 유무 2벌을 곱해 **(18+40)×2 = 116차원**이다.
*"13-zone density"*는 **후속 연구·공개 커널에서 붙은 것이지 원논문이 아니다.**
(`decisions.md:1531`의 「13-zone 밀도 특징」 항목도 이 전제 위에 있다 — 출처를 다시 달아야 한다.)

### 4-1. 그런데 정규화는 **한다** — 그것도 우리와 같은 방식으로

**기하 특징 (§III-B)**:
> "**Because the wafer maps vary in size, the attributes must be normalized by dividing
> appropriate constants**" — (6) 최대 영역 면적 **÷ 웨이퍼 면적**,
> (7) 둘레 **÷ 웨이퍼 반지름**, (8)(9) 중심까지 최대·최소 거리, (10)(11) 타원 축 비.

**Radon 특징 (§III-A)**:
> "**To ensure that response G is comparable among wafer maps**, minmax normalization
> is applied to G" — 식 (5).

**전부 「웨이퍼 내부 기준 대조비」다.** §3-12에서 우리가 도달한 결론
— *"성공한 특징의 공통점은 전부 웨이퍼 내부 기준 대조비. 절대 통계가 아니다"* —
**과 같은 설계 원칙이 원논문에 이미 있다.**

> **이건 우리 설계의 외부 확인이다.** 우리는 형상 특징 3종이 기대와 반대로 나온 뒤
> **면적 정규화로 되살아난 경험**(§3-12, Center 3.24배)에서 그 원칙에 도달했고,
> 원논문은 처음부터 그렇게 설계했다. **결론이 같다는 것이 요점이다.**

### 4-2. 하지만 **교란 실측은 여전히 0건이다**

**정규화(normalization)와 교란 통제(confound control)는 다른 일이다.**
§3-13이 *"통제와 제거는 다르다"*로 가른 것과 같은 종류의 구분이다.

| | 원논문 | 우리 |
|---|---|---|
| 웨이퍼 크기를 나눠서 제거 | ✅ 설계상 | ✅ |
| **크기가 성능에 얼마나 실려 있는지 측정** | ❌ | ✅ `w4_size.md` (+0.111 중 **84.3%가 F1a·F1b 중복**) |
| **결함 개수를 고정하고 재측정** | ❌ | ✅ §3-2 (F3 0.927 → **0.743**) |
| **귀무값을 순열로 만들어 합격선 검증** | ❌ | ✅ §3-13 (`none` CV 0.828, 귀무 0.830) |

원논문은 다이 개수 분포를 **관찰은 한다** — Fig. 9,
*"the number of dice **varies considerably**"*. **거기서 멈춘다.**
그 변동이 분류 성능에 얼마나 실리는지는 묻지 않는다.

**→ 판정: 8편 중 「개수·크기를 교란으로 놓고 측정한」 사례 0건.**
**정규화까지 포함하면 원논문이 부분적으로 앞선다 — 정규화는 우리 발명이 아니다.**
**「최초」라고 쓰지 않는다. 쓸 수 있는 것은 「읽은 8편에 교란 실측이 없었다」뿐이다.**

---

## 5. ★ 수확 ① — **회전·플립 불변성을 아무도 측정하지 않았다 (원논문 포함)**

### 5-0. v1보다 판정이 세졌다

v1은 *"우리가 5개 파일에서 주장만 하고 측정 기록이 0건"*이라 썼다.
**원논문을 읽으니 원논문도 똑같다.** 그리고 **원논문은 그게 논문의 핵심 주장이다.**

**W 초록**:
> "a set of novel **rotation- and scale-invariant features** is proposed"

**W §III-A** (Radon 특징):
> "The Rμ and Rσ **appear to be** rotation-invariant and scale-invariant."

**W §III-B** (기하 특징):
> "the geometry-based features were obtained by calculating the regional, statistical,
> and linear attributes, **all of which are rotation- and scale-invariant**"

> ***"appear to be"*** — **근거는 Fig. 2(c)(d)의 곡선을 눈으로 본 것이다.**
> 회전시킨 웨이퍼로 특징값을 다시 계산해 비교한 실험은 **논문에 없다.**
> 제목과 초록이 내건 성질을 **측정하지 않았다.** 그 논문이 **377회 인용됐다.**

### 5-1. 우리 상태 — 같다

| 위치 | 문장 |
|---|---|
| `src/azimuth.py:20` | *"회전 불변량인 circular variance만 쓴다"* |
| `src/edge_band.py:90` | *"회전 불변량이므로 notch 방향 정보가 없어도 유효하다"* |
| `src/radial.py:46` | *"이 파일의 특징 2종 (회전 불변)"* |
| `src/shape_feats.py:22` | *"회전 불변만 쓴다"* |
| `docs/coordinate_alignment.md:48` | *"F3는 이미 회전 불변 지표로 설계했으므로"* |

**5개 파일에서 주장한다. 측정 기록은 0건이다.**
**C(Sci Rep 2023)만 회전·플립 test set을 따로 만들어 실측한다** — 8편 중 1편이다.

### 5-2. 왜 값싼 검사이고, 왜 귀무값이 정확한가

웨이퍼 맵은 다이 격자다. **90°·180°·270° 회전과 좌우/상하 플립은 격자 정렬이라
리샘플링이 없다** — 보간 오차가 원천적으로 0이다. 따라서 §3-13이 요구하는
**「이 통계량은 귀무 상태에서 정확히 얼마인가」에 답이 있다**:

> **6종 특징값은 8개 이면군(dihedral) 변환 전부에서 부동소수점 오차 내로 동일해야 한다.**
> 하나라도 어긋나면 **불변성 주장이 틀렸거나 구현에 버그가 있다.**
> *"직관으로 정한 합격선"*이 아니라 **계산된 귀무값**이다.

**예상 30분~1시간.** §3-3의 상한 안이다.

### 5-3. ✅ 실행 완료 (2026-09-10) — `docs/w6_checks.md`

> **6종 중 4종은 오차 0.000e+00으로 통과, 2종은 깨졌다.**
> `circular variance`는 **±π bin 경계 동점**(전수 72.60%가 경계 위 점 보유),
> `cc_compact`는 **최대 연결성분 크기 동점**(전수 0.50%)이다.
> **후자는 회전과 무관하게 원래 미정의였다** — 회전 검사가 잡아낸 것이
> 회전 문제가 아니었다. 상세·판정·수정 보류 근거는 `w6_checks.md` §2.

---

## 6. ★ 수확 ② — **중복 웨이퍼맵을 한 번도 세지 않았다** (v2 신설)

### 6-0. 원논문이 통제한 것은 lot이 아니라 중복이었다

> "there are only **696 599 unique wafer maps**" (전체 811,457 중)
> "**both training and test sets comprised unique wafer maps**"

**원본의 14.2%가 중복이고, 원저자는 그것을 제거하고 분할했다.**

**우리 쪽 상태** (`grep 중복|duplicat|unique`):
`docs/`와 `src/`의 「중복」은 **전부 특징 간 상관**(D-010)이다.
**표본 중복 — 같은 웨이퍼 맵이 두 번 들어있는가 — 을 센 기록은 0건이다.**

### 6-1. lot 그룹 분할이 이걸 덮어주지 않는다

같은 lot 안의 중복은 `StratifiedGroupKFold`가 막는다.
**하지만 서로 다른 lot에 있는 동일 맵은 그대로 fold를 넘는다.**
`Near-full`처럼 lot당 1.1장인 클래스에서는 그룹 분할의 보호가 약하다.

**검사**: 8종 라벨 부분집합(25,519장)에서
① 완전 동일 맵이 몇 쌍인가 ② 그 중 fold를 교차하는 것이 몇 개인가.
`hash(map.tobytes())` 한 줄이면 끝난다. **10분.**

> **결과가 0이면 그것도 결과다** — D-003의 누수 방어가 중복까지 커버한다는 증거가 된다.
> **0이 아니면 `w3_*` 성능 수치의 해석에 단서를 달아야 한다.**

### 6-2. ✅ 실행 완료 (2026-09-10) — `docs/w6_checks.md`

> **0이 아니었다.** 중복 그룹 **93개 / 187장(0.73%)**이고
> **93개 전부가 lot을 넘는다** — `StratifiedGroupKFold`가 하나도 막지 못한다.
> fold를 실제로 넘는 것은 **0.58%**(seed 0). `Near-full`은 **10.07%**가 중복이다.
> 완전 동일 맵에 **상충 라벨 3건**(그중 2건이 헤드라인 쌍 `Edge-Loc↔Edge-Ring`).
> 상세는 `w6_checks.md` §1.

---

## 7. 등급이 낮은 후보 (기록만)

- **train set 크기별 성능 곡선** (C가 보고) — 우리는 없다. **등급 중.**
  전이 실험·외부 검증이 이미 있어 우선순위가 밀린다
- **잡음 제거 유무 2벌을 모두 쓰는 설계** (W) — 원논문은 median filter 적용/미적용
  특징을 **둘 다 넣는다**. *"어느 쪽이 맞는지 모르니 둘 다"*라는 태도다.
  우리 전처리에 대응물이 없다. **등급 하** — 새 관찰이 아니라 성능 시도다

---

## 8. 판정 요약 (v2)

| 질문 | 답 | 우리 대응 |
|---|---|---|
| ① lot 단위 분할이 표준인가 | ❌ **8편 중 1편.** **원논문도 아니다**(전문가 선별) | **D-003 유지.** *"표준을 따랐다"* 금지. §7 서술 정밀화 |
| ② accuracy가 주 지표인가 | ⚠ **8편 중 6편.** **뿌리가 원논문(94.63%)이다** | **§7 accuracy 금지 유지** |
| ③ 개수·크기 교란을 통제하는가 | ⚠ **정규화는 원논문부터 있다.** **교란 실측은 8편 중 0건** | *"최초"* 금지. **정규화는 우리 발명이 아니다** |
| ★ 빠뜨린 대조가 있나 | ✅ **2건 — 회전·플립 불변성 실측 · 중복 웨이퍼맵 계수** | **둘 다 실행 완료 (`w6_checks.md`). 둘 다 무언가를 잡았다** |

**초록에 미치는 영향: 0.** 참고문헌 추가가 물리적으로 불가하고(1건=2페이지),
초록에 신규성 주장이 없다. **이 스캔의 결과로 초록의 어떤 문장도 바뀌지 않는다.**

**포스터에는 재료가 생겼다** — §3-B(원저자도 `Loc` 경계 혼동을 *"두 유형에 걸쳐 있다"*로
설명) · §4-1(대조비 설계 원칙의 외부 확인) · §5(불변성 미검증이 이 분야의 관행).

---

---

## 9. 출처와 접근 — **재현하려면 이게 필요하다**

v1의 「못 읽은 것」 절은 v2에서 사라졌다(원논문을 받았으므로).
**하지만 접근 조건 자체는 기록으로 남겨야 한다** — 다음 사람이 같은 벽에 부딪힌다.

### 9-A. 입수 경로

| 논문 | 경로 |
|---|---|
| A·B·D·E·F·G (arXiv 6편) | `curl https://arxiv.org/pdf/<id>` — **무인증, 즉시** |
| C (Sci Rep) | `https://www.nature.com/articles/s41598-023-34147-2.pdf` — 웹 뷰는 인증 리다이렉트, **`.pdf` 직링크는 열린다** |
| **W (Wu 2015, IEEE TSM)** | ⛔ **스크립트로 못 받는다.** **사용자가 기관 접근으로 받아 `docs/`에 넣었다** |

**Wu 2015 접근 실패 실측 (2026-09-10)**

- `ieeexplore.ieee.org/iel7/66/7027932/06932449.pdf` → **본문이 `<script>`**(17KB). 봇 차단
- Unpaywall (`api.unpaywall.org/v2/10.1109/TSM.2014.2364237`) → **`is_oa: false`**
- Semantic Scholar → **`openAccessPdf: {status: "CLOSED"}`**
- Semantic Scholar 검색 API → **HTTP 429** (무키 호출 제한)

> **합법적인 공개 사본이 없다. 기관 구독이 유일한 경로다.**
> 받은 PDF에는 *"Authorized licensed use limited to: Chungnam National University"*가
> 찍혀 있다. **`.gitignore`에 `docs/*.pdf`를 넣어 리포에 올라가지 않게 했다.**

### 9-B. 서지 확정은 Crossref로 했다 (무인증)

```
curl -s https://api.crossref.org/works/10.1109/TSM.2014.2364237
→ IEEE Trans. Semicond. Manuf. 28(1):1-12, 2015-02, 피인용 342(Crossref)/377(S2)
```

**검색 결과에 적힌 권·호·페이지를 그대로 믿지 않고 DOI로 확정했다.**

### 9-C. ⚠ 본문 추출 — **요약 API를 쓰지 않았다**

두 번 데였다.

1. **웹 검색 요약이 여러 논문을 한 문단에 섞었다.** *"lot-group split을 쓴다"*를
   출처 없이 서술했는데 **실제로는 8편 중 1편에만 있는 문장**이었다.
   그대로 인용했으면 **§2의 판정이 정반대**가 됐다
2. **PDF 요약 도구가 arXiv PDF를 못 읽었다.** *"corrupted or improperly encoded
   PDF stream"* / *"content appears incomplete"*를 반환하고,
   **그러면서도 그럴듯한 답을 지어냈다**

**→ 내려받아 로컬에서 전문을 뽑았다.**

```python
import pymupdf, re
t = "\n".join(p.get_text() for p in pymupdf.open(path))
# 방법론 절만: split|lot|stratif|leak|fold / macro-F1|per-class|balanced accuracy
#              defect density|number of defect|die count|wafer size|confound
# 히트 전후 160자 -> 걸린 문단은 통째로 다시 읽는다
```

- **콘솔이 깨지면 `PYTHONIOENCODING=utf-8`** (cp949 기본값 — CLAUDE.md §7과 같은 함정)
- 추출 스크립트와 PDF는 **스크래치패드에만** 두었다 (저작권)

> **이 절의 요점**: *"요약을 읽고 논문을 읽었다고 하지 않는다."*
> §0에 *"원문을 읽었다"*고 쓴 것의 실제 내용이 이것이다.


---

*v1 2026-09-10 오전 (7편) / **v2 2026-09-10 오후 (원논문 추가, §4·§5·§6 정정)**.
소요 누계 약 2시간. 원문 추출 스크립트와 PDF는 리포 밖에 둔다 — `.gitignore: docs/*.pdf`.*
