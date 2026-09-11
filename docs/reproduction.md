# 재현 부록 — 방법 개요가 생략한 정의와 상수

**2026-09-11 신설.** 외부 검토가 *"개요만으로는 재구현할 수 없다"*며 17항목을 지적해 만들었다.
개요(`method_overview.md`)는 **주장**을 담고, 이 문서는 **그 주장을 다시 만드는 데 필요한 것**만 담는다.

---

## 0. 환경 — 이걸 틀리면 수치가 재현되지 않는다

| 항목 | 값 |
|---|---|
| **scikit-learn** | **1.7.2 고정** (D-022) |
| 왜 | 1.9.0에서 6종 macro **0.837 → 0.830**, 5종 `Loc↔Scratch` 0.168 → 0.196, **1단계 최대 병목 쌍이 `Center↔Loc` → `Loc↔Scratch`로 뒤바뀐다.** 데이터·코드는 동일했고 변수는 sklearn뿐이었다 |
| 실행 | `./.venv/Scripts/python.exe src/*.py` (base anaconda 아님) |

## 1. 데이터

- 원본 `data/LSWMD.pkl` (WM-811K). die 값 **0 = 웨이퍼 영역 밖 / 1 = 정상 / 2 = 불량**
- **패턴 8종 25,519장** (`none` 제외 — D-002): `Center`(4,294) `Donut`(555) `Edge-Loc`(5,189)
  `Edge-Ring`(9,680) `Loc`(3,593) `Near-full`(149) `Random`(866) `Scratch`(1,193)
- **리사이즈하지 않는다** (D-004). 맵 크기가 제각각인 채로 특징을 계산한다
- 그룹 키는 `lotName`

## 2. 층(layer)과 밴드 — 가장자리 3종의 토대

**반경 비율이 아니라 침식(erosion) 기반이다.** `R_CUT` 방식은 폐기됐다 (D-007).

```python
# 유효 영역(값 != 0)에서 8-이웃 침식을 반복해 최외곽부터 한 층씩 벗긴다
층 l_i = (침식 i-1회 결과) ∖ (침식 i회 결과)
밴드   = 최외곽 EDGE_LAYERS 층의 합집합
```

| 상수 | 값 | 쓰임 |
|---|---|---|
| `EDGE_LAYERS` | **1** | 밴드 두께 (D-007) |
| `MAX_LAYERS` | 8 | 층별 프로파일 관측 범위 |
| `INNER_FROM, INNER_TO` | **3, 6** | `edge_contrast`의 내부 기준층 (2층은 링 번짐이 섞인다) |
| `N_WEIGHT_BINS` | 36 | `circ_var` 가중치 정규화 |
| `MIN_FAIL` | 12 | 이 미만이면 `circ_var` = NaN |
| `SUBSAMPLE_N` | 20 | 개수 통제용 재표집 크기 (D-008) |
| `MAX_FAIL` | 4000 | 초과하면 `cc_compact` = NaN |
| `ALPHA` | 1.0 | Laplace 평활 |
| `DISK` | √π/2 ≈ 0.8862 | 원판 기준선 |

## 3. 특징 6종 — 전부 **웨이퍼 내부 기준 대조비**다

| 이름 | 정의 | 겨냥 |
|---|---|---|
| `coverage` | 밴드 안 불량 die 수 / 밴드 die 수 | 둘레를 얼마나 덮었나 |
| `edge_contrast` | `(f₁+α)/(n₁+2α)` ÷ `(f₃₋₆+α)/(n₃₋₆+2α)` | 가장자리 대 내부 |
| `circular variance` | 밴드 위 방위각의 가중 원형분산. 가중치 = 1/(그 방위각 bin의 밴드 die 수) | 어떻게 흩어졌나 |
| `F1a` (`radial_contrast`) | `rate(r<R1) / rate(r≥R2)` | `Center` |
| `F1b` (`mid_peak`) | `rate(R1≤r<R2) / max(안, 밖)` | `Donut` |
| `cc_compact` | `(n/L) / (DISK·√n)`, `n`=최대 8-이웃 연결성분 크기, `L`=주축 투영 범위+1 | 선형성 |

- `rate(mask)` = `(불량 수 + α) / (die 수 + 2α)`
- **반경 `r`은 타원 정규화다** — 유효 영역 bounding box 중심에서 **축별 반장으로 나눈다**
  (`r = hypot((y−cy)/hy, (x−cx)/hx)`). 원형 가정으로 짜면 F1a·F1b가 달라진다
- **`R1 = 1/3, R2 = 2/3`은 임의값**이다 (유효 반경 3등분). 성능으로 고르지 않았다는 것만 보장된다
- `cc_compact`의 주축은 `np.linalg.eigh(cov)`의 **최대 고유값 쪽 고유벡터**다

**⚠ 결측은 대치하지 않고 NaN 그대로 RandomForest에 넣는다.** NaN 비율이 클래스마다 다르므로
0으로 채우면 다른 수치가 나온다.

## 4. 분류기와 평가

```python
RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                       class_weight="balanced", random_state=0, n_jobs=-1)
```

- **주 분할**: `StratifiedGroupKFold(5, shuffle)` on `lotName`, **seed 0/1/2**.
  fold별 out-of-fold 예측을 모아 **25,519장 전수**에 예측을 붙이고 macro-F1을 낸 뒤 **seed 3개 평균**
- **혼동 쌍(상호 오분류율)** = `(M[i,j] + M[j,i]) / (M[i].sum() + M[j].sum())`.
  개요의 0.188 / 0.168 / 0.102가 전부 이 양이다
- **외부 검증**: 원저자 `trainTestLabel`(`split_orig`) — **같은 25,519장을 다시 나눈 것**이고
  test 비율 30.9%, **단일 fit 1회**(반복 불가). 클래스 사전확률이 달라
  **절대 수치를 직접 대지 않고 재현율과 혼동 쌍 순위를 본다**

## 5. 알려진 구현 결함 — 재현자가 알아야 한다

| 결함 | 상태 |
|---|---|
| **`azimuth.auc`의 동점 처리** — 서수 순위라 완전 동점 입력이 0.5가 아니라 0이 됐다 | **2026-09-11 정정**(평균 순위). 연속 특징은 동점 ≤0.11%라 보고 수치 불변(0.313 동일) |
| **`cc_compact` 동점 2종** — ⓐ 최대 연결성분 크기 동점(128장) ⓑ **PCA 고유값 동점(9장)**. 둘 다 값이 정의되지 않는다 | **`w6_impact.cc_variant(mode="nan")`에 수정 완료**(귀무 3.9e-15). **`shape2.py` 본판은 아직 원판이고 적용은 11월** |
| **`circular variance`의 회전 불변성** | **구조적으로 못 고친다.** bin 경계가 축이거나 대각인데 격자점이 둘 다 밟는다. 수정안 5개 전부 귀무 불합격 (D-024) |

## 6. 스크립트 대응

| 산출물 | 스크립트 |
|---|---|
| 분할 | `src/split.py` → `data/processed/split_folds.npz` |
| 가장자리 3종 | `src/edge_band.py`, `src/edge_contrast.py` |
| 반경 2종 | `src/radial.py` |
| 형상 1종 | `src/shape2.py` |
| 8종 최종 모델·혼동 쌍 | `src/model8_final.py` |
| 순서 의존성·leave-one-out | `src/order_dep.py`, `src/order_seed.py` |
| 외부 검증 | `src/external_check.py` |
| 회전 불변성·중복 | `src/w6_checks.py`, `src/w6_impact.py` |
