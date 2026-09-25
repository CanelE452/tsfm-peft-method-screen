# Residual Dependence Go/No-Go v1

## 목적

동결된 TSFM의 각 미래 시점별 주변 예측분포는 그대로 유지하고, **미래 24시간 예측오차의 시간적 의존성만 적응**했을 때 24시간 누적 전력수요의 확률예측이 좋아지는지 확인한다.

새 학습형 모듈을 먼저 만들지 않는다. 다음 세 질문을 순서대로 판정한다.

1. 미래 시점을 독립적으로 샘플링하는 것보다 시간 의존성을 넣는 것이 실제로 필요한가?
2. 원 시계열의 과거 경로 의존성보다 **TSFM이 남긴 예측오차의 의존성**을 쓰는 것이 더 나은가?
3. 고정 residual dependence보다 최근 오차를 이용해 순차 갱신한 dependence가 더 나은가?

3번까지 통과해야 다음 단계에서 작은 학습형 dependence adapter를 만들 근거가 생긴다.

## 데이터/분할

저장소가 이미 사용해 온 `electricity.txt.gz`의 첫 네 채널, 1시간 간격을 사용한다.
과거 336시간(14일) → 미래 24시간.

- TRAIN residual fit: origin 336 ... <6000, 24시간 간격
- CAL: 6000 ... <6600, 24시간 간격
- PILOT: 6600 ... <7200, 24시간 간격
- 기존 evaluation_start=7200 이후의 **수치 값은 이번 실행에서 파싱하지 않는다.**

이 분할은 논문 최종 TEST가 아니라 주제 선택용 screening이다. 동일 데이터는 기존 저장소에서 사용된 이력이 있으므로 양성 결과도 독립 확증으로 보지 않는다.

## 공통 주변분포

모든 방법은 동일한 Chronos-2 quantile forecast를 쓴다. TRAIN PIT(probability integral transform)만으로 horizon별 공통 marginal recalibration을 만들고 모든 방법에 동일하게 적용한다. 따라서 방법 간 차이는 **시간 의존성**뿐이다.

## 방법

- `IND`: 미래 24시점을 독립 샘플링.
- `RAW_SCHAAKE`: 과거 실제 24시간 경로의 rank pattern으로 동일 marginal sample을 재배열.
- `RESID_FIXED`: TRAIN에서 얻은 Chronos-2 calibrated PIT residual의 24x24 Gaussian-copula correlation을 채널별로 고정 사용.
- `RESID_RECENT`: 현재 origin보다 과거에 이미 관측된 residual만 사용해 28일 half-life로 correlation을 순차 갱신하고, fixed correlation에 25% shrink.

`RESID_RECENT`는 현재 origin이나 미래 정답을 fitting에 사용하지 않는다.

## 주 지표

각 sample path의 24시간 합계에 대해:

- Primary: 24h total CRPS (낮을수록 좋음)
- Secondary: 80%/90% coverage, interval width, median absolute total error
- channel별 결과
- origin-day paired bootstrap 90% CI

## Go/No-Go

1. `DEPENDENCE_RELEVANT`
   - CAL과 PILOT 모두: dependence-aware 최선 방법이 IND 대비 CRPS 3% 이상 개선
   - PILOT bootstrap 90% CI lower bound > 0
   - 최소 3/4채널에서 개선

2. `RESIDUAL_STRUCTURE_VALUE`
   - CAL/PILOT 모두: residual 방법이 RAW_SCHAAKE 대비 2% 이상 개선
   - PILOT CI lower bound > 0
   - 최소 3/4채널 개선

3. `ADAPTATION_NEEDED`
   - CAL/PILOT 모두: RESID_RECENT가 RESID_FIXED 대비 1.5% 이상 개선
   - PILOT CI lower bound > 0
   - 최소 3/4채널 개선
   - 80%/90% coverage nominal deviation가 fixed보다 3%p 이상 악화되지 않음

최종 token:

- 세 gate 모두 PASS → `GO_ONLINE_RESIDUAL_ADAPTER`
- 1,2 PASS / 3 FAIL → `STATIC_RESIDUAL_SUFFICIENT`
- 1 PASS / 2 FAIL → `SIMPLE_RAW_DEPENDENCE_FIRST`
- 1 FAIL → `NO_GO_DEPENDENCE_AXIS`
- 기술 실패 → `TECHNICAL_FAILURE`

`STATIC_RESIDUAL_SUFFICIENT`는 학습형 adapter를 만들 필요가 약하다는 중단 신호다.
