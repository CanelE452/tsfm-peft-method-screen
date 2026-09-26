# RAW vs RESIDUAL 2×2 Go/No-Go

목적은 새 모델을 만드는 것이 아니다. 이전 실행에서 혼입되어 있던 두 요소
(자료원 RAW/RESIDUAL, 의존성 모델 Schaake/Gaussian)을 분리한다.

모든 셀은 **동일한 Chronos-2 forecast cache, 동일 PIT-calibrated marginal,
동일 Gaussian copula, 동일 shrinkage, 동일 scenario 수, 동일 base Gaussian draws**를 쓴다.
달라지는 것은 딱 두 축이다.

|            | FIXED | RECENT |
|------------|-------|--------|
| RAW        | RAW_FIXED | RAW_RECENT |
| RESIDUAL   | RESID_FIXED | RESID_RECENT |

- RAW: 실제 24시간 경로를 TRAIN 기준 horizon별 empirical CDF → normal score로 바꾼 뒤 상관을 추정.
- RESIDUAL: Chronos-2 예측분포에 대한 PIT를 사용하되, RAW와 똑같이 TRAIN-only channel×horizon empirical CDF → normal score로 변환해 상관을 추정.
- FIXED: TRAIN으로 한 번 fit한 correlation을 CAL/PILOT 모두 끝까지 고정.
- RECENT: 해당 origin보다 과거에 이미 관측된 벡터만 사용. 28일 half-life, fixed prior 25%, shrinkage 0.10.
- PILOT recent는 TRAIN+CAL+이미 끝난 PILOT origin만 사용할 수 있다. 현재/미래 정답은 금지.

중요: 기존 run의 `RAW_SCHAAKE`는 이번 주 비교에서 쓰지 않는다. 이번 질문은
"residual 자체가 가치 있는가"이므로 RAW와 RESIDUAL에 **같은 Gaussian estimator**를 적용한다.

주 지표:
- 채널별 24h total CRPS를 origin-day에서 평균한 뒤 CAL/PILOT 비교
- 80/90% interval coverage/width는 보조
- PILOT paired circular moving-block bootstrap, block length=3 days, 5000 reps, 90% CI
- 4채널 방향도 기록

Go/No-Go:
1. RESIDUAL_SOURCE_SUPPORTED:
   RESID_FIXED vs RAW_FIXED가 CAL/PILOT 모두 >=2% 개선,
   PILOT block-bootstrap CI low >0, 3/4채널 이상 개선.
   RECENT 축의 RESID_RECENT vs RAW_RECENT도 방향이 음수가 아니어야 한다.

2. RESIDUAL_ADAPTATION_SUPPORTED:
   RESID_RECENT vs RESID_FIXED가 CAL/PILOT 모두 >=1.5% 개선,
   PILOT block-bootstrap CI low >0, 3/4채널 이상 개선.

최종:
- source PASS + adaptation PASS: GO_SMALL_ADAPTIVE_RESIDUAL_MODULE
- source PASS + adaptation FAIL: STATIC_RESIDUAL_POSTPROCESSING_ONLY
- source FAIL, RAW dependence가 independence보다 의미 있게 우세한 경우: RAW_DEPENDENCE_BASELINE_ONLY
- source FAIL + dependence 자체도 약함: STOP_DEPENDENCE_TOPIC
- 통계적으로 경계면: INCONCLUSIVE_SOURCE_EFFECT

이번 결과가 양성이어도 같은 Electricity 4채널에서 나온 screening이다. 독립 데이터 확증은 별도다.
