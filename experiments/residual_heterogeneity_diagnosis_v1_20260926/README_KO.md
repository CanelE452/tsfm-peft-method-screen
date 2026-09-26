# Residual-source 이질성 원인 진단 v1

## 목적

2×2 결과에서 `RESID_FIXED`가 `RAW_FIXED`보다 PILOT 평균 CRPS는 3.93% 좋았지만
채널별 방향은 0/3 개선, 1/2 악화였다.

이번 작업은 새 방법을 만들지 않는다. 딱 하나를 확인한다.

> **왜 어떤 채널에서는 residual dependence가 raw dependence보다 유리하고,
> 어떤 채널에서는 그렇지 않은가?**

Chronos 재추론, backbone/LoRA/adapter 학습, threshold tuning은 전부 금지한다.
기존 forecast cache와 기존 2×2 per-case 결과를 그대로 사용한다.

## 사전에 고정한 진단 축

### A. source-to-target dependence alignment
모델이 실제로 맞혀야 하는 target dependence는 PILOT에서 관측된
Chronos residual normal-score의 24×24 시간 상관구조로 정의한다.

채널별로:
- TRAIN RAW correlation → PILOT residual target distance
- TRAIN RESIDUAL correlation → PILOT residual target distance
- normalized Frobenius distance
- lag profile MAE, lag={1,2,3,4,5,6,12,23}

Residual이 이기는 채널에서 residual source가 target에 더 가까운지를 본다.

주의: PILOT target matrix는 사후 진단용이다. 모델 fitting/selection에 사용하지 않는다.

### B. aggregate-dispersion alignment
24시간 합의 분산에 직접 영향을 주는
`1^T R 1 / H`를 각 source와 PILOT target에서 계산한다.

- RAW source aggregate factor
- RESIDUAL source aggregate factor
- PILOT residual target aggregate factor
- 각 source의 absolute factor error

Residual 방식의 CRPS 이득이 "aggregate uncertainty 크기를 더 맞춘 효과"와 연결되는지 본다.

### C. stability
TRAIN first-half vs second-half, TRAIN vs CAL에서:
- RAW correlation stability
- residual correlation stability
를 같은 shrinkage로 비교한다.

### D. marginal quality
의존성 방법은 marginal을 공유하므로, channel별 Chronos marginal 품질이
source 효과와 상호작용하는지만 진단한다.

PILOT에서:
- raw PIT mean/variance
- TRAIN-calibrated PIT KS-to-uniform
- 평균 quantile pinball loss

### E. 실제 forecast score
기존 2×2 `PILOT_CASES.jsonl`을 그대로 읽어:
- RAW_FIXED → RESID_FIXED CRPS gain
- 3-day circular moving-block bootstrap 90% CI
- 80/90 coverage와 width
를 채널별로 다시 요약한다.

## 해석 규칙

자동으로 "논문 가능"을 선언하지 않는다.
채널이 4개뿐이라 원인 설명은 exploratory다.

다만 다음 진단 token을 기록한다.

- `STRUCTURAL_ALIGNMENT_PLAUSIBLE`
  - 4개 중 >=3개 채널에서
    CRPS gain의 부호와 "residual source가 PILOT target에 더 가까운가"의 부호가 일치하고,
    winning channel 중 >=1개는 block-bootstrap CI low > 0.

- `DISPERSION_ALIGNMENT_PLAUSIBLE`
  - 위 조건은 약하지만 4개 중 >=3개에서
    CRPS gain 부호와 aggregate-factor closeness 부호가 일치.

- `MARGINAL_LIMITATION_PLAUSIBLE`
  - CRPS losing channel들이 winning channel들보다 PILOT calibrated-PIT KS가 모두 더 크고,
    구조/dispersion alignment는 3/4 일치하지 않음.

- `HETEROGENEITY_UNEXPLAINED`
  - 위 어느 것도 충족하지 않음.

이 token은 후속 연구 방향 선택용 진단이지 통계적 인과 결론이 아니다.

## 다음 행동

- STRUCTURAL_ALIGNMENT_PLAUSIBLE:
  residual source 아이디어를 더 많은 채널/독립 데이터에서 재현.
  아직 adaptive module은 만들지 않는다.
- DISPERSION_ALIGNMENT_PLAUSIBLE:
  residual-specific 구조보다 단순 aggregate variance scaling baseline을 먼저 검토.
- MARGINAL_LIMITATION_PLAUSIBLE:
  dependence가 아니라 marginal calibration 문제를 먼저 해결.
- HETEROGENEITY_UNEXPLAINED:
  이 4채널 결과로 method를 설계하지 말고 주제 우선순위를 낮춘다.
