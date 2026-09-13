# 기존 TSFM PEFT 실험 전체 재검토 — 2026-09-14

검토 기준: `CanelE452/tsfm-peft-method-screen`, `main`, `c7b0f2f`.
기존 결과·기준은 보존하고 연구 해석을 다시 평가했다. 이번 작업은 학습 0회, optimizer update 0회다.

**현재 결과로 논문 수준의 우위가 확인된 후보는 없다. 그러나 모든 STOP을 같은 종류의 실패로 취급할 수 없다. 재진단 우선순위는 일반 prediction anchoring과 DualClock이다.** 전자는 raw-loss LoRA에 대한 개선이 여러 개발 조건에서 관찰됐고, 후자는 작지만 양의 효과와 학습 예산 끝에서 선택된 checkpoint가 있다. 둘 다 신규성·독립 재현·최선 기준선 우위가 해결된 상태는 아니다.

**검토 범위와 재현**

초기 7개, Candidate05 구현 복구, Freshness v2, 두 메모리 진단, Forecast-query 학습·checkpoint 진단·동일시간 후속, Block shape, 야간 3개, Calibration anchor를 대조했다. 총 **270 completed fits와 9 stream attempts(8 완료, 과거 1 중단)**다. 비교군 fits가 포함된 수치이며 270개 새 방법이나 독립 반복을 뜻하지 않는다. Distillation의 8 fits는 전부 raw 비교군이고 학생 fits는 0이다. 워밍업·진단 backward·smoke update를 fits에 더하지 않았다.

이번에 `verify_all.py`와 야간/Calibration의 `--verify-only`를 CPU에서 다시 실행했다. 초기 예측, 복구 스트림, v2, 메모리 gradient, Query, Block shape, 동일시간 비교, 야간 376개와 Calibration 286개 prediction cache 검증이 모두 통과했다. 재검토 산술 스크립트는 결과 파일 **700개**의 SHA256 불변도 확인했다. 이는 저장된 실행과 수치의 일관성 검증이며, 모든 방법의 과학적 타당성이나 데이터 독립성을 보증하지 않는다. 실행 명령·종료 코드·로그는 [verification_commands.json](verification_commands.json), 집계는 [audit.json](audit.json), [fit_inventory.csv](fit_inventory.csv)에 있다.

재계산: `CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_full_reassessment.py`.
기존 예측·gradient replay에는 로컬 ignored cache가 필요하다. 이 스크립트 자체는 기록된 수치의 산술 비교를 재계산한다.

**판정 기준에서 바로잡을 점**

초기 7개 기준은 원래 [사용자 실험 지시](../../docs/USER_PROTOCOL.md)에 명시돼 있었다. 예를 들어 DualClock 1% 개선, PatchPhase 분산 30% 감소, Maturity 계산 overhead 20% 제한이다. 후속 실험은 별도 프로토콜에서 0.5% 또는 1% 개선과 두 데이터셋·seed 조건을 묶었다. 모든 기준을 제가 임의로 같은 방식으로 만들었다는 설명도 정확하지 않다. 제가 바로잡아야 할 부분은 **고정 개발 기준의 STOP을 전체 연구 주제의 가능성 부족으로 넓혀 설명한 것**이다.

학습 중 한 checkpoint가 악화했다고 즉시 FAIL한 구조는 아니다. 대부분 계획된 학습을 완료하고 V에서 checkpoint/LR를 선택했다. 예외는 사전에 정한 구성·신규성·gradient·teacher 진입 조건이다. 최근 품질 gate는 두 데이터셋 모두 최선 비교군을 넘고 각 seed 조건도 만족해야 하는 AND 규칙이라 작은 pilot의 유망 신호를 놓칠 수 있다.

또한 초기 표의 `%F0 = 100×(비교군−후보)/F0`와 후속 표의 `%baseline = 100×(비교군−후보)/비교군`은 분모가 다르다. 서로 같은 개선율처럼 합산하거나 전체 평균을 내면 안 된다. [comparisons.csv](comparisons.csv)는 두 분모를 명시한다. 아래 표도 초기 수치에 `%F0`를 붙였다.

**전체 후보별 재판독**

| 실험·후보 | 확인된 핵심 결과 | 이번 재해석 | 현재 우선순위 |
|---|---|---|---|
| 01 Freshness gate | Feature보다 −0.735%F0, clean 손해 0.989%F0 | 손상 문제는 있지만 특별한 gate의 추가 가치는 없음 | 현 구현 보류 |
| Freshness v2 | 두 seed 평균 −0.171%F0; 각 seed 최선 비교군에 패배 | F0보다 좋아진 효과를 conditioning 효과로 해석하면 안 됨 | 반복 음성 근거로 낮음 |
| 02 DualClock | EventSummary 대비 +0.0140%F0, 63.7% series에서 양의 차이, 상위 10% 효과 제외 시 평균 음수 | 작고 불안정한 양의 신호. 충분한 학습 여부 미해결 | 제한적 재진단 2순위 |
| 03 PatchPhase | unseen phase −0.0137%F0, phase 분산 0.276% 감소 | 정확도 차이는 작지만 주 메커니즘 효과도 작음. 경계 민감성 문제 자체는 남음 | 문제는 유지, 현 gate는 낮음 |
| 04 FR-LoRA | 비교군 포함 선택 4개 모두 step 0 | 학습된 FR의 추가 가치가 선택되지 않은 실험. 주제 전체 반증은 아님 | 적응 가능한 조건 확인 전 보류 |
| 05 Maturity 복구 | TAFAS-like 대비 −4.298%F0, F0보다 12.344% 악화, overhead +103.7% | 한 조건에서 근소하게 진 것이 아니라 online 손실과 비용이 모두 불리 | 현 규칙 종료 유지 |
| 06 Conditional Path | 신규성 중복 판단으로 fits 0 | 정확도 실패 데이터가 전혀 없음. 기여 차별화의 문제 | 차별화 근거 없으면 재학습 불필요 |
| 07 Censor Preserve | Censor-only 대비 −0.0322%F0, censored 위치 손실도 악화 | 작은 평균 차이지만 보존항을 넣을 이유가 드러나지 않음 | 현 보존항 보류 |
| Global temporal compression | 같은 retained bytes INT8보다 gradient 오차가 매우 큼: 66–77% | forecasting 실패가 아닌 압축 primitive의 불리한 gradient 진단 | 현 primitive 종료 유지 |
| Local residual backward | A-gradient 오차 12.9–36.3%, local FP16보다 peak도 큼 | 예측 품질은 미측정. 가까운 비교군 대비 자원·근사 이점이 없음 | 현 primitive 종료 유지 |
| Forecast-query 초기 | Standard 대비 ETTm2 +0.519%, Electricity −0.326%; peak 28.13% 증가 | 일부 정확도 신호와 메모리 목표 실패가 함께 있음 | 후속 결과와 함께 판단 |
| Query checkpoint 진단 | 양쪽 checkpoint-on에서 peak 6.69% 감소, step 15.9–18.6% 느림 | 조건부 자원 교환관계는 있음. 예측 학습 실험은 아님 | 시스템 대안으로 보관 |
| Query 동일시간 | Standard 대비 ETTm2 −0.369%, Electricity −1.155%; F0·Side보다 평균 좋음 | 모든 이점이 사라진 것은 아니지만 동일 active-time 정확도 우위는 없음 | 정확도 중심 주제는 낮음 |
| Block shape | 고정된 보수적 비교군 대비 −0.407%/−0.269%, pooled보다 이점 없음 | shape 업데이트는 작동했으나 block별 수락 규칙의 추가 가치가 없음 | 현 규칙 보류 |
| Prediction anchor | 초기 최선 비교군 대비 +0.196%/+0.341%, 다른 seed 조건 통과 | 1% 기준 때문에 중단된 실제 양의 신호. 후속 native 비교는 음성 | 제한적 재진단 1순위 |
| Drift gate | +0.012%/−1.282%, moment gate와 작은 차이 | 유용한 조건부 개선의 범위를 아직 입증하지 못함 | 조건 진단 전 확장 보류 |
| Context distillation | V teacher가 raw LoRA보다 4.08%/4.22% 나쁨; 학생 학습 0 | 이 teacher를 쓸 근거 부족. 증류 일반의 실패는 아님 | teacher 개선 가능성부터 |
| Calibration anchor | native 대비 −1.153%/−0.633%; uniform·shuffled에도 패배 | 단순 anchoring 효과와 calibration 가중치 효과를 구분해야 함 | 현 가중치 규칙 종료 유지 |

원본: [초기 7개](../../results/screening_summary/latest_review.md), [v2](../../results/candidate_01_v2/RESULT.md), [Maturity 복구](../../results/candidate_05_repaired/RESULT.md), [global](../../results/memory_feasibility/RESULT.md), [local](../../results/local_backward_feasibility/RESULT.md), [Query](../../results/forecast_query_pilot/RESULT.md), [checkpoint](../../results/forecast_query_checkpoint_diagnostic/RESULT.md), [동일시간](../../results/forecast_query_equal_time/RESULT.md), [Block](../../results/block_shape_pilot/RESULT.md), [야간](../../results/overnight_20260913/REPORT.md), [Calibration](../../results/calibration_anchor_20260914/REPORT.md).

**재검토에서 더 명확해진 네 가지**

1. **Prediction anchoring의 신호는 학습 목적함수에 의존한다.**

| 개발 평가 | 균일 anchor 대 raw-loss LoRA 개선 | 균일 anchor 대 native-loss LoRA 개선 |
|---|---:|---:|
| 야간 ETTh1 | +0.284% | +0.489% |
| 야간 Traffic | +1.498% | +0.341% |
| 후속 ETTh1 | +1.777% | −1.144% |
| 후속 Traffic | +1.042% | −0.520% |

야간 `prediction_anchor`와 후속 비교군 `full_anchor`는 raw pinball + 균일 F0 보존항이다. seed와 E 구간은 바뀌었지만 같은 두 원천·기존 train/V를 사용했다. 따라서 독립 4개 데이터셋 재현이 아니다. 그래도 raw-loss 비교에서는 네 조건 모두 평균이 좋아졌다는 관찰이 남는다.

반면 이 효과가 native loss를 쓰는 LoRA보다 항상 나은 것은 아니다. 최신 calibration 가중치는 uniform 대비 ETTh1 0.00885%, Traffic 0.11311% 나쁘고 shuffled 대비도 나빴다. **지금 근거가 있는 부분은 일반 보존 규제이며, calibration별 가중치의 추가 기여는 아니다.** native/raw 목적함수 × 보존항 유무를 비교하는 작은 요인 실험이 먼저다. native+anchor는 기존에 측정되지 않았으므로 성공을 예측할 근거로 쓰지 않는다. 보존 강도와 선택 조건은 train/V에서 고정하고 별도 E로 검증해야 한다.

2. **DualClock과 PatchPhase는 학습 예산 끝에서 선택됐다.**

두 실험 모두 세 방법 전부 LR 1e-4 / step 360이 선택됐다. [validation_endpoints.csv](validation_endpoints.csv). DualClock V는 180→240에 잠깐 악화했다가 360에서 더 낮아졌다. 단조 개선이나 수렴 증명은 아니다. 모든 방법이 마지막 후보를 선택했으므로 현재 상한이 충분했는지는 모른다. 추가 학습은 후보뿐 아니라 같은 비교군에도 제공해야 한다.

DualClock의 +0.0140%F0는 신규성이나 강한 예측 이득으로 부르기 어렵다. 다만 양의 series 비율·median·train-defined zero-heavy 효과가 함께 있어, 다음 후보를 무작정 발명하기 전에 **event branch의 기여와 train/V 학습곡선을 제한적으로 재진단할 근거**는 있다. 고정 데이터와 선택된 기간에서 효과가 작다는 사실도 그대로 남는다. PatchPhase는 거의 동률인 정확도에 더해 핵심 분산 개선이 0.276%에 그쳐 같은 우선순위를 주지 않는다.

3. **Maturity의 좋은 구간만 남겨도 특별한 보존항이 유리해지지 않는다.**

| 시간 구간 | F0 | Immediate LoRA | Maturity | TAFAS-like |
|---|---:|---:|---:|---:|
| 첫 25 origins | 0.590883 | 0.590846 | 0.596079 | 0.625768 |
| 마지막 5 origins | 0.548728 | 0.926390 | 0.955182 | 0.656163 |

기존 진단에서 마지막 5개를 제외하면 TAFAS-like에 대한 이득이 양수였지만, 그 구간에서도 Immediate와 F0가 더 좋다. 마지막 구간에서는 Maturity가 더 크게 악화했다. 이 분해는 기록된 시간 순서에 대한 사후 설명이며, 성능이 나쁜 구간을 제외하는 평가 규칙이 아니다. 보존항이 online 붕괴를 막았다는 근거가 없고, 시점별 분포 변화가 단일 원인이라고도 확정할 수 없다.

4. **Query는 정확도 주제와 자원 주제로 나눠 읽어야 한다.**

동일시간 학습에서 query-off peak는 약 948MiB, standard-off는 약 1830MiB였으므로 그 선택끼리는 약 48% 차이가 있다. 그러나 별도 고정 상태 진단에서 standard-on은 733MiB였고 query-on은 684MiB였다. checkpoint라는 강한 대안을 무시한 48% 메모리 우위 주장도, 20% 목표 미달 때문에 메모리 이점이 전혀 없다는 주장도 부정확하다. 측정 단계·학습 상태가 다른 표를 하나의 Pareto 실험으로 합치지 않는다. 실제 고정 메모리 제한에서 달성 가능한 품질과 전체 시간이 추가로 필요하다. 현재 자료는 같은 시간 정확도 우위의 논문 근거가 약하다는 판단까지다.

**기준만 느슨하게 바꾸면 살아나는가**

최근 gate에서 최소 평균 개선만 1%→0.5%→0%로 바꾸고 기존 seed 조건은 유지하는 사후 민감도 계산을 했다. [threshold_sensitivity.csv](threshold_sensitivity.csv).

- 초기 Prediction anchor만 0% 조건에서 두 데이터·seed 조건을 모두 만족한다. 0.5%에서는 미달이다.
- Drift와 Calibration은 0%로 내려도 통과하지 못한다.
- 이 계산은 새 PASS 판정이 아니다. 기존 STOP의 어느 부분이 수치 문턱에 의존했는지 보여준다.
- 메모리 오차·신규성·teacher 조건을 품질 0% 기준으로 바꿔 평가할 수 없다.

초기 7개도 대부분 단순히 1%를 못 넘긴 것이 아니라 핵심 비교군을 넘지 못했다. DualClock은 이미 WEAK로 구분돼 있었다. 앞으로는 작은 양의 신호, 차이 불명확, 측정상 음성, 학습 미실행, 구현 중단, 신규성 중복을 별도 기록해야 한다. 이번의 '거의 차이 없음'은 통계적 동등성 검정을 통과했다는 뜻이 아니다.

**공통 한계와 다음 검증의 설계**

270 fits를 실행했어도 한 Chronos-2 backbone, 자주 재사용된 train/V, 소수 데이터와 짧은 개발 평가가 중심이다. 최근 두 seed는 optimizer 반복이며, 네 채널은 Traffic의 풍부한 다채널 구조를 대표한다고 보장할 수 없다. 마지막 Calibration E는 데이터셋당 8 origins다. 구간 bootstrap은 네 시간 block뿐이고 anchor 및 calibration 차이의 참고 구간이 0을 포함한다. 다중 후보 탐색을 거친 결과를 독립 검정처럼 해석할 수도 없다.

비교군 중 E 최선값을 쓰거나 seed별 최선값을 합치는 보수적 gate는 배포 가능한 하나의 검증 선택 baseline보다 엄격할 수 있다. 그것이 후보의 checkpoint를 E로 선택했다는 뜻은 아니다. 향후에는 각 baseline과의 개별 비교, V로 선택한 실용 baseline, 보수적 E 최선 비교를 구분해서 제시하는 편이 낫다.

Time-PEFT 공식 설명은 시간적·다채널 복잡도로 미세조정의 개선 여지를 파악하고 해당 특성을 다루는 어댑터를 제안한다. 이 설명에서 얻을 설계상 교훈은 **효과가 예상되는 조건을 먼저 정의하는 것**이다. 전체 결과표의 개별 승패를 이번 검토에서 확인한 것은 아니다. [공식 저장소](https://github.com/kaist-dmlab/TimePEFT).

Conditional Path의 기존 중단은 중앙 copula 아이디어의 차별화 부족 판단이다. 기존 TSFM에서 상관된 경로를 만드는 copula 접근 자체는 선행 논문에서 확인된다. 모든 저랭크 공분산 확장이 무조건 논문이 될 수 없다는 정리로 해석하지 않는다. 새로 진행하려면 해결되지 않은 제약과 실제 방법 차이를 먼저 보여야 한다. [Baron 등, 2025](https://arxiv.org/abs/2510.02224). 다른 후보의 전체 최신 문헌 조사를 새로 완료했다는 주장은 하지 않는다.

재실험을 한다면 우선순위는 다음과 같다.

1. **목적함수에 따른 보존 규제 효과**: raw/native와 보존항 유무를 분리하고, 단순 uniform anchor를 강한 기준선으로 유지한다. 지금의 작은 개선을 새 PEFT 방법의 신규성으로 포장하지 않는다. 조건부 규칙은 train/V에서 규정한 뒤 검증해야 한다.
2. **DualClock의 학습·event branch 기여**: 같은 예산 확장을 모든 arm에 제공해 V 수렴과 event 정보를 사용하는 효과를 먼저 확인한다. 방법 효과가 미미하면 더 많은 데이터셋으로 숫자만 늘리지 않는다.

각 주제는 예상 적용 조건, baseline, 평균 집계, 악화 허용 범위, 불확실성 보고 방식, 예산 상한을 새 E를 보기 전에 정해야 한다. 승·무·패와 실패 조건도 모두 보고한다. 이미 본 데이터에서 잘 나온 조건만 골라 성공을 선언하지 않는다. 특정 조건의 재현 가능한 이득과 고유 구성요소의 기여가 확인되면 방법론 연구로 발전할 수 있지만, 이번 재검토만으로 논문 PASS를 선언할 수는 없다.

**이번 작업의 종료 상태**: 전체 기록 재검증과 사후 분석 완료. 과거 실행·판정은 불변. 새 학습이나 자동 연구 반복은 실행하지 않았다.
