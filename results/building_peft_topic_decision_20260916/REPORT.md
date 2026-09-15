# 짧은 이력 건물 PEFT — 한 후보의 직접 비교

[확인] **COMPLETE / NO_ADDED_VALUE_IN_THIS_PILOT / UNRESOLVED**. 이번 실행에서 새 방법론 주제 미확보. 이 문서는 실행 성공과 방법 발견을 분리한다. 본학습 120/120 fits 완료, 19680 optimizer updates. Smoke 10 updates 별도. 미완료 논리 셀 0/120.

## 문제와 변경의 이유

[설계] 새 건물의 3일/14일 관측만으로 다음 24시간을 예측할 때, 타깃만 학습하는 rank1 LoRA에 잔차 기반 보존 방향 배분을 추가하면 표준 LoRA와 균일 보존보다 도움이 되는지 물었다. 기존 결과에서는 학습량과 건물에 따라 수준 편향과 형상 오차가 서로 다르게 움직였다. 그 분해는 사후 진단이며 인과 증명이나 미래 입력이 아니다. [EVIDENCE.md](EVIDENCE.md)에 반례와 과거 원점수를 보존했다.

[설계] STD는 native supervised loss, SIMPLE은 균일한 F0 출력 보존항, CANDIDATE는 같은 보존항의 레벨·형상 가중치만 달리한다. lambda 척도와 파라미터·학습 기회는 동일하다. Frozen F0의 이력 내부 잔차로 가중치를 계산하며, 평가 정답으로 보정 크기를 맞추지 않는다. 정확한 식과 동치는 [MECHANISM_SPEC.md](MECHANISM_SPEC.md)에 있다.

## 정보와 비교의 공정성

[확인] DISCOVERY는 이미 노출된 기존 tune4, LOCKED는 노출 감사에서 성능 사용이 발견되지 않은 기존 heldout6이다. 새로운 건물 교체 없이 canonical ID 순으로 수요일/토요일 origin을 고정했다. 한 건물당 origin 하나이며 두 H와 두 seed를 독립 건물로 세지 않았다. 모든 worker는 H와 모델만 받는다. LOCKED 채점은 선택 봉인 뒤에 수행했다.

[설계] native Chronos-2 revision29ec3766d36d6f73f0696f85560a422f50e8498c, FP32/batch1, rank1/alpha2, 96 projections/147456 trainable scalars. LR2개×budget5종의 전역 선택 기회가 세 방법에 동일하다. 각 fit은 H3 120/H14 208 updates까지 실행했다. 선택 checkpoint와 fixed120은 같은 trajectory에서 읽었다.

- CANDIDATE: LR=3e-05, ZERO, DISCOVERY primary=1.087340406.
- SIMPLE: LR=3e-05, ZERO, DISCOVERY primary=1.087340406.
- STD: LR=3e-05, ZERO, DISCOVERY primary=1.087340406.

[확인] 선택 ZERO는 정확히 F0 반환이다. ZERO가 선택된 방법의 차이를 학습 효과라고 해석하지 않는다. 후보 개발 점수의 부호를 LOCKED 입장 gate로 사용하지 않았다. 초기 CPU 수식 prototype 작성은 STD 종료 직전이었다는 준비 순서 차이를 [RESEARCH_REVIEW.md](RESEARCH_REVIEW.md)에 기록했다. 후보 GPU 비교는 topic seal 후 하나의 고정 구현으로 진행했다.

## 실제 원점수와 추가 가치

Primary는 episode median RMSE / H의 population std(floor1e-6)를 건물 내 H·seed 평균한 뒤 건물 평균한 값이다. Raw RMSE/MAE와 scaled 2-pinball도 같은 셀에서 보고한다. 분위수 산술평균을 조건부 평균으로 부르지 않는다.

| 방법 | Primary | Raw RMSE | MAE | scaled 2-pinball | H3 primary | H14 primary |
|---|---:|---:|---:|---:|---:|---:|
| STD | 0.642842 | 18.712691 | 15.109893 | 0.367812 | 0.729294 | 0.556391 |
| STD_FIXED120 | 0.605488 | 14.604021 | 11.486483 | 0.347345 | 0.699484 | 0.511493 |
| SIMPLE | 0.642842 | 18.712691 | 15.109893 | 0.367812 | 0.729294 | 0.556391 |
| SIMPLE_FIXED120 | 0.615378 | 15.914313 | 12.378200 | 0.355946 | 0.694338 | 0.536419 |
| CANDIDATE | 0.642842 | 18.712691 | 15.109893 | 0.367812 | 0.729294 | 0.556391 |
| CANDIDATE_FIXED120 | 0.617506 | 15.856057 | 12.315974 | 0.358203 | 0.699949 | 0.535064 |
| F0 | 0.642842 | 18.712691 | 15.109893 | 0.367812 | 0.729294 | 0.556391 |
| AFFINE | 0.671559 | 17.421456 | 13.573358 | 0.419451 | 0.800973 | 0.542146 |
| SEASONAL24 | 0.613556 | 13.836929 | 10.611951 | 0.472039 | 0.697635 | 0.529478 |


- STD 대비 후보: 0.000000%, 건물 paired bootstrap 95% [0.000, 0.000]%, leave-one-building-out [0.000, 0.000]%.
- SIMPLE 대비 후보: 0.000000%, 건물 paired bootstrap 95% [0.000, 0.000]%, leave-one-building-out [0.000, 0.000]%.
- STD_FIXED120 대비 후보: -6.169243%, 건물 paired bootstrap 95% [-29.622, 3.974]%, leave-one-building-out [-8.514, 1.255]%.

사전 SCREEN 조건 각각의 결과:

- STD_gain_ge_1pct: False
- STD_each_seed_positive: False
- STD_CI_lower_positive: False
- STD_neither_H_worse_over_1pct: True
- SIMPLE_gain_ge_1pct: False
- SIMPLE_each_seed_positive: False
- SIMPLE_CI_lower_positive: False
- SIMPLE_neither_H_worse_over_1pct: True
- no_fixed_control_better: False
- not_worse_than_F0: True

[확인] F0→STD는 타깃 학습 자체, STD→SIMPLE은 균일 출력 보존, SIMPLE→CANDIDATE는 잔차 기반 방향 배분의 추가 가치를 비교한다. 선택형과 fixed120형을 함께 공개했으며 평가 뒤 더 좋은 checkpoint로 배포 설정을 바꾸지 않았다. 전체 building/seed/H 점수는 [locked_selected_and_fixed_scores.csv](locked_selected_and_fixed_scores.csv), 건물별 값은 [building_summary.csv](building_summary.csv), 효과·CI·최악 건물은 [paired_building_effects.json](paired_building_effects.json)과 [macro_scores.csv](macro_scores.csv)에 있다.

## 자원과 실제 실행

| 범위 | 방법 | Fits | Updates | Gradient 초 | Fit wall 초 | Median peak allocated MiB |
|---|---|---:|---:|---:|---:|---:|
| DISCOVERY | STD | 16 | 2624 | 274.63 | 420.36 | 525.64 |
| DISCOVERY | SIMPLE | 16 | 2624 | 278.01 | 493.99 | 525.66 |
| DISCOVERY | CANDIDATE | 16 | 2624 | 277.24 | 495.32 | 525.66 |
| LOCKED | STD | 24 | 3936 | 414.16 | 960.86 | 525.64 |
| LOCKED | SIMPLE | 24 | 3936 | 416.18 | 972.09 | 525.66 |
| LOCKED | CANDIDATE | 24 | 3936 | 415.99 | 976.63 | 525.66 |


[확인] 모든 군의 trainable 수는147456으로 파라미터 절감은0%다. 추가 fit당 실제 wall에는 모델 로딩·보존 통계·예측·파일 저장이 포함되며 gradient 시간과 다르다. 선택 checkpoint까지의 gradient 시간은 배포 adaptation의 일부 비용이지 초기화·통계·최종 추론을 모두 포함한 지연이 아니다. 선택 비용은 macro_scores에, 전체 연구 비용은 resources.csv에 있다. Calibration metadata의 시간은 history/method별 첫 호출만 보존하므로 전체 반복 통계 비용이라고 합산하지 않았다. 후보가 더 빠르거나 메모리가 적다고 사전 가정하지 않는다.

GPU monitor 8249 samples, 최소 free 8192 MiB, 비승인 외부 compute 0 samples, free<1GiB 0 samples. RustDesk만 기존 사용자 승인 예외다. 원시 monitor는 gpu_*.jsonl에 보존한다. 준비의 syntax 수정1회는 본학습 전이며 preparation_repair.json에 기록했다. 실제 오류·미실행은 [execution_errors.json](execution_errors.json), [UNEXECUTED.md](UNEXECUTED.md)에 있다.

## 독립 검산과 재현 범위

[확인] 독립 scalar 재계산 3300개 값, 최대 절대차 1.4210854715202004e-14. 이전 1810개 파일 hash 불변. 실제 checkpoint 600개, 이력·최종 예측 5100개를 fit마다 새 모델에서 복원해 exact equality 검사했다. GPU 검산 optimizer updates=0. CPU 수식/gradient 검사, 실제모델 smoke, 동결/finite/step0/poison 검사와 데이터·소스·모델 seal을 함께 검증했다. 검산 오류: 없음.

[확인] GitHub에는 코드·프로토콜·점수·hash manifest를 공개한다. 원자료, npz 예측, pt 가중치는 ignored 로컬 cache에 남아 있으므로 GitHub만으로 수치 replay가 가능한 완전 배포물이 아니다. 새 검산 코드는 실험 소스와 분리했으며 최종 hash를 검산 기록에 남겼다.

## 신규성 한계와 남은 구멍

[확인] 비등방 정규화와 사전학습점 보존은 알려진 원리다. [L2-SP](https://proceedings.mlr.press/v80/li18a.html), [TILDE-Q](https://arxiv.org/abs/2210.15050), [Meta-LoRA](https://arxiv.org/html/2608.12389v1)와의 경계 및 나머지 직접 읽은 선행은 [LITERATURE_BOUNDARY.md](LITERATURE_BOUNDARY.md)에 정리했다. 특정 잔차 배분 식이 동일하다는 확인과 최초성 입증 모두 현재 없다. 성능이 좋아도 UNRESOLVED를 임의 승격하지 않는다.

[추정] H3의 두 과거 창에서 잔차 일관성을 추정하는 것은 불안정할 수 있다. 다음날 운영이 바뀌면 과거 일관성이 미래 편향 수정 방향을 보장하지 않는다. 이 실행은 경쟁 설명을 모두 인과 분리하지 못한다. 건물6개/원점1개/공개 원천1개, 동일 site 가능성, 전체기간 연속 관측 필터 재사용, foundation pretraining 중복 미확인은 일반화 한계다. Bootstrap은 건물 단위2000회이며 작은 표본의 정교한 모집단 추정으로 해석하지 않는다.

[확인] 최종 투자 판정은 **NO_METHOD_TOPIC_THIS_RUN**, 다음 결정은 **종료**. 계약의 판정 기준·학습률·데이터·후보를 결과에 맞춰 바꾸지 않았다. 이번 run에서 새 후보를 자동 생성하거나 추가 학습하지 않는다.


## 구성요소별 추가 가치와 자원 이득

아래 비교는 사전에 남기기로 한 선택 recipe와 고정120을 분리한다. 평가에서 더 좋은 고정120을 찾아 기존 ZERO 선택을 바꾸는 배포 정책이 아니다. 양수는 비교 대상보다 좋은 방향이다.

| 추가 구성요소 | 기준 primary | 변경 primary | 이득 % | 건물 paired95% CI % | raw RMSE 이득 % |
|---|---:|---:|---:|---|---:|
| 타깃 LoRA 학습 / 선택 recipe | 0.642842150 | 0.642842150 | 0.000000 | [0.000, 0.000] | 0.000000 |
| 균일 보존 / 선택 recipe | 0.642842150 | 0.642842150 | 0.000000 | [0.000, 0.000] | 0.000000 |
| 잔차 방향 배분 / 선택 recipe | 0.642842150 | 0.642842150 | 0.000000 | [0.000, 0.000] | 0.000000 |
| 타깃 LoRA 학습 / 고정120 | 0.642842150 | 0.605488118 | 5.810763 | [-4.138, 22.853] | 21.956595 |
| 균일 보존 / 고정120 | 0.605488118 | 0.615378497 | -1.633456 | [-8.951, 1.949] | -8.972131 |
| 잔차 방향 배분 / 고정120 | 0.615378497 | 0.617506410 | -0.345789 | [-1.124, 0.376] | 0.366066 |

| 방법 vs STD | 파라미터 감소 % | 전체 trajectory median peak allocated 감소 % | fixed120 gradient 시간 감소 % |
|---|---:|---:|---:|
| SIMPLE | 0 | -0.004180 | -0.504847 |
| CANDIDATE | 0 | -0.004180 | -0.454534 |

[확인] 세 방법의 선택은 모두 ZERO이므로 선택된 배포의 adaptation updates는0이고, 학습 시간 비율0/0은 N/A다. 표의 메모리는 전체 trajectory에서 측정한 peak이며 ZERO 배포의 추가 메모리 측정치가 아니다. fixed120 시간은 gradient 구간만의 값이다. Fit wall은 점점 커지는 원장의 저장 비용과 고정 실행 순서에도 영향을 받으므로 방법 자체의 속도 차이라고 단정하지 않는다. 메모리·시간이 같다는 수학적 주장은 하지 않는다.

[확인] 공통 참조 20 episodes의 계산을 독립 복원했고 affine ill-conditioning fallback은 0회였다. 이력 20개의 native F0와 과거 잔차로 lambda·레벨/형상 가중치를 독립 재계산했다. [후처리 개발 기록](POSTPROCESSING_REVIEW.md)은 학습 오류와 별도로 보존한다.


평균 개선과 불확실성, 선택 전달 실패 및 새 변경의 추가 가치를 분리한 해석은 [INTERPRETATION.md](INTERPRETATION.md)에 정리했다.
