# C3/MAG 학습 원인 분리 — 고정된 2×2×2 통제 비교

2026-09-18. 사용자 ‘해줘’에 따라 B0·어댑터 초기화·학습 순서를 독립적으로 통제한다. 기존 결과를 본 뒤 설계한 사후 기전 진단이며 독립 확인 시험이나 새 후보 선택이 아니다. 기존 C3/MAG, 데이터·입력·loss·optimizer·정보 권한은 바꾸지 않는다.

## 범위와 예산

- source Electricity, ETTm1. Electricity의 기존 source4·전이16계열과 ETTm1 기존4계열을 평가한다. 모든 기존 표준10조건·형태9조건 및 FAULT 집계를 유지한다.
- 세 이진 요인은 B0 checkpoint의 seed, 추가 어댑터 초기화 seed, epoch별 permutation seed다. 각 요인 수준은81551/81552. 이 두 실행은 기존 진단에서 같은 방향이지만 크기가 크게 달랐던 두 실행이므로 선택했다.81553과 기존 모든 음성 결과는 원 보고서·원고에서 그대로 유지한다. 이번 범위는 세 seed 전체 또는 랜덤 초기화 모집단으로 일반화하지 않는다.
- 2source×8조합×2gate=32경로. 대각선 b=i=o의8경로는 원 checkpoint·명세·receipt를 확인해 재사용한다. 새 경로는24개×1024=24,576 본학습 updates. smoke는2source×2gate×3=12updates. 최대 합계24,588. 재학습으로 기존 결과를 복원하거나 다른 후보를 추가하지 않는다.
- 두 gate는 같은 source/B0/초기값/순서 조건에서 짝지어진다. LR은 Electricity0.0003, ETTm10.0001, FP32/TF32off, batch32/microbatch32, AdamW(0.9,0.999), eps1e-8, decay0, clipnorm1, 동일 TRAIN·draws·정답·32epochs. LR 검색 없음.
- 주 분석은 모든 경로의 고정1024 checkpoint. 보조 분석은 기존 동일 V objective와0/256/512/768/1024 중 최소오차(동률 작은 step)의 선택 결과다. 이전 selected 효과와 고정종료 효과를 혼동하지 않는다. 선택이 끝나기 전 E를 채점하지 않으며 전체 E 예측을 먼저 저장한다.

## 추정 대상과 분석

- 주 결과는 전력 전이 SHIFT8의 d=nMAE(C3)−nMAE(MAG), 양수는 MAG 방향. 나머지 패널·조건을 전부 같은 방식으로 보고한다.
- 각 요인 주효과는 나머지 두 요인을 동일 가중 평균한 d(high)−d(low). gate별 원오차의 주효과도 따로 보고한다. 요인과 gate의 차등 효과를 기본 예측 난이도와 구분한다.
- 3요인7개 직교 factorial 성분과 조합별 paired d를 보존한다. 요인 contrast는2*mean(d*sign_product), 절편은mean(d). 세 요인의 조합은 각gate1회씩 학습하므로 셀 내 확률적 반복이나 seed 모집단 분산을 추정하지 않는다. 주효과끼리의 크기와 상호작용을 구분하고 가장 큰 상호작용을 단일 원인으로 바꾸어 말하지 않는다.
- 기존 date-block 방식(7 index-days)으로2000회 paired bootstrap, 고정 seed89418. 같은 자료의 조건별 재추출을 공유한다. 7contrast의 Bonferroni family 구간을 병기한다. 구간은 선택한 모델·요인 수준에 조건부이며 E 재사용·사후 설계의 한계를 제거하지 않는다. 음성 조건·계열·날짜를 제외하지 않는다.
- selected와fixed1024의 차이는 checkpoint 선택 민감도이며 추가 요인의 무작위 인과효과로 해석하지 않는다.

## 안전·검사·중단

RustDesk만 기존 승인된 외부compute 예외. GPU 30초 안전 확인, 실행 중 상시감시·외부학습 시 pause, GPUfree<1GiB/CPU RAM<2GiB/disk<10GiB/대기600초/전체4시간 시 중단. 기존GPUlock 공유. 수치오류·불명확한 update는 결과를 보존하고 자동 재시도하지 않는다. epoch 경계의 정확한 optimizer 상태만 resume한다.

실제 모델 smoke에서 B0 초기 동일성, 동결가중치·buffer 보존, 추가파라미터 갱신, 저장/복원, residual-off B0 복원, 입력 비변경을 검사한다. 합성 factorial 수학 검산과 원점별 scalar metric 독립 계산을 수행한다. 모든 선택·예측·부모hash를 봉인한다.

최종 한국어 REPORT.md, FINAL_DECISION.md, 기여도·원점수CSV, 그림, 독립검산을 작성하고 논문 근거에 추가한다. 단순한 효과가 크면 그대로 인정한다. 새 구조나 자동 후속학습은 시작하지 않는다.
