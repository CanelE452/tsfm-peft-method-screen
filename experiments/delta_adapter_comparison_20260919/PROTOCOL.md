# δ-Adapter와의 통제된 직접 비교

2026-09-19. 기준 `36d6853`. 새 후보를 만들거나 C3/MAG/PLAIN을 재튜닝하지 않는다. 전체 방법론 목표에서 빠진 가까운 선행의 실제 학습 비교를 보완한다. 이 단위의 예산은 새16fits / 16,384main +8smoke optimizer updates다. 초기 방향 진단을 성능 gate로 사용하지 않는다.

## 선행 구현과 이전 결과 재사용

공식 `Anoise/Adapter` commit `0add06ea7b4d2e0a84c364a8be72eef2676a92f2`의 `Adapter-X+Y/experiments/exp_online_xy_add.py`를 기준으로 한다. Linear→ReLU→InstanceNorm1d 두 번→Linear→tanh×δ인 additive XY cell, δ=.1이다. 원래 별도 Y-only 파일은 BatchNorm/δ1로 다르므로 그것을 동일 cell이라고 하지 않는다. 독립 구현의 CPU output/input-gradient를 원본 class와 비교한다. 논문의 전체 backbone·MSE 프로토콜 재현은 아니다.

공식 XY의 output-net 생성은 seq_len을 input dimension으로 쓰지만 실제 입력은 예측값이다. 512→64에 연결할 때 output dimension을64로 지정해야 한다. 공식512 hidden 폭을 유지한 DELTA_XY_DEFAULT와, 기존8,712개 추가 adapter와 가깝게 parameter-count 차이를 최소화한 hidden7의 DELTA_XY_BUDGET 두 가지를 고정한다. 용량을 줄인 대조만으로 선행을 약화시키지 않는다. budget군은 완전 동일 parameter count가 아니며 정확한 수와 비용을 보고한다.

## Chronos 연결 및 정보 권한

기존 trained B0와 Chronos-Bolt-small을 그대로 고정한다. x는 관측512, sigma는 기존 TRAIN population scale다. center는 관측 x의 평균이다. 입력 cell은 (x-center)/sigma를 읽고 ±.1sigma 이내 보정을 원래 x에 더한다. B0 출력9quantile 각각을 같은 output cell에 넣고 각 quantile의64개 horizon을 함께 읽어 ±.1sigma 잔차를 더한다. quantile index/state/원본 clean x/생성 mask/합성 delta/future y는 cell 입력이 아니다. output cell은 모든 quantile에 공유한다. 이는 point-forecast 선행을 quantile backbone에 연결하는 공개된 변경이며 공식 quantile-calibrator의 재현이 아니다.

공식 cell의 초기화는 유지한다. random residual 때문에 초기 예측이 B0와 완전히 같을 필요는 없지만, adapter off 경로의 B0 동일성·동결 가중치 보존·실제 update·checkpoint 복원을 검사한다. 원래 C3/MAG/PLAIN 결과를 이 초기화로 바꾸지 않는다. 기본9개 quantile normalized2pinball은 기존 비교와 동일하다.

## 공정한 학습·선택

Electricity/ETTm1의 기존 봉인 TRAIN/V/E origins·4채널·32epoch synthetic draws·labels·sigma를 유지한다. train arrays는 이전 C2/C3와 hash 동일성을 확인한다. batch/micro32, FP32/TF32off/dropout0. AdamW(beta .9/.999, eps1e-8, wd0), gradclip1, 1024updates, checkpoint0/256/512/768/1024. 같은 source/seed/epoch의 순서는 기존 rng(84100,source,seed,epoch)로 동일하다.

두 LR 1e-4/3e-4는 selection seed81550에서 각각 학습한다(2원천×2군×2LR=8fits). 기존 다섯 V 상태 REFERENCE/POINT8/BURST8/SHIFT4/SHIFT8의 평균 median nMAE로 LR와checkpoint를 선택한다. 동률은 낮은LR/이른step. 그 LR로81551/81552를 반복한다(추가8fits). 전체16fits를 완료하며 중간 성능이 나쁘다는 이유로 다른 군을 생략하지 않는다. 학습 실패/안전 차단은 성능과 구분한다. GPU는 RustDesk만 외부 compute 예외로 유지한다.

기존 B0/PLAIN/C3/MAG_ONLY의 2seed·selected/fixed1024 예측96views를 해시로 재사용한다. 새로운 두 군은2seed×2정책×3패널(Electricity/ETTm1/같은전력의16계열)×2입력형식으로48views다. 전체144views를 보관한 뒤 E labels를 채점한다. trained B0 식별/seed/source가 맞는지 확인한다. 같은 source의 새 계열을 독립 dataset이라고 부르지 않는다.

## 해석 범위

주 비교는 기존 narrow claim인 전력16계열 selected SHIFT8에서 C3와MAG 각각 대 δ두 군의4개 차이다. 날짜 index-week block bootstrap2000, family4 Bonferroni95% 구간을 사용한다. 두seed의 방향·원점수 및 fixed1024를 함께 낸다. 유의성만으로 논문 PASS를 선언하지 않는다. 모든 원천의 REFERENCE/FAULT/SHIFT4/SHIFT8/SHIFT_POINT와9개 형태 결과를 보존한다. PULSE의 구별 불가능성으로 SHIFT8의 손해를 설명하지 않는다.

C3가δ보다 좋아도 MAG에 대한 지속성 규칙의 추가 가치가 자동 생기지는 않는다. MAG가 좋아도 사후 선택한 규칙의 신규성·독립 확인이 완료된 것은 아니다. 정확도와 parameter/GPU memory/wall 비용을 분리한다. 공식δ의 다른 변형·δ범위·hidden폭·LR grid를 추가 탐색하지 않으며 모든 설정을 결과 전에 봉인한다.

완료 후 한국어 REPORT/FINAL_DECISION과 원점수·검산·그림을 commit/push한다. 이 비교에서 새 모델 구조/다른 dataset/추가학습을 자동 연결하지 않는다.
