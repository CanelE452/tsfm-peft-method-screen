# PETSA 보정 부품 대조 — 추가 실행 승인 전 고정안

상태: PREPARED_NOT_AUTHORIZED. 이전 자동시작 승인은 16 fits를 완료하며 소진했다. 이 문서는 추가8 fits 승인 요청의 정확한 범위이며, GPU smoke를 포함한 optimizer update는 사용자 승인 전 실행하지 않는다. 승인 시 이 문서 SHA를 AUTHORIZATION.json에 기록한다. MAG를 다시 설계하거나 학습하지 않는다. 이번 군은 선행 부품 대조이며 새 제안 후보가 아니다.

## 고정 방법과 선행 변경점

공식 PETSA 87853d888e98311ac94e64be920d17b57143b20c의 GCM, var_wise=True, rank16, gamma 초기 .01, A Kaiming, B/bias0을 사용한다. 공유 단일계열 입력512 cell과 출력64 cell: 16897+2113=19010 추가params. source/seed에 대응하는 학습된 Chronos-Bolt-small B0 전체를 동결한다. 초기 예측은 B0와 정확히 같다.

좌표는 이전 δ XY 대조와 같이 관측창 mean과 기존 TRAIN sigma를 쓴다. x에 sigma*GCM.correction((x-mean)/sigma)를 더하고 B0에 넣는다. 예측9quantiles 각각에 같은 출력 GCM을 적용한다. zero correction일 때 불필요한 normalize/inverse 반올림을 피하도록 correction만 raw 값에 더한다. quantile 정렬·추가 보정은 하지 않고 crossing을 보고한다. gate나 model에 future/clean x/state/fault mask/true delta를 주지 않는다.

이는 공식 부품의 **offline 통제 이식**이다. 공식 PETSA의 online partial/delayed labels·복합 loss·전체 runner를 재현한 것으로 부르지 않는다. Chronos 좌표·quantile 공유는 이식 변경이며 공식 설정으로 위장하지 않는다. MAG8712와 파라미터 수가 같지는 않다.

## 고정 학습과 예산

두 원천 Electricity/ETTm1 × 한 군 PETSA_XY_OFFLINE. selection81550에서 LR1e-4/3e-4 =4 fits, 선택LR로81551/81552 반복=4 fits. 총8 fits,8192 main updates. 각1024updates=32epochs×32, checkpoint0/256/512/768/1024. 기존 TRAIN draws·labels·sigma·B0를 hash 재사용한다. batch/micro32,FP32,TF32off,dropout0,AdamW(.9,.999),eps1e-8,wd0,gradclip1. 초기cell seed=seed+200000, 순서rng(84100,source,seed,epoch). normalized2pinball 학습; V REFERENCE/POINT8/BURST8/SHIFT4/SHIFT8 평균nMAE로만 선택. 동률 낮은LR/이른step. 기존 각 대조와 학습·선택 기회를 맞춘다.

GPU smoke는 두원천×2updates=4회만. 실제 update·초기B0·동결parameters/buffers·cell gate 갱신·fresh restore를 검사한다. CPU 검사는 공식cell의 nonzero 출력 및 input/parameter gradients와 실제Chronos 초기연결·복원이다. CPU 통과는 GPU smoke/본학습 완료가 아니다. 서로 다른precision/batch update의 완전동일성을 요구하지 않는다.

RustDesk만 외부compute 예외. 시작free>=4GiB, 실행중>=1GiB, disk>=10GiB, wall cap3시간. 공통 GPU lock과 사용량 기록을 적용한다. 애매한 optimizer intent나 partial epoch는 임의 replay하지 않는다. 안전/구현 중단과 성능 부진을 구분한다. 한 조건이 나쁘다고 다른 조건·seed를 생략하지 않는다.

## 평가·정보 노출

기존 학습형gate 결과192prediction views를 SHA로 재사용한다. 새군의 두repeat seed×selected/fixed1024×electricity/electricity_transfer/ettm1/NESO후반에 대응하는 trained source×standard/9shape =32 새views. 총224. 선택봉인→새 전체예측저장→새 E채점. 동일checkpoint alias는 hash로 식별하고 실제추론수와view수를 구분한다.

**모든 E 패널은 이제 이미 채점한 개발 평가다.** NESO2026 Jul–Aug55일도 앞 실험에서 노출됐으며 새 독립 평가·새 원천으로 부르지 않는다. origin·조건·채널·sigma·원본변형을 변경하지 않는다. 추가자료·날짜·seed를 확보하는 실험이 아니다.

## 사전 판단과 산출물

주family2: selected SHIFT8에서 MAG 대 PETSA cell, electricity_transfer와NESO후반 각각. 고정두seed·채널 조건부 index7일block bootstrap2000, 양측Bonferroni family2 95%구간. RNG91942. 제한된 추가 대조 지지는 두보정구간하한>0, 네개seed별gain>0, 두패널REFERENCE/FAULT에서MAG가B0/PLAIN/PETSA cell보다 평균nMAE를 악화시키는 정도가각각<=1%일 때다. 평균손해한도와통계적비열등성을구분한다. 미충족은 이 대조의 추가 지지 미확보이지 기존 양성결과 삭제가 아니다. 기존family4·역사탐색을이번family2로보정했다고하지않는다.

전체 원점수·seed·9shape·ETTm1·fixed1024·crossing·자원·단순대조추가가치를 한국어 REPORT.md/FINAL_DECISION.md와그림에 남긴다. 모든원점metric·선택·해시·구간 독립검산 후scoped push. 순수속도비는과거측정경계가달라주장하지않는다. 승인 단위종료후자동후속0. 공식방법전체우위·범용PEFT우위·논문PASS는선언하지않는다.

실행 진입점: `.venv/bin/python -m experiments.petsa_cell_comparison_20260919.runner`. 승인기록이 없으면 첫 단계에서 차단한다. 실제 모델의 CPU 연결·gradient·복원은 검산했다. GPU smoke와 본학습은 미실행이며, 데이터·코드 hash 봉인과 승인 전 차단 검사를 별도로 기록한다.
