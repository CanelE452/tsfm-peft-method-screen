# 방법론 전제 검사 — 새 학습 전 고정 범위

2026-09-19. 전체 목표는 새 PEFT 방법론의 근거 확보다. 본 검사는 그 목표의 준비 단계이며 분석논문으로 목표를 바꾸지 않는다. 기존 TRP/C3 실험과 설정·결과를 합치지 않는다. 이 파일을 봉인한 뒤 실제 gradient를 계산한다.

## 완료한 결과의 보존 및 관측 한계

기준 commit `9d688dda3baec90c521c42e029790ceaf4c676ff`의 실험은 종료 상태로 보존한다. 이미 알려진 PULSE/PAIRED_SHIFT의 동일 입력·다른 정답을 저장 예측으로 수치화한다. 3패널×8군×2seed×2checkpoint 정책의96개 shape 예측 view만 재사용한다. 새 inference/fit/update0. 이것은 TRP 주 비교의 원인이 아니다. TRP 주 비교는 별도의 standard SHIFT8이다. oracle interval은 사후 검산 전용이며 새 모델 입력으로 쓸 수 없다.

## 실제 모델의 초기 방향 점검

- 새 가설: 서로 다른 TRAIN 날짜 블록에서 동의하는 gradient 방향을 이용하면, 단순 평균 gradient의 고유방향보다 이후 TRAIN 기간의 손실 감소 방향을 더 잘 잡을 수 있는가?
- 일반 예측 범위의 탐색이다. 기존 합성 오류·변화 상태, C3, TRP, IDEAL의 학습은 재개하지 않는다.
- 모델: 기존 hash가 검증된 **원본 Chronos-Bolt-small F0**. 학습된 B0 checkpoint를 불러오지 않는다. 기존 raw B0 wrapper를 만들더라도 LoRA의 B=0을 검사해 초기 출력이 원본과 같아야 한다.
- 데이터: 기존 Electricity/ETTm1의 TRAIN256 origins, 기존4채널, 입력512/출력64, TRAIN population sigma를 그대로 읽는다. 합성 변형0. V/E 파일은 이 gradient runner에서 열지 않는다.
- 각 원천의 날짜순 첫96 origins는 calibration, 마지막96은 후반 TRAIN 진단, 가운데64는 버린다. 각8origins×4channels=32개의12블록. 두 집합 사이 input+target 비중첩을 검사한다. 각 집합 내부의 겹침은 기록하며 독립 표본이라고 가정하지 않는다.
- FP32, TF32 off, dropout0, 모델 eval. 기본 quantile9개 평균 normalized 2pinball loss. 모든 q/v의36개512×512 weight에 대한 dense gradient를 읽는다. weights를 업데이트하지 않는다.
- 최대48개의 실제 autograd 호출(2원천×24블록), 추가 원본 출력 동일성 forward2회. **optimizer update0, 새 fit0, V/E 평가0**. gradient 상태의 CPU 수학 검사는 별도다. GPU에 RustDesk 외 외부 compute가 있거나 공유 lock을 얻지 못하면 실행하지 않는다.
- 각 층에서 calibration gradient G_k로 M=mean(G), S=M^T M, Q=mean(G_k^T G_k), C=(K*S-Q)/(K-1)을 계산한다. C는 알려진 off-diagonal U-statistic이며, 시간 블록의 독립성·동일 분포를 실제로 보장하지 않는다. 새로운 일반 수식의 발명이라고 쓰지 않는다.
- rank8 고정. 비교는 RANDOM_ORTHO(seed91932+층번호), MEAN_SVD(S의 상위8), SECOND_MOMENT(Q의 상위8), CROSS_BLOCK(C의 상위8) 네 개. full gradient는 용량이 다른 진단 상한 참고이며 PEFT 대조군이 아니다. C의 음의 고유값과 양의 rank 수를 기록하며 결과를 보고 clip·rank·ridge·블록 경계를 조정하지 않는다.
- 각 방향 D=M*V*V^T를 고정한 뒤 후반 TRAIN의 G_hold로 내적 `<G_hold,D>` 및 cosine을 계산한다. positive inner product는 음의 update 방향에 대한 1차 손실 감소 예측이다. 실제 학습 이득이나 Adam의 최종 경로와 동일하지 않다. 층을 합친 전체 inner product와 전체 update norm 기준 cosine을 함께 보고 층을 골라 승자를 만들지 않는다.
- 이 검사는 계열·날짜 수·LR·seed를 탐색하는 학습 sweep이 아니다. calibration M에서의 손실 포착은 MEAN_SVD가 최적이므로 이를 CROSS의 성공 기준으로 사용하지 않는다. 후반 TRAIN의 두 원천 모두에서 CROSS가 MEAN보다 방향 정렬이 좋고 양의 감소 방향이어도, 이는 후속 구현 검토 근거일 뿐 논문 PASS·학습 자동 시작 조건이 아니다.
- 선행: LoRA-GA, LoRA-One, FILet, FCCA와 비교해야 한다. MEAN_SVD는 공식 LoRA-GA 재현이 아니다(양쪽 행렬 초기화·스케일·base 보상과 다름). 현재 검사는 새 방법의 학술적 독창성을 확정하지 않는다.

## 실행 후 처리

원점수·블록·gradient hash·GPU 비용·동결 weight hash·구현검사를 한국어 REPORT에 남긴다. V/E 선택·학습 결과를 만들지 않는다. 실패한 가설은 종료 상태로 보존하고 이번 한도에서 다른 후보로 자동 전환하지 않는다. 후속 본학습은 이 진단 및 선행과의 실제 차이를 검토한 뒤 별도 고정 실험 단위로만 구성할 수 있다.
