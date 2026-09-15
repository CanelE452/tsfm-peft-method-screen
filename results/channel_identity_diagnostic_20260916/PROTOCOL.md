# 표현 교체와 초기 표현 보존의 제한 진단

2026-09-16. 기준 cbb1f844be82419f2077f39aee4ae643ad598e8c. 이 문서는 첫 신규 GPU 업데이트 전에 봉인한다.

## 목적과 범위

[설계] 큰 채널 모듈의 추가 가치가 없었던 이유 중, 사전학습 표현을 처음부터 교체하는 경로가 불리했는지 확인한다. 기존 120-fit 건물 실행은 종료 상태 그대로 보존한다. 이는 사용자의 실패 원인 재검토·가능한 연구 방향 탐색 요청 아래 별도로 수행하는 개발 진단이다. 새 방법론 후보를 발견했다고 전제하지 않는다.

[설계] 신규 비교군 하나는 기존 SHARED_BUDGET의 출력 u(h,f)를 h + a*u(h,f)로 바꾼다. a는 0으로 초기화한 학습 가능한 스칼라이고 나머지 모듈은 기존 동일 seed·초기화다. 알려진 ReZero 연결의 적용이다. 추가 파라미터 1개이며 기존 SHARED_BUDGET보다 가벼워지지 않는다. backbone LoRA와 head는 동일하게 학습한다. 초기 eval 출력은 같은 seed의 LH와 동일해야 한다. 학습 중 adapter dropout의 난수 소비 때문에 LH와 완전히 동일한 학습 궤적은 아니다.

핵심 대조: 기존 완료 LH 및 SHARED_BUDGET의 같은 dataset/seed 결과를 해시 검증 후 재사용한다. 각각 4 fits를 다시 학습하지 않는다. 원래 Time-PEFT 전체 재현으로 부르지 않는다. Time-PEFT와 달리 이전 파일럿의 작은 shared-channel 구조를 그대로 대조한다.

## 고정 실행

- 기존 R2 electricity/traffic 첫 32채널, 동일 train70%/V10%/E20%, context96/horizon96. 기존 E는 이미 노출됐으며 이번에도 DISCOVERY_REUSED_E다. 독립 잠금 평가나 확증 PASS는 없다.
- MOMENT-small 기존 로컬 revision과 환경 .venv-channel, LoRA rank8 alpha32 q/k/v, BF16·FFT FP32, effective/micro batch8 그대로.
- AdamW lr0.001, weight_decay0, betas(0.9,0.999), eps1e-8; norm clip1; StepLR5epochs gamma0.5.
- 기존 schedules.json 그대로, 20epochs 최대, V min_delta0.0001 patience5. INIT 포함 최소 V MSE checkpoint 선택. 모든 설정·데이터·선택 규칙 변경 금지.
- electricity/traffic × seed41000/41001 = 신규 최대4 fits, 최대5080 updates. smoke는 각 원천2 updates, 총4. 실패 attempt도 예산 차감; 재튜닝·재시도·대체 fit 없음.
- 네 fits의 V 선택이 끝나면 봉인 후 기존 E에서 측정. 학습 이후 다른 후보 자동 실행 없음.
- 한 GPU 한 worker, 기존 gpu.lock. RustDesk만 예외. 안정30초/free>=4GiB, running free>=1GiB/비승인compute면 대기, 누적대기600초, 실행wall3600초. 오류는 성능 실패와 분리.

## 검사와 판단

출처·데이터·재사용 점수/예측 해시, 초기 LH eval 동일성(FP32 atol1e-6 rtol1e-5, BF16은 같은 shape에서 exact), frozen 불변, finite loss/gradient/output, 실제 gate/adapter/head/LoRA 업데이트, checkpoint 새 모델 재생 exact, 독립 float64 MSE/MAE 검산. 첫 step의 gate 외 adapter gradient0은 정상이다.

주지표는 기존 normalized channel-macro MSE. raw MAE, V 선택epoch, 업데이트·시간·메모리를 함께 공개한다. 기존 SHARED 및 LH 각각 대비 모든 원천/seed 효과를 그대로 보고한다. E로 epoch/설정을 재선택하지 않는다. 과거 E를 이미 본 탐색이므로 신뢰구간을 독립 확증처럼 제시하지 않는다.

[설계] 원천별 두 seed 모두 SHARED보다 좋아지면 '표현 보존의 개발 신호'; LH도 모두 넘어야 '복잡한 모듈의 추가 가치에 관한 개발 신호'로 구분한다. 이는 과거 PASS 기준의 변경이 아니라 새 원인 진단의 서술 규칙이다. 자원 이득과 예측 이득은 분리하고, 좋아져도 KNOWN_METHOD_USEFUL 이하로만 기록한다. 좋지 않아도 모든 PEFT가 불가능하다는 결론을 내리지 않는다.

[한계] 잔차 경로와 0 gate를 동시에 도입하므로 둘의 개별 인과효과는 분리되지 않는다. CPU 구조 검사는 수치적 성능 실패의 원인 증명이 아니다. 파라미터 공유와 추론 중 타 채널 의존성은 다르다.
