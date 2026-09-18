# C3 내부 표현·gradient·B0 조합 진단

2026-09-18. 사용자 ‘확인해줘’에 따른 사후 기전 진단. 기준 c41c96e의 두 source·32경로·기존 입력·모델·origin을 유지한다. 신규 fit와 optimizer update는0회다. 새 방법·LR·seed·선택을 만들지 않는다.

## 봉인 범위

1. 초기 gradient: 실제 TRAIN epoch0의1024문맥 전체를 index순 batch32로 사용한다. 이는 전체 epoch의 공통 gradient 측정이며 실제 첫 optimizer batch와 구분한다. 실제 first batch는 기존 rng(84100,source,order_seed,0)의 첫32개로 별도 비교한다. epoch0만의 증강 노출 범위(주로 amplitude4, SHIFT4)를 숨기지 않는다.
2. B0 두 개의 입력 patch 표현 h, 초기 GELU feature z=GELU(Dh+d), gate 및 초기 예측을 대조한다. U=0,u=0에서 ∇D=0, ∇U=Σ(cap·gate·r)zᵀ, ∇u=Σ(cap·gate·r), r=∂loss/∂수정patch 표현의 닫힌 식을 실제 autograd와 검산한다. 상대L2≤1e-4 및 최대절대차≤2e-6. 기대한 방향이나 성능은 통과 조건이 아니다.
3. 초기 loss 출력 gradient v와 downstream Jacobian의 역할을 분리한다. 두 B0에서 v를 서로 바꿔 J_Bᵀv와 추가 어댑터 gradient를 계산한다. 이는 고정 함수의 미분 진단이고 virtual cotangent 교환이지 정당한 타깃 교체·학습·새 방법이 아니다. ΔG를 두 변경 순서의 평균으로 J항+v항에 정확히 분해한다. norm·cosine을 보고하며 원인 확률·최종 성능 기여율로 바꾸지 않는다.
4. 모든32경로의0/256/512/768/1024 checkpoint에서 같은128 TRAIN probe를 사용한다. epoch[0,5,18,23] 각각8개 균등 index-day×4채널=32개씩. REFERENCE/POINT/BURST/SHIFT가 균형이고 SHIFT4/8 및24/48 길이를 포함한다. 실제 훈련 중 그 순간의 batch gradient가 아닌 고정 공통 probe다. loss, Up/Down gradient, residual 크기, tanh 포화, 초기값 대비 파라미터 이동과 gate 간 gradient 방향을 기록한다. 저장 final Adam moments도 순서 수준 간 대조한다. order 요인은 배치 구성과 순서를 함께 바꾼다.
5. fixed1024의32개 저장 어댑터를 다른 B0에 그대로 붙여 학습 B0×수신 B0를 교차한다. 기존96개 E view를 재사용하고 교환96개를 새로 예측한다. 세 패널의 표준10조건·형태9조건 전부 포함한다. gate·초기화·순서·어댑터 가중치를 유지한 함수 개입이다. 원래 B0와의 짝이 언제 유리한지 평가하되 배포 모델로 선택하지 않는다.
6. 전체192개 E 예측이 저장된 뒤 채점한다. 원점 nMAE/pinball과 원래96view의 집계를 재현한다. 평균 pairing penalty=mean(교환 오차−원래 오차), 수신 B0별·gate별·init별·order별 원점수 전부 보고한다. 2000회7 index-day paired bootstrap, base seed90418+panel index, 패널·조건별 C3/MAG2대조 Bonferroni 구간. 고정 모델·두 B0에 조건부이며 독립 test가 아니다.

## 계산·안전·해석

GPU autograd 호출은최대1280회, 새 전체 E prediction view는최대96개. 함수 검사용 소규모 forward는 별도 기록한다. optimizer 객체 생성·step 및 신규 적합은 금지한다. 물리적으로 저장된 모델·buffer가 변하지 않았는지 hash로 검사한다. CPU 합성 chain-rule·분해 검산을 추가한다.

기존 GPU lock·30초 안전 확인·RustDesk만 승인 예외·실행 중 외부학습 감지 pause를 사용한다. free GPU1GiB/RAM2GiB/disk10GiB 하한, 대기600초, runner2시간 한도. 오류는 보존하고 결과에 맞춰 tolerance·probe·대조를 바꾸지 않는다.

초기 gradient 경로, 저장 가중치의 학습 후 변화, B0 교환의 실제 예측 효과를 구분한다. gradient 변화가 최종 성능 변화의 완전한 원인이라는 주장은 하지 않는다. 내부 한 층의 해석을 전체 시스템으로 확대하지 않는다. 최종 REPORT.md와 FINAL_DECISION.md, 논문 주장 보충·그림·검산을 작성하고 scoped commit/push한다. 자동 후속학습은 없다.
