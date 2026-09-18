# 고정 가중치의 C3–MAG_ONLY 영향 진단

사용자의 “어떤 게 영향을 주는지 확인” 요청에 따른 사후 진단이다. 기준은 c07d4af의 완료된 30-fit 비교다. 이전 규약을 다시 실행하거나 후속 학습을 자동 시작하지 않는다.

- 기존 Electricity4/transfer16/ETTm1의 E 원점, 10개 standard 조건, 9개 shape, seed81551/52/53을 모두 유지한다. 이미 본 개발 E이며 독립 확증이 아니다.
- 가중치와 선택 checkpoint를 고정한다. A=(C3 weights,C3 gate), B=(C3 weights,MAG gate), C=(MAG weights,C3 gate), D=(MAG weights,MAG gate). A/D의 기존 36개 예측을 검증 후 재사용하고 B/C의 36개 교차 예측만 저장한다. 새 fits/optimizer updates=0. 다른 규칙·checkpoint·threshold·rank를 선택하지 않는다.
- total=A−D, gate=0.5[(A−B)+(C−D)], weights=0.5[(A−C)+(B−D)], interaction=A−B−C+D. 오차 단위이므로 양수는 MAG 방향의 이득이며 total=gate+weights를 검산한다. 같은 가중치에서 규칙을 교체한 유한 차이는 실제 함수 개입이다. weights 항은 학습과 checkpoint 선택의 합성 차이이며 학습 gate만의 인과효과가 아니다. 게이트 교체는 학습 시 분포와 달라지는 진단이며 새 실사용 후보가 아니다.
- 원자료/FAULT6종 및 평균/SHIFT4/SHIFT8/SHIFT_POINT/모든shape/모든seed를 보고한다. nMAE 주분해, MAE·pinball 원점수도 보존한다. 전체 swap 예측 저장 후 채점한다.
- 관측 입력만으로 gate 동일 여부, robust magnitude, 같은 부호 extreme 연속 길이, 마지막 extreme 위치, gate 차이의 최근128/32 관측 질량을 계산한다. I=1(|d|>3), p=최근8개의 같은 부호 extreme 비율×I, g_C3−g_MAG=patchmean(I−p)≥0. 연속 길이≥8이면 p=1이므로 차이는 각 연속 run의 첫7개 위치에만 있다는 명제를 전수 검사한다. 이는 실제 변화 시작 사건을 식별했다는 뜻이 아니다.
- 사전 진단 그룹은 ALL, GATE_EQUAL/GATE_DIFFERENT; peak magnitude≤3/(3,6]/(6,12]/>12; longest same-sign extreme run=0/1–7/8–31/≥32; last extreme=none/old0–383/recent384–511. 같은 그룹들의 완전 분할을 검사하고 미포함·유리 subgroup만 선택하지 않는다. 입력 층화는 서로 상관될 수 있으며 인과변수 중요도로 해석하지 않는다.
- 날짜별 채널·draw를 함께 묶고 고정 seed 평균을 사용한다. 7일 paired bootstrap2000회, seed918301+panelindex, 전체E128일 블록 기반. 사후 기술적95% 구간이며 유의성 승격·새 PASS 기준 없음. seed별 분해, 원점/채널별 기여와 leave-one-channel-out의 범위도 기록한다.
- checkpoint/기존 예측 hash, 저장된 diagonal 재현, 동결 모델 보존, 모델 off=B0, 동일 mask에서 동일 가중치 출력 일치, scalar gate·metric와 분해 재집계 검산. 모든 과거 결과 보존.
- GPU startup4GiB30초/runtime1GiB/RAM2GiB/disk10GiB/2시간 cap/단일 lock. RustDesk만 기존 승인 예외. 안전 차단 시 완료 범위를 보존하고 보고한다.
- 출력: 한국어 REPORT/FINAL_DECISION, 분해·입력 특성·seed/원점/계열 결과, 그림, 검산, scoped commit/push. 성능을 개선했다고 주장하지 않고 원인 근거와 아직 확인하지 못한 인과를 구분한다. 새 학습/새 자료/자동 후속0개.
