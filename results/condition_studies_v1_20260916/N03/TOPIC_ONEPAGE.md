# N03 CLOCK


7. N03 CLOCK — 같은 실제 미래를 다른 관측 밀도로 예측
======================================================================

데이터: ETTm1의15분 원자료,4채널. C336slots=84h, H48slots=12h.
ETTh1과 같은 물리적 시스템/다른 해상도일 수 있으므로 독립 도메인으로 세지 않는다.
문제: 같은84h 관측 구간을 듬성듬성 읽을 때 실제시간을 보존하는 단순처리로 충분한가?
선행: t-PatchGNN, FlowState(ICML2026). 아래 kernel은 두 논문의 재현이 아니다.
초기 실험은 기록된 규칙 자료의 관측을 가리는 모사다. 실제 irregular benchmark 성공 아님.

관측:
TRAIN epoch마다 dense(delta=1)와60분(delta=4) 교대.
각 채널의 sampling phase는 hash(origin,channel,epoch)로 정한 0..delta-1.
모든 군에 같은 관측값·시각·mask·age 제공.
V_SELECT는 delta1/4 같은 가중치. E 주평가는 미학습delta2(30분)와
간격{1,3}를 교대로 섞은 irregular pattern(평균2slot)의 같은 가중치.
irregular의 시작phase/순서는 고정hash. 마지막관측은 항상 origin미만.
E의 dense1/60분4는 부가표. 예측은 모든 경우 같은 미래12h의15분 grid.
현재 원점 사례에서 가린 값은 입력·보간·native context normalization·보조 복원 손실에서 읽지 않는다.
공통 TRAIN mu/sigma는 dense 학습 조건에서 관측을 허용한 TRAIN 자료만으로 고정한다.
이 실험은 TRAIN 전체가 불규칙 관측인 설정이 아니라, dense/60분 조건으로 학습한 뒤
새 관측 밀도에 평가하는 설정이다. 실제 완전 미관측 TRAIN 값을 알고 정규화했다고 혼동하지 않는다.

4개 학습군:
C0 GRID: canonical15분 grid의 누락을 NaN으로 유지 + M/A + LoRA.
C1 HOLD: forward-fill(이전 관측 유지), 첫 관측 전은 TRAIN mean + M/A + LoRA.
C2 KERNEL: 누락grid t를 이용 가능한 가장 가까운4개 관측으로 kernel 보간 + M/A + LoRA.
   w_i=exp(-|t-t_i|/tau0), tau0=4 raw slots. 관측 위치는 원래 값 그대로.
   양쪽 관측 모두 origin 이전이면 현재 forecast에서 이용 가능하다.
   실시간 각 과거 t에 이미 알려졌다는 주장이 아니라 원점o에 합법적인 smoothing이다.
C3 LEARN_KERNEL: C2와 같고 채널별 tau_c=0.5+7.5*sigmoid(theta_c).
   tau_c=4가 되도록 초기화. 추가4파라미터. 인접4개 선택은 고정, kernel만 학습.
   누락이 없는 창은 C2/C3가 입력상 동일하고 tau gradient0일 수 있다.

대조의 정보 권한은 동일. C0도 mask/age와 canonical 실제시간 위치를 받는다.
단순 재표본화 baseline에서 시간정보를 일부러 빼지 않는다.
primary: 동일 실제12h의 normalized RMSE (delta2/irregular 평균).
기대: 학습에서 본 관측밀도에만 맞춘 성능이 아니라 중간/불규칙 조건의 이득.
반례: C0/C1/C2로 충분 -> 학습 kernel 추가가치 없음.
이 결과만으로 다양한 관측률의 모든 TSFM이나 실제 비동기자료를 해결했다고 쓰지 않는다.

필수검사: 모든 방법 동일 observed tuple(value,timestamp), 정답 physicaltime 일치,
원점 이후 관측 접근0, kernel row weight 합1, observed overwrite0,
tau0일 때C3=C2, raw timestamp/단위 변환 검산.


[설계] 전체 계약의 공통 조항도 적용. 실행/효과/단순 대안/신규성을 분리한다.
