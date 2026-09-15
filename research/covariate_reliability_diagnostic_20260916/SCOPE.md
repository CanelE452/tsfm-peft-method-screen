# 예보 신뢰도 진단 — 계산 전 범위

기준 fa5be2988562249d45bd8609b84ecf14d0f8690c. 이전 참조 실험은 완료 상태로 보존한다. 이번은 동일한 28개 C/D 원점의 저장 입력·예측·정답을 사용하는 사후 원인 진단이다. 새 학습0, 신경망 추론0, GPU0, 새로운 배포 선택기0. 과거 PASS/FAIL 및 teacher 조건은 바꾸지 않는다.

질문: 예보가 서로 다를수록 실제 부하 예측에서 최신 기상을 덜 믿어야 하는가? 예보의 기상 오차와 부하 예측 손해를 구분한다.

고정 진단량:
1. INPUT_SPREAD: 네 미래 경로의 각 feature/time population std / 해당 원점의 과거 기상336h std(floor1e-6), 세 feature·24h 평균. 예측 시점에 사용 가능한 simulated-as-of 입력량.
2. OUTPUT_SPREAD: 네 경로의 저장 부하 median 예측 population std / 부하 context std, 24h 평균. 가용량이나 이미 네 경로 추론이 필요한 진단.
3. WEATHER_ERROR_PROXY: dataset weather_measurements의 해당 미래24h와 각 미래 예보 경로의 절대 차이 / 위 과거 feature std, 세 feature·24h 평균. 평가 후에만 알 수 있는 기상 오차 proxy이며 운영 입력·보정·선택에 사용하지 않는다. Open-Meteo historical field는 정확한 현장 관측 참값이라고 단정하지 않는다. available_at이 없어 실시간 오차 피드백 계약도 아니다.
4. DAMAGE = latest primary - F0 primary, 양수면 기상 입력이 손해. MIXTURE_GAIN = latest primary - raw mixture primary, 양수면 혼합이 도움. 일자와 날씨 오차만으로 손해 원인을 인과적으로 단정하지 않는다.

C12·D16을 따로 보고한다. INPUT/OUTPUT_SPREAD 각각 C median으로 low(<=)/high(>)를 정해 C/D 원점수를 집계한다. 평가 D threshold 재선택·최적 cut 탐색 없음. 세 신호와 DAMAGE/MIXTURE_GAIN의 Spearman 상관을 표시하며, 24시간을 독립 표본으로 세지 않는다. 신호들은 exploratory이며 유의성/인과성/일반화 증거로 주장하지 않는다.

저장된 네 개 path의 primary와 기상 오차 proxy를 모두 계산하고 k0/k3의 같은 원점 차이를 보고한다. 오래된 예보가 부하 성능을 망치는지, 기상 오차만 늘어나는지 구분한다. 최적 path의 oracle을 새 예측 정책으로 만들지 않는다.

각 path의 pinball scalar 검산, weather proxy의 scalar 검산, 상관의 별도 average-rank 검산, 파일 hash·기존 결과 보존. 후보 gate를 학습하지 않는다. 원인 가설을 지지/제한하는 범위와 가까운 선행의 알려진 구성요소를 한국어 REPORT에 남긴다.
