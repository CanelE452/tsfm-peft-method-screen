# 표준 LoRA 개발 비교 봉인

[설계] 첨부 EXECUTION_CONTRACT 전체를 이 run의 계약으로 보존한다. 기존 transfer tune4와 같은 origin의 H3/H14, rank1(alpha2) native Chronos-2, seed61680, LR3e-5/1e-4, ZERO/EPOCH1/EPOCH4/EPOCH16/FIXED120을 모두 비교한다. 16 fits, 최대2624 optimizer updates. 별도 실제 모델 STD smoke2 updates, 전체 smoke한도24. 본학습 전체120 attempts/19680updates 한도는 후속 단계와 공유한다.

[설계] 표준 모델과 기존 동일-shape FP32 비교는 normalized prediction 기준 atol1e-5/rtol1e-5, 기준 크기와 raw 최대차도 기록한다. Step0 disabled-vs-enabled 및 target poison은 exact equality. 새 모델 checkpoint 복원은 FP32 같은 입력에서 exact equality. 실패 후 tolerance를 늘리지 않는다. 각 LoRA A tensor가 첫 step에서0 gradient인 것은 허용하고 finite/all-gradient-present와 전체 intended update를 확인한다.

[설계] worker fit 함수는 h 배열과 id/method/role/building/days/origin/seed/lr만 받는다. target file handle이나 전체 시계열은 받지 않는다. Coordinator만 fit 후 DISCOVERY target을 읽는다. LOCKED target은 선택 봉인 전 차단한다. 양의 OLS ill-conditioning은 미리 정한 F0 fallback을 사용하며 발생을 기록한다. SEASONAL24는 직전24시간 반복의 결정론적 참조이며 quantile forecast로서 calibration을 주장하지 않는다.

[설계] STD 비교 뒤 연구 근거와 후보 한 장을 먼저 봉인한다. STD 결과의 부호를 후보 비교 진입 gate로 삼지 않는다. 현재 미정인 후보 코드를 이미 구현/승인/성능검증했다고 쓰지 않는다. 후보가 정의되면 candidate.py/compare.py를 별도 봉인하며 현재 STD 실행 소스를 바꾸지 않는다.

[설계] 각 fit 최대 checkpoint까지 동일 길이의 순열 stream을 실행한다. Global method recipe는 DISCOVERY의 building-balanced primary로 선택하고 정확한 동률은 적은 예상 update -> 작은 LR. ZERO 포함. 동일 trajectory의 선택·fixed120 비용을 별도로 표기한다. Failed attempt은 상한에 포함하고 재시도 대체 금지. RUNNING인 실제 중단 fit만 저장된 model/optimizer/update에서 동일 ID로 복원할 수 있다. optimizer state는 update마다 atomic 저장한다. 오류·자원 중단은 과학적 실패로 집계하지 않는다.

[설계] GPU lock/guard는 이전 검산된 경로를 재사용한다. 초기30초 free≥4GiB, 실행중free≥1GiB 및 비승인 compute 없음. RustDesk만 기존 승인 예외이며 다른 PID 종료 없음. 누적대기600초, controller wall14400초. 데이터·model·STD 코드 hash와 metric quantile0.5 index를 보존한다.
