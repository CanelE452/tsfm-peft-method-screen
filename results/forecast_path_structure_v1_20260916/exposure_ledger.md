# 평가 구간 노출 이력

이번 고정 TEST는 2024-10-01~12-30 후보 원점이다. 기존 로컬 covariate/vintage/order/calibration의 seal·코드·결과를 확인했다. T0 OS Gorredijk의 모델 비교는 3~5월 C, 6~9월 D를 사용했고 그 결과는 이미 개발에 사용했다. 이번 TRAIN/V/CAL과 과거 개발 기간의 중복을 숨기지 않는다. 기존 순서 대조는 120 updates이며 이번 64원점/512 updates의 재사용 경로가 아니다.

기존 availability 감사는 T0의 2~12월 가용성·결측과 12월27일 기상값 예제를 읽었다. 이는 TEST 파일·입력에 대한 접근 이력이며, 10~12월 모델 예측 점수 노출과는 구별한다. 해당 기간의 모델 성능 열람은 기존 관련 manifest에서 찾지 못했다. 따라서 T0는 **기존 개발 타깃의 held-out time-period replication**으로만 한정한다. 원자료 자체가 완전히 미열람인 새 독립 데이터라고 쓰지 않는다.

T1 SS Ureterp와 T2 OS Waarderpolder는 기존 관련 예보 실험의 타깃 목록에 없었다. 이름 canonical SHA256 순서와 TRAIN 유효성만으로 선택했으며 이번에 처음 적응·평가한다. 같은 데이터셋·그룹의 타깃이므로 세 독립 도메인이라고 세지 않는다. 타깃 선정에 V/CAL/TEST 수치·성능·metadata upper/lower_limit을 사용하지 않았다.

이번에도 입력 함수를 만들 때 TEST 원점의 합법적 과거 부하를 읽는다. 뒤 TEST 원점의 context가 앞 TEST 정답 시각을 포함할 수 있지만 매번 관측 가능한 과거만 쓰며 학습 업데이트는 없다. 모든 TEST 정답은 예측·선택·보정 봉인 이후 scorer가 추출한다. 9월은 이 실행에서 학습·선택·보정·채점하지 않으나 옛 T0 연구에서는 이미 개발에 사용했던 달이다.

Chronos-2의 사전학습 corpus와 이 자료의 비중복은 입증하지 못했다. 실제 발행 로그가 아닌 SIMULATED_ASOF이며, 새 모집단에 대한 독립 시험이나 사전학습 무노출 확증으로 부르지 않는다.
