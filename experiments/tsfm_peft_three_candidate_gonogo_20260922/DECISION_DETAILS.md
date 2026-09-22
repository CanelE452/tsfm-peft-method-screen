# TEST 접근 전 판정·검산 세부 고정

TEST 예측/정답 채점 전에 기록했다. 계약의 threshold를 바꾸지 않는다.

- F의 특정 client 손해 5%는 seed별 client와 두 seed 평균 client 모두 공개하고, 하나라도 초과하면 보수적으로 TRADEOFF_HOLD로 표시한다. 두 seed 평균만으로 특정 반복의 큰 손해를 가리지 않는다.
- 통계 코드의 mechanical_category와 표준 방법의 유용성 해석을 구분한다. GO_STANDARD_ONLY는 추가 arbitrary threshold 없이, 공개한 baseline 상대 점수와 목적상 제약을 근거로 사람이 해석한다. 원래 수치 판정은 덮어쓰지 않는다.
- F 추론 자원표는 client0을 대표 입력으로 한 실측이다. 네 client 각각의 학습 시간/peak와 storage/communication은 FIT에 별도로 남긴다. 전체 workflow checkpoint 크기를 client 한 모델 크기라고 부르지 않는다.
- Q 초기화의 원래 각 fit별 시간이 별도로 계측되지 않았다. 학습/전체 프로세스 시간은 실측하고, 동일 고정 초기화의 runtime-only 재생(optimizer0, sensitivity 재측정0)으로 초기화 비용을 별도 표시한다. 원래 fit의 직접 타이밍이라고 부르지 않는다.
- 학습 전후 frozen hash는 state_dict의 모든 동결 가중치와 persistent buffer를 포함한다. nonpersistent quantiles metadata는 학습 중 전후 hash 대상이 아니었음을 공개하며, 배포 artifact roundtrip에서 실제 metadata buffer와 공식 예측 일치를 별도 검증한다. 검산 PASS는 이 범위를 뜻한다.
- 실제 성능 선택을 모두 봉인한 뒤 34개 선택 모델/untouched baseline의 TEST 예측을 저장한다. 모든 prediction SHA가 검증된 뒤 독립 scalar replay와 TEST 채점을 수행한다.
- 최종 추천은 GO_SCREEN을 만족하는 후보 중 최대 하나이다. 해당 후보가 없으면 NONE. 후속 확인은 자동 실행하지 않는다.
