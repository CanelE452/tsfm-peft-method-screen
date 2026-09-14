# 파생 개선율 정정

원래 seed 열은 100×anchor/plain이며 개선율이 아니다. 올바른 식은 100×(plain−anchor)/plain이다.

30개 기존 E 원점수 → 식별자로 결합한 12 seed 대조 → 6 cell의 평균 손실 비율 → 3-source 균형 macro와 interaction → 원래 decision을 각각 재계산했다. seed 열의 오류는 현재 확인한 최종 판정을 바꾸지 않는다. 검산기는 원래 예측 primary만 검증했으므로 이 파생 열의 오류를 검출하지 못했다. 이번에는 파생 집계도 별도 단위 테스트로 검증한다.

기존 보고/봉인/소스는 수정하지 않았다. ETTm2 동일 점수는 개선율 0%다. Beijing dense의 plain 대비 개선과 F0 대비 작은 이득을 구분한다.
