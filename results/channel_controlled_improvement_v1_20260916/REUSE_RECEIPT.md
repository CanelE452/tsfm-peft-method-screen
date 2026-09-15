# 기존 학습 경로 재사용 감사

같은 모델/초기값·표본순서·학습 recipe와 INIT부터 epoch20까지 체크포인트·업데이트 기록이 확인된 8개 경로만 재사용한다. 과거 점수는 새 선택 점수로 복사하지 않으며 두 V 패널에서 다시 예측한다. 다른 28개 cell은 새 학습이다.

| 원천/seed/방법 | 결정 | 이유/업데이트 |
|---|---|---|
| electricity_41000_LH | REUSE_ACCEPTED | 1280 updates, INIT+20 checkpoints |
| electricity_41000_SIDE | REUSE_ACCEPTED | 1280 updates, INIT+20 checkpoints |
| electricity_41000_PRIOR | REUSE_ACCEPTED | 1280 updates, INIT+20 checkpoints |
| electricity_41001_LH | NEW_REQUIRED | Partial path: no complete optimizer/scheduler/RNG state |
| electricity_41001_SIDE | NEW_REQUIRED | Partial path: no complete optimizer/scheduler/RNG state |
| electricity_41001_PRIOR | NEW_REQUIRED | Partial path: no complete optimizer/scheduler/RNG state |
| electricity_41002_LH | NEW_REQUIRED | No historical seed |
| electricity_41002_SIDE | NEW_REQUIRED | No historical seed |
| electricity_41002_PRIOR | NEW_REQUIRED | No historical seed |
| traffic_41000_LH | REUSE_ACCEPTED | 1260 updates, INIT+20 checkpoints |
| traffic_41000_SIDE | REUSE_ACCEPTED | 1260 updates, INIT+20 checkpoints |
| traffic_41000_PRIOR | REUSE_ACCEPTED | 1260 updates, INIT+20 checkpoints |
| traffic_41001_LH | REUSE_ACCEPTED | 1260 updates, INIT+20 checkpoints |
| traffic_41001_SIDE | NEW_REQUIRED | Partial path: no complete optimizer/scheduler/RNG state |
| traffic_41001_PRIOR | REUSE_ACCEPTED | 1260 updates, INIT+20 checkpoints |
| traffic_41002_LH | NEW_REQUIRED | No historical seed |
| traffic_41002_SIDE | NEW_REQUIRED | No historical seed |
| traffic_41002_PRIOR | NEW_REQUIRED | No historical seed |

이 표의 NEW_REQUIRED는 원천/seed/방법의 기존 경로 감사다. LR=0.0003의 18개 cell은 모두 새 학습이며 여기에 추가된다. 기존 조기 종료 경로에는 Adam/scheduler/RNG 완전 상태가 없어 best weights부터 이어가지 않는다. 새 학습은 완전 상태를 원자적으로 저장한다.

자원 측정용 canonical 상태는 선택된 가중치와 새로 비운 Adam/RNG를 저장한 것이다. 모든 옵션을 같은 상태로 복원하고 두 warmup updates로 Adam 상태를 형성한다. 과거 학습 Adam을 복구했다는 의미가 아니다.

표기 주의: REUSE_RECEIPT.json의 `epochs: 21`은 INIT 포함 저장 상태 개수다. 실제 학습은 모든 재사용 경로에서 20 epochs이며 trajectory epoch0..20과 실제 업데이트 수로 감사했다. 학습 epoch를 21로 늘린 것이 아니다.
