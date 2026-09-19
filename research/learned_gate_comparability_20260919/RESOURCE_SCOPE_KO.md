# 자원 비용의 비교 범위

학습 중 코드를 읽어 확인한 측정 범위다. 새 학습·추론·benchmark는0이며 현재 실행과 사전 기준을 변경하지 않았다. 원래 자원 receipt도 수정하지 않는다.

| 항목 | 기존 MAG/PLAIN/C3 | 새 TOKEN_GATE 두 군 | 허용되는 해석 |
| --- | --- | --- | --- |
| 추가 학습 파라미터 | 8,712 | 9,225 | 추가513개(5.89%); 전체 B0 훈련비를 대체하는 수가 아님 |
| optimizer_seconds | forward/backward/clip + optimizer.step/sync; intent 저장 제외 | zero_grad 전부터 step/sync까지; intent 저장 포함 | 원래 정의 그대로 보고하며 순수 계산 속도비로 해석하지 않음 |
| validation_seconds | validation forward와 metric 계산 | 같은 부모 validation 함수 | 서로 다른 측정 시각·GPU 상태의 단일 기록; 반복 benchmark 아님 |
| peak_allocated | PyTorch allocator peak | PyTorch allocator peak | nvidia-smi 전체 사용량이나 전체 시스템 메모리와 구분 |
| invocation_wall_seconds | guard/I/O 등 포함, runner 버전별 비용 차이 | 매 update GPU 점검 포함 | runner 전체 비용이지 adapter 계산 속도만의 차이가 아님 |

두 timer 모두 이름이 optimizer_seconds지만 forward/backward를 포함한다. 새 코드에서는 BEFORE_OPTIMIZER intent JSON 저장이 그 시간 구간 안에 있고, 이전 코드에서는 두 측정 구간 사이에 있다. 이 저장의 실제 시간을 별도로 측정하지 않았으므로 사후 추정해서 빼지 않는다. GPU 점검 빈도 및 과거 측정 시각도 달라 wall시간 우위를 주장하지 않는다.

검산은 각 runner의 ledger 합계와 receipt 일치를 확인하는 것이며, 그 일치가 runner 간 timing boundary 동일성을 증명하지는 않는다. 학습gate 두 군끼리는 같은 runner 정의를 사용하지만 한 번의 실행 시간 차이를 일반적인 속도비로 과장하지 않는다.

이번 논문 증거에서 확실히 비교 가능한 자원 항목은 부가 학습 파라미터 수·정해진 update 수·실제 peak 정의다. 최종 정확도·비용 절충에는 이 정의와 공통 학습된 B0의 비용을 함께 명시한다. 새 timing benchmark는 승인된16fits 범위 밖이므로 자동 추가하지 않는다.

[코드 hash 및 기계 판독 기록](RESOURCE_SCOPE.json) · [선택·정보 공정성 감사](COMPARABILITY_KO.md)
