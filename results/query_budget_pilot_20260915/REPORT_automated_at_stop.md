# Query 자원 제약 파일럿 — simulated tensor budget / 기존 평가 구간을 재사용한 개발 비교

A: 폐기용 optimizer updates 120회 (120 attempts). B: 0/0 fits 완료, 0 updates. 종료 상태: **INCONCLUSIVE_NUMERICS**.

## 1. 비교 목적

동일 context4096, 4채널 origin 그룹, 1,179,648 학습 파라미터에서 Standard/Side/Query의 자원 제약을 비교했다. Standard에는 CP0/3/6/9/12와 origin microbatch1/2를 허용했다. Censor 재튜닝이나 새 어댑터 개발은 하지 않았으며 과거 판정은 보존한다. Query/Side의 frozen encode/cache를 매 forward 비용에 포함한다.

## 2. 예산과 옵션

1/2/4/8GiB는 실제 장비 요구가 주어지지 않아 사전에 고정한 tensor allocated 예산 시뮬레이션이다. allocated, reserved, NVML 및 실제 free VRAM은 다른 양이다. 1GiB GPU에서 실행된다는 뜻이 아니다. [전체 budget 표](resource_budget_table.csv), [timing 반복](resource_measurements.csv), [준비·cold/warm/FP32 포함 모든 실행](resource_all_executions.csv)을 구분한다.

## 3. 수치 무결성과 Standard 대조

저장 artifact로 독립 재계산한 parity 42개 중 실패 4개. 실패 옵션을 제외해서 Standard를 불가능으로 분류하지 않는다. 수치 무결성을 확보하지 못하면 이번 조합 전체의 A가 판정 불가다.

| 옵션 | 검사 | precision | 실패한 주요 양 |
| --- | --- | --- | --- |
| ettm2_standard_cp0_mb1 | microbatch | fp32 | {"clipped_gradient": {"max_absolute": 1.6312114894390106e-06, "reference_norm": 0.9999996600550529, "relative_l2": 3.7796536415505915e-05}, "raw": {"max_absolute": 1.1444091796875e-05, "reference_norm": 3228.142828365484, "relative_l2": 4.029466897778121e-08}, "raw_gradient": {"max_absolute": 4.194676876068115e-06, "reference_norm": 2.5662232073278672, "relative_l2": 3.784510616151414e-05}, "update": {"max_absolute": 8.473580237478018e-07, "reference_norm": 0.07115482449479663, "relative_l2": 5.361744498231507e-05}} |
| electricity_standard_cp0_mb1 | microbatch | fp32 | {"clipped_gradient": {"max_absolute": 1.0366784408688545e-06, "reference_norm": 0.9999998963394301, "relative_l2": 3.577182601086688e-05}, "raw": {"max_absolute": 0.00048828125, "reference_norm": 29462.571111846093, "relative_l2": 1.1771006105138378e-07}, "raw_gradient": {"max_absolute": 9.953975677490234e-06, "reference_norm": 9.627341518210692, "relative_l2": 3.580436331012271e-05}, "update": {"max_absolute": 6.863847374916077e-07, "reference_norm": 0.0716438461183946, "relative_l2": 3.9329564337505345e-05}} |
| electricity_side_cp0_mb1 | microbatch | fp32 | {"raw": {"max_absolute": 0.00048828125, "reference_norm": 29232.38821309482, "relative_l2": 1.3311525382270693e-07}} |
| electricity_query_cp0_mb1 | microbatch | fp32 | {"raw": {"max_absolute": 0.000732421875, "reference_norm": 29401.077397715337, "relative_l2": 1.383779364974655e-07}} |

## 4. B 실행 여부와 예측 결과

예측 비교 미실행; INCONCLUSIVE_NUMERICS.

새 E 점수와 최강 예측 대조군은 측정하지 않았다. 과거 점수를 이번 자원 설정의 새 정확도로 가져오지 않는다.

## 5. 원점수·반복·비용

[metrics.csv](metrics.csv)는 새로 평가한 원점수만 담는다. [fit_attempts.csv](fit_attempts.csv), [resources.csv](resources.csv), [trajectories.csv](trajectories.csv)에 실제 시도·시간·V 기회를 구분한다. 기록이 비어 있으면 미실행이며 0 성능을 뜻하지 않는다. Native raw loss는 원천 간 직접 평균하지 않는다. 개선율은 100×(baseline−Query)/baseline이다.

## 6. 한계

개발 데이터 재사용, 2 seeds, 길이4096 하나, 제한된 CP/micro 옵션과 단일 LR의 비교다. 4개 시간 블록 bootstrap은 기술적 불확실성 표시이며 재사용 편향을 보정하지 않는다. 실제 소형 VRAM 장비, 다른 길이·원천, 신규성은 검증하지 않았다. F0와 각 arm의 step0 차이를 별도 저장하며 초기 반올림 차이를 모두 학습 이득으로 세지 않는다.

Side는 [Ladder Side-Tuning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html)의 원리와 관련된 저장소 대조군이며 공식 전체 재현이 아니다. [PyTorch checkpoint 설명](https://pytorch.org/blog/activation-checkpointing-techniques/)과 로컬 PyTorch API를 확인했다. Query의 큰 아이디어가 최초라는 주장을 하지 않는다.

## 7. 종료와 다음 판단

현재 구현의 FP32/BF16 microbatch 및 checkpoint 동등성 문제를 먼저 이해해야 자원 이점이나 새 학습을 판단할 수 있다. 이번 실행 안에서 허용오차를 늘리거나 실패 옵션을 빼고 진행하지 않는다. 자동 추가 학습·seed/LR/budget 탐색은 없다.

검증: 이전 파일 1217개 해시 보존, scalar metric 최대 차이 0. [independent_verification.json](independent_verification.json)은 기록의 재현성을 확인하며 실패 판정을 PASS로 바꾸지 않는다.

종료 근거: `NumericalError('FP32_CP_OR_MICROBATCH_EQUIVALENCE_FAILED')`
