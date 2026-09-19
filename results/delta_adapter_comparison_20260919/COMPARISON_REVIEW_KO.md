# 직접 비교의 통계·비용·방법론 해석 보완

기존에 봉인된 비교의 CPU 재검산과 기술 통계다. 새 학습·모델 선택·성공 기준은 없다. 과거 분석 논문을 방법론 논문이라고 이름만 바꾸지 않는다.

## 신뢰구간이 다루는 불확실성

주 단위 합계를 먼저 구한 뒤 봉인된 2,000개 재표집에 적용하는 별도 구현으로 모든 효과행의95%/family4 구간을 재계산했다. 원 runner는 origin 가중치를 사용한다. 두 계산이 일치했다. family4 보정의 주장은 이번 단위의 사전 지정 네 비교에만 적용한다. 이전 후보 탐색 전체의 선택 편향을 보정하거나 이미 본 E를 확증 자료로 바꾸는 것은 아니다. 나머지 수치는 탐색적이다.

이는 두 학습 seed와 현재 채널·자료를 고정한 시간 블록 변동성이다. seed 모집단·독립 source·최적 설정 탐색 불확실성까지 포함하지 않는다. index-week는 표본 index 기준7일 블록이며 달력의ISO week라고 부르지 않는다. E 구간은 기존 개발 자료다.

| panel | origins | distinct_index_days | index_weeks | index_span_days | seed_count |
| --- | --- | --- | --- | --- | --- |
| electricity | 128 | 128 | 32 | 215 | 2 |
| electricity_transfer | 128 | 128 | 32 | 215 | 2 |
| ettm1 | 128 | 128 | 21 | 143 | 2 |

## 기존 구성요소 대조를 삭제하지 않는다

다음 전력16계열 selected SHIFT8의 seed별 결과도 함께 읽어야 한다. δ 대조에서의 이득이 C3 지속성 규칙의 추가 가치를 대신 입증하지 않는다. MAG는 관측 크기만 사용하는 더 단순한 대조군이다. 모든 상태·seed·fixed1024 대조는 DESCRIPTIVE_SEED_CONTRASTS.csv에 있다.

| proposed | baseline | seed | gain_pct |
| --- | --- | --- | --- |
| C3 | B0 | 81551 | 5.964495 |
| C3 | PLAIN | 81551 | 0.944230 |
| C3 | MAG_ONLY | 81551 | -0.191989 |
| MAG_ONLY | B0 | 81551 | 6.144687 |
| MAG_ONLY | PLAIN | 81551 | 1.134042 |
| C3 | B0 | 81552 | 11.446064 |
| C3 | PLAIN | 81552 | 5.691144 |
| C3 | MAG_ONLY | 81552 | -0.630873 |
| MAG_ONLY | B0 | 81552 | 12.001225 |
| MAG_ONLY | PLAIN | 81552 | 6.282383 |

## 동일 반복 경로의 자원

각 군의 두 원천×두 반복 seed, 총4개1024-update 경로 평균이다. 선택된 checkpoint hash를 원 receipt와 맞춰서 연결했다. optimizer 시간은 실제 기록이며 검증·선택·I/O의 전체 비용이 아니다. 서로 다른 시각에 기록했으므로 엄밀한 동시 성능 벤치마크나 하드웨어 불변 speedup으로 주장하지 않는다. allocator peak는 PyTorch 할당량이며 nvidia-smi 전체 사용량과 다르다. GPU 메모리와 파라미터 수를 혼동하지 않는다.

| arm | parameters | optimizer_seconds | peak_allocated_MiB |
| --- | --- | --- | --- |
| C3 | 8712.000000 | 32.033104 | 362.108887 |
| DELTA_XY_BUDGET | 8766.000000 | 35.756697 | 366.435059 |
| DELTA_XY_DEFAULT | 1116736.000000 | 35.938141 | 379.352051 |
| MAG_ONLY | 8712.000000 | 34.863896 | 362.108887 |
| PLAIN | 8712.000000 | 31.450155 | 362.108887 |

검증에서 선택된 실제 step은 다음과 같다. step0은 경로 전체1,024 updates를 실행한 뒤 사전 규칙에 따라 초기 가중치를 선택했다는 뜻이다. δ의 초기 잔차는 random이므로 δ step0을 B0와 같다고 볼 수 없다. 추가 학습의 이득과 초기 구조의 영향을 혼동하지 않는다.

| source | arm | seed | selected_step |
| --- | --- | --- | --- |
| electricity | PLAIN | 81551 | 512 |
| electricity | PLAIN | 81552 | 1024 |
| electricity | C3 | 81551 | 768 |
| electricity | C3 | 81552 | 768 |
| electricity | MAG_ONLY | 81551 | 768 |
| electricity | MAG_ONLY | 81552 | 768 |
| electricity | DELTA_XY_BUDGET | 81551 | 768 |
| electricity | DELTA_XY_BUDGET | 81552 | 1024 |
| electricity | DELTA_XY_DEFAULT | 81551 | 256 |
| electricity | DELTA_XY_DEFAULT | 81552 | 1024 |
| ettm1 | PLAIN | 81551 | 0 |
| ettm1 | PLAIN | 81552 | 1024 |
| ettm1 | C3 | 81551 | 0 |
| ettm1 | C3 | 81552 | 1024 |
| ettm1 | MAG_ONLY | 81551 | 0 |
| ettm1 | MAG_ONLY | 81552 | 0 |
| ettm1 | DELTA_XY_BUDGET | 81551 | 768 |
| ettm1 | DELTA_XY_BUDGET | 81552 | 512 |
| ettm1 | DELTA_XY_DEFAULT | 81551 | 0 |
| ettm1 | DELTA_XY_DEFAULT | 81552 | 0 |

이 파라미터 수는 추가 어댑터만 센다. 모두 기존 B0 LoRA294,912개와 frozen Chronos backbone을 따로 유지하므로 전체 저장·배포 크기가8,712개라는 뜻이 아니다. B0를 얻는 사전 적응 비용도 추가 학습 비용에 포함하지 않았다. δ는 입력까지 gradient를 전파하지만 기존 latent adapter는 다른 위치에 있어 파라미터 수만으로 훈련 메모리·시간을 예측할 수 없다. 위치·초기화·연산·출력 제한을 한 번에 바꾼 비교이므로 속도/효과 차이의 단일 원인을 분리한 실험은 아니다.

## 분위수 순서 진단

| arm | crossing_pairs | total_pairs | crossing_fraction |
| --- | --- | --- | --- |
| B0 | 268300 | 182452224 | 0.001471 |
| C3 | 344742 | 182452224 | 0.001889 |
| DELTA_XY_BUDGET | 370320 | 182452224 | 0.002030 |
| DELTA_XY_DEFAULT | 494336 | 182452224 | 0.002709 |
| MAG_ONLY | 351795 | 182452224 | 0.001928 |
| PLAIN | 369595 | 182452224 | 0.002026 |

인접 quantile 쌍의 crossing 비율을 저장된 모든 view에서 합산했다. selected/fixed 동일 checkpoint의 중복 view도 포함된 기술 통계이며 독립 표본 수로 쓰지 않는다. 이 지표를 보고 재정렬·재보정하거나 선택 규칙을 바꾸지 않았다.

## 방법론 주장에 남은 과제

이번 단위는 공식 XY cell을 고정 과제에 옮긴 직접 비교를 채웠다. 원 논문의 전 설정 재현과 모든 robust PEFT 대비 우위는 이번 실험이 확인한 범위 밖이다. 방법론 논문에 모든 데이터·상태에서의 승리를 요구하는 것은 아니다. 현재 남은 핵심은 C3 지속성 요소의 추가 가치와 신규성, 이미 반복 사용한 E 밖에서의 확인이다. 실제 확인된 좁은 이득은 남기되 논문 전체 목표를 완료로 표시하지 않는다. 추가 실험은 자동 시작하지 않았다.
