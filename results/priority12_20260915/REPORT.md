# PEFT 우선순위 1·2 실행 결과 — 기존 원천을 재사용한 개발 비교

**Q: BLOCKED_GPU_BUSY. C: BLOCKED_GPU_BUSY.** 이번에 완료한 구현·CPU 검사와 실제 GPU 학습을 아래에서 분리한다. 수치 인증 미실행과 예측 성능 FAIL을 혼동하지 않는다.

## 1. 실제 실행 횟수와 미실행

| 작업 | 폐기용 실모델 GPU updates | 본학습 fits 완료/시도 (상한) | 본학습 updates |
| --- | ---: | ---: | ---: |
| Q 수치진단/자원 | 0 / 0 | 0/0 (12) | 0 |
| C 구조점검 | 0 | 0/0 (24) | 0 |

C에는 별도로 tiny FP64 구조 검사의 실제 optimizer 30회가 있다. 초기 검사와 prepare 회귀 검사에서 각각5개 작은 block×3회였다. 실모델 CPU optimizer는0이며 이 toy 업데이트를 예측 fits로 세지 않았다. 보수적으로 구조점검72회 예산에서 차감했다. [장부](../channel_basis_pilot_20260915/cpu_update_ledger.json).

Q 시작 대기 600.19초, C 시작 대기 600.23초. 원래1290개 파일은 불변이다. GPU 진입 조건은 외부 compute 없음30초와 free4GiB 이상이다. 다른 작업의 프로세스를 종료하지 않았다.

## 2. Q 수치 중단이 해소됐는가

새 단일/연속 인증 결과는 [Q 수치검증](../query_budget_numeric_v2_20260915/numeric_verification.json)에 있다. 실제 새 GPU 진단 update는 0회다. 기존42개 artifact 비교는 원래 v1 기준으로 재계산했다. v1의4개 microbatch 기준 초과와 INCONCLUSIVE_NUMERICS는 보존한다.

v2 기준은 사용자가 이전 중단을 보고 명시적으로 개정한 정책이다. raw 단위 절대값 관문을 Train std 보정으로 바꾸고 gradient/update 및5-step 경로 한계를 새로 고정했다. 결과에 맞춰 반복 조정하지 않았다. 실제 인증이 미실행이면 정책 구현만으로 중단 원인을 규명하거나 NUMERICS_BOUNDED라 할 수 없다.

## 3. 강한 Standard 대조 뒤 Query의 자원 이점

자원 비교 실제 update는 0회다. [Q 보고서](../query_budget_numeric_v2_20260915/REPORT.md)와 [예산 표](../query_budget_numeric_v2_20260915/resource_budget_table.csv)를 참조한다. 측정이 없으면 B*, 반복 속도 이점, 최속 Standard/Side를 정할 수 없다. 예산은 실제 소형 GPU 실행이 아닌1/2/4/8GiB tensor allocated 시뮬레이션이다.

## 4. C의 구조와 정확한 파라미터 수

Time-PEFT의 MOMENT-small·LoRA q/k/v·frequency/down/head를 기준으로 채널 up의 공유 형태만 비교한다. 실제 D512, patch8,64patches, horizon96을 확인했다. 두seed에서 공식 SPECIFIC 출력/목적식과 wrapper가 CPU FP32에서 차이0으로 일치했다. SPECIFIC~BASIS4의 초기 출력 차이도0이었다. 이 관찰은 GPU backward 검증과 다르다.

| arm | 채널 블록 | 전체 trainable | 총 모델 |
| --- | ---: | ---: | ---: |
| LORA_HEAD | 0 | 3,317,856 | 38,655,264 |
| SPECIFIC | 8,684,800 | 12,265,312 | 47,602,720 |
| SHARED | 395,008 | 3,975,520 | 39,312,928 |
| SHARED_WIDE | 790,017 | 4,370,529 | 39,707,937 |
| GROUP4 | 789,760 | 4,370,272 | 39,707,680 |
| BASIS4 | 790,016 | 4,370,528 | 39,707,936 |

BASIS4의 전체 trainable은 SPECIFIC의 35.633%로 64.367% 감소한다. 채널 블록만의 절약을 전체 절약이라고 바꾸지 않았다. 파라미터 조건만으로 예측1% 이내 유지 조건을 만족했다고 볼 수 없다.

Electricity 원본26304×321, Traffic17544×862에서 Train-only 조건의 첫64개 열을 골랐다. V/E origins는 각각54/54 및36/36이며 원본4채널을 복제하지 않았다. [자료 manifest](channel_data_manifest.json).

## 5. 같은 예산 공유/고정 그룹보다 BASIS4가 좋은가

신규 MSE/MAE 원점수가 없다. BASIS4 대 SHARED_WIDE/GROUP4의0.5% 평균 이득 및 두seed 같은 방향 조건은 미판정이다. CPU 구조 정상 작동을 계수 활용의 예측 이득으로 승격하지 않는다.

## 6. SPECIFIC과 LORA_HEAD 대비 실제 가치

SPECIFIC 대비1% 이내 예측 유지, LORA_HEAD 대비 예측 우위 모두 확인하지 못했다. 가장 강한 예측 대조군을 선정할 근거가 없다. LORA_HEAD는 전체3,317,856 trainable로 BASIS4보다 작다는 구조적 사실만 확인됐다.

| 작업 | 신규 원점수 | 자원 이득 | 예측 이득 |
| --- | --- | --- | --- |
| Q | 미실행 | 미판정 | 미판정 |
| C | 미실행 | 파라미터 수 확인; GPU 속도/peak는 실측 여부 별도 | 미판정 |

## 7. 계속할 근거와 남은 한계

이번 구현은 제한 실험을 실행할 준비와 CPU 의미 검사를 제공한다. GPU 시작이 막힌 경우에는 성능을 근거로 연구 방향을 채택하거나 폐기할 수 없다. CPU 검사는 Q 전체112개, C 전용6개가 통과했다. C의 끝 불완전 시간 블록을 포함하는 bootstrap도 추가 모델 실행 없이 검산했다.

[Time-PEFT 공개 구조](https://github.com/kaist-dmlab/TimePEFT/blob/ea4e7e1887bb35587bab7ea93e2af3685ac55852/run.py)의 통제 변형이며 원문 전체 재현이 아니다. [C-LoRA](https://arxiv.org/html/2407.17246v1) 등 채널 공유·저랭크 조합은 이미 알려진 원리다. 공식 C-LoRA 같은 백본 비교, 가까운 혼합 adapter와의 차이, 새 원천 독립 확증 및 최적화 조건 확장이 남아 있다. 이번 작업만으로 신규 방법이나 논문 PASS를 선언하지 않는다. [선행/환경 차이](../../sources/RELATED_WORK.md).

공개 requirements와 momentfm 메타데이터의 Transformers 버전 충돌은 격리된 C 환경에서 공개 코드의4.44.2를 적용해 해결했다. Q 환경·드라이버는 변경하지 않았다. C의 best-V 저장/복원은 upstream의 마지막 state 반환과 다르며 모든 arm에 동일 적용한다.

**이번 두 실행은 종료한다.** 같은 run을 덮어쓰거나 이름만 바꾼 재시도, 추가 seed/LR/후보/자동 후속 학습은 실행하지 않는다. 코드·계약·검증·실패 근거는 GitHub에 보관하고, 원자료·가중치·큰 cache는 로컬에 둔다. GPU 학습 경로가 미실행이면 그 경로의 실동작까지 검증됐다고 하지 않는다.

[Q 상세](../query_budget_numeric_v2_20260915/REPORT.md) · [C 상세](../channel_basis_pilot_20260915/REPORT.md) · [C 검증](../channel_basis_pilot_20260915/independent_verification.json)
