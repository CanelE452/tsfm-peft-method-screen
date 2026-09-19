# δ-Adapter 공개 XY cell과의 통제 비교

**16/16 본학습·선택·144개 예측 view의 채점·독립 검산 완료.** 이 완료 상태는 새 PEFT 방법론의 신규성이나 논문 PASS를 뜻하지 않는다. C3/MAG를 수정하지 않고 가까운 선행의 실제 비교를 추가했다.

## 실행과 구현 범위

새16fits = 두 원천×두δ구성×selection seed81550의두LR(8fits) + 선택LR의81551/81552 반복(8fits). 본학습16,384updates, 별도smoke8updates. 기존 B0/PLAIN/C3/MAG의96개 view는 hash 검증 후 재사용했다. 새δ48view(새 전체 추론40개, 같은 checkpoint 재사용8개)는 선택을 봉인한 뒤 저장했으며, 동일checkpoint alias를 제외한 고유 파일 수는 전체116개다. 전체 예측 저장 후 새 E정답을 채점했다. 원점별/채널별 원점수와 불리한 조건을 모두 보관한다.

공식 `Anoise/Adapter` commit0add06e의 additive XY cell을 독립 구현하고 입력 길이512/64×hidden7/512의4개 CPU 경우에서 output/input-gradient/batch permutation의 exact parity를 확인했다. 두δ×두원천의 실제 모델에서 학습·off 경로 B0 동일성·동결 가중치 보존·fresh checkpoint 복원을 검사했다. 원래 random residual 초기화를 유지해 초기 출력은 B0와 다를 수 있다.

공개 기본 폭512와8,712개 기존 어댑터에 가까운 폭7을 함께 사용했다. 작은δ는8,766개로0.62% 더 많고, 기본δ는1,116,736개다. 동일 parameter count라고 쓰지 않는다. 공개 cell을512→64 예측에 연결하면서 output 입력 차원을64로 지정했고, 관측 평균/기존 TRAIN sigma로 단위를 맞췄다. output cell은9개 분위수에 공유했다. MSE 대신 기존 normalized2pinball·동일 synthetic TRAIN/V를 사용하므로 **공식 논문 전체 재현이 아니라 현재 과제로 옮긴 통제 비교**다. 공식 Y-only/feature-selector/quantile-calibrator와 같은 실험이 아니다.

## 사전 주 비교

양수는 C3/MAG가δ보다 낮은 오차라는 뜻이다. 이 비교는 과거에 확인한 전력16계열 SHIFT8의 좁은 주장에 대응하며 모든 상태의 범용 우위를 뜻하지 않는다.

| proposed | baseline | proposed_nmae | baseline_nmae | gain_pct | bonf4_low | bonf4_high | seed_gains |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C3 | DELTA_XY_BUDGET | 0.364538 | 0.396176 | 7.985661 | 6.785668 | 9.218057 | {"81551": 5.2931204208315314, "81552": 10.48711250732569} |
| C3 | DELTA_XY_DEFAULT | 0.364538 | 0.397764 | 8.353055 | 7.093093 | 9.646950 | {"81551": 5.244766490569286, "81552": 11.215780158447142} |
| MAG_ONLY | DELTA_XY_BUDGET | 0.363040 | 0.396176 | 8.363972 | 7.085480 | 9.639065 | {"81551": 5.474599251536905, "81552": 11.048285286548376} |
| MAG_ONLY | DELTA_XY_DEFAULT | 0.363040 | 0.397764 | 8.729855 | 7.376570 | 10.056687 | {"81551": 5.4263379778457965, "81552": 11.772384785950408} |

- C3 대 DELTA_XY_BUDGET: +7.986%, family4 보정95% 구간 [6.786, 9.218] (양의 구간).
- C3 대 DELTA_XY_DEFAULT: +8.353%, family4 보정95% 구간 [7.093, 9.647] (양의 구간).
- MAG_ONLY 대 DELTA_XY_BUDGET: +8.364%, family4 보정95% 구간 [7.085, 9.639] (양의 구간).
- MAG_ONLY 대 DELTA_XY_DEFAULT: +8.730%, family4 보정95% 구간 [7.377, 10.057] (양의 구간).

같은 전력16계열 SHIFT8에서 C3는 PLAIN보다3.396% 좋지만 MAG 대비 이득은-0.413%다. 즉 **가까운 선행 대비 이득은 확인됐으나 지속성 규칙을 추가할 이유는 여전히 입증되지 않았다.** 모든 source·상태에서 이겨야 한다는 판정은 사용하지 않는다. 기여로 주장한 구성요소와 단순 대조의 차이를 별도로 요구하는 것이다.

## 원자료·오류·변화의 원점수

| panel | condition | B0 | C3 | DELTA_XY_BUDGET | DELTA_XY_DEFAULT | MAG_ONLY | PLAIN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| electricity | FAULT | 0.171460 | 0.171995 | 0.173392 | 0.174537 | 0.172018 | 0.171823 |
| electricity | REFERENCE | 0.166524 | 0.166853 | 0.168567 | 0.169383 | 0.166890 | 0.166614 |
| electricity | SHIFT8 | 0.210032 | 0.201709 | 0.207450 | 0.209685 | 0.201424 | 0.204404 |
| electricity_transfer | FAULT | 0.255210 | 0.255619 | 0.256425 | 0.258797 | 0.255727 | 0.255428 |
| electricity_transfer | REFERENCE | 0.239856 | 0.240185 | 0.241200 | 0.243960 | 0.240273 | 0.239930 |
| electricity_transfer | SHIFT8 | 0.399762 | 0.364538 | 0.396176 | 0.397764 | 0.363040 | 0.377355 |
| ettm1 | FAULT | 0.423198 | 0.423214 | 0.424755 | 0.425459 | 0.423198 | 0.423652 |
| ettm1 | REFERENCE | 0.407336 | 0.407226 | 0.408415 | 0.409590 | 0.407336 | 0.407596 |
| ettm1 | SHIFT8 | 0.545368 | 0.547599 | 0.547086 | 0.549553 | 0.545368 | 0.540239 |

나머지 상태와9개 변화 형태, fixed1024, seed별 값은 [RAW_SCORES.csv](RAW_SCORES.csv), [EFFECTS.csv](EFFECTS.csv), [CHANNEL_SCORES.csv](CHANNEL_SCORES.csv)에 있다. 같은 과거의 PULSE 손해도 삭제하지 않았다. ETTm1 또는 원자료/오류 조건에서의 손해를 전체 평균으로 감추지 않는다. C3와MAG 사이의 과거 판정은 이 비교로 바뀌지 않는다. 특히 전력16계열의 더 긴 STEP12_D63 형태에서 C3 대 기본폭δ의 이득은-3.872%로 손해다. SHIFT8의 양성을 모든 변화 형태로 확대하지 않는다.

## 자원과 남은 한계

다음은16개경로의 군별 평균이며 LR선택 경로도 포함한다. 실제 시간은 [RESOURCES.csv](RESOURCES.csv)에 경로별로 남겼다. 같은update 수는 같은FLOPs나같은wall time과 같지 않다.

| arm | trainable_parameters | optimizer_seconds | peak_allocated_MiB |
| --- | --- | --- | --- |
| DELTA_XY_BUDGET | 8766.000000 | 35.792860 | 366.435059 |
| DELTA_XY_DEFAULT | 1116736.000000 | 36.023694 | 379.352051 |

GPU guard 표본4601개, 비승인 외부 compute0개, 최소 여유8255MiB. RustDesk만 허용했다. 별도 quantile crossing 비율은 [QUANTILE_CROSSING.csv](QUANTILE_CROSSING.csv)에 있으며 이는 사후 진단이고 선택 기준을 바꾸지 않는다.

가까운 선행을 추가했어도 C3 고유의 지속성 요소가 MAG보다 필요하다는 근거가 자동으로 생기지는 않는다. 작은 latent adapter와 boundary adapter의 차이는 위치·연산·초기화가 함께 다르므로 이 비교만으로 개별 원인의 인과기여를 정하지 않는다. 데이터는 이미 사용한 개발E이며 전력의16개 다른계열도 독립자료가 아니다. 실제 센서사건 label·독립확인·정식 선행의전체 설정 재현·충분한 신규성은 남아 있다. 확인된 좁은 이득은 보존하지만 방법론 논문의 전체 목표는 아직 완료로 표시하지 않는다.

모델·origin·조건·loss·LR·seed·update·checkpoint·선택 규칙은 사전 봉인을 유지했다. 새 후보·추가 학습을 자동 실행하지 않았다. [고정 계약](../../experiments/delta_adapter_comparison_20260919/PROTOCOL.md), [공식 코드 receipt](PRIOR_CODE_RECEIPTS.json), [독립 검산](PUBLICATION_AUDIT.json), [별도 bootstrap·비용·seed 검토](COMPARISON_REVIEW_KO.md), [재검산 절차](../../research/delta_adapter_review_20260919/REPRODUCTION_KO.md). 원자료·모델·예측 cache는 로컬 보관이며 GitHub에는 코드·해시·점수·그림이 있다.
