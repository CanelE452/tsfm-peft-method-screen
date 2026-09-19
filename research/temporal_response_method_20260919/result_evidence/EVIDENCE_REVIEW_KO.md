# TRP 방법론 근거 추가 검토

봉인된 학습·평가 완료 여부는 원 REPORT.md를 따른다. 고정 투자 신호=False. 이 자료의 새 학습·모델 추론·예측 채점은0회다.

## 주 비교와 seed

| baseline | TRP_nmae | baseline_nmae | gain_pct | bonferroni4_low | bonferroni4_high |
| --- | --- | --- | --- | --- | --- |
| PLAIN | 0.399733 | 0.377355 | -5.9304 | -7.14091 | -4.8058 |
| ANCHOR | 0.399733 | 0.399759 | 0.0063426 | 0.00463543 | 0.00812557 |
| SHUFFLE | 0.399733 | 0.398086 | -0.413852 | -0.460938 | -0.368017 |
| IDEAL | 0.399733 | 0.358506 | -11.4998 | -14.2058 | -9.1527 |

| baseline | seed | TRP_nmae | baseline_nmae | gain_pct |
| --- | --- | --- | --- | --- |
| PLAIN | 81551 | 0.384274 | 0.364847 | -5.32476 |
| PLAIN | 81552 | 0.415193 | 0.389863 | -6.49719 |
| ANCHOR | 81551 | 0.384274 | 0.384318 | 0.0115835 |
| ANCHOR | 81552 | 0.415193 | 0.415199 | 0.00149148 |
| SHUFFLE | 81551 | 0.384274 | 0.38376 | -0.133969 |
| SHUFFLE | 81552 | 0.415193 | 0.412412 | -0.674291 |
| IDEAL | 81551 | 0.384274 | 0.362817 | -5.91389 |
| IDEAL | 81552 | 0.415193 | 0.354195 | -17.2216 |

## 강한 기존 방법과의 관계

| panel | condition | baseline | gain_pct | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| electricity | FAULT | B0 | 0.00101457 | 0.000332419 | 0.00183963 |
| electricity | FAULT | C3 | 0.311717 | 0.0907917 | 0.563179 |
| electricity | FAULT | MAG_ONLY | 0.325264 | 0.10988 | 0.56646 |
| electricity | REFERENCE | B0 | 0.000968199 | 0.000297928 | 0.00175544 |
| electricity | REFERENCE | C3 | 0.198088 | -0.0171067 | 0.441328 |
| electricity | REFERENCE | MAG_ONLY | 0.22001 | 0.00690566 | 0.46135 |
| electricity | SHIFT8 | B0 | 0.00442881 | 0.00322557 | 0.0057464 |
| electricity | SHIFT8 | C3 | -4.12165 | -5.05339 | -3.30595 |
| electricity | SHIFT8 | MAG_ONLY | -4.26905 | -5.22714 | -3.4147 |
| electricity_transfer | FAULT | B0 | 0.00148168 | 0.000858476 | 0.0021069 |
| electricity_transfer | FAULT | C3 | 0.161329 | 0.0344705 | 0.299201 |
| electricity_transfer | FAULT | MAG_ONLY | 0.203656 | 0.0693439 | 0.348339 |
| electricity_transfer | REFERENCE | B0 | 0.0015647 | 0.0010257 | 0.00209901 |
| electricity_transfer | REFERENCE | C3 | 0.138615 | -0.0157117 | 0.302041 |
| electricity_transfer | REFERENCE | MAG_ONLY | 0.175127 | 0.0108604 | 0.353729 |
| electricity_transfer | SHIFT8 | B0 | 0.00713141 | 0.00577258 | 0.00855787 |
| electricity_transfer | SHIFT8 | C3 | -9.65467 | -10.9103 | -8.47916 |
| electricity_transfer | SHIFT8 | MAG_ONLY | -10.1074 | -11.4261 | -8.85295 |
| ettm1 | FAULT | B0 | 0.00178235 | -0.00387525 | 0.00809807 |
| ettm1 | FAULT | C3 | 0.00575022 | -0.160121 | 0.159179 |
| ettm1 | FAULT | MAG_ONLY | 0.00178235 | -0.00387525 | 0.00809807 |
| ettm1 | REFERENCE | B0 | 0.0030786 | -0.00471079 | 0.0115331 |
| ettm1 | REFERENCE | C3 | -0.0241175 | -0.179297 | 0.129265 |
| ettm1 | REFERENCE | MAG_ONLY | 0.0030786 | -0.00471079 | 0.0115331 |
| ettm1 | SHIFT8 | B0 | 0.0156842 | 0.00174465 | 0.0310403 |
| ettm1 | SHIFT8 | C3 | 0.422933 | 0.0873689 | 0.758812 |
| ettm1 | SHIFT8 | MAG_ONLY | 0.0156842 | 0.00174465 | 0.0310403 |

C3/MAG는 기존 학습률 정책을 사용한 맥락 비교다. 이를 이번 동일 loss 대조군처럼 쓰지 않으며, 후보가 단순 MAG보다 불리하면 네 새 대조군의 일부 이득만으로 실용적 우위를 주장하지 않는다.

## 추가 보정 억제 여부

TRAIN_LOSS_COMPONENTS.csv는 실제 training ledger의 초반/중반/후반 supervised·regularizer·gradient norm이다. PARAMETER_MOVEMENT.csv는 checkpoint0→1024의 행렬 이동량이며 함수 보정량이나 성능 인과 기여율과 다르다. 이 수치만으로 특정 gradient 경로가 최종 성능의 원인이라고 판단하지 않는다.

## 비용·한계

| source | arm | train_seconds | validation_seconds | peak_allocated_GiB |
| --- | --- | --- | --- | --- |
| electricity | ANCHOR | 88.6614 | 5.0837 | 0.503421 |
| electricity | IDEAL | 85.5494 | 4.91017 | 0.503879 |
| electricity | SHUFFLE | 86.2055 | 4.91807 | 0.503421 |
| electricity | TRP | 89.3009 | 5.16543 | 0.503879 |
| ettm1 | ANCHOR | 87.4429 | 4.99373 | 0.503879 |
| ettm1 | IDEAL | 87.218 | 4.96698 | 0.503879 |
| ettm1 | SHUFFLE | 87.1773 | 5.00142 | 0.503879 |
| ettm1 | TRP | 87.2362 | 4.99489 | 0.503879 |

아직 독립 새 source·추가 seed·정식 PEFT 선행 비교는 없다. 최적 λ/LR을 탐색하지 않은 고정 개발 파일럿이므로 실패를 모든 가능한 반응 보존 방법의 반증으로 확대하지 않는다. 반대로 탐색하지 않았다는 이유만으로 현 후보를 성공으로 간주하지도 않는다.
