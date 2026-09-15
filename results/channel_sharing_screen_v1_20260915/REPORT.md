# R2 채널 변환 예산 비교 — 한국어 결과

실행 상태 **COMPLETE**. smoke 24/24 updates, 본학습 24/24 fits, 9602 updates.

## 어떤 파라미터를 공유했나, 왜 필요한가

LH는 LoRA+HEAD 참고군이다. 다섯 channel arm은 같은 LoRA/frequency/HEAD를 사용한다. 독립 up, 완전 공유 up, 채널별 left×공통 right, 공통 affine+3 residual basis를 고정 예산에서 비교했다. 다른 채널 입력을 읽는 새 attention은 없다.

## 단순 공유·저랭크와의 차이 및 예산

BASIS는 vectorized affine weights를 공통 bank와 채널 계수로 factorization한 알려진 매개변수화다. FACTOR는 feature-to-hidden 행렬의 일반 저랭크 대조군이다. 폭도 다르므로 basis만의 단일 인과효과 비교가 아니다. [선행 경계](../../research/peft_rank12_20260915/NOVELTY_BOUNDARY.md), [상류 차이](../../research/peft_rank12_20260915/UPSTREAM_DIFF.md).

| arm | 채널 블록 | 전체 trainable | 전체 모델 |
| --- | ---: | ---: | ---: |
| LH | 0 | 761952 | 36099360 |
| INDIV_REF | 4474112 | 5498720 | 40836128 |
| INDIV_BUDGET | 295952 | 1320560 | 36657968 |
| SHARED_BUDGET | 295103 | 1319711 | 36657119 |
| FACTOR_BUDGET | 295935 | 1320543 | 36657951 |
| BASIS_BUDGET | 295103 | 1319711 | 36657119 |

동일 예산군은 INDIV_BUDGET/SHARED_BUDGET/FACTOR_BUDGET/BASIS_BUDGET이다. LH와 큰 INDIV_REF는 별도 참조군이다. HEAD는 모든 arm에서 같은 shape/초기값으로 학습했다. [실제 목록](PARAMETER_BUDGET.csv).

## 원점수와 추가 가치

| 원천 | seed | arm/role | MSE | MAE |
| --- | --- | --- | ---: | ---: |
| electricity | None | SEASONAL_NAIVE/seasonal_naive | 0.525609095 | 0.445602675 |
| electricity | 41000 | SHARED_BUDGET/INIT | 1.04903389 | 0.767514712 |
| electricity | 41000 | SHARED_BUDGET/selected | 0.489744416 | 0.512276884 |
| electricity | 41001 | SHARED_BUDGET/INIT | 0.930316013 | 0.716808821 |
| electricity | 41001 | SHARED_BUDGET/selected | 0.488275894 | 0.505573265 |
| electricity | 41001 | INDIV_BUDGET/INIT | 0.91941648 | 0.720408637 |
| electricity | 41001 | INDIV_BUDGET/selected | 0.734953723 | 0.632336064 |
| electricity | 41000 | FACTOR_BUDGET/INIT | 1.07812416 | 0.778577752 |
| electricity | 41000 | FACTOR_BUDGET/selected | 0.766962164 | 0.652613421 |
| electricity | 41001 | INDIV_REF/INIT | 1.0083409 | 0.754285591 |
| electricity | 41001 | INDIV_REF/selected | 0.928597866 | 0.727705993 |
| electricity | 41000 | LH/INIT | 0.803364798 | 0.67807591 |
| electricity | 41000 | LH/selected | 0.465861103 | 0.481035834 |
| electricity | 41001 | LH/INIT | 0.845955876 | 0.695740772 |
| electricity | 41001 | LH/selected | 0.472898829 | 0.495081531 |
| electricity | 41001 | BASIS_BUDGET/INIT | 1.00266415 | 0.749018929 |
| electricity | 41001 | BASIS_BUDGET/selected | 0.546493559 | 0.532266974 |
| electricity | 41001 | FACTOR_BUDGET/INIT | 1.01654738 | 0.744905336 |
| electricity | 41001 | FACTOR_BUDGET/selected | 0.707287387 | 0.624525518 |
| electricity | 41000 | BASIS_BUDGET/INIT | 0.960052306 | 0.735206382 |
| electricity | 41000 | BASIS_BUDGET/selected | 0.537314491 | 0.541616175 |
| electricity | 41000 | INDIV_REF/INIT | 0.979217964 | 0.737265783 |
| electricity | 41000 | INDIV_REF/selected | 0.930391708 | 0.720433999 |
| electricity | 41000 | INDIV_BUDGET/INIT | 0.955430572 | 0.735361429 |
| electricity | 41000 | INDIV_BUDGET/selected | 0.781029583 | 0.664277238 |
| traffic | None | SEASONAL_NAIVE/seasonal_naive | 0.981448809 | 0.488784538 |
| traffic | 41001 | INDIV_BUDGET/INIT | 1.62753796 | 0.938820091 |
| traffic | 41001 | INDIV_BUDGET/selected | 1.62753796 | 0.938820091 |
| traffic | 41001 | INDIV_REF/INIT | 1.64969021 | 0.921792955 |
| traffic | 41001 | INDIV_REF/selected | 1.64969021 | 0.921792955 |
| traffic | 41001 | SHARED_BUDGET/INIT | 1.59865465 | 0.903791305 |
| traffic | 41001 | SHARED_BUDGET/selected | 1.59865465 | 0.903791305 |
| traffic | 41000 | FACTOR_BUDGET/INIT | 1.5558304 | 0.900674453 |
| traffic | 41000 | FACTOR_BUDGET/selected | 1.5558304 | 0.900674453 |
| traffic | 41000 | INDIV_BUDGET/INIT | 1.54886103 | 0.891811974 |
| traffic | 41000 | INDIV_BUDGET/selected | 1.54886103 | 0.891811974 |
| traffic | 41001 | FACTOR_BUDGET/INIT | 1.45738903 | 0.866725327 |
| traffic | 41001 | FACTOR_BUDGET/selected | 1.45738903 | 0.866725327 |
| traffic | 41000 | BASIS_BUDGET/INIT | 1.6265802 | 0.905530181 |
| traffic | 41000 | BASIS_BUDGET/selected | 1.6265802 | 0.905530181 |
| traffic | 41001 | BASIS_BUDGET/INIT | 1.52721301 | 0.887556918 |
| traffic | 41001 | BASIS_BUDGET/selected | 1.52721301 | 0.887556918 |
| traffic | 41000 | SHARED_BUDGET/INIT | 1.70333552 | 0.951600065 |
| traffic | 41000 | SHARED_BUDGET/selected | 1.70333552 | 0.951600065 |
| traffic | 41001 | LH/INIT | 1.29348354 | 0.796298418 |
| traffic | 41001 | LH/selected | 1.25046529 | 0.80610458 |
| traffic | 41000 | INDIV_REF/INIT | 1.55990005 | 0.894667029 |
| traffic | 41000 | INDIV_REF/selected | 1.55990005 | 0.894667029 |
| traffic | 41000 | LH/INIT | 1.24295948 | 0.763470514 |
| traffic | 41000 | LH/selected | 1.22025851 | 0.793952514 |

MSE/MAE는 Train 표준화 공간이다. [채널별 원단위 MAE·유효 분모](channel_metrics.csv). INIT는 새 예측 head의 초기 상태이며 유효한 pretrained zero-shot이라고 하지 않는다. seasonal-naive는 직전24시간 반복이다.

| 원천 | 대조 | 기준 평균 MSE | BASIS 평균 MSE | 개선율(%) | paired 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| electricity | INDIV_REF | 0.929494787 | 0.541904025 | +41.6991 | [+39.3888, +43.9970] |
| electricity | INDIV_BUDGET | 0.757991653 | 0.541904025 | +28.5079 | [+27.2896, +29.9274] |
| electricity | SHARED_BUDGET | 0.489010155 | 0.541904025 | -10.8165 | [-11.7437, -9.6196] |
| electricity | FACTOR_BUDGET | 0.737124775 | 0.541904025 | +26.4841 | [+24.2032, +28.6947] |
| electricity | LH | 0.469379966 | 0.541904025 | -15.4510 | [-16.9670, -13.9113] |
| electricity | V_SELECTED_BUDGET | 0.489010155 | 0.541904025 | -10.8165 | [-11.7437, -9.6196] |
| electricity | INIT | 0.981358228 | 0.541904025 | +44.7802 | [+41.2945, +48.1104] |
| electricity | SEASONAL_NAIVE | 0.525609095 | 0.541904025 | -3.1002 | [-10.5611, +3.2049] |
| traffic | INDIV_REF | 1.60479513 | 1.57689661 | +1.7384 | [+1.0888, +2.3128] |
| traffic | INDIV_BUDGET | 1.5881995 | 1.57689661 | +0.7117 | [-0.5127, +1.7884] |
| traffic | SHARED_BUDGET | 1.65099508 | 1.57689661 | +4.4881 | [+3.8853, +5.2248] |
| traffic | FACTOR_BUDGET | 1.50660972 | 1.57689661 | -4.6652 | [-6.2312, -2.8732] |
| traffic | LH | 1.2353619 | 1.57689661 | -27.6465 | [-33.0124, -23.0532] |
| traffic | V_SELECTED_BUDGET | 1.57375784 | 1.57689661 | -0.1994 | [-1.2202, +0.7170] |
| traffic | INIT | 1.57689661 | 1.57689661 | +0.0000 | [+0.0000, +0.0000] |
| traffic | SEASONAL_NAIVE | 0.981448809 | 1.57689661 | -60.6703 | [-75.6063, -48.1951] |

두 seed 원점수 평균 후 개선율을 계산했다. [seed별 비교](comparisons.csv). V_SELECTED_BUDGET은 V에서 3개 대조군을 고른 사전 봉인 정책이며 선택 비용은 source/seed당3 fits다. E 최저 모델을 사후 정책 점수로 쓰지 않았다.

- electricity: EXECUTION_VALID=True, BUDGET_SIGNAL=False, COMPRESSION_SIGNAL=True, REF가 LH보다 나음=False, 학습6군 중 최저 MSE=LH, NOVELTY_STATUS=KNOWN_PARAMETERIZATION.
- traffic: EXECUTION_VALID=True, BUDGET_SIGNAL=False, COMPRESSION_SIGNAL=True, REF가 LH보다 나음=False, 학습6군 중 최저 MSE=LH, NOVELTY_STATUS=KNOWN_PARAMETERIZATION.

## 실제 자원과 선택

| fit | epochs/updates | active 초 | wall 초 | peak allocated GiB |
| --- | --- | ---: | ---: | ---: |
| 00_electricity_41000_SHARED_BUDGET | 8/512 | 27.239 | 75.430 | 1.0832 |
| 01_traffic_41001_INDIV_BUDGET | 5/315 | 21.965 | 52.662 | 1.0823 |
| 02_traffic_41001_INDIV_REF | 5/315 | 21.862 | 52.008 | 1.1482 |
| 03_traffic_41001_SHARED_BUDGET | 5/315 | 16.831 | 46.384 | 1.0842 |
| 04_electricity_41001_SHARED_BUDGET | 15/960 | 51.335 | 139.430 | 1.0854 |
| 05_electricity_41001_INDIV_BUDGET | 6/384 | 26.771 | 63.878 | 1.0857 |
| 06_traffic_41000_FACTOR_BUDGET | 5/315 | 17.019 | 46.889 | 1.0867 |
| 07_traffic_41000_INDIV_BUDGET | 5/315 | 21.986 | 52.686 | 1.0857 |
| 08_electricity_41000_FACTOR_BUDGET | 6/384 | 20.896 | 57.092 | 1.0867 |
| 09_traffic_41001_FACTOR_BUDGET | 5/315 | 17.034 | 46.758 | 1.0796 |
| 10_electricity_41001_INDIV_REF | 6/384 | 26.620 | 64.003 | 1.1465 |
| 11_traffic_41000_BASIS_BUDGET | 5/315 | 17.499 | 47.702 | 1.0862 |
| 12_electricity_41000_LH | 6/384 | 18.551 | 53.761 | 1.0431 |
| 13_electricity_41001_LH | 8/512 | 25.082 | 71.902 | 1.0475 |
| 14_traffic_41001_BASIS_BUDGET | 5/315 | 17.571 | 47.204 | 1.0896 |
| 15_electricity_41001_BASIS_BUDGET | 6/384 | 21.515 | 57.423 | 1.0862 |
| 16_electricity_41001_FACTOR_BUDGET | 6/384 | 20.901 | 56.786 | 1.0857 |
| 17_traffic_41000_SHARED_BUDGET | 5/315 | 16.839 | 46.686 | 1.0879 |
| 18_traffic_41001_LH | 6/378 | 18.568 | 53.438 | 1.0458 |
| 19_electricity_41000_BASIS_BUDGET | 7/448 | 24.629 | 66.439 | 1.0872 |
| 20_traffic_41000_INDIV_REF | 5/315 | 20.771 | 50.378 | 1.1501 |
| 21_electricity_41000_INDIV_REF | 9/576 | 37.674 | 91.029 | 1.1479 |
| 22_electricity_41000_INDIV_BUDGET | 6/384 | 25.007 | 60.705 | 1.0867 |
| 23_traffic_41000_LH | 6/378 | 17.364 | 52.436 | 1.0458 |

같은 epoch·노출 상한 비교이며 동일 wall-time 비교가 아니다. [자원 상세](resources.csv). 미완료 fit에서 없는 시간 값은 측정 완료를 뜻하지 않는다.
BUDGET_LIMITED 0 fits, INIT 선택 10 fits. min_delta는 patience 리셋에만 쓰고 실제 최저 V는 작은 개선도 저장했다.

## V에서만 수행한 계수 개입

| 원천 | seed | 개입 | 원래 V MSE | 개입 V MSE |
| --- | --- | --- | ---: | ---: |
| traffic | 41000 | permuted | 1.24243336 | 1.24243336 |
| traffic | 41000 | channel_mean | 1.24243336 | 1.24243336 |
| traffic | 41001 | permuted | 1.20572463 | 1.20572463 |
| traffic | 41001 | channel_mean | 1.20572463 | 1.20572463 |
| electricity | 41001 | permuted | 0.634157725 | 0.636166826 |
| electricity | 41001 | channel_mean | 0.634157725 | 0.632094838 |
| electricity | 41000 | permuted | 0.545530214 | 0.543751477 |
| electricity | 41000 | channel_mean | 0.545530214 | 0.540914699 |

계수 순열과 채널 평균 대체는 고정 사후 진단이며 selection·threshold를 바꾸지 않았다. 분포 밖 개입일 수 있어 고유한 인과 기여 증명은 아니다.

## 검증·신규성·평가 한계

저장 예측 257개를 독립 scalar float64로 다시 계산했다. 최대 허용 절대차1e-10이며 실제 값은 [검산 기록](independent_verification.json)에 있다. 기존 1498개 결과·연구 파일과 모델/소스 해시를 확인했다.

Time-PEFT-inspired controlled pilot이다. 32채널·길이96·BF16·최대20epochs·고정LR·2seed·이미 본 두 원천의 결과다. 이전64채널 결과 이후의 설계 변경이므로 독립 확증이나 공정한 단일 변수 전후 비교로 포장하지 않는다. 공식 Time-PEFT 전체 성능 재현 및 공식 C-LoRA/MoLA 직접 비교는 수행하지 않았다.

NEXT_ACTION: 이번 예산 비교의 결과와 최강 단순 대조군을 검토해 후속 투자 여부를 결정한다. 후속 학습은 자동 실행하지 않는다.
