# PETSA 공개 보정 부품의 offline 직접 대조

실행·검산 완료. 판정: ADDITIONAL_BASELINE_SUPPORT_NOT_ESTABLISHED. 공식 PETSA 전체 재현이나 논문 PASS를 뜻하지 않는다.

## 실제 실행과 검산

8/8 새 fits, 8192 main +4 smoke updates. 기존192개 prediction views를 hash 재사용하고 새32개를 추가해 총224개를 검산했다. 미실행 승인 학습0. 모델 선택 봉인→새 전체 예측 저장→새 E 채점 순서, 40개 checkpoint, 원점별 독립 metric 555492개, 기존 점수 1920행의 재현과 bootstrap 구간 재계산을 검사했다. 실제 초기 B0 동일성, 두 cell의 update, 동결 보존, fresh restore를 확인했다. 미승인 외부 GPU compute 표본 0개. raw/weights/predictions는 로컬 캐시이며 공개 저장소만으로 모든 수치가 재생된다고 하지 않는다.

## 방법과 비교 범위

공식 PETSA GCM의 rank16·초기 gate .01·var-wise 수식을 이식했다. 관측 mean/기존 TRAIN sigma 좌표에서 입력·출력 correction을 더하며 출력 cell은9개 quantile에 공유한다. 이 좌표계와 probabilistic output 연결은 로컬 이식 선택이다. 공식 온라인 partial/delayed-label 적응과 복합 loss는 재현하지 않았다. 같은 B0·TRAIN·두LR·선택seed·두 반복seed·1024updates·V nMAE 조건으로 normalized2pinball을 학습했다. MAG는 변경하거나 재튜닝하지 않았다. PETSA cell19010 vsMAG8712 추가params이며 예산이 같은 adapter 대조라고 부르지 않는다. 기존 B0의 LoRA294912와 학습 비용도 따로 존재한다.

## 주 비교와 손해

| panel | baseline | proposed_nmae | baseline_nmae | gain_pct | bonf2_low | bonf2_high | seed_gains |
| --- | --- | --- | --- | --- | --- | --- | --- |
| electricity_transfer | PETSA_XY_OFFLINE | 0.363040 | 0.366591 | 0.968776 | 0.371636 | 1.506718 | {"81551": 0.7953206066216634, "81552": 1.1394251329955973} |
| neso_2026_jul_aug | PETSA_XY_OFFLINE | 0.277940 | 0.279586 | 0.588908 | -0.583206 | 1.669347 | {"81551": 0.754297333359244, "81552": 0.42870239146978184} |

양수는 MAG 이득이다. 두 primary 비교에 한한 Bonferroni family2 구간이며, 고정 두seed/채널에 조건부인 index7일 block bootstrap2000회다. 역사 전체의 후보 탐색이나 seed 모집단의 불확실성을 보정하지 않는다. 제한된 추가 대조 기준 충족: False. REFERENCE/FAULT의1% 손해 한도는 평균점수 기준이지 신뢰구간 비열등성 보장이 아니다. 기존 학습형 gate 비교의 family4와 이번 family2를 합쳐 하나의 사전 확증이라고 주장하지 않는다.

| panel | condition | baseline | gain_pct | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| electricity_transfer | FAULT | PETSA_XY_OFFLINE | -0.248779 | -0.534565 | -0.045779 |
| electricity_transfer | FAULT | B0 | -0.202587 | -0.347322 | -0.067649 |
| electricity_transfer | FAULT | PLAIN | -0.117303 | -0.220529 | -0.023910 |
| electricity_transfer | REFERENCE | PETSA_XY_OFFLINE | -0.287149 | -0.626123 | -0.028928 |
| electricity_transfer | REFERENCE | B0 | -0.173867 | -0.358162 | -0.007743 |
| electricity_transfer | REFERENCE | PLAIN | -0.142947 | -0.262851 | -0.033354 |
| neso_2026_jul_aug | FAULT | PETSA_XY_OFFLINE | -0.848323 | -1.586667 | -0.177835 |
| neso_2026_jul_aug | FAULT | B0 | -0.400960 | -1.055725 | 0.201049 |
| neso_2026_jul_aug | FAULT | PLAIN | -0.443399 | -0.845257 | 0.007668 |
| neso_2026_jul_aug | REFERENCE | PETSA_XY_OFFLINE | -0.778187 | -1.724661 | 0.130012 |
| neso_2026_jul_aug | REFERENCE | B0 | -0.242490 | -1.036023 | 0.630020 |
| neso_2026_jul_aug | REFERENCE | PLAIN | -0.300761 | -0.773854 | 0.206631 |

## 원점수와 비용

| panel | condition | B0 | C3 | MAG_ONLY | PETSA_XY_OFFLINE | PLAIN | TOKEN_GATE | TOKEN_GATE_ENTROPY |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| electricity | FAULT | 0.171460 | 0.171995 | 0.172018 | 0.171718 | 0.171823 | 0.171734 | 0.171725 |
| electricity | REFERENCE | 0.166524 | 0.166853 | 0.166890 | 0.166811 | 0.166614 | 0.166511 | 0.166629 |
| electricity | SHIFT4 | 0.191307 | 0.189130 | 0.189160 | 0.190285 | 0.189093 | 0.189147 | 0.189325 |
| electricity | SHIFT8 | 0.210032 | 0.201709 | 0.201424 | 0.202196 | 0.204404 | 0.204271 | 0.202533 |
| electricity | SHIFT_POINT | 0.196221 | 0.193432 | 0.193520 | 0.195063 | 0.193546 | 0.193779 | 0.194141 |
| electricity_transfer | FAULT | 0.255210 | 0.255619 | 0.255727 | 0.255093 | 0.255428 | 0.255584 | 0.255194 |
| electricity_transfer | REFERENCE | 0.239856 | 0.240185 | 0.240273 | 0.239585 | 0.239930 | 0.240011 | 0.239613 |
| electricity_transfer | SHIFT4 | 0.326449 | 0.319635 | 0.319098 | 0.320531 | 0.318945 | 0.319229 | 0.317829 |
| electricity_transfer | SHIFT8 | 0.399762 | 0.364538 | 0.363040 | 0.366591 | 0.377355 | 0.375400 | 0.373421 |
| electricity_transfer | SHIFT_POINT | 0.342813 | 0.335071 | 0.334295 | 0.336919 | 0.334722 | 0.336123 | 0.334341 |
| ettm1 | FAULT | 0.423198 | 0.423214 | 0.423198 | 0.423198 | 0.423652 | 0.423588 | 0.423193 |
| ettm1 | REFERENCE | 0.407336 | 0.407226 | 0.407336 | 0.407336 | 0.407596 | 0.407482 | 0.407124 |
| ettm1 | SHIFT4 | 0.551405 | 0.549718 | 0.551405 | 0.551405 | 0.548210 | 0.547819 | 0.548012 |
| ettm1 | SHIFT8 | 0.545368 | 0.547599 | 0.545368 | 0.545368 | 0.540239 | 0.540690 | 0.545137 |
| ettm1 | SHIFT_POINT | 0.589807 | 0.588500 | 0.589807 | 0.589807 | 0.586404 | 0.586056 | 0.586334 |
| neso_2026_jul_aug | FAULT | 0.204939 | 0.205647 | 0.205761 | 0.204030 | 0.204852 | 0.205128 | 0.205280 |
| neso_2026_jul_aug | REFERENCE | 0.191119 | 0.191506 | 0.191583 | 0.190103 | 0.191008 | 0.191050 | 0.191465 |
| neso_2026_jul_aug | SHIFT4 | 0.266394 | 0.265091 | 0.263951 | 0.262343 | 0.264542 | 0.265078 | 0.265461 |
| neso_2026_jul_aug | SHIFT8 | 0.291422 | 0.278363 | 0.277940 | 0.279586 | 0.283443 | 0.283995 | 0.282803 |
| neso_2026_jul_aug | SHIFT_POINT | 0.259068 | 0.255970 | 0.255654 | 0.256942 | 0.256566 | 0.257173 | 0.257787 |

모든seed 원점수는 RAW_SCORES.csv, 날짜별 점수는 ORIGIN_SCORES.csv.gz, 채널별 점수는 CHANNEL_SCORES.csv다. selected/fixed1024, 모든오류·변화형태·ETTm1을 보존했다. QUANTILE_CROSSING.csv에 crossing 빈도를 남기며 출력 정렬로 결과를 바꾸지 않았다.

| arm | trainable_parameters | optimizer_seconds | peak_MiB |
| --- | --- | --- | --- |
| PETSA_XY_OFFLINE | 19010.000000 | 40.466223 | 367.540527 |

optimizer_seconds는 intent 저장 시간을 포함한다. 과거 MAG의 순수 계산 측정과 다르므로 속도비를 주장하지 않는다. 같은 관측 권한은 동일한 특징 표현·개입 위치를 의미하지 않는다.

## 논문 해석과 종료

모든 E 패널은 이번 대조를 설계하기 전에 이미 채점됐다. NESO 후반 기간도 이제 재사용 개발 평가이며 새 독립 검증이라고 부르지 않는다. 긍정 결과는 가까운 공개 보정 부품보다 특정조건에서 유리하다는 근거를 더할 뿐, 공식 PETSA/Time-PEFT 전체 우위나 gating 최초성을 입증하지 않는다. 부정 결과는 그 추가 근거가 확보되지 않았다는 뜻이며 기존 완료 실험의 좁은 양성 결과를 삭제하지 않는다. C3 지속성 규칙의 성공으로 옮기지 않는다. 자동 후속 학습0.
