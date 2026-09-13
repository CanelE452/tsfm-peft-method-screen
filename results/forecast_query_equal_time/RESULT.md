# Forecast-query 동일 학습 시간 비교 결과

판정: **STOP_CURRENT_QUERY**. 기존 메모리·품질 gate의 FAIL은 유지한다.

## 범위와 실행

실행 commit: 7918edd0fb5c29d66e014c633ec794a99e70c7e3. 고정 계획 commit: a33f2456fe573df513a3e062f999ff0ef4d93cc6.
32/32 fits 완료, 실제 학습 updates=12120. Preflight80 updates는 별도다.
총 active training=961.50s, fit wall 합계=1084.63s, 전체 실행 wall=1374.65s.
각 fit은 명목30초+최대0.5초 경계 초과 이내이며, 선택은 V의0/10/20/30초와2LR만 사용했다.
V/E/저장/대기 시간은 active clock 밖이고 wall 장부에 포함한다. end-to-end 동일시간 비교로 해석하지 않는다.

## Primary loss (낮을수록 좋음)

| 데이터 | 방식 | seed30000 | seed30001 | 평균 |
|---|---|---:|---:|---:|
| ettm2 | F0 | 0.26379622 | 0.26379622 | 0.26379622 |
| ettm2 | standard | 0.25973042 | 0.26423198 | 0.26198120 |
| ettm2 | head | 0.26600557 | 0.26379622 | 0.26490090 |
| ettm2 | side | 0.26379622 | 0.26448664 | 0.26414143 |
| ettm2 | query | 0.26271872 | 0.26317801 | 0.26294836 |
| electricity | F0 | 0.08708946 | 0.08708946 | 0.08708946 |
| electricity | standard | 0.08417056 | 0.08381967 | 0.08399511 |
| electricity | head | 0.08708946 | 0.08708946 | 0.08708946 |
| electricity | side | 0.08708946 | 0.08446112 | 0.08577529 |
| electricity | query | 0.08490077 | 0.08503007 | 0.08496542 |

Query는 두 데이터셋 모두 F0와 Side보다 평균 손실이 낮았으나, Standard LoRA보다 ETTm2 약0.3692%, Electricity 약1.1552% 높은 손실을 얻었다.
네 Query 선택 모두 시간>0이고 자체 초기 E 예측보다 개선되어 실제 학습 효과는 확인됐다. 두 데이터셋에서 요구한0.5% 우위는 충족하지 못했으며, ETTm2 seed30000 및 Electricity seed30001은 해당 seed 최선 baseline 대비1% 악화 제한도 넘었다.
따라서 이 예산·데이터·recipe 범위에서 현재 Query의 추가 확장을 중단한다. 전체 연구 질문의 불가능성이나 모든 예산의 성능 우위를 증명한 결과는 아니다.

## 데이터셋별 고정 판정

- ettm2: 최선 기준선=standard, Query 평균 개선율=-0.3692%, 지속 gate=False.
  - seed30000: 최선 기준선 대비 loss ratio=1.011505, 선택 시간>0=True, 자체 초기 출력보다 개선=True.
  - seed30001: 최선 기준선 대비 loss ratio=0.997656, 선택 시간>0=True, 자체 초기 출력보다 개선=True.
- electricity: 최선 기준선=standard, Query 평균 개선율=-1.1552%, 지속 gate=False.
  - seed30000: 최선 기준선 대비 loss ratio=1.008675, 선택 시간>0=True, 자체 초기 출력보다 개선=True.
  - seed30001: 최선 기준선 대비 loss ratio=1.014441, 선택 시간>0=True, 자체 초기 출력보다 개선=True.

## 저장 방식과 비용

| 데이터 | 방식 | checkpoint | preflight median step s | preflight peak MiB |
|---|---|---|---:|---:|
| ettm2 | standard | False | 0.13218 | 1831.18 |
| electricity | standard | False | 0.13200 | 1831.18 |
| ettm2 | head | False | 0.04156 | 611.24 |
| electricity | head | True | 0.04357 | 611.24 |
| ettm2 | side | False | 0.09189 | 759.00 |
| electricity | side | False | 0.09247 | 759.00 |
| ettm2 | query | False | 0.14741 | 949.03 |
| electricity | query | False | 0.15985 | 949.03 |

### 실제 30초 학습에서의 비용

| 데이터 | 방식 | updates 범위 | step 중앙값 s | 최대 peak MiB |
|---|---|---:|---:|---:|
| ettm2 | standard | 245–251 | 0.12079 | 1830.31 |
| ettm2 | head | 654–702 | 0.04392 | 610.84 |
| ettm2 | side | 366–379 | 0.08018 | 759.00 |
| ettm2 | query | 211–218 | 0.13851 | 947.97 |
| electricity | standard | 237–250 | 0.12191 | 1830.31 |
| electricity | head | 665–684 | 0.04403 | 611.59 |
| electricity | side | 372–380 | 0.07931 | 759.00 |
| electricity | query | 216–222 | 0.13615 | 947.97 |

학습 중 각 fit의 실제 업데이트 수·peak·validation/checkpoint/wall 비용은 fits.json에 전부 기록했다.
Preflight3회 시간 차이는 공유 데스크톱의 작은 표본이며 확정적 시스템 우위가 아니다.

## 무결성

32 preflight on/off 쌍 독립 재계산, 최대 수치 차이=0.
166개 prediction cache와16개 선택 checkpoint replay; primary 최대 오차=1.11e-16.
기존 결과 242개 파일과 frozen backbone 파라미터 불변.
GPU 점검 367회, 실행 단계 최소 관측 free=7182MiB, 외부 compute 관측(대기 포함)=True; 실행 단계=False.
모든16 선택을 봉인한 뒤에만 E 평가를 열었다. raw 파일 mechanical staging과 E scoring은 구분한다.

## 해석의 한계

Train/V 일부는 과거 실험의 개발 데이터다. E는 기록상 이전 scoring과 겹치지 않는 이후 구간이며 같은 원천의16 origins뿐이다.
ETTm2와 Electricity는 시간 해상도와 달력상 horizon이 다르다. seed2개는 optimization 반복이며 독립 데이터2개가 아니다.
paired_differences.json의4 chronological block bootstrap 구간은 참고용이며 통계적 유의성이나 논문 PASS의 근거가 아니다.
이번 gate는 같은 명목 최적화 시간의 품질 개선에 관한 개발 지속 조건이다. 메모리20% gate를 바꿔 실패를 성공으로 재분류하지 않는다.
조건을 만족해도 신규성·독립 데이터 일반화·논문 기여는 별도 검증 대상이다. 미충족이면 이 Query의 추가 확장을 중단한다.

## 재현

scripts/with_cuda.sh .venv/bin/python scripts/finalize_forecast_query_equal_time.py --verify-only
원시 예측과 파라미터/Adam cache는 ignored .cache/forecast_query_equal_time에 있다.
GPU runner는 기존 결과 디렉터리를 덮어쓰지 않는다.
