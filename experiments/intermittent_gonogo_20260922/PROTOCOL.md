# 간헐 수요 TSFM 과소예측 PEFT — go/no-go 실행 계약

작성·실행 2026-09-22. 외부 계획 문서(MASTER CLI, 2026-09-22)를 받아 실행 전 검증 후 축약 실행한 기록이다.

## 목적

TSFM PEFT 방법론 논문의 후보 방향 하나 — "간헐 수요에서 TSFM의 과소예측(음의 편향)을 PEFT로
고친다" — 에 추가 GPU·시간을 투입할지 판정한다. 방법의 우월성이나 논문 PASS를 묻지 않는다.

## 판정 사슬

```
게이트 1 (G1)  과소예측이 M5 일 단위에도 있고, 중앙값 추출 탓만은 아닌가   문제 존재
게이트 2-A     단순 보정(분위수 선택·스칼라 스케일링)으로 설명되지 않는가  틈 존재
게이트 2-B     목표 데이터 적응 여지가 있는가                              상한 확인
게이트 3 (G3)  LoRA 가 그 틈을 메우는가                                    방법 가능성
```

2-B 실패 시 3은 실행하지 않는다(적응 여지가 없으면 PEFT 여지도 없다).

## 데이터·분할

```
M5 일 단위 item x store 최하위 계열  (github/m5dataset/data/sales_train_evaluation.csv)
test        d_1914 - d_1941 (28일)
validation  d_1886 - d_1913 (28일)   alpha* 계산 전용
적격성·SBC 분류·LightGBM 학습        d_1885 이하 관측만
context     512,  h = 28
적격 기준   d_1885 까지 양수 관측 >= 20  ->  30,458 / 30,490
```

표본 추출을 하지 않고 적격 전수를 쓴다(전수 추론 15초, 층화로 줄일 계산상 이유가 없다).

## 정의

```
점예측 QMEAN  요청 분위수 13개의 사다리꼴 적분. 양 끝 [0,최저]·[최고,1] 은 끝값 상수 연장,
              결과 음수는 0 절단.   (모델 학습 분위수 21개 전부가 요청 13개를 포함 -> 보간 없음)
PRB           sum_i sum_h (yhat - y) / sum_i sum_h y        음수 = 과소예측
RMSSE_i       sqrt( mean_h (y-yhat)^2 / mean_{t=s_i+1..n} (y_t - y_{t-1})^2 ),  s_i = 첫 양수 시점
alpha*        sum_validation y / sum_validation yhat        validation 에서만 계산
SBC 분류      ADI = (첫 양수 ~ 마지막 양수 구간 길이) / 양수 개수,  CV^2 = (양수 수요 변동계수)^2
              경계 ADI 1.32, CV^2 0.49
```

ADI 정의는 선행 논문 저장소의 고정 프로파일(`results_release/raf_full38/raf_dataset_profile.json`,
mean 8.649364 / median 8.1)을 RAF 데이터로 역추적해 특정했다. 외부 계획 문서가 쓴 "첫 양수 관측
이후 구간" 정의는 같은 데이터에서 9.997 / 9.222 가 되어 논문과 맞지 않는다.

## 서비스·재고 평가기

선행 논문 재현 저장소 `sfcheng-research/icdm-2026-reproduction`(HEAD 067687b1) 의
`src/scripts/10_experiments/run_raf_instancenorm.py` 정의를 이식했다.

```
점예측      clip(0)
sigma_hat   IQR / 1.35          (분위수 0.75 - 0.25)
정책        order-up-to.  prot = L + R,  S = max(0, mean*prot + z(tau)*std*sqrt(prot)),
            s = max(0, S - mean*R)
충족 대리   양수 수요 기간 중 완전 충족 비율. 수요 없는 창은 1점
재고        평균 보유재고(on-hand) 의 시간평균
M5 설정     L = 7, R = 7, tau = 0.90        (원 스크립트는 RAF 기준 L=1, R=6)
Stock@90    alpha 격자에서 평균 충족이 처음 0.90 이상이 되는 두 점 사이 선형 보간
alpha 격자  0.50 - 4.00, 0.05 간격, 71점. alpha 는 점예측과 분산에 함께 곱한다
```

비영 조건부 정규화(NZ)는 같은 파일의 `patch_instancenorm` 을 이식했다. context 안 양수 관측만으로
중심·척도를 계산하고, 양수가 없으면 원래 통계, 척도 0이면 eps 로 치환한다. 설치 패키지를 직접
수정하지 않고 `inner_model.instance_norm.forward` 를 런타임 교체한다.

## arm

```
ZS-QMEAN, ZS-Q80, ZS-Q90   원 정규화 zero-shot Chronos-2
NZ-QMEAN                   비영 조건부 정규화 zero-shot (학습 없음)
SBA, TSB                   원점 이하 관측으로만 적합
REF-LGB                    목표 데이터 학습 LightGBM (게이트 2-B 전용)
sNaive(주간)               편향 대조용 (기존 m5dataset 산출물 재사용)
```

## 판정 기준 (실행 전 고정)

```
G1 (i)    ZS-QMEAN PRB 95% CI 상한 < 0
G1 (ii)   [ZS-QMEAN 충족 - max(SBA, TSB 충족)] CI 상한 < 0
G1 (iii)  [(intermittent+lumpy) PRB - smooth PRB] CI 상한 < 0
추출 교란  ZS-MED 상한 < 0 인데 ZS-QMEAN CI 가 0 을 포함하면 통과로 치지 않는다
게이트 2-A 재고 개선률(기준 ZS-best, arm NZ-QMEAN) >= 3.0% 이고 CI 하한 > 0
           ZS-best = {ZS-QMEAN, ZS-Q80, ZS-Q90} 중 test Stock@90 최소 (사후 최강)
게이트 2-B 정확도 개선률(기준 ZS-QMEAN x alpha*, arm REF-LGB x alpha*) >= 1.0% 이고 CI 하한 > 0
통계       계열 부트스트랩 2,000회, seed 20260922, percentile 95% CI
```

## 실행 순서

```
run_precheck_a.py + ci.py   점예측 추출 규칙 민감도, SBC 정의 2종, PRB 3가중, G1 (i)(iii)
run_precheck_b.py           LoRA 배선 smoke 10 step (seed / 체크포인트 / 시간·VRAM)
run_gonogo.py               ZS·NZ 추론, SBA/TSB, alpha 곡선, Stock@90, G1 (ii), 게이트 2-A
val_pred.py                 validation 구간 예측 (alpha* 용)
g2b.py / g2b_full.py / g2b_full12.py   게이트 2-B 를 세 설정으로 (공변량 유무 x 학습원점 4/12)
```

## 실행하지 않은 것

- 게이트 0 (선행 논문 재현으로 측정 도구를 검증하는 단계). 평가기는 코드 이식 수준에서만 담보된다.
- RAF 부품 수요 데이터. 원본은 확보했으나(아래) 측정하지 않았다.
- 봉인 절차. 빠른 판정을 택했으므로 이 기록은 사전등록된 판정이 아니라 탐색적 측정이다.
- 게이트 3 (LoRA 16 fits). 게이트 2-B 실패로 규칙상 진행하지 않았다.

## 외부 자료

```
논문 저장소   github.com/sfcheng-research/icdm-2026-reproduction  HEAD 067687b1
RAF 원본      github.com/danieldehaan96/spdf
              "RAF data - 7 years demand  - 5000 items.xls"
              3,001,344 bytes  sha256 1165f2d54759bf7661becc39a2e80265626b59965c61f2ad8ad12a3d7940cf09
              논문 고정 프로파일과 zero_fraction 0.8983452380952381 까지 일치 확인
환경          Ubuntu 22.04, RTX 3080 10GB, chronos-forecasting 2.3.2, torch 2.8.0+cu128,
              amazon/chronos-2 (로컬 캐시), lightgbm 4.6.0
```
