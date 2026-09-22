# 서비스 축 PEFT 여지 검증 (B') — 실행 계약

2026-09-22. 외부 실행 계약 MASTER CLI v2 를 받아 실행한 기록이다. 선행 실험
[간헐 수요 go/no-go (v1)](../intermittent_gonogo_20260922/PROTOCOL.md)에서 편향 축이 닫힌 뒤,
남은 서비스 축을 검증한다.

## 질문

M5 일 단위 간헐 수요에서, 방법과 무관한 공정한 재고 정책으로 비교했을 때 학습 없는 최선의 선택지
(zero-shot TSFM, 비영 정규화, 고전 간헐 기법)보다 목표 데이터로 학습한 모델이 같은 충족률을 더 적은
재고로 달성하는가.

## 판정 사슬

```
V0  측정 도구·배선 검증 (v1 에서 생략한 논문 재현 포함)      실패 -> STOP_DEBUG
V1  zero-shot TSFM 이 고전 기법보다 서비스에서 뒤지는가       서사를 정한다
V2  v1 의 +43.78% 가 어디서 왔는가 (한 번에 한 요인)          보고 전용
V3  목표 데이터 학습 모델이 학습 없는 최선보다 나은가         결정 게이트
RAF 같은 비교의 방향이 부품 수요에서도 같은가                 방향 확인 전용
V4  LoRA 가 V3 의 여지를 잡는가                               V3 통과 + 승인 후에만
```

## 창 (M5)

```
TRAIN   학습형 모델의 학습 표적 <= d_1549
CAL-A   o in {1549, 1577}                      학습형 모델 선택 전용
CAL-B   o in {1605, 1633, 1661, 1689}          ERP 잔차, 임계값, 기준선 선택
EVAL    o in {1717,1745,1773,1801,1829,1857,1885,1913}   판정 전용
창 = d_{o+1} ~ d_{o+28}.  모든 예측의 context 는 o 이하 관측만 쓴다.
적격: d_1 ~ d_1549 양수 관측 >= 20  ->  29,502 / 30,490
```

## 정의

```
QMEAN21   모델 학습 분위수 21개를 max(Q,0) 로 자른 뒤 사다리꼴 적분, 양 끝 상수 연장
SBC       ADI = (첫 양수 ~ 마지막 양수 구간) / 양수 개수,  CV^2 = (양수 크기 변동계수)^2
          경계 ADI 1.32, CV^2 0.49.  논문 프로파일(mean 8.649364)을 RAF 로 역추적해 특정한 정의다
ERP       경험적 잔차 정책 (방법 무관). M5: L=7, R=1, P=8
          잔차 e = (연속 P일 실제 합) - mu_hat x P,  CAL-B 4창 x 21블록 = 계열당 84개
          S(tau) = max(0, mu_hat x P + Q_tau(e)),  tau 격자 12점 (0.50 ~ 0.99)
시뮬레이터 lost sales. t일 마감 발주분은 t+L+1 일 입고 단계에서 들어온다(파이프 길이 L+1)
지표      U = 총충족/총수요,  I = 계열 평균 마감보유의 합,  I@90 = U=0.90 의 I 를 선형보간
판정      Delta_w = 100 x (I@90_기준 - I@90_arm) / I@90_기준
          PASS <=> mean >= T_stock,  95% t-하한 > 0 (df=7),  양수 창 >= 7/8
임계      T_stock = max(3.0, SBA-TSB 격차)  ->  CAL-B 에서 0.773% 가 나와 3.000% 적용
```

## arm

```
학습 없음 TSFM   ZS-MED(보고용), ZS-Q(QMEAN21), NZ-Q(비영 조건부 정규화)
학습 없음 고전   SNAIVE(season 7), SES(optimized), SBA(CrostonSBA), TSB(alpha_d=alpha_p=0.1)
학습형          REF-LGB-U  univariate 특성만 (lag 1/7/14/28, rolling mean 7/28/56/365,
                zero-rate 56/365, rolling sd 365, 표적일 요일, horizon)
                학습 원점 1521-28k (k=0..11), CAL-A 2창으로 early stopping, best_iter 168
V4 전용         LORA-ORIG, LORA-NZ  (미실행)
```

## 봉인

CAL-B 만 채점해 ERP 잔차·T_stock·S-best·best_iter 를 정하고 SEAL.json 에 hash 를 남긴 뒤 EVAL 을 열었다.
EVAL 은 선택·임계값 결정에 쓰지 않았다.

## 실행 순서

```
v0a_paper_repro.sh     V0a 논문 환경 재현 (uv sync -> run_raf_instancenorm -> 고정 참조 대조)
v0_rest.py             V0b(v1 재현) V0c(시뮬레이터) V0d(추론 재현) V0e(고전) V0g(누설)
v0f_nz_wiring.py       V0f NZ 배선. orig/nz 를 반드시 다른 프로세스로 실행
gen_predictions.py     14창 x 9 arm 예측 생성 (tsfm_zs / tsfm_nz / classical / lgb)
decide.py              CAL-B 채점 -> 봉인 -> EVAL V1·V3
ladder.py              V2 사다리 L0-L6
prb_windows.py         12창 x arm PRB
raf_direction.py       RAF 방향 확인 (zs / nz / classical / lgb / score)
verify_recompute.py    저장 곡선에서 판정 표 독립 재계산
core.py                시뮬레이터·ERP·지표·판정 통계 공통 모듈
```

## 환경

```
Ubuntu 22.04, RTX 3080 10GB
chronos-forecasting 2.3.2, torch 2.8.0+cu128, amazon/chronos-2 (bf16, context 512)
statsforecast 2.1.1, lightgbm 4.7.0, xlrd 2.0.2, numpy 2.4.6, scipy 1.17.1, pandas 2.3.3
논문 저장소 sfcheng-research/icdm-2026-reproduction @ 067687b1 (external/, gitignored)
RAF 원본 sha256 1165f2d54759bf7661becc39a2e80265626b59965c61f2ad8ad12a3d7940cf09
```

계열 단위 예측·시뮬레이션 원본은 `.cache/service_axis_v2_20260922/` 에 로컬 보존한다(gitignored).
