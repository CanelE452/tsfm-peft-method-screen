# LoRA와 같은 파라미터 수의 마지막 adapter 대조

**동일 용량 대조4회와 평가를 완료했다. 알려진 adapter의 폭을 한 번 고정한 진단이며 새로운 방법론 PASS가 아니다.**

실제4/4fits, 4632/5080 본학습updates, smoke4updates. 기존 LH/head-only/POINTWISE16 각4fits는 재학습하지 않았다. 미완료fit0, 실행오류/재시도0. 추가후속학습0.

## 무엇을 확인했는가

공통 head589,920개를 제외한 적응 용량을172,032개로 맞췄다. 마지막 residual h+B GELU(Ah)의 폭은172032/(2×512)=168이다. 점수를 보면서 폭을 선택하지 않았다. 기존 POINTWISE16과는 폭만 달라지고, 모델·head초기함수·sampling·원점·정규화·학습률·epoch 상한을 유지했다. encoder와 초기0-update LoRA는 동결했다. [사전 프로토콜](PROTOCOL.md).

## 원점수와 효과

| 원천 | recipe | 폭168 MSE | LH MSE | 폭16 MSE | head-only MSE |
|---|---|---:|---:|---:|---:|
| electricity | selected | 0.290154 | 0.280010 | 0.290063 | 0.297230 |
| electricity | matched_old_epoch | 0.325419 | 0.312357 | 0.332379 | 0.345150 |
| traffic | selected | 0.407270 | 0.386548 | 0.420738 | 0.436876 |
| traffic | matched_old_epoch | 0.524193 | 0.507064 | 0.532362 | 0.553986 |

각방법의 V 최저 checkpoint와, 사전에 정한 같은epoch/updates 비교를 함께 공개한다. 모든 E는 이미 노출된 개발 자료다. [seed별 MSE·MAE·rawMAE](scores.csv), [대조별 효과](comparisons.csv), [전체 채널](channel_scores.csv).

| 원천 | V 선택 대조 | 폭168 개선율 | 두seed 개선 | 설명용95% 구간 |
|---|---|---:|---|---|
| electricity | BALANCED_LH | -3.623% | False | [-5.140%, -2.407%] |
| electricity | HEAD_ONLY | +2.381% | True | [1.823%, 3.077%] |
| electricity | POINTWISE | -0.031% | False | [-0.328%, 0.310%] |
| traffic | BALANCED_LH | -5.361% | False | [-7.360%, -3.609%] |
| traffic | HEAD_ONLY | +6.777% | True | [4.839%, 8.884%] |
| traffic | POINTWISE | +3.201% | True | [1.989%, 4.534%] |

구간은 두seed를 동일원점에서 평균하고 시간순8블록으로2000회 paired resample한 설명용 결과다(seed9018). 독립 검증이나 새로운 PASS gate가 아니다. 추가 hyperparameter 선택이나 노출 편향을 보정하지 않는다.

## 자원·검산

| fit | epochs | updates | 학습peak MiB | 학습step 합계 초 |
|---|---:|---:|---:|---:|
| 00_electricity_41000_CAPACITY_MATCHED | 20 | 1280 | 288.342 | 28.03 |
| 01_electricity_41001_CAPACITY_MATCHED | 13 | 832 | 290.607 | 18.28 |
| 02_traffic_41000_CAPACITY_MATCHED | 20 | 1260 | 288.107 | 27.85 |
| 03_traffic_41001_CAPACITY_MATCHED | 20 | 1260 | 288.107 | 27.58 |

Controller 552.6초, 최소GPU여유 8731MiB, 비승인compute 0표본. 새모델 trainables761,952개가 LH와 정확히 같다. 따라서 LH 대비 학습파라미터 절감은0이다. 총 모델은 고정LoRA를 보존한채 residual adapter를 더하므로 작아졌다고 주장하지 않는다. 역전파 경로와 활성 저장량이 달라 학습 peak는 별도로 측정했다. 선택checkpoint까지 step 시간과 총 연구step시간은 scores.csv에서 구분한다.

고유예측 113개/113개 MSE·MAE·rawMAE 기록을 독립 float64 scalar로 검산했다. 최대MSE차 1.11e-16. 선택checkpoint4개의 새모델 V 재생은exact. 전체frozen/buffer불변·intended parameters 실제변경·초기예측exact·미래poison불변, source/model/staged data/hash 및 기존2101개 결과 파일 보존을 확인했다. [검산](verification.json).

## 해석 한계와 남은 작업

같은 파라미터 수가 같은 표현력이나 최적화 기하를 보장하지 않는다. 폭 변경은 초기 파라미터 수와 학습 동역학도 바꾸므로 차이를 순수한 삽입 위치 인과효과로 단정하지 않는다. 단일lr/폭/20epoch cap의 개발 대조다. cap도달은 수렴 증명이 아니다. 이전 입력 조건화 후보의 추가 가치 없음 판정은 그대로 보존한다.

NOVELTY=KNOWN_CAPACITY_CONTROL, 독립SCREEN_PASS=NOT_EVALUATED, 새방법론주제=NOT_CONFIRMED. 후보 설계와 미노출 평가·다른backbone 일반화는 아직 미실행이다. [연구 검토](RESEARCH_REVIEW.md). 코드·원점수·manifest·검산은GitHub에, raw data/weights/예측배열은로컬cache에보존한다.

학습 peak allocated 절감 범위는 LoRA 대비 72.818~73.070%다. 실제4 fits의 max allocated를 같은 dataset/seed끼리 비교한 값이다. 전력41001의 공통 조기 종료로448updates를 쓰지 않았으며, 미완료fit은 없다. [최종 판단](FINAL_DECISION.md), [원인·주제 검토](INTERPRETATION.md), [정책 재검산](completion_audit.json).
