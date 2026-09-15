# 균형 표본에서 head-only와 LoRA 추가 가치

**head-only 4회 학습과 고정 평가를 완료했다. 이것은 구성요소 대조이며 새 PEFT 방법론 PASS가 아니다.**

신규4/4fits·5080/5080updates, smoke4updates. 기존 balanced LoRA+head4fits는 재학습하지 않았다. 미완료fit0. 추가 후보·seed·재튜닝·재시도0.

## 문제와 비교

직전 sampling 변경으로 회복된 성능에 encoder LoRA가 추가로 기여하는지 확인했다. 모델 forward와 초기 head는 같고, 이번에는 encoder를 동결해 head589920개만 학습했다. LoRA+head는761952개다. 두 방법 모두 균형 train 창, 동일 순열·학습률·V/E·정규화를 쓴다. INIT을 포함한 V 최저오차 checkpoint를 각각 선택한다. 별도로 사전 고정 epoch(전력seed41000=1/41001=3, 교통둘=1)에서 동일 updates 비교를 한다.

## 원점수

양수인 LoRA 추가 이득은 LoRA+head가 head-only보다 낮은 MSE임을 뜻한다. 백분율은100×(head-only−LoRA+head)/head-only다.

| 원천 | seed | recipe | head epoch | LoRA epoch | head-only MSE | LoRA+head MSE | LoRA 추가 이득 |
|---|---:|---|---:|---:|---:|---:|---:|
| electricity | 41000 | selected | 16 | 16 | 0.298453 | 0.278283 | +6.758% |
| electricity | 41000 | matched_old_epoch | 1 | 1 | 0.351940 | 0.321743 | +8.580% |
| electricity | 41001 | selected | 16 | 8 | 0.296008 | 0.281737 | +4.821% |
| electricity | 41001 | matched_old_epoch | 3 | 3 | 0.338359 | 0.302970 | +10.459% |
| traffic | 41000 | selected | 18 | 17 | 0.433989 | 0.385908 | +11.079% |
| traffic | 41000 | matched_old_epoch | 1 | 1 | 0.561704 | 0.509221 | +9.344% |
| traffic | 41001 | selected | 17 | 18 | 0.439764 | 0.387187 | +11.956% |
| traffic | 41001 | matched_old_epoch | 1 | 1 | 0.546267 | 0.504907 | +7.571% |

| 원천 | recipe | head 평균 MSE | LoRA+head 평균 MSE | LoRA 추가 이득 | 두seed 모두 LoRA 우세 |
|---|---|---:|---:|---:|---|
| electricity | selected | 0.297230 | 0.280010 | +5.794% | True |
| electricity | matched_old_epoch | 0.345150 | 0.312357 | +9.501% | True |
| traffic | selected | 0.436876 | 0.386548 | +11.520% | True |
| traffic | matched_old_epoch | 0.553986 | 0.507064 | +8.470% | True |

MAE·원단위MAE·선택epoch까지 optimizer step 시간과 같은 epoch 비교의 updates는 [comparisons.csv](comparisons.csv)에 모두 있다. 두 원천의 원단위 지표를 섞지 않는다. E는 이미 사용한 개발 구간으로 독립 시험이 아니다. E를 보고 새 checkpoint를 고르거나 기준을 바꾸지 않았다.

사후 진단 [채널별 오차](channel_comparisons.csv)와 [잔차 분해](residual_components.csv)는 모든 채널을 보존한다. 잔차의 평균·24시간 반복 성분·나머지는 정답을 사용하는 설명용 분해이며, 배포 보정이나 새 채널 선택 규칙이 아니다. RMSE를 가산 분해한 것이 아니라 MSE의 직교 분해다.

## 자원과 실제 실행

| fit | epochs | updates | peak MiB | 학습 step 합계 초 |
|---|---:|---:|---:|---:|
| 00_electricity_41000_HEAD_ONLY | 20 | 1280 | 286.373 | 27.53 |
| 01_electricity_41001_HEAD_ONLY | 20 | 1280 | 286.139 | 27.81 |
| 02_traffic_41000_HEAD_ONLY | 20 | 1260 | 286.139 | 27.16 |
| 03_traffic_41001_HEAD_ONLY | 20 | 1260 | 286.139 | 27.32 |

전체 controller 607.0초, 최소 GPU 여유 8398MiB. 비승인compute 0개; RustDesk만 허용했다. head-only는 학습 파라미터172032개(22.578%)를 줄였다. encoder backward도 없어지지만 고정된0-update LoRA forward 모듈을 남겼으므로 최소 LP 구현의 자원 최적화는 아니다. 기존 실행과의 시간 차이를 모든 환경의 속도 개선으로 일반화하지 않는다. 총 연구비용과 선택 checkpoint까지 적용비용을 분리한다.

## 검산과 연구 판단

고유 예측 104개·104개 MSE/MAE/원단위MAE 기록을 독립 scalar 계산으로 검산했다. 최대 MSE차 1.11e-16. 선택 checkpoint4개의 새 모델 V 재생 차이0, 초기 head/실모델 초기 예측 일치, 학습 대상 변경·전체동결가중치/버퍼 보존·미래 poison 불변을 확인했다. 이전 결과·연구 2009개 파일을 보존했다. [검산](verification.json).

EXECUTION=COMPLETE, EVIDENCE=EXPOSED_DEVELOPMENT_COMPONENT_ABLATION, NOVELTY=DIRECT_EQUIVALENCE_LINEAR_PROBING. 이 비교는 알려진 LP와 LoRA의 가치 분해다. 어느 쪽이 좋아도 신규 알고리즘 성공으로 바꾸지 않는다. 두 원천·두seed·한 E 시작 위상에 한정되고 건물/Query 실패 원인으로 확대할 수 없다. 새 PEFT 주제 확보에는 강한 단순 대조를 넘는 구체적 변경, 가까운 선행과의 차이 및 노출되지 않은 평가가 여전히 필요하다.

[프로토콜](PROTOCOL.md), [사전 연구 검토](RESEARCH_REVIEW.md), [판단](decision.json). 원자료·weights·예측배열은 로컬cache에 보존하고 GitHub에는 코드·원점수·manifest·검산을 공개한다.

후속 연구 질문과 판단 변화는 [INTERPRETATION.md](INTERPRETATION.md), 신규성 경계는 [FOLLOWUP_BOUNDARY.md](FOLLOWUP_BOUNDARY.md)에 분리했다.
