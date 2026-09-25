# 실제 TSFM Go/No-Go v1

이 패키지는 메타데이터 사전점검이 아니다. **실제로 Weather 데이터를 받고 Chronos-2를 로드해 후보 A/B를 학습하고 후보 C를 실제 예측으로 평가한다.**

이전 P0 지시문과 달리 다음을 명시적으로 허용한다.

- 공식 공개 Weather 데이터 다운로드
- `amazon/chronos-2`가 캐시에 없으면 Hugging Face에서 모델 다운로드
- 실제 gradient/backward/optimizer step
- LoRA 후보 A의 실제 학습
- 채널 압축/잔차 후보 B의 실제 학습
- A의 I8 예측을 이용한 C의 CAL 결합 가중치 선택
- 결과 commit/push

설치된 저장소 `.venv`의 `chronos-forecasting==2.3.2`와 PyTorch를 사용한다. 새 pip 설치는 하지 않는다. 의존성이 없으면 기술적 실패로 기록하되, **'데이터/모델이 로컬에 없다'는 이유로 사전점검에서 멈추지 않는다.**


## 연구 범위 주의

이 실행은 MSFT/AdaPTS 공식 수치 재현이 아니다. 두 논문에서 가져온 **핵심 메커니즘 후보를 동일한 Chronos-2/Weather 환경에서 빠르게 반박하거나 유지하기 위한 mechanism screen**이다. 따라서 결과는 원 논문의 성능과 직접 비교하지 않고, 후보를 계속 투자할지 판단하는 데만 사용한다.

## 후보 A — multi-scale LoRA sharing

동일한 물리 시간 구간을 scale 1/2/4로 평균 다운샘플하고 같은 Chronos-2에서 예측한다. 모든 Chronos-2 time/group attention의 q/k/v/o에 실제 LoRA를 삽입한다.

- I8: scale별 독립 rank-8 LoRA
- S8: 모든 scale이 완전히 같은 rank-8 LoRA
- I2: scale별 독립 rank-2 LoRA
- P8: 공유 rank-8 A/B + scale별 rank gate

각 step에서 세 scale 모두 실제 forward/backward를 수행한다. scale별 loss를 이용한 학습 가중치도 실제로 업데이트한다.

## 후보 C — train weighting vs prediction weighting

A의 I8을 한 번 학습한 뒤 같은 checkpoint의 scale별 CAL/PILOT 예측을 저장한다.

- C_NATIVE: 학습 중 scale loss weighting에서 얻은 scale weight
- C_UNIFORM: 동일 가중
- C_SINGLE: CAL에서 가장 좋은 단일 scale
- C_STACK: CAL에서 simplex grid로 선택한 convex weight

PILOT 결과만으로 판단한다. 추가 신경망 학습은 없다.

## 후보 B — compressed channel adaptation

Weather의 모든 수치 채널을 사용한다.

- U4: 4 latent channels + frozen Chronos + decoder
- U8: 8 latent channels
- R4: U4 + compression residual에만 적용되는 작은 shared temporal head
- X4: U4 + raw history에 적용되는 같은 크기의 temporal head
- PCA4_R: PCA encoder/decoder 고정 + 같은 residual head
- LINEAR: 작은 shared temporal head만

Encoder/decoder는 PCA 초기화에서 시작하지만 U4/U8/R4/X4에서는 실제 optimizer로 학습한다. R4의 목적은 **U8처럼 latent channels를 늘리는 것보다 싸게 정확도를 회복하는가**를 보는 것이다.

## split

Weather 전체의 앞 70%는 TRAIN. 다음 10%를 시간순으로 절반 나눠 CAL / PILOT. 뒤 20% TEST는 코드에서 모델 선택·점수 계산에 사용하지 않는다.

이 패키지는 한 seed의 screening이므로 GO는 `GO_TO_CONFIRMATION`이지 논문 결론이 아니다.
