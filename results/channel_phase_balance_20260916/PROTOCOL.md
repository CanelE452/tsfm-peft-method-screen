# 동일 예산의 시작 위상 균형 LoRA 대조

2026-09-16. 기준1ce4200cabaf6e0dc520d7b3fb97010c94c9b4a1. 이전120-fit 건물 계약과24/4-fit 채널 실행은 종료 상태로 보존한다. 사용자의 계속된 실패 원인 검토·후보 탐색 요청에 따라 확인된 sampling 가설 하나를 별도4-fit 대조로 검사한다. 새 방법론 PASS를 만드는 실행이 아니다.

## 고정 가설과 단일 변경

[확인] R2의 학습 forecast origin은 모두 row mod24=0이고, V는electricity4/traffic16, E는양쪽19였다. [추정] 시작 위상 편중이 주기성 전이와 예측 head 일반화를 어렵게 했을 수 있다.

[설계] 모델은 기존 LH와 완전히 같다. 동일 원자료·32채널·train/V/E70/10/20 경계, train 기준 평균/std, context96/horizon96, MOMENT-small 로컬 revision, LoRA rank8 alpha32 q/k/v, trainable head, BF16·micro/effective batch8을 유지한다. 원점 수electricity512/traffic504도 같다.

변경은 train 원점만이다. 기존 오름차순 origin a_i에 phi_i=i mod24를 더해 o_i=a_i+phi_i로 정한다. o_i+96이 train 끝을 넘으면24를 뺀다. 따라서 기존 날짜 부근(−23~+23시간)을 사용하며24개 위상별 수 차이는 최대1이다. 원시 load나 V/E 점수를 보고 원점을 고르지 않는다. 중복·범위·위상 분포와 미래값 오염 불변을 CPU에서 확인한다. 매 epoch permutation은 기존과 같은 np.random.default_rng(seed) 알고리즘으로 새 목록을 섞는다. 위상별로 전체기간에서 별도 linspace 표본을 뽑아 특정 날짜에 묶는 방식은 사용하지 않는다.

## 예산과 평가

- 신규4 fits = electricity/traffic × seed41000/41001, 상한5080updates. 실제 smoke2updates×2원천=4 별도. 실패도 attempt에 포함하고 재튜닝·교체·반복 없음.
- 기존 AdamW lr0.001, weight_decay0, betas(.9,.999),eps1e-8,clip1. StepLR5epochs gamma0.5, 최대20epochs, V min_delta0.0001/patience5. INIT 포함 최소 V MSE 선택. 기존 LH4fits는 재실행하지 않는다.
- 네 fit의 V 선택 후 봉인하고 기존 E를 평가한다. E는 이미 노출돼 DISCOVERY_REUSED_E이며 독립 확인이 아니다.
- E에서 두 결과를 사전에 정해 모두 공개한다: 새 V 선택 checkpoint, 그리고 기존 LH의 V 선택 epoch와 같은 epoch의 새 checkpoint. 후자는 같은 수의 optimizer updates끼리 비교하며 E에서 좋은 epoch로 바꾸지 않는다. 동일 checkpoint면 예측 캐시를 참조하여 중복 추론을 피한다.
- 비교는 기존 LH 및 seasonal24. 과거 online 일별 잔차 대조는 추가 E 정답을 사용하므로 같은 정보량의 기준선으로 섞지 않는다. 추가 온라인 적응은0이다.
- 원점수 normalized channel macro MSE/MAE, 원단위MAE, seed별/원천별 효과, 선택과 동일updates 결과, trainables·시간·메모리·전체실행량을 보고한다. sampling-only 변경으로 신경망 신규성은 없음. 모든 source×seed 결과를 그대로 기록하고 새1%/유의성 PASS 기준을 추가하지 않는다.

## 검사와 운영

원자료/모델/기존 예측·source 해시를 재검사한다. 각seed 초기 trainable state가 기존 epoch0 checkpoint와 같아야 한다. 실모델 BF16 smoke에서 기존 LH INIT V 예측(같은 batch shape)을 exact 재생하고 intended parameters 실제 업데이트·frozen weight/buffer 불변·finite를 확인한다. 새 선택 checkpoint의 새 모델 재생은 exact, MSE/MAE는 vector/float64 scalar atol1e-12 rtol1e-12를 고정한다. 임계값 변경 금지.

한GPU 한worker, 기존gpu.lock. RustDesk만 예외. 시작30초 안정/free>=4GiB, 실행중1GiB미만 또는 비승인compute면 경계에서 대기. 누적대기600초/wall3600초 상한. 오류와 성능 결과를 분리한다. 보고서만 작성하고 멈추지 않고 사전 검사 통과 시4fits를 실행한다. 이 단일 원인 대조를 끝낸 뒤 자동으로 다른 후보를 학습하지 않는다.

## 비판적 해석

위상 균형 sampling은 알려진 표본 설계이며 그 자체가 새로운 PEFT 방법이 아니다. training windows가 바뀌므로 노출된 개별 값과 순서도 달라지고, sampling intervention의 전체 효과를 측정한다. 순수한 시간 위상 효과만 분리한 수학적 인과 증명으로 쓰지 않는다. 기존 평가 원점은 바꾸지 않아 여전히 E의 한 시작 위상에 대한 개발 확인이다. 성능이 좋으면 교정된 강한 baseline으로 향후 방법을 다시 비교해야 하며, 좋지 않으면 위상 편중만으로 과거 실패를 설명할 근거가 약해진다.
