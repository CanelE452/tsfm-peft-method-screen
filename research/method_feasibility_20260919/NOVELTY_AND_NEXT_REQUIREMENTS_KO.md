# 방법론의 기여 범위와 가까운 선행

2026-09-19. 현재 새 방법의 추가 가치와 학술적 신규성은 미확보다. 이 문서는 새 후보의 성공을 주장하거나 기존 분석 원고를 방법론 논문으로 이름만 바꾸지 않는다.

## 실제 확인한 선행

1. **Time-PEFT**: 공식 구현 commit `ea4e7e1887bb35587bab7ea93e2af3685ac55852`의 FrequencyAdapter/ChannelAdapter/forward를 다시 읽었다. patch 축 FFT top-k 필터와 projection, 공유 down 및 채널별 up projection이 포함된다. 본 검사는 이 방법의 재현이 아니며, 그 논문의 방법론 기여를 따르려면 명확한 구조적 문제와 직접 비교가 필요하다는 참고다. [공식 코드](https://github.com/kaist-dmlab/TimePEFT/blob/ea4e7e1887bb35587bab7ea93e2af3685ac55852/run.py), [이전 코드 해시](../../results/temporal_response_peft_20260919/PRIOR_CODE_AUDIT.json).
2. **LoRA-GA**: 논문 §3과 공개 구현의 `estimate_gradient`/`lora_ga_init`을 확인했다. batch gradient를 평균하고 SVD의 서로 다른 방향에서 A/B를 초기화한다. scale 선택과 기존 weight에서 초기 BA를 빼는 보상도 수행한다. 단순 평균-gradient 부분공간과의 비교를 공식 LoRA-GA 전체 재현이라고 부를 수 없다. [논문](https://arxiv.org/html/2407.05000v1), [고정 코드의 초기화](https://github.com/Outsider565/LoRA-GA/blob/c4cd5372c75b290924214b348008891f744512ef/peft/src/peft/tuners/lora/layer.py), [gradient 수집](https://github.com/Outsider565/LoRA-GA/blob/c4cd5372c75b290924214b348008891f744512ef/peft/src/peft/utils/lora_ga_utils/lora_ga_utils.py). 두 파일의 원본·SHA는 [receipt](PRIOR_CODE_RECEIPTS.json)에 기록했다. 공식 코드를 실행한 학습 재현은 아니다.
3. **LoRA-One**: 공식 README가 한 번의 full gradient에 기반한 초기화와 conditioning을 다룬다. gradient에 맞춘 초기 방향이라는 넓은 주장만으로 차별화할 수 없다. 이번 검토는 README 및 공개 설명 범위이며 구현 전체 재현은 하지 않았다. [공식 저장소](https://github.com/kyoonji/lora_initialization_hessian).
4. **FILet**: 원문 §2–3은 Fisher/데이터 민감도에 의한 LoRA 초기 방향 선택을 다룬다. 데이터 의존 부분공간 선택 자체를 새 원리라고 주장할 수 없다. [원문](https://arxiv.org/html/2605.01046v1). 본 검토에서는 preprint로 취급하며 정식 게재 상태나 전체 수치 재현을 확인한 것은 아니다.
5. **FCCA**: signed input–error cross-covariance와 Fisher whitening을 사용하는 근접 preprint를 확인했다. 이것은 이번 서로 다른 날짜 블록 gradient 사이의 cross-moment와 정의가 다르지만, task signal을 이용한 작은 학습 공간이라는 넓은 영역이 이미 연구 중임을 보여준다. [원문](https://arxiv.org/html/2609.00762v1). 구현 재현과 정식 성능 비교는 미실행이다.

이 검색으로 모든 선행을 다 찾았다고 주장하지 않는다. 수식 차이의 존재, 코드 구현 완료, 실험 양성, 충분한 신규성은 각각 별개의 요건이다.

## 이번 프로토타입의 정확한 차이

TRAIN 날짜 블록 k의 dense gradient를 G_k라고 하면 M=mean(G_k), Q=mean(G_k^T G_k)다. CROSS_BLOCK은 C=(K M^T M−Q)/(K−1)의 상위 rank8 고유벡터를 한쪽 LoRA 초기 부분공간으로 쓴다는 가설이다. 이는 서로 다른 블록 곱의 평균과 같고, 같은 평균·독립 noise 가정을 두면 자기 noise 항을 제거하는 알려진 off-diagonal moment 추정이다. 시간 블록은 실제로 독립·동일분포라고 보장할 수 없으며, 변화가 큰 블록의 신호를 버리는 부작용도 가능하다.

MEAN_SVD와 동일한 calibration labels/gradient/랭크를 쓰고, 나중 TRAIN의 모든36개층 gradient에 정렬되는지를 본다. 이 프로토타입은 LoRA-GA의 양방향 초기화·scale·offset과 다르며, 실제 A/B를 학습한 최종 방법이 아니다. 일반 U-statistic이나 gradient-aware PEFT를 새로 발명했다는 주장은 불가하다.

이번 실모델 검사에서는 CROSS의 후반 TRAIN cosine이 Electricity0.104025 대 MEAN0.115521, ETTm1 0.082757 대0.093216으로 둘 다 낮았다. 동일 크기의 단순 대조보다 좋아진 근거가 없어 이 고정 가설의 본학습을 추천하지 않는다. 이는 gradient 진단이며 새로운 방법의 실제 forecasting 성능 실패를 관측한 것과 구분한다.

## 방법론 논문을 위해 아직 필요한 것

- 기존에 잘 학습되는 LoRA를 기준으로, 어느 계산·표현 제약을 새 연산으로 해결하는지 명확히 정의해야 한다. ‘seed에 따라 달라진다’, ‘작은 모듈을 붙인다’만으로는 연구 기여가 아니다.
- 직접 가까운 선행과 가장 싼 대안에 대해 같은 입력 권한·학습 표본·선택 기회로 비교해야 한다. 보조 gradient 지표를 최종 forecasting 결과의 대체물로 쓰지 않는다.
- 모든 데이터셋·상태에서 무조건 이겨야 한다는 요구는 하지 않는다. 주장의 범위를 사전에 정하고 그 범위에서 구성요소의 추가 가치, 손해 조건, 비용을 함께 검증해야 한다.
- 일반 예측으로 범위를 넓히더라도 과거 개발 E가 독립 test가 되지 않는다. 기존 결과를 본 뒤 설계한 방법에는 미사용 기간 또는 외부 source의 확인이 필요하다.
- 새 방법의 수식·학습 절차·구현을 갖추고, 단순 대조로 설명되지 않는 재현 가능한 이득을 보여야 한다. 현재 이 요건을 충족한 후보는 없다. 이 사실을 숨기고 원고의 명칭만 바꾸지 않는다.

이번 실행 단위는 새로운 학습 없이 과제의 관측 한계와 초기화 가설을 실제 데이터에서 검사했다. 전체 방법론 목표의 완료가 아니며, 새 주제로 자동 전환하는 실행기는 연결하지 않았다.
