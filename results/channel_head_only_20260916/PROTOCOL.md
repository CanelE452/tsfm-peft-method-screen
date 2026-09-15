# 균형 표본에서 LoRA 추가 가치: head-only 4-fit 대조

2026-09-16. 기준56d17ec61e040f8696a4f50ddbdba8336658c5fa. 완료된 건물120-fit 계약과 채널 실험은 재개하지 않는다. 사용자의 실패 후 근거 있는 PEFT 후보 탐색 목표에 따라, 최근 큰 성능 회복을 LoRA 효과로 오해하지 않도록 단일 구성요소를 분리한다. 새 방법이나 PASS를 미리 가정하지 않는다.

## 질문과 유일한 변경

동일하게 균형 잡힌 학습 창에서 encoder LoRA를 학습하는 것이 예측 head만 학습하는 것보다 유용한가? 기존 BALANCED_LH는761952개, 이번 HEAD_ONLY는589920개 head 파라미터만 학습한다. 모델 생성·head 초기화·전체 forward는 같은 LH factory를 사용하고 encoder 전체 requires_grad만False로 바꾼다. 초기 LoRA B=0을 고정하므로 encoder 함수는 미적응 함수다. 사용하지 않는 LoRA 모듈은 보존돼 그 forward 연산/상주 메모리 비용은 남는다. 최소 구현의 linear probing 자원 최적화 결과라고 주장하지 않는다. feature cache도 쓰지 않는다. train/eval 모드와 기존 dropout 정책을 유지한다.

MOMENT-small revision·32채널·context96/horizon96·원자료·train70%/V10%/E20%·train 정규화·sampling·seed·epoch별 순열·학습률 모두 직전 봉인과 동일하다. train 시작점은 직전 균형 표본을 그대로 복사하고 새로 선택하지 않는다. 두 데이터electricity/traffic, seed41000/41001. AdamW lr.001 betas(.9,.999) eps1e-8 weight_decay0 clip1, StepLR5 gamma.5, BF16 micro/effective8, cap20epochs, V patience5 min_delta1e-4, INIT 포함 최소 V MSE 선택.

## 유한 예산과 사전 평가

본학습 최대4 fits/5080updates, smoke2updates×2원천=4 별도. 실패도 attempt에 포함, 재튜닝/seed교체/재시도 없음. 기존 balanced LH4fits는 재학습하지 않는다. 동일 GPU guard와 lock: RustDesk만 예외, 시작30초 안정/free>=4GiB, 실행중free<1GiB 또는 비승인compute면 경계 대기, 누적600초/wall3600초. 공통 안전 문제면 실행 중단과 원인을 기록한다.

모든 V 선택을 봉인한 뒤 이미 노출된 E(DISCOVERY_REUSED_E)를 평가한다. 주 비교는 각 방법의 V 선택 checkpoint. 추가 비교는 직전 sampling 계약에서 사전 고정했던 기존 LH 선택 epoch(electricity41000=1,41001=3,traffic둘=1)를 양쪽에 적용한 동일 updates 비교다. 이 checkpoint가 실제 trajectory에 없으면 미실행으로 보고하며 학습을 연장하지 않는다. E를 보고 epoch를 고르지 않는다. 두 recipe와 모든 seed를 공개하고 head-only가 나빠도 숨기지 않는다. normalized channel macro MSE/MAE·원단위MAE, source별 macro, 파라미터·시간·메모리를 구분한다. 새 PASS 허용오차나 성능 문턱은 만들지 않는다.

## 검사

모델·소스·데이터·이전 결과 해시 보존. 네 seed/source의 head 초기 tensor는 원래 LH epoch0와 exact. LoRA B=0과 encoder 전체 동결 확인. 실제 BF16 같은 batch INIT V 예측을 이전 LH cache와 exact 비교. smoke head 실제 변경, 전체 frozen parameters/buffers 불변, finite loss/gradient/output 확인. 전체 train 창 미래값 poison 불변. 선택 checkpoint4개 새 모델 V 재생 exact, 모든 저장 예측 MSE/MAE float64 scalar/vector atol/rtol1e-12. 원점·target/std와 일정 일치 독립 검사. 학습 횟수와 연속 update 원장을 교차 검산한다.

## 해석 한계와 다음 판단

head-only는 알려진 linear probing 대조이며 신규성은DIRECT_EQUIVALENCE다. LoRA가 더 좋아도 표준 LoRA의 구성요소 가치이지 새 방법론 성공이 아니다. head-only가 충분하면 단순 head/기존 표현을 넘는 근거 없는 새 encoder adapter 주장은 보류한다. 실제 차이가 있으면 그 차이를 보존하는 더 효율적인 적응이라는 후속 질문의 근거가 된다. seed2·원천2·한 E 시작 위상·노출된 E의 개발 비교이며 독립 검증 또는 일반화 보장은 없다. 기존 건물/Query 실패의 원인으로 확대하지 않는다. 이 runner는4fits와 평가 후 종료하며 후보 생성 분기가 없다.
