# 예보 경로 연결을 보존하는 LoRA 학습의 직접 비교

질문은 동일 시점별 예보값을 시간적으로 연결해서 보여주는 학습이 그 값들을 시간별 rank로 섞어 보여주는 학습보다 유용한가다. PATH는 알려진 증강, POINT는 합성 통제이며 새 PEFT 구조를 주장하지 않는다.

고정 Chronos-2 FP32 rank1 LoRA에 LATEST/DROP/PATH/POINT를 적용한다. 동일 타깃64 TRAIN origin, 512 updates, 두 LR(1e-4/3e-5), 두 seed61730/61731이며 최대3타깃·48 fits다. PATH/POINT는 4epoch 내 origin/hour/feature multiset을 exact 일치시킨다. 세 변수는 한 시간 안에서 같은 rank다. 초기화·원점 순서·정답 노출·선택 기회도 일치한다.

TRAIN만으로 선정·정규화하고 V_SELECT로 checkpoint/LR, 별도 V_CALIBRATE로21상수 보정을 결정한다. 모두 봉인한 뒤 TEST 예측과 해시를 만들고 정답을 채점한다. raw primary와 동일 보정 효과, FIXED512, 네 실제 rank case, 세 타깃·두 seed·월별 결과를 모두 남긴다. 테스트값이나 중간 성능에 맞춰 조건을 변경하지 않는다.

정보 계약은 SIMULATED_ASOF이다. 전력망 타깃 세 개가 독립 도메인 세 개는 아니며 사전학습 corpus 중복 부재도 입증하지 못했다. FROZEN_WEATHER/HISTORY 및 LAST_DAY는 추가 neural fit0 대조다. controller4시간, main24576+smoke24+폐기resource최대24=총24624updates 이내이며 한 GPU에 한 worker다. 기존 승인 RustDesk 예외만 유지한다.

최종 추천은 계약상 하나만 남기고 새 구조·다른 데이터셋·후속 학습은 시작하지 않는다. 실행 중단은 성능 실패와 구별한다. 구체적 모든 기준은 PROTOCOL.md 하나에 따른다.
