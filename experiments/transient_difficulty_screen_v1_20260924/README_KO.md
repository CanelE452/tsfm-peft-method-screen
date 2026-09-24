# Transition-memory difficulty screen v1

이번 단계는 새 PEFT 학습이 아니다. 좁은 히트펌프·바닥난방 환경에서 **현재 상태와 target history를 알고도 과거 command history가 미래 예측에 추가 정보를 주는가**를 먼저 확인한다.

고정 흐름:
1. BOPTEST `bestest_hydronic_heat_pump`에서 8개 episode를 미리 정한 날짜·명령 일정으로 수집한다.
2. 성능 확인 전에 TRAIN 5 / DEV 2 / RESERVE 1로 고정한다. RESERVE는 수집·hash만 하고 이번 실행에서 읽거나 채점하지 않는다.
3. 다섯 target/horizon 후보를 TRAIN episode-group OOF로 screen한다.
4. CURRENT predictor는 target의 과거 궤적 요약 + **현재 command**를 사용한다.
5. HISTORY predictor는 같은 정보에 **과거 command history descriptor**를 추가한다.
6. `History Gain = (MAE_CURRENT-MAE_HISTORY)/MAE_CURRENT`가 TRAIN 5개 중 4개 이상 양수이고 평균 2% 이상인 후보만 DEV로 간다. 2%는 논문 기준이 아니라 후속 투자용 screen threshold다.
7. TRAIN에서 학습한 CURRENT/HISTORY 두 예측의 차이로 `TMD = mean(|pred_history-pred_current|)/train_target_std`를 계산한다. TMD는 DEV future label 없이 계산 가능하다.
8. DEV 두 episode 모두 History Gain > 0일 때만 zero-shot Chronos-2 F0를 같은 정보 권한(target history + command history + 이미 알려진 future command plan)으로 평가한다.
9. F0 error와 TMD의 관계를 보고하되, 이번 단계에서 LoRA/제안 PEFT는 학습하지 않는다.

주의: TMD를 보편적인 시계열 complexity라고 선언하지 않는다. 이것은 이 좁은 도메인에서 history dependence를 측정하기 위한 **difficulty proxy 후보**다.
