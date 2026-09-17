# 최종 결정

실행 COMPLETE: 48 fits / 49,152 main +24 smoke updates. 논문 PASS가 아니다.

**고정 B5 후보는 종료한다. 후속으로 자동 투자할 새 방법 후보는0개다.** 실행은 성공적으로 완료했지만 PERSISTENCE_METHOD_SIGNAL 요건을 충족하지 못했다. 임의의1% 문턱 때문이 아니라, raw augmentation B0 및 단순 rule B3 대비 큰 persistent-shift 손해가 두 seed에서 남았기 때문이다.

판정 태그는 두 원천 모두 TRAINING_EXPOSURE_EXPLAINS_PART(서술적·부분 설명), HARD_CLIP_MISMATCH, NO_ADDED_METHOD_EVIDENCE다. Electricity에서는 RAW_AUGMENTATION_SUFFICIENT도 적용한다. ETTm1에서는 B3의 SHIFT_POINT 개선을 그대로 인정하며 B0가 모든 패널에서 우월하다고 쓰지 않는다. PERSISTENCE_PREPROCESSING_SUFFICIENT를 두 원천 전체의 비열등성 선언으로 사용하지 않는다.

노출 부족 진단, matched B2의 SHIFT8 손해 축소, B3의 일부 효과, B5/B4의 작은 양수는 남길 근거다. 그 근거가 현재 learnable PEFT 후보의 추가 필요성이나 신규성을 확보한 것은 아니다. 이 고정 비교의 결론을 모든 robust PEFT나 실제 오류·실제 사건으로 일반화하지 않는다.

1. **기존 실패의 학습 노출 mismatch: 부분 설명을 지지한다.** 역사적 참고로 기존 A5의 SHIFT8 nMAE는 Electricity0.682569, ETTm1 1.620576이었고, 새 matched B2는0.261049/0.588650이다. 이 값들은 서로 다른 프로토콜의 기록이므로 하나의 평균으로 합치거나 노출만의 개선율로 해석하지 않는다. 새 실험 안에서 B2는 B0보다 SHIFT8 오차가24.29%/7.94%, SHIFT4가18.33%/11.32% 높다. 노출을 맞춘 것만으로 residual 경로의 필요성이 확보되지는 않았다. 특히 B2의 ETTm1 FAULT/REFERENCE 오차도 B0보다4.98%/5.85% 높았다.

2. **Hard clipping의 손해가 남는다.** 같은 augmentation에서 B1은 B0보다 SHIFT4 오차가34.97%/48.40%, SHIFT8이61.86%/29.86% 높다(Electricity/ETTm1 순). 두 seed 모두 악화이고 각 time-block CI도 같은 방향이다. 이 고정 clip6 경로의 손해를 지지하지만, 모든 clipping 설정이나 robust PEFT를 반증하지 않는다. 두 군의 선택된 LR/checkpoint는 공통 규칙에 따른 것이므로 최적화·선택 절차까지 포함한 직접 비교다.

3. **단순 persistence rule의 부분 효과는 있다.** B3는 clip 계열의 큰 SHIFT 손해를 대부분 피한다. ETTm1 SHIFT_POINT는 B0 대비+2.3441% [95% block CI 1.2847,3.5076], seed별+1.5295/+3.1379%다. ETTm1 FAULT 평균도+0.7434% [0.2527,1.2181]지만 seed별−0.3353/+1.8009%로 엇갈린다. 두 seed를 평균한 시간 bootstrap이 optimizer 안정성을 보장하지 않는 실제 사례다. ETTm1 SHIFT8은−0.8704% [−1.7462,0.0373]이며 CI0 포함을 동등성 증명으로 부르지 않는다. Electricity에서는 B3의 FAULT/REFERENCE가 B0보다0.5974%/0.7615% 나쁘고 각각 CI도 악화 방향이다. 따라서 B3를 두 원천의 보편적 개선 또는 비열등 방법이라고 선언하지 않는다. B0를 기본으로 두고 B3는 단순 대조로 보존한다.

4. **Learnable persistence는 제한된 반응을 보이지만 새 방법의 필요성은 지지되지 않는다.** B5는 B4보다 SHIFT4를0.1655%/0.1823%, SHIFT8을0.4102%/0.2442% 개선했다. 이 작은 양수를 무시하지 않는다. 해당 time-block CI는0보다 높고 두 seed 방향도 일치한다. 다만 extra scalar1개 차이와 B4 intercept 중복을 포함한 비교다. V에서 p를0으로 만들면 SHIFT8 오차가0.6258%/0.2258%, 고정 permutation이면0.5945%/0.2204% 증가해 persistence 의존성은 관찰된다.

그러나 핵심 B5/B0에서는 SHIFT4 오차가32.95%/46.58%, SHIFT8이60.18%/27.24%, SHIFT_POINT가26.50%/28.93% 높다. B5/B3도 SHIFT4/SHIFT8/SHIFT_POINT에서 큰 손해가 남는다. B5가 B4보다 조금 좋은 것이 raw augmentation이나 fixed rule보다 필요한 방법이라는 결론을 만들지는 못한다. B5/B2에서는 ETTm1 FAULT/REFERENCE 이득이4.47%/3.41%지만 SHIFT4/SHIFT8 손해가31.68%/17.89%로 남아 단일 우승자로 만들지 않았다.

**Gate 동작의 한계:** B5는 학습됐다(유한하고 비영인 gradient, scalar 변경, V ablation 반응). 하지만 E SHIFT8 clipped slots의 평균 복원율은 Electricity2.7451%, ETTm1 2.2538%로, B3의84.9698%/89.6983%보다 작고 hard clip에 가까웠다. 초기 intercept−4, 공통 작은 LR와 고정1024updates에서 이 동작을 관찰했다. 초기화·최적화 조건과 방법 개념 자체의 한계를 분리해야 한다. 더 큰 LR나 다른 초기값이 해결한다고 입증한 것은 아니며 재튜닝하지 않는다.

**자원 절충:** 반복seed 평균에서 B5는 B0보다 optimizer compute가약5.56%/5.76%, E추론 시간이약7.57%/7.11% 늘었고, peak allocated는두 원천 모두약11.51MiB 늘었다. 측정은 이 GPU와 두 반복 경로의 기술통계다. 학습 파라미터4개 추가가 이 구현의 자원 이득을 뜻하지 않는다.

남길 구현: 정보 권한을 분리한 입력 변형, 같은 x/y를 사용하는 직접 대조, clip 노출 진단, 고정 예산 LoRA runner, journal/resume·봉인·전체 예측 선저장·독립 검산.

남길 비교 구현: **B0 raw augmentation LoRA**를 기본 기준으로, **B3 fixed persistence**를 source/condition별 단순 대조로 보존한다. B2/B4/B5의 코드·결과도 재현용으로 보존하지만, 현재 B5와 기존 residual 후보를 살리기 위한 추가 튜닝은 중단한다. 후속 연구 투자 후보는0개이며 새 후보로 자동 전환하지 않는다.

실제 오류/사건 레이블, 가까운 강건 선행 전체 비교, 독립 source/backbone 검증이 남아 있다. 새 seed/LR/rank/threshold/window/v3 또는 후속 학습은 시작하지 않는다.
