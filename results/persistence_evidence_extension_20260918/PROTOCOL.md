# 기존 C3의 논문 근거 추가 검증 — 2026-09-18

사용자의 이번 추가 검증 요청에 따른 별도 작업이다. 이전 58-fit 계약과 판정은 보존한다. 목표는 양성 결과를 만들어내는 것이 아니라 전력의 좁은 추가 이득의 설명과 외부 적용 범위를 직접 검증하는 것이다.

- 새 fit / backward / optimizer update: **0**. 모든 기존 selected 가중치와 원래 관측 입력을 유지한다. Chronos-Bolt-small FP32, TF32 off, dropout0, microbatch32, context512/horizon64, 기존 sigma/gate/rank 그대로다.
- 원래 Electricity/전이16계열/ETTm1: 기존에 core만 평가한 동일8형태+짝 SHIFT8에 학습된 MEAN/ROTATE16/RECENCY를 적용한다. 3패널×3군×3seed=27개 예측view.
- 같은3패널의 standard와 shape: C3 가중치+RECENCY gate 및 RECENCY 가중치+C3 gate를 **설명용 개입**으로만 적용한다. 3×2종류×2개입×3seed=36view. 원래 가중치는 변경하지 않고, 이 개입을 선택·배포할 새 후보로 삼지 않는다. 이미 학습된 함수에서의 gate/weight 분해이며 학습 과정의 인과분리는 아니다.
- 외부 원천은 NESO 2025 National Demand(ND) 한 계열로 사전 고정한다. TSD/EW 수요 등 유사 계열을 독립 표본으로 더하지 않는다. 공식 공개 CSV의 날짜·settlement period를 Europe/London에서 UTC로 변환하고 두 반시간 MW의 평균으로 시간별 수요를 만든다. 누락·중복·유효하지 않은 DST 일은 실행 전 차단하며 성능에 따라 대체 자료를 찾지 않는다.
- NESO sigma는2025-01-01~06-30 UTC의 population std다. target 학습·V 선택은0회다. E는07-01 이후이며 horizon64가 연말 범위에 온전히 포함되는 원점만 사용한다. 기간 전체에서 서로 다른128일을 먼저 균등 선택하고 phase를 분산한다(seed88301). 기존 양성 전력의 selected7군×3seed를 standard/shape로 그대로 이전:42view. SHRINK/BIAS는 원래 전력 V의 alpha/beta 그대로 적용:6view. NESO 결과가 나빠도 source나 기간을 교체하지 않는다.
- 총 새 예측view 상한 **111**(105개 모델 view+6출력대조). ETTm2는 새 기전 대조 가중치가 없으므로 기존 결과만 참조한다. C1 step0으로 C0와 같은 가중치인 NESO view는 실제 parity 확인 후 예측을 재사용한다.
- 표본·입력 변형·정답은 군별로 동일하다. 측정 오류는 입력만, 지속 변화는 과거·미래를 함께 바꾸는 원래 generator를 재사용한다. state/clean input/delta/future labels는 모델과 gate에 제공하지 않는다.
- 새 결과 채점 전 모든 예측을 저장하고 manifest를 고정한다. 기존 결과는 이미 본 개발 결과이며 숨기지 않는다. NESO는 다른 제공자와 국가 집계 수준의 후속 외부 평가이지, 개인계량기 전이나 실제 사건 검증과 동의어가 아니다. 수정된 역사적 outturn이며 당시 공개된 vintage는 아니다.
- 주 질문은 NESO standard SHIFT8의 C3/C0,C3/C2,C3/RECENCY 세 비교를 함께 보는 것이다. FAULT,REFERENCE,SHIFT4,SHIFT_POINT와8형태 손해를 모두 공개한다. 모든 주 비교가 필요하며 유리한 하나의 CI로 PASS를 선언하지 않는다.
- 원래 nMAE와 rawMAE/channelNRMSE/2pinball/crossing, 모든 seed를 유지한다. 평균오차 비율로 gain을 계산한다. 2,000회7일 paired block bootstrap, seed88700+1000×[electricity,transfer,ettm1,ettm2,NESO]의 index. 패널 내 모든 조건/seed/방법/채널에 같은 draw. 전력16계열에는 series+time 구간도 보고한다. 기존 기전3대비에는Bonferroni3구간을 병기한다. 형태·개입은 탐색적이며 여러 조건의 양성을 확증으로 선별하지 않는다.
- C3/RECENCY의 오류 차이는 동일 학습 가중치/교체gate의2×2 평균 차이로 분해한다. 총차이=gate평균효과+weight평균효과인지 수치 검산한다. 이 결과로 새로운 gate를 튜닝하지 않는다.
- 같은 GPU guard/oneworker/RustDesk 예외, startup4GiB 안정30초, runtime1GiB/RAM2GiB/disk10GiB, 최대3시간 및 외부사용 대기10분. 자원 부족과 성능 손해는 구분한다.
- 정상 구성된 비교는 모두 실행·검산한다. 전후 state hash, 원래 selected forward exact parity, 복원, 독립 scalar nMAE/2pinball과 전체 효과 재계산을 확인한다. GPU normalizedmax1e-5 또는rtol1e-4, CPU rtol1e-10/atol1e-12를 유지한다.
- 최종 한국어 REPORT, PAPER_REVISION, FINAL_DECISION, 비용·실행·선행 한계를 작성하고 scoped commit/push한다. 새 구조·학습·후속 원천 자동 시작은 없다.
