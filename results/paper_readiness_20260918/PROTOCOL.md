# 논문 준비 보강 실험 — 평가 전 고정

2026-09-18 사용자가 논문에 필요한 실험·시각화 수행을 명시적으로 요청했다. 이전 계약과 결과는 변경하지 않는다. 새 방법을 탐색하거나 C3를 튜닝하지 않는다.

1. 질문 A: 추가 adapter 학습 효과와 추론 gate 직접 효과를 분리할 수 있는가? Electricity source의 동일 LR 0.0003, 동일 초기값·B0·학습 순서·1,024 업데이트 C2/C3 가중치 × 추론 C2(all-one)/C3 gate의 2×2 네 구성을 비교한다. 원래 Electricity, 같은 source 전이16계열, NESO2025 × standard/shape × 기존3seed =72 논리적 view. 정확히 같은 checkpoint/gate/input 예측은 hash 확인 후 재사용한다. selected 주 결과를 fixed1024로 교체하지 않는다. 기존 ETTm1은 C2/C3 LR가 달라 이 통제 실험에 넣지 않으며 기존 음성 결과는 전체 논문 표에 보존한다.
2. 질문 B: NESO에서 기존 C3/RECENCY 설명은 재현되는가? selected C3 가중치/RECENCY gate와 그 역방향, standard/shape×3seed=12 view. 이미 사용한 평가 자료의 후속 기전 진단이며 독립 확증이 아니다. 원래 두 구성은 재사용한다.
3. 질문 C: 출발 모델과 단순 예측보다 이득이 있는가? 미적응 Chronos-Bolt-small F0, 마지막 관측 유지(PERSISTENCE), 계절 반복(SEASONAL) 각5패널×2종류=30 view. F0는 pretrained 가중치 그대로이며 LoRA 학습 없음. 단순 예측은 관측 입력만, 계절 길이는 기존24/96을 사용한다. 동일한 점예측을9개 slot에 저장하되 확률 예측 우위의 근거로 사용하지 않는다. deterministic baseline은 seed0 한 번만 계산하며3seed로 복제하여 표본을 늘리지 않는다. NESO의 B0는 Electricity source 적응 모델이며 NESO 현지 적응 모델이 아니다.
4. 총114 논리적 view 상한(신규 추론+정확 재사용 합), 학습0회/optimizer update0회. 추가 seed/LR/데이터/모델 없음. 모든 예측 저장 및 manifest 봉인 후 채점한다. 이미 알려진 결과의 사후 진단임을 명시한다.
5. 입력512/출력64,FP32,TF32 off,microbatch32,기존 sealed origins·sigma·변형·부호·정답 유지. FAULT는POINT/BURST 여섯 조건 동등 평균, REFERENCE/SHIFT4/SHIFT8/SHIFT_POINT와8형태+pairedSHIFT8을 모두 기록한다. 하위자료를 성능으로 제외하지 않는다.
6. 기전 분해의 주 조건은 standard SHIFT8, REFERENCE와FAULT는 손해 대조다. 대칭2×2 gate/weights/interaction을 절대nMAE로 보고한다. 같은 총효과도 비선형 상호작용을 포함하며 보편적 학습 인과 기전이나 새로운 배포 후보로 주장하지 않는다.
7. 원점별 nMAE,rawMAE,nRMSE,2pinball,crossing을 기록. gain은 평균오차비율. 7 index-day/calendar-day block,2,000 draws,seed89700+1000×기존panel index,패널 내 같은 draws. CIs는 관측된seed/기간에 조건부. 새로운 비교는 탐색적이고 다중 검정 보정 없는 구간을 확증으로 사용하지 않는다.
8. 시각화: 전체5패널의 직접효과와seed, 기전4구성, 모든변화형태, 기존학습곡선, 자원절충, 첫E원점·첫채널·첫draw·첫seed의 예시. 가상운영혼합은SHIFT8비중w∈[0,1], 나머지 중FAULT비중q∈{0,0.1,0.5,1}; 기존점수의선형혼합이며실제발생빈도·최적운영정책추정이아니다. 유리한w를골라주성공으로삼지않는다.
9. GPU 단일worker, RustDesk만예외. startup4GiB안정30초,runtime1GiB,RAM2GiB,disk10GiB,최대3시간/외부사용대기10분. 잘못된구성·GPU안전차단을성능실패와구분한다.
10. 원래forward/동결hash/복원/단순예측수학검사/독립scalar/전체효과재집계 확인. 최종 한국어 근거 보고서, 논문용 표·그림, 주장-근거 대응, 재현 안내, 남은 실제 제출 요건을 작성하고 scoped commit/push. 이 보강 완료를 선행 전체 우위나 논문 채택 보장으로 해석하지 않는다.
