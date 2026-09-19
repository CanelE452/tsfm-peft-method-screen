# LoRA 선행 여부 직접 대조 — 실행 전 고정 사항

단일 실행 계약은 [CONTRACT.txt](CONTRACT.txt)이며 2026-09-19 사용자 메시지가 이 실험에 한해 학습 금지를 해제했다. 다른 실험·joint training·자동 successor는 승인되지 않는다.

F0는 pinned Chronos-Bolt-small 원본이다. F0_PLAIN/F0_MAG는 LoRA 부착 없이 모든 foundation parameter/buffer를 동결하고 기존 residual8712개만 학습한다. PLAIN은 gate1, MAG는 기존 median/MAD·floor0.1sigma·threshold3·patch16 수식 그대로다. 기존 MAG_ONLY는 B0_MAG로 표기하며 no-LoRA가 아니다. down seed+200000, up0, 동일 seed 두 군 초기값 동일.

Electricity/ETTm1의 기존32epoch TRAIN arrays·labels·sigma, V arrays, E origins·변형을 hash 재사용한다. 증강·표본 새 생성 없음. selection81550×LR1e-4/3e-4 후 repeats81551/81552; source/arm당4fits, 총16fits/16384updates+smoke8. batch32 FP32 TF32off dropout0, AdamW(.9,.999)/eps1e-8/wd0, gradclip1, scheduler없음. 순서rng(84100,source,seed,epoch), checkpoint0/256/512/768/1024. 기존 five-condition V nMAE로 선택, 낮은LR/이른step 동률 규칙. step0 fallback은 성공으로 세지 않는다.

기존 B0/PLAIN/MAG 선택과 prediction을 읽어 hash 검증하고 재사용한다. B0의 fixed1024 표기는 과거 표와 마찬가지로 고정된 selected B0 위 추가 어댑터의1024updates 비교이며 B0를 새로1024checkpoint로 바꾸는 의미가 아니다. 이전 학습 비용과 deployed parameter303624를 standalone8712와 분리한다.

모든 LR/checkpoint 선택을 봉인한 뒤 F0 및 새 두 군의 전체4패널×2seed×selected/fixed1024×standard/9shape 예측을 저장한다. 이후 E 정답 채점. 기존 세 군96views+새 세 군96views=192views. F0의 seed/stage는 정렬용 동일 모델 반복 표기이며 독립 반복 실험이 아니다. 미래/clean/생성state/true delta/mask를 forward에 전달하지 않는다.

패널별·조건별 paired index7-day block bootstrap2000회, RNG91942, 두seed/채널/조건/draw를 독립 날짜로 세지 않는다. 기존 learned_gate panel 통계 방식의 고정 seed 조건부 양측95%구간을 재사용한다. 계약이 새 전역 PASS 문턱이나 다중검정family수를 지정하지 않았으므로 만들지 않는다. 모든4패널·standard조건·9shape·selected/fixed1024를 보고하고 전역 우승 점수 없음. 주표의 네 직접 대조와 F0_PLAIN vs F0를 제공한다. seed별 부호, 절대 nMAE 차이, 상대%와구간을 함께 표시한다. 관측 양성이나 날짜CI를 seed 모집단·독립검증·논문PASS로 해석하지 않는다.

interaction은 계약의 두 절대오차 차분식을 그대로 쓰고 같은 time block draws로 구간을 계산한다. 사후 causal identification 아님. REPORT 첫 부분에서 다섯 질문을4패널×5조건 숫자로 답한다. standalone/second-stage의 근거는 조건별로 판단하고 혼재·불확실을 유지한다. 불리한 조건은 제외하지 않는다.

nRMSE는 각 채널에서 모든원점·draw의 normalized squared error를 평균한 후 sqrt하고 채널평균한다. FAULT는 기존 POINT/BURST6조건 동등가중이다. nMAE/rawMAE/2pinball/crossing도 보존하고 output sort하지 않는다. 시간비는 기존 측정기 차이가 있으므로 순수 학습속도 우위로 쓰지 않는다.

실제모델 no-LoRA name/module 검사, 초기F0·off정확동일, 동결보존, finite gradients/change, restore정확, permutation정합 검사를 수행한다. permutation에는 기존 GPU smoke의 fixed 허용식 normalized max≤1e-5 또는 allclose(rtol1e-4,atol0)를 그대로 사용한다. FP32/batch shape 간 업데이트 완전동일검사는 추가하지 않는다. intent/journal+epoch resume, 모호한update재실행금지. RustDesk만외부GPU예외, 동시작업발견시자원차단. 실행오류와성능불리함분리.

완료 또는 차단 결과를 한국어보고서·manifest·검산으로 보존, scoped commit/push한다. 이 실험 이후 학습금지가 다시 적용되며 자동후속0.
