# 현재 학습 금지 지시

2026-09-19 사용자 지시: “현재 학습은하지 말아줘”.

방법론 논문 목표가 활성 상태여도 새 fit, optimizer update, 재학습, 추가 seed/LR 학습을 시작하지 않는다. 사용자가 다시 명시적으로 학습을 허용하기 전까지 유지한다.

지시 수신 당시 활성 프로세스는 mag_inference_cost_20260919의 기존 모델 추론 비용 측정이다. 이 측정은 inference_mode, backward0, optimizer0이며 가중치·입력·전후 예측 보존을 검산한다. 현재 측정의 마무리·CPU 검산·문서·scoped push만 계속한다. 평가 정답 채점이나 후속 학습을 새로 연결하지 않는다.

## 2026-09-19 명시적 한정 해제

사용자: “이번 요청은 저장소의 이전 새 학습 금지 지시를 이 실험 범위에 한해서만 해제하는 명시적 승인이다.”

[mag_standalone_ablation_v1_20260919 단일 계약](../experiments/mag_standalone_ablation_v1_20260919/CONTRACT.txt)에 한해 F0_PLAIN/F0_MAG 최대16fits /16384main +8smoke updates 및 지정 평가·검산·게시를 허용한다. 기존 B0/PLAIN/MAG 재학습, joint LoRA+MAG, 추가 탐색은 금지한다. 종료 후 자동 후속 학습 없이 기존 학습 금지로 복귀한다. 이전 지시와 기록은 보존한다.
