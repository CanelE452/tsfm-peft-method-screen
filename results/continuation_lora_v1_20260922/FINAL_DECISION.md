# 최종 판단

EXECUTION: COMPLETE
SCIENTIFIC_DECISION: PRACTICAL_DIRECTION_POSITIVE_PILOT_SIGNAL
PRIMARY: CONTINUATION affine versus F0_NATIVE affine, block65–128
PRIMARY_RELATIVE_GAIN_PERCENT: +1.2181964
PRIMARY_ABSOLUTE_EFFECT_CI95: [-0.0055825, +0.0109154]
PRACTICAL_POSITIVE_SEEDS: 2 / 2
STRUCTURAL_AFFINE_GAIN_PERCENT: -0.4952448
FITS: 4 / 4
MAIN_UPDATES: 2048 / 2048
SMOKE_UPDATES: 4 / 4
ATTENUATION_OPTIMIZER_UPDATES: 0
NOVELTY: NOT_ESTABLISHED; Aurora already supports from_second LoRA
PAPER_PASS: NOT_CLAIMED
AUTOMATIC_FOLLOWUP: NONE

INTERPRETATION: 양seed에서 보정 F0 대비 작은 이득은 관찰됐지만 신뢰구간은0을 포함한다. SHARED가 F0 대비1.70% 개선으로 CONTINUATION의1.22%보다 좋으며, 후반 전용 구조의 추가 정확도 우위는 없다. 후반 전용은 첫64 보존과 이번 실행의 학습 계산 시간20.82% 감소를 보였다. 새로운 원리나 우월한 새 방법이 확인됐다는 뜻은 아니다.

두 seed에서 보정 F0 대비 개선 방향이 반복됐다. 제한된 practical-direction pilot 신호이며 확정적인 우위 판정은 아니다.

Solar의 사전고정8열·512→128·두seed에 한정한다. 단순 보정 F0 대비 실용 비교와 SHARED 대비 구조 비교는 별도다. gating+고정F0경로+첫loss제거의 묶음 효과이며 각 요소의 독립 인과효과나 새로운 원리의 증명이 아니다. 첫64 보존이 후반 개선을 보장하지 않는다. 기존 ETTh2 감쇠 진단은 VAL만 사용하며 이전TEST는 재평가하지 않았다. 새 설정·후보를 자동 실행하지 않고 종료했다.
