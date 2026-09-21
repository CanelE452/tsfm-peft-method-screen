# 최종 판단 검토 초안

상태: REVIEW_REQUIRED

정해진 학습·선택·추론·채점은 완료됐습니다. 아래 관찰값으로 과학적 범주를 검토해야 하며 자동 METHOD_SIGNAL/PASS로 분류하지 않았습니다.

- Electricity ordered, additional_width_information: 개선 0.000031 (0.018%), 원점 CI [0.000008, 0.000061].
- Electricity ordered, total_U_vs_R: 개선 0.000575 (0.321%), 원점 CI [0.000237, 0.000854].
- Electricity affine, additional_width_information: 개선 0.000028 (0.015%), 원점 CI [0.000005, 0.000057].
- Electricity affine, total_U_vs_R: 개선 0.000308 (0.172%), 원점 CI [0.000014, 0.000598].
- ETTh1 ordered, additional_width_information: 개선 0.000003 (0.001%), 원점 CI [-0.000032, 0.000041].
- ETTh1 ordered, total_U_vs_R: 개선 -0.003319 (-0.857%), 원점 CI [-0.006560, -0.000021].
- ETTh1 affine, additional_width_information: 개선 0.000002 (0.000%), 원점 CI [-0.000033, 0.000041].
- ETTh1 affine, total_U_vs_R: 개선 -0.004725 (-1.223%), 원점 CI [-0.008904, 0.000513].

선택용 seed 제외. 좋은 결과도 신규성/논문 PASS가 아니며, 음수 결과도 rollout PEFT 전체 반증이 아닙니다. 실측 자원과 seed별 반례를 함께 검토하십시오. 자동 후속 실험 없음.
