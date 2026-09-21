# 최종 판단

실행 상태: COMPLETED_WITH_DISCLOSED_PROTOCOL_DEVIATION

과학적 분류: **STATE_OR_CALIBRATION_SUFFICIENT**

이번 고정 예산에서는 U의 예측 폭 전달이 S보다 실용적으로 의미 있는 추가 가치를 보인다고 판단하기 어렵다. Electricity의 작은 양의 평균 차이는 인정하지만, S가 R 대비 이득의 대부분을 설명하고 U−S는 seed별 방향이 갈린다. ETTh1에서는 U와 S가 거의 같고 둘 다 R보다 전체 확률점수가 나쁘다. 효과가 정확히 0이라는 결론은 아니다.

U의 S 대비 전체256 개선(양수는 U가 좋음):

```text
     source baseline candidate baseline_variant  effect_a_minus_b  relative_gain_percent  ci95_low  ci95_high
Electricity        S         U          ordered          0.000031               0.017546  0.000008   0.000061
Electricity        S         U           affine          0.000028               0.015438  0.000005   0.000057
      ETTh1        S         U          ordered          0.000003               0.000815 -0.000032   0.000041
      ETTh1        S         U           affine          0.000002               0.000471 -0.000033   0.000041
```

일반 R rollout LoRA는 F0 중앙값 경로보다 ordered 점수를 Electricity 2.798%, ETTh1 8.740% 개선했다. 이것은 rollout 학습을 포함한 LoRA 적응 전체의 결과이며, teacher-forcing 대조가 없으므로 rollout 노출만의 인과효과가 아니다.

Chronos-2는 두 자료의 평균 확률점수와 실측 latency에서 강한 대안이다. 그러나 더 많은 메모리, 80% coverage의 차이, ETTh1 점예측 오차와 보정 후 PCE의 손익이 있어 `DIRECT_LONG_MODEL_DOMINATES`라는 전면 우월 분류는 채택하지 않는다. 현재 공식 branching도 중앙값-only보다 개선되므로 이를 약한 중앙값 기준선과 혼동하지 않는다.

24 fits × 512 = 본학습 12,288 updates, 폐기 smoke 12 updates, 합계 12,300으로 종료했다. 26개 CAL/TEST 모델과 104개 자원 조합을 완료했다. 선택 seed 92120은 반복 평균에서 제외했다. 추가 fit·LR·rank·epoch·seed·자료·후속 실험은 0이다.

구현 이탈을 별도 공개한다. 첫 두 Electricity R 선택 fit은 전체 실행 코드 봉인 전에 별도 runner로 실행됐고, 원학습 전후 frozen weights/buffers digest가 측정되지 않았다. 1,024 updates와 원 checkpoint를 보존해 재학습하지 않고 복구했다. 이 두 fit은 LR 선택에 사용됐으므로 완전히 영향 없는 사건이라고 하지 않는다. 나머지 22 fits는 전후 digest를 직접 확인했다. 수치 검산 성공이 이 기록 누락을 소급해 없애지는 않는다. [복구 기록](PRESEAL_EXECUTION_RECOVERY.json), [원 실행 코드](../../experiments/rollout_uncertainty_peft_v1_20260921/FIRST_FIT_RUNNER_ORIGINAL.txt).

자료 실패와 자원 차단은 없었다. 위 구현 사건은 복구됐으며, U의 추가 가치 미확보는 별도의 과학적 관찰이다. 512 updates·두 반복의 제한 아래 판단하며, 신규성이나 논문 PASS를 선언하지 않는다. rollout PEFT 전체의 반증도 아니다. **자동 후속 0으로 종료한다.**
