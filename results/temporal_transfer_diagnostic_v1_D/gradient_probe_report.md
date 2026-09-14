# 고정 체크포인트 기울기·미소 교란 진단 D

진단 완료: 12개 상태, forward 180회, backward 60회, 미소 교란 48개. 새 학습·optimizer update·E 배열 접근·영구 가중치 변경은 모두 0이다.
실행 시간 2.12분(모델/데이터 사전 해시 검사 및 구현·보고서 작성 시간 제외). GPU 최대 할당 0.91 GiB.

판정: LOCAL_TEMPORAL_CONFLICT. 국소 목적 충돌 0/12 상태, 국소 시간 구간 충돌 4/12 상태.
이는 고정 step450 상태와 각 구간 마지막 2개 origin에 한정된 결과다. 실제 Adam 업데이트나 장기 학습 실패의 원인을 입증하지 않으며, 모든 상태·시간 구간으로 일반화할 수 없다.

| 상태 | native→S | native→D | eval→S | eval→D |
|---|---|---|---|---|
| beijing_233_34000_native | WORSENS | WORSENS | WORSENS | WORSENS |
| beijing_233_34000_native_anchor | WORSENS | WORSENS | WORSENS | WORSENS |
| beijing_233_34001_native | WORSENS | WORSENS | WORSENS | WORSENS |
| beijing_233_34001_native_anchor | WORSENS | WORSENS | WORSENS | WORSENS |
| electricity_new_233_34000_native | IMPROVES | WORSENS | IMPROVES | WORSENS |
| electricity_new_233_34000_native_anchor | IMPROVES | WORSENS | IMPROVES | WORSENS |
| electricity_new_233_34001_native | IMPROVES | WORSENS | IMPROVES | WORSENS |
| electricity_new_233_34001_native_anchor | IMPROVES | WORSENS | IMPROVES | WORSENS |
| ettm2_later_233_34000_native | IMPROVES | IMPROVES | IMPROVES | IMPROVES |
| ettm2_later_233_34000_native_anchor | IMPROVES | IMPROVES | IMPROVES | IMPROVES |
| ettm2_later_233_34001_native | IMPROVES | IMPROVES | IMPROVES | IMPROVES |
| ettm2_later_233_34001_native_anchor | IMPROVES | IMPROVES | IMPROVES | IMPROVES |

두 epsilon 모두 부호가 일치하고 1차 예측과 근사가 맞는 경우만 IMPROVES/WORSENS로 읽었다. NONLOCAL_OR_NUMERICAL에는 효과가 FP32 수치 바닥보다 작거나, 두 크기의 부호가 다르거나, 근사가 나쁜 경우가 포함된다. epsilon이나 임계값을 결과에 맞춰 변경하지 않았다.
수치 바닥은 8×FP32 epsilon×max(1,|baseline loss|), 허용 근사 오차는 |예측|의 50%+수치 바닥이다. 이것은 진단의 수치 신뢰성 필터이며 성능 PASS 기준이 아니다. 원시 효과와 1차 예측은 local_perturbation_effects.csv에 보존했다.

BF16/FP32 평가 손실 차이의 최대 절댓값: 0.0006509167. 원래 저장 BF16 V 예측과 재계산의 최대 절댓값 차이: 3.0517578e-05.
FP32 baseline sort tie 0개, pinball target tie 0개. 결측값 마스크와 채널별 유효 관측 수, train-only scale을 유지했다.
native task는 기존 native_loss/21이며 anchor는 기존 0.1×정렬된 F0 출력 L1/채널 scale이다. native arm에서도 anchor 기울기를 기하 비교용으로 계산했다. 모든 gradient는 동일한 LoRA 1,179,648개 좌표다.

48개 교란마다 trainable/frozen/checkpoint/RNG 복원 또는 보존 검사를 통과했다. 교란 48개는 모두 폐기했다. 새 optimizer는 생성하지 않았다. 원본 A–C의 D 보류 기록은 당시 상태를 나타내므로 그대로 보존했다.
GPU에는 학습 프로세스가 없었고 RustDesk 표시 서비스만 있었다. 정확한 실행 경로와 512MiB 이하 메모리에만 예외를 허용했으며, 다른 계산 프로세스·메모리 부족·기존 RAM/2시간 제한을 검사했다.

## A–D 통합 해석과 다음 한 가지 제안

A는 seed별 gain 보고식 오류를 수정했지만 기존 주된 실패 판정은 바뀌지 않았다. B는 시간 구간에 따라 선택 가능한 체크포인트의 유효성이 달라지는 사례와 F0 선택이 뒤 구간의 개선 후보를 놓치는 사례를 분리했다. C는 S에서 고른 중간 출력 보정 강도가 4/24 셀에서 뒤 구간의 두 끝점보다 좋았지만 한 seed에 집중되었다.
D는 위 표의 제한된 상태에서만 목적/시간 방향을 판독한다. 서로 다른 실험의 상태·origin·선택 규칙이므로 D만으로 B/C의 현상을 모두 설명했다고 주장할 수 없다.

다음 우선순위는 기존 제안인 새 미래 구간에서 RECENT4와 SPREAD4 선택 규칙을 같은 검증 비용으로 비교하는 0-fit 실험 하나다. D에서 국소 충돌이 관찰되어도 바로 gradient surgery나 새로운 PEFT 방법을 채택하지 않는다. 이 후속 실험은 제안만 유지하며 이번 실행에서 시작하지 않았다.
방법론 논문 후보를 평가하기 전에 선택 절차가 실제 미래 구간으로 전달되는지 분리해서 확인하는 단계다. 논문 PASS나 성능 개선을 보장하는 결과가 아니다.

[A–C 원본 보고서](../temporal_transfer_diagnostic_v1/REPORT.md) · [후속 실험 설계](../temporal_transfer_diagnostic_v1/next_experiment_proposal.md) · [실행 영수증](execution_receipt.json) · [수치 검산](artifact_verification.json)
