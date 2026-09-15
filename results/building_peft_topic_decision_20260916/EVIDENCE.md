# 기존 증거 재검토

[확인] 기존 두 건물 실험의 local 예측·체크포인트를 hash 확인 후 읽었다. 모든 과거 성능 표본은 DISCOVERY다. 새 GPU 학습은 이 재검토에0 fits다. 아래 raw는 정렬 여부와 별개인 원래 부하 단위 RMSE이며 출력 정렬 여부를 별도 표기한다.

| 실험/비교 | 표본과 직접 비교 범위 | raw/scaled 오차 | 편향/평균 제거 분해 | 학습량에 따른 변화 |
| --- | --- | --- | --- | --- |
| transfer F0/LOCAL/fixed120/AFFINE | 같은 episode끼리만 paired; rank1·seed61600 선택 epoch1와 fixed120 | episode CSV 전체 공개 | 각 출력의 b²와 centered MSE 공개 | 같은 trajectory checkpoint raw/q 둘 다 공개 |
| transfer tune rank1/rank8 | 동일 건물·origin·H에서 rank별; 이번 신규seed61680과 같다고 하지 않음 | rank/step별 공개 | 사후 정답 분해 | 0/n/4n/16n/120 checkpoint |
| coverage F0/STANDARD/AFFINE | 기존 rank8·선택recipe; transfer rank1과 직접 대조 금지 | 원점수와 분해 공개 | 같은 시점 안의 비교만 | 최종120 및 별도 recipe 곡선은 기존 장부 참조 |

[확인] 상세: [episode 분해](evidence_episode_decomposition.csv), [학습량별 raw/정렬 변화](evidence_training_changes.csv), [직접 선택-vs120 기존 표](../building_transfer_subspace_v1_20260915/budget_control_comparison.csv).

[확인] MSE=b²+centered MSE 독립 분해의 최대 절대차 1.82e-12. RMSE의 선형 합을 사용하지 않는다. 평균 잔차는 미래 정답으로 계산한 진단이며 배포 입력이나 oracle 선택 성능이 아니다.

| 경쟁 설명 | 지지 관찰 | 반례 | 없는 근거 |
| --- | --- | --- | --- |
| 적응량 선택이 건물별로 달라짐 | 기존 tune와 dev의 fixed120 순위 역전 | fixed120도 일부 건물에서 좋음 | target 미래 없이 신뢰할 선택 규칙 |
| 레벨과 형상 보정의 혼동 | affine 보정으로 일부 내부 적응 손해 회복 | affine 자체가 raw/일부 건물에서 악화 | 두 잔차 항을 분리한 변경의 직접 비교 |
| 단순 local LoRA/보정만으로 충분 | 기존 fixed120이 최저 dev macro | 짧은 H3에서는 악화 | 새 변경이 가장 가까운 단순 대조를 넘는 잠금 평가 |

[설계] 새 STD16 fits는 동일 origin이지만 seed61680·LR2개·ZERO 선택 포함으로 기존 실행과 다르다. 동일실험을 이름만 바꿔 반복하지 않는다. 이 새 seed의 결과를 받은 후 한 후보의 필요성과 반례를 명세한다.
