# TSFM PEFT 시간적 전달 진단 — 결과

**A~C 완료, D는 DEFERRED_GPU_BUSY. 신규 fits=0, optimizer updates=0, GPU forward/backward=0.**

## 1. 무엇이 문제였나

48-fit 연구는 정상 완료됐지만 적은 윈도에서 anchor가 더 유리하다는 가설은 지지되지 않았다. 보고서 seed 열에는 개선율 대신 loss ratio를 출력한 별도 오류가 있었다. 이를 고쳐도 원래 평균 효과와 판정은 바뀌지 않았다.

## 2. 왜 중요한가

Beijing dense는 anchor가 plain보다 +4.172% 좋아도 F0 대비는 약 +0.186%다. 적응 손해 회복과 기본모델을 넘는 이득은 다르다. ETTm2 원래 선택은 step0/F0였다. Study35는 동결 백본 HEAD도 V 이득/D 손해를 보였으므로 백본 표현 손상 하나로 모든 실패를 설명할 수 없다.

## 3. 왜 이 대조인가

B는 같은 후보를 선택하는 창 배치만 바꿨다. RECENT4/SPREAD4는 같은 4 origins, ALL8은 8 origins다. C는 RECENT4 한 checkpoint에서 출력 보정 크기만 바꿨다. H3/H4용 국소 gradient 개입 D는 미수행이므로 관측된 선택/축소 차이를 목적함수 충돌이나 망각의 인과 증거로 바꾸지 않는다.

## 4. 설정 근거

3 sources×2 budgets×2 seeds×2 arms=24개의 서로 의존하는 cell. 기존 V16을 S8/D_diag8로 재사용했다. 원천 수는3이며 step0 중복을 합친 동일7후보를 각 arm 안에서 비교했다. 원래 타깃/마스크/scale/분위 정렬, 채널별 분자/분모를 보존했다. 모든 시간 경계와 4채널 비교를 확인했고 누락 캐시는 없었다.

## 5. 무엇을 지지하나

- SELECTION_TRANSFER_PATTERN: R2/R3의 예측이 8/24 cell에서 달랐으며 S 개선/D 악화는 RECENT4 5, SPREAD4 4, ALL8 8 cell이다. 원천별 반례와 원손실은 선택 보고서/CSV에 모두 포함했다.
- SIMPLE_SHRINKAGE_EXPLAINS_PART: S로 선택한 interior alpha가 원모델과 F0를 모두 넘은 경우는4/24, Beijing/Electricity dense의 seed34001 두 arm이다. seed34000에서는 alpha1이며 보편 재현이 아니다.
- ETTm2는 선택 후보가 F0로 같았지만 저장된 다른 적응 후보 중 D에서 좋은 것이 있었다. ALL8은8/8 해당 cell에서 그런 후보를 선택하지 못했다. 사후 oracle은 진단값이다.
- NUMERICAL_ERRATUM_ONLY는 seed 파생 열에 한정된다. corrected_seed_gains.csv와 aggregate_replay.json이 원점수부터 판정까지의 의존 경로를 기록한다.

## 6. 무엇은 아직 모르나

8개 D origins와 재사용 V로 새 일반화·유의성·동등성을 선언할 수 없다. 일부 축소 효과만으로 새 loss 필요성은 약하다. H3/H4는 D 미실행 때문에 INSUFFICIENT_EVIDENCE다. 외부 YOLO 작업과 RustDesk가 GPU에 있었고 순간 여유 약5.7GiB만으로 최대 점유/간섭 안전성을 보장할 수 없어 CPU 작업만 병행했다. 기존 GPU guard를 유지하고 예약·무한 대기를 만들지 않았다.

## 7. 다음 한 개 계획

새 기간에서 동일 창 수 RECENT4/SPREAD4의 선택 전달을 검증하는 **0-new-fit** 계획을 제안한다. 기존84개 dense checkpoint를 재사용하므로 24-fit 상한 아래이며 새로운 모델 학습을 반복하지 않는다. 구체적 분할·추론 예산·반증 대안은 next_experiment_proposal.md에 있고 실행하지 않았다.

| 관찰 | 현재 지지되는 설명 | 배제하지 못한 설명 | 다음 행동 |
|---|---|---|---|
| 선택 창을 바꾸면 일부 D 손실이 달라짐 | 선택의 시간적 전달 패턴 | 작은 표본/선택 잡음/기간별 목표 차이 | 미사용 기간에서 동일 창 수 선택 대조 |
| 일부 S-only 축소가 원모델과 F0를 넘음 | 보정 크기가 일부 사례 설명 | 선택 예산 증가/seed 변동 | 복잡한 보존 기법 발명 보류 |
| 국소 개입 미실행 | 없음 | objective mismatch/temporal conflict | GPU 충돌 없이 별도 확인 전 인과 주장 금지 |

## 검증·변경·실행 명령

예측 캐시228개(기존 E30개 포함), max primary 재계산 오차1.11e-16, sufficient-stat 재집계312회, checkpoint192개 hash 확인. 과거/소스1193개 파일 불변. 정정·시간 경계·선택 누출·alpha endpoint 8개 테스트 통과. CPU 계산 약5초(자료 조사·보고서 시간 제외), CPU peak RSS 약889MiB. 참조 저장소 main의 선택 파일28개 조사; 이전 미커밋 작업은 보존.

새 코드: scripts/run_temporal_transfer_diagnostic_v1.py, scripts/report_temporal_transfer_diagnostic_v1.py, tests/test_temporal_transfer_diagnostic_v1.py. 산출물은 results/temporal_transfer_diagnostic_v1/, 지시문은 research/temporal_transfer_diagnostic_v1/USER_INSTRUCTION.txt. commit/push/브랜치 변경/자동 후속 없음.

```bash
CUDA_VISIBLE_DEVICES='' scripts/with_cuda.sh .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_temporal_transfer_diagnostic_v1.py
CUDA_VISIBLE_DEVICES='' scripts/with_cuda.sh .venv/bin/python scripts/run_temporal_transfer_diagnostic_v1.py
.venv/bin/python scripts/report_temporal_transfer_diagnostic_v1.py
```

위 진단 실행 명령은 이미 완료됐다. 동명 출력이 있으면 거부하므로 다시 실행하려면 새 run suffix와 재실행 이유를 고정해야 한다. 기존48fit를 재학습하지 않는다.

시간순 검증 및 선택 편향의 표준 배경: [FPP3](https://otexts.com/fpp3/tscv.html), [Cawley & Talbot](https://jmlr.org/papers/v11/cawley10a.html). 이번 S/D 선택과 alpha 조합은 사후 개발 진단이며 신규 방법이나 보편 검정이 아니다.
