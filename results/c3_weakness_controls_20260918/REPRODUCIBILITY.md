# 재현 절차와 공개 범위

## 실행 규약

- 유일한 사용자 설계: [EXECUTION_CONTRACT.md](EXECUTION_CONTRACT.md). 다운로드 원본과 동일한 문서다.
- [PROTOCOL.md](PROTOCOL.md), [SEAL.json](SEAL.json), [PLAN.json](PLAN.json): 결과 이전에 고정한 과학적 조건과 코드·입력 해시. 사전 등록 커밋은 `396de8d`다.
- 설명 대조 3개 × 원천 2개 × (선택 경로 2개 + 반복 경로 3개) = 30 fits. 경로당 1,024 updates, 폐기 smoke 12 updates를 합해 총 30,732 updates다.
- 기존 B0/C3와 완료된 대조는 재사용했다. 원점·학습 자료·변형·평가 패널은 부모 결과에 고정된 것을 사용했다.

## 완료 결과 검산

저장소 루트에서, 기존 로컬 캐시와 `.venv`가 있는 환경의 명령이다. 모델 학습이나 새로운 평가 예측 생성 없이 선택·예산·원점수·파일 해시를 검사하고 보고서를 만든다.

```bash
.venv/bin/python -m experiments.c3_weakness_controls_20260918.final_audit
.venv/bin/python -m experiments.c3_weakness_controls_20260918.report
.venv/bin/python -m experiments.c3_weakness_controls_20260918.decision
```

학습과 평가를 실제 수행한 명령은 아래다. **완료 결과를 읽기 위해 다시 실행할 필요가 없다.** 기존 영수증과 체크포인트의 해시가 일치할 때만 완료 경로를 재사용하며, 불명확한 optimizer update를 삭제하고 다시 세는 방식으로 재개하지 않는다. 예산을 추가할 권한도 부여하지 않는다.

```bash
.venv/bin/python -m experiments.c3_weakness_controls_20260918.runner
```

최초 실행의 CPU 참조 검사와 실모델 smoke 검사는 각각 CPU_CHECKS.json과 SMOKE.json에 있다. smoke와 본학습의 optimizer 변경은 별도 원장으로 남겼다. 모든 선택은 EVALUATION_SEAL.json으로 고정했고 전체 54개 예측을 저장한 뒤 ALL_PREDICTIONS_SAVED.json을 만들고 채점했다.

## 파일별 의미

- `fits/*/receipt.json`, `FIT_LEDGER.csv`, `UPDATE_LEDGER.jsonl`: 실제 경로·선택 checkpoint·시간·학습 횟수.
- `RAW_SCORES.csv`: 방법·패널·조건·seed별 nMAE, 원단위 MAE, channel-mean nRMSE, normalized twice-pinball, crossing.
- `NEW_ORIGIN_SCORES.csv.gz`: 새 대조의 원점·채널별 점수. 두 합성 draw의 오차를 평균한 값이다.
- `ALL_ORIGIN_SCORES.csv.gz`: 검증한 기존 대조의 점수를 함께 넣은 분석 입력. 원래 예측값 파일을 대신하지 않는다.
- `MATCHED_CONTRASTS.csv`, `SEED_EFFECTS.csv`: 전체 대조·조건 및 seed별 효과. 양수는 `new`가 `baseline`보다 낮은 nMAE다.
- `PAIRED_ORIGIN_DIFFERENCES.csv`: 세 주요 대조의 같은 원점별 오차 차이.
- `PARAMETER_DYNAMICS.json`: 최종 학습의 실제 변경량과 V로 선택된 gate/gamma를 구분한다. checkpoint 0 선택도 보존한다.
- `RESOURCE_REPORT.csv`, `RESOURCE_COMPARISON.csv`: 동일 입력 수의 새 추론 측정 및 기존 C3 등과의 비교. 측정 시점 차이가 있으므로 작은 시간 차이는 확정적 속도 우위가 아니다.
- `VERIFICATION.json`: 독립 계산으로 확인한 범위. 각 새 view/조건의 첫·마지막 원점/채널에서 scalar metric을 검사하고 전체 원점 점수를 다시 집계한다. 모든 원점을 scalar 방식으로 재생했다는 뜻은 아니다.

날짜 bootstrap은 고정된 세 seed의 평균에 조건부이다. 3개 주 대조의 Bonferroni 구간은 **각 패널 안**의 다중성만 보정한다. 전체 연구 이력이나 모든 패널·조건의 다중성을 제거하지 않는다. 전력 전이의 계열+시간 보조 구간도 함께 남긴다.

## 로컬 의존성과 논문 한계

GitHub에는 코드, 규약, 봉인, scalar 원점수, 검산 기록, 보고서와 그림을 공개한다. 원자료·모델 가중치·전체 예측은 `.cache/`의 ignored 파일이다. 부모 실험의 캐시와 정확한 해시가 필요하므로 **GitHub만 내려받으면 완전한 수치 재생이 가능하다는 주장은 하지 않는다.** 필요한 파일과 해시는 SEAL.json, BASELINE_MANIFEST.json, DATA_MANIFEST.json, PREDICTIONS.json에서 추적한다.

현재 E는 반복 사용한 개발 구간이며 실제 사건 레이블이 없다. 새 방법, 추가 seed/LR, 다른 자료 학습은 자동 실행하지 않는다. LCL 준비 및 공식 온라인 TAFAS/COSA 재현은 이번 범위 밖이다.
