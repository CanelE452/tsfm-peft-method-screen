# 동일 MASTER_CLI 완료 결과 재사용 감사

상태: **VERIFIED_REUSE / 계약 내 필수 작업 완료**. 이번 요청에서 새 학습0회, optimizer update0회. 기존58 fits를 또 수행하지 않았다. 계획만 작성한 상태가 아니라 실제 완료 결과의 파일·선택·모델 복원·원점수를 다시 검산한 상태다.

## 단일 실행 계약 확인

다운로드의 `additive_persistence_paper_cli_20260917.txt`와 `(1).txt`가 기존 [MASTER_CLI.txt](../additive_persistence_validation_v1_20260917/MASTER_CLI.txt)와 byte 단위로 일치했다. SHA-256은 `409af4da5ae2fb77f9c25f24081631a064083fe0012b646cace20f01ed4a16f5`이다. 별도의 ZIP나 원래 제공된 reference_checks.py는 확인하지 못했으며, 과거 작업에서 문서에 따라 작성한 CPU 참조검사를 그대로 재실행했다. 이를 첨부 원본 PY 복원이나 실제 모델 검사라고 부르지 않는다.

감사 시작 HEAD는 `2f4ebb4c62819511fa2e324ac2de6b09d2b3c259`, origin/main과 일치했고 사용자 변경은 없었다. 계약 기준 `196820e` 이후 해당 계약의 구현·학습·평가·보고서가 이미 추가되어 있었다. NESO 추가 연구나 다른 계약의 결과를 이번 비교에 합치거나 해당 학습을 재개하지 않았다. C3 구조·mask·threshold·window·rank와 B0 가중치를 변경하지 않았다.

## 학습·평가 장부

| 계약 블록 | 이미 완료한 본학습 | 이번에 추가한 학습 |
|---|---:|---:|
| 원래 두 source의 세 번째 seed B0/C1/C2/C3 | 8 | 0 |
| MEAN/ROTATE16/RECENCY 선택·반복 | 30 | 0 |
| ETTm2 별도 B0 및 C1/C2/C3 절차 재현 | 20 | 0 |
| 합계 | 58 | 0 |

각 fit의1~1,024 step 연속성과 중복 없음, 마지막 optimizer intent의 JOURNALED 상태를 확인했다. 본학습59,392 updates와 smoke36 updates의 합59,428로 전체 상한을 이미 사용했다. forward 복원 검사는 update를 추가하지 않는다. 기존 미학습 계열 전이·변화 형태·출력 보정 대조를 포함한246개 prediction views가 존재한다. 246은 독립 모델 또는 학습 횟수가 아니다.

**이 계약에서 남은 필수 학습·평가는0개**다. ETTm2의 MEAN/ROTATE16/RECENCY 학습은 원래 계약이 명시적으로 제외한 범위이므로 누락된24/30 fits처럼 세지 않는다. 정식 COSA/TATO 전체 비교·다른 backbone·실제 사건 레이블·독립 source 인증은 기존부터 계약 밖 미완료 범위다.

## 이번에 실제 다시 확인한 것

- 기존 결과 파일348개를 감사 전후 hash로 비교했고 모두 불변이었다.
- source/data/model/결과의1,382개 고유 파일 hash를 확인했다. 새58 fits의 checkpoint290개(각0/256/512/768/1024)와 기존 selected 가중치를 검사했다.
- CPU 수학검사20개를 재실행했다.
- selected 모델54개를 실제 Chronos 경로에 복원하고 저장 forward와 대조했다. 적용 대상의 adapter-off B0 정확 복원도 다시 확인했다. 실제 학습의 동결 보존·초기 B0 동일성은 기존 smoke/초기검사/fit receipt와 그 hash를 확인해 재사용했다. 추가 smoke optimizer update는 실행하지 않았다.
- 모든246 prediction view의 hash를 검사했다. 각 view/조건의 첫·마지막 원점과 첫·마지막 채널, 두 draw를 포함한8,400항목에서 nMAE/pinball scalar를 다시 계산했다. 이것은 모든 원점의 scalar 재계산이라는 뜻은 아니다. 별도로 모든 원점 저장 점수의 vector 재집계를 수행했다.
- source/arm LR 선택10개와 V 기반 alpha/beta 선택9개를 재계산했다. 원점수에서 직접 효과660행을 다시 확인했다. HISTORY_SUBSET의 별도 비율 행은 이번660행에 포함하지 않으며 해당 보조표의 기존 봉인 결과는 보존한다.
- 기존 통계 코드 수정은 E 채점 전에 동일 block draw를 계약에 맞추기 위한 것이었고 변경 파일은 statistics.py와 MASTER_PROTOCOL.md 두 개뿐임을 확인했다. 모델·학습·선택·데이터를 결과에 맞춰 변경한 것이 아니다.
- 선택 seal → 전체 예측 저장 순서와 manifest를 재검증했다. 새 선택·출력 보정·E 기반 튜닝은 없었다.

자세한 범위는 [재검증](VERIFICATION.json), [기존 verifier 재실행 기록](REPLAY_VERIFICATION.json), [선택·효과 재계산](SELECTION_EFFECT_REPLAY.json), [hash 장부](CHECKED_HASHES.json)에 있다.

## 재사용하는 과학적 결과

아래는 이번 계약의 원래 비교 결과이며 새로운 성능 실험으로 표시하지 않는다. 양수는 nMAE 감소다.

| SHIFT8 비교 | 결과 |
|---|---:|
| 전력16계열: 일반 어댑터 C2/B0 | +4.828439% |
| 전력16계열: C3/B0 | +7.112013% |
| 전력16계열: C3/C2 | +2.399429% |
| 전력16계열: C3/MEAN | +2.009914% |
| 전력16계열: C3/ROTATE16 | +2.544647% |
| 전력16계열: C3/RECENCY | +0.180396% |
| ETTm1: C3/C2 | −1.847203% |
| ETTm2: C3/C2 | −0.044834% |

전력16계열 C3/C2는 세 seed 모두 양수지만 C3/RECENCY의 세 번째 seed는−0.019538%였다. 전력 전이 REFERENCE/FAULT는 B0보다 각각0.118831%/0.173071% 악화했다. ETTm2 C2/C3는 두 seed에서step0이 선택됐고 그 seed도 평균에 남았다. 불리한 결과를 제외하거나 학습이 실행되지 않은 것으로 바꾸지 않는다.

전이16계열 중 과거 성능 노출 확인4개·불명12개로 검증된 완전 미사용 계열은0개다. B0가 학습하지 않은 계열이라는 사실과 독립 자료라는 주장은 다르다. 같은 전력 원천·날짜의 탐색적 전이이며 ETTm2도 노출된 benchmark family다. 실제 오류/변화 사건 레이블은 없다.

원래 계약의 과학 판정은 좁은 전력 SHIFT8의 추가 이득을 보존하면서 RECENCY 대비 고유 기여와 source 일반성을 제한한다. 논문 PASS·범용 강건성·정식 선행 우위로 선언하지 않는다.

## 자원·필수 최종 문서

기존58 fits의 training compute는 약31.94분, 전체 run-all wall은 약62.14분이었으며 기존 B0·이전 additive 학습비용은 별도 [비용 장부](../additive_persistence_validation_v1_20260917/COST_ACCOUNTING.json)에 있다. 추가 어댑터8,712개뿐 아니라 B0 LoRA294,912개가 배포 시 남는다. 이를 같은 비율의 GPU 메모리 절감이라고 해석하지 않는다. SHRINK의 두 forward 비용도 기존 RESOURCE_REPORT에 포함됐다.

이번 복원·검산 GPU guard 구간은67.08초(안전 확인 대기30.27초 포함), 최소 여유8,575MiB, 승인되지 않은 외부 compute0건이었다. guard의 `external_compute_samples`는 승인된 RustDesk까지 포함하는 원래 필드이므로 위반 횟수가 아니다. 보고서의 새학습0회와 과거58회 비용을 구분한다.

필수 한국어 네 문서는 이미 완성됐고 hash 검증 후 원본 그대로 재사용한다:

- [REPORT.md — 전체 원점수·seed·형태·손해·비용](../additive_persistence_validation_v1_20260917/REPORT.md)
- [FINAL_DECISION.md — 실행과 과학 판정 구분](../additive_persistence_validation_v1_20260917/FINAL_DECISION.md)
- [PAPER_CLAIM_EVIDENCE.md — 주장·근거·미완료 비교](../additive_persistence_validation_v1_20260917/PAPER_CLAIM_EVIDENCE.md)
- [PAPER_OUTLINE.md — 증거에 맞춘 논문 개요](../additive_persistence_validation_v1_20260917/PAPER_OUTLINE.md)

code/규약/원점수/검산은 commit·push하지만 raw data/weights/prediction cache는 로컬 보존이다. GitHub만으로 전체 수치 재생이 완결된다고 주장하지 않는다. 자동 후속 학습·새 구조 탐색은 시작하지 않는다.
