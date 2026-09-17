# 입력 오류 강건성·지속 변화 보존 — 부분 진행 보고

**상태: PARTIAL / BLOCKED_MISSING_REFERENCE_FILES. 학습 0/48 fits, optimizer update 0. 성능 실패 판정이 아니다.**

## 완료한 실제 작업

기준 및 원격 main은 `72eda9650a121d484f5246da725d4b6a6ed5733a`였다. 정확히 같은 완료 실험은 없었다. 다운로드의 `outlier_signal_peft_cli_20260917.txt`를 유일한 계약으로 복사했으며 이전 실험을 재개하지 않았다.

공개 두 모델의 revision과 가중치 hash, TSFM-Biases와 TATO의 실제 코드를 확인했다. 두 원자료 hash를 확인하고 TRAIN만으로 첫 네 적격 채널 및 population std를 고정했다. ETTm1 날짜는 단조·중복 없음·15분 격자를 통과했다. Electricity에는 원본 timestamp가 없어 index-day를 쓴다.

| 원천 | 채널 | TRAIN/V/E 서로 다른 날짜 | 선택 날짜 span: TRAIN/V/E |
|---|---|---|---|
| Electricity | 원본 열 0/1/2/3 | 256/64/128 | 630/214/216일 |
| ETTm1 | HUFL/HULL/MUFL/MULL | 256/64/128 | 428/142/144일 |

각 역할의 full eligible day에서 날짜를 먼저 고르고 phase roster를 한 번 섞었다. 모든 phase count max-min≤1이다. day/week/phase/overlap/correlation 원표는 `origin_audit.json`에 있다. 입력과 정답은 별도 캐시에 보관했다. E 정답은 complete-case 검사와 분리 저장에만 접근했고 예측 성능을 채점하지 않았다. 이 원천의 개발 기간은 기존에 사용했으므로 독립 test라고 부르지 않는다.

실제 Bolt-small의 CPU native forward와 명시적 wrapper를 연결했다. q/v 36개에 LoRA rank8/alpha16을 붙였다. A0/A2의 학습 파라미터는 294,912개, A4/A5는 299,784개다. 실제 Electricity 8개 입력에서 native-wrapper 및 A4/A5 초기 출력과 A2의 TRAIN-scale 정규화 최대 차이는 모두 0이었다. A4/A5 초기 추가 가중치 hash가 같고 원래 본체와 head hash가 보존됐다. 이는 초기 연결 검사이며 학습·복원 검사를 대신하지 않는다.

별도로 작성한 `test_local_components.py`의 CPU 검사 5개를 실행해 통과했다. 실제 입력 8개에 과거 오류를 가한 clip/Hampel/feature의 NumPy-Torch 대조, scalar 2-pinball 및 단위 불변성, full-day 선정·부족 날짜 거부, adapter bound와 초기 동일성을 확인했다. **이 파일은 첨부된 test_reference.py가 아니며 그 검사를 통과했다고 주장하지 않는다.**

## 멈춘 조건과 정확한 미실행 범위

사용자가 지정한 `reference_core.py`, `test_reference.py`를 Downloads·Documents·Codex attachments 및 홈 파일명 검색에서 찾지 못했다. Downloads의 ZIP 내부 파일 목록에도 없다. 동명 fsspec 테스트는 무관한 라이브러리 파일이다. 사용자 지시의 “reference_core.py와 test_reference.py를 검토하고 CPU 검사를 실행해”를 아직 충족하지 못했다. 자체 구현으로 해당 첨부 검사까지 통과한 것으로 바꾸지 않았다. 두 파일 또는 ZIP의 로컬 경로가 필요하다.

아직 전체 학습 runner를 완성한 상태가 아니다. generator/label stream 권한 검사, 첨부 코드 대조, GPU microbatch 봉인, 24 smoke updates, 선택 24 fits, 반복 24 fits, 최대 98 series 제한 재현, 선택 봉인, E 예측·채점, 선택 모델 자원 측정·독립 검산 모두 미실행이다. 본학습 49,152 updates와 smoke 24 updates 예산은 전부 남아 있다. `FIT_LEDGER.csv`의 48행은 NOT_STARTED 계획 슬롯이며 실행 횟수가 아니다. GLOBAL_EVALUATION_SEAL·선택·점수 파일을 가짜로 만들지 않았다.

현재 실행 중인 학습·백그라운드 worker는 없다. GPU 조회 당시 RTX3080의 여유 메모리는 9,086 MiB였고 외부 compute는 허용된 RustDesk 272 MiB뿐이었다. GPU 안전 자체가 이번 차단 원인은 아니다. CPU 검사가 사용한 자원과 향후 GPU 학습 비용을 혼동하지 않는다. 학습 peak/time 및 추론 자원 절충은 미측정이다.

## 과학적 질문과 아직 판단할 수 없는 것

측정 오류는 입력만 바꾸고 지속 변화는 과거 끝과 미래를 함께 바꾸어, 오류 감소와 유용한 변화 보존을 분리하려는 비교다. 실제 오류·사건의 전문가 레이블은 확보하지 않았다. 원자료는 UNMODIFIED_REFERENCE이고, SYNTHETIC_MEASUREMENT_FAULT / SYNTHETIC_PERSISTENT_SHIFT / HISTORY_TRIGGERED_SUBSET은 서로 다르다. 동일한 관측 과거에 다른 정답이 있을 때 이를 oracle scenario로 구분해서는 안 된다.

A1-A0는 증강, A2/A3-A1은 전처리, A5-A4는 같은 크기 adapter에서 제거된 관측 차이의 추가 가치, A5-A2는 추가 경로 전체 가치를 분리할 예정이다. 아직 비교 점수가 없으므로 오류 감소·원자료 성능·지속 변화 보존·추가 비용 모두 N/A다. 단순 방법의 충분성이나 새 구성요소의 필요성도 판단하지 않았다. 좋은 결과라도 논문 PASS로 부르지 않는다.

## 재개

첨부 두 파일을 확보해 hash와 CPU 검사 및 현재 구현 대조부터 진행한다. 현재 원점·모델 캐시는 hash 검증 후 재사용하며 중복 준비·다운로드를 피한다. 현 시점의 실행 가능 명령은 CPU 감사와 자체 검사뿐이다:

```bash
.venv/bin/python -m experiments.outlier_signal_peft_v1_20260917.audit
.venv/bin/python -m unittest experiments.outlier_signal_peft_v1_20260917.test_local_components -v
```

이 명령은 학습을 시작하지 않는다. 전체 runner 구현·봉인·실제 GPU correctness 이후에만 승인된 48 fits 범위까지 진행할 수 있다. 원자료·가중치·입력/정답 캐시는 GitHub에 올리지 않으므로 GitHub만으로 수치 재실행이 완결되는 것은 아니다.
