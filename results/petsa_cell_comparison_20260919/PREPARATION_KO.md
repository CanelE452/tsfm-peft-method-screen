# PETSA 부품 직접 비교 준비 결과

상태: **PREPARED_NOT_AUTHORIZED — 구현·CPU 검산 완료, 추가 학습 승인 대기**. 본학습0/8회, main0/8192updates, GPU smoke0/4updates. 기존 16 fits 학습형 gate 비교는 완료 상태로 보존했다. 이 기록을 실제 PETSA 성능 결과로 사용하지 않는다.

## 완료한 작업

- 공식 GCM과 비영(0이 아닌) 가중치 상태의 출력·입력 gradient·모든 파라미터 gradient를 FP64 CPU에서 정확히 대조했다.
- 실제 Chronos-Bolt-small의 Electricity/ETTm1 B0를 CPU에서 복원했다. TRAIN4개 입력씩으로 초기 B0 동일성, 입력·출력 보정의 gradient, 동결 parameter/buffer 보존, 새 인스턴스 복원을 확인했다. optimizer update 없이 수동으로 바꾼 검사 가중치는 버렸다.
- 학습·V 선택·전체 예측 저장·채점·독립 검산·한국어 보고서·그림·scoped publication 코드를 준비했다. 실제 GPU 전체 경로가 실행됐다고 주장하지 않는다.
- 임시 합성 자료로224 prediction views, 두 primary 비교와12개 손해 비교의 scorer를 검사했다. 동일 예측의 이득·구간은0이었다. 이 합성 점수는 실제 결과 파일에 넣지 않았다.
- 기존192 prediction views, TRAIN/V, B0, 평가 표본, 의존 코드와 새 실행 코드의399개 hash를 확인했다. 과거 예측을 재추론하거나 옛 학습을 복원하지 않았다.
- 승인 파일이 없는 상태에서 runner를 호출하면 GPU 접근·학습 전에 PermissionError로 종료되고 결과 파일을 변경하지 않음을 확인했다.

## 승인 요청 범위

[고정 계약](../../experiments/petsa_cell_comparison_20260919/PROTOCOL.md): 한 대조군 PETSA_XY_OFFLINE, 두 원천, 두LR 선택과두seed 반복의 **8 fits /8192main +4smoke updates**다. MAG/B0·날짜·조건·LR·seed·판정 기준을 튜닝하지 않는다. 이 범위가 끝나면 추가 후보나 후속 학습을 연결하지 않는다.

PETSA cell19010개는 MAG8712개와 같은 크기가 아니다. 공개cell을 관측 mean/기존TRAIN sigma 좌표와9quantile에 연결한 offline 통제 비교이며, 공식 PETSA 온라인 방법 전체 재현이 아니다. 모든 E는 이전에 채점한 개발 평가다. NESO후반 기간도 새 독립 검증으로 부르지 않는다. 긍정 결과가 나오더라도 충분한 신규성·독립원천·공식전체 우위가 자동 확보되지 않는다.

GPU smoke에서 실제 업데이트·동결·복원을 검사하고 안전 조건을 통과한 뒤 본학습한다. 구현 오류나 안전 차단은 성능 실패와 구분한다. 부분 update가 애매하면 자동 재시작하지 않는다. CPU 통과는 GPU 수치 결과를 미리 보장하지 않는다.

## 보존과 재현

[CPU 실제 모델 검사](CPU_CHECKS.json), [합성 scorer 검사](SCORER_FIXTURE_CHECK.json), [현재 봉인](SEAL.json), [최초 준비 봉인](PREPARATION_SEAL_00.json), [학습 전 검사 보강 기록](PREPARATION_AMENDMENT_01.json)을 함께 보존한다. 보강은 GPU 접근 전 확인과 독립 선택 검산뿐이며 수식·표본·예산·기준은 바뀌지 않았다. 승인·학습·새 E채점 전에 이루어졌다.

CPU 검사: `.venv/bin/python -m experiments.petsa_cell_comparison_20260919.checks`.
합성 scorer 검사: `PYTHONPATH=. .venv/bin/python research/petsa_cell_readiness_20260919/check_scorer.py`.
승인 후 실행/게시 진입점: `.venv/bin/python -m experiments.petsa_cell_comparison_20260919.execute`.

실행은 사용자 승인의 정확한 메시지와 이 PROTOCOL.md의SHA, scope=PETSA_XY_OFFLINE_8_FITS, main_cap=8192, smoke_cap=4가 AUTHORIZATION.json에 기록된 경우에만 허용한다. 준비 도구가 승인 기록을 자동 생성하지 않는다. raw/weights/예측 캐시는 로컬에 있으므로 GitHub만으로 완전 재생되는 패키지는 아니다.
