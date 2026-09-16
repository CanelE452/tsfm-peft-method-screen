# 아홉 조건 PEFT 비교 — 전체 결과

단일 MASTER_PROTOCOL 계약으로 N01 → N02 → R05 → N06 → N03 → R04 → N07 → R08 → R09 순서로 실행한다. 서로 다른 목적을 합산한 순위는 만들지 않는다.

| ID | 주제 | 실행 | 근거 | 신규성 | 완료경로 | 업데이트 | smoke |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N01 | ASYNC | COMPLETE | NEGATIVE_WITHIN_SCOPE | UNVERIFIED_VARIANT | 16 | 8192 | 8 |
| N02 | ARCHIVE | PARTIAL | NOT_MEASURED | UNVERIFIED_VARIANT | 4 | 2048 | 8 |
| R05 | VINTAGE | PREPARED | NOT_MEASURED | KNOWN_CONTROL | 0 | 0 | 0 |
| N06 | JOINT | PREPARED | NOT_MEASURED | KNOWN_CONTROL | 0 | 0 | 0 |
| N03 | CLOCK | PREPARED | NOT_MEASURED | UNVERIFIED_VARIANT | 0 | 0 | 0 |
| R04 | REVISION | PREPARED | NOT_MEASURED | UNVERIFIED_VARIANT | 0 | 0 | 0 |
| N07 | SPECTRAL | PREPARED | NOT_MEASURED | UNVERIFIED_VARIANT | 0 | 0 | 0 |
| R08 | LEAD | PREPARED | NOT_MEASURED | UNVERIFIED_VARIANT | 0 | 0 | 0 |
| R09 | MIXED | PREPARED | NOT_MEASURED | UNVERIFIED_VARIANT | 0 | 0 | 0 |

현재 controller 상태: `RUNNING`. 새 본학습 업데이트 10286/59,392, smoke 16/96, native forwards 37710/200,000. [세부 예산](BUDGET_LEDGER.csv).

- [N01 ASYNC 한국어 보고서](N01/REPORT.md)
- [N02 ARCHIVE 한국어 보고서](N02/REPORT.md)
- [R05 VINTAGE 한국어 보고서](R05/REPORT.md)
- [N06 JOINT 한국어 보고서](N06/REPORT.md)
- [N03 CLOCK 한국어 보고서](N03/REPORT.md)
- [R04 REVISION 한국어 보고서](R04/REPORT.md)
- [N07 SPECTRAL 한국어 보고서](N07/REPORT.md)
- [R08 LEAD 한국어 보고서](R08/REPORT.md)
- [R09 MIXED 한국어 보고서](R09/REPORT.md)

[원천·노출 장부](SOURCE_AND_EXPOSURE_LEDGER.md). 학습 실험의 E는 기존 프로그램에서 노출된 개발 평가이고 R05/N06는 기존 평가 점수의 재분석이다. 기존 결과는 덮어쓰지 않는다. 전체 적응 성공과 신규성은 별도 축이다. 단순방법으로 충분하면 새학습법을 만들어 살리지 않는다.

미실행 부분은 QUEUE_STATUS와 후보 STATUS에 명시한다. 공통 안전/예산 중단 시 `scripts/with_cuda.sh .venv/bin/python scripts/run_condition_studies.py resume-all`을 쓰며 해시와 완전한epoch상태가 일치해야 한다. partial epoch/모호한update는 자동 재실행하지 않는다.
