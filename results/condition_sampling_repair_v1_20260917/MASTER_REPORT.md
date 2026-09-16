# 날짜 다양성 교정 — 전체 보고서

단일 [실행 계약](MASTER_PROTOCOL.md), [저장소 감사](REPOSITORY_AUDIT.md), [GPU 전 원점 감사](ORIGIN_REPAIR_AUDIT.md).

| track | execution | fits | evidence | tags |
| --- | --- | --- | --- | --- |
| N01 | BLOCKED_DIVERSITY | 0 | NOT_MEASURED |  |
| N02 | PREPARED | 0 | NOT_MEASURED |  |
| N03 | PREPARED | 0 | NOT_MEASURED |  |
| R04 | BLOCKED_DIVERSITY | 0 | NOT_MEASURED |  |
| N07 | BLOCKED_DIVERSITY | 0 | NOT_MEASURED |  |
| R08 | BLOCKED_DIVERSITY | 0 | NOT_MEASURED |  |

| track | main_fits | main_updates | smoke_updates |
| --- | --- | --- | --- |
| N01 | 0 | 0 | 0 |
| N02 | 0 | 0 | 0 |
| N03 | 0 | 0 | 0 |
| R04 | 0 | 0 | 0 |
| N07 | 0 | 0 | 0 |
| R08 | 0 | 0 | 0 |

본학습 상한96fits/49,152updates, smoke 상한48updates, 총49,200이다. 다양성 미충족 트랙의 예산은 다른 학습에 사용하지 않는다. 실행 완료를 논문 성공으로 해석하지 않는다.

## UNAFFECTED_REFERENCE

R05/N06는 기존 저장 예측 CPU 분석이고 이번 selector와 무관하다. R05의 알려진 혼합 정책은 재사용 E 및 두 모델 비용을 포함한다. N06의 AR1 합계 CRPS 개선은 알려진 단순 결합이며 새 PEFT 근거가 아니다. R09는 기존 E64가64 index-days에 분산되어 대상 밖이다. NULL 대 MIXED의 부정적 결과와 FINE_ONLY의 고유 상세 정답16 대 MIXED64의 정보 차이를 유지한다. 세 결과를 이번 새 증거나 우승 점수로 합산하지 않았다.

## 후보별 보고서

- [N01](N01/REPORT.md)
- [N02](N02/REPORT.md)
- [N03](N03/REPORT.md)
- [R04](R04/REPORT.md)
- [N07](N07/REPORT.md)
- [R08](R08/REPORT.md)

[최종 결정](FINAL_DECISION.md). 큰 원자료·예측 배열·가중치는 로컬 ignored cache에 보존하며 GitHub만으로 수치 재생이 된다고 주장하지 않는다.
