# R1 Query 수치 복구와 조건부 자원 비교

상태 **INCONCLUSIVE_NUMERICS_V2**. 실제 수치 updates 84/96, 자원 updates 0/480, 본학습 0/0 attempts(상한12).

새 배치와 RMS 기준은 이전 실패를 본 뒤 사용자 지시로 고정했다. 과거120-update/180-update 결과와 판정은 그대로 보존한다. 동일 shape checkpoint와 다른 shape microbatch 검사를 분리하고, FP64 변환 뒤 parameter delta를 계산했다.

| 종류 | 정밀도 | 통과/검사 |
| --- | --- | --- |
| checkpoint | fp32 | 24/24 |
| microbatch | fp32 | 12/12 |
| microbatch | bf16 | 0/12 |

미충족 검사는 수치 무결성 미확정이며 예측 성능 실패가 아니다. 의미 오류가 입증되지 않으면 코드나 허용치를 재조정하지 않는다. [수치 원기록](numeric_checks.json), [한정 위치 추적](bounded_localization.json), [독립 재계산](numeric_verification.json).

자원 또는 본학습 미실행은 NOT_RUN이며 성능0이나 자원 이점0을 측정한 것이 아니다. 조건을 통과하면 같은 계약의 자원·학습 절차만 이어간다. 별도 후보·재튜닝은 없다.
