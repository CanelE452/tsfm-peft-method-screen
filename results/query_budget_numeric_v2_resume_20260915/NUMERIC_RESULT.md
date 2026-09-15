# Query v2 재개 — 수치 진단 종료

**180 disposable updates, INCONCLUSIVE_NUMERICS_V2. 자원 측정0, 본학습0/12 fits.**

| 검사 | FP32 통과/전체 | BF16 통과/전체 |
| --- | --- | --- |
| 단일 업데이트 | 9/12 | 0/12 |
| 5-step 경로 | 5/6 | 2/6 |

origin 그룹·frozen hash·same-shape replay 검사는 통과했다. FP32에서 pinball 부호 뒤집힘이 없는 비교에서도 Standard gradient/update 차이가 기준을 넘었다. 따라서 pinball kink만으로 모든 차이를 설명할 수 없다. BF16에서는 모든 단일 비교가 기준을 넘었고 부호 뒤집힘도 관측됐다. 그 출력 미분 변화의 상한을 별도 기록했으며 파라미터 gradient 차이에 대한 인과 분해를 완료했다고 하지 않는다.

Electricity Standard의 FP32 5-step total-delta relative-L2는 0.001180421로 지정1e-3을 넘었다. BF16 경로에서도 scaled output 차이가 지정0.05를 넘는 경우가 있었다. raw 단위 기준만 바꾸어 끝낼 문제가 아니며 이번 run에서는 허용치·정밀도·학습 설정을 추가 변경하지 않는다.

v2는 옛 실패를 본 후 사용자가 고정한 정책 개정이다. 기존 v1 판정은 불변이다. 새 인증 쌍과 장부는 [계약](contract.json), [단일 검사](numeric_checks.json), [연속 검사](path_checks.json), [그룹/불변 검사](leakage_checks.json), [독립 검증](numeric_verification.json)에 있다. 본 예측 비교는 미실행이므로 예측 성능 FAIL이나 Query의 전체 연구 가치를 판정하지 않는다.

Q는 종료했고 C는 성능과 독립적으로 진행한다. RustDesk만 사용자가 예외 허용했으며 GPU 사용량은 원본 로그에 남았다. 완전 유휴 GPU 시간 측정이라고 주장하지 않는다.
