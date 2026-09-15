# Query v2 — simulated tensor budget / 기존 개발 구간 재사용

**BLOCKED_GPU_BUSY. 새 GPU 수치진단0 updates, 자원측정0 updates, 본학습0/12 fits.**

실행기는 실제 진입했으나 외부 GPU compute가 계속되어 시작 안전 조건을 얻지 못했다.

누적 대기 600.19초로 고정600초 한도를 적용했다. 최소 free VRAM 727MiB, 외부 compute 관측 1112회. 다른 프로세스를 종료하거나 동시 학습하지 않았다.

## 수치 중단의 해소 여부

이번 지시문은 과거 실패를 본 뒤 새로 정한 v2 수치 정책이다. raw 출력은 Train std로 보정하고 단일 FP32 gradient/update 상대 허용치1e-4, 짧은 경로1e-3 등을 한 번 고정했다. 이전 run의1e-5 판정을 바꾸지 않았다. 새 인증 배치와 5-step 경로는 GPU 대기로 실행하지 못했으므로 NUMERICS_BOUNDED로 인증되지 않았다.

기존 저장 비교 42개를 원래 정책으로 재계산해 원 판정을 확인했다. CPU FP64의 partition/mask/gradient 검사 및 새 수치 정책 단위 검사는 통과했다. 이는 새 실제 모델 수치 인증을 대체하지 않는다.

## 자원과 예측

1/2/4/8GiB 예산별 신규 반복 측정이 없어 B*, 최속 옵션, Query 자원 이점을 정하지 않았다. 신규 V/E 점수도 없어 Standard/Side/F0 중 최강 대조군이나 예측 이득을 비교할 수 없다. 값0으로 성능을 표시하지 않는다. 기존 결과를 새 실행의 점수로 가져오지 않았다.

Q 전용 종료는 C의 성능 관문이 아니다. C는 별도600초 대기 예산으로 독립 진입했다. [통합 보고서](../priority12_20260915/REPORT.md), [원래 재검산](historical_parity_replay.json), [대기 기록](numeric_wall.json).

현재 Q 실행은 종료했으며 새 후보나 자동 재시도는 없다. 구현된 GPU 수치/자원/학습 경로는 이번 실행에서 검증되지 않은 부분으로 남는다.
