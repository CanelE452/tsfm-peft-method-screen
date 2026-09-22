# Residual feedback PEFT 두 후보 통합 결과

[확인] A와 B를 독립적으로 완료했다. 신규 offline16경로 **8192updates**, A 현장 **6144updates**, B 온라인 **12288updates**, smoke **14updates**다. A 결과로 B를 취소하거나 두 모델을 합치지 않았다. 16경로는 offline 학습만 센 값이다. A 현장 비교는 DEV/TEST를 합쳐768개의 독립 episode trajectory, B는128개의 독립 고객 stream이며 비용을 별도로 포함했다.

```text
track                label       baseline  general_static_gain  feedback_gain  time_vs_set_gain  primary_gain     seed1     seed2
    A NO_GO_CURRENT_RECIPE STATIC_LONG512            -0.598218      -0.000853          0.013032     -3.046222 -2.975281 -3.116922
    B NO_GO_CURRENT_RECIPE STATIC_LONG320             0.000000       0.000000          0.000000     -2.054057 -2.141102 -1.966585
```

각 gain은 서로 다른 질문이다. general_static_gain은 G0 이후 donor META 적응, feedback_gain은 NOERROR/FULLGEN 대비, time_vs_set_gain은 같은 정보의 일반 생성기 대비, primary_gain은 DEV에서 고정한 최강 합법 대조 대비다. 서로 더하거나 가장 좋은 항목으로 주장을 바꾸지 않는다.

- [A_AMORTIZED 한국어 보고서·그림](A/REPORT_KO.md) / [판정](A/FINAL_DECISION.md)
- [B_PARTIAL 한국어 보고서·그림](B/REPORT_KO.md) / [판정](B/FINAL_DECISION.md)
- [공통 검산](VERIFICATION.json), [독립 데이터 재검산](DATA_POSTRUN_AUDIT.json), [체크포인트·수치 재검산](CHECKPOINT_AND_METRIC_AUDIT.json), [실모델 사전검사](PREFLIGHT.json), [학습 장부](UPDATE_LEDGER.jsonl), [환경](ENVIRONMENT.json), [모델](MODEL_MAP.json), [원자료](DATA_SOURCE.json), [계열분리](SERIES_SPLIT.json), [시간분리](TIME_SPLIT.json), [reference manifest](REFERENCE_FORECAST_MANIFEST.json).

실행 정상 여부와 연구 효과를 구분했다. 사후 검산기의 잘못된 content-hash 유일성 조건을 수정했다. 같은 초기 모델의 서로 다른 방법이 동일 예측을 저장할 수 있기 때문이다. 발행 key/path 유일성과 파일 hash 일치는 계속 검사했으며 [초기 실패와 수정 근거](AUDIT_CORRECTION.json)를 보존했다. 학습·예측은 변경하지 않았다. 기존 부분정답 Maturity 규제의 FAIL도 보존한다. 새 후보는 기존 STOP을 해제하거나 추가설정을 탐색하지 않았다. 비용에는 공유 준비와 단독 사용 추정을 구분하며 optimizer 장부 I/O에 의한 시간 비교 한계를 공개했다. 신규성·논문 PASS를 선언하지 않으며 자동 후속 실험 없이 종료한다.
