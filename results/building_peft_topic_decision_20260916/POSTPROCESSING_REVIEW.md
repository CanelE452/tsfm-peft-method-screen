# 실험 후처리 검산기의 개발·검사 범위

[확인] finalizer는 봉인한 runtime/runner/candidate/compare 파일과 분리했다. 기존 worker가 종료된 후 한 번 실행하도록 실제 PID와 생성시각을 확인한 연결 프로세스가 대기한다. 이 연결은 새 fit·재시도·설정 선택을 수행하지 않는다.

[확인] 독립 지표·건물 bootstrap·0분모·전역 학습량 정책·최강 고정 대조·신규성 분리 검사10개를 합성 자료에서 통과했다. 개발 scalar snapshot1245개 값 최대차1.4210854715202004e-14, 과거 대조8개 episode를 재계산했다. 최종 실제 전체 검산은 각 별도 verification 파일의 완료 상태를 확인해야 한다.

[확인] 정적 검토에서 LOCKED episode ID에 요일이 포함됨을 확인해 building/days/origin으로 manifest를 참조하도록 했다. CSV 줄바꿈을 LF로 정리했다. 대조의 수치 계산은 통과했지만 검산 기록 JSON 출력에 NumPy bool 형식 오류가 한 번 발생해 Python bool로 바꾸고 동일 기록을 저장했다. 이 변경들은 후처리 개발이며 봉인 실험의 재실행·학습 수치 수정이 아니다. 실제 변경 hash와 오류는 verification_source_revision.json, publication_formatting.json, additional_verifier_checks.json, reference_audit_preflight.json, finalizer_preparation_errors.json에 보존한다.

[확인] 권한 제한 환경에서 py_compile의 bytecode cache 쓰기가 차단된 호출1회가 있었다. 파일 쓰기가 없는 AST parse로 문법 검사를 완료했다. 실험 worker와 GPU 검사는 중단되지 않았다. 이를 학습 실패로 세지 않는다.

[설계] 독립 scalar 값은 절대/상대1e-10으로 비교하고 FP32 같은 shape의 checkpoint replay는 exact equality로 검사한다. 이 수치 검산 허용오차와 SCREEN_PASS의 성능 문턱은 서로 다르다. 전자는 오류 탐지, 후자는 사전에 고정한 투자판정 기준이다. 실패 뒤 성능 문턱을 바꾸지 않는다.
