# wind 전용 재개 실행기

## 고치는 문제
원래 run.py는 패키지 manifest만으로 만든 .started를 한 번 생성하고, 설치 대상 experiments 폴더도 새로 만들어야 했다.
따라서 데이터 경로가 나중에 확보돼도 원래 명령은 재실행할 수 없다. 잠금만 지워도 source 폴더 존재 검사에 걸린다.
이 실행기는 기존 잠금과 폴더를 수정하는 대신, 해시가 확인된 기존 triage.worker를 wind 한정으로 호출한다.

## 변경되지 않는 것
기존 source 22개 파일을 해시로 검사한다(원 manifest 포함).
wind.py, predictors.py, statistics.py, engine.py, RUN_CONFIG.json은 한 바이트도 수정하지 않는다.
풍력 원시 관측 6시간 -> 미래 발전량 1시간, 원래 터빈/날짜 분할/난이도 점수/선택 기준/모델/정규화 설정을 유지한다.
명령행 wind CSV 경로와 위치 파일 경로만 기존 입력 인터페이스로 제공한다.
respiration과 coordinates는 호출하지 않는다. 두 결과를 새 결과처럼 재게시하지 않는다.

## 실행과 증거
- 기존 .venv를 사용하며 설치/모델 다운로드/데이터 다운로드를 하지 않는다.
- 원래 고정 Figshare 메타데이터 캐시에서 정확한 이름/크기/MD5를 대조한다. 캐시가 없으면 중단한다.
- 원래 wind 결과가 데이터 선정 단계에서 차단됐고, 평가 점수/봉인이 생성되지 않았는지 확인한다.
- 기존 패키지·모든 기존 결과·원래 잠금을 실행 전후 검산한다.
- wind 작업은 900초 한 번. 같은 새 승인으로 중복 실행은 거절한다.
- 새 실행기의 사전 실패는 .cache/wind_resume_v1_20260925/LAST_PRECONDITION_FAILURE.json에 기록한다.
- 작업 시작 이후 기술 실패도 안전한 Git 상태에서 새 결과·로그로 게시한다.
- 소스나 이전 기록의 무결성이 달라지면 게시도 거부한다.

## 결과 경로
results/wind_resume_v1_20260925/run_<UTC>/wind/RESULT.json
results/wind_resume_v1_20260925/run_<UTC>/WIND_ONLY.json
results/wind_resume_v1_20260925/run_<UTC>/INPUT_VERIFICATION.json
results/wind_resume_v1_20260925/run_<UTC>/RUN_MANIFEST.json
.cache/wind_resume_v1_20260925/EXECUTION_RECEIPT.json

원자료는 .cache 내부의 Git 비추적 파일이다. .cache가 저장소 디렉터리 밖이라는 뜻은 아니다.

## 한계
새 실행기는 데이터 획득 뒤의 재개만 허용한다. 지표의 과학적 타당성이나 모델 성능을 개선하는 수정이 아니다.
MD5 대조는 원래 내려받은 공식 메타데이터 캐시 기준이다. 작성 환경에서는 현재 Figshare 서버를 재검증하지 못했다.
작성 환경에는 native Chronos가 없고 사용자의 334MB 원자료도 없으므로 wind 실제 분석은 여기서 재실행하지 않았다.
모의 데이터/임시 Git 검증과 실제 연구 결과는 AUTHOR_VALIDATION.json에서 구분한다.
