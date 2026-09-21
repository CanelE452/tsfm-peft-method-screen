# ISO-NE 역사 기간 확대: coverage-only 봉인 계약

봉인 시점: 2026-09-21. 기준 commit: b3761e51ff851961676fcf90dc48e1d6e2c5d221.
이번 사용자 요청은 **자료 확보·품질·고정 past-only 조건 빈도**에 한정한다. optimizer/model prediction/target error scoring은 모두0이고, 충족하더라도 학습을 자동 시작하지 않는다.

## 범위 및 역사 제공 감사

동일한 ISO-NE Hourly Real-Time System Demand / 공식 API `hourlysysload/day/{day}/location/32.json` / NEPOOL AREA / `Load`만 사용한다. 공식 business document는 rolling7년을 안내하며 service info에는 latest만 있고 earliest 필드가 없다. 목표 시작일2020-01-01 API200,24시간,00:00~23:00 EST를 직접 확인했다. 따라서 요청한2020-01-01~2026-08-31의 전체 범위를 고정한다. 2020년 이전 자료가 전혀 없다고 주장하지 않는다. 완전성은 각 날짜 전체 다운로드 후 감사하며 시작/끝을 유리한 조건 빈도로 이동하지 않는다.

원본 API query date는 America/New_York 현지 날짜다. 통계·split은 기존과 같은 UTC다. 현지2020-01-01 이전 자료를 추가하지 않으므로 UTC2020-01-01의 최초5시간이 없을 수 있다. 이를 missing으로 남기고512시간 context와64시간 target이 모두 유한한 원점만 사용한다. 현지2026-08-31의 UTC9월1일 시간들은 원본에는 보존하지만 분석 구간에서 제외한다. DST fold/gap은 원본 offset과 America/New_York 규칙을 검산해 처리한다.

## 고정 temporal split (모두 UTC, 왼쪽 포함/오른쪽 제외)

| split | 시작 | 종료 |
|---|---|---|
| TRAIN |2020-01-01 00:00|2023-01-01 00:00|
| V_SELECT |2023-01-01 00:00|2024-01-01 00:00|
| E_CONFIRM |2024-01-01 00:00|2026-09-01 00:00|

이 calendar boundary는2020-01-01의 제공 여부만 확인한 뒤 봉인했다. 확대 기간의 S빈도나 성능을 보고 정하지 않았다. 이후 수정하지 않는다. TRAIN 약3년, 다음 완전한1년V, 나머지후기E다. 각 target64시간 전체가 해당 split 안에 있어야 하고 context512시간은 이전 split으로 넘어갈 수 있다. missing은 보간하지 않는다. 중복 충돌·시간대 오류를 임의로 수정하거나 잘못된 데이터를 합산하지 않는다.

sigma_train = 위 TRAIN의 finite raw Load에 대한 population std(ddof=0). 예전2026년TRAIN sigma를 가져오지 않고 새로 봉인한 초기TRAIN에서만 계산한다. 이는 동일한 TRAIN-only 정의의 적용이며 threshold나 score 변경이 아니다.

## 고정 past-only score

예측 origin 직전 실제512시간 x에 대해:
- reference = x[-160:-32] (128시간)
- recent = x[-32:] (32시간)
- m_ref = median(reference), m_recent = median(recent)
- r_ref = max(1.4826 * median(abs(reference-m_ref)), 0.1*sigma_train)
- S = abs(m_recent-m_ref)/r_ref

기존 `audit_isone.shift_score`를 그대로 호출한다. 새 threshold/조건/모델/gate/MAG를 만들지 않는다. MAG threshold3도 변경하지 않는다. 미래64시간은 유한성 및 split 적법성 확인에만 사용하고 S/정답오차/미래지속 판정에는 넣지 않는다.

E_CONFIRM의 모든 적법 hourly origin을 대상으로 S<1 / 1<=S<2 / 2<=S<3 / S>=3의 네 bin만 계산한다. 원점 수와 unique UTC dates를 보고하며 target overlap을 독립 표본으로 해석하지 않는다. 전체 결과 외 연도별 표는 같은 고정E 전체의 기술적 분해이며 기간선택에 사용하지 않는다.

## 고정 판단 및 종료

S>=3 origins>=50 AND unique UTC dates>=14이면 `READY_FOR_REAL_CONFIRMATION_TRAINING`으로 기록하고 멈춘다. 이는 coverage충족일 뿐 모델검증·성능성공·학습승인이 아니다. 수년 확대 후에도 미달이면 `REAL_SHIFT_CONDITION_TOO_RARE_FOR_PRIMARY_STUDY`로 기록하고, 현재 실제 변화 조건을 primary로 삼아 MAG주력 방법론 연구를 계속할 타당성이 낮다는 판단을 COVERAGE_DECISION.md에 별도로 적는다. 다운로드/품질오류로 확인불가능하면 기술적BLOCKED로 구분하고 결측을 조건부재로 바꾸지 않는다.

기존 b3761e5의2026Jan-Aug9원점/2일 결과는 `INCONCLUSIVE_DATA_COVERAGE`이며 성능실패가 아니다. 기존결과파일과243개원본 및 UTC변환파일의hash를보존한다. 이미받은2026자료는 hash검증 후 재사용하며 덮어쓰지 않는다. 신규자료는 별도 cache에 저장한다.

예산: optimizer0, modelprediction0, targeterrorscoring0, GPU사용0, 자동학습0. 기존모델학습스크립트를호출하지않는다. 최종산출물: 날짜별receipt/hash, availability/quality/DST/duplicate/missing감사, legalorigin목록, 네S-bin·연도별분포, 독립scalar검산, 한국어REPORT.md/COVERAGE_DECISION.md. 자격증명·원본은Git에서제외하고scopedcommit/push한다.
