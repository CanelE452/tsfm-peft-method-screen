# ISO-NE 자료 확보 및 검증

**공식 API 인증을 통해 2026년1~8월 243일 원본 다운로드를 완료했다.** 과거의 CSV403 및 NYISO schema 차단 기록은 당시 상태로 보존한다. 현재 ISO-NE API 자료 확보 문제는 해소됐지만, 모델 학습·예측·성능 평가는 아직 실행하지 않았다.

## 출처와 정보 권한
[공식 보고서](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/dmnd-rt-hourly-sys)의 전체 시스템 부하를 공식 API `/hourlysysload/day/{day}/location/32.json`으로 받았다. Location32=`NEPOOL AREA`, target=`Load`, 단위는 시간별 MW다. [공식 business mapping](https://www.iso-ne.com/static-assets/documents/2017/06/webservices_documentation.xlsx)과 실제 response의 시스템 식별자를 확인했다. NativeLoad/ArdDemand/zone settlement demand로 대체하지 않았다. 웹 CSV와 API의 수치별 parity는 CSV 접근이 차단되어 미확인이며, 이 한계도 남긴다.

사용자가 로컬에 설정한 인증은 공식 API에만 전달했다. 공개물에는 비밀번호·가입 이메일·Authorization header·쿠키가 없다. 원본은 로컬 ignored cache에 보존하고 공개 receipt에는 날짜·URL·시각·bytes·SHA256을 기록했다. 기존 연구 노출 감사는 부모 기록을 참조하며, 이번 접근에서 예측 성능은 계산하지 않았다.

## 기간·품질 검산

- 일별 JSON243개, 원본 시간별 행 5831개; 원본 hash 전부 확인.
- UTC 중복 0개, America/New_York offset roundtrip 불일치 0개.
- DST 전환일2026-03-08은23시간이며 실제 offset으로 UTC 변환했다.
- 계약 UTC기간의 5832시간 중 finite 5827시간, missing 5시간.
- missing 목록: 2026-01-01T00:00:00+00:00、2026-01-01T01:00:00+00:00、2026-01-01T02:00:00+00:00、2026-01-01T03:00:00+00:00、2026-01-01T04:00:00+00:00. 로컬1월1일부터 받은 자료의 시작 경계에 해당하며 보간하지 않았다.
- sigma는 TRAIN의 finite 2875시간만으로 계산했고 population std를 별도 scalar 계산으로 검산했다.
- 적법 origin: TRAIN 2300, V_SELECT 1401, E_CONFIRM 1425. 각 context512/target64는 유한하고 target전체가 split안에 있다.

## 미래 오차를 보지 않은 조건 분포

계약의 과거128시간 median/MAD와 최근32시간 median을 사용했다. S계산에 미래 y·예측 오차를 넣지 않았다.

| 고정 S bin | 원점 수 | 서로 다른 UTC날짜 |
|---|---:|---:|
| S<1 | 1058 | 55 |
| 1<=S<2 | 330 | 33 |
| 2<=S<3 | 28 | 5 |
| S>=3 | 9 | 2 |

**INSUFFICIENT_REAL_SHIFT_COVERAGE**: primary S≥3은 9원점/2일이다. 계약 최소50원점/14일과 비교한 표본 충분성 검사이며 MAG의 이득·손해가 아니다. 기준이 부족해도 threshold나 source를 바꾸지 않는다.

## 아직 하지 않은 작업
새 학습0fits/optimizer0, 예측0, 점수0이다. 이번 자료 확보를 MAG의 실제 효과 확인이나 논문 성공으로 부르지 않는다. source-specific runner 연결·고정 모델 검산·학습·V선택·E예측선저장·원점수/통계 검산은 별도 미완료다. 본 문서는 데이터 확보 단계만 보고한다.

다운로드 코드: `download_isone.py`; 검산 코드: `audit_isone.py`. 이미 받은 날짜는 hash확인 후 재사용한다. 모델 학습을 실행하는 코드는 이 다운로드 스크립트에 없다.
