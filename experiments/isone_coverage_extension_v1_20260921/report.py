"""Korean coverage decision; stop without starting any model operation."""
from .common import *
import pandas as pd


def main():
    check_seal(); preserved = check_parent()
    s = read(OUT/'COVERAGE_SUMMARY.json')
    v = read(OUT/'VERIFICATION.json')
    q = read(OUT/'DATA_QUALITY.json')
    primary = s['bins'][-1]
    ready = s['status'] == 'READY_FOR_REAL_CONFIRMATION_TRAINING'
    bins = '\n'.join(f"| {r['bin']} | {r['origins']:,} | {r['unique_utc_dates']:,} | {100*r['origin_fraction']:.3f}% |" for r in s['bins'])
    annual = pd.read_csv(OUT/'ANNUAL_BIN_COUNTS.csv')
    annual_text = '\n'.join(f"| {r.year} | {r.bin} | {r.origins:,} | {r.unique_utc_dates:,} |" for r in annual.itertuples(index=False))
    rationale = (
        '고정 primary 조건의 원점 수와 날짜 수가 두 최소 요건을 모두 충족했다. 이는 같은 ISO-NE에서 실제 조건 비교를 구성할 표본이 있다는 판단이며, MAG의 예측 이득이나 일반 adapter 대비 우위를 뜻하지 않는다. 겹치는 64시간 target과 인접 원점은 독립 사건이 아니므로 향후 성능 비교에서도 상관을 고려해야 한다. 지금은 coverage 검사만 승인됐으므로 모델 검산·학습·예측·채점을 시작하지 않고 종료한다.'
        if ready else
        '기간을 수년으로 확대해도 고정 primary 조건의 최소 원점 수 또는 날짜 수를 충족하지 못했다. 따라서 이 조건을 primary로 삼아 현재 MAG를 주력 방법론 연구로 계속하는 타당성이 낮다. 이 판단은 실제 조건의 희소성에 관한 것이며, MAG의 예측 성능이 나쁘다는 증거나 모든 robust PEFT의 반증이 아니다. threshold·source·gate를 바꾸어 주제를 살리거나 추가 학습을 시작하지 않고 종료한다.'
    )
    (OUT/'COVERAGE_DECISION.md').write_text(f'''# Coverage 판단

**{s['status']}**

- 고정 E_CONFIRM: 2024-01-01 00:00 UTC 이상 ~ 2026-09-01 00:00 UTC 미만.
- S≥3: **{primary['origins']:,} origins / {primary['unique_utc_dates']:,} unique UTC dates**.
- 요구조건: origins≥50 AND unique UTC dates≥14.
- optimizer 0 / model prediction 0 / target error scoring 0 / 자동 학습 0.

{rationale}

기존 b3761e5의 2026년 7~8월 E 결과 9 origins / 2 days는 `INCONCLUSIVE_DATA_COVERAGE`로 보존한다. 이번 확대 결과와 원래 결과는 E 기간 및 TRAIN sigma가 다르므로 동일 표본의 성능 개선으로 표현하지 않는다. 새 source·MAG·threshold는 만들지 않았다.
''')
    (OUT/'REPORT.md').write_text(f'''# 동일 ISO-NE 역사 확장: 실제 변화 조건의 발생 빈도

**{s['status']} — S≥3은 {primary['origins']:,}원점 / {primary['unique_utc_dates']:,} UTC일이다.** 이번 작업은 모델 없이 수행한 데이터 품질·조건 빈도 검사다.

## 보존과 사전 봉인

기준 b3761e5의 2026 Jan–Aug 획득 결과, 243개 원본, 기존 UTC 파일을 포함한 **{preserved}개 파일의 hash가 모두 일치**한다. 기존 9원점/2일은 성능 실패가 아니라 `INCONCLUSIVE_DATA_COVERAGE`다. 기존 파일을 수정하거나 삭제하지 않았다.

2020-01-01의 공식 API 200 응답과 24시간 자료를 확인한 뒤, 확대 S 분포를 계산하기 전에 [프로토콜](COVERAGE_EXTENSION_PROTOCOL.md)을 봉인했다. 문서 SHA256은 `{s['protocol_sha256']}`다.

| split | 시작 UTC | 종료 UTC (미포함) |
|---|---|---|
| TRAIN |2020-01-01|2023-01-01|
| V_SELECT |2023-01-01|2024-01-01|
| E_CONFIRM |2024-01-01|2026-09-01|

이 경계를 이후 변경하지 않았다. 계절·S 빈도·성능을 보고 특정 기간을 고르지 않았다.

## 공식 자료 범위와 품질

동일한 [ISO-NE 공식 hourly system demand API](https://webservices.iso-ne.com/docs/v1.1/rest.hourlysysload.day.day.location.locationId.html)의 location 32, NEPOOL AREA, `Load` 필드다. [공식 business documentation](https://www.iso-ne.com/static-assets/documents/2017/06/webservices_documentation.xlsx)은 rolling 7년을 안내한다. service info에는 latest만 있고 earliest는 명시되지 않는다. 요청한 **2020-01-01~2026-08-31의 {q['daily_files']:,}개 현지 날짜**를 모두 확보했다. 목표 시작일이 제공됨을 확인한 것이며, 2020년이 API의 절대 최초 연도라는 뜻은 아니다.

- 원본 {q['raw_rows']:,}행, 분석 UTC grid {s['utc_grid_hours']:,}시간.
- finite {s['finite_utc_hours']:,}시간, missing {s['missing_utc_hours']:,}시간. 누락은 보간하지 않았다.
- 원본 중복 관련 행 {q['duplicate_rows']}개: 모든 필드가 같은 복사본 {q['exact_duplicate_copies_collapsed']}개만 정규 시계열에서 제거했다. 원본은 보존했고 충돌값은 없었다.
- DST 규칙 불일치 {len(q['dst_mismatches'])}, 일별 unexpected hour {q['unexpected_hours']}.
- 23/25시간 DST 전환일을 원본 offset과 America/New_York 규칙으로 검사했다.

현지 2020년 1월 1일부터 받은 자료는 UTC 시작 경계의 최초 5시간을 포함하지 않는다. 원본에는 보존된 UTC 2026년 9월 1일 시간은 분석에서 제외했다. 누락 시각 전체는 `MISSING_UTC_HOURS.csv`, 일별 검사는 `DAILY_QUALITY.csv`, schema 및 DST 감사는 `DATA_QUALITY.json`에 기록했다. 추가 필드의 schema 차이는 기록하되, system ID와 Load를 확인해 native/zone 수요로 대체하지 않았다.

2026년의 243일은 기존 hash를 확인해 재사용했다. 새 원본은 별도 cache에 저장하고, URL·시각·bytes·SHA256을 공개했다. 원본과 인증 정보는 Git에서 제외했다. API와 접근이 차단된 웹 CSV의 직접 수치 parity는 미확인이라는 기존 한계도 유지한다.

## 적법 원점과 고정 score

| split | 적법 hourly origins |
|---|---:|
| TRAIN |{s['legal_origins']['TRAIN']:,}|
| V_SELECT |{s['legal_origins']['V_SELECT']:,}|
| E_CONFIRM |{s['legal_origins']['E_CONFIRM']:,}|

context 512시간과 target 64시간이 모두 finite이고, target 전체가 split 안에 있는 원점만 사용했다. context는 앞 split의 과거를 사용할 수 있다. 미래값은 target 유한성 확인에만 접근했고, 모델 예측이나 target 오차는 계산하지 않았다.

기존 reference128/recent32, median/MAD, floor0.1sigmaTRAIN, S≥3을 그대로 사용했다. MAG threshold3도 수정하지 않았다. TRAIN sigma는 **{s['sigma_train_population']:.12g}**이며, 초기 TRAIN의 finite {s['sigma_train_finite_hours']:,}시간에서 population std로 계산했다. 같은 TRAIN-only 정의를 새로 봉인한 기간에 적용했으므로, 예전 2026년 TRAIN sigma와 수치가 다를 수 있지만 수동으로 조정한 것이 아니다.

| 고정 S bin | origins | unique UTC dates | E origin 비율 |
|---|---:|---:|---:|
{bins}

서로 다른 bin의 날짜는 겹칠 수 있으므로 날짜 열은 합산하지 않는다. 전체 E의 원점 날짜는 {s['E_unique_dates']:,}일이다. 동일 target hour의 중복은 최대 {s['target_overlap']['max_origin_coverage_per_target_hour']}회이며, 원점 수를 독립 사건 수로 해석하지 않는다. 이 조건은 과거 관측으로 정의한 high-shift stratum이지 실제 운영 사건의 원인 label은 아니다.

## 동일 E의 연도별 분해

| 연도 | S bin | origins | unique UTC dates |
|---|---|---:|---:|
{annual_text}

유리한 연도만 선택하지 않고 고정 E 전체로 판단했다. 2024/2025년은 완전한 연도이고, 2026년은 1~8월이다. 2026년 원자료와 짧은 E의 coverage는 이미 확인했으며 이번 확장은 그 표본 부족을 계기로 설계했다. 전체를 완전히 미열람한 독립 확증 자료라고 표현하지 않는다.

## 독립 검산과 판단

첫 품질 검사에서 중복을 일괄 차단했으나, 원본의 모든 필드가 동일한 반복 행임을 확인해 동일 시간 관측 하나로 표현했다. 이전 검사 결과는 `quality_attempt_1/`에 보존했다. UTC grid 생성의 naive/aware 타입 불일치도 수정했으며, 두 조치는 S 계산 전에 이루어졌고 봉인한 경계·score·판정 기준은 바꾸지 않았다.

모든 {q['daily_files']:,}개 일별 파일 hash와 부모 {preserved}개 hash를 확인했다. E의 {v['independent_scalar_E_scores']:,}개 S값을 `statistics.median`과 scalar 표준편차로 독립 재계산했고, bin 및 날짜 수가 일치했다. 모든 {v['direct_legal_origin_checks']:,}개 후보 원점의 적법성을 별도 직접 finite-window 계산으로 재검산했다. protocol과 기존 score 코드 hash는 바뀌지 않았다.

{rationale}

실제 **optimizer 0, model prediction 0, target error scoring 0, GPU 모델 초기화 0**이다. B0/PLAIN/MAG/OUTPUT_CONTEXT는 어느 것도 실행하지 않았다. [COVERAGE_DECISION.md](COVERAGE_DECISION.md)에 이번 연구 판단을 기록하고 멈췄다. 후속 학습·새 후보·source 추가·threshold 변경은 시작하지 않았다.
''')
    save(OUT/'FINAL_STATUS.json',{'status':s['status'],'coverage_only_complete':True,'training_authorized_by_this_task':False,'optimizer_updates':0,'model_predictions':0,'target_error_scores':0,'automatic_successors':0,'parent_files_unchanged':True})
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!='PUBLICATION_AUDIT.json']
    save(OUT/'PUBLICATION_AUDIT.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in files},'raw_data_local_only':True,'credentials_not_published':True})
    print(s['status'],primary)

if __name__=='__main__':main()
