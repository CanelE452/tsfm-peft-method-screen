# 고정 MAG 실제 전력 부하 확인 — 데이터 단계 차단

## 1. 질문
실제 관측의 past-defined high-shift 구간에서 MAG가 일반 second-stage adapter보다 유용한가?

## 2. 데이터 독립성과 확보 상태
기준 local/remote main은 `40a80f6ddf0d69978a51051bc95aaa6e8a473554`다. 저장소의 기존 tracked text와 로컬 results/research 및 cache 파일명 검색에서 NYISO/ISO-NE의 이전 성능 사용 흔적을 찾지 못했다. 이는 기록 밖의 노출까지 없었다는 증명은 아니다. 부모 보고서·학습 코드·모델 코드 hash 및 pinned Chronos-Bolt-small snapshot hash를 확인했다.

[NYISO 공식 archive](https://mis.nyiso.com/public/P-58Blist.htm)의 완료된2026년1~8월 ZIP8개와 일별 CSV243개, 총785,334행을 검사했다. 컬럼은 Time Stamp / Time Zone / Name / PTID / Load다. 11개 지역만 있고 공식 system-total 행이 없어 `BLOCKED_SCHEMA_NO_SYSTEM_TOTAL`이다. N.Y.C.는 뉴욕시 지역이며 시스템 전체로 대체하지 않았다. timestamp/entity중복0, 누락·비유한값0, EST/EDT를 반영한 UTC↔America/New_York roundtrip 불일치0이다. 개별 sampling interval 빈도도 schema audit에 기록했다. 이 품질 검사는 모델 성능 평가가 아니다.

NYISO 예측·target score0회에서 [ISO-NE 공식 Hourly Real-Time System Demand](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/dmnd-rt-hourly-sys)를 확인했다. 페이지200응답에는 과거 다운로드 CAPTCHA가 있고, 동일 보고서의2026-01-01 공식 CSV 요청은403이었다. fallback은 `BLOCKED_DOWNLOAD_ACCESS`다. 인증/접근 제어를 우회하지 않았으며 다른 데이터 제품이나 provider로 대체하지 않았다.

## 3. Primary 결과
**미실행.** S>=3의 MAG 대 PLAIN 원점수, seed92101/92102 이득, block interval은 모두 N/A다. 적합한 시스템 시계열을 확보하지 못했으므로 S>=3 원점 수/날짜 수 자체도 계산하지 않았다. 이는 효과0이나 coverage 부족을 관측했다는 뜻이 아니다.

## 4. 전체 raw trade-off
F0/B0/B0_PLAIN/B0_MAG/OUTPUT_CONTEXT의 실제 E 예측과 점수 모두 N/A다. synthetic 자료로 대신 평가하지 않았다. 실제 raw 관측에서의 MAG 추가 가치는 이번 실행으로 확인되지 않았다.

## 5. 조건별 결과
S<1, 1≤S<2, 2≤S<3, S≥3의 네 bin 모두 미실행이다. 빈 CSV는 점수0이 아니라 NOT_RUN이다.

## 6. 비용과 미실행 범위
새 학습 **0/16 fits**, main **0/16,384 updates**, smoke **0/8 updates**, 모델 예측0회, GPU 학습 시간0이다. source 전처리·origin 선정·모델 검산·학습·선택·raw E 평가·통계가 남았으며, 차단 종료 때문에 실행하지 않았다.

| 방법 | first-stage params | second-stage params | deployed adaptation params | 적응 단계 | 실제 새 updates |
|---|---:|---:|---:|---:|---:|
| F0 |0|0|0|0|0|
| B0 |294912|0|294912|1|0|
| B0_PLAIN |294912|8712|303624|2|0|
| B0_MAG |294912|8712|303624|2|0|
| OUTPUT_CONTEXT |294912|4673|299585|2|0|

파라미터 표는 고정된 기존 구현의 구성값이고 이번 원천에서 모델을 만들고 측정한 결과가 아니다. 다운로드 archive와 응답 원문은 로컬 ignored cache에 있으며 공개 receipt/hash만으로 원자료가 GitHub에 포함됐다고 주장하지 않는다.

## 7. 판단
**BLOCKED_NO_INDEPENDENT_REAL_LOAD_SOURCE**

NYISO는 계약의 전체 시스템 합계가 없고 ISO-NE는 공식 과거 CSV 접근이 차단됐다. 따라서 MAG의 실제 성능 실패나 계속 개발 근거를 판단할 수 없다. 기존 synthetic 양성 결과는 development stress-test evidence로 보존한다. MAG 구조·threshold·rank·seed·source를 바꿔 rescue하지 않는다. 새 후보 및 자동 후속 학습은0이다.

## 논문 주장 경계
현재 주장 범위는 already-adapted forecaster의 작은 second-stage residual에 대한 magnitude-aware restriction의 조건부 추가 가치다. 이번 데이터 차단은 이를 강화하거나 반증하지 않는다. Time-PEFT의 joint LoRA/frequency/channel, MSFT의 multi-scale, AdaPTS의 multivariate 설정 전체와 직접 비교하지 않았다. δ-Adapter/COSA의 frozen predictor 보강은 관련 큰 설정이지만 정보 권한과 개입 위치가 다르고 OUTPUT_CONTEXT는 공식 전체 재현이 아니다. 독립 실제 자료 성능과 정식 선행 우위는 여전히 미확인이다.
