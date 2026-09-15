# 미래 입력 제안의 데이터 가용성 감사 — 사전 범위

- 이전 건물 120-fit 계약은 완료·종료 상태로 보존한다. 본 감사는 이후 명시된 연구 목표 아래, 2026-09-13의 미실행 제안에 필요한 정보 계약만 확인한다.
- 신규 학습 0 fits, optimizer updates 0, 모델 추론 0, GPU 사용 0. 점수와 성공 문턱을 만들지 않는다.
- 원천: OpenSTEF/liander2024-energy-forecasting-benchmark, revision dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044.
- 대상: 공식 튜토리얼의 OS Gorredijk / mv_feeder 한 개. 부하·예측 성능을 보고 선택하지 않는다. 다른 타깃으로 교체하지 않는다.
- 허용 다운로드: README, target metadata, 공식 로딩/가용시각 코드와 이 타깃의 load_measurements, weather_measurements, weather_forecasts_versioned 세 파일. 최신 예보 weather_forecasts는 사용하지 않는다. 파일당 32MiB, 전체 데이터 64MiB 한도.
- 감사 원점: UTC 2024-02-01부터 2024-12-29까지 매일 08:00. 각 원점부터 48h의 15분 valid timestamps 192개. 모델의 최종 예측 기간이나 분할을 사전 등록하는 것은 아니다.
- 선택 규칙: available_at <= origin인 행만 남기고 valid timestamp별 가장 늦은 available_at을 선택. 같은 key 중복은 보고하고 상충하면 중지. 모든 feature의 finite 여부를 검사한다.
- 숫자 성능·부하 크기·상관·oracle 이득은 계산하지 않는다. 메타데이터의 전체기간 upper/lower_limit는 학습 scale이나 타깃 선택에 사용하지 않는다.
- 독립 행 검색과 12개 원점의 as-of 선택을 대조. 원점 뒤에 가용해지는 행을 오염시켜도 선택 출력은 불변이어야 한다.
- 예보 vintage는 서로 다른 발행 시점의 예보다. 그 분산을 보정된 미래 확률분포나 독립 ensemble member로 주장하지 않는다. simulated available_at의 생성 근거가 없으면 실제 공개시각 보장을 미확인으로 남긴다.
- 결과는 데이터 감사 COMPLETE 여부와 후속 예측 비교 가능성만 구분한다. 새로운 PEFT PASS, teacher의 이득, 신규성은 모두 미평가다.
