# 예보 vintage 참조 비교 — 유한 개발 실험

2026-09-16. 기준 311bbfc. 이전 건물 120-fit 계약은 종료 상태로 유지한다. 사용자 후속 연구 목표에 따른 별도 참조 검증이며 새 PEFT 방법 또는 실제 ensemble 재현이 아니다. 부하/예측 성능을 읽기 전에 범위를 고정한다.

## 정보와 데이터

- 이전 가용성 감사의 pinned Liander revision dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044와 OS Gorredijk 한 타깃만 사용한다. 다른 타깃·데이터로 교체하지 않는다.
- SIMULATED_AS_OF, SINGLE_SOURCE_DISCOVERY. available_at은 실제 발행 로그로 검증되지 않았다. 동일 발행 cycle의 확률 ensemble을 가정하지 않는다. 결과는 equal-weight vintage-path mixture의 경험적 가치만 설명한다.
- 15분 값 네 개를 해당 시각으로 시작하는 한 시간의 평균으로 집계한다. 원점 o의 context는 [o-336h,o), target은 [o,o+24h). 모든 context 원시값의 available_at <= o를 검사한다. 평균은 네 값이 모두 finite일 때만 유효하다.
- 기상 입력은 temperature_2m, wind_speed_10m, shortwave_radiation 세 개. 수요·풍력·태양광과 관련한 대표 연속 변수라는 설계 선택이며 상관/점수로 고르지 않는다. 나머지 변수를 조사해 대체하지 않는다.
- 과거 기상도 versioned 예보에서 원점 이전에 가용한 최신 행만 사용한다. 공개시각이 없는 weather_measurements, 실제 미래 기상, 가격, 타깃 metadata upper/lower_limit는 사용하지 않는다.
- 미래 각 15분 시점에서 available_at<=o인 최신 네 행을 시간 역순으로 뽑아 k=0,1,2,3 경로를 만든다. 각 경로를 시간별 평균한다. 한 경로가 단일 기상 발행 run이라는 보장은 없다. 네 행 부족/입력 또는 target 결측이면 해당 원점을 임의 제거하지 않고 데이터 조건 미충족으로 종료한다.
- 보정 C: 2024년 3·4·5월 각 1·8·15·22일 UTC08시, 총12원점. 개발 D: 6·7·8·9월 같은 네 날짜·시간, 총16원점. 전체28원점으로 달력만 사용한 고정 분리. 기존 감사로 결측 위치는 노출됐으나 부하 예측 점수는 미열람이다. 10~12월은 이번 점수 계산에 사용하지 않는다. 모든 데이터는 한 타깃·한 원천이다.

## 예측과 비교군

Chronos-2 revision29ec3766d36d6f73f0696f85560a422f50e8498c, 기존 local weights, FP32/eval/dropout0/TF32off, context336/horizon24, seed61700. cross_learning=False. batch_size16(타깃+공변량 수 기준)로 네 경로를 한 번에 처리한다. 기본 native21분위수와 raw output을 보관하고 모든 비교에서 분위수 축 정렬을 공통 적용한다. 분포의 전체 joint law는 주장하지 않는다.

- F0: 부하 context만.
- PAST: 같은 부하+과거 기상, 미래 기상 없음.
- LATEST: 같은 과거 입력+가용한 최신 미래 경로(k0). PATHS 배치의 첫 결과 재사용.
- MEAN_INPUT: 같은 과거 입력+네 미래 경로의 산술평균.
- MIXTURE: 네 경로 각각의 조건부 예측분포를 동일 가중 CDF 혼합. 분위수 평균이 아니다.
- 각 F0/PAST/LATEST/MEAN_INPUT/MIXTURE에 같은 C-only 편향·폭 보정 기회를 추가한다. b=median((y-q50)/context_std), s∈{0.5,0.75,1,1.25,1.5,2}. q'=q50+b*std+s*(q-q50). C primary 최소 s, 동률 작은 s. 원본과 보정본 모두 보고한다. 보정 추정5건/후보 비교30개이며 neural fits에 섞지 않는다.
- SEASONAL24/168: 해당 lag의 context 반복 예측에 C의 (y-pred)/std 전체 잔차의 empirical quantiles(np.quantile, linear)를 더해 확률 대조를 만든다. 보정 추정2건. 알려진 단순 대조다.

MIXTURE 분포 정의: 정렬된 native q(tau)의 분위수 함수를 probability 축에서 선형 보간하고 양끝은 q(.01),q(.99) 상수로 확장한다. 이 명시적 유계 분포들의 CDF를 평균해 64-step 이분법으로 generalized inverse를 구한다. 꼬리 모델링을 새 방법 기여로 쓰지 않는다. 동일분포 불변·분리된 uniform 혼합·상수분포 원자 질량을 검사한다. 평가 원점의 scale은 context population std floor1e-6. 모든 원점 같은 비중.

Primary: native21 tau에서 평균 2-pinball / context_std. 이것을 연속 CRPS 정확값이라 부르지 않는다. Secondary: raw2-pinball·medianRMSE/MAE·.1-.9포함률/폭, 월별 원점수, 비교 상대개선. 원점별 저장 예측을 독립 scalar float64로 검산(절대/상대1e-10). 보정은 C만 읽어 봉인한 뒤 D를 채점한다.

이번은 PEFT PASS 판정이 아니다. MIXTURE_CAL이 D에서 강한 비-mixture 고정 대조 각각보다 평균 primary>=2% 좋고 4개월 중3개월 이상 방향이 좋으면 TEACHER_SIGNAL_TO_INVESTIGATE, 양성이나 불충족이면 POSITIVE_BUT_UNCERTAIN, 단순 대조보다 나쁘면 NO_ADDED_VALUE_IN_THIS_REFERENCE. 2%는 이번 단계 투자 기준이고 결과를 보고 바꾸지 않는다. 신뢰구간은4개월을 묶음으로 paired bootstrap2000(seed61701), 탐색적이고 독립 재현이 아니다. teacher가 좋더라도 일반 LoRA 및 동일정보 증류 대조는 아직 미실행이다.

## 예산·안전·검증

신규 neural fits0/optimizer updates0. 28원점×7 target forecasts=196(4PATHS+MEAN+PAST+F0). 기본 pipeline calls112. 첫 C원점 네 호출 재생7 forecasts/4calls. 자원 probe는 첫 C입력의 LATEST와4PATHS, 각warmup1+측정5=30forecasts/12calls. 합계 최대233target forecasts/128pipeline calls. primitive forward hook 실제 횟수도 기록한다. 비용 비교는 동일 batch16에서 네 경로 병렬 처리 대비 단일경로이며 4배 속도 이득을 가정하지 않는다. 보정·혼합CPU시간은 별도 기록한다.

단일GPU/worker, 기존lock/Watch와 RustDesk만 예외 재사용. 시작30초안정/free4GiB, 실행free1GiB, 외부 compute는 정지경계에서 대기, 총대기600초/controllerwall3600초. OOM/nonfinite/예산 초과에 batch·기간·경로수·설정을 바꾸지 않는다. 준비/API 오류 한 번 수정 가능, 시도 원장 보존. observer timeout은 job 재시작 사유가 아니다.

데이터와 프로토콜·코드·모델 receipt hash 봉인, 미래 부하 poison에 입력 불변, 시간 경계·네값 집계 검사, frozen weights/buffers hash 불변, 첫 C전체 출력 재생(동일shape FP32 normalized abs<=1e-6), 원점별 저장·원점수 독립 검산. 재생은 bitwise 여부도 함께 보고한다. 기존 results/research hash를 보존한다. 종료 시 실제횟수·미실행·오류·원점수·자원·신규성 한계를 한국어 REPORT.md로 작성하고 push한다. 자동 학생학습이나 별도 후보 추가 없음.
