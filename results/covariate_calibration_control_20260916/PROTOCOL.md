# 예보 버전 LoRA 이득의 단순 출력 보정 대조

기준 e9c9859, 2026-09-16. 기존 6-fit 실행과 원래 건물120-fit 계약은 종료된 기록으로 보존한다. 이번 단위는 사용자의 후속 연구 목표 아래 저장 예측으로 경쟁 설명을 점검하는 제한 분석이며 원래 계약의 추가 학습이 아니다.

질문: VINTAGE fixed120의 이득이 V에서 구한 간단한 편향·분위수 폭 보정 뒤에도 남는가? 이것은 새 PEFT 후보 또는 독립 검증이 아니다. 기존 V4/D16 모두 노출된 DISCOVERY 자료다. 계산 전 범위를 고정하되 사전 무노출 가설이라고 쓰지 않는다.

- Neural fits/updates/GPU inference=0. 기존6fits 재사용, 모든 군 FIXED120만. 기존 checkpoint 선택 변경 없음.
- 군={INIT0 1개, STD/EXODROP/VINTAGE 각seed61710/61711}, 총7개.
- V=5월4원점 최신k0, D=6~9월16원점 k0/k3 각각. 새 날짜·타깃·원천 없음.
- 각 군에 같은 affine quantile 보정: b=median((y-q50)/context_std), pooled V96시점에서 계산. s∈{0.5,0.75,1,1.25,1.5,2}, V4 평균 primary 최소 선택, 동률 작은s. q'=q50+b·context_std+s·(q-q50).
- b는 중앙값 잔차 기준의 고정 추정이며 b/s를 primary에 대해 연속 최적화했다고 주장하지 않는다. s는 앞선 vintage-reference 보정의 동일 grid다. 추가 값, clip, 월별/경로별 계수, D 기반 재선택 없음.
- CPU 통계 보정기7개, scale 후보42개·V 원점 채점168회. 모든 계수를 봉인한 뒤 D 원점수 계산. RAW와CAL 둘 다224행, 총448행. 기존 selected결과의 판정에 소급 반영하지 않는다.
- 지표=기존 scaled 2-pinball(primary), raw pinball/RMSE/MAE/80% 포함률/폭. 비교는 VINTAGE 대 STD/EXODROP/INIT0, 같은보정끼리 및 보정STD 대 원본VINTAGE(단순대체 가능성)로 고정. 최신/과거는 합쳐 새 primary로 만들지 않는다.
- 평균은 seed→원점→월에 동일 비중. 두seed와월별결과 공개. paired 월block bootstrap2000(seed61712); 한타깃4개월의 기술구간이며 독립 검정 아님.
- median MSE를 원점별 평균오차 제곱과 중심화 오차 분산으로 분해. 이는 정답을 쓰는 설명 통계이며 배포 보정이나 인과 증명이 아니다.
- 입력/예측/코드 해시, 원본224점수 재생, scalar primary abs/rel1e-10, 보정grid 선택 독립 검산, 분위수 순서/median/폭 변환, 미래 D label poison에도 계수 불변(보정 함수는 V만 받음), 기존 결과 보존을 검사한다.
- 유리하면 단순 보정 뒤에도 남는 개발 신호, 불리하면 단순 대조가 설명하는 범위를 보고한다. 어떤 결과도 새방법 PASS로 올리지 않는다. 오류·미실행과 과학적 결론 구분. 추가 후보 학습0.
