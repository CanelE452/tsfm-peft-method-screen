# N02 ARCHIVE


6. N02 ARCHIVE — 최근 창 밖의 관측 완료된 과거 사례
======================================================================

데이터: Traffic 4채널, 기본 C336h/H48h.
문제: 이미 관측 완료된 예전의 유사한 패턴과 그 이후 값이 추가 예측정보를 주는가?
선행: Retrieval Augmented Time Series Forecasting(ICML2025, RAFT).
검색 자체/검색 후값 사용은 알려진 방법. 아래 간단한 보정도 신규성 미확인.

과거 은행:
채널별 r>=336인 TRAIN 원점에서 stride24로 (과거336, 이후48)을 저장.
현재 o에서 검색 가능한 r은 r+48 <= min(TRAIN_end, o-336-24).
현재 문맥과 은행의 전체 정답 사이에도24h 간격을 둔다.
TRAIN query에도 동일 규칙. query의 target 또는 query와 겹치는 continuation 검색 금지.
TRAIN query는 o>=2048부터 선택한다. 은행후보>=32인 원점만 기계적으로 유효 처리.
은행 최대2048개: 많으면 TRAIN 원점의 deterministic 균등 인덱스.
V/E 은행은 TRAIN에 고정. V/E의 새로운 과거로 은행을 갱신하지 않는다.

검색:
채널별 query의 마지막336과 은행past336을 각 창의 mean/std로 표준화.
Euclidean mean squared distance로 상위K=2. 동률 r가 이른 것을 선택.
정답 continuation을 유사도 계산에 넣지 않는다.
과거와 후속 전체를 query의 past mean/std로 단위 변환한 두 사례를 만든다.
후속값이 현재에서 이미 과거인 시각이라는 증명을 각 retrieval에 남긴다.

4개 학습군:
B0 SHORT: 원래 C336 그룹 + LoRA.
B1 LONG: 원래 C1344 그룹 + 동일 LoRA. H48 동일.
   더 긴 문맥은 강한 직접 대안. 입력 정보량/비용 차이를 명시.
B2 RETRIEVE: C336 원래 그룹 + 채널마다K2의 검색past를 보조 행으로,
   검색continuation을 그 보조행의 'known future feature'로 제공 + LoRA.
   이 feature는 실제 미래 측정값이 아니라 이미 관측된 과거 사례를 정렬한 합성feature다.
B3 DELTA_ADAPT: B2와 같고 검색 future만 다음처럼 조정:
   u_last = 검색past의 마지막 값(이미 query 단위로 변환)
   u_future' = u_last + exp(clip(a_c,-2,2))*(u_future-u_last) + b_c*sigma_query.
   a_c=b_c=0 초기. 추가8 scalar. 검색past 자체는 바꾸지 않는다.
   모든 검색행의 affine를 past/future에 같이 적용하면 native normalization이
   효과를 지울 수 있으므로 그런 구현으로 대체하지 않는다.
   clipping saturation율과 실제 a/b gradient를 기록.

추가 CPU 대조: 검색continuation 평균만 예측, B0 예측과 그 평균의 전역 convex blend.
blend alpha={0,.25,.5,.75,1}, V_SELECT에서만 선택. 비용에 검색시간 포함.
본체가 같은 조합은 prediction reuse. CPU fitting도 기록.

주지표: 뒤 구간 전체 normalized RMSE.
핵심 대비 B3 vs B2, B3 vs LONG, B3 vs 단순 blend.
B2가 SHORT만 이기는 결과는 추가정보 효과이지 새 PEFT 효과가 아님.
LONG으로 충분하면 문맥 제한을 의도적으로 줄여 retrieval을 살리지 않는다.

필수검사:
query target poison -> 검색순위/변환/입력 불변,
모든 r+H <= o-C-24, bank TRAIN 경계, 자기 자신/겹침 exclusion,
B3 초기=B2, 바꾸는 future delta가 native 전처리 후에도 실제로 달라짐,
보조후속은 label이 아니라 합법적 이미 관측된 feature임을 시간표로 증명.


[설계] 전체 계약의 공통 조항도 적용. 실행/효과/단순 대안/신규성을 분리한다.
