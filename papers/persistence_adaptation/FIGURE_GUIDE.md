# 논문 그림과 표 사용 안내

모든 그림은 같은 이름의 PNG,PDF,SVG로 제공한다. PDF/SVG는 벡터 편집용이며 PNG는 빠른 검토용이다. 표의 원자료는 `tables/`에 있다. 양수 gain/benefit은 C3의 오차 감소다.

1. **F1_main_seed_effects** — selected SHIFT8의 C3/B0,C2,RECENCY 비교. 검은 구간은 탐색적 날짜 조건부95%구간,주황×는개별seed. ETTm2 RECENCY는 미실행 표시. 서로 다른 데이터의 gain을 합산하지 않는다.
2. **F2_foundation_simple_baselines** — 원자료·FAULT·SHIFT8에서 F0,마지막값유지,계절반복,B0,C2,C3,RECENCY의 nMAE. 로그축이며 빠진 비교군은0이 아니다. 단순 점예측을 학습 seed로 복제하지 않는다.
3. **F3_matched_factorial** — 동일 LR·1,024 updates의 C2/C3 가중치와 추론 gate 교차 결과. 막대는 대칭분해의 gate/weights 항,×는합. 서로 다른 항의 크기는 순수한 기전 중요도 순위가 아니다.
4. **F4_all_shapes** — 등록된8형태와 pairedSHIFT8 전체. D17 양성과D63/pulse 음성을함께제시. standardSHIFT8과pairedSHIFT8의draw/표본차이를설명해야한다.
5. **F5_operating_mix_sensitivity** — 가상운영혼합에대한평균오차차이. w는SHIFT8비중,q는나머지중FAULT비중. q=0,.1,.5,1을모두표시하며 실제빈도·운영최적정책 추정이아니다.
6. **F6_validation_trajectories** — 기존selected LR에서0/256/512/768/1024 V점수와선택시점. 가중치를다시학습한곡선이아니다. ETTm1의C2와C3 LR는다르므로matched-training그림으로해석하지않는다.
7. **F7_cost_accuracy** — 기존RTX3080 FP32 128입력profile의seed별중앙값평균과원자료/SHIFT8점수. source B0학습비용은별도이며 latency비교를GPU메모리절감률로바꾸지않는다.
8. **F8_fixed_examples** — Electricity전이와NESO의첫E원점·첫채널·첫draw·seed81551,REFERENCE/POINT8/SHIFT8. 유리한예측으로선별한대표사례가아니다. 검은선은합성정답이며실제사건label이아니다.

9. **F9_gate_identifiability** — 평가 후 CPU 감사. C3/RECENCY mask의 context 단위 완전 일치율과 C3 all-one 비율을 전체 standard 조건에서 표시. NESO SHIFT8에서 두 규칙이 동일 mask를 만드는 사실은 학습 효과와 추론 위치 효과를 구분하는 데 필요하다.

본문 추천 배치는 F1,F3,F4,F9이며 F2,F5,F6,F7,F8은 주장과 분량에 따라 본문 또는 부록에 둔다. 원자료·오류 손해와 seed 역전은 본문에서 언급하고 부록에만 숨기지 않는다.
