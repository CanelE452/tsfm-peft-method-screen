# 세 설명 대조 이후의 논문 주장

최신 다운로드 설계의 §4–6을 실행해 C3의 약점을 직접 비교했다. **30개 학습 경로·54개 평가 예측·독립 검산을 완료했다.** C3와 B0 자체는 바꾸지 않았다. 실행 전 봉인 커밋은 `396de8d`다.

전력 16계열의 SHIFT8에서 C3는 일반 C2보다 2.3994%, 정적 위치 대조보다 0.6943%, 출력 보정보다 2.8989% 좋았다. 그러나 연속 길이를 쓰지 않는 MAG_ONLY보다 0.2515% 나빴다. 날짜 및 계열+날짜 보정 구간도 이 방향을 지지한다. ETTm1에서는 정적 위치 대조가 C3보다 좋았다.

따라서 기존 전력 이득을 지울 이유는 없지만, 연속성 계산이 성능 개선에 필수라는 주장은 강화되지 않았다. 단순 크기 대조와 원자료·오류에서의 작은 절충을 함께 제시해야 한다. 모든 조건에서의 승리를 요구하는 것이 아니라, 제안 구성요소가 단순 대안 위에 무엇을 더하는지 구분해야 한다. 이번 결과만으로 새 PEFT 방법론의 일반 우위나 논문 PASS를 선언하지 않는다.

- [상세 해석과 주장 경계](../../results/c3_weakness_controls_20260918/INTERPRETATION_KO.md)
- [한국어 전체 보고서](../../results/c3_weakness_controls_20260918/REPORT.md) · [최종 판단](../../results/c3_weakness_controls_20260918/FINAL_DECISION.md)
- [주 비교와 seed 그림](../../results/c3_weakness_controls_20260918/figures/01_primary_controls.png)
- [원자료·오류·변화 절충](../../results/c3_weakness_controls_20260918/figures/02_tradeoffs.png)
- [학습된 위치 가중치](../../results/c3_weakness_controls_20260918/figures/03_selected_position_gates.png)
- [모든 변화 형태](../../results/c3_weakness_controls_20260918/figures/04_all_shapes.png)
- [검산 기록](../../results/c3_weakness_controls_20260918/VERIFICATION.json) · [재현 범위](../../results/c3_weakness_controls_20260918/REPRODUCIBILITY.md)

그림은 같은 폴더의 PDF/SVG로 편집·제출할 수 있다. 기존 원고와 PDF를 이번 결과 반영 완료본으로 취급하지 않는다. 재사용 개발 E, 합성 사건, 고정 세 seed의 조건부 통계, 공식 온라인 선행 미재현이라는 한계는 남는다. LCL 준비나 후속 학습은 자동으로 시작하지 않았다.
