# 최종 결정 C

LH가 두 원천의 주 비교에서 가장 정확하다. 교정된 LH를 작동 기준으로 보존한다. SIDE는 측정된 저메모리 대조군으로만 보관하며 PRIOR 확장과 새 attention 후보 자동 생성을 중단한다.

실행은 EXECUTION_COMPLETE이며 성능 또는 논문 PASS와 다르다. PRIOR_COMPONENT_SIGNAL 미충족; 평균 양성은 WEAK_OR_UNCERTAIN_SIGNAL, 정확도·자원 손익은 TRADEOFF_ONLY로 구분한다.

기존 구현·원점수·부정적인 seed와 원천을 모두 보존한다. 신규성은 확보되지 않았다. 사용한 E_FIXED/E_MIXED는 개발 기간 재사용이며 독립 test가 아니다. 새 연구·새 seed·추가 학습은 자동 실행하지 않는다.

## 구현의 역할

- LH: 교정된 TRAIN의 정확도 기준 구현으로 보존.
- SIDE: 동일 예산의 저메모리 대조 구현으로 보존.
- PRIOR: 추가 연구 후보 확장과 후속 학습 중단. 기존 코드·가중치·결과는 재현용 기록으로 보존.

LH의 native checkpoint 옵션도 자원 대조 구현으로 보존한다. 이번 동일 배치 2-update 검사는 두 원천에서 bitwise 통과했고 peak는 SIDE-fast보다 낮았다. 전체 20-epoch checkpoint 학습을 추가 실행한 것은 아니며, 시간 비용은 더 컸다.
