# 관련 연구와 기여 범위 — 한국어 원고용

시계열 기반 모델의 적응에서는 작은 학습 모듈을 어디에 두고 어떤 정보로 갱신하는지가 서로 다르다. [Time-PEFT 공개 구현](https://github.com/kaist-dmlab/TimePEFT/blob/ea4e7e1887bb35587bab7ea93e2af3685ac55852/run.py)은 MOMENT encoder의 표현에 주파수·채널별 adapter를 연결하고 head와 LoRA도 학습한다. 본 연구는 이미 학습된 Chronos-Bolt B0를 유지한 채 작은 추가 잔차의 적용 강도를 제어한다. 따라서 현재 실험은 Time-PEFT 전체와의 우열을 판정한 것이 아니다.

입력 의존 gating 자체도 새로운 원리는 아니다. [GateRA](https://arxiv.org/html/2511.17582v1)는 입력 표현에 따른 PEFT 조절을 제안했다. 본 연구의 TOKEN_GATE와 TOKEN_GATE_ENTROPY는 이 원리에서 착안한 내부 통제군이며 공식 HiRA 구조의 전체 재현이 아니다. 또한 [PETSA](https://arxiv.org/html/2506.23424v1)는 gating을 포함하는 저랭크 입력·출력 보정 모듈을 부분 또는 지연 정답으로 온라인 적응시킨다. 이 선행들은 “시계열 PEFT 최초의 gate” 또는 “최초의 작은 추가 보정기”를 본 연구의 기여로 내세울 수 없음을 보여준다.

본 연구에서 검증한 구체적 설계는 원래 관측을 잘라내지 않고, 관측창의 robust 진폭 통계로 추가 patch 잔차만 제한하는 고정 MAG이다. 일반 잔차, 위치 대조, 학습형 embedding gate와의 비교를 통해 이 제약이 특정 합성 지속 변화에서 제공하는 추가 가치를 조사한다. 학습형 gate와는 관측 권한은 같지만 특징 표현과 초기 gate가 다르므로, 결과를 비학습성 하나의 인과효과로 해석하지 않는다. 실제 fault와 regime change를 식별하는 방법이라는 주장도 하지 않는다.

완료된 직접 비교에서는 큰 합성 지속 변화(SHIFT8)의 전력 계열 전이와 사전 고정한 NESO 후반 기간에서 양성 근거를 얻었다. 그러나 ETTm1, 원자료·오류 조건, 긴 변화 형태의 손해를 함께 보고한다. 이미 사용한 개발 평가와 새로운 시간 구간을 구분하며, 새로운 시간 구간도 같은 원천의 자료다. 가까운 공식 방법 전체와의 비교 및 충분한 방법론 신규성은 아직 확립되지 않았다. 이 결과는 제한된 설계의 경험적 근거이며 범용 PEFT 우위나 논문 채택의 보장이 아니다.

공개 코드의 정확한 버전, CPU 검사 범위, 후속 비교에 필요한 조건은 [선행 호환성 감사](../../../research/method_baseline_compatibility_20260919/BASELINE_COMPATIBILITY_KO.md)에 기록했다. [방법 절](METHOD_SECTION_KO.md)과 [주장–실험 근거](PAPER_CLAIM_EVIDENCE_KO.md)를 함께 사용한다. 새 학습이나 기존 결과의 재선택 없이 작성한 원고 구성 요소이며 완성 원고를 의미하지 않는다.
