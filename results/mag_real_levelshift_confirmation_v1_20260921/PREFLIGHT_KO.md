# 실행 전 부모 근거 감사

새 원천 정답을 열기 전에 작성했다. 기존 개발 SHIFT8에서 B0_MAG 대 B0_PLAIN은 electricity_transfer 약 +3.79%, NESO 약 +1.94%, ETTm1 추가 가치 없음/음수다. no-LoRA SHIFT8에서 MAG는 PLAIN보다 나빴다. 현재 근거는 already-adapted B0의 second-stage MAG에 한정한다. 실제 raw 부하의 추가 가치는 아직 확인하지 않았다.

이번 계약 하나만 적용한다. MAG 구현을 변경하지 않는다. 원천 schema가 계약에 맞지 않으면 학습하지 않는다. 기존 결과는 보존한다.
