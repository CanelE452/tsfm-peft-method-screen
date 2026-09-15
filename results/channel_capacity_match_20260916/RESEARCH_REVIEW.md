# 사전 연구·코드 검토

[확인] 직전12-fit에서는 같은 용량의 POINTWISE/UNIFORM/CONDITIONED 사이에 후보의 실용적 추가 가치가 없었다. 그러나 LH와의 비교는 적응 파라미터16,384 대172,032로 달랐다. 이번에는 일반POINTWISE만172,032개로 맞춰 이 교란을 직접 확인한다. 이전조건의FAIL/보류 판정을 새 기준으로 바꾸지 않는다.

[설계] 기존 h+B GELU(Ah)의 폭만168로 바꾼다. 원시입력 가중치나 주기 가설을 보강하는 실행이 아니다. 두 선형층 bias를 넣지 않아2Dr의 정확한 등식을 유지하고 head589,920개를 더해761,952개다. 다른 폭을 탐색하지 않는다. 독립 RNG에서 동일초기화 규칙을 사용하며 B=0으로 초기 함수는 기존 모델과 같다. 같은 수의 파라미터가 같은 표현력/최적화 난이도를 뜻하지 않음도 명시한다.

[신규성] 알려진 bottleneck adapter의 용량 비교다. [Houlsby et al., ICML2019](https://proceedings.mlr.press/v97/houlsby19a.html)와 이전 [선행 검토](../channel_phase_transport_20260916/RESEARCH_REVIEW.md)의 경계를 유지한다. 새 논문 제목이나 새로운알고리즘으로 포장하지 않는다. 기존 source의검증된 Model/Transport를 독립이름 importlib로 로드해 upstream 'run/model' 이름 충돌을 피한다.

[한계] 학습률·초기화규칙·epoch cap을 유지한 단일비교이므로 각폭별최적튜닝의상한을검증하지않는다. 모든E는이미노출됐고, 사전학습중복이나다른backbone으로의일반화는미확인이다. 만약큰adapter가LH에근접해도그것은강한단순대조를확보한것이지새방법의가치가아니다.
