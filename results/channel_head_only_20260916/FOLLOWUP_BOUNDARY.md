# 후속 방법 설계에서 피해야 할 신규성 오해

2026-09-16. 이번4-fit 대조를 실행하는 동안 확인한 선행 경계다. 학습 계약 변경이나 추가 후보 실행 명세가 아니다.

[확인] [Time-LlaMA: Adapting Large Language Models for Time Series Modeling via Dynamic Low-rank Adaptation (ACL 2025 Student Research Workshop)](https://aclanthology.org/2025.acl-srw.90.pdf)의 §3.2–3.3 수식을 직접 읽었다. 각 층의 마지막 토큰 표현을 router 입력으로 삼아 선형층·softmax·top-k로 Q/K/V/O/G/U/D 중 활성 LoRA 모듈을 고른다. 입력과 층에 따라 선택이 달라지고 load-balancing 손실을 더한다. 시계열 입력별 LoRA routing이라는 큰 아이디어는 이미 존재한다. 논문 전체 실험 재현이나 저자 코드 검증은 하지 않았다. Student Research Workshop 논문을 ACL 본회의 논문으로 표기하지 않는다.

[확인] [TRACE v1](https://arxiv.org/html/2503.16991v1)의 초록·서론은 큰 forecasting head 축소와 LoRA 중요도 기반 선택을 이미 다룬다. 따라서 'head를 작게 만들기', '중요한 LoRA만 쓰기', '입력마다 adapter를 고르기'만으로 새 방법이라고 할 수 없다.

[미확인] [Adaptive Refinement of Time Series Foundation Models via Pattern and Context-Awareness](https://openreview.net/pdf?id=6Rai3jnoWj)는 검색 결과에서 FFT/DWT 기반 경량 adapter와 retrieval을 설명하지만 원문 열기는 브라우저 인증 페이지로 막혔다. 검색 요약만으로 구현 동치·최초성 또는 정식 게재를 판정하지 않는다.

이번 LP 대조는 좋은 단순 대안을 빠뜨리지 않기 위한 증거다. 결과가 LoRA의 추가 가치를 지지해도 위 선행들과 구분되는 구체적 메커니즘은 따로 필요하다. 아직 새 후보 수식은 봉인하지 않았고 이 문서로 추가 GPU 실행을 시작하지 않았다.
