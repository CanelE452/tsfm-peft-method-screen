# 선행과 구현의 경계

확인일: 2026-09-22. 최상위 계약의 설계를 그대로 시험하며, 아래는 신규성 주장이 아니라 해석 범위를 제한하기 위한 출처이다.

- [bitsandbytes 공식 설치 문서](https://huggingface.co/docs/bitsandbytes/main/en/installation): Windows/CUDA 지원을 확인한 뒤 0.50.2 공식 wheel의 PyPI SHA와 실제 GPU packed NF4 연산을 검사했다. 지원 표만으로 실행 성공을 가정하지 않았다.
- [PEFT 공식 quantization 안내](https://huggingface.co/docs/peft/developer_guides/quantization): QLoRA의 all-linear 범위와 `replace_lora_weights_loftq`의 one-step 제한을 확인했다. 이 실험은 반복 LoftQ 재현이 아니다.
- [QERA 공식 코드](https://github.com/ChengZhang-98/QERA/tree/bd7fc86a2e44d41f95b9b0421f27f5624dd37064): input RMS의 대각 통계, residual×scale SVD, inverse-scale factor 복구를 대조했다. 실제 공식 함수 두 개만 AST로 추출해 동일 packed matrix에서 factor product를 수치 비교했다. 전체 repository를 실행하지 않았고 전체 QERA 벤치마크 재현을 주장하지 않는다.
- [TQS 공개본 v3](https://arxiv.org/html/2606.13300v3): 예측 궤적의 민감도를 이용한 양자화/정밀도 배치가 이미 연구되고 있다. 여기서는 native64 discrepancy로 LoRA rank를 배치하는 작은 개발 탐침이다. 논문의 rollout 동역학 분석 전체를 재현하거나 최초 forecast-sensitivity allocation이라고 주장하지 않는다.
- [FFA-LoRA, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4e243e95c913b367775d71d7182b99d9-Abstract-Conference.html): nonzero A를 고정하고 zero-initialized B만 학습하는 기존 원리를 사용한다. B 평균의 delta 동치가 이 실험의 새로운 방법론은 아니다. 이 실험에는 DP noise가 없다.
- [Trans-LoRA, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/708fdc7911f11585ee7161518e509ae6-Abstract-Conference.html): 모델 교체 시 적응 지식 이전은 기존 문제이다. 본 T는 실제 최근 BRIDGE target을 활용하는 예측 residual transfer이며, synthetic-data 기반 Trans-LoRA나 data-free 이전을 재현한다고 부르지 않는다.

통제의 수치적 성공은 논문 신규성, 다른 데이터 일반화, 제출 가능성의 검증을 대신하지 않는다. 최초50% scale과 공개 Electricity는 기존 개발/사전학습 노출 가능성이 있는 개발 화면이다.

학습 후 TEST 채점 전 추가 확인: QERA 함수 docstring의 balanced sqrt(S) 분배와 실제 helper 본문의 U*S / Vh 분배가 다르다. 로컬 구현은 전자이고 product만 수치 일치한다. factor scaling과 최적화까지 동일하다는 주장을 하지 않는다. Q_QERA/Q_IO16/Q_FORECAST에 같은 로컬 변형을 사용했으며 학습을 다시 실행하지 않았다. 결과의 QERA_FACTOR_BALANCE_AUDIT.json에 차이를 보존한다.
