# 공개 Time-PEFT 대비 구현 차이

원본 `sources/timepeft_ea4e7e1/`은 바이트 그대로 보존한다. README의 Apache-2.0 배지와 저자 표기는 유지하며, 이 지정 archive에 별도 LICENSE 파일이 없다는 기존 기록도 보존한다.

R2는 MOMENT-small revision411e288267f82cce86296dbe4d6c8bc533cc162f, 입력96/예측96,32개 고정 채널을 사용한다. pretrained encoder/embedder/normalizer는 고정하며 q/k/v LoRA(rank8,alpha32), HEAD를 모든 arm에서 학습한다. 다섯 channel arm은 top3 frequency projection을 학습한다. FFT와 주파수 projection은 명시적으로 FP32에서 계산하고 나머지는 공통 BF16 autocast다.

공개 forward의 normalizer/tokenizer/patch/encoder/head와 INDIV_REF의 down→ReLU/dropout→채널별 up→LayerNorm을 유지한다. 같은 state와 FP32 eval의 실제 모델 출력을 CPU에서 비교한다. 초기 up bias0·채널간 초기 up 복사, 작은 모델·자체 split·epoch 상한·최저 V checkpoint의 실제 저장/새 모델 복구가 공개 기본 실행과 다른 점이다.

예산별 폭·FACTOR·BASIS는 새 비교 모듈이다. frozen backbone 자체를 바꾸거나 C-LoRA 공식 삽입 구조를 재현했다고 하지 않는다. 일반 소형 원천에서 REF가 LH보다 나빠도 공개 논문 전체 반증이라고 해석하지 않는다.

환경은 기존 격리 `.venv-channel`을 재사용한다. 공개 requirements의 transformers4.44.2를 momentfm0.1.4 메타데이터의4.33.3 대신 사용한 기존 override를 그대로 기록하며, R1 `.venv`·시스템 torch/CUDA/드라이버는 변경하지 않는다.
