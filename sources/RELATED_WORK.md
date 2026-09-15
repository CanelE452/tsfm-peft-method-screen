# 채널 공유 파일럿의 선행연구 경계

## Time-PEFT에서 유지하는 것

[공개 코드 ea4e7e1](https://github.com/kaist-dmlab/TimePEFT/blob/ea4e7e1887bb35587bab7ea93e2af3685ac55852/run.py)의 MOMENT 정규화·patch·encoder·head 경로, q/k/v LoRA, top-3 Fourier filtering/projection, 공유 down과 채널별 up, ReLU/dropout/LayerNorm을 기준으로 삼는다. 보존한 원본은 `timepeft_ea4e7e1/`에 있다. 저자는 Jihye Na, Patara Trirat, Chanyoung Park, Jae-Gil Lee다. README는 Apache-2.0 배지를 포함하지만 지정 commit의 archive에는 별도 LICENSE/NOTICE 파일이 없다. 원본 README와 소스는 바이트 그대로 보존했다.

## 이번 비교에서 바꾸는 것

채널 up을 완전 공유, 예산을 맞춘 폭 증가, 고정 4그룹, signed 정적 채널 계수와 4개 공통 기저로 바꾼다. 초기 함수와 공통 가중치를 맞추고, 공개 train의 마지막 state 반환 대신 모든 arm에 동일한 best-V 저장·복원을 적용한다. 작은 MOMENT/표준 원천의 개발 비교이며 원문 전체 재현이 아니다. 기본 constructor의 encoder checkpoint 상태는 공통 preflight에서 명시적으로 선택한다.

## 이미 알려진 원리 및 남은 비교

[C-LoRA, CIKM 2024](https://arxiv.org/html/2407.17246v1)는 공유 표현과 채널별 저랭크 변환을 결합한다. Eq.5–6은 채널별 행렬과 공유 행렬을 곱하고 ReLU를 적용해 입력 표현에 결합한다. 이번 정적 선형 기저합 `W_c=sum_k a_ck B_k`는 그 구현과 완전 동일하다고 확인된 것은 아니지만, 채널 인식 공유/분해 자체는 알려진 원리다. 따라서 이번 BASIS4는 Time-PEFT의 통제된 파라미터 공유 변형으로 다루며 최초 PEFT나 새 방법 확정을 주장하지 않는다. 공식 C-LoRA 및 가까운 공유·혼합 adapter와 같은 백본/예산에서의 직접 비교와 더 넓은 선행 조사는 이번 24 fits 밖의 남은 과제다.

## 환경 차이

공개 requirements는 transformers 4.44.2를 지정하지만 momentfm 0.1.4 메타데이터는 4.33.3을 요구한다. 격리 `.venv-channel`에서 공개 실행 코드의 4.44.2를 명시적으로 override했다. Q `.venv`는 변경하지 않았다. 실제 설치 lock과 import/모델 검사 결과를 별도 보관한다. GPU 드라이버와 시스템 CUDA는 변경하지 않았다.
