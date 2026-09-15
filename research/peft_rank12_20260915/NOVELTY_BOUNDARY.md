# 공유 변환 비교의 선행·신규성 경계

| 비교 | 수식·공유 축 | 삽입 위치와 학습 정책 | 예산/확인 깊이 |
| --- | --- | --- | --- |
| [Time-PEFT 공개 소스](https://github.com/kaist-dmlab/TimePEFT/blob/ea4e7e1887bb35587bab7ea93e2af3685ac55852/run.py) | 공통 down, 채널별 affine up | MOMENT encoder 뒤 frequency/채널 변환과 head; encoder q/k/v LoRA | 지정 commit 원본 코드와 실제 wrapper forward 비교. 공식 논문 전체 재현 아님 |
| [Channel-Aware Low-Rank Adaptation](https://arxiv.org/html/2407.17246v1) | 채널별 저랭크 성분과 공유 표현 | 채널 인식 보정을 backbone에 결합 | 저자본의 방법과 출판 정보를 확인. 공식 코드를 이 환경에서 실행하지 않았으며 FACTOR를 공식 C-LoRA로 부르지 않음 |
| 일반 shared factorization / 이번 FACTOR | A_c V, A_c는 h×q, V는 q×D | 같은 channel up 위치; 별도 중간 비선형 없음 | h95/q51, 채널 블록295,935. 일반 저랭크 행렬 곱의 대조 |
| 이번 BASIS | U_c=U0+Σ a_ck Uk, bias도 같은 합 | 정적 채널 ID 계수; forecast step router 없음 | h95/잔차3, 채널 블록295,103. vectorized affine weight 공간의 알려진 factorization |
| [MoLA 공개본](https://arxiv.org/abs/2505.17872) | 예측 step별 LoRA expert의 부분 공유 | step-specific 적응/가중 expert; 이번 고정 channel ID와 축이 다름 | 초록과 공개 버전 정보를 확인. 공식 구현·같은 백본 직접 비교·정식 게재 상태는 미확인 |

BASIS의 채널×vectorized-weight 행렬은 공통 행렬에 rank≤3인 편차를 더한 매개변수화다. 따라서 `KNOWN_PARAMETERIZATION`으로 기록한다. 이 수학적 표현을 새 발명이라고 하지 않으며, 같은 예산과 이 삽입 위치에서 추가 가치가 있는지가 실험 질문이다. 이전64채널 연구와 폭·분할·초기화·정밀도·학습 recipe가 달라 중복 결과는 없지만, 알려진 원리 또는 이전 부진과 무관한 독립 확인은 아니다.

추가 문헌 탐색이나 새 후보 생성은 하지 않는다. 후속 연구에서 공식 C-LoRA/MoLA와 같은 모델·예산의 직접 비교 및 미노출 원천 확인이 남는다.
