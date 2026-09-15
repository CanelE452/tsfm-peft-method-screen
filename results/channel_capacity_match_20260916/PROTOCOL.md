# 동일 용량 pointwise adapter 대조 — 4 fits

2026-09-16. 기준15e094efd3e9d1b453603b484918a9d3cdf1dc88. 직전12-fit 입력 조건화 후보는 종료·보류 상태로 보존한다. 사용자 활성 목표의 실패 후 원인 검토·후보 탐색에 따라 적응 용량 교란 하나를 별도 대조한다. 기존 후보를 키워서 PASS로 재분류하는 작업이 아니다.

## 고정 질문과 변경

직전 일반 POINTWISE adapter의 적응 파라미터16,384개와 encoder LoRA의172,032개는10.5배 차이였다. head는 양쪽589,920개로 같다. 마지막 residual adapter가 열세인 이유에 용량 차이가 얼마나 관계되는가?

알려진 pointwise residual h'=h+B GELU(Ah) 하나를 쓴다. bias없음, hidden512, bottleneck168. 2×512×168=172,032이므로 head 포함761,952개가 기존 LH와 정확히 같다. 168은 점수 탐색이 아니라 이 등식으로 정한 유일한 폭이다. rank16 결과는 그대로 재사용하고, 다른 폭·학습률·표본을 추가하지 않는다. gaussian kernel/같은위상 평균/입력 조건화는 넣지 않는다. 기존 POINTWISE 구현을 참조하고 adapter만 같은 seed+202 초기화 규칙으로 폭168로 만든다. A는Linear 기본 초기화, B=0, head 초기값과 전체 초기 함수는 기존 LH와 같다. backbone 전체와 초기0-update LoRA는 동결한다.

MOMENT-small revision·원자료·32채널·96입력/96출력·70/10/20 split·train 통계·균형 train 원점·epoch별 순열은 기존 그대로다. trainable count가 같아도 표현력·삽입 위치·최적화 기하가 같지는 않다. 폭 변경은 학습 동역학도 바꾸므로 완전한 단일 원인의 인과 증명으로 해석하지 않는다.

## 예산·운영·학습

electricity/traffic×seed41000/41001=최대4fits,5080updates. smoke는2updates×2원천=4별도. 기존 LP/LH/POINTWISE16 각4fits는 재학습하지 않는다. 오류도attempt에 포함, 재시도·대체fit·새 후보·튜닝없음. 실제검사 통과시 고정4fits를 실행하고 성능의 부호로 중간에 조건을 추가하지 않는다.

AdamW lr.001 betas(.9,.999),eps1e-8,weight_decay0,clip1; StepLR5epochs gamma.5; BF16 batch8; cap20epochs; Vpatience5,min_delta1e-4; INIT포함 최소V MSE 선택. 기존 dropout/train/eval 정책 유지. 모든 모델의 초기 함수와 같은 seed/sample stream을 유지하되 서로 다른 폭의 파라미터 자체가 같다고 주장하지 않는다.

한GPU/한worker/기존lock. RustDesk만 기존 승인예외. 시작30초안정/free4GiB이상, 실행중1GiB미만 또는 외부compute는 경계대기. 누적대기600초/controller3600초. 환경·수치 오류는 성능FAIL과 분리한다. 검증한 코드·원점수·한국어보고서를 기존 승인대로push하고 큰weights/cache/원자료는 로컬에 보존한다.

## 평가·해석

네 V 선택을 먼저 봉인하고 노출된E(DISCOVERY_REUSED_E)를 평가한다. 각방법의 V 선택과 사전고정epoch(전력41000=1/41001=3, 교통둘=1) 결과를 모두 공개한다. 후자는 같은updates 비교이며 E에서 epoch를 고르지 않는다. 해당epoch가 실제trajectory에 없으면 미실행기록하며 연장하지 않는다.

비교군은 균형LH,head-only,POINTWISE16. 원점수MSE/MAE/rawMAE·source별/seed별 효과·실제학습/선택비용·학습peak·총모델/학습파라미터를 구분한다. 이 실행에서 새로운PASS 문턱이나 자원 목표를 만들지 않는다. 폭168이 좋아지면 이전 폭16의 용량 제한이 관련됐다는 개발 근거로, 개선되지 않으면 용량만 늘리는 접근으로는 설명/해결하지 못했다는 근거로 기록한다. 기존 입력 조건화 후보의 추가 가치 없음 판정은 변경하지 않는다.

새 방법이 아닌 알려진 adapter 용량 대조다. 성능이 좋아도 KNOWN_METHOD_USEFUL 수준의 근거이며, 독립SCREEN_PASS나 새 방법론 주제 확보로 표시하지 않는다.

## 검사

기존 POINTWISE 수식·FP64 gradient 검사 기록을 참조하고 함수 구현은 변경하지 않는다. 새 모델의 실제761,952 trainables·초기head/hash·encoder LoRA B=0·동결목록을 확인한다. BF16 INIT V8 예측이 기존LH와exact, smoke후head/down/up 실제변경·frozen/buffer불변·finite 확인. 미래poison 학습창불변, staged/model/source/과거결과hash보존, 선택checkpoint4개 새모델 재생exact, 전저장예측 MSE/MAE 독립float64 scalar atol/rtol1e-12, 선택봉인·epoch·원점·target/std·update원장 검산. 결과에 따라 허용오차를 변경하지 않는다.
