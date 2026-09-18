# C3 추가 적응 연구: 최신 결과 감사와 약점 해결 실험 설계안

작성일: 2026-09-18
감사 기준: CanelE452/tsfm-peft-method-screen @ 2f4ebb4c62819511fa2e324ac2de6b09d2b3c259
문서 상태: 검토·합의용 설계안. 이 문서 작성으로 저장소 변경이나 모델 학습을 실행하지 않았다.

## 1. 목적과 현재 결론

최상위 목적은 현재 C3의 조건부 예측 이득을 인정하면서, 일반적인 추가 어댑터·위치 사전정보·크기 정보·간단한 출력 보정 이상으로 필요한 부분이 있는지 판단하는 것이다. 소비처는 사용자와 지도교수의 방법론 논문 개발 결정이다.

[확인] 최신 작업은 기존 가중치의 NESO2026 H1 전이와 C3/RECENCY 마스크 동일·상이 집단 분석이다. 60개 새 예측 view, 새 학습/optimizer 0회다. 128일/26주간 블록을 사용했다.

[확인] SHIFT8에서 C3 nMAE=0.349945, B0=0.362939, 일반 C2=0.357029, RECENCY=0.350159다. C3의 개선은 각각 3.5802%, 1.9843%, 0.0612%다. RECENCY 대비 구간은 0을 포함한다. 원자료 조건은 B0보다 0.4871%, F0보다 7.5176% 악화했다.

[확인] NESO2025와 NESO2026 H1의 SHIFT8에서 각각 256개 입력 모두 C3/RECENCY 마스크가 같았다. 추가 전력16계열의 다른 마스크 집단에서는 전체 이득 약0.0854%가 남았지만, 고정 함수 분해의 gate 항은 음수이고 weights 항은 양수였다.

[판단] 새 기간의 C3 대 일반 C2 이득은 보존할 근거다. 현재 관측은 정확한 지속성 위치를 사용해야 한다는 설명을 강하게 뒷받침하지 않는다. 단, RECENCY도 C3의 지속성 값들을 정렬하므로, 지속성 정보를 전혀 쓰지 않는 대조가 아니다.

## 2. 먼저 바로잡을 논문 문장

- 금지: '기존 PEFT는 모든 시간 위치를 동일한 크기로 수정한다.' 공유 파라미터를 써도 수정분은 입력에 따라 달라진다.
- 대체: '현재 일반 어댑터에는 관측된 지속성을 이용해 수정 강도를 명시적으로 제한하는 항이 없다.'
- 금지: '일반 어댑터가 지속 변화를 망가뜨리므로 보호가 필요하다.' C2 자체가 B0보다 좋아진 결과도 있다.
- 대체: '일반 어댑터의 추가 이득 위에서 명시적 지속성 가중이 더 유리한 조건이 있는지 검증한다.'
- 금지: 'RECENCY와 같으므로 지속성은 무의미하다.' RECENCY는 지속성에서 계산된 값의 배치 대조다.
- 금지: '가중치 항이 크므로 train-gate의 인과효과를 증명했다.' 현재 2×2는 저장된 함수의 오차 분해다.
- 금지: '마스크가 다른 하위집단의 이득만으로 독립 성공이다.' 해당 분석은 이미 본 자료의 설명 분석이다.

## 3. 가까운 선행과 주장 경계

아래는 논문의 공식 게재 정보/저자 자료에서 확인한 내용이다. 정확히 같은 C3 전체 수식을 발견하지 못했다는 사실은 신규성의 증명이 아니다.

1) Parameter-Efficient Transfer Learning for NLP(2019/ICML)
   - 기존 모델을 고정하고 작은 어댑터를 학습하는 원리가 이미 존재한다.
   - https://proceedings.mlr.press/v97/houlsby19a.html
2) UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning(2022/ACL)
   - 여러 PEFT 모듈을 결합하고 gate로 활성도를 학습하는 원리가 존재한다.
   - https://aclanthology.org/2022.acl-long.433/
3) Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation(2025/AAAI), TAFAS
   - 기존 시계열 예측기를 보존하면서 gated calibration과 도착한 정답으로 적응한다.
   - 원래 프로토콜은 온라인 적응이며 현재 C3는 오프라인 학습 후 동결한다.
   - https://ojs.aaai.org/index.php/AAAI/article/view/33965
4) COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting(2026/ICLR)
   - 동결 예측기의 출력에 context-aware gated residual을 더한다.
   - 최근 도착한 정답으로 online update한다는 정보 계약을 그대로 구분해야 한다.
   - https://proceedings.iclr.cc/paper_files/paper/2026/hash/2a8ce71baac4c89bf9ff479d8240c7d9-Abstract-Conference.html
5) Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models(2026/ICLR), TATO
   - 문맥 선택·정규화·이상값 보정의 조합을 선택하여 동결 모델을 적응시킨다.
   - 현재 고정 clip 한 개와 TATO 전체는 다르다.
   - https://proceedings.iclr.cc/paper_files/paper/2026/hash/48c5226582f41254026748c7e35d4ac2-Abstract-Conference.html

TAFAS/COSA를 정식 재현했다고 주장하려면 공개 코드·정답 도착 시점·업데이트 순서·선택 규칙을 실제로 재현해야 한다. 아래 출력 보정은 정보 접근을 현재 오프라인 설정에 맞춘 구조 대조이지, 원 논문 전체의 재현이 아니다. 온라인 기준선 숫자와 오프라인 C3 숫자를 정보 권한 설명 없이 순위화하지 않는다.

## 4. 왜 지금 새 구조 대신 세 대조를 추가하는가

공통 고정: 현재 C3 구조·마스크·B0 checkpoint·소스별 학습/검증 자료·증강·기존 손실을 변경하지 않는다. C0/C1/C2/C3/MEAN/ROTATE16/RECENCY는 검증된 완료 가중치를 재사용한다.

### E1. POS_ONLY: 실제 입력값을 보지 않는 위치별 gate

목적 A: 단순한 학습된 최근성/위치 사전정보로 충분한지 확인.
목적 B: 기존 RECENCY가 여전히 입력별 지속성 값을 사용한다는 빈칸을 채움.

[설계] 32개 패치의 gate를 g_j=sigmoid(a_j)로 둔다. a는 source 모델별로 공유하는 32개 학습 계수이며 입력값·p·사건 레이블을 받지 않는다. a_j=0으로 시작한다. C2와 같은 8,712개 어댑터에 이를 곱한다. 총 추가 학습 계수는 8,744개로 정확히 보고한다. C3와 같은 파라미터 수라고 주장하지 않는다.

초기에 g=0.5여도 adapter up weight/bias=0이므로 예측은 B0와 동일하다. 초기 원래 입력을 바꾸는 과거 B5와 다르다. 수식만으로 초기 동일성을 가정하지 않고 실제 모델에서 검사한다.

adapter LR 두 후보: 1e-4, 3e-4. 위치 계수 LR은 1e-2로 고정한 별도 그룹을 제안한다. 두 설정만 검증에서 비교하며 추가 탐색을 하지 않는다. 이 값들은 이번 설계값이지 최적값이라는 주장이 아니다. 낮은 gate LR 때문에 거의 움직이지 않았던 과거 문제를 반복하지 않도록 계수·gate·gradient 변화를 보고한다. 수치 오류는 구현 차단으로, 나쁜 성능은 결과로 구분한다.

예상 결과/판정:
- POS_ONLY≈C3: 관측된 지속성을 매 입력에서 계산할 필요성이 약해짐.
- C3>POS_ONLY: 입력에 따른 가중이 정적 위치 대안보다 유용할 수 있음. 이것만으로 지속성 자체가 원인이라고 단정하지 않음.
- 어느 쪽이든 원자료 손해를 함께 보고.

### E2. MAG_ONLY: 값의 크기만 사용하고 연속성을 사용하지 않는 gate

목적 A: 같은 방향의 연속 길이가 필요한지 확인.
목적 B: 큰 값이 있는 패치의 보정을 줄이는 것으로 설명되는지 확인.

[설계] C3와 같은 observed median, robust scale, threshold3을 쓴다. I_t=1[abs(d_t)>3], g_j=1-mean_patch(I_t). trailing window, 같은 부호의 지속 횟수는 사용하지 않는다. raw 입력과 B0 normalization은 그대로다. C2와 같은 어댑터 8,712개만 학습한다.

기존 학습이 완료된 C3를 inference 때 바꾸기만 하는 비교가 아니라, 이 gate를 사용해 같은 B0에서 새 어댑터를 처음부터 학습한다.

예상 결과/판정:
- MAG_ONLY≈C3: 현재 자료에서는 연속성보다 크기 정보로 충분할 가능성.
- C3>MAG_ONLY: 같은 크기 기준을 넘어 지속성 정보를 사용하는 제한된 추가 근거.
- 평가 마스크가 같은 입력에서는 위치/규칙 효과를 식별할 수 없음을 유지.

### E3. OUTPUT_CONTEXT: 같은 정보로 수행하는 저비용 출력 보정

목적 A: 내부 패치 수정이 필요한지 확인.
목적 B: TAFAS/COSA가 보여준 간단한 보정 원리와의 차이를 같은 정보 계약에서 살펴봄.

[설계] B0 전체를 동결하고 그 .5 분위수 예측64개와 observed context512개를 8구간으로 나눈 평균8개를 사용한다. 모든 feature는 B0의 동일 observed-context loc/scale로 정규화한다. 미래 y나 모사 state는 feature가 아니다.

u ∈ R^72,
r = W u + b ∈ R^64,
y_new(q,h)=y_B0(q,h)+s_context*tanh(gamma)*r_h.

W:64×72, b:64, gamma:1, 추가 계수4,673개. W는 작은 비영 초기값, b=0, gamma=0. W와 gamma를 동시에 0으로 만들어 양쪽 학습 경로가 막히지 않도록 한다. gamma LR은 adapter와 같은 두 LR 후보를 사용한다. 초기 예측은 B0와 동일해야 한다.

동일 수정분을 모든 분위수에 더하므로 기존 분위수 간격을 바꾸지 않는 제한된 대조다. C3는 분위수 간격도 바꿀 수 있으므로 확률예측 전체의 완벽한 capacity match라고 주장하지 않는다. 학습 손실은 모든 군의 normalized 2-pinball을 유지한다.

명칭: 'COSA에서 동기를 얻은 offline output-residual control'. 원래 COSA의 online context 수집·정답 도착·업데이트 스케줄을 재현했다고 부르지 않는다.

예상 결과/판정:
- 출력 보정으로 충분하면 내부 persistence adapter의 필요성은 약해짐.
- C3가 주 조건에서 더 좋고 비용/원자료 손해를 고려해도 의미가 남으면, 내부 정보 사용의 후속 근거가 됨.

## 5. 세 대조의 공동 프로토콜 — 승인 후 수행할 범위

[설계] 두 원천 Electricity/ETTm1, 각 세 대조. 기존 source별 selected B0를 공유한다. ETTm2의 RECENCY 학습이 없으므로 이번에 이를 채우기 위한 별도 학습을 자동 추가하지 않는다.

- 각 대조: 선택용 seed81550에서 두 학습률, 고른 학습률로81551/81552/81553 반복.
- 원천2 × 대조3 × (선택2 + 반복3) = 새 30 fits.
- 각1,024 updates → 본학습30,720 updates.
- 두 원천×세 대조×2 discarded smoke =12 updates.
- 합계30,732 optimizer updates 상한. 추가 LR/seed/구조/원천 자동 확대 없음.
- 원래 C3가 사용한 matched TRAIN, 원점, 32epochs/effectivebatch32, checkpoint0/256/512/768/1024, native nine quantiles, loss 및 V objective를 재사용한다.
- 일반적인 optimizer 설정은 기존 계약을 따른다. POS_ONLY의 작은 gate 그룹만 앞 절의 별도 LR을 명시한다.
- 같은 추가 학습 기회를 쓰더라도 최적화 난이도가 완전히 같다는 주장은 하지 않는다.
- 모델을 검사한 뒤 V로 선택하고, 이미 본 E는 개발 비교로만 채점한다. NESO2026을 또 채점해 새로운 확증이라고 부르지 않는다.
- 같은 입력/정답 노출과 baseline hashes, 실제 학습 변화, frozen baseline 보존, 초기/off 복원, 선택·scoring 순서를 검사한다.
- 일부 seed가 step0를 선택하면 추가 개선 없음으로 기록한다. 실제 학습을 했으므로 구현 실패라고 부르지 않는다.

## 6. 지표·해석

주 비교: 고정된 SHIFT8에서 C3 대 POS_ONLY/MAG_ONLY/OUTPUT_CONTEXT. 일반 C2 및 기존 RECENCY 대비도 병기한다.
안전/손익 패널: REFERENCE, FAULT, SHIFT4, SHIFT_POINT와 이미 정의한 전체 shape 평가. F0는 계속 포함한다.

- nMAE/원단위 MAE, seed별 효과, paired 원점 차이를 보고한다.
- 같은 날짜의 채널·draw·조건을 독립 날짜로 세지 않는다.
- 기존 seven-day paired bootstrap과 conditional-on-seeds 해석을 유지한다.
- 셋을 사전 주 비교로 정하면 3개 대조의 다중성 구간도 병기한다. 이는 누적된 연구 선택 전체를 제거하지 않는다.
- average gain의 크기와 구간을 함께 본다. CI0 포함은 동등성 증명도, 절대적인 미래 투자 금지 조건도 아니다.
- SHIFT8에서 좋은 결과만 골라 FAULT/REFERENCE 손해를 누락하지 않는다.
- 임의의 성공 문턱을 새로 만들어 결과를 PASS로 바꾸지 않는다.

## 7. 실제 사건 근거를 추가하는 별도 데이터 준비 단계

[확인] London Datastore의 Low Carbon London 자료는 30분 전력 소비와 2013년 dynamic time-of-use 가격 신호 일정을 공개한다. 약1,100가구에 높은/낮은/일반 요금의 시간표가 하루 전 통지되었다고 설명한다.
공식 자료: https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d
사업자: https://innovation.ukpowernetworks.co.uk/projects/low-carbon-london

[미검증] 이 일정이 실제 소비의 지속적 수준 변화 레이블이라는 것은 아니다. 가격 변경, 소비 반응, 영구적인 regime change는 서로 다르다. 원자료 파일·tariff schedule의 연결 및 유효 가구/날짜는 아직 확인하지 않았다.

제안하는 준비는 새 모델 학습이 아니라 다음 정보 감사다.
1) 공개 파일의 가구ID·시각·요금군·일정이 연결되는지 확인.
2) 관측 시각/DST/집계 단위·결측·권한·기존 노출을 확인.
3) 모델 오차를 보기 전에 event windows와 non-event 비교 기간을 schedule로 정의.
4) 사건 이전/이후에서 사용하는 context와 미래 정답을 명시. 반응이 큰 가구만 결과를 보고 선택하지 않음.
5) 같은 기간의 여러 가구는 같은 tariff event를 공유하므로 날짜/사건 군집으로 통계 처리.
6) 실제 y에 합성 delta를 더하지 않음.
7) tariff 정보를 모델에 주는지 여부를 모든 군에 동일하게 고정. 현재 univariate C3에만 사건 정보를 은밀히 주지 않음.
8) 일정만 있고 충분한 관측 반응을 검증할 수 없으면 '외부 사건 일정이 있는 전력 평가'로 표현을 제한. 지속적 변화 해결의 확증이라고 부르지 않음.

이 감사가 끝나기 전에는 실제 사건 연구의 fit 수/성능 목표를 임의로 확정하지 않는다. 해당 단계는 별도 합의이며 자동 실행/대규모 다운로드 대상이 아니다.

## 8. 하지 않을 작업

- 이미 완료한 SAME/DIFFERENT 집단 분석을 새 핵심 결과로 다시 계산하기.
- 같은 tail SHIFT8에서 C3/RECENCY가 동일한데 기간만 더 늘려 위치 기전을 증명하려 하기.
- C3가 이기는 입력만 골라 새로운 조건 이름 붙이기.
- 온라인 TAFAS/COSA에 도착한 정답을 주고 오프라인 C3와 동일 정보의 비교라고 쓰기.
- 인과 기전을 완벽히 증명해야만 방법론 논문이 가능하다고 요구하기.
- 새 방법 효과가 없다는 것을 감추기 위해 문제를 새 분야/새 지표로 자동 이동하기.

## 9. 분기별 결론

- C3가 일반 C2와 위 세 대조보다 특정 조건에서 반복적으로 유리하면: 그 조건부 메커니즘 주장을 유지하고, 독립 자료/실제 사건/정식 선행 중 남은 범위를 진행한다.
- POS_ONLY로 충분하면: 입력별 persistence 필요성이 약하다. 위치별 적응 제한의 기존 연구와 더 가까운 주장으로 낮춘다.
- MAG_ONLY로 충분하면: 연속성의 고유 가치가 약하다. 단순 큰 값 기반 대조를 보존한다.
- OUTPUT_CONTEXT로 충분하면: 복잡한 내부 변경의 필요성이 약하다. 알려진 보정 원리의 유용성으로 정리한다.
- 불확실하면: 효과 크기와 범위를 그대로 남긴다. 모든 조건 무승부/모든 seed 승리를 기계적인 논문 자격으로 삼지 않는다.
- 어떤 결과도 원고의 신규성을 자동 보장하지 않는다. 성능 차이, 정보 권한, 신규성, 현실 타당성을 별도 판단한다.

## 10. 기대 산출물

실행 전에: CLAIM_BOUNDARY.md, BASELINE_MAPPING.md, DATA_AND_EXPOSURE.md, 실행 예산·선택 규칙.
실행 뒤: 대조별 원점수/seed/선택checkpoint/자원, MATCHED_CONTRASTS.csv, 한국어 REPORT.md.
추가 필수: 공식 선행 전체와 offline 구조 대조의 차이를 표로 남긴다. E가 재사용이면 명시한다.
기존 결과는 삭제하거나 덮어쓰지 않는다. 출력 문서 수나 그림 수를 새 방법의 근거로 세지 않는다.

현재 완료 범위: 저장소 코드·문서 검토, 공식 논문/데이터 출처 확인, 반올림 원점수·파라미터 수·학습 예산 산술 검산. 실제 신규 학습·원자료 전수 재생·LCL 파일 연결은 하지 않았다.
