# 선행 경계와 공식 코드 감사

2026-09-18에 지정된 다섯 논문의 공식 발표 페이지와 공개 구현을 확인했다. 코드 commit/파일 URL/hash는 official_source_receipts.json, 읽은 원문 파일은 로컬 literature cache에 있다. 아래 대조는 정식 방법 전체를 재현한 벤치마크가 아니다. 직접 충돌 후보는 지정된 COSA/TATO를 우선 확인했고 별도의 무제한 문헌 탐색은 하지 않았다. 정확히 동일한 공개 방법을 확인하지 못했지만 신규성 부재를 배제한 것은 아니다: NOVELTY_OPEN / FORMAL_BASELINE_COMPARISON_PENDING.

| 논문 / 정식 발표 | 문제 | 입력 | 적응 시점 | 학습 파라미터 | 보존 방식 | 이번 C3와 실제 차이 | 정식 재현 여부 | 남은 비교 |
|---|---|---|---|---|---|---|---|---|
| [Houlsby et al.](https://proceedings.mlr.press/v97/houlsby19a.html), ICML 2019 확인 | NLP 전이의 작은 추가 파라미터 | 텍스트 hidden state | 오프라인 task 학습 | bottleneck adapters 등 | 원래 network를 유지하며 near-identity로 시작 | C3는 Chronos의 입력 patch residual에 고정된 과거 지속성 mask를 적용. 일반 adapter 자체는 기존 원리 | 아님. C2는 단순 residual 대조이며 BERT 배치 전체 복제 아님 | 최신 TSFM 적응 대조와 공정한 직접 비교 |
| [ReZero](https://proceedings.mlr.press/v161/bachlechner21a.html), UAI 2021 확인 | 깊은 network 최적화 | 각 layer activation | 학습 중 | residual branch와 0에서 시작하는 scalar weight | 시작 시 identity | C3는 출력층 0 초기화, frozen B0, 관측으로 계산한 고정 patch mask. ReZero 전체 architecture 아님 | 아님 | identity 시작 자체를 기여로 주장하지 않기 |
| [COSA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2a8ce71baac4c89bf9ff479d8240c7d9-Abstract-Conference.html), ICLR 2026 확인 | 예측기의 test-time 적응 | 고정 예측과 최근 context, 도착한 정답 | 온라인 delayed/partial-ground-truth update | output adapter와 gate | backbone 고정, output residual | C3는 오프라인 학습 후 동결된 patch-space adapter. 평가 중 정답 update 없음. 실제 코드의 SimpleOutputAdapter는 tanh gate와 output correction을 사용 | 아님. BIAS는 COSA 재현이 아님 | 동일 정보 권한과 도착 정답 정책을 정의한 정식 비교 |
| [TATO](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48c5226582f41254026748c7e35d4ac2-Abstract-Conference.html), ICLR 2026 확인 | 고정 TSFM에 맞는 입력 변환 | 시계열과 변환 pipeline | 변환 설정 탐색·선택 | backbone weights는 고정, 변환 configuration 최적화 | frozen model, 전후 변환 | C3는 raw 입력을 변경하지 않고 이미 학습된 B0의 추가 residual만 조절. 공식 BasePipeline과 Optuna Tuner를 확인 | 아님. 단순 전처리를 전체 TATO로 부르지 않음 | 공식 전체 탐색 예산·선택 규칙을 포함한 비교 |
| [Cawley & Talbot](https://jmlr.org/papers/v11/cawley10a.html), JMLR 2010 확인 | 모델 선택과 평가 편향 | 유한 평가 표본·선택 절차 | 반복 선택 | 해당 없음 | 해당 없음 | 기존 E의 반복 사용, 새 seed와 CI가 개발 선택 편향을 해결하지 않는다는 한계에 적용 | 방법 성능 재현 대상 아님 | 완전히 미사용인 평가 자료가 필요 |

공식 구현에서 확인한 핵심: adapter-bert/modeling.py의 feedforward_adapter는 down/GELU/up과 residual을 사용한다. ReZero rztx.py는 residual weight를 0으로 초기화한다. COSA tta/cosa.py는 고정 예측에 gated correction을 더하고 도착 정답으로 adapter를 업데이트한다. TATO pipeline/base.py는 변환 전후 처리, tuner/base.py는 Optuna 탐색을 수행한다. 이 공통 원리들을 새 기여로 계산하지 않는다.
