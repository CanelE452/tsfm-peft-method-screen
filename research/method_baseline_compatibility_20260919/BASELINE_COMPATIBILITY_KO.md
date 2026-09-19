# 공식 선행 코드와 현재 MAG 비교의 호환성

작성: 2026-09-19. 기존 학습형 gate 비교가 완료된 뒤 실시한 코드 검토와 합성 CPU 검사다. 새 학습 0회, optimizer update 0회, 실제 자료 예측 0회다. 기존 16 fits의 조건·선택·결론을 변경하지 않는다.

## 결론

현재 MAG의 제한된 양성 근거는 유지된다. 다만 “시계열 PEFT에 처음 gate를 도입했다”는 주장은 사용할 수 없다. PETSA도 작은 시계열 적응 모듈에 gating을 사용한다. 우리의 차이는 학습된 B0와 원래 관측을 유지하고, 관측창의 robust 진폭 통계로 추가 내부 잔차만 제한하는 구체적 설계에 있다. 이 차이가 충분한 방법론 신규성인지는 별도 판단이 필요하다.

Time-PEFT를 “다른 채널의 관측을 더 보므로 비교할 수 없다”고 제외해서도 안 된다. 검토한 공식 adapter는 채널별 파라미터를 사용하지만 다른 채널 값을 직접 섞지 않았다. 필요한 것은 정보 권한을 이유로 한 제외가 아니라 backbone·학습 대상·채널 전이 규칙의 명시다.

## Time-PEFT: 코드 확인과 CPU 검산

검토 버전: [공식 run.py, ea4e7e1](https://github.com/kaist-dmlab/TimePEFT/blob/ea4e7e1887bb35587bab7ea93e2af3685ac55852/run.py). 기존 캐시 SHA-256을 기존 receipt와 대조했다.

FrequencyAdapter는 각 채널·특징의 patch 축 FFT에서 큰 성분을 선택한다. ChannelAdapter는 같은 채널의 backbone/frequency 출력을 이어 붙이고, 공유 down projection과 채널별 up projection을 적용한다. 전체 pipeline은 MOMENT encoder 뒤에 이 모듈들을 연결하며, adapter·forecast head·encoder LoRA를 학습한다. 현재 MAG는 이미 학습된 Chronos B0를 동결하고 attention 이전의 추가 잔차만 학습하므로 개입 위치와 학습 대상이 다르다.

공식 두 class를 AST로 추출해 CPU에서 직접 실행했다. 합성 입력 [2,4,32,512], top-k=3, hidden ratio=2에서 다른 채널만 바꿨을 때 채널0 출력 차이와 다른 채널에 대한 Jacobian은 모두 0이었다. 채널 순서를 바꾸고 되돌렸을 때는 채널별 파라미터 때문에 출력이 달랐고, 4채널 모듈에 1채널 입력을 넣으면 오류가 났다. 이 검사는 adapter 정보 흐름만 검증하며 전체 MOMENT의 정규화·추론·성능을 검증하지 않는다.

이 크기에서 frequency+channel 파라미터는 4채널 1,052,416개, 1채널 657,664개다. head와 LoRA를 제외한 부품 수치이며 공식 전체 모델 비용이나 MAG 대비 동일 예산 성능으로 쓰지 않는다. 공개 train 함수는 validation으로 조기 종료하지만 best checkpoint를 복원하지 않고 마지막 model을 반환한다. 향후 이 동작을 바꾸면 재현 변경으로 기록해야 한다.

새 16계열/단일 NESO에 기존 채널별 가중치를 어떻게 대응시킬지는 별도 문제다. head 공유·평균·재학습은 각각 방법을 변경할 수 있다. 이번에는 어느 것도 적용하지 않았다.

## PETSA: 가까운 시계열 gating 선행

자료: [논문 v1](https://arxiv.org/html/2506.23424v1), [공식 tta/petsa.py, 87853d8](https://github.com/BorealisAI/PETSA/blob/87853d888e98311ac94e64be920d17b57143b20c/tta/petsa.py). 다운로드한 코드·설정·README의 URL, commit, SHA-256은 [receipt](PETSA_CODE_RECEIPTS.json)에 있다.

PETSA는 작은 입력·출력 보정 모듈을 부분/지연 정답으로 온라인 갱신한다. 공식 GCM은 채널별 학습 계수 gamma로 tanh(gamma*x)를 만들고 저랭크 시간 변환과 bias를 더한다. 이는 현재 TOKEN_GATE의 embedding→sigmoid나 MAG의 patch별 잔차 배율과 같은 수식은 아니다. 공개 전체 runner는 Huber·주파수·구조 항을 포함하는 loss를 사용한다.

공식 GCM을 CPU에서 실행해 B=0, bias=0 초기화가 정확한 identity임을 확인했다. 단일 채널, rank16, 입력512/출력64를 가정한 두 cell은 각각16,897/2,113개, 합계19,010개 파라미터다. 합성 파라미터를 수동으로 바꾸면 출력이 바뀌는 것도 확인했다. 학습·Chronos 연결·온라인 갱신·예측 성능은 검사하지 않았다.

현재 계약은 E 전에 모든 가중치를 고정한다. PETSA 전체 온라인 절차는 정답 도착 시점과 예측 발행 시점을 따로 정해야 하므로 같은 실험이라고 간주할 수 없다. 이것은 PETSA에 누수가 있다는 판정이 아니다. GCM만 현재 offline pinball 학습에 넣는다면 **공식 부품의 통제 이식**이지 PETSA 전체 재현이 아니다. CPU 통과만으로 실험 runner가 완성됐다고 기록하지 않는다.

## 현재 결과가 비교한 것과 남은 것

| 비교 | 완료 범위 | 아직 주장할 수 없는 것 |
| --- | --- | --- |
| MAG 대 PLAIN·B0 | 같은 기반 예측기에서 추가 잔차의 가치, 실제 학습·평가 | 모든 변화·원천에서의 우위 |
| MAG 대 두 embedding gate | 동일 관측 권한·학습 기회의 실제 직접 비교 | 고정/학습 여부만의 인과효과, 공식 GateRA 전체 우위 |
| MAG 대 δ XY cell | 두 폭의 공식 부품 통제 이식 및 학습 | 공식 전체 δ 방법 재현 |
| Time-PEFT | 코드·adapter 정보 흐름·부품 파라미터 검사 | 전체 MOMENT 성능 비교 |
| PETSA | 코드·GCM identity·부품 파라미터 검사 | Chronos 연결 또는 전체 온라인 성능 비교 |

가장 직접적인 실험 증거의 빈칸은 같은 B0에서 가까운 공개 시계열 보정 부품과의 비교다. 그 비교를 추가하더라도 전체 온라인 방법 우위와 구분해야 한다. Time-PEFT 전체 비교에는 MOMENT와 채널 전이 절차를 별도로 정의해야 한다. 두 방법을 단순히 현재 성적표에 이름만 추가해서는 안 된다.

승인된 16 fits는 모두 종료했다. 이 문서는 후속 학습 승인이나 새로운 실험 계약이 아니다. 비교 범위·계산 예산·정답 도착 권한을 새로 고정하지 않은 추가 학습을 자동 시작하지 않았다. 기존 공개 보고서의 양성 결과와 부정 조건을 그대로 보존했다.

## 재현

저장소 루트에서 `.venv/bin/python research/method_baseline_compatibility_20260919/probe.py`를 실행한다. 결과는 [CPU_PROBES.json](CPU_PROBES.json)이다. probe는 hash를 확인한 공식 class만 추출하며 optimizer를 생성하지 않는다. 공식 소스 캐시와 기존 Time-PEFT receipt가 필요하므로 GitHub checkout만으로 오프라인 재실행이 가능하다고 주장하지 않는다. PETSA 원본 코드는 공개 URL로 추적할 수 있고 캐시는 저장소에 포함하지 않았다.
