# LoRA Q/K·V 진단: 시간 연결 수정만으로 충분하다는 근거는 없음

**신규 학습0회로16개 검증 예측과 검산을 완료했다. Q/K와 V 어느 한쪽을 제거해도 네 checkpoint 모두 악화했고, V 제거 비용이 더 컸다.**

## 원점수

같은 학습된 head를 고정한 상태다. FULL은 기존 V 선택 예측과exact하다. NONE은 LoRA와 공동 학습한 head를 남긴 상태이므로, 별도로 학습한 head-only의 성능으로 읽으면 안 된다. 모든 점수는 재사용 V이며 E 점수가 아니다.

| 원천 | FULL | Q/K만 유지 | V만 유지 | 모두 제거 |
|---|---:|---:|---:|---:|
| electricity | 0.270338 | 0.523560 | 0.360247 | 1.330163 |
| traffic | 0.275670 | 0.678395 | 0.412503 | 1.791871 |

| 원천 | seed | Q/K 제거 MSE 증가 | V 제거 MSE 증가 | 전체 LoRA 이득 중 V 대칭배분 |
|---|---:|---:|---:|---:|
| electricity | 41000 | 0.109681 | 0.228456 | 56.19% |
| electricity | 41001 | 0.070138 | 0.277989 | 58.96% |
| traffic | 41000 | 0.130103 | 0.363766 | 57.74% |
| traffic | 41001 | 0.143564 | 0.441684 | 59.79% |

대칭 배분은 두 구성요소를 넣는 순서의 기여를 평균한 설명용 계산이다. 절대 MSE 감소량의 합을 보존하지만, 재학습한 방법의 독립 가치나 보편적 인과 비중을 측정한 것은 아니다. [seed별 원점수](scores.csv), [분해](effects.csv), [사전 프로토콜](PROTOCOL.md).

## 다음 주제에 주는 근거와 반례

[확인] 같은 용량의 마지막 adapter가 LoRA를 따라가지 못한 뒤, encoder 내부 변화의 역할을 분리했다. 이번 저장 모델에서는 V 경로도 크게 작용하고 Q/K 경로도 제거할 수 없었다. 앞서 제안한 시간 패치 관계만의 경량 적응을 정답으로 선택할 근거는 확보되지 않았다.

[한계] V는 뒤 layer의 입력을 바꾸므로 그 뒤 attention 관계에도 영향을 준다. q/k를 바꾸면 전달 내용도 달라진다. 이 실험은 각 projection의 국소적 변경을 제거했으며, 시간과 내용을 완전히 분리한 실험이 아니다. 또한 head와 encoder가 공동 적응했으므로 삭제 손실을 그대로 학습 기여로 해석할 수 없다. QK_ONLY/V_ONLY를 처음부터 학습하거나 예산을 맞춘 비교는 미실행이다.

[제안] 다음 설계에서 설명해야 할 것은 **시간 연결과 전달 내용의 공동 적응을 어떻게 유지하는가**다. 단순한 시간 평균·입력 유사도 weighting·adapter 폭 확대만으로 충분하다는 주장은 현재 증거가 지지하지 않는다. 그러나 이 요구조건 자체가 새 알고리즘은 아니다. 구체적 수식과 가까운 단순 대조의 차이를 정당화하기 전에는 후보 확보나 PASS라고 기록하지 않는다.

## 선행과 신규성 경계

[TRACE, arXiv v1(2025), §4.2](https://arxiv.org/html/2503.16991v1)는 다른 LoRA 모듈을 mask할 때 남은 모듈 중요도가 달라지는 문제와 gate를 이용한 반복 masking·선택을 다룬다. 따라서 이번 상호작용 관찰이나 중요도에 따라 projection을 선택한다는 발상만으로 신규성을 주장할 수 없다. 이 논문의 모든 주장·실험을 재현한 것은 아니다.

[Beyond LoRA, arXiv v1(2024), §2](https://arxiv.org/html/2409.11302v1)는 Chronos에 BitFit·LayerNorm tuning·VeRA·FourierFT를 비교한다. 기존 PEFT를 다른 이름으로 적용하는 것과 새 방법론 개발을 구분한다. 현재 MOMENT 진단의 결과를 해당 ICU 과제나 Chronos-2 건물 실험에 일반화하지 않는다.

## 실행·검산·미실행 범위

실제fits0, optimizer updates0, V 예측16/16, E 예측0. Controller 39.3초, 최소 GPU 여유8810MiB, 비승인compute0표본. 메모리는 순전파 진단 비용이며 학습 메모리 절감의 새 측정이 아니다.

16개 예측의 MSE·MAE·rawMAE를 독립float64 scalar로 검산했다. MSE 최대차2.22e-16, FULL4개 예측은기존checkpoint출력과exact다. head·frozen tensor/buffer 불변, 지정한B만0, 원상복구·source/checkpoint/model/data hash 및 기존2139개 결과파일 보존을 확인했다. [검산](verification.json).

새 후보 구현·학습, 재학습에 의한 구성요소 비교, 미노출 평가, 다른backbone 일반화는 아직 미실행이다. EXECUTION=COMPLETE, 새방법론주제=NOT_CONFIRMED, 독립SCREEN_PASS=NOT_EVALUATED. 예측배열·원자료·가중치는 로컬, 코드·원점수·manifest·보고서는GitHub에 보존한다.
