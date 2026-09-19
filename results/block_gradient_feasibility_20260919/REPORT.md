# 일반 예측 PEFT 초기 방향의 실모델 전제 검사

**실행 완료, 새 방법의 추가 근거 미확보.** 두 원천에서 CROSS_BLOCK은 단순 MEAN_SVD보다 후반 TRAIN gradient와의 정렬이 낮았다. 새 방법론 논문의 목표는 미달이며 이번 검사를 논문 PASS나 실제 예측 성능 실패로 표시하지 않는다.

## 실제로 수행한 것

원본 Chronos-Bolt-small F0에서 Electricity/ETTm1의 기존 raw TRAIN만 사용했다. 학습된 B0/C3/TRP 가중치나 합성 상태를 사용하지 않았다. 첫96일과 마지막96일 사이에 Electricity3270 / ETTm1 9875 time slots의 input+target embargo를 확인했다. 두 집합 안에는 target overlap이 남아 있으므로12블록을 독립 반복이라고 부르지 않는다. Electricity는 timestamp가 없는 index-day다.

GPU gradient 계산48회, 추가 native 출력 동일성 forward2회, **새 학습0 fits·optimizer update0회·V/E 평가0회**다. CPU 수학 검사에는 작은 LoRA의3회 autograd가 별도로 포함된다. 원본 출력 동일성과 전체 모델 state hash 보존을 두 원천 모두 확인했다. 각 층의 rank는8로 고정했다. 36개 층 모두를 집계했으며 불리한 층을 빼지 않았다.

## 원점수

| 원천 | 방식 | 후반 TRAIN과 방향 내적 | 전체 cosine | update 방향 norm |
|---|---|---:|---:|---:|
| electricity | RANDOM_ORTHO | 6.20537584e-06 | 0.016029 | 0.006489 |
| electricity | MEAN_SVD | 0.000297321475 | 0.115521 | 0.043140 |
| electricity | SECOND_MOMENT | 0.000264000224 | 0.108746 | 0.040692 |
| electricity | CROSS_BLOCK | 0.000250110255 | 0.104025 | 0.040300 |
| electricity | FULL_GRADIENT | 0.000343287798 | 0.114724 | 0.050156 |
| ettm1 | RANDOM_ORTHO | 3.91330847e-05 | 0.017220 | 0.014380 |
| ettm1 | MEAN_SVD | 0.00142280452 | 0.093216 | 0.096586 |
| ettm1 | SECOND_MOMENT | 0.00137073724 | 0.092446 | 0.093826 |
| ettm1 | CROSS_BLOCK | 0.00123845109 | 0.082757 | 0.094696 |
| ettm1 | FULL_GRADIENT | 0.0015863573 | 0.096530 | 0.103991 |

내적이 양수라는 것은 해당 방향의 음의 미소 update가 후반 TRAIN 손실을 줄일 것이라는 **1차 근사**다. 실제 Adam 학습·장기 경로·예측 오차의 개선율은 아니다. CROSS가 MEAN보다 낮다는 것은 이번 초기화 가설의 직접적인 초기 근거가 없다는 뜻이며, 모든 후속 비선형 학습의 실패를 증명하지 않는다. FULL_GRADIENT는 rank 제한 없는 참고값이므로 동등 예산 PEFT 대조가 아니다.

평균 gradient의 방향을 쓰는 알려진 대조 자체는 random 직교 초기 방향보다 두 원천에서 훨씬 정렬이 높았다. 이를 우리의 새 기여로 세지 않는다. CROSS의 covariance subtraction은 이 고정 자료에서 후반 방향 정렬을 더 높이지 못했다. 진단을 본 뒤 rank·날짜 블록·감산 계수·seed를 바꾸지 않았다.

## 0-gradient 처리와 비용

각 원천 decoder self-attention q6개는24개 블록에서 모두 gradient0이다. 이 계산 경로에서는 decoder query가 하나여서 자기 attention 가중치가1이 되는 구조와 부합한다. 초기 CPU 집계의0분모 assertion이 중단됐으며, 원래 봉인 코드를 보존하고 별도 [집계 보정](ANALYSIS_AMENDMENT_01.json)으로 모든0행을 남겼다. 360행 중60행의 cosine은 미정의다. GPU 재계산·추가 update는0이다. 이 알려진 퇴화 구조 제거를 새 방법이라고 주장하지 않는다.

| 원천 | gradient 호출 | source 처리 초 | peak allocated MiB |
|---|---:|---:|---:|
| electricity | 24 | 5.62 | 388.79 |
| ettm1 | 24 | 6.30 | 396.49 |

전체 GPU guard 구간44.39초 중 안전 대기30.27초. 최소 여유8239MiB, 비승인 외부 compute 표본0. generic wall 기록의 external_compute_samples는 허용된 RustDesk를 포함하므로 비승인 간섭 횟수가 아니다. CPU eigendecomposition/파일 검산 시간은 위 GPU source 처리 시간과 별개다. gradients의 dense 저장은 진단 비용이며 새 PEFT의 메모리 절감으로 보고하지 않는다.

## 선행과 주장 한계

LoRA-GA 공식 코드는 여러 batch의 gradient를 평균하고 SVD로 A/B 양쪽 방향을 초기화하며 scale와 base offset 보상을 적용한다. 이번 MEAN_SVD는 A의 한쪽 직교 부분공간 비교로, LoRA-GA 전체 재현이 아니다. CROSS의 일반적인 off-diagonal moment 수식은 알려진 통계 구성이다. 시간 블록을 적용한 구현 차이만으로 충분한 신규성이 생기지 않는다. [실제 코드 출처와 문헌 검토](../../research/method_feasibility_20260919/NOVELTY_AND_NEXT_REQUIREMENTS_KO.md).

현재 근거로 CROSS 본학습을 추가 투자 대상으로 올리지 않는다. 이 판단은 수치 오류·자료 부재 때문이 아니며, 새 예측 방법의 실제 성능을 검증 완료했다는 뜻도 아니다. 가까운 초기화 선행과의 구체적 차별성, 같은 계산 예산의 본학습 이득, 다른 기간/원천의 확인이 모두 남아 있다. 새 후보나 추가 학습을 자동 연결하지 않았다.

## 재현과 검산

[봉인 계약](../../research/method_feasibility_20260919/PROTOCOL.md), [블록](BLOCKS.json), [원점 감사](ORIGIN_AUDIT.csv), [360개 층별 점수](LAYER_RESULTS.csv), [독립 검산](PUBLICATION_AUDIT.json), [그림](gradient_direction.png). GPU 단계는 기존 complete marker를 확인해 중복 실행하지 않는다. 최초 분석 entry point는0방향 처리 오류를 그대로 보존했다. 현재 집계 명령은 `OPENBLAS_NUM_THREADS=4 .venv/bin/python research/method_feasibility_20260919/analyse_gradients.py`다.

gradient 배열·사전학습 가중치·raw TRAIN은 ignored 로컬 cache이고 GitHub에는 hash/원점수/코드가 있다. GitHub만으로 수치 replay가 즉시 가능한 것은 아니다. 자료는 기존 개발 연구에서 사용한 원천이므로 새 독립 시험이라고 부르지 않는다.
