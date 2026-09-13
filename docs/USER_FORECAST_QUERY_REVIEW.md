# Local backward 결과 검토와 다음 개발안

검토 기준: `CanelE452/tsfm-peft-method-screen`, `main`, `39ab580b512f1f8054c49aae592816703c539c27`.
실험 실행 커밋: `7c08a264f72b68b6d9f0f31766e70d78d3c5f148`.

## 1. 최종 판정

[확인] 새 local backward 후보는 forward, LoRA B-gradient, 입력으로 전달하는 gradient를 그대로 유지하고 A-gradient만 근사하도록 구현되었다. 결과 기록은 216회 backward, optimizer update 0회, 신규 학습 fit 0회다. V/E를 이용한 새 예측 성능은 없다.

[판정] 고정된 잔차 샘플링 방식의 중단은 타당하다. local FP16보다 더 많은 저장 공간을 사용하고, 측정 peak와 시간도 불리하며, A-gradient 추정 오차도 크다. BF16+checkpoint가 현재 측정한 자원 대조에서는 더 강하다. 따라서 현재 primitive를 16회 예측 학습으로 확장할 근거는 없다.

[정정] 이전 제안에서 FP32 평균 저장량과 local FP16의 단순한 byte lower bound를 먼저 비교했어야 했다. 이 부족함은 실험을 지시한 설계에도 있다. CLI가 구현을 잘못했다고 책임을 돌릴 근거는 없다.

## 2. 확인 범위

직접 읽은 핵심 파일:
- `results/local_backward_feasibility/RESULT.md`
- `results/local_backward_feasibility/status.json`
- `results/local_backward_feasibility/verification.json`
- `docs/LOCAL_BACKWARD_PROTOCOL.md`
- `scripts/profile_local_backward.py`
- `src/tsfm_peft_screen/memory/local_backward.py`
- `src/tsfm_peft_screen/memory/common.py`
- `src/tsfm_peft_screen/memory/compression.py`
- `tests/test_local_backward.py`

독립 수행:
- 저장 데이터 형식으로부터 payload byte 수 재계산.
- 평균/차이 gradient 항등식의 float64 CPU 검사.
- 작은 확률 공간 전수 열거를 통한 잔차 gradient 추정의 기대값과 분산 공식 검사.
- 보고서에 반올림되어 실린 peak/time 수치의 비교 비율 계산.
- 제안하는 query-branch의 초기 출력 일치와 gradient 흐름을 toy Transformer에서 확인.

미수행:
- Chronos-2 GPU 학습 또는 profiling 재실행.
- 로컬 gradient/forecast cache의 128개 파일 직접 재검산.
- 제안하는 다음 방법의 실제 Chronos-2 구현·정확도·메모리 측정.

`reported_rounded_table.csv`는 RESULT.md의 전사 자료다. 원시 metrics.json 전체를 독립적으로 다시 집계한 것으로 부르면 안 된다.

## 3. 완료된 실험의 의미

8 methods × 2 datasets × 2 context lengths = 32 cases.
시간 반복은 최초 3회이며, PRAC와 residual은 case당 총16개 codec draw다.
총200 measured backward +4 reference +12 integration parity =216.
이는 216회 학습이 아니다. 고정된 warmed parameter state의 역전파 진단이다.

실험은 다음 상태를 확인했다.
- FP32 codec forward: bit-identical.
- 입력 gradient: 기록상 상대오차 0.
- B-gradient: 기록상 상대오차 0.
- 모든 temporal detail 보존: A-gradient 상대오차 최대 1.5061e-7.
- 원래 parameter state 변화 없음.

주의: 같은 state에서 propagation이 정확하더라도 이후 A를 근사 gradient로 업데이트하면 모델 경로는 달라진다. 전체 학습이 standard LoRA와 같다는 보장이 아니다.

## 4. 주요 수치

Context4096, batch8, horizon48, FP32 parameters와 AdamW moment buffers를 상주시킨 forward/backward 측정이다. 실제 optimizer.step peak는 포함하지 않았다.

| 자료 | 방식 | Peak MiB | ms | A 상대오차 % |
|---|---|---:|---:|---:|
| ETTm2 | FP32 standard | 1990.0 | 202.56 | 0 |
| ETTm2 | FP32 checkpoint | 772.7 | 276.30 | 0 |
| ETTm2 | local FP16 | 1770.6 | 203.77 | 0.00618 |
| ETTm2 | residual | 1809.8 | 206.93 | 12.93511 |
| ETTm2 | BF16 autocast+checkpoint | 735.0 | 178.15 | 3.89087 |
| Electricity | FP32 standard | 1990.0 | 200.28 | 0 |
| Electricity | FP32 checkpoint | 772.7 | 276.62 | 0 |
| Electricity | local FP16 | 1771.5 | 203.49 | 0.01098 |
| Electricity | residual | 1810.4 | 208.03 | 36.33118 |
| Electricity | BF16 autocast+checkpoint | 736.4 | 177.84 | 4.12640 |

BF16의 오차는 FP32 기준과의 수치 차이다. FP32 checkpoint와 달리 FP32 exact라고 부르지 않는다. A-gradient 오차를 forecasting loss 악화율이라고 해석하면 안 된다.

## 5. 사전에 확인할 수 있었던 저장량 하한

과거 토큰 N개, hidden 폭 d, FP32=4bytes, FP16=2bytes라고 하자.

- 모든 과거 토큰 FP16: 2Nd bytes.
- 인접 쌍 평균 N/2개를 FP32로 저장: 4(N/2)d = 2Nd bytes.

즉 현재 방식은 **평균만 저장해도 모든 토큰 FP16 저장량과 같다.** 여기에 residual sample, index, inverse probability를 더하므로 더 작아질 수 없다.

실제 구현의 dimensions:
- batch8, history256, special4, hidden768.
- pair128, sampled detail32.
- shared payload48개.

payload 하나당:
- 평균 FP32: 3,145,728 bytes.
- special FP32: 98,304 bytes.
- detail FP32: 786,432 bytes.
- int64 index: 2,048 bytes.
- inverse probability FP32: 1,024 bytes.

합계:
- residual: 193,609,728 bytes.
- local FP16: 153,354,240 bytes.
- residual이 26.25% 크고, 38.390625 MiB를 더 저장한다.

이는 기록된 payload 수치와 일치한다. payload 누적합은 GPU peak와 다른 지표지만, 실제 peak 차이 약39MiB와도 방향·크기가 부합한다.

평균을 FP16으로 바꾸면 이 특정 하한은 달라진다. 하지만 그것만으로 실제 peak, gradient variance, BF16+checkpoint 대비 가치까지 해결되는 것은 아니다.

## 6. 불편 추정과 좋은 학습은 다르다

한 pair의 정확한 기여:

u1 h1^T + u2 h2^T
= 2 u_bar h_bar^T + 1/2 (u1-u2)(h1-h2)^T.

새 구현은 첫 항을 보존하고 두 번째 항을 sampling probability로 보정한다. 작은 전수 열거 실험에서는 기대값 오차 8.88e-16이었으며, 평균/차이 항등식도 같은 수준으로 일치했다.

그러나 m회 복원추출의 오차 분산은:

E||Ghat-G||_F^2 = (1/m)[sum_i ||R_i||_F^2/p_i - ||sum_i R_i||_F^2].

현재 p_i는 forward에서 알 수 있는 activation detail norm에 기반한다. R_i에는 backward의 gradient difference도 들어가므로 detail norm만으로 가장 중요한 gradient 기여를 알 수는 없다.

실제 기록:
- ETTm2: 한 번 RMS A오차12.935%,16회 평균오차3.303%.
- Electricity: 한 번36.331%,16회 평균7.213%.

16회 평균은 진단용이다. 학습 시16회를 실행해 평균내면 그 비용을 모두 포함해야 한다. Adam·clipping이 비선형이므로 raw estimator의 기대값이 맞아도 optimizer update나 최종 품질이 보장되지 않는다.

## 7. 중단 gate의 올바른 해석

기존 gate는 A-gradient 오차가 local FP16보다 작을 것을 요구했다. 이는 계산 투자 여부를 정한 보수적 기준이지, 과학적으로 모든 유용한 근사학습에 필요한 조건이 아니다.

다만 이번 중단은 그 엄격한 오차 기준 하나 때문이 아니다. 현재 residual은 local FP16보다 payload/peak/time도 불리하다. 오차 문턱을 완화해도 자원상 추가 가치가 새로 생기지 않는다.

BF16+checkpoint는 현재 동일-batch 측정에서 residual보다 약59.3~59.4% 낮은 peak, 약13.9~14.5% 짧은 forward/backward 시간을 보였다. 이 역시 반올림 보고표에서 계산한 수치이며 full training이나 예측 품질 우열의 증거는 아니다.

## 8. 선행 대조의 범위

CARE/PRAC 항목은 scoped primitive다. 공식 학습 결과를 재현한 것이 아니다.

- CARE: LoRA 저랭크 값과 ridge decoder로 A-gradient를 근사.
- PRAC: rank8 randomized principal + rank8 random tail을 매번 새로 생성. 원 논문의 SVD/lazy-update 전체절차와 다름.
- 두 압축은 residual과 byte matched가 아님.

따라서 ‘CARE보다 gradient가 작으니 더 좋은 방법’, ‘PRAC는 학습이 안 된다’ 같은 결론은 불가하다.

근거:
- CARE-LoRA (2026/arXiv; 이 검토에서 정식 채택 확인 못 함): https://arxiv.org/html/2607.11940v1
- PRAC 원문: https://arxiv.org/html/2602.23111v1
- LoRA-FA 원문 (2023/arXiv 공개): https://arxiv.org/abs/2308.03303

## 9. 다음 개발에 대한 권고

[판정] 현 residual primitive는 보존하고 종료한다. BF16+checkpoint를 메모리 측정의 현실적인 강한 대조로 사용한다. 이를 새 PEFT 기여라고 주장하지 않는다.

‘메모리 효율 PEFT’라는 목적은 유지하되, 새 시도는 **큰 backbone의 과거-token backward 자체를 만들지 않는 방식**으로 좁히는 것이 더 직접적이다. 아래는 아직 결과가 없는 방법 후보이며, 무조건 성공할 것이라는 권고가 아니다.

### 후보: 과거 표현 고정·미래 토큰 적응

- F0를 no_grad로 한 번 실행하여 각 층의 과거 표현/필요한 K,V를 확보한다.
- 미래 예측 토큰과 필요한 REG 토큰만 별도의 학습 경로로 전파한다.
- 각 층에서 미래 query는 같은 입력창의 고정된 과거 표현을 읽는다.
- 미래 토큰의 attention projection에만 LoRA 효과를 적용한다.
- native output head는 유지한다.
- 과거 입력을 평균으로 없애지 않는다. 다른 샘플의 과거나 미래 정답을 읽지 않는다.

핵심적으로 ‘단순히 context.detach()를 붙인다’로 구현하지 않는다. 전체 MLP 등을 autograd로 먼저 실행하면 비용이 남는다. 본체 F0 forward는 no_grad, 미래 토큰 쪽 계산만 backward graph에 있어야 한다.

Chronos-2는 causal prefix LM과 다르다. 공식 encoder mask는 key validity mask이며, 일반적인 과거-only causal cache를 가정해서는 안 된다. 본체의 전체 frozen forward에서 층별 과거 표현을 얻고, adaptation 이후에도 그 과거 표현을 고정한다는 새 계산모델을 명시해야 한다.

처음에는 F0 출력과 일치하도록 구성할 수 있지만, 적응 후 일반 all-token LoRA와 같은 모델이라고 주장하면 안 된다. 이것은 근사 gradient codec이 아니라 제한된 적응 구조다.

### 기대효과와 실패조건

기대: 긴 과거 전체의 MLP/attention backward를 제거하여 더 낮은 메모리·학습시간을 얻으면서 forecast 품질을 유지.

위험:
- 타깃 도메인이 과거 표현 자체를 바꿔야 하면 품질이 나빠질 수 있다.
- frozen K/V 캐시도 context 길이에 비례한다.
- full frozen forward 및 미래 branch 재계산, cache 관리 비용 포함 필요.
- 한 입력창의 캐시를 다른 원점/다른 horizon에 무조건 재사용하면 안 된다.
- 미래 쿼리 수가 커지면 이점이 줄 수 있다.

4096관측을 patch16으로 나누면 과거256토큰이고,48-step출력은3개 forecast patch다. 이것은 역전파 범위를 달리할 근거이지,64배 속도 향상의 증명이 아니다.

### 선행 경계

- LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning (2022/NeurIPS): frozen backbone의 중간표현을 별도 side network가 읽고 본체 역전파를 제거. https://papers.neurips.cc/paper_files/paper/2022/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html
- Activated LoRA: Fine-tuned LLMs for Intrinsics (2025/NeurIPS): 일부 이후 토큰에서만 LoRA를 활성화하여 base prefix KV cache 재사용. https://papers.neurips.cc/paper_files/paper/2025/hash/4d0b6303d4a4811445f69f357bf6def5-Abstract-Conference.html
- EfficientFSL: Enhancing Few-Shot Classification via Query-Only Tuning in Vision Transformers (2026 공개본): frozen ViT의 중간표현을 task queries로 읽는 접근. https://arxiv.org/abs/2601.08499
- TS-Memory (2026/arXiv 공개본): frozen TSFM과 별도 horizon-query 예측모듈을 결합하지만, 해당 모듈은 raw context를 입력으로 하고 retrieval teacher를 증류. https://arxiv.org/html/2602.11550v1

따라서 query-only/side-tuning/부분토큰LoRA 자체는 새롭지 않다. 양방향 시계열 백본에서 native forecast 경로를 보존한 제약과 실제 메모리–예측 품질 효용이 차이를 만드는지 검증해야 한다. 정확한 최초성은 미확정이다.

## 10. 실제로 다음에 할 실험안

[제안; 승인 전 실행하지 않음]

장기간의 새 압축 진단 대신, 구현 최소 검사 후 실제 학습 파일럿으로 간다.

기본 비교:
1. Standard LoRA + BF16 + checkpoint.
2. Frozen backbone + parameter-budgeted forecast head.
3. LST-style side baseline (공식 결과를 재현한 것이 아니면 adapted baseline임을 명시).
4. 과거 표현 고정·미래토큰 적응 후보.
F0는 학습 없이 공통 평가한다.

두 dataset × 두 사전 고정 optimizer recipe × 네 학습 방식 × 첫 seed = 최대16fits.

서로 구조가 다른 무작위 초기화 side와 LoRA에 동일LR만 강제하지 않는다. 각 방식의 합리적인 두 recipe를 학습 전에 고정하고 같은 validation 기회를 준다. 평가 E를 본 뒤 grid/period를 바꾸지 않는다. 자료는 fit-only 두 진단 원점이 아니라 실제 chronological train/V/E windows를 사용한다. 이전에 노출된 기간이면 development로 표기한다.

필수 측정:
- Native probabilistic loss 및 같은 raw-scale 평가.
- 실제 optimizer.step을 포함한 peak allocated/reserved.
- cache 생성, frozen forward, train branch, validation을 포함한 total time.
- 학습 파라미터 수, forward parity, valid temporal/group mask.
- fixed batch 비교 후에만 fixed GPU-memory-budget 비교.

진행 기준 예시(학회 기준 아님):
- BF16+checkpoint LoRA 대비 실제 peak를20% 이상 줄이고,
- 평가 primary loss 악화를1% 상대 이내로 유지하며,
- 단순 frozen head/LST 대조로 같은 품질·자원 이득이 설명되지 않을 것.
- all-arm step0 선택이면 ‘성능 유지 성공’이 아니라 adaptation value 미확보로 보고.

실제 peak가 줄지 않거나 단순 대조보다 추가 가치가 없다면 후보를 확장하지 않는다. 숫자가 안 좋다는 이유로 또 새 저장소나 새 이름부터 만들지 않는다.

## 11. 첨부 CPU 검증의 한계

`independent_math_checks.py`:
- 현재 residual estimator의 정확한 작은 항등식/기대값/분산과 payload byte 수를 검사.
- 보고표 기반 비율은 rounded input이라는 점을 출력에 명시.

`forecast_token_toy.py`:
- 작은 bidirectional Transformer에서 frozen full-context pass를 이용한 query branch의 zero-adapter 출력 일치를 검사.
- toy 최대차 약5e-16, zero-adapter에서 B-gradient가 비영, B변경시 예측변경 확인.
- Chronos의 RoPE, group attention, native quantile head를 구현한 코드가 아니다.
- GPU 메모리 절감, 데이터 예측 정확도, 신규성의 증거가 아니다.

## 12. 요약

지금 결과는 ‘구현 실패’가 아니라 ‘현재 압축 설계가 단순 대조보다 유리하지 않음’이다. 최종 품질은 미측정이지만, 현재 저장량의 하한만으로 local FP16에 비해 불리한 이유를 설명할 수 있다.

다음 개발은 잔차 비율을 계속 바꾸는 것보다, 긴 과거 표현은 frozen memory로 읽고 적은 미래 예측 토큰의 내부 계산만 학습하는 제한적 PEFT 후보를 실제 학습으로 비교하는 편을 제안한다. 이 방향도 선행 기반의 후보이며 확정된 논문 주제가 아니다.