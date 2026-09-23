# Chronos-2 삽입 지점 감사

2026-09-23. 소스만 읽었다. 구현·학습·추론 0회.
경로: `.venv/lib/python3.11/site-packages/chronos/chronos2/`

## 0. 판정

```
IMPLEMENTABLE_WITH_CLASS_LEVEL_PATCH
```

wrapper 만으로는 부족하다. `fit()` 이 config 에서 모델을 **새로 만들고** `state_dict` 만 복사하기
때문에(아래 §5), 인스턴스에 붙인 custom module 은 학습 경로로 전달되지 않는다. v2 실험에서
같은 성질을 실측으로 확인한 바 있다(`results/service_axis_v2_20260922/V0F_NZ_WIRING.json`,
클래스 수준 교체 시 학습 경로 loc 3.0, 인스턴스 패치였다면 1.5).

## 1. 기본 LoRA target_modules — 출력층이 포함된다

`pipeline.py:207–216`

```python
lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=[
        "self_attention.q", "self_attention.v",
        "self_attention.k", "self_attention.o",
        "output_patch_embedding.output_layer",   # <- native 출력층
    ],
)
```

**attention-only config 작성은 가능하다.** `lora_config` 인자에 `LoraConfig` 또는 dict 를 주면
기본값을 쓰지 않는다(`pipeline.py:217–222` 의 분기). 출력층 동결을 지키려면 위 리스트에서
마지막 항목을 빼고 명시해야 한다.

## 2. native output head 가 실제로 동결됐는지 검사하는 방법

`model.py:265` 에서 `self.output_patch_embedding = ResidualBlock(...)`, `:732` 에서
`quantile_preds = self.output_patch_embedding(forecast_embeds)`.

검사 절차(학습 전후 비교):

```
1. fit 전   output_patch_embedding 의 모든 파라미터를 flatten -> sha256
2. fit 후   반환된 pipeline 에서 같은 해시를 다시 계산
3. 일치해야 동결.  추가로 named_parameters() 에서 requires_grad=True 인 항목을 전부 출력해
   output_patch_embedding 이 없는지 확인한다.
```

v2 에서 같은 방식(기본 가중치 hash 불변 확인)을 이미 썼다.

## 3. 공변량 텐서 shape 과 토큰 경로

```
preprocess.py:22   future_covariates: (n_variates, prediction_length), float32
preprocess.py:63   미지정 시 NaN 으로 채움
model.py:355       future_covariates.shape[-1] <= num_output_patches * output_patch_size 강제
model.py:465       future_covariates 의 비마스크 위치 NaN 을 거부
model.py:226       input_patch_size == output_patch_size 를 assert
model.py:241       in_dim = input_patch_size * 3   (값·마스크·? 3채널로 보인다 [추정])
model.py:250       patch_size=input_patch_size, patch_stride=input_patch_stride
model.py:268       out_dim = num_quantiles * output_patch_size
```

## 4. group / time mask 와 patch indexing

```
model.py:99   class Chronos2Encoder
model.py:124–128  group_ids 로 group_mask = group_ids[:,None] == group_ids[None,:]
model.py:139  group_time_mask = rearrange(group_time_mask, "q b t -> t 1 q b")
model.py:165  inputs_embeds 와 결합
pipeline.py:391–404  group_ids 를 quantile 축으로 확장하고
                     group_ids * len(unrolled_quantiles) + arange 로 재배치
```

즉 토큰은 **(그룹 × quantile) × 시간패치** 로 배치된다. 같은 group_id 이고 같은 quantile 인
계열끼리만 섞인다.

## 5. fit 이 모델을 재생성하는 방식 — 여기가 핵심 제약

`pipeline.py:200–203`

```python
# Create a copy of the model to avoid modifying the original
config = deepcopy(self.model.config)
model = Chronos2Model(config).to(self.model.device)
model.load_state_dict(self.model.state_dict())
```

**함의 세 가지.**

1. custom module(tau, state generator)을 인스턴스에 붙여도 `Chronos2Model(config)` 는 그것을
   만들지 않는다. config 에 없고 state_dict 에도 없다.
2. 따라서 `Chronos2Model.__init__` 또는 `forward` 를 **클래스 수준에서** 바꿔야 새 인스턴스에도
   적용된다. ORIG 와 수정본은 반드시 다른 프로세스에서 돌린다(클래스 교체가 전역이므로).
3. `load_state_dict` 는 기본이 strict 다. custom 파라미터를 추가하면 키 불일치가 날 수 있으므로,
   custom 파라미터는 **LoRA 어댑터 쪽(peft 가 관리)** 에 얹거나 `load_state_dict` 이후에
   부착하는 순서를 지켜야 한다. 이 순서는 구현 시 실측으로 확인해야 한다 [미검증].

## 6. checkpoint save/load 에서 custom 파라미터 보존

v2 실측: `extra_trainer_kwargs` 로 `save_strategy="steps"`, `save_steps` 를 넘기면 중간
체크포인트가 저장되고, `Chronos2Pipeline.from_pretrained` 가 adapter config 를 찾아 되읽는다.
그러나 **tau·state generator 가 peft adapter 바깥에 있으면 그 경로로 저장되지 않는다** [추정].
별도 저장·복원 경로를 만들고 재생 일치를 검사해야 한다.

## 7. m(t) 를 어느 토큰에 대응시킬 것인가

m 은 시간축 상태이고 토큰은 패치 단위이므로, **`input_patch_size` 간격으로 집계**해야 한다.

```
m_raw   (batch, T_context + H)        15분 격자 등 원 시간축
        |  패치 경계에서 집계 (마지막 값 또는 평균 — 봉인 전 확정)
m_patch (batch, n_patches)
        |  group·quantile 축 확장 (pipeline.py:391–404 과 같은 규칙)
m_tok   (batch * n_groups * n_quantiles, n_patches)
        |  c(·) 로 rank 차원 투영
c_vec   (..., n_patches, r)
```

삽입 pseudocode (LoRA 가 붙는 각 attention projection 에서):

```
# h        : (B, N_tok, d_model)      백본 은닉
# A        : (d_model, r)             LoRA down
# Bm       : (r, d_model)             LoRA up,  초기 0
# c_vec    : (B, N_tok, r)            m_patch 로부터 생성, 초기 0
z  = h @ A                            # (B, N_tok, r)
z  = z * (1.0 + c_vec)                # diag(1 + c(m)) 에 해당
out = h @ W0 + z @ Bm                 # W0 동결, Bm=0 이므로 초기 출력 = F0
```

`c_vec` 을 0 으로 초기화하면 B=0 과 무관하게 초기 동일성이 보장된다. 첫 수 회 업데이트에서
gradient 가 실제로 c 경로로 흐르는지 확인해야 한다(B=0 때문에 첫 step 상류가 0인 것과
gradient 단절을 혼동하지 않는다).

## 8. future-known command plan 정렬

`future_covariates` 는 `(n_variates, prediction_length)` 이고 `num_output_patches *
output_patch_size` 이내여야 한다(`model.py:355`). 명령 계획을 여기에 넣으면 horizon 과 자동
정렬된다. 다만 **명령 발행시각·유효시각·실행값의 구분**은 데이터 쪽 계약이며 이 API 가
보장해 주지 않는다.

## 9. 아직 확인하지 못한 것

- `model.py:241` 의 `in_dim = input_patch_size * 3` 에서 3채널의 정확한 의미 [추정]
- `load_state_dict` strict 모드에서 custom 파라미터 추가 시의 실제 동작 [미검증]
- peft adapter 바깥 파라미터의 체크포인트 보존 여부 [미검증]
- 패치 경계 집계 규칙(마지막 값 대 평균)이 성능에 주는 차이 — 봉인 전 TRAIN-only 에서 확정
