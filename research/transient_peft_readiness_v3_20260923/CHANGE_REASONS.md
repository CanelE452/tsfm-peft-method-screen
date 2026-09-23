# 정정 사유 — v2 문서의 주장 대 이번 실측

2026-09-23. 기준 커밋 `877236b`. **v2 파일은 수정하지 않았다.**

| # | 이전 주장 (v2) | 근거 | 이번 교정 | 미확인 |
|---|---|---|---|---|
| A | 토큰은 "(그룹 × quantile) × 시간패치" 로 배치된다 | `pipeline.py:391–404` 의 분위수 확장을 일반 forward 구조로 오독 | **틀렸다.** 그 확장은 `_prepare_inputs_for_long_horizon_unrolling` 경로다. 일반 forward 내부는 `(행, 토큰, hidden)` = 실측 `[3, 6, 768]` | long-horizon unrolling 경로의 축 배치는 이번 검사 범위 밖 |
| B | — | — | 패치 특징은 `p*3 = 48` (값·마스크·시간인코딩). 실측 `input_patch_embedding in=[3,3,48]`, `p=16` | 3채널 각각의 정확한 내용은 소스 주석 수준 [추정] |
| C | — | — | **분위수 축은 출력 projection 에서 생긴다.** `out_features 336 = 21 quantiles × 16 output_patch_size`, 실측 일치 | — |
| D | "`c_vec` 을 0 으로 초기화하면 B=0 과 무관하게 초기 동일성이 보장된다" | 수식 오독 | **틀렸다.** `c=0` 이면 `out = W0h + kappa·B·A·h` 로 **일반 LoRA** 가 될 뿐 F0 가 아니다. F0 보존에는 `B=0` 이 필요하다. CPU float64 검산으로 확정(차이 0.3626) | — |
| E | — | — | 입력 특징을 추가한 F0 와 raw F0 는 다를 수 있다. 본실험에서 `F0_RAW`/`F0_FEATURE` 를 분리 저장해야 한다 | 이번에 두 view 의 차이를 수치로 재지는 않았다 |
| F | BOPTEST 판정 `AUTH_OR_POLICY_BLOCKED` | DNS 실패를 정책 차단으로 해석 | **과한 판정이었다.** 관찰된 것은 `DNS_RESOLUTION_FAILED` 뿐이다(curl exit 6 = CURLE_COULDNT_RESOLVE_HOST). 인증 요구도 서비스 중단도 확인하지 않았다 | 서비스 생존, 인증 필요 여부, live 버전 |
| G | STAR 는 "학습·추론 모두 patch 경계 안에서 causal" | 초록·본문 요약에서 추정 | **단정할 수 없다.** 인과 마스크의 명시적 근거를 찾지 못했다. "인과성 미확인" 으로 남긴다 | §3.3.1–3.4 의 마스크 서술, SVD 기저의 고정/학습 구분 |
| H | 2×2 로 "교란이 제거된다" | — | 같은 초기 정의가 **같은 최종 학습 상태**를 뜻하지 않는다. LI 와 LM 의 최종 tau/m 은 달라질 수 있으므로 "위치만 바꾼 인과 실험" 이라고 쓰지 않는다 | — |
| I | 삽입 제약을 `IMPLEMENTABLE_WITH_CLASS_LEVEL_PATCH` 로 단정 | v2 감사 | **과했다.** 이번엔 site-packages 를 고치지 않고 `peft.get_peft_model` + 별도 상태 생성기 모듈만으로 gradient·roundtrip 이 통과했다. 클래스 수준 패치가 **필수라는 근거는 없다** | `fit()` 진입점을 쓸 때의 custom module 보존은 미검증 (이번은 `model.forward` 직접 경로) |
| J | (v2 `PRIOR_OVERLAP_MATRIX`) "O 가 이겨도 새 정보가 아니다" | 기존 PETSA 결과 이식 | **폐기한다.** 기존 결과는 다른 데이터·정보·목표다. 이번 설비 과제에서 O 가 좋으면 그것은 "출력 보정으로 충분하다" 는 이 과제의 판단 근거가 된다 | — |

## 새로 드러난 것 — v2 에 없던 사실

**native loss 의 행 평균 분모에 공변량 행이 포함된다.**

```
target 행만        batch=1  per_row=[0.200691]            scalar=0.200691
target+공변량 2행   batch=3  per_row=[1.271318, 0, 0]     scalar=0.423773
공변량 행 기여      0.0, 0.0        (분자에는 0, 마스킹 정상)
target_row/scalar   3.0000 = batch  (분모에는 포함)
```

`_compute_loss` 의 `loss.mean(dim=-1).sum(dim=-1).mean()` 에서 마지막 `.mean()` 분모가 행 수다.
INPUT arm(FI/LI)은 상태를 행으로 넣으므로 같은 target 오차에 대해 더 작은 scalar loss 를 받고,
MOD arm(FM/LM)은 행이 늘지 않는다. **FI/LI 대 FM/LM 을 "순수 위치 효과" 라고 쓸 수 없다.**
`LEARNING_SCALE_CONFOUND` 로 기록하고, 공통 보정 계약을 본실험 전에 정해야 한다.
학습률로 상쇄하거나 native loss 를 다른 목적함수로 바꾸지 않는다.
