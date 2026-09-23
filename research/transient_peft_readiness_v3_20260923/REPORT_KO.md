# v3 준비 검사 결과

2026-09-23. 기준 커밋 `877236b`. 감사 문서를 다시 만들지 않고 **실제로 실행한 검사 결과**만 적는다.

## 1. 한 줄

모델 배선은 통과했고 데이터는 여전히 막혔다. **연구 fit 0회, optimizer update 0회.**
새로 드러난 것은 native loss 의 행 평균 분모가 INPUT/MOD 비교를 오염시킨다는 사실이다.

## 2. 검사 결과 — 25 PASS / 0 FAIL / 1 NOT_RUN / 교란 1 · 확인 1

| 검사 | 기준 | 관측 | 상태 |
|---|---|---|---|
| 토큰 축 순서 | 분위수 축이 출력 projection 이후인가 | `out_features 336 = 21 × 16` 일치 | PASS |
| encoder 축 형상 | `(행, 토큰, hidden)` | attention q in `[3, 6, 768]` | PASS |
| 패치 특징 차원 | `p × 3` | `input_patch_embedding in=[3,3,48]`, p=16 | PASS |
| 수식 B=0 | 원 선형층과 동일 | 차이 0.0 | PASS |
| 수식 c=0 | 일반 LoRA 와 동일 | 차이 0.0 | PASS |
| 수식 c=0 ≠ F0 | **F0 가 아니어야 한다** | 차이 0.3626 | PASS |
| 수식 c≠0 | 출력이 바뀌어야 한다 | 바뀜 | PASS |
| 스칼라 fixture | W0=2,A=B=h=1,κ=1,c=0 → 3 | 3.0 (원 출력 2) | PASS |
| **loss 분모** | 공변량 행이 분모에 드는가 | `target_row/scalar = 3.0000 = batch` | **교란 확인** |
| FI 고정성 | 상태 학습 파라미터 0 | 0, grad None | PASS |
| **LI gradient** | log_tau 까지 유한 gradient | `grad_norm = 0.012278` | PASS |
| LoRA 공유 | 두 arm 모두 attention LoRA 학습 | 1,179,648 (0.978%) | PASS |
| roundtrip tau | 새 프로세스 정확 복원 | 일치, strict 로드 unexpected 0 | PASS |
| 동결 출력층 | 해시 보존 | `d6c8634b24a455b8` 일치 | PASS |
| **상태 정렬 6개** | §4.4 | 아래 §4-1 | **6/6 PASS** |
| 토큰 대응 표 | §4.3 | 6 토큰 · 행 3개 매핑 | PASS |
| MOD 배선 | §6.3 c 초기 0, B=0 | 첫 step mod/tau grad 0 (정상) | PASS |
| MOD active-path | tau 까지 gradient | mod 3.116e-03, tau 3.661e-05 | PASS |
| F0_RAW vs F0_FEATURE | §2-E 다른가 | 정규화 차이 0.5630 | **확인** |
| L0 의 F0_RAW 보존 | B=0 | 0.0e+00 | PASS |
| FI 의 F0_FEATURE 보존 | B=0 | 0.0e+00 | PASS |
| BOPTEST smoke | §3.3 | 미실행 (도달 실패) | NOT_RUN |

## 3. 데이터 트랙 — `DNS_RESOLUTION_FAILED`

```
getent ahosts api.boptest.net          exit 2
curl -I https://api.boptest.net        exit 6 = CURLE_COULDNT_RESOLVE_HOST
재검사 1회                              동일 (일시 오류 아님)
대조 ibpsa.github.io                    HTTP 200
```

API 호출 0회(select/initialize/advance/stop 전부), smoke 미수행, 설정 변경 0건.
**서비스 생존·인증 필요 여부·live 버전·testcase 목록은 모두 미확인이다.**
v2 의 `AUTH_OR_POLICY_BLOCKED` 는 과한 판정이었고 `DNS_RESOLUTION_FAILED` 로 정정한다.

## 4. 모델 트랙 — 무엇이 실제로 확인됐나

**토큰 구성이 확정됐다.** `[context 패치 3] + [REG 1] + [출력 패치 2] = 6`
(`model.py:600–604`, `use_reg_token`). m(t) 를 붙일 때 REG 와 전부 마스킹된 패치는 중립으로 둔다.

**F/L 축이 배선 수준에서 성립한다.** FI 는 상태 학습 파라미터 0, LI 는 log_tau 까지 유한한
gradient 가 도달한다. 두 arm 모두 attention LoRA 를 학습하며 출력층은 대상에서 제외했다.

**저장·복원이 된다.** custom tau 는 peft adapter 경로로 저장되지 않으므로 별도 파일이 필요하고,
새 프로세스에서 `strict=True` 로 정확히 복원됐다. native 출력층 해시도 보존됐다.

**v2 의 구현 판정이 과했다.** `IMPLEMENTABLE_WITH_CLASS_LEVEL_PATCH` 라고 썼으나, site-packages 를
고치지 않고 `peft.get_peft_model` + 별도 상태 생성기만으로 통과했다. 다만 이번은 `model.forward`
직접 경로이며 **`fit()` 진입점에서의 custom module 보존은 여전히 미검증**이다.


## 4-1. 상태 정렬 (§4.4) — 6/6

| 검사 | 관측 | 결과 |
|---|---|---|
| 행 순서 바꿔도 같은 task 결과 동일 | target 행 차이 5.72e-06 | PASS |
| 다른 task 명령을 바꿔도 이 task 불변 | 차이 0.00e+00 | PASS |
| 패치 경계 명령이 의도한 위치에 반영 | 원시 recurrence 와 정확 일치, REG 중립, 좌측 padding 15 | PASS |
| batch 간 상태 누출 없음 | 같은 입력이면 항상 같은 상태 | PASS |
| 미허용 미래 target 을 바꿔도 예측 불변 | 차이 0.00e+00 | PASS |
| 허용된 명령 계획 변경은 상태에 반영 | 미래 상태 차이 5.0000, 과거 0.0 | PASS |

토큰 대응은 `[context 패치 3] + [REG 1] + [출력 패치 2]` 이고,
각 패치의 원시 인덱스·좌우 padding·미래 패치가 대표하는 명령 구간을 `STATE_ALIGNMENT.json` 의
`token_map` 에 기록했다. 패치 상태는 **마지막 유효 시점**으로 집계하고 REG 와 전부 padding 인
패치는 중립(0)으로 둔다. 이는 최소 구현 선택이며 성능 최적값이 아니다.

## 4-2. MOD 경로 (§6.3)

`c` 를 0 으로 초기화하고 `B=0` 인 첫 step 에서 modulator·tau gradient 가 0 인 것은 정상이다.
별도 active-path fixture 에서 `B` 에 작은 비영값을 넣으니 gradient 가 modulator
(3.116e-03)와 tau(3.661e-05)까지 도달했다.
판정 `MOD_PATH_WIRED_OK`. 이 fixture 는 배선 검사 전용이며 본학습 초기화 변경이 아니다.

## 4-3. 입력 view (§2-E)

`F0_RAW` 와 `F0_FEATURE` 는 실제로 다르다(정규화 차이 0.5630).
따라서 초기 동일성은 **같은 view 안에서만** 요구한다. raw view 를 쓰는 L0/FM/LM 은 F0_RAW 와,
feature view 를 쓰는 FI/LI 는 F0_FEATURE 와 비교했고 둘 다 B=0 에서 보존됐다.

## 5. 새로 드러난 교란 — `LEARNING_SCALE_CONFOUND`

```
target 행만        batch=1  per_row=[0.200691]            scalar=0.200691
target+공변량 2행   batch=3  per_row=[1.271318, 0, 0]     scalar=0.423773
```

공변량 행은 분자에 0 을 기여하지만(마스킹 정상) `loss.mean(-1).sum(-1).mean()` 의 마지막 평균
분모에는 들어간다(`target_row/scalar = 3.0000`). INPUT arm 은 상태를 행으로 넣어 행이 늘고,
MOD arm 은 늘지 않는다. **FI/LI 대 FM/LM 을 "순수 위치 효과" 로 해석할 수 없다.**

본실험 전에 모든 arm 에 적용할 공통 보정 계약을 정하고 검산해야 한다. 학습률로 상쇄하거나
native loss 를 다른 목적함수로 바꾸는 방식은 쓰지 않는다.

## 6. 세 번의 실패와 원인 (보존)

| 시도 | 증상 | 원인 | 수정 |
|---|---|---|---|
| 1 | `element 0 does not require grad` | backbone 전체 동결 + LoRA 미부착 → 학습 파라미터 0. v3 §6 요구를 빠뜨림 | peft LoRA(q,k,v,o) 부착, 출력층 제외 |
| 2 | `log_tau.grad = nan` | target 행의 future_covariates 를 NaN 으로 넣고 마스크 미지정 | 값 0 + `future_covariates_mask` 명시 |
| 3 | `log_tau.grad = nan` (다른 원인) | anomaly detection 으로 특정: `AsinhBackward0` NaN. `_compute_loss` 가 NaN 이 든 future_target 을 instance_norm→arcsinh 에 통과시키고, loc_scale 이 graph 위 context 에 연결돼 NaN 이 역전파 (`model.py:532`, `chronos_bolt.py:120`) | `future_target` NaN → 0, `future_target_mask` 명시 |

허용오차·assertion 을 느슨하게 바꾸지 않았다. 실패는 원인을 특정한 뒤에만 수정했다.
시도별 코드는 `/tmp/attempt{1,2,3}_*.py` 로 보존했다.

## 7. 현재 가설 상태

```
연구 데이터 성능   미평가 (연구 fit 0회)
정보 가치          미검증 (설비 데이터 없음)
신규성             미확정 (STAR §3.3.1–3.4 인과 마스크 미확인)
```

배선이 통과한 것은 **구현이 가능하다**는 뜻이지 방법이 유효하다는 뜻이 아니다.

## 8. 예산 사용

```
허용            forward 64 / backward 24 / update 24 / CPU 30분 / GPU 30분
실제            model_load 10, forward 24, backward 10, optimizer update 0
                연구 fit 0, API select/advance/stop 0, 설치 0
검사 장치        전부 CPU. GPU 미사용
```

정확한 집계는 `results/.../RUN_MANIFEST.json` 의 `actual_counts` 와 각 JSON 의 `ledger` 에 있다.
