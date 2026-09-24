# v4 readiness — 구현·비교 공정성 수정 결과

2026-09-23. 기준 HEAD `63beed9`. 연구 fit 0회, 실데이터 0건. 전부 가상 fixture·CPU.

## 0. 외부에서 확인하는 법

주장만 읽지 말고 근거 파일을 직접 열어 대조할 수 있게 경로를 먼저 둔다.

| 확인하고 싶은 것 | 볼 파일 |
|---|---|
| 12개 기준 전부 통과했나 | `results/transient_peft_readiness_v4_20260923/TEST_RESULTS.json` |
| 어떤 모듈에 무엇이 붙었나 | 같은 폴더 `MODULE_SCOPE.json` (이름 48개 + 각 모듈 shape) |
| hook 이 실제로 무슨 값을 썼나 | `ACTIVE_MOD_ALIGNMENT.json` 의 `hook_debug_audit` 48건 |
| loss 보정이 맞나 | `LOSS_CONTRACT.json` (FP64, atol 1e-10) |
| 파라미터가 진짜 갱신됐나 | `UPDATE_AUDIT.json` (5 arm x 3 step) |
| 저장·복원이 같은 예측을 주나 | `FULL_ROUNDTRIP.json` (prediction max_abs 0.0) |
| 초기 동일성 | `INITIAL_VIEWS.json` |
| 실행 횟수·해시·재시도 | `RUN_MANIFEST.json` |
| 검사 stdout | `raw_logs/*.txt` |

직접 돌려보려면 `python experiments/transient_peft_readiness_v4_20260923/run_all.py` 다.
CPU 로 수 분이면 끝나고 연구 fit 은 0회다.

## 1. 목적

v3 의 25 PASS 를 구현 완료로 보지 않고, 지적된 결함 9개를 실제 코드로 고쳐
"데이터만 확보되면 실험을 시작해도 되는가" 를 두 축으로 판정한다.

## 2. 기준 HEAD / 환경

```
HEAD      63beed98113546dbf0a747d073e7aae082093ed8 (기준과 일치, dirty 0)
env       python 3.11.15 / torch 2.8.0+cu128 / chronos 2.3.2 / peft 0.18.1 / CPU
```

## 3. v3 에서 고친 실제 버그

| # | v3 결함 | 무엇이 문제였나 | v4 수정 |
|---|---|---|---|
| P1 | MOD hook 이 축을 shape 로만 매칭 | `GroupSelfAttention` 은 `layers.py:402` 에서 `batch time d -> time batch d` 로 전치한다. shape 매칭이면 group 에서 조용히 생략되거나, 행 수 == 토큰 수일 때 엉뚱한 축에 적용될 수 있었다 | 클래스(`TimeSelfAttention`)로 모듈을 고르고 축을 이름으로 안다. 불일치는 `ShapeMismatch` 예외 |
| P2 | c 가 한 task 만 가정 | 여러 task 가 한 batch 에 있으면 서로 다른 state 를 받아야 하는데 공유됐다 | `build_c_rows` 가 각 row 의 `group_id` 로 자기 task state 를 lookup |
| P3 | 정렬 검사가 frozen 경로 | active MOD 가 붙은 모델에서 검증한 것이 아니었다 | B≠0·c≠0 활성 경로에서 8개 재검사 |
| P4 | m=0 과 c=0 을 혼동 | Modulator 에 bias 가 있으면 학습 후 `c(0)=bias` 로 중립이 깨진다 | `bias=False` + `token_valid_mask` 를 곱한다. mask 는 observation mask 를 실제 인자로 받는다 |
| P5 | MOD fixture 가 target-only | 최종 계약에서는 모든 arm 이 같은 raw command 를 받아야 한다 | fixture 가 target row + raw command row 를 모든 arm 에 제공 |
| P6 | loss row 분모 교란 | 공변량 행이 분모를 희석해 INPUT/MOD 의 learning scale 이 달랐다 | `TARGET_ROW_MEAN_CORRECTION` 구현·검산 |
| P7 | backward 까지만 | "gradient 가 있다" 와 "optimizer 가 파라미터를 갱신한다" 는 다르다 | 5 arm × 3 step 실제 update |
| P8 | roundtrip 이 tau 중심 | 전체 prediction parity 를 새 프로세스에서 안 봤다 | modulator·LoRA·state·hook 재구성 후 prediction SHA 일치 확인 |
| P9 | `fit()` custom module 보존 미검증 | — | site-packages 를 고치지 않고 `TransientPEFTModel` wrapper + 공통 trainer 경로로 대체. `pipeline.fit()` 자체는 여전히 미검증 |

## 4. module scope

```
num_layers 12   TimeSelfAttention LoRA 48   GroupSelfAttention LoRA 48   (12 x 4 각각)
modulation 적용   time 48 / 48,  group 0,  skipped 0,  hook 호출 48
group LoRA 는 여전히 trainable (48개 전부 확인, 표본 4개가 아니다)
```

`MODULE_SCOPE.json` 에 time/group/전체 LoRA module 이름 목록과 48개 module 각각의 실제
`A out` / `c` shape 를 남겼다. 둘은 전부 `[4, 4, 8]` 로 일치한다.

설계 범위를 명시한다. response-state modulation 은 TimeSelfAttention LoRA 에만 적용했다.
"모든 attention 을 상태 조건화했다" 가 아니다. group attention 조건화는 이번 범위에서 제외했다.

## 5. active multitask modulation — 8/8

| 검사 | 결과 |
|---|---|
| 같은 task 의 target/command row 가 같은 state | 차이 0.0 |
| 다른 task row 는 다른 state | 차이 0.0566 |
| row permutation + group/state mapping 동시 변경 후 재정렬 일치 | 차이 0.0 |
| task B command 변경 → task A 불변 | 차이 0.0 |
| task A command 변경 → task A 변함 | pred·c 모두 변함 |
| patch boundary impulse 가 의도한 토큰에만 | patch0 0.0, patch1 >0, REG 0.0 |
| batch 교차 호출 후 carry-over 없음 | 차이 0.0, holder 비움 확인 |
| D 반례: 행 수(4) == 토큰 수(4) | hook 48회 정상 (과거 shape hook 이면 축이 모호했을 fixture) |

hook 48건 각각에 module_name, row 수, group_ids, 토큰별 `|c|` 최대, row별 `|c|` 최대를
`ACTIVE_MOD_ALIGNMENT.json` 의 `hook_debug_audit` 로 남겼다. 예: `group_ids [0,0,1,1]`,
토큰별 `[0.0339, 0.0104, 0.0, 0.0041]` — 세 번째가 REG 라 0 이다.

## 6. neutral token contract — 6/6

REG, 완전 padding, all-masked command patch, H 밖 future token 은 학습된 weight 에서도 `c` 가 정확히 0.0 이다.
Modulator 는 `bias=False` 이며, bias 를 넣으면 검사가 실패한다는 것도 확인했다(`c(m=0)=0.3`).
부분 관측 패치는 마지막 valid 시점만 쓴다.

## 7. 초기 동일성 — F0_BASE / F0_STATE_INIT (7/7)

`INITIAL_VIEWS.json`. optimizer 를 쓰지 않는 forward 전용 검사다.

```
F0_BASE        = raw rows(target + command), LoRA B=0, c=0          rows 4
F0_STATE_INIT  = F0_BASE rows + 초기 고정 상태 행                    rows 8
```

| 검사 | max_abs |
|---|---|
| L0 step0 == F0_BASE | 0.0 |
| FM step0 == F0_BASE (modulator init 0 → c=0) | 0.0 |
| LM step0 == F0_BASE | 0.0 |
| FI step0 == F0_STATE_INIT | 0.0 (정의상) |
| LI step0 == F0_STATE_INIT | 0.0 |
| FI·LI 가 같은 tau init 에서 출발 (상태 행 자체 비교) | 0.0 |

`F0_BASE` 와 `F0_STATE_INIT` 의 target row 예측 차이는 2.1395 다. 이것은 실패가 아니라
정보/표현 효과이며, 서로 다른 input view 끼리 동일 예측을 요구하지 않는다는 계약대로 기록만 한다.

FI 항목은 `F0_STATE_INIT` 을 FI 의 step0 으로 정의했으므로 정의상 0 이다. 실질 검사는 LI 쪽,
즉 tau 를 학습 파라미터로 둔 구성이 같은 초기값에서 같은 예측을 내는가다.

## 8. loss correction — TARGET_ROW_MEAN_CORRECTION

FP64, atol 1e-10. 고정 prediction/target/mask 로 모델과 분리해 검산했다.

```
공변량 행   native scalar    corrected scalar   correction factor
    0       2.11462178       2.11462178              1.0
    1       1.05731089       2.11462178              2.0
    2       0.70487393       2.11462178              3.0
    4       0.42292436       2.11462178              5.0
```

native 는 행 수대로 1/N 희석되고 corrected 는 불변이다. target-prediction gradient 도 모든 경우 동일하고하며,
zero-loss 행의 gradient 는 0, 방법 A(배수)와 방법 B(active row mean)가 일치한다.

underlying quantile loss 는 native 와 동일하고 row reduction 만 고쳤다.
"완전히 native scalar loss 와 동일" 이라고 부르지 않는다.

## 9. optimizer updates — 5/5 OK

```
arm  trainable(lora / modulator / state)   loraB변화  mod변화  tau변화  head동결
L0   1,179,648 / 0  / 0                      O         -        -        O
FI   1,179,648 / 0  / 0                      O         -        -        O
LI   1,179,648 / 0  / 4                      O         -        O        O
FM   1,179,648 / 16 / 0                      O         O        -        O
LM   1,179,648 / 16 / 4                      O         O        O        O
```

optimizer 는 `requires_grad=True` 를 명시 인벤토리한 뒤 만들었다. 상태 생성기·modulator 가 모델 바깥
객체라 누락되는 일이 없음을 파라미터 수로 확인했다. 총 15 update (상한 15).

## 10. full roundtrip — 10/10

새 프로세스에서 base 재로드 → LoRA config 재생성 → adapter load → wrapper/state/modulator 재생성 →
hook 재설치 → checkpoint load → 같은 fixture forward.

```
log_tau exact / modulator exact / frozen head hash 동일 / base weight hash 동일
prediction SHA 동일, max_abs 차이 0.0 (사전 고정 tol 1e-6) / c_rows SHA 동일
hook coverage 동일 / unexpected keys 0
```

state_gen·modulator 는 `strict=True` 로 올렸다. adapter 쪽 `missing_keys` 는 170개인데, 이는 adapter
체크포인트가 base weight 를 담지 않기 때문이고 base 는 사전에 재로드된다. 그래서 판정은 `unexpected_keys == 0`
과 prediction SHA 일치로 한다. "전부 strict" 라고 쓰지 않는다.

scope 는 checkpoint inference roundtrip 이다. training resume parity 는 별도 검사이며 하지 않았다.

## 11. data access

```
DATA_ACCESS_BLOCKED / UPSTREAM_DNS_ZONE_UNSERVED
API 호출 0, smoke 0, 설정 변경 0
```

작업 시작 시의 1회 확인은 `DNS_RESOLUTION_FAILED` 였다. 이후 사용자가 직접 같은 요청을 실행해
같은 실패를 확인했고, 원인이 우리 망인지 서비스인지 가르기 위해 1회 진단했다 (`DATA_ACCESS.json`).

```
github/pypi/huggingface 해석 정상          -> 이 PC 의 DNS 는 살아 있다
api/www/apex boptest.net 전부 실패          -> 서브도메인이 아니라 zone 단위
dig +cd 에도 SERVFAIL                      -> 우리 쪽 DNSSEC 검증 문제 아님
+trace: .net 이 cloudflare NS 2대로 위임     -> 등록은 살아 있다
그 NS 2대에 직접 질의 -> REFUSED             -> zone 을 서빙하지 않는다
같은 NS 에 cloudflare.com 질의 -> NOERROR    -> 서버는 정상, boptest.net zone 만 없다
Anthropic 측 fetch -> getaddrinfo ETIMEOUT   -> 우리 망 밖에서도 해석 안 됨
```

전 세계 어떤 resolver 에서도 해석되지 않는 상태다. 따라서 다른 네트워크로 옮기거나 resolver 를
바꿔도 열리지 않는다. 공식 문서는 여전히 `https://api.boptest.net/<request>` 를 안내하지만
2026-09-23 기준 그 zone 이 죽어 있다.

## 12. 판정 — 두 축

```
MODEL_READY            §13 의 12개 기준 전부 PASS
DATA_ACCESS_BLOCKED    서비스에 도달하지 못함
```

데이터 실패를 MODEL_NOT_READY 로 바꾸지 않았고, 모델 통과를 DATA_READY 로 올리지도 않았다.

## 13. 아직 과학적으로 미검증인 것

```
TRANSIENT difficulty 존재   미검증
일반 LoRA residual          미검증
LM > LoRA                   미검증
novelty                     미확정
```

이번 작업의 성공은 "논문 방법이 된다" 가 아니라 "구현·비교 공정성 문제가 제거되어 실제 데이터
실험을 시작해도 되는 상태" 까지다.

## 14. 여섯 번의 실패와 원인 (보존)

| # | 증상 | 원인 | 수정 |
|---|---|---|---|
| 1 | 검사3 c_diff=0.0 | fixture 의 `cmd_shift` 가 `t==1` 고정이라 "task A 를 바꾼다" 가 성립 안 함 | `shift_task` 인자 |
| 2 | beyond_H c≠0 | `H=p+1` 이라 두 번째 출력 패치에도 유효 구간이 있었다. 검사 전제가 틀림 | 완전히 H 밖인 토큰 사용 |
| 3 | backward a second time | `state_rows` 를 루프 밖에서 만들어 graph 재사용 | 매 step batch 재생성 |
| 4 | LM tau_changed=False | `exp(log_tau)` 표시값 비교로 작은 delta 를 놓침 | log_tau 원값 비교 |
| 5 | LM tau_changed=False (다른 원인) | gradient 1.65e-04 는 도달하나 lr=1e-3 에서 delta 1.65e-07 이 float32 정밀도 아래 | 모든 arm 동일하게 lr 1e-1. PASS 기준(delta>0)은 바꾸지 않았다 |
| 6 | D 반례가 PASS 인데 전제가 거짓 | `n_ctx_patch=1` 이면 token 3, row 4 다. 두 수를 주석에 적어 놓고 "행 수 == 토큰 수" 라고 이름 붙였다. 검사는 통과했지만 의도한 조건을 한 번도 만들지 않았다 | `n_ctx_patch=2` 로 token 4 = row 4 를 실제로 만들고 전제를 `assert` 로 고정. 재실행 8/8 |

허용오차·assertion 을 완화한 곳은 없다. 시도별 코드는 `RUN_MANIFEST.json` 의 `retries` 에 경로를 남겼다.

### 지시문 재대조에서 찾은 누락 4건

1차 보고 후 지시문 원문을 절별로 다시 읽어 아래를 보완했다. 셋은 기록 누락이고 하나는 검사 자체가 없었다.

| 조항 | 빠졌던 것 | 보완 |
|---|---|---|
| §2 | module name 전체 목록, module별 A out/c shape (표본 2개만 있었다) | 48개 전부 기록. group LoRA trainable 도 표본 4개 → 48개 전수 |
| §4 | hook debug audit (module_name / row / group_id / token index / c 요약) | 48건 저장 |
| §7 | 초기 동일성 검사 자체가 없었다. `§13-7` 을 fixture 구조 설명만으로 PASS 처리했다 | `test_initial_views.py` 신설, 7/7 실측 |
| §8·§10·§14 | trainable 이름, prediction max_abs, base weight hash, 소요시간 | 전부 기록. 단 trainable 이름은 사후 재구성이라고 명시 |

§7 이 제일 무겁다. 검사 없이 PASS 로 적혀 있었고, 지금은 L0/FM/LM 이 `F0_BASE` 와 max_abs 0.0,
LI 가 `F0_STATE_INIT` 과 0.0 임을 실제로 측정했다.

## 15. 실행 횟수와 raw log

```
model_loads 45   forward 101   backward 18   optimizer_updates 15 (상한 15)
api_calls 0   installs 0   research_fits 0
wall  module_scope 4.4s / multitask 4.8s / initial_views 11.0s / roundtrip 10.1s
```

optimizer_updates 와 backward 는 계측값이다. 보완 재실행분의 model_loads·forward 는 코드상
호출 지점에서 센 산정값이라 계측이 아니다. 1차 실행의 per-test wall time 은 남기지 않았고,
`loss_contract` 와 `optimizer_updates` 는 재실행하지 않아 미계측이다.

`results/.../raw_logs/*.txt` 에 검사 stdout 을 남겼다. `test_optimizer_updates` 로그는 없다 —
update 상한을 이미 소진해 재실행하지 않았고, 그 검사의 근거는 `UPDATE_AUDIT.json` 이다.
같은 이유로 §8 이 요구한 trainable '이름' 목록은 update 실행 당시 기록이 아니라
`INITIAL_VIEWS.json` 의 `trainable_inventory_names`(같은 builder, optimizer step 0)로 채웠다.

## 16. 다음 딱 한 단계

공개 서비스는 선택지에서 뺀다. 위 진단으로 "다른 환경에서 응답을 받아 온다" 는 경로가 막혔다 —
zone 자체가 서빙되지 않으므로 망을 바꿔도 같다.

남은 선택지는 셋이고, 어느 것도 이번 단계의 금지사항(설치·Docker) 안에서는 실행할 수 없다.
사용자 결정이 필요하다.

| 선택지 | 내용 | 걸리는 것 |
|---|---|---|
| 로컬 BOPTEST | `docker compose up web worker provision` → `http://127.0.0.1:8000` (swagger `/docs`) | v4 는 설치·Docker 금지. BOPTEST 의 표준 사용법이라 기술적 장애는 없다 |
| 공개 서비스 복구 대기 | zone 이 복구되면 원래 계획대로 | 복구 시점 미상, 통제 불가 |
| 다른 설비 도메인 데이터 | 과도응답이 있는 다른 공개 데이터 | difficulty 정의부터 다시 해야 한다 |

로컬 BOPTEST 가 가장 곧다. 승인해 주시면 다음 단계는 `bestest_hydronic_heat_pump` 를 띄워
`/inputs`·`/measurements` 로 신호 이름·단위·시각 계약을 채우는 것이다.
