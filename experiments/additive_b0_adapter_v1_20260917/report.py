"""Render the completed additive comparison without automatic paper PASS rules."""
import pandas as pd
from .common import *
from experiments.outlier_signal_peft_v1_20260917.report import table

def report():
    assert read(OUT/'verification.json')['status']=='VERIFIED';assert read(OUT/'publication_audit.json')['status']=='VERIFIED'
    d=read(OUT/'scientific_decision.json');summary=pd.read_csv(OUT/'panel_summary.csv');seed=pd.read_csv(OUT/'seed_panel_summary.csv');scores=pd.read_csv(OUT/'scores_by_condition.csv');effects=pd.read_csv(OUT/'paired_effects.csv')
    costs=pd.read_csv(OUT/'resources.csv');costs=costs[costs.seed.isin([81551,81552])].groupby(['source','arm'])[['trainable_parameters','additional_train_seconds','validation_seconds','train_peak_allocated_mib','E_inference_seconds','E_peak_allocated_mib']].mean().reset_index()
    selected=pd.DataFrame(read(OUT/'MODEL_SELECTION.json'))[['source','arm','seed','lr','step','objective']]
    raw=scores.groupby(['source','arm'])[['mae','nrmse','pinball','crossing']].mean().reset_index()
    ab=pd.read_csv(OUT/'V_ablation_effects.csv');ab=ab[ab.condition.isin(['REFERENCE','POINT8','BURST8','SHIFT4','SHIFT8'])].groupby(['source','condition','mode'])[['normal_nmae','altered_nmae','error_increase_pct']].mean().reset_index()
    res=read(OUT/'execution_resource_summary.json');v=read(OUT/'verification.json');baseline=read(OUT/'shared_baseline_cost.json')
    body=f'''# B0 유지 + 추가 어댑터 비교 결과

실행 **COMPLETE**. 추가 학습 24/24경로, 본학습 24,576회와 smoke 검사 12회의 optimizer update를 완료했다. C0의 기존 학습·예측은 hash를 확인한 후 재사용했다. 학습 완료와 새 방법을 지지하는 근거는 구분한다.

{d['headline']}

## 무엇을 비교했는가

사용자가 승인한 “잘되는 B0에 추가 요소가 얼마만큼 더해지는가”를 직접 시험했다. 원래 입력을 clipping하거나 교체하지 않았다. 이번 계약은 [PROTOCOL.md](PROTOCOL.md)이며, 구현 선택은 MACHINE_CONTRACT.json에 학습 전에 고정했다.

| 군 | 출발점 | 추가 학습 |
|---|---|---|
| C0 | 기존 selected B0 | 0; 저장 예측 참조 |
| C1 | 같은 source/seed B0 | 기존 q/v LoRA294,912개 추가 학습 |
| C2 | 같은 B0의 전체 가중치 동결 | 일반 residual adapter8,712개 |
| C3 | C2와 동일 초기값/크기 | persistence mask를 적용한 adapter8,712개 |

C2/C3는 512차원 patch embedding에 512→8→512 bottleneck residual을 더한다. 출력층의 weight와 bias를 0으로 초기화해 시작 예측이 B0와 bitwise 동일하다. C3만 residual에 1−mean_patch(p)를 곱한다. p는 과거 8개 관측 중 현재와 같은 부호의 극단값이 차지하는 비율이다. 원래 B0 경로는 유지하며, 지속적으로 큰 변화가 보이는 patch에서는 추가 보정을 줄인다. 8개 관측은 Electricity 8시간, ETTm1 2시간이다. 합성 오류·변화의 실제 생성 상태는 gate에 제공하지 않는다.

일반 residual adapter 및 near-identity 초기화는 알려진 구조다([Houlsby et al.](https://proceedings.mlr.press/v97/houlsby19a.html)). 이번 mask의 신규성을 검증한 것은 아니다. C3/C0 이득만으로 충분하지 않고 C3/C2, C3/C1에서 추가 필요성이 있어야 한다. 동결된 B0 가중치는 보존되지만 어댑터on일 때 예측 성능이 자동 보존되지는 않는다.

## 데이터·선택·정보 권한

기존 Electricity/ETTm1의 TRAIN 256일·V 64일·E 128일과 4채널, context 512/horizon 64, 고정된 Chronos-Bolt-small revision `772f3d25d38aec6d914c8949dab4462e2d46f5d8`을 사용했다. v2의 32 epochs 입력·정답과 V/E 변형 배열도 그대로 재사용했다. 실제 오류나 사건 레이블은 없다. 원래 깨끗한 입력, 생성 상태·mask·true delta·미래 정답을 어댑터 입력으로 제공하지 않았다.

B0의 출발 checkpoint도 기존 V 선택을 유지했다. 선택 seed 81550에서 두 LR(1e−4/3e−4)을 각 군에 동일하게 비교하고, 반복 seed 81551/81552에서는 선택된 LR만 사용했다. 각 군은 대응하는 source·seed의 동일한 B0 checkpoint에서 출발한다. C1을 포함해 모든 추가 학습의 optimizer moments는 0에서 새로 시작했다. 출발 checkpoint와 예측의 hash는 BASELINE_MANIFEST.json에 있다.

세 학습군 모두 32 epochs/1,024 updates, effective batch 32, FP32, TF32 off, dropout 0을 사용했다. AdamW(.9,.999), eps 1e−8, weight decay 0, gradient clip 1이며 스케줄러는 없다. loss는 TRAIN population sigma로 정규화한 2-pinball이다. V 선택은 REFERENCE/POINT8/BURST8/SHIFT4/SHIFT8의 동가중 nMAE이며 checkpoint 0/256/512/768/1024 중 고른다. 동률이면 작은 LR, 이른 checkpoint를 택한다. step 0을 선택하면 B0로 남으며 이를 새 요소의 개선으로 세지 않는다.

추가 학습 3군 × 2 source × (선택 LR 2개 + 반복 seed 2개) = 24경로다. 이전 B0 학습 {baseline['fits']}경로/{baseline['updates']} updates는 모든 군이 공유하는 기존 비용이며 이번에 반복하지 않았다. C2/C3의 8,712개는 추가 학습 파라미터 수다. 배포 시에는 기존 LoRA 294,912개도 남는다. 학습 파라미터 감소가 같은 비율의 계산량·메모리 감소를 뜻하지 않으므로 실측 자원을 별도로 보고한다.

E는 이미 여러 연구에 노출된 개발 기간이며 새로운 독립 test가 아니다. 새 12개 모델의 전체 E 예측을 먼저 저장하고 기존 C0의 4개 예측을 hash로 확인한 후 채점했다. ALL_E_PREDICTIONS_SAVED의 labels_scored=false는 채점 직전 시점의 기록이다. 채점 완료는 verification.json과 scores 파일에 기록돼 있다. 결과를 보고 seed·LR·방법·기간을 추가하지 않았다.

## 원점수와 추가 이득

두 반복 seed 평균 nMAE이며 낮을수록 좋다. FAULT는 POINT/BURST 6조건을 같은 비중으로 평균했다. SHIFT4, SHIFT8, 혼합 조건 SHIFT_POINT를 분리한다.

{table(summary)}

### seed별 원점수

{table(seed)}

### 제안 요소와 모든 직접 대조

양수 gain은 개선, 음수는 악화다. source별 7일 시간 block으로 2,000회 paired bootstrap을 수행했다. 같은 원점의 조건·채널·합성 draw를 독립 날짜로 세지 않는다. 두 optimizer seed를 각 bootstrap draw 안에서 평균하므로 이 구간이 optimizer seed 전체의 불확실성을 충분히 반영하지는 않는다. 구간에 0이 포함돼도 동등성의 증거는 아니다.

{table(effects[effects.panel.isin(['FAULT','REFERENCE','SHIFT4','SHIFT8','SHIFT_POINT'])][['source','new','baseline','panel','gain_pct','ci_low_pct','ci_high_pct','seed_81551_gain_pct','seed_81552_gain_pct']])}

SHIFT 평균, history subset, 원점수 차이와 block 수는 paired_effects.csv에 있다. raw MAE, channel NRMSE, 2-pinball, crossing은 scores_by_condition.csv에 10조건·seed별로 보존했다. 출력에 별도 보정이나 quantile sorting을 적용하지 않았다. 서로 다른 source·목표를 합쳐 우승자를 만들지 않았다.

## 구성요소와 학습량의 해석

{d['interpretation']}

선택된checkpoint:

{table(selected)}

선택된 C3의 V 입력에 p=0(추가 residual을 감쇠하지 않음) 또는 고정 permutation을 적용한 진단이다. optimizer update는 0회다. 양수는 해당 조작 시 오차 증가를 뜻한다. mask 의존성을 확인하는 보조 근거이며, C2보다 나은지는 직접 비교로 판단한다.

{table(ab)}

## 자원 및 검산

두 반복 seed의 평균 추가 자원:

{table(costs)}

추가본학습compute {res['main_optimizer_seconds']:.2f}초, V {res['validation_seconds']:.2f}초, checkpointI/O {res['checkpoint_io_seconds']:.2f}초, 새E추론 {res['E_inference_seconds']:.2f}초. 본실행wall {res['run_all_wall_seconds']/60:.2f}분. C0는예측cache를재사용했으므로이번추론시간을0초모델로해석하지않는다. 기존B0학습비용도shared_baseline_cost.json에별도로남겼다.

최소GPU여유 {res['minimum_free_gpu_mib']}MiB,허용되지않은외부compute표본 {res['unapproved_external_compute_samples']}개. RustDesk만기존승인예외다. 새cache {res['new_cache_bytes']/2**30:.3f}GiB,기존data/model/checkpoint는공유한다.

CPU수학검사4개,6개의출발B0×3학습군초기bitwise parity,12smoke updates의gradient·frozenB0보존·restore·batch permutation을검사했다. main24,576 unique journal,독립scalar {v['independent_scalar_rows']}행×2지표,선택12checkpoint복원,추가adapter8개의off시기존저장B0E예측정확복원,기존C0원점수재검산과LR선택재계산을통과했다. 기존결과·학습표본·checkpoint hash도그대로다. 허용오차는실행전FP32normalizedmax1e−5또는rtol1e−4,CPUrtol1e−10/atol1e−12로고정했다. 초기/off B0복원은bitwise같음을요구했다.

필수 미실행 범위는 없다. 독립된 미사용 기간·source·backbone 검증, 실제 오류 레이블, 가장 가까운 선행 방법과의 정식 비교는 이번 범위에 없으며 실행하지 않았다. 원자료·weights·prediction은 로컬 ignored cache에 보존하고 GitHub에는 manifest·점수·검산을 공개한다. GitHub만으로 cache 전체를 복원할 수 있다는 뜻은 아니다.

## 그림

![원점수](figures/raw_scores.png)

![추가이득과시간block구간](figures/incremental_effects.png)

## 최종 판단

{d['decision']}

새 어댑터가 나빠도 모든 robust PEFT를 반증한 것은 아니다. 좋아도 실제 센서 오류·regime change 해결이나 논문 PASS를 뜻하지 않는다. 다음 후보나 추가 학습은 자동으로 실행하지 않는다.
'''
    (OUT/'REPORT.md').write_text(body)
    (OUT/'FINAL_DECISION.md').write_text('# 최종 결정\n\n실행 COMPLETE: 추가 학습 24경로 / 본학습 24,576 updates + smoke 12 updates. C0 재학습은 0회다.\n\n'+d['headline']+'\n\n'+d['decision']+'\n\n'+d['interpretation']+'\n\n'+d['retain']+'\n\n기존 E는 노출된 개발 기간이며 논문 PASS를 선언하지 않는다. 모든 자료를 보존하며 추가 학습이나 새 후보를 자동으로 시작하지 않는다.\n')
    status(execution='COMPLETE',reports_complete=True,automatic_followup=False)

if __name__=='__main__':report()
