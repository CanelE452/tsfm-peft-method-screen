"""Korean report renders verified scores; scientific interpretation is explicit."""
import pandas as pd
from .common import *
from .prepare import OLD
from experiments.outlier_signal_peft_v1_20260917.report import table

def report():
    assert read(OUT/'verification.json')['status']=='VERIFIED'
    assert read(OUT/'publication_audit.json')['status']=='VERIFIED'
    scores=pd.read_csv(OUT/'scores_by_condition.csv');effects=pd.read_csv(OUT/'paired_effects.csv');res=pd.read_csv(OUT/'resources.csv')
    scores['panel']=scores.condition.map(lambda x:'FAULT' if x.startswith(('POINT','BURST')) else x)
    panel=scores.groupby(['source','arm','panel']).nmae.mean().unstack().reset_index()
    byseed=scores.groupby(['source','arm','seed','panel']).nmae.mean().unstack().reset_index()
    raw=scores.groupby(['source','arm','panel'])[['mae','nrmse','pinball','crossing']].mean().reset_index()
    costs=res[res.seed.isin([81551,81552])].groupby(['source','arm'])[['optimizer_seconds','validation_seconds','checkpoint_io_seconds','train_peak_allocated_mib','E_inference_seconds','E_peak_allocated_mib']].mean().reset_index()
    compare=effects[(effects.new=='B5')&(effects.panel.isin(['FAULT','REFERENCE','SHIFT4','SHIFT8','SHIFT_POINT']))]
    selected=pd.DataFrame(read(OUT/'MODEL_SELECTION.json'))[['source','arm','seed','lr','step','objective']]
    usage=pd.read_csv(OUT/'adapter_usage.csv');exposure=pd.read_csv(OUT/'clip_exposure_summary.csv')
    ex=exposure[(exposure.state=='SHIFT')|((exposure.role=='E_DISCOVERY')&exposure.state.isin(['SHIFT4','SHIFT8']))][['source','role','state','shift_clipped_fraction','shift_nonzero_residual','normalized_mass','longest_run']]
    gate=pd.read_csv(OUT/'gate_behavior.csv');gate=gate[(gate.role=='E_DISCOVERY')&(gate.subset=='clipped_slots')&gate.condition.isin(['POINT8','BURST8','SHIFT4','SHIFT8'])].groupby(['source','arm','condition'])[['mean','median','p10','p90']].mean().reset_index()
    ab=pd.read_csv(OUT/'B5_validation_ablation_effects.csv');ab=ab[ab.condition.isin(['REFERENCE','POINT8','BURST8','SHIFT4','SHIFT8'])].groupby(['source','condition','mode'])[['normal_nmae','ablated_nmae','error_increase_pct']].mean().reset_index()
    old_ab=pd.read_csv(OLD/'A5_validation_diagnostic_effects.csv')
    old_ab=old_ab[old_ab.condition.isin(['POINT8','BURST8','SHIFT4','SHIFT8'])].groupby(['source','condition','mode'])[['normal_nmae','altered_nmae','altered_error_change_pct']].mean().reset_index()
    r=read(OUT/'execution_resource_summary.json');v=read(OUT/'verification.json');d=read(OUT/'scientific_decision.json')
    body=f'''# 입력 오류·지속 변화 후속 비교 결과

실행 **COMPLETE**. 48/48 fits, 본학습 **49,152 updates**, smoke **24 updates**. 기존 실패 진단 및 선택 모델의 V ablation은 optimizer0회다. 실행 완료는 논문 성공과 다르다. 자동 후속 학습은 없다.

핵심 결과: 기존 persistent-shift 학습 노출 부족을 확인했지만, matched 학습 후에도 B5는 B0보다 SHIFT8 오차가 Electricity60.18%·ETTm1 27.24% 높았다. 고정 B3의 ETTm1 SHIFT_POINT +2.34%와 B5/B4의 작은 이득은 남기되, 현재 B5 후보는 종료한다.

## 범위와 재사용

유일한 실행 계약은 [PROTOCOL.md](PROTOCOL.md)다. 기준 commit e2b1ea4의 v1 결과·코드·원점·checkpoint를 감사하고 기존 파일을 보존했다. 새 구현은 사용자의 Python 직접 작성 승인 아래 작성했다. 기존 Electricity/ETTm1 TRAIN256·V64·E128 distinct days와 네 채널, pinned Chronos-Bolt-small FP32를 그대로 재사용했다. 모델 revision은 `772f3d25d38aec6d914c8949dab4462e2d46f5d8`이며 정확한 파일 hash는 download_receipts.json에 있다.

E는 이미 사용한 개발 기간이다. 새로운 독립 시험이 아니다. 원자료를 완전한 clean ground truth라 부르지 않는다. 실제 품질/사건 레이블이 없으며 synthetic measurement fault는 입력만, synthetic persistent shift는 최근 입력과 미래를 같은 delta로 바꾼다. 모델은 observed context와 TRAIN population sigma만 받는다. generator state·fault mask·clean x0·true delta·future y는 gate 입력에 없다.

## 기존 실패 기전: 업데이트 0회 진단

{table(ex)}

shift_clipped_fraction은 변화가 주입된 위치 중 clip된 비율이다. 전체 context discarded mass와 구분해야 한다. Electricity에서는 기존 TRAIN의 다른 극단값 때문에 전체 normalized mass가 E SHIFT8보다 클 수도 있다. 그러므로 전체 mass 하나로 persistent-shift 노출을 판단하지 않았다. TRAIN state별 severity/duration/fault-count 세부는 train_clip_exposure.csv, V/E 세부는 eval_clip_exposure.csv, median/p90/p99/max는 clip_exposure_distribution.csv에 있다.

{d['exposure_interpretation']}

기존 A5의 실제 V adapter 사용량:

{table(usage[usage.state.isin(['SHIFT4','SHIFT8','POINT8','BURST8'])])}

사용량은 native patch embedding에 추가된 delta의 RMS 비율이다. patch별 집계와 상관계수는 adapter_usage_by_patch.csv, adapter_usage.csv에 있다. 기존 residual-zero/permutation V 결과와 함께 경로 사용의 근거로 해석하며, 우월성 또는 독립적인 인과 증명으로 해석하지 않는다. near-zero 문턱을 사후 생성해 ADAPTER_UNDERUTILIZED라고 자동 판정하지 않았다.

기존 v1 selected A5의 V residual-zero/permutation 결과를 다시 참조한다(기존 optimizer seeds81501/81502 평균, 새 학습 아님). 양수는 feature를 제거/섞을 때 오차 증가다.

{table(old_ab)}

## 고정된 직접 비교

| 군 | 입력 처리 | LoRA 외 학습 파라미터 |
|---|---|---:|
| B0 RAW_AUG | raw + matched augmentation | 0 |
| B1 HARD_CLIP | observed median ±6r | 0 |
| B2 CLIP_RESIDUAL | 기존 A5 embedding residual 동일 구조 | 4,872 |
| B3 PERSIST_FIXED | trailing8 same-sign extreme 비율로 복원 | 0 |
| B4 PERSIST_MAG | magnitude-only sigmoid gate | 3 |
| B5 PERSIST_LEARNED | magnitude·persistence·상호작용 sigmoid gate | 4 |

공통 LoRA는 q/v rank8 alpha16, 294,912개다. B4/B5의 1 scalar 차이를 숨기거나 dummy parameter로 맞추지 않았다. B5>B4만으로 순수 persistence 효과라고 부르지 않는다. 또한 B4에서 실제 복원량이 0이 아닌 clipped 위치는 항상 I=1이므로 a0+a2가 하나의 intercept처럼 작용한다. 이 수식의 중복성과 최적화 차이도 직접 대비의 해석 한계이며 수식을 사후 수정하지 않았다. B5 gate 초기값은 문서에 없으므로 학습 전 B4와 같은 intercept−4, 나머지0으로 봉인했다. gate는 동일 LR로 end-to-end forecasting loss만 사용했다. 8은 observation slots이며 Electricity8시간/ETTm1 2시간이다.

각 예제는32epochs 동안 REFERENCE/POINT/BURST/SHIFT 각8회. fault amplitudes4/8/16과 shift4/8이 겹친다. 모든 군은 같은 x/y cache를 읽는다. V/E 변형 배열도 v1과 hash가 동일하다. AdamW(β=.9/.999, eps1e−8, wd0), clipnorm1, scheduler 없음, effective batch32, FP32/TF32off/dropout0를 고정했다. CPU 허용오차1e−10/1e−12와 FP32 normalized max1e−5 또는 rtol1e−4도 결과 전에 고정했다.

선택seed81550에서 LR1e−4/3e−4를 비교하고, 선택 LR로81551/81552를 반복했다. 모든 경로는1024updates, checkpoint0/256/512/768/1024다. V objective는 REFERENCE/POINT8/BURST8/SHIFT4/SHIFT8 동가중 nMAE다. 이 가중치를 실제 배포 빈도라 주장하지 않는다. 작은 LR·이른 checkpoint tie rule을 썼다. 선택 전부를 봉인한 후 선택24개 모델의 **전체 E prediction을 먼저 저장하고** 정답을 채점했다.

### 원점수

반복 두 seed 평균 nMAE, 낮을수록 좋다. FAULT는 POINT/BURST6조건 동가중이다. SHIFT4/SHIFT8을 분리했다.

{table(panel)}

### seed별 원점수

{table(byseed)}

raw MAE·채널 NRMSE·TRAIN-scale2-pinball·quantile crossing 원점수:

{table(raw)}

원점별 세부는 scores_by_origin.csv, 10조건×seed별 세부는 scores_by_condition.csv다. 채널·모사draw·조건을 독립 날짜로 세지 않는다. 음수 synthetic 값도 제외하지 않았다.

### B5의 핵심 직접 대비

gain_pct 양수=개선, 음수=악화. 7일 절대 시간 block(Electricity168slots, ETTm1 672slots), paired bootstrap2,000회. 두 optimizer seed는 draw 안에서 평균했다. 이 구간은 optimizer seed 모집단 전체의 불확실성을 포괄하지 않는다. CI가0을 포함한다고 equivalence라고 쓰지 않는다.

{table(compare[['source','baseline','panel','gain_pct','ci_low_pct','ci_high_pct','seed_81551_gain_pct','seed_81552_gain_pct','origins','time_blocks']])}

필수 B3/B0, B2/B0, B1/B0 및 SHIFT평균/HISTORY_SUBSET 참고 대비도 paired_effects.csv에 전부 있다. 서로 다른 원천의 p-value나 목표 점수를 합쳐 우승자를 만들지 않았다.

## 구성요소별 판단

{d['component_interpretation']}

기존 A1/A5는 역사적 참고다. TRAIN 분포뿐 아니라 V objective와 seed도 바뀌었으므로 v1→v2 차이를 노출 변경만의 인과 효과라고 부르지 않는다. 새 실험 내 B2/B0는 같은 노출·선택 규칙에서의 직접 대비다.

## Gate 사용과 optimizer0 ablation

아래는 clipped slots에서 선택 B3/B4/B5 gate 분포의 seed 평균 요약이다. 원래 clip되지 않는 점은 g가 달라도 x_eff가 같으므로 별도 집계했다. 표의 median/p10/p90은 seed별 분위수의 평균이며 합친 분포의 분위수와 구분한다. 전체 슬롯과 seed별 quantiles/histogram은 gate_behavior.csv에 있다. 초기·선택·1024update 최종 scalar는 gate_parameters.json에 있다.

{table(gate)}

선택 B5의 V에서 p=0 또는 고정 permutation, 추가 학습0회:

{table(ab)}

이는 의존성 진단이다. B4/B3보다 필요한 구성요소라는 결론은 직접 비교를 통해서만 판단한다. 초기 gate가 hard clip에 가까운 점과 현재 LR/update예산의 최적화 제약도 해석에 포함한다. 결과를 보고 초기값·LR·rank·W·threshold를 바꾸지 않았다.

## 자원과 검산

반복seed 평균 자원:

{table(costs)}

전체48fits optimizer compute {r['main_optimizer_seconds']:.2f}초, V {r['validation_seconds']:.2f}초, checkpoint I/O {r['checkpoint_io_seconds']:.2f}초, E 추론 {r['E_inference_seconds']:.2f}초. 학습 시간은 forward/backward/clip/optimizer 구간이며 V와I/O를 분리했다. 자원 이득은 자원 표로만 판단하며 예측 이득과 합산하지 않았다.

모니터링 최소 GPU 여유 {r['min_free_gpu_mib']}MiB, 허용되지 않은 외부 compute 표본 {r['unapproved_external_compute_samples']}개. RustDesk만 기존 승인 예외로 기록했다. 신규 cache {r['cache_bytes_excluding_symlink_files']/2**30:.3f}GiB(공유 모델/데이터는 기존 cache). one GPU/one worker였다.

12 source/arm의2-update 학습, gate gradient, frozen backbone/head 보존, exact checkpoint restore, batch 순서 불변성, native raw parity, 기존 clip/A5 구조 일치를 확인했다. 총49,152 main journal은 중복 없이각 fit1..1024다. 독립 scalar 검산 {v['independent_scalar_rows']}행×2지표, 선택24checkpoint의 E복원, LR/checkpoint 선택 재계산, 원자료/모델/기존결과/입력 hash를 검산했다. [verification.json](verification.json), [publication_audit.json](publication_audit.json).

초기 준비 단계에서 audit metadata의 NumPy scalar JSON 직렬화 오류1건을 수정했다(학습0회, 설정 변경 없음). 이후 중단/재학습 여부는 optimizer journal과 실행 오류 기록을 따른다. 본학습 중 추가 독립 generator replay로 두 원천 각각32,768개 TRAIN 예제의 정답·비변형 위치·shift 위치·fault magnitude도 재검산했다(independent_generator_replay.json). 필수 미실행 범위: 없음. 원자료·모델·checkpoint·prediction은 로컬 ignored cache에 있어 GitHub만으로 수치 replay가 완결되지는 않는다. resume는 journal과 epoch state가 일치할 때만 허용하며 불명확한 update를 자동 replay하지 않는다.

## 그림

![기존 모든 state의 TRAIN/E clip 노출](figures/all_state_clip_exposure.png)

![기존 TRAIN/E shift 노출](figures/training_evaluation_clip_exposure.png)

![FAULT·REFERENCE·SHIFT4·SHIFT8](figures/fault_reference_shift_tradeoff.png)

![복원 gate 분포](figures/restore_fraction_distribution.png)

![사전 고정 origin 예시](figures/predeclared_transform_examples.png)

예시는 첫 sealed V origin·channel0·generator0로 실행 전에 지정했고 성능 좋은/나쁜 사례를 골라내지 않았다.

## 최종 과학적 판단과 신규성

{d['final_interpretation']}

이 결과는 고정한 두 원천의 synthetic development evidence다. 실제 센서 고장이나 실제 regime change 해결을 보인 것이 아니고, 나쁜 결과도 robust PEFT 전체를 반증하지 않는다. sigmoid gate/robust clipping/persistence 수식 자체의 신규성을 확정하지 않았다. 정식 TATO 또는 가까운 강건 선행 전체 비교, 실제 품질/이벤트 레이블, 독립 source/backbone 검증은 수행하지 않은 범위다. 후속이 필요하더라도 여기서는 보고서에만 남기며 자동 실행하지 않는다.
'''
    (OUT/'REPORT.md').write_text(body)
    (OUT/'FINAL_DECISION.md').write_text('# 최종 결정\n\n실행 COMPLETE: 48 fits / 49,152 main +24 smoke updates. 논문 PASS가 아니다.\n\n'+d['final_interpretation']+'\n\n'+d['component_interpretation']+'\n\n남길 구현: 정보 권한을 분리한 입력 변형, 같은 x/y를 사용하는 직접 대조, clip 노출 진단, 고정 예산 LoRA runner, journal/resume·봉인·전체 예측 선저장·독립 검산.\n\n'+d['retained_candidates']+'\n\n실제 오류/사건 레이블, 가까운 강건 선행 전체 비교, 독립 source/backbone 검증이 남아 있다. 새 seed/LR/rank/threshold/window/v3 또는 후속 학습은 시작하지 않는다.\n')
    status(execution='COMPLETE',reports_complete=True,automatic_followup=False)

if __name__=='__main__':report()
