"""Korean report and paper evidence, rendered only after independent verification."""
import pandas as pd
from .common import *
from experiments.outlier_signal_peft_v1_20260917.report import table
from .prepare import SHAPES

def report():
    assert read(OUT/'VERIFICATION.json')['status']=='VERIFIED';assert read(OUT/'PUBLICATION_AUDIT.json')['status']=='VERIFIED'
    d=read(OUT/'SCIENTIFIC_DECISION.json');summary=pd.read_csv(OUT/'SUMMARY_TABLE.csv');effects=pd.read_csv(OUT/'UNCERTAINTY.csv');seed=pd.read_csv(OUT/'SEED_EFFECTS.csv');scores=pd.read_csv(OUT/'PANEL_SCORES.csv');cost=read(OUT/'COST_ACCOUNTING.json');v=read(OUT/'VERIFICATION.json');cal=read(OUT/'CALIBRATION_SELECTION.json')
    main=summary[(summary.seed_scope=='all3') & summary.arm.isin(['C0','C1','C2','C3'])];original=summary[(summary.seed_scope=='original2') & summary.arm.isin(['C0','C1','C2','C3'])]
    mech=effects[(effects.kind=='standard') & (effects.condition=='SHIFT8') & (effects.new=='C3') & effects.baseline.isin(CONTROLS+['C2_SHRINK','C0_BIAS']) & (effects.seed_scope=='all3')]
    transfer=effects[(effects.panel=='electricity_transfer')&(effects.kind=='standard')&(effects.condition=='SHIFT8')&(effects.new=='C3')&effects.baseline.isin(['C0','C2'])]
    trade=effects[(effects.kind=='standard') & (effects.seed_scope=='all3') & (effects.ci_type=='time') & (effects.new=='C3') & effects.baseline.isin(['C0','C2']) & effects.condition.isin(['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT'])]
    shape=effects[(effects.kind=='shape') & (effects.seed_scope=='all3') & (effects.ci_type=='time') & (effects.new=='C3') & (effects.baseline=='C2')].pivot(index='condition',columns='panel',values='gain_pct').reset_index()
    legacy=effects[(effects.kind=='legacy_pulse') & (effects.seed_scope=='all3') & (effects.ci_type=='time') & (effects.new=='C3') & effects.baseline.isin(['C0','C2'])]
    seedtable=seed[(seed.kind=='standard')&(seed.condition=='SHIFT8')&(seed.new=='C3')&seed.baseline.isin(['C0','C2'])]
    rawseed=scores[(scores.kind=='standard')&(scores.stage=='selected')&scores.arm.isin(['C0','C1','C2','C3'])].pivot(index=['panel','arm','seed'],columns='metric_panel',values='nmae').reset_index()
    cols=['panel','seed_scope','new','baseline','gain_pct','ci_type','ci_low_pct','ci_high_pct']
    calibration=pd.DataFrame([dict(source_seed=k,alpha=r['alpha'],beta=r['beta']) for k,r in cal.items()])
    choices=pd.DataFrame(read(OUT/'MODEL_SELECTION.json'))[['source','arm','seed','lr','step']]
    storage=pd.read_csv(OUT/'STORAGE_SUMMARY.csv');storage['GiB']=storage.logical_bytes/2**30
    resources=pd.read_csv(OUT/'RESOURCE_REPORT.csv');train=resources[resources.kind=='training'].groupby(['source','arm'])[['trainable_parameters','retained_adaptation_parameters','optimizer_seconds','validation_seconds','train_peak_allocated_mib','train_peak_reserved_mib']].median().reset_index()
    inference=resources[resources.kind=='inference_profile'].groupby(['panel','arm'])[['median_seconds','range_seconds','inference_peak_allocated_mib','forward_models']].median().reset_index()
    body=f'''# 지속성 마스크 추가 적응 — 논문 후속 검증

실행 COMPLETE. 새 본학습 **58/58경로, 59,392 updates**, 폐기용 smoke **36 updates**, 전체 **59,428 updates**. 기존 B0/C1/C2/C3를 중복 학습하지 않았다. 모든 평가 예측 저장 후 채점하고 독립 검산까지 완료했다. 실행 성공과 방법의 과학적 근거를 구분한다.

{d['summary']}

## 계약·자료·가중치

유일한 계약은 [MASTER_CLI.txt](MASTER_CLI.txt)다. 기존 C3의 구조·mask·rank·threshold·window를 바꾸지 않았다. 잘 학습된 B0와 원래 관측 입력을 유지하고, 오프라인 추가 적응 후 평가 중에는 정답으로 업데이트하지 않는다. 시작 시 delta=0이며 어댑터 off로 B0를 복원할 수 있지만, 학습 후 성능 보존을 수학적으로 보장하지는 않는다.

C0는 이미 적응된 B0, C1은 같은 B0의 LoRA 추가 학습, C2는 B0 동결+일반 어댑터, C3는 같은 어댑터에 고정 관측 지속성 mask를 적용한 것이다. 새 MEAN/ROTATE16/RECENCY는 보정량·위치·최근성 설명의 대조이며 새 후보가 아니다. 어댑터8,712개 외에 기존 B0 LoRA294,912개가 남아 배포 적응 파라미터는303,624개다.

원래 Electricity/ETTm1의 TRAIN256/V64/E128일과4채널을 그대로 재사용했다. 추가 Electricity16계열은 원래 B0의 미학습 계열이지만 미사용 이력이 검증된 계열은0개다. 명세에 따라 hash순 fallback 탐색 패널로 선택했고 과거 성능 노출 확인4개와 노출 불명12개를 모두 유지했다. 같은 E 날짜를 사용하고 대상 TRAIN sigma에도 접근하므로 새로운 독립 자료나 target-data-free 전이가 아니다. 물리적 meter group은 없으며 계열 ID 분리만 확인했다. 사전학습 원천 중복은 UNKNOWN_PRETRAINING_OVERLAP이다.

ETTm2는 공식 commit/hash와 기존 로컬 파일이 같고 실제15분 간격·중복 없음·69,680행을 확인했다. 다른 변압기 지점이지만 같은 ETT 계열이며 이전 연구에서 이미 성능을 본 자료다. 별도 B0 학습 후 같은 적응 절차를 적용한 재현이고 ETTm1 가중치의 zero-training transfer가 아니다. TRAIN/V/E의60/20/20 분할은 통상 ETT benchmark split이라고 부르지 않는다. target은 HUFL/HULL/MUFL/MULL의 pooled univariate 예측이며 OT forecasting이 아니다.

계열 ID·원점 분산·출처·과거 노출은 [SERIES_PANEL_MANIFEST.json](SERIES_PANEL_MANIFEST.json), [DATA_EXPOSURE_MANIFEST.csv](DATA_EXPOSURE_MANIFEST.csv), [ORIGIN_AUDIT.csv](ORIGIN_AUDIT.csv), [ETTM2_DATA_RECEIPT.json](ETTM2_DATA_RECEIPT.json)에 있다. 미래 성능으로 계열·날짜를 골라 바꾸지 않았다. 실제 오류·사건 레이블은 없다.

## 실행과 선택

원래 두 source의 세 번째 seed81553 B0/core8경로 → 세 기전 대조30경로 → ETTm2 B0/core20경로 순서로 완료했다. 그중 새 B0는7경로다. main58경로 모두1,024updates를 실행했으며 불리한 중간 성능 때문에 생략하지 않았다. 선택 seed는 기존81550, ETTm2 85550이고 반복 ETTm2는85551/85552/85553이다.

모든 군이 같은 matched TRAIN 입력·정답·상태 빈도·순서와 초기 어댑터를 공유했다. FP32/TF32off/dropout0, effectivebatch32, AdamW(.9,.999), eps1e−8, wd0, clipnorm1, scheduler없음이다. 실제 microbatch는 세 source 모두32였다. LR1e−4/3e−4와 checkpoint0/256/512/768/1024, 다섯 V 조건 동가중 nMAE 선택을 유지했다. 원래 source의 core 세 번째 seed는 기존 LR 선택을 상속했다. C1 optimizer moments는 새로 시작했다.

step0은 B0 유지이며 새 요소의 개선으로 세지 않는다. fixed1024 결과는 별도 [FIXED1024_COMPARISON.csv](FIXED1024_COMPARISON.csv)에 보존했고 E에서 더 나은 쪽으로 주 결과를 바꾸지 않았다. full prediction manifest는246개 view로, 기존 가중치/예측과 동일 checkpoint의 alias, 출력 대조, 동일 과거 PULSE 재사용을 포함한다. 이를246개 새 학습이나246개 독립 모델로 세지 않는다.

선택 checkpoint:

{table(choices)}

V에서 선택한 출력 대조:

{table(calibration)}

C2_SHRINK는 원래 source V에서 alpha∈{{0,.25,.5,.75,1}}를 선택하고 동률이면 작은 alpha를 썼다. alpha0/1은 C0/C2와 동일하다. C0_BIAS는 같은 V 조건의 normalized median residual 한 개다. target transfer V/E로 다시 선택하지 않았다. 두 대조는 COSA 또는 TATO 전체 재현이 아니다.

## T1. 기존 관찰·계열 전이·세 번째 seed

아래 nMAE는 낮을수록 좋다. FAULT는 POINT/BURST6조건 동가중이다. source와 목표를 섞은 종합 우승 점수는 없다.

세 seed 평균 core 원점수:

{table(main)}

기존 두 seed만의 별도 원점수:

{table(original)}

핵심 전이 SHIFT8 비교. 양수 gain은 개선이며 분모는 해당 기준 방법의 평균 오차다. 날짜 구간은 seed 평균 후7일 block을 재표집한 조건부 구간이다. 계열+날짜 구간도 함께 보고하며 채널×시간을 독립 표본으로 flatten하지 않았다.

{table(transfer[cols])}

{d['series']}

seed별 SHIFT8 효과:

{table(seedtable[['panel','seed','new','baseline','new_nmae','baseline_nmae','gain_pct']])}

core의 모든 주요 조건·seed 원점수:

{table(rawseed)}

전이의 노출 확인/불명 그룹은 [EXPOSURE_SCORES.csv](EXPOSURE_SCORES.csv)에 분리했다. 이 혼합 평균을 미사용 계열의 확증으로 해석하지 않는다.

## T2. 보정량·위치·최근성 설명

MEAN은 example별 gate 평균을 모든 patch에 적용한다. ROTATE16은32개 gate를16patch 이동하고 RECENCY는 gate를 내림차순 정렬해 최근 patch를 더 보호한다. ROTATE/RECENCY는 C3와 gate multiset·합·제곱합을 보존하지만 실제 residual norm, delta와의 정렬, gradient와 학습 경로까지 같게 만들지는 않는다.

아래는 각 대조를 같은 예산으로 학습하고 V로 선택한 pipeline의 비교다. nominal95%와 기전3대비의 Bonferroni 동시구간을 구분한다. SHRINK/BIAS는 그 세 대조 보정 집합에 포함하지 않는다.

{table(mech[cols+['bonferroni3_low_pct','bonferroni3_high_pct']])}

{d['location']}

gate와 실제 추가 residual RMS는 GATE_RESIDUAL_STATS.json과 그림4에 있다. 관측 mask를 사후 교체한 것만으로 학습 대조를 대신하지 않았다. CI에0이 포함된다는 이유만으로 동등성을 확정하지 않는다.

## T3. ETTm1·ETTm2와 원자료/오류 조건의 손해

{d['site']}

모든 source의 주요 조건, C3/C0와 C3/C2 총·추가 효과:

{table(trade[['panel','condition','new','baseline','new_nmae','baseline_nmae','absolute_error_difference','gain_pct','ci_low_pct','ci_high_pct']])}

REFERENCE는 수정하지 않은 원자료이며 반드시 깨끗한 자료라는 뜻은 아니다. 손해를 감수할 실사용 허용폭은 아직 정해지지 않았고 임의1%를 합격선으로 삼지 않았다. 오류와 지속 변화의 현재 합성 비중은 실제 사건 빈도가 아니다.

## T4. 변화 크기·길이·경계·ramp/pulse

기존 E128 중 index순 등간격64원점, 양·음 부호 동가중으로 지정된8형태를 검사했다. 학습이나 선택에는 쓰지 않았다. 아래는 C3의 C2 대비 nMAE gain이다. C0 대비 효과와 구간까지 UNCERTAINTY.csv에 보존했다.

{table(shape)}

{d['shape']}

PULSE의 과거는 지속되는 SHIFT8과 같고 미래는 다르다. 양 부호 균형 패널의 PAIRED_SHIFT8_D32는 그 짝이며 새로운 학습 형태가 아니다. 다음 표는 별도로 기존 SHIFT8의 원점·채널·draw·부호·입력·예측을 정확히 재사용하고 미래 offset만0으로 둔 검사다. 동일 관측 과거로 서로 다른 미래를 완벽히 구별할 수 있다고 요구하지 않는다.

{table(legacy[['panel','new','baseline','gain_pct','ci_low_pct','ci_high_pct']])}

기존 source의 history-triggered subset은 기존 입력 기반 정의를 유지한 보조표이며 실제 사건 레이블이 아니다. seed별 점수와 CI 모두 원점 동가중을 사용한다.

평가 전 감사에서 조건별 재표집 seed가 공통 날짜 draw 지시와 다른 것을 발견했다. 정확한 epoch 경계에서 잠시 멈추고 통계 코드와 해설 명세만 바로잡았다. 원래 봉인·오류 상태·코드는 pre_E_statistics_repair에, 변경 사유와 진행 횟수는 SEAL_AMENDMENT_01.json에 보존했다. 모델·학습·선택·데이터·주 지표는 바뀌지 않았고 당시 새 E 예측·채점도 시작하지 않았다. 완료된38 fits는 재사용하고 중간 경로의576updates 이후부터 정확히 이어갔다.

## T5. 파라미터와 실제 자원

{d['cost']}

전체 새 본학습 compute {cost['new_training_compute_seconds']:.2f}초, 학습 중 checkpoint V 검증 {cost['new_validation_seconds']:.2f}초, checkpoint I/O {cost['new_checkpoint_io_seconds']:.2f}초. 두 실행 구간을 합한 run-all wall {cost['new_run_all_wall_seconds']/60:.2f}분이다. 이 V 시간은 checkpoint 선택 검증이며, 이후 alpha/beta 선택용 V 추론은 별도 active-time을 계측하지 않았고 전체 wall에 포함된다. 새 학습 횟수58에는 새 B0 7개를 포함한다. 과거 공유 B0 {cost['old_shared_B0_fits']}경로/{cost['old_shared_B0_updates']}updates와 이전 additive 비교 {cost['old_additive_comparison_fits']}경로/{cost['old_additive_updates']}updates는 새 비용과 분리했다.

기존 공유 자료·모델·checkpoint와 새 cache의 논리적 파일 크기를 따로 집계했다. 파일시스템의 압축·block 할당량이나 모델 사전학습 비용을 측정한 수치가 아니다. 공유 자료는 이번 감사에 등록한 파일 범위이며 과거 모든 실험 cache의 총량이 아니다.

{table(storage)}

source·방법별 학습 자원 중앙값(선택과 반복 fit 모두 포함):

{table(train)}

동일 panel에서128개 입력 series, 같은 microbatch32/FP32로 fresh profile을 수행했다. 각 seed의3회 raw timing과 범위는 RESOURCE_REPORT.csv에 있다. 아래는 seed별 측정 중앙값들의 중앙값이다. C2_SHRINK는 C0와 C2의 두 forward 측정 비용을 합산했으며 단일 forward가 아니다. blend의 CPU overhead는 합산값에 포함하지 않아 명시적으로 하한 비용이다. 이전 예측 cache를 재사용한 C0를0초 모델로 보고하지 않는다.

{table(inference)}

최소 GPU 여유 {cost['minimum_free_gpu_mib']}MiB, 허용되지 않은 외부 compute 표본 {cost['unapproved_compute_samples']}개. RustDesk만 승인된 예외다. PyTorch allocated/reserved peak는 전체 드라이버/원격 화면 메모리와 다르다. 중간에 재개한 ETTm2 B0 선택1경로의 fit receipt peak는 마지막 실행 구간의 값이며, 전체 구간의 GPU 여유는 통합 monitor log로 보존했다. frozen backbone도 patch 어댑터의 gradient 전파에 activation이 필요하므로 파라미터 감소율을 GPU 메모리 감소율로 바꾸지 않는다. 작은 timing 차이를 보편적 속도 우위로 주장하지 않는다.

## 검산·보존·미실행

CPU 참조20개는 로컬 작성한 합성 수식 검사다. 실제 GPU 검사와 구분한다. 모델 검사에서 원래 학습된 C3 출력 일치, 초기/off B0 일치, 같은 adapter 초기값, g=1 대조 동일성, 실제 parameter 변화, 동결 가중치·buffer 보존, 유한 gradient, checkpoint 복원을 확인했다. ETTm2의 사전 smoke에는 아직 학습되지 않은 wiring proxy를 썼고 모든 실제 추가 main fit 직전에 해당 seed의 학습된 B0 checkpoint/hash 및 초기 예측 동일성을 다시 확인했다.

main59,392 unique journal과 smoke36개를 검산했다. 선택54모델을 복원했고 원점수 전체를 벡터 재집계했다. 독립 scalar replay는 **{v['independent_scalar_rows']}행×2지표**로, 모든 저장 예측 view/조건의 첫·마지막 원점과 첫·마지막 채널, 두 draw를 확인했다. 전체 원점에 대한 scalar replay를 했다고 쓰지 않는다. 기존 점수/hash 보존과 V LR/alpha/beta 선택 재계산도 통과했다. CPU rtol1e−10/atol1e−12, GPU normalizedmax1e−5 또는 rtol1e−4 기준은 바꾸지 않았다.

필수 실행의 미완료 범위는 없다. 그러나 독립 계열 미사용 확인은 UNRESOLVED다. 실제 사건 레이블·완전히 미사용 source/기간·공식 COSA/TATO 전체 비교·다른 backbone 검증은 본 예산에 없으며 실행하지 않았다. 원자료·weights·예측은 local ignored cache에 보존했고 GitHub에는 manifest·전체 점수·검산·코드가 있다. GitHub만으로 cache 없이 전체 수치 재현이 가능하다는 뜻은 아니다.

## 신규성과 논문 주장

{d['novelty']}

[선행 경계](LITERATURE_BOUNDARY.md), [주장–근거표](PAPER_CLAIM_EVIDENCE.md), [논문 개요](PAPER_OUTLINE.md)를 함께 본다. 일반 adapter와 identity 시작은 알려진 원리다. C3는 관측 지속성으로 추가 수정 위치를 제한한다는 좁은 가설이며, 새 이름을 붙인 것만으로 신규성을 확정하지 않는다.

## 그림

![데이터 흐름](figures/01_dataflow.png)
![전이 seed별 효과](figures/02_transfer_seeds.png)
![주요 손익](figures/03_tradeoffs.png)
![gate와 실제 residual](figures/04_gate_and_residual.png)
![변화 형태](figures/05_shape_stress.png)

## 최종 결정

{d['decision']}

새 구조·새 LR·추가 seed·다른 데이터·후속 학습은 자동으로 시작하지 않는다. 불리한 조건과 이미 확인된 좁은 양성 관찰을 함께 보존한다.
'''
    (OUT/'REPORT.md').write_text(body)
    (OUT/'FINAL_DECISION.md').write_text('# 최종 결정\n\n실행 COMPLETE:58새fits /59,392main +36smoke =59,428updates.\n\n'+d['summary']+'\n\n'+d['decision']+'\n\n'+d['series']+'\n\n'+d['location']+'\n\n'+d['site']+'\n\n'+d['novelty']+'\n\n독립 자료 PASS나 논문 PASS를 선언하지 않는다. 추가 자동 학습은 없다.\n')
    claim=pd.DataFrame(d['claim_rows']);(OUT/'PAPER_CLAIM_EVIDENCE.md').write_text('# 논문 주장–근거 연결\n\n'+table(claim)+'\n\n실행 완료는 논문 성공과 다르다. 원자료 노출 및 범위 한계를 삭제하지 않는다. 전체 표와 음수 결과는 REPORT.md/RAW_SCORES.csv에 보존했다.\n')
    outline='''# 논문 개요 — 근거에 맞춘 초안 구조

작업명: Persistence-masked residual adaptation. 신규성 확정 명칭이 아니다.

1. 문제: 잘 적응된 raw-input B0 위에서 작은 추가 적응이 언제 유익한가. 원래 오류 강건성 탐색에서 Electricity SHIFT8의 양성 결과를 보고 좁힌 후속 가설이라는 발견 경로를 공개한다.
2. 방법: B0 동결, zero-init bottleneck residual, 관측 과거만으로 계산한 fixed persistence mask. 일반 adapter·identity 초기화의 선행을 인정한다. 초기/off 복원과 학습 후 성능 보존을 구분한다.
3. 반박 가능한 설명: 일반 추가 용량(C2), 추가 계산(C1), 보정량(MEAN/SHRINK), 위치(ROTATE16), 최근성(RECENCY), 간단한 bias(C0_BIAS). 동일 mask 분포가 순수 인과 분리를 보장하지 않는다는 한계.
4. 절차: 사전 봉인·58 fits cap·V-only 선택·전체 예측 저장 후 E 채점·과거 자료 노출·target TRAIN sigma 접근·미학습 계열과 독립 자료의 차이.
5. 결과: T1 원래 관찰과 전이/두 seed·세 seed, T2 기전과 단순 대안, T3 ETTm1/ETTm2 및 오류 손해, T4 형태와 pulse 식별성 반례, T5 공유 B0+추가 학습+두 forward 비용. 유리한 조건만 남기지 않는다.
6. 한계: 실제 사건 레이블 부재, 반복 개발 선택, 사전학습 중복 불명, 공유 날짜/날씨, optimizer seed3의 조건부 CI, 공식 선행/다른 backbone 비교 미실행.
7. 결론: 아래 검증된 범위를 넘어 일반 오류 강건성·인과성·독립 재현·논문 PASS를 주장하지 않는다.

'''+d['summary']+'\n\n'+d['decision']+'\n\n표·그림의 원점수와 계산식은 REPORT.md 및 CSV를 참조한다. 새로운 연구 실행은 이 개요의 작성으로 승인되지 않는다.\n'
    (OUT/'PAPER_OUTLINE.md').write_text(outline)
    status(execution='COMPLETE',reports_complete=True,automatic_followup=False)

if __name__=='__main__':report()
