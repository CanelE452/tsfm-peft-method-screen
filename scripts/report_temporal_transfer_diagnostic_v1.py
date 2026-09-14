"""Render the one-shot temporal audit; consumes completed outputs, never trains."""
import csv,json,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/temporal_transfer_diagnostic_v1'
RESEARCH=ROOT/'research/temporal_transfer_diagnostic_v1'
REF=ROOT/'.cache/temporal_transfer_references'

def read(p):return json.loads(p.read_text())
def save(name,text):
    p=OUT/name;assert not p.exists(),p;p.write_text(text+'\n')
def csvsave(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

s=read(OUT/'analysis_summary.json');agg=read(OUT/'aggregate_replay.json');receipt=read(OUT/'execution_receipt.json')
rows=s['comparisons'];shr=s['shrinkage'];diag=s['selection_cells']
assert len(rows)==120 and len(shr)==24 and len(diag)==24
scope=read(OUT/'verification_scope.json')
source_names=['beijing','ettm2_later','electricity_new']
lines=['# 같은 후보의 시간순 선택 진단','',
'24 source×budget×seed×arm cell, 각각 LR 2개×step 4개에서 step0 중복만 합친 7개 후보. 총 192개 원본 예측·체크포인트 identity를 기록했다. 모든 cell의 서로 다른 후보가 확인됐고 누락 캐시는 없다.','',
'S는 기존 V 앞 8 origins, D_diag는 뒤 8 origins다. 학습 타깃은 S보다 먼저 끝나며 S/D horizon 타깃은 겹치지 않는다. D_diag의 문맥에 S 관측이 들어가는 것은 예측 당시 과거 정보다. D_diag는 사후 재사용 V이며 새 holdout이 아니다.','',
'표는 각 source/budget/arm의 두 seed에 대한 %F0 평균이다. 원손실·paired 차이·각 seed·앞4/뒤4·유효 타깃 수는 temporal_selection_comparisons.csv에 있다. 서로 다른 원천의 raw loss를 섞어 순위를 만들지 않았다.','',
'| Source | Windows | Arm | FIXED %F0 | RECENT4 %F0 | SPREAD4 %F0 | ALL8 %F0 |',
'|---|---:|---|---:|---:|---:|---:|']
for src in source_names:
    for b in (32,233):
        for arm in ('native','native_anchor'):
            pool=[r for r in rows if (r['source'],r['budget'],r['arm'])==(src,b,arm)]
            vals=[sum(r['D_gain_vs_f0_percent'] for r in pool if r['rule']==rule)/2 for rule in ('FIXED','RECENT4','SPREAD4','ALL8')]
            lines.append(f'| {src} | {b} | {arm} | '+' | '.join(f'{v:+.4f}' for v in vals)+' |')
lines+=['','| Source | RECENT4 S 개선/D 악화 | SPREAD4 | ALL8 | R2/R3 동일 예측 | 좋은 후보 있으나 ALL8 미선택 | 모든 후보 F0 이하 |',
'|---|---:|---:|---:|---:|---:|---:|']
for src in source_names:
    cells=[r for r in diag if r['source']==src]
    counts=[sum(r['S_improves_D_worsens'] for r in rows if r['source']==src and r['rule']==rule) for rule in ('RECENT4','SPREAD4','ALL8')]
    lines.append(f'| {src} | '+' | '.join(str(v) for v in counts+[sum(r['same_R2_R3_prediction'] for r in cells),sum(r['useful_candidate_missed_by_ALL8'] for r in cells),sum(r['no_candidate_beats_F0'] for r in cells)])+' |')
lines+=['','각 원천의 분모는 8개의 서로 의존하는 budget/arm/seed cell이다. 24개를 독립 데이터셋으로 세지 않는다. 같은 예측 반환은 16/24이며 나머지 8/24에서 선택 창 배치에 따른 차이가 관찰된다.','',
'RECENT4와 SPREAD4는 선택 origin 수가 4개로 같고, ALL8은 정보량도 더 많다. 선택 규칙에 따라 앞부분 이득/뒷부분 손해의 횟수는 5/4/8로 달라졌다. 이 횟수는 S 전체 8개와 D 전체 8개의 F0 대조이며, 규칙별 실제 선택 부분집합 점수는 별도 열이다.','',
'ETTm2의 모든 S 기반 규칙은 F0를 선택했지만 D_diag에서는 F0보다 좋은 후보가 있었다. 따라서 "후보가 모두 같다"와 "선택된 결과가 F0 같다"를 구분한다. sparse FIXED의 두 seed 평균 이득은 native +4.4324%F0, anchor +3.9865%F0다. 이것은 새 test 이득이나 배포 가능한 사후 oracle 성능이 아니다.','',
'각 후보의 S/D 순위 상관, ties, 사후 D 최선과 S 최선의 D 손실 차이는 selection_diagnostics.json에 모두 보존했다. 사후 최선은 낙관적 후보집합 진단이지 회수 가능한 이득의 확증이 아니다.','',
'원점수는 origin별 scaled pinball 분자와 유효 타깃 수를 채널별로 합쳐 재구성했다. per_origin_scores.csv의 분자는 분위 평균과 train scale 나눗셈을 포함한다. 원점수는 mean_channel(sum_origin numerator / sum_origin count)이다. Beijing 결측 때문에 origin별 점수 평균을 대신 쓰지 않았다. 모든 비교 블록은 공통 4채널에 유효 타깃이 있으며 채널을 제거하지 않았다.','',
'시간순 평가의 원칙은 [FPP3](https://otexts.com/fpp3/tscv.html), 유한한 검증 기준의 변동성과 선택 편향 구분은 [Cawley & Talbot (2010)](https://jmlr.org/papers/v11/cawley10a.html)를 참고했다. 이번 분석은 rolling-origin 재학습 CV나 공식 편향 보정 알고리즘이 아니다.']
save('selection_transfer_report.md','\n'.join(lines))
lines=['# 고정 보정 크기 진단','',
'각 arm/cell의 RECENT4 checkpoint를 고정하고 정렬된 분위 예측 사이에서 alpha={0,.25,.5,.75,1}를 공통 적용했다. alpha는 S 전체 8개에서만 고르고 동점은 작은 alpha다. R2의 4개 선택 창 외에 S 전체와 5-alpha 탐색을 추가 사용하므로 R2와 선택 예산이 같지 않다.','',
'| Source | Windows | Arm | Seed | S 선택 alpha | D %F0 | 원 R2 대비 개선 % |',
'|---|---:|---|---:|---:|---:|---:|']
for r in shr:lines.append(f"| {r['source']} | {r['budget']} | {r['arm']} | {r['seed']} | {r['S_selected_alpha']} | {r['D_gain_vs_f0_percent']:+.4f} | {r['D_gain_vs_original_percent']:+.4f} |")
lines+=['','S-only interior alpha가 원모델과 F0를 모두 넘은 것은 4/24 cell이다. Beijing dense/native와 dense/anchor, Electricity dense/native와 dense/anchor의 seed34001에서 각각 관찰됐다. 각 쌍의 seed34000은 alpha1이 선택됐다. 두 원천에서 한 seed씩 나온 사례를 독립 4원천 재현으로 해석하지 않는다.','',
'ETTm2 8개 cell은 R2 자체가 F0이므로 모든 alpha가 같은 예측이다. alpha0 동점 선택은 축소가 학습된 보정을 구조적으로 고쳤다는 근거가 아니다. 나머지 16개 적응된 R2 중 4개에서 interior 개선을 확인했지만 24개 전체를 함께 보고한다. Beijing sparse/anchor seed34001은 alpha.75로 원모델 손해를 일부 줄여도 F0보다 여전히 약1.99% 나쁘다.','',
'correction_scale_curves.csv에 모든 S/D 곡선, correction_scale_selection.csv에 S 선택과 D 사후 최선이 있다. 사후 최선 alpha는 실행 가능한 모델 성능으로 쓰지 않는다. q는 분위 예측이며 조건부 평균·확률분포 mixture가 아니다. 출력 보간은 학습 중 anchoring이나 LoRA weight scaling과 일반적으로 다르다. alpha endpoint는 정확히 일치했다.','',
'기존 mltimeseries hospital_shared_strength_v1에서도 F0–shared LoRA 출력 강도 선택을 이미 수행했다. 이번 국소 곡선을 새 PEFT 기법으로 이름 붙일 근거는 없다. 당시 GLOBAL은 alpha1, INDIVIDUAL은 GLOBAL보다 나빴다. 이번 결과는 그 실험의 새 재현이 아니라 다른 패널에서의 사후 진단이다.']
save('correction_scale_report.md','\n'.join(lines))

# Evidence inventory: paths are tied to inspected revisions, and parameter axes
# are kept distinct from training objective/precision/selection.
refs=read(REF/'inspected_files.json')
commits={r['repo']:r['commit'] for r in refs if r['status']=='READ'}
def row(family,repo,status,code,result,data,params,notes,axis='PEFT family'):
    return dict(family=family,axis=axis,repository=repo,inspected_ref=commits.get(repo,'3750490033ca40adf51fe132fe578fb1b4babd75'),
        status=status,code_path=code,execution_or_result_path=result,model='Chronos-2',datasets=data,trainable_parameters=params,notes=notes)
inv=[]
inv.append(row('original parameter subset: output head','mltimeseries','TRAINED_AND_EVALUATED','experiments/peft_adaptation_scope_v1/modeling.py','_docs/notes/tsfm_topics/02_adaptation_scope/04_peft_adaptation_scope_s1_results_20260908.md','ETTm2; Jena','H_FULL 3653280','Original output_patch_embedding only; not bias-only or norm-only'))
inv.append(row('bias-only / norm-only','all four inspected scopes','NOT_FOUND_IN_INSPECTED_SCOPE','inspected_files.json and local src/scripts text search','none found','not established','not established','Head-only evidence does not prove BitFit/norm-only training'))
inv.append(row('low-rank weight correction','tsfm-peft-method-screen','TRAINED_AND_EVALUATED','src/tsfm_peft_screen/lora.py','results/anchor_window_study_20260914/anchor/fits.json','Beijing; ETTm2; Electricity','1179648 (rank8 alpha16 96 attention projections)','Anchor changes the objective, not the parameter family'))
inv.append(row('low-rank weight correction','forecast-revision-peft','TRAINED_AND_EVALUATED','src/fr_peft/lora.py; src/fr_peft/train.py','results/pilot_v1/summary.json','BMRA; Jena','rank8 alpha16; 96 attention projections; head frozen','40 fits recorded; FR/anchor are loss axes; Jena selected F0'))
inv.append(row('internal additive/conditional adapter','tsfm-peft-method-screen','TRAINED_AND_EVALUATED','src/tsfm_peft_screen/candidates/dualclock.py','results/candidate_02/integrity.json; results/reassessment_diagnostics_20260914/dualclock/fits.json','M5','LoRA 1179648 plus adapter 12704 (summary) or 16192 (DualClock)','Internal residual intervention; not an official Houlsby/ReFT reproduction'))
inv.append(row('head/readout residual adaptation','mltimeseries','TRAINED_AND_EVALUATED','experiments/peft_initial_headroom_v1/fit.py; experiments/peft_head_convergence_v1/fit.py','results/peft_head_convergence_v1/metrics.csv; _docs/notes/tsfm_topics/07_research_direction/34_initial_headroom_20260911.md','BMRA; Jena','HEAD 589301; WIDE/JOINT 1768949','JOINT includes LoRA; HEAD/WIDE freeze backbone; historical D arrays not found locally'))
inv.append(row('head/readout adaptation','tsfm-peft-method-screen','TRAINED_AND_EVALUATED','src/tsfm_peft_screen/forecast_query/model.py:Head','results/forecast_query_equal_time/fits.json; results/forecast_query_equal_time/evaluation.json','ETTm2; Electricity','1179648','Frozen backbone with residual head; not Houlsby Adapter'))
inv.append(row('side network','tsfm-peft-method-screen','TRAINED_AND_EVALUATED','src/tsfm_peft_screen/forecast_query/model.py:Side','results/forecast_query_equal_time/fits.json; results/forecast_query_equal_time/evaluation.json','ETTm2; Electricity','1179648','12 small side layers fed frozen features; separate from LoRA'))
for family in ['soft prompt / prefix','activation scaling: official IA3','representation intervention: official ReFT']:
    inv.append(row(family,'all four inspected scopes','NOT_FOUND_IN_INSPECTED_SCOPE','inspected_files.json; local src/scripts','none found','not established','not established','Absence in the stated inspected scope is not proof of absence in all past branches'))
inv.append(row('custom conditional low-rank gating','tsfm-peft-method-screen','TRAINED_AND_EVALUATED','src/tsfm_peft_screen/lora.py:LowRank','results/candidate_01/integrity.json','Jena','1182720','Freshness gate is not an official IA3 reproduction'))
inv.append(row('anchor / FR / censor losses','tsfm-peft-method-screen','TRAINED_AND_EVALUATED','src/tsfm_peft_screen/reassessment.py; src/tsfm_peft_screen/candidates/fr_lora.py; src/tsfm_peft_screen/candidates/censor.py','results/anchor_window_study_20260914; results/candidate_04; results/candidate_07','Beijing; ETTm2; Electricity; M5','LoRA parameters; no family inferred from loss','Separate objective axis',axis='training objective'))
inv.append(row('activation INT8/local backward','tsfm-peft-method-screen','GRADIENT_OR_RESOURCE_PROBE_ONLY','src/tsfm_peft_screen/memory/compression.py; src/tsfm_peft_screen/memory/local_backward.py','docs/MEMORY_FEASIBILITY_PROTOCOL.md; docs/LOCAL_BACKWARD_PROTOCOL.md','diagnostic batches','no accuracy-training fit established by these probes','Activation quantization is not QLoRA',axis='precision/storage/recomputation'))
inv.append(row('gradient checkpointing/BF16','tsfm-peft-method-screen','TRAINED_AND_EVALUATED','src/tsfm_peft_screen/forecast_query/model.py; scripts/run_reassessment_diagnostics.py','results/forecast_query_equal_time/fits.json; results/reassessment_diagnostics_20260914/anchor/fits.json','ETTm2; Electricity; ETTh1; Traffic','underlying LoRA/head/side parameters','Execution strategy; not a new PEFT family',axis='precision/storage/recomputation'))
inv.append(row('QLoRA','all four inspected scopes','NOT_FOUND_IN_INSPECTED_SCOPE','inspected_files.json; local src/scripts','none found','not established','not established','No verified low-bit backbone + LoRA training; activation INT8 does not qualify',axis='low-bit backbone + LoRA'))
inv.append(row('FULL fine tuning','mltimeseries','TRAINED_AND_EVALUATED','experiments/peft_adaptation_scope_v1/modeling.py','_docs/notes/tsfm_topics/02_adaptation_scope/04_peft_adaptation_scope_s1_results_20260908.md','ETTm2; Jena','all base parameters; count not independently recovered here','Full-FT comparison, not PEFT',axis='full-training reference'))
inv.append(row('frozen forecasting / admission rules','covariate-trust-pilot','NOT_FOUND_IN_INSPECTED_SCOPE','src/covariate_trust/chronos_adapter.py; src/covariate_trust/acquisition_models.py','results/report.md; README.md','synthetic; later acquisition tasks','TSFM weights frozen in inspected pilot; sklearn model parameters separate','Chronos adapter is an API wrapper; fitted admission regression is not TSFM PEFT',axis='non-PEFT prediction/selection'))
inv.append(row('F0/output interpolation/strength choice','mltimeseries','TRAINED_AND_EVALUATED','experiments/hospital_shared_strength_v1/summarize_completed.py (tree verified; not read here)','results/hospital_shared_strength_v1/summary.json; _docs/notes/tsfm_topics/08_hospital_shared_strength/24_hospital_results_20260910.md','767 hospital series','shared LoRA 1206912; alpha selection not trained PEFT weights','Status refers to underlying LoRA experiment; no new PEFT training in posthoc alpha',axis='selection/postprocessing'))
csvsave('peft_family_inventory.csv',inv)
write_scope=dict(remote_files=refs,remote_main_commits=commits,local_main_commit='3750490033ca40adf51fe132fe578fb1b4babd75',
    local_forecast_revision_commit='237d2456991fbdabc764a4be1e4e18d2fbbd16fd',
    uncommitted_preexisting_paths=['automation/research_followup/','research/window_study_followup_20260914/'],
    Study35_raw_predictions='NOT_FOUND in checked timeseries/mlmltime/forecast-revision-peft runs locations; no local mltimeseries checkout. No reconstruction training.',
    branch_scope='Main only; recursive file trees read for three remote repositories, not every file body or historical branch. 28 selected remote files downloaded and inspected plus local src/scripts/result receipts.')
(OUT/'reference_inspection.json').write_text(json.dumps(write_scope,indent=2)+'\n')
save('peft_coverage.md','# PEFT 실행 이력 범위\n\nLoRA 외에도 head/readout-only와 side network가 실제 학습·평가됐고, 내부 residual/conditional adapter도 LoRA와 결합해 실행됐다. H_FULL은 원래 출력층만 학습하는 선택적 파라미터 적응이다. 반면 bias/norm-only, soft prompt/prefix, 공식 IA3/ReFT, QLoRA는 이번 검사 범위에서 실행 근거를 찾지 못했다. 이것은 전체 과거의 부재 증명이 아니다.\n\npeft_family_inventory.csv에 코드·결과·데이터·파라미터·상태와 별도 축을 나눴다. 내부 additive adapter를 Houlsby 재현으로, conditional LoRA gate를 IA3 재현으로, activation INT8을 QLoRA로 세지 않았다. Full-FT는 PEFT가 아니며 F0와 출력 혼합도 자체 PEFT 학습이 아니다. Covariate-trust의 Chronos adapter라는 이름은 모델 학습 어댑터가 아니라 호출 wrapper였고, 별도의 회귀 선택기는 TSFM PEFT로 세지 않았다.\n\n검사 커밋: '+json.dumps(commits)+'\n\n참조 main의 tree와 28개 선택 파일을 읽었다. 모든 과거 branch/file을 전수 조사했다고 주장하지 않는다. Study35 원예측은 확인한 로컬 위치에 없어서 저장 점수/코드만 참조했고 alpha 곡선을 새로 계산하지 않았다. Study35 HEAD도 D에서 F0보다 악화됐으므로 전체 문제를 백본 표현 손상만으로 설명할 수 없다. JOINT에서 표현 손상이 기여했을 가능성은 미확정이다.\n\n주 저장소 HEAD는 지시문의 3750490과 같고 새 완료 학습 결과는 없었다. 앞서 만든 미커밋 감지/정정 파일은 보존했고 이 작업은 commit/push/자동 재개를 하지 않았다.')

proposal='''# 다음 한 개 계획 — 실행 승인 전 제안만

질문: 같은 학습 후보에서 검증 origin 수를 같게 유지하고 시간 배치만 바꾸면, 다음 미사용 기간의 선택 손해가 줄어드는가?

주 비교는 RECENT4 대 SPREAD4다. F0, 사전 고정 LR3e-5/step150, ALL16은 참조다. ALL16과의 차이는 정보량 차이도 포함한다. 새 학습형 selector/controller/adapter와 alpha 재탐색은 만들지 않는다.

새 학습 예산은 **0 fits / optimizer updates 0**이다(사용자가 허용한 제안 상한24 fits 이내). 기존 dense233의 3 sources×2 seeds×2 arms, 각 7개 checkpoint=84상태를 그대로 재사용한다. 따라서 또 native/raw 학습을 반복하는 계획이 아니다. 반복 학습 전에 선택 전달 질문만 분리한다.

승인 후 고정할 다음 기간 후보는 Beijing/Electricity S_new=18432:96:19872, D_new=20992:96:23968, ETTm2 S_new=45056:96:46496, D_new=47616:96:50592다. 모두 기존에 점수화한 E 다음이지만 같은 원천/채널이다. 원시 길이·누락·노출 이력을 다시 검사해 미사용 여부를 확인하고, 겹치면 점수를 열지 않고 BLOCKED로 보고한다. 이 계획서 작성 중 이 새 기간의 예측/점수를 생성하지 않았다.

S_new 16 origins에서 RECENT4=[12,13,14,15], SPREAD4=[0,5,10,15], ALL16=[0..15]. horizon48, context1024, 원래 4채널/마스크/train scale, 동일 7후보·동점 처리. native와 anchor를 따로 유지한다. 모든 선택을 봉인한 뒤 D_new 32 origins에서 선택 모델/F0만 평가한다. D_new 사후 oracle용 전 후보 추론은 하지 않는다.

추론은 batch당 최대2 origins×4채널, GPU forward 상한1536회, backward0, wall4시간으로 제안한다. 실제 캐시 중복/같은 선택 모델은 재사용하며 비용을 기록한다. GPU 공유 안전성이 확인되지 않으면 실행하지 않는다. 이 예산은 이번 D의 256-forward 상한을 늘려 실행하라는 뜻이 아니라 **다음 별도 승인 계획**의 상한이다.

primary는 원천별 paired D_new loss 차이와 source-balanced %F0 차이다. seed는 최적화 반복이며 세 원천을 늘려 세지 않는다. 단일 만능 PASS 수치는 만들지 않는다. 예상은 SPREAD4의 S 좋은/D 나쁜 패턴과 F0 대비 손해 감소다. 반증 대안은 최근성이 실제로 더 유용하거나 어느 S 규칙도 미래 순위를 전달하지 못하는 것이다. 평균·원천별 효과·반례가 시간 배치 개선을 지지하지 않으면 선택 규칙 확장 필요성을 내려놓고 추가 selector 튜닝을 중단한다. 이 결과를 기다리지 않고 새 PEFT 구조를 만들어서는 안 된다.

기존 hospital shared-strength는 alpha/계열 강도 선택이었다. 이 계획은 학습된 후보의 선택 시간 배치만 바꾸고 shrinkage를 새 이름으로 반복하지 않는다. 표준 시계열 검증 설계에 가까우며 신규 방법 우위를 주장하지 않는다. 지지되더라도 새 PEFT 필요성은 별도 미확정이다.

사용자가 전체 계획에 동의하기 전 실행하지 않는다. 이번 진단은 이 제안을 작성하고 종료한다.
'''
save('next_experiment_proposal.md',proposal)
main=['# TSFM PEFT 시간적 전달 진단 — 결과','',
'**A~C 완료, D는 DEFERRED_GPU_BUSY. 신규 fits=0, optimizer updates=0, GPU forward/backward=0.**','',
'## 1. 무엇이 문제였나','',
'48-fit 연구는 정상 완료됐지만 적은 윈도에서 anchor가 더 유리하다는 가설은 지지되지 않았다. 보고서 seed 열에는 개선율 대신 loss ratio를 출력한 별도 오류가 있었다. 이를 고쳐도 원래 평균 효과와 판정은 바뀌지 않았다.','',
'## 2. 왜 중요한가','',
'Beijing dense는 anchor가 plain보다 +4.172% 좋아도 F0 대비는 약 +0.186%다. 적응 손해 회복과 기본모델을 넘는 이득은 다르다. ETTm2 원래 선택은 step0/F0였다. Study35는 동결 백본 HEAD도 V 이득/D 손해를 보였으므로 백본 표현 손상 하나로 모든 실패를 설명할 수 없다.','',
'## 3. 왜 이 대조인가','',
'B는 같은 후보를 선택하는 창 배치만 바꿨다. RECENT4/SPREAD4는 같은 4 origins, ALL8은 8 origins다. C는 RECENT4 한 checkpoint에서 출력 보정 크기만 바꿨다. H3/H4용 국소 gradient 개입 D는 미수행이므로 관측된 선택/축소 차이를 목적함수 충돌이나 망각의 인과 증거로 바꾸지 않는다.','',
'## 4. 설정 근거','',
'3 sources×2 budgets×2 seeds×2 arms=24개의 서로 의존하는 cell. 기존 V16을 S8/D_diag8로 재사용했다. 원천 수는3이며 step0 중복을 합친 동일7후보를 각 arm 안에서 비교했다. 원래 타깃/마스크/scale/분위 정렬, 채널별 분자/분모를 보존했다. 모든 시간 경계와 4채널 비교를 확인했고 누락 캐시는 없었다.','',
'## 5. 무엇을 지지하나','',
'- SELECTION_TRANSFER_PATTERN: R2/R3의 예측이 8/24 cell에서 달랐으며 S 개선/D 악화는 RECENT4 5, SPREAD4 4, ALL8 8 cell이다. 원천별 반례와 원손실은 선택 보고서/CSV에 모두 포함했다.\n- SIMPLE_SHRINKAGE_EXPLAINS_PART: S로 선택한 interior alpha가 원모델과 F0를 모두 넘은 경우는4/24, Beijing/Electricity dense의 seed34001 두 arm이다. seed34000에서는 alpha1이며 보편 재현이 아니다.\n- ETTm2는 선택 후보가 F0로 같았지만 저장된 다른 적응 후보 중 D에서 좋은 것이 있었다. ALL8은8/8 해당 cell에서 그런 후보를 선택하지 못했다. 사후 oracle은 진단값이다.\n- NUMERICAL_ERRATUM_ONLY는 seed 파생 열에 한정된다. corrected_seed_gains.csv와 aggregate_replay.json이 원점수부터 판정까지의 의존 경로를 기록한다.','',
'## 6. 무엇은 아직 모르나','',
'8개 D origins와 재사용 V로 새 일반화·유의성·동등성을 선언할 수 없다. 일부 축소 효과만으로 새 loss 필요성은 약하다. H3/H4는 D 미실행 때문에 INSUFFICIENT_EVIDENCE다. 외부 YOLO 작업과 RustDesk가 GPU에 있었고 순간 여유 약5.7GiB만으로 최대 점유/간섭 안전성을 보장할 수 없어 CPU 작업만 병행했다. 기존 GPU guard를 유지하고 예약·무한 대기를 만들지 않았다.','',
'## 7. 다음 한 개 계획','',
'새 기간에서 동일 창 수 RECENT4/SPREAD4의 선택 전달을 검증하는 **0-new-fit** 계획을 제안한다. 기존84개 dense checkpoint를 재사용하므로 24-fit 상한 아래이며 새로운 모델 학습을 반복하지 않는다. 구체적 분할·추론 예산·반증 대안은 next_experiment_proposal.md에 있고 실행하지 않았다.','',
'| 관찰 | 현재 지지되는 설명 | 배제하지 못한 설명 | 다음 행동 |',
'|---|---|---|---|',
'| 선택 창을 바꾸면 일부 D 손실이 달라짐 | 선택의 시간적 전달 패턴 | 작은 표본/선택 잡음/기간별 목표 차이 | 미사용 기간에서 동일 창 수 선택 대조 |',
'| 일부 S-only 축소가 원모델과 F0를 넘음 | 보정 크기가 일부 사례 설명 | 선택 예산 증가/seed 변동 | 복잡한 보존 기법 발명 보류 |',
'| 국소 개입 미실행 | 없음 | objective mismatch/temporal conflict | GPU 충돌 없이 별도 확인 전 인과 주장 금지 |','',
'## 검증·변경·실행 명령','',
'예측 캐시228개(기존 E30개 포함), max primary 재계산 오차1.11e-16, sufficient-stat 재집계312회, checkpoint192개 hash 확인. 과거/소스1193개 파일 불변. 정정·시간 경계·선택 누출·alpha endpoint 8개 테스트 통과. CPU 계산 약5초(자료 조사·보고서 시간 제외), CPU peak RSS 약889MiB. 참조 저장소 main의 선택 파일28개 조사; 이전 미커밋 작업은 보존.','',
'새 코드: scripts/run_temporal_transfer_diagnostic_v1.py, scripts/report_temporal_transfer_diagnostic_v1.py, tests/test_temporal_transfer_diagnostic_v1.py. 산출물은 results/temporal_transfer_diagnostic_v1/, 지시문은 research/temporal_transfer_diagnostic_v1/USER_INSTRUCTION.txt. commit/push/브랜치 변경/자동 후속 없음.','',
'```bash\nCUDA_VISIBLE_DEVICES=\'\' scripts/with_cuda.sh .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_temporal_transfer_diagnostic_v1.py\nCUDA_VISIBLE_DEVICES=\'\' scripts/with_cuda.sh .venv/bin/python scripts/run_temporal_transfer_diagnostic_v1.py\n.venv/bin/python scripts/report_temporal_transfer_diagnostic_v1.py\n```','',
'위 진단 실행 명령은 이미 완료됐다. 동명 출력이 있으면 거부하므로 다시 실행하려면 새 run suffix와 재실행 이유를 고정해야 한다. 기존48fit를 재학습하지 않는다.','',
'시간순 검증 및 선택 편향의 표준 배경: [FPP3](https://otexts.com/fpp3/tscv.html), [Cawley & Talbot](https://jmlr.org/papers/v11/cawley10a.html). 이번 S/D 선택과 alpha 조합은 사후 개발 진단이며 신규 방법이나 보편 검정이 아니다.']
save('REPORT.md','\n'.join(main))
(RESEARCH/'REPORT.md').write_text('# 완료 보고서\n\n[전체 진단 결과](../../results/temporal_transfer_diagnostic_v1/REPORT.md)\n\nA~C 완료. D는 외부 GPU 작업의 안전한 동시 실행 여유가 확인되지 않아 DEFERRED_GPU_BUSY. 신규 학습·자동 후속·push 없음.\n')
print('Reports and inventory written; no training or scheduling.')
