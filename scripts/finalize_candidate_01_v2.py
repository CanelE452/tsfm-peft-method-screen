"""Independent numerical/artifact audit and deterministic v2 report generation."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from tsfm_peft_screen.reproducibility import ROOT,sha,write_json
from tsfm_peft_screen.selection import require_seal
from tsfm_peft_screen.metrics import score,replay
OUT=ROOT/'results/candidate_01_v2';CACHE=ROOT/'.cache/candidate_01_v2'
def read(p):return json.loads(p.read_text())
def csvrows(p):
    with open(p) as f:return list(csv.DictReader(f))
contract=read(OUT/'contract.json');cfg=contract['config'];status=read(OUT/'status.json')
assert status['verdict'] in ['PASS','WEAK','FAIL'],status
assert read(OUT/'job_exit.json')['exit_code']==0
commit=contract['execution_commit']
for name,h in contract['source_hashes'].items():
    content=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=ROOT)
    assert hashlib.sha256(content).hexdigest()==h,('Execution hash mismatch',name)
assert sha(OUT/'sampling_manifest.json')==contract['sampling_sha256']
require_seal(OUT/'selection.json',contract)
assert sha(OUT/'selection.json')==read(OUT/'evaluation_open.json')['selection_sha256']
assert read(OUT/'evaluation_open.json')['all_twelve_fits_complete_before_open']
assert read(OUT/'round0.json')['status']=='PASS'
with np.load(CACHE/'gate_clean.npz') as current, np.load(ROOT/'.cache/candidate_01/gate_clean.npz') as original:
    for key in ['prediction','target','scale']:np.testing.assert_array_equal(current[key],original[key])
attempts=read(OUT/'attempts.json');usage=read(OUT/'resource_usage.json');integ=read(OUT/'integrity.json')
expected={f'{seed}_{arm}_{recipe["id"]}' for seed in cfg['seeds'] for arm in cfg['arms'] for recipe in cfg['recipes']}
assert {a['fit'] for a in attempts}==expected and len(attempts)==12
assert all(a['status']=='COMPLETE' for a in attempts)
assert len(usage['fits'])==12 and sum(u['optimizer_steps'] for u in usage['fits'])==4320
assert integ['status']=='PASS' and len(integ['fits'])==12
for fit in integ['fits']:
    assert fit['identity_max_abs']==0 and fit['checkpoint_replay_max_abs']==0 and fit['frozen_unchanged'] and fit['optimizer_covers_all_trainables']
    if 'STANDARD_LORA' not in fit['fit']:
        assert fit['conditional_smoke']['train_state_replacement_prediction_max_abs']>0
trajectories=csvrows(OUT/'trajectories.csv');selected=read(OUT/'selection.json')['winners']
assert len(selected)==6
for fit in expected:
    rows=[r for r in trajectories if r['fit']==fit]
    assert [int(r['step']) for r in rows]==cfg['checkpoints']
    for r in rows:
        np.testing.assert_allclose(float(r['validation_loss']),np.mean([float(r['V_'+v]) for v in contract['train_variants']]),rtol=0,atol=1e-12)
for selection in selected:
    rows=[r for r in trajectories if r['arm']==selection['arm'] and int(r['seed'])==selection['seed']]
    best=min(rows,key=lambda r:(float(r['validation_loss']),int(r['step']),r['recipe']))
    assert selection['step']==int(best['step']) and selection['recipe']==best['recipe']
    assert selection['validation_loss']==float(best['validation_loss'])
    assert sha(selection['checkpoint'])==selection['checkpoint_sha256']
schedules=read(OUT/'sampling_manifest.json')
for seed in cfg['seeds']:
    assert len(schedules[str(seed)])==360
    assert all(sum(r['variant']==variant for r in schedules[str(seed)])==90 for variant in contract['train_variants'])
# Ensure the original screen plus recovered Candidate05 were not modified.
historical='1f7ac730bdff95108815261b0f23f35ba0876a20'
folders=[ROOT/'results'/f'candidate_{i:02}' for i in range(1,8)]+[ROOT/'results/candidate_05_repaired']
for folder in folders:
    for path in folder.rglob('*'):
        if path.is_file():
            content=subprocess.check_output(['git','show',f'{historical}:{path.relative_to(ROOT)}'],cwd=ROOT)
            assert hashlib.sha256(content).hexdigest()==sha(path),('Original artifact modified',path)
metrics=csvrows(OUT/'metrics.csv');assert len(metrics)==35
from tsfm_peft_screen.data import Panel
panel=Panel('jena');assert sha(panel.root/'manifest.json')==contract['manifest_sha256'];panel.open_e(OUT/'selection.json',contract)
yexpected=np.stack([panel.window(int(o))[1] for o in panel.origins['evaluation']])
errors=[];metric_hashes={};origin_losses={}
for row in metrics:
    file=f"E_{row['seed']}_{row['arm']}_{row['variant']}.npz";path=CACHE/file
    assert sha(path)==row['prediction_sha256'];metric_hashes[file]=sha(path);errors.append(replay(path))
    with np.load(path) as z:
        np.testing.assert_array_equal(z['target'],yexpected)
        np.testing.assert_array_equal(z['scale'],panel.scale)
        np.testing.assert_array_equal(z['origins'],panel.origins['evaluation'])
        calculated=score(z['prediction'],z['target'],z['scale'])
        for k,v in calculated.items():np.testing.assert_allclose(v,float(row[k]),rtol=0,atol=1e-12)
        origin_losses[file]=[score(z['prediction'][j:j+1],z['target'][j:j+1],z['scale'])['scaled_2pinball'] for j in range(30)]
def loss(arm,seed,variant=None,clean=False):
    rr=[float(r['scaled_2pinball']) for r in metrics if r['arm']==arm and (arm=='F0' or int(r['seed'])==seed) and (r['variant']==variant if variant is not None else (r['variant']=='clean')==clean)]
    assert len(rr)==(1 if variant is not None or clean else 4)
    return float(np.mean(rr))
f0=loss('F0',30000);detail=[];regimes=[]
for seed in cfg['seeds']:
    best=min(cfg['arms'][:2],key=lambda arm:loss(arm,seed));proposed=loss('AFFINE_V2',seed)
    gain=100*(loss(best,seed)-proposed)/f0
    clean=100*(loss('AFFINE_V2',seed,clean=True)-min(loss(a,seed,clean=True) for a in cfg['arms'][:2]))/loss('F0',30000,clean=True)
    detail.append(dict(seed=seed,strongest_baseline=best,baseline_primary=loss(best,seed),proposed_primary=proposed,gain_percent_f0=gain,clean_degradation_percent_f0=clean))
    for variant in contract['evaluation_variants']:
        regimes.append(dict(seed=seed,variant=variant,gain_vs_feature_percent_f0=100*(loss('FEATURE_LORA',seed,variant)-loss('AFFINE_V2',seed,variant))/loss('F0',seed,variant),gain_vs_standard_percent_f0=100*(loss('STANDARD_LORA',seed,variant)-loss('AFFINE_V2',seed,variant))/loss('F0',seed,variant)))
mean=float(np.mean([r['gain_percent_f0'] for r in detail]));positive=all(r['gain_percent_f0']>0 for r in detail)
passed=positive and mean>=1 and all(r['clean_degradation_percent_f0']<=.5 for r in detail)
verdict='PASS' if passed else 'WEAK' if positive else 'FAIL'
assert status['verdict']==verdict
np.testing.assert_allclose(mean,status['mean_gain_percent_f0'],rtol=0,atol=1e-12)
for row,record in zip(detail,status['seeds']):
    for key,value in row.items():
        if isinstance(value,float):np.testing.assert_allclose(value,record[key],rtol=0,atol=1e-12)
        else:assert value==record[key]
branch=csvrows(OUT/'branch_diagnostics.csv')
for fit in expected:
    if 'STANDARD_LORA' in fit:continue
    smoke=next(r for r in branch if r['fit']==fit and r['kind']=='optimizer' and int(r['step'])==8)
    assert float(smoke['conditional_gradient_norm'])>0 and float(smoke['conditional_update_norm'])>0
    if 'AFFINE_V2' in fit:
        assert float(smoke['scale_gradient_norm'])>0 and float(smoke['shift_gradient_norm'])>0
# Check the actual scheduled, clean-centered training state design, not E.
from tsfm_peft_screen.candidates.freshness_v2 import corrupt
state_cache={};support={}
for seed in cfg['seeds']:
    samples=[]
    for item in schedules[str(seed)]:
        for origin in item['origins']:
            key=(origin,item['variant'])
            if key not in state_cache:
                x,_=panel.window(origin,target=False);_,r=corrupt(x,item['variant'],origin)
                tokens=np.concatenate([r.reshape(4,21,16,3).mean(2),np.repeat(r[:,-1:,:],4,axis=1)],axis=1)
                state_cache[key]=tokens.reshape(-1,3).astype(np.float64)-[1.,0.,1.]
            samples.append(state_cache[key])
    design=np.concatenate(samples);singular=np.linalg.svd(design,compute_uv=False)
    support[str(seed)]=dict(centered_state_rank=int(np.linalg.matrix_rank(design)),singular_values=singular.tolist(),scheduled_token_rows=len(design))
    assert support[str(seed)]['centered_state_rank']==3
if '--verify-only' in sys.argv:
    stored=read(OUT/'verification.json')
    assert stored['prediction_sha256']==metric_hashes and stored['verdict']==verdict
    assert read(OUT/'train_state_support.json')==support
    print('V2 VERIFICATION PASS: 12 fits, 4320 updates, 35 E prediction caches, all original artifacts unchanged; verdict',verdict)
    raise SystemExit(0)
write_json(OUT/'verification.json',dict(status='PASS',verdict=verdict,execution_commit=commit,fit_count=12,optimizer_updates=4320,metric_replay_max_abs=max(errors),source_hashes_verified=True,original_clean_f0_gate_predictions_exact=True,selection_independently_recomputed=True,targets_unchanged=True,original_artifacts_unchanged=True,conditional_branches_active=True,prediction_sha256=metric_hashes))
write_json(OUT/'origin_losses.json',origin_losses)
write_json(OUT/'train_state_support.json',support)
from tsfm_peft_screen.runners.freshness_v2 import csv_write
csv_write(OUT/'regime_effects.csv',regimes)
# Metadata-only corruption receipts across the fixed splits: no E tuning.
from tsfm_peft_screen.candidates.freshness_v2 import corrupt
corruptions=[]
for split in ['gate','train','validation','evaluation']:
    variants=contract['evaluation_variants'] if split=='evaluation' else contract['train_variants']
    for origin in panel.origins[split]:
        x,_=panel.window(int(origin),target=False)
        for variant in variants:
            a,r=corrupt(x,variant,int(origin));observed=np.isfinite(a)
            corruptions.append(dict(split=split,origin=int(origin),variant=variant,unique_channel_masks=int(np.unique(observed,axis=0).shape[0]),availability_per_channel=observed.mean(1).tolist(),mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest()))
write_json(OUT/'corruption_receipts.json',corruptions)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi':150,'savefig.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for ax,seed in zip(axes,cfg['seeds']):
    arms=['F0']+cfg['arms'];values=[loss(a,seed) for a in arms]
    bars=ax.barh([a.replace('_',' ') for a in arms],values,color=['#167f78' if a=='AFFINE_V2' else '#7d91ad' for a in arms]);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.2)
    for b,v in zip(bars,values):ax.text(v+.008,b.get_y()+b.get_height()/2,f'{v:.6f}',va='center',fontsize=8)
    ax.set_title(f'Seed {seed}');ax.set_xlabel('Mean unseen corruption loss (lower is better)')
fig.suptitle(f'Freshness v2 — {verdict}');fig.tight_layout();fig.savefig(figdir/'primary.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for ax,seed in zip(axes,cfg['seeds']):
    for arm in cfg['arms']:
        for recipe in cfg['recipes']:
            rr=[r for r in trajectories if int(r['seed'])==seed and r['arm']==arm and r['recipe']==recipe['id']]
            ax.plot([int(r['step']) for r in rr],[float(r['validation_loss']) for r in rr],label=f"{arm} {recipe['id']}",lw=1)
    ax.set_title(f'Seed {seed} validation');ax.set_xlabel('Optimizer update');ax.set_ylabel('Mean V loss');ax.legend(fontsize=6)
fig.tight_layout();fig.savefig(figdir/'learning_curves.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
labels=[str(r['seed']) for r in detail]
axes[0].bar(labels,[r['gain_percent_f0'] for r in detail],color='#167f78');axes[0].axhline(1,ls='--',color='gray',label='Mean target >=1%');axes[0].axhline(0,lw=.7,color='black');axes[0].set_title('Gain vs strongest simple baseline');axes[0].set_ylabel('% F0');axes[0].legend(fontsize=8)
axes[1].bar(labels,[r['clean_degradation_percent_f0'] for r in detail],color='#7d91ad');axes[1].axhline(.5,ls='--',color='gray',label='Per-seed limit 0.5%');axes[1].axhline(0,lw=.7,color='black');axes[1].set_title('Clean degradation vs best clean baseline');axes[1].set_ylabel('% F0');axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(figdir/'diagnostics.png');plt.close(fig)
table='| Seed | Strongest baseline | Baseline loss | Affine loss | Gain %F0 | Clean degradation %F0 |\n| --- | --- | ---: | ---: | ---: | ---: |\n'
for r in detail:table+=f"| {r['seed']} | {r['strongest_baseline']} | {r['baseline_primary']:.9f} | {r['proposed_primary']:.9f} | {r['gain_percent_f0']:.6f} | {r['clean_degradation_percent_f0']:.6f} |\n"
maxstep=sum(s['step']==360 for s in selected);stepzero=sum(s['step']==0 for s in selected)
text=f'''# Candidate01 v2 — {verdict}

[문제·범위]
동일 시점에 손상시킨 원래 gate와 구분해 채널별 갱신 간격·offset·block 위치가 다른 관측 과정을 검사했다. [사전 고정 계약](../../docs/CANDIDATE_01_V2.md), [사용자 검토문](../../docs/USER_DESIGN_REVIEW_2026_09_13.md). 이전 실패 결과와 05 복구 결과는 byte 단위로 그대로 보존했다.

[방법·강한 대조]
Standard LoRA, 동일 관측 상태를 사용하는 기존 additive Feature LoRA, clean-centered scale+shift Affine v2. 표준 LoRA1,179,648 parameters; Feature 추가3072, Affine 추가4608. 같은 frozen Chronos-2/native head와 상태·시간·마스크 정보. Affine의 clean-reference 추가항은0이지만 공유 LoRA 학습 때문에 F0 clean 성능 보존을 보장하지 않는다. Affine modulation 자체의 신규성은 주장하지 않는다.

[데이터·학습]
Jena hourly4채널, context336/horizon48, 원래 train/V/E split과 train scale. Train mixture clean/refresh/block/combined 각25%. 3 arms × 2 recipes × 2 seeds, 12 fits × 360 updates =4320 updates. LoRA LR3e-5/1e-4, conditional LR3e-4/1e-3를 Feature/Affine에 동일하게 제공했다. 6개 최종 모델을 V로 선택·봉인한 뒤 E를 열었다. 선택된 step0 모델 {stepzero}/6, step360 모델 {maxstep}/6. E는 이미 연구에 사용한 개발 구간이며 독립 holdout이 아니다.

[문제 gate]
{json.dumps(read(OUT/'round0.json'),ensure_ascii=False)}

[raw·relative 결과]

{table}
![Primary comparison](figures/primary.png)

F0 unseen 평균 손실 {f0:.9f}. Gain은 `100*(baseline-proposed)/F0`; 일반 baseline-relative 개선율이 아니다. 두 seed 평균 gain **{mean:.6f}%F0**. Affine의 F0 대비 이득은 seed30000 {100*(f0-loss('AFFINE_V2',30000))/f0:.6f}%F0, seed30001 {100*(f0-loss('AFFINE_V2',30001))/f0:.6f}%F0다. 그러나 첫 seed의 Feature 대비 이득 {100*(loss('FEATURE_LORA',30000)-loss('AFFINE_V2',30000))/f0:.6f}%F0는 더 강한 Standard 대비 이득으로 이어지지 않았고, 두 번째 seed에서는 Feature보다도 나빴다. 전체35개 arm/seed/regime 지표는 metrics.csv, 개별 corruption 효과는 regime_effects.csv에 있다. Clean과 unseen 결과를 혼합해 primary를 만들지 않았다.

[성공·실패 판정]
**{verdict}**. 두 seed 각각 양의 효과, 두 seed 평균 ≥1%F0, 각 seed clean 손해 ≤0.5%F0를 모두 요구했다. E 결과로 설정·checkpoint·threshold를 바꾸지 않았다. 양의 미세 차이를 통계적 유의성으로 해석하지 않는다.

[무결성·조건부 계산]
CPU 기능 테스트와 실제 train-only 첫8 scheduled updates에서 Feature/Affine의 조건부 gradient/update 및 상태 교체에 따른 예측 변화가 확인됐다. Affine scale·shift 양쪽 gradient도 비영이다. 실제 학습 schedule에서 clean-centered state 행렬의 rank는 두 seed 모두3이다(train_state_support.json). 세부 범위·gradient/update norm은 branch_diagnostics.csv, 선택 checkpoint의 상태 민감도는 integrity.json에 있다. 초기 F0 identity와 checkpoint replay 오차0, frozen weights 불변. 저장된 E 예측35개를 독립 계산한 지표 replay 최대 오차 {max(errors):.3g}; 타깃이 원래 값과 같고 selection이 V 최소값인지 독립 검증했다. 별도 regularizer는 없다.

[계산량]
이번 추가12 fits, wall {usage['wall_seconds']:.3f}s, peak allocated GPU {max(u['gpu_peak_bytes'] for u in usage['fits'])} bytes. Gate 및 진단 forward 시간은 candidate wall에 포함된다. job_exit.json은 worker 프로세스 시작·종료 시간을 포함한 총시간이다. 실행 전 GPU 유휴 대기 시간은 이 두 기록에서 제외된다. 누적 실험46 standard fits +9 stream attempts (8 complete,1 historical abort). 기존38-fit screen 예산을 새 실험까지 포함한 것처럼 표기하지 않는다.

[한계·후속]
두 seed·단일 개발 데이터의 한정된 학습 예산이다. 각 origin의 관측 episode마다 주기·offset 배치를 생성하므로, 전체 기간에 하나의 지속적인 센서 달력이 있는 상황까지 검증한 것은 아니다. 실제 비동기 센서 분포, 독립 원천 일반화, 충분한 수렴, 논문 수준 신규성을 입증하지 않는다. 새 모듈을 충분히 학습하면 성공한다거나 본 실패가 아이디어 전체를 반증한다고 주장하지 않는다. {'조건을 통과했으므로 설정을 고정한 새 원천 검토 대상이다. 새 원천 실험은 실행하지 않았다.' if passed else '사전 규칙에 따라 이 v2 분기를 종료한다. 추가 LR·seed·threshold 탐색이나 새 원천 실험은 진행하지 않는다.'} Round2 실행0.

[학습 경로·진단 그림]

![Validation learning curves](figures/learning_curves.png)

![Fixed threshold diagnostics](figures/diagnostics.png)

[실행 리비전]
`{commit}`. 계약·소스·선택·캐시 해시 검증은 verification.json에 기록했다. [원래 7개 후보 결과](../screening_summary/latest_review.md)와 v2는 별도 실험이다.
'''
(OUT/'RESULT.md').write_text(text)
summary=ROOT/'results/screening_summary'
write_json(summary/'freshness_v2_review.json',dict(verdict=verdict,execution_commit=commit,status_sha256=sha(OUT/'status.json'),seeds=detail,mean_gain_percent_f0=mean,additional_fits=12,cumulative_standard_fits=46,cumulative_stream_attempts=9,round2_executed=False,new_source_eligible_for_review=passed,original_screen_unchanged=True))
(summary/'freshness_v2_review.md').write_text(f'# Freshness v2 development review — {verdict}\n\n'+table+f'\nMean gain: {mean:.6f}%F0. [Full report](../candidate_01_v2/RESULT.md). Original seven-candidate results remain unchanged. Cumulative46 standard fits +9 stream attempts. No Round2 or new-source experiment executed.\n')
print(json.dumps(dict(verification='PASS',verdict=verdict,seeds=detail,mean_gain_percent_f0=mean,metric_replay_max_abs=max(errors)),indent=2))
