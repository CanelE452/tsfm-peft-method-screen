"""Post-selection CPU analysis; never trains or changes selections."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/additive_b0_mpl')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
from .reference_core import STATES
from .model import correction_gate
from .prepare import PREV,PREVC,prepare

@torch.no_grad()
def analysis():
    setup();assert read(OUT/'verification.json')['status']=='VERIFIED';prepare();check_seal()
    scores=pd.read_csv(OUT/'scores_by_condition.csv');scores['panel']=scores.condition.map(lambda x:'FAULT' if x.startswith(('POINT','BURST')) else x)
    summary=scores.groupby(['source','arm','panel']).nmae.mean().unstack().reset_index();summary.to_csv(OUT/'panel_summary.csv',index=False)
    scores.groupby(['source','arm','seed','panel']).nmae.mean().unstack().reset_index().to_csv(OUT/'seed_panel_summary.csv',index=False)
    # Baseline metric replay is numerically identical to the old paired B0, not an aggregate substitute.
    old=pd.read_csv(PREV/'scores_by_condition.csv');a=scores[scores.arm=='C0'].sort_values(['source','seed','condition']);b=old[old.arm=='B0'].sort_values(['source','seed','condition'])
    np.testing.assert_allclose(a[['nmae','mae','pinball','nrmse','crossing']].to_numpy(),b[['nmae','mae','pinball','nrmse','crossing']].to_numpy(),rtol=1e-10,atol=1e-12)
    # V mask ablations vs stored normal V predictions, zero optimizer.
    rows=[];diag=read(OUT/'C3_validation_diagnostics.json')
    for r in read(OUT/'MODEL_SELECTION.json'):
        if r['arm']!='C3':continue
        source,seed=r['source'],r['seed'];key=f'{source}_C3_{seed}'
        p=np.load(CACHE/'predictions'/f'{key}_V.npy',mmap_mode='r');y=np.load(CACHE/'conditions'/source/'V_SELECT_y.npy');s=np.tile(read(OUT/'DATA_MANIFEST.json')[source]['sigma_train_population'],len(y)//4)
        normal=(np.abs(p[:,4].astype(float)-y)/s[:,None]).mean(-1).reshape(10,2,64,4).mean((1,2,3))
        for mode,vals in diag[key].items():
            for condition,x,z in zip(STATES,normal,vals):rows.append(dict(source=source,seed=seed,condition=condition,mode=mode,normal_nmae=float(x),altered_nmae=z,error_increase_pct=float(100*(z/x-1)),optimizer_updates=0))
    pd.DataFrame(rows).to_csv(OUT/'V_ablation_effects.csv',index=False)
    # Mask distribution is input-only and seed-independent.
    gates=[]
    for source in SOURCES:
        for role in ['V_SELECT','E_DISCOVERY']:
            f=CACHE/'conditions'/source;x=np.load(f/f'{role}_x.npy',mmap_mode='r');s=np.load(f/f'{role}_sigma.npy',mmap_mode='r');chunks=[]
            for lo in range(0,len(x),256):chunks.append(correction_gate(torch.tensor(x[lo:lo+256].copy()),torch.tensor(s[lo:lo+256].copy())).numpy())
            g=np.concatenate(chunks).reshape(10,-1)
            for condition,values in zip(STATES,g):gates.append(dict(source=source,role=role,condition=condition,mean_correction_gate=float(values.mean()),p10=float(np.quantile(values,.1)),median=float(np.median(values)),p90=float(np.quantile(values,.9)),zero_fraction=float((values==0).mean()),one_fraction=float((values==1).mean())))
    pd.DataFrame(gates).to_csv(OUT/'mask_behavior.csv',index=False)
    inf=read(OUT/'inference_resources.json');ledger=[];resources=[];selections=[]
    for p in sorted((OUT/'fits').glob('*/receipt.json')):
        r=read(p);key=f"{r['source']}_{r['arm']}_{r['seed']}";cost=inf.get(key)
        ledger.append({k:r[k] for k in ['fit','source','arm','seed','lr','status','updates','optimizer_seconds','validation_seconds','checkpoint_io_seconds','peak_allocated','peak_reserved']})
        selections.append(dict(source=r['source'],arm=r['arm'],seed=r['seed'],lr=r['lr'],selected_step=r['selected']['step'],objective=r['selected']['objective'],unchanged_baseline_selected=r['selected']['step']==0))
        resources.append(dict(source=r['source'],arm=r['arm'],seed=r['seed'],lr=r['lr'],trainable_parameters=294912 if r['arm']=='C1' else 8712,additional_train_seconds=r['optimizer_seconds'],validation_seconds=r['validation_seconds'],checkpoint_io_seconds=r['checkpoint_io_seconds'],train_peak_allocated_mib=r['peak_allocated']/2**20,E_inference_seconds=cost['inference_seconds'] if cost else None,E_peak_allocated_mib=cost['peak_allocated']/2**20 if cost else None))
    pd.DataFrame(ledger).to_csv(OUT/'FIT_LEDGER.csv',index=False);pd.DataFrame(resources).to_csv(OUT/'resources.csv',index=False);pd.DataFrame(selections).to_csv(OUT/'selection_summary.csv',index=False)
    baseline=[read(p) for p in (PREV/'fits').glob('*/receipt.json') if read(p)['arm']=='B0']
    save(OUT/'shared_baseline_cost.json',dict(historical=True,not_new_training=True,fits=len(baseline),updates=sum(r['updates'] for r in baseline),optimizer_seconds=sum(r['optimizer_seconds'] for r in baseline),scope='all v2 B0 LR selection and repeat fits; same upstream cost shared by C0/C1/C2/C3'))
    guards=[json.loads(l) for p in OUT.glob('gpu_*.jsonl') for l in open(p)]
    save(OUT/'execution_resource_summary.json',dict(main_optimizer_seconds=sum(r['additional_train_seconds'] for r in resources),validation_seconds=sum(r['validation_seconds'] for r in resources),checkpoint_io_seconds=sum(r['checkpoint_io_seconds'] for r in resources),E_inference_seconds=sum(v['inference_seconds'] for v in inf.values()),run_all_wall_seconds=read(OUT/'run-all_wall.json')['seconds'],minimum_free_gpu_mib=min(r['free_mib'] for r in guards),unapproved_external_compute_samples=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in r['apps']) for r in guards),new_cache_bytes=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file() and not p.is_symlink())))
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    effects=pd.read_csv(OUT/'paired_effects.csv')
    panels=['FAULT','REFERENCE','SHIFT4','SHIFT8','SHIFT_POINT']
    fig,axes=plt.subplots(2,5,figsize=(18,7))
    for i,source in enumerate(SOURCES):
        for j,panel in enumerate(panels):
            f=summary[summary.source==source].set_index('arm');axes[i,j].bar(['C0','C1','C2','C3'],[f.loc[arm,panel] for arm in ['C0','C1','C2','C3']]);axes[i,j].set_title(source+' / '+panel);axes[i,j].set_ylabel('nMAE (lower better)')
    plt.tight_layout()
    for ext in ['png','pdf']:plt.savefig(figures/f'raw_scores.{ext}',dpi=160)
    plt.close()
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for i,source in enumerate(SOURCES):
        for j,base in enumerate(['C0','C1','C2']):
            f=effects[(effects.source==source)&(effects.new=='C3')&(effects.baseline==base)].set_index('panel').loc[panels]
            axes[i,j].hlines(np.arange(5),f.ci_low_pct,f.ci_high_pct,color='tab:blue');axes[i,j].plot(f.gain_pct,np.arange(5),'o');axes[i,j].axvline(0,color='grey');axes[i,j].set_yticks(np.arange(5),panels);axes[i,j].set_title(f'{source}: C3 vs {base}');axes[i,j].set_xlabel('gain %; 95% paired time-block CI')
    plt.tight_layout()
    for ext in ['png','pdf']:plt.savefig(figures/f'incremental_effects.{ext}',dpi=160)
    plt.close()
    # Exact accounting and V-only LR selection independently reconstructed.
    from .train import fit_id
    choices=read(OUT/'LR_SELECTION.json')
    for source in SOURCES:
        for arm in ARMS:
            fits=[read(OUT/'fits'/fit_id(source,arm,81550,lr)/'receipt.json') for lr in [1e-4,3e-4]]
            chosen=min(fits,key=lambda r:(r['selected']['objective'],r['lr'],r['selected']['step']));assert choices[source][arm]['fit']==chosen['fit']
    smoke=[json.loads(l) for p in OUT.glob('smoke_*.jsonl') for l in open(p)];assert len(smoke)==12
    for path,h in read(OUT/'SOURCE_MANIFEST.json')['immutable_hashes'].items():assert sha(ROOT/path)==h
    save(OUT/'publication_audit.json',dict(status='VERIFIED',baseline_scores_exact_replay=True,LR_selection_recomputed=True,old_results_and_arrays_unchanged=True,smoke_journal_updates=12,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in EXP.glob('*.py')},score_hashes={name:sha(OUT/name) for name in ['scores_by_origin.csv','scores_by_condition.csv','paired_effects.csv']}))
    print('POST_ANALYSIS_VERIFIED',flush=True)

if __name__=='__main__':analysis()
