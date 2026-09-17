"""Reporting-only analysis after verification. No training or model selection."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/persistence_followup_mpl')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .common import *
from .prepare import SHAPES,historical
from .train import choice

def analyze():
    setup();assert read(OUT/'VERIFICATION.json')['status']=='VERIFIED';check_seal();historical()
    effects=pd.read_csv(OUT/'UNCERTAINTY.csv');scores=pd.read_csv(OUT/'PANEL_SCORES.csv');raw=pd.read_csv(OUT/'RAW_SCORES.csv');seed=pd.read_csv(OUT/'SEED_EFFECTS.csv')
    # Independent direct-gain replay from per-condition raw scores, not bootstrap internals.
    checked_effects=0
    for row in effects[effects.condition!='HISTORY_SUBSET'].to_dict('records'):
        f=raw[(raw.panel==row['panel'])&(raw.kind==row['kind'])&(raw.stage=='selected')]
        if row['seed_scope']=='original2':f=f[f.seed.isin([81551,81552])]
        f=f[f.condition.str.startswith(('POINT','BURST'))] if row['condition']=='FAULT' else f[f.condition==row['condition']]
        a=float(f[f.arm==row['new']].nmae.mean());b=float(f[f.arm==row['baseline']].nmae.mean())
        np.testing.assert_allclose([row['new_nmae'],row['baseline_nmae'],row['gain_pct']],[a,b,100*(1-a/b)],rtol=1e-10,atol=1e-10);checked_effects+=1
    panels=['FAULT','REFERENCE','SHIFT4','SHIFT8','SHIFT_POINT'];sources=['electricity','electricity_transfer','ettm1','ettm2']
    # Both old two and all three optimizer seeds, with no winner-based selection.
    summaries=[]
    for source in sources:
        for scope in (['original2','all3'] if source!='ettm2' else ['all3']):
            s=scores[(scores.panel==source)&(scores.kind=='standard')&(scores.stage=='selected')]
            if scope=='original2':s=s[s.seed.isin([81551,81552])]
            f=s.groupby(['arm','metric_panel']).nmae.mean().unstack().reset_index();f.insert(0,'seed_scope',scope);f.insert(0,'panel',source);summaries.append(f)
    summary=pd.concat(summaries,ignore_index=True);summary.to_csv(OUT/'SUMMARY_TABLE.csv',index=False)
    # Preserve all fixed1024 scores; no replacing primary selected scores.
    fixed=scores[scores.kind=='standard'].groupby(['panel','stage','arm','metric_panel']).nmae.mean().reset_index();fixed.to_csv(OUT/'FIXED1024_COMPARISON.csv',index=False)
    fit=pd.read_csv(OUT/'FIT_LEDGER.csv');fit['train_peak_allocated_mib']=fit.peak_allocated/2**20;fit['train_peak_reserved_mib']=fit.peak_reserved/2**20
    fit['trainable_parameters']=fit.arm.map(lambda x:294912 if x in ['B0','C1'] else 8712)
    fit['retained_adaptation_parameters']=fit.arm.map(lambda x:294912 if x in ['B0','C1'] else 303624)
    costs=[]
    for r in fit.to_dict('records'):costs.append(dict(kind='training',**r))
    inf=read(OUT/'INFERENCE_RESOURCES.json')
    for key,r in inf.items():
        if '__standard__selected__' not in key:continue
        panel,kind,stage,arm,seedkey=key.split('__');costs.append(dict(kind='inference_profile',prediction=key,panel=panel,arm=arm,seed=int(seedkey),raw_seconds=json.dumps(r.get('raw_seconds')),median_seconds=r.get('median_seconds'),range_seconds=r.get('range_seconds'),profile_series=r.get('profile_series'),inference_peak_allocated_mib=r.get('peak_allocated',float('nan'))/2**20,forward_models=r.get('forward_models'),full_prediction_seconds=r.get('full_prediction_seconds'),notes=r.get('meaning','fresh common-size profile excludes model loading and gate instrumentation; full_prediction_seconds may include residual diagnostic hooks')))
    pd.DataFrame(costs).to_csv(OUT/'RESOURCE_REPORT.csv',index=False)
    # Separate historical B0 and previous additive comparison costs from the new 58-fit ledger.
    oldb=[read(p) for p in (V2/'fits').glob('*/receipt.json') if read(p)['arm']=='B0'];olda=[read(p) for p in (OLD/'fits').glob('*/receipt.json')]
    guards=[json.loads(l) for p in OUT.glob('gpu_*.jsonl') for l in open(p)]
    wall=read(OUT/'run-all_wall.json')['seconds']+read(OUT/'pre_E_statistics_repair/run-all_wall.json')['seconds']
    save(OUT/'COST_ACCOUNTING.json',dict(new_fits=len(fit),new_B0_fits=int((fit.arm=='B0').sum()),new_main_updates=int(fit.updates.sum()),new_smoke_updates=36,new_training_compute_seconds=float(fit.optimizer_seconds.sum()),new_validation_seconds=float(fit.validation_seconds.sum()),new_checkpoint_io_seconds=float(fit.checkpoint_io_seconds.sum()),new_run_all_wall_seconds=wall,old_shared_B0_fits=len(oldb),old_shared_B0_updates=sum(r['updates'] for r in oldb),old_shared_B0_compute_seconds=sum(r['optimizer_seconds'] for r in oldb),old_additive_comparison_fits=len(olda),old_additive_updates=sum(r['updates'] for r in olda),old_additive_compute_seconds=sum(r['optimizer_seconds'] for r in olda),minimum_free_gpu_mib=min(r['free_mib'] for r in guards),unapproved_compute_samples=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in r['apps']) for r in guards),new_cache_bytes=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file() and not p.is_symlink()),training_parameters_not_memory_saving_ratio=True))
    # LR and output-control selection are independently reconstructed from V-only receipts.
    for source,d in read(OUT/'LR_SELECTION.json').items():
        for arm,r in d.items():assert choice(source,arm,85550 if source=='ettm2' else 81550)==r
    calibration=read(OUT/'CALIBRATION_SELECTION.json')
    for key,r in calibration.items():
        source,seedkey=key.rsplit('_',1);p0=np.load(CACHE/'V_predictions'/f'{source}_C0_{seedkey}.npy');p2=np.load(CACHE/'V_predictions'/f'{source}_C2_{seedkey}.npy');y=np.load(CACHE/'conditions'/source/'V_SELECT_y.npy');s=np.tile(read(OUT/'DATA_MANIFEST.json')[source]['sigma_train_population'],len(y)//4);n=len(y)//20
        from .evaluate import VSTATES
        from .prepare import STATES
        ids=np.concatenate([np.arange(STATES.index(state)*2*n,(STATES.index(state)+1)*2*n) for state in VSTATES]);obj=[]
        for alpha in [0.,.25,.5,.75,1.]:
            p=p0 if alpha==0 else p2 if alpha==1 else p0+alpha*(p2-p0);obj.append((float(np.mean(abs(p[ids,4]-y[ids])/s[ids,None])),alpha))
        assert min(obj)[1]==r['alpha'];np.testing.assert_allclose(np.median(((y[ids]-p0[ids,4])/s[ids,None]).reshape(-1)),r['beta'],rtol=1e-10,atol=1e-12)
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    def finish(name):
        plt.tight_layout()
        for ext in ['png','pdf']:plt.savefig(figs/f'{name}.{ext}',dpi=170)
        plt.close()
    # (i) Original B0 information path and extra correction; known identity-start adapter principle.
    fig,ax=plt.subplots(figsize=(12,4));ax.set_xlim(0,12);ax.set_ylim(0,4);ax.axis('off')
    boxes=[(.2,1.7,2,1,'Observed past x\nTRAIN sigma'),(3,1.7,2,1,'Frozen B0\npatch embedding e'),(6.2,2.4,2.3,1,'Zero-init residual\ndelta(e), 8,712 params'),(3,.2,2.5,1,'Fixed persistence\ng(x), observed-only'),(9.2,1.7,2.5,1,'e + g * delta\nfrozen encoder/head')]
    for x,y,w,h,t in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',fc='#edf3fa',ec='#335b80'));ax.text(x+w/2,y+h/2,t,ha='center',va='center')
    for a,b in [((2.2,2.2),(3,2.2)),((5,2.2),(9.2,2.2)),((5,2.7),(6.2,2.9)),((8.5,2.9),(9.2,2.6)),((1.2,1.7),(3,.8)),((5.5,.7),(10,1.7))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->'})
    ax.text(6,.0,'No clean input, fault mask, scenario ID, synthetic delta or future labels enter g/model.',ha='center',fontsize=9);finish('01_dataflow')
    # (ii) All three seed effects on the preregistered transfer question.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,base in zip(axes,['C0','C2']):
        f=seed[(seed.panel=='electricity_transfer')&(seed.kind=='standard')&(seed.condition=='SHIFT8')&(seed.new=='C3')&(seed.baseline==base)].sort_values('seed')
        ax.bar(f.seed.astype(str),f.gain_pct);ax.axhline(0,color='grey');ax.set_title('Series transfer: C3 vs '+base);ax.set_ylabel('nMAE gain % (positive better)')
    finish('02_transfer_seeds')
    # (iii) Every target/source and all gains and harms, no outcome-based filtering.
    fig,axes=plt.subplots(2,4,figsize=(18,8))
    for col,source in enumerate(sources):
        for row,base in enumerate(['C0','C2']):
            f=effects[(effects.panel==source)&(effects.kind=='standard')&(effects.seed_scope=='all3')&(effects.ci_type=='time')&(effects.new=='C3')&(effects.baseline==base)].set_index('condition').loc[panels];ax=axes[row,col];ax.hlines(range(5),f.ci_low_pct,f.ci_high_pct);ax.plot(f.gain_pct,range(5),'o');ax.axvline(0,color='grey');ax.set_yticks(range(5),panels);ax.set_title(source+' / C3 vs '+base);ax.set_xlabel('gain %, conditional time-block 95% CI')
    finish('03_tradeoffs')
    # (iv) Same gate values can have different locations and realized residual norms.
    stats=read(OUT/'GATE_RESIDUAL_STATS.json');fig,axes=plt.subplots(2,3,figsize=(15,7))
    for col,source in enumerate(['electricity','electricity_transfer','ettm1']):
        for arm in ['C2','C3']+CONTROLS:
            vals=[v['SHIFT8'] for k,v in stats.items() if k.startswith(source+'__standard__selected__'+arm+'__')]
            assert len(vals)==3,(source,arm,len(vals))
            axes[0,col].plot(np.mean([v['gate_by_patch'] for v in vals],0),label=arm);axes[1,col].plot(np.mean([v['applied_rms_by_patch'] for v in vals],0),label=arm)
        axes[0,col].set_title(source+' / SHIFT8');axes[0,col].set_ylabel('mean gate');axes[1,col].set_ylabel('actual added residual RMS');axes[1,col].set_xlabel('patch position: old -> recent');axes[0,col].legend(fontsize=7)
    finish('04_gate_and_residual')
    # (v) Include pulse counterexamples and all fixed lengths/amplitudes.
    names=SHAPES+['PAIRED_SHIFT8_D32'];fig,axes=plt.subplots(2,2,figsize=(13,11))
    for ax,source in zip(axes.flat,sources):
        for base,shift in [('C0',-.12),('C2',.12)]:
            f=effects[(effects.panel==source)&(effects.kind=='shape')&(effects.seed_scope=='all3')&(effects.ci_type=='time')&(effects.new=='C3')&(effects.baseline==base)].set_index('condition').loc[names];ys=np.arange(len(names))+shift;ax.hlines(ys,f.ci_low_pct,f.ci_high_pct);ax.plot(f.gain_pct,ys,'o',label='vs '+base)
        ax.axvline(0,color='grey');ax.set_yticks(range(len(names)),names);ax.set_title(source);ax.legend();ax.set_xlabel('gain %, conditional time-block 95% CI')
    finish('05_shape_stress')
    required=['RAW_SCORES.csv','PRIMARY_CONTRASTS.csv','MECHANISM_CONTRASTS.csv','SEED_EFFECTS.csv','UNCERTAINTY.csv','RESOURCE_REPORT.csv','VERIFICATION.json']
    save(OUT/'PUBLICATION_AUDIT.json',dict(status='VERIFIED',all_new_counts_correct=True,direct_gain_rows_independently_replayed=checked_effects,LR_selection_recomputed=True,alpha_beta_V_selection_recomputed=True,old_results_preserved=True,execution_seal_amended_before_E_scoring=True,model_train_selection_data_hashes_unchanged=True,required_artifact_hashes={n:sha(OUT/n) for n in required},reporting_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [EXP/'analysis.py',EXP/'report.py'] if p.exists()}))
    print('REPORT_ANALYSIS_VERIFIED',flush=True)

if __name__=='__main__':analyze()
