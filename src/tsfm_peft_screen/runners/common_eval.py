import gc,json
import numpy as np
import torch
from ..reproducibility import write_json,sha,guard
from ..metrics import score,replay
from ..lora import restore
from ..candidates.censor import cdf

def demand_metrics(p,y,panel):
    sorted_p=np.sort(p,axis=2);med=sorted_p[:,:,10]
    rmsse=float(np.mean(np.sqrt(np.mean((med-y)**2,axis=(0,2))/panel.rms_scale)))
    # Threshold 0.5 defines positive integer demand; quantile-CDF linear tails.
    flat=torch.tensor(sorted_p.reshape(-1,21,48));threshold=torch.ones(flat.shape[0],48)*.5
    probability=(1-cdf(flat,threshold)).numpy().reshape(y.shape)
    positive=y>0
    return dict(rmsse=rmsse,occurrence_brier=float(np.mean((probability-positive)**2)),positive_demand_mae=float(abs(med-y)[positive].mean()))

def evaluate(candidate,panel,contract,winners,out,cache):
    from .common_fit import build,predictions,eval_variants,csv_write,ARMS
    panel.open_e(out/'selection.json',contract)
    write_json(out/'evaluation_open.json',dict(selection_sha256=sha(out/'selection.json'),evaluation_file_sha256=sha(panel.root/'evaluation.npz'),policy='single sealed evaluation pass; all arms fixed before opening'))
    rows=[];pred={};replay_errors=[]
    for arm,selection in [('F0',None)]+[(w['arm'],w) for w in winners]:
        m,adapter=build(candidate,arm)
        if selection:
            assert sha(selection['checkpoint'])==selection['checkpoint_sha256']
            restore(torch.load(selection['checkpoint'],weights_only=True),m,adapter)
        for variant in eval_variants(candidate,'evaluation'):
            p,y=predictions(m,adapter,panel,candidate,'evaluation',variant);pred[(arm,str(variant))]=p
            path=cache/f'E_{arm}_{variant}.npz';np.savez_compressed(path,prediction=p,target=y,scale=panel.scale,origins=panel.origins['evaluation'])
            replay_errors.append(replay(path));metrics=score(p,y,panel.scale)
            if candidate in (2,7):metrics.update(demand_metrics(p,y,panel))
            if candidate==7:
                mask=y>panel.caps[None,:,None]
                metrics.update(censored_pinball=score(p,y,panel.scale,mask)['scaled_2pinball'],uncensored_pinball=score(p,y,panel.scale,~mask)['scaled_2pinball'],censored_underprediction_bias=float(np.mean(np.maximum(y-np.sort(p,axis=2)[:,:,10],0)[mask])))
            rows.append(dict(arm=arm,variant=str(variant),**metrics,prediction_sha256=sha(path)));print('E',candidate,arm,variant,metrics['scaled_2pinball'],flush=True)
        del m,adapter;gc.collect();torch.cuda.empty_cache();guard()
    csv_write(out/'metrics.csv',rows)
    variants=[str(v) for v in eval_variants(candidate,'evaluation') if not (candidate==1 and v=='clean') and not (candidate==3 and v==0)]
    def primary(arm,metric='scaled_2pinball',vs=variants):return float(np.mean([row[metric] for row in rows if row['arm']==arm and row['variant'] in vs]))
    arms=ARMS[candidate];proposed=arms[-1];baselines=arms[:-1];best=min(baselines,key=primary);f0=primary('F0');pval=primary(proposed)
    gains={b:100*(primary(b)-pval)/f0 for b in baselines+['F0']};diagnostics={};passed=False
    if candidate==1:
        cleanbest=min(primary(b,vs=['clean']) for b in baselines)
        clean_degrade=100*(primary(proposed,vs=['clean'])-cleanbest)/primary('F0',vs=['clean'])
        worsening={v:100*(primary(proposed,vs=[v])-min(primary(b,vs=[v]) for b in baselines))/primary('F0',vs=[v]) for v in variants}
        diagnostics=dict(clean_degradation_percent=clean_degrade,corruption_worsening_percent=worsening)
        passed=gains['FEATURE_LORA']>=1 and gains['STANDARD_LORA']>0 and clean_degrade<=.5 and max(worsening.values())<=3
    if candidate==2:
        a=pred[(proposed,'clean')];b=pred[('EVENT_SUMMARY_LORA','clean')];base=pred[('F0','clean')]
        effects=[]
        for c in range(256):
            args=(y[:,c:c+1],panel.scale[c:c+1]);effects.append(score(b[:,c:c+1],*args)['scaled_2pinball']-score(a[:,c:c+1],*args)['scaled_2pinball'])
        effects=np.array(effects);heavy=panel.zero_fraction>=np.quantile(panel.zero_fraction,.75)
        trimmed=float(np.mean(np.sort(effects)[:-26]));heavy_gain=float(effects[heavy].mean())
        diagnostics=dict(improved_series_fraction=float((effects>0).mean()),median_series_effect=float(np.median(effects)),gain_excluding_best10percent=trimmed,zero_heavy_effect=heavy_gain)
        csv_write(out/'series_effects.csv',[dict(channel=str(panel.channels[i]),loss_difference=float(d),zero_fraction=float(panel.zero_fraction[i])) for i,d in enumerate(effects)])
        passed=gains['STANDARD_LORA']>=1 and gains['EVENT_SUMMARY_LORA']>=1 and trimmed>0 and heavy_gain>0 and np.median(effects)>0
    if candidate==3:
        def variance(arm):return float(np.mean(np.var(np.stack([pred[(arm,v)] for v in ['0','4','12']])/panel.scale[None,None,:,None,None],axis=0)))
        vb=variance('PHASE_AUGMENTED_ADAPTER');vp=variance(proposed);reduction=100*(vb-vp)/max(vb,1e-15)
        degradation=100*(primary(proposed,vs=['0'])-min(primary(b,vs=['0']) for b in baselines))/primary('F0',vs=['0'])
        diagnostics=dict(baseline_phase_variance=vb,proposed_phase_variance=vp,variance_reduction_percent=reduction,canonical_degradation_percent=degradation)
        passed=gains['PHASE_AUGMENTED_ADAPTER']>=.5 and reduction>=30 and degradation<=.5
    if candidate==4:
        def revision(arm,correction):
            p=pred[(arm,'clean')].astype(float)
            if correction:p=p-pred[('F0','clean')]
            d=p[:-1,:,:,24:]-p[1:,:,:,:24];d=d/panel.scale[None,:,None,None]
            return float(np.mean(np.where(abs(d)<1,.5*d*d,abs(d)-.5)))
        diagnostics={a:dict(raw_revision=revision(a,False),correction_revision=revision(a,True)) for a in ['F0']+arms}
        passed=gains['STANDARD_LORA']>=1 and gains['RAW_STABILITY_LORA']>=1 and gains['F0_ANCHOR_LORA']>0
    if candidate==7:
        cp=primary(proposed,'censored_pinball');cb=primary('CENSORED_LOSS_LORA','censored_pinball');cf=primary('F0','censored_pinball')
        biasp=primary(proposed,'censored_underprediction_bias');biasb=primary('CENSORED_LOSS_LORA','censored_underprediction_bias')
        unc=100*(primary(proposed,'uncensored_pinball')-primary('CENSORED_LOSS_LORA','uncensored_pinball'))/primary('F0','uncensored_pinball')
        diagnostics=dict(censored_pinball_gain_percent=100*(cb-cp)/cf,censored_bias_proposed=biasp,censored_bias_baseline=biasb,uncensored_degradation_percent=unc,censored_fraction=float((y>panel.caps[None,:,None]).mean()))
        passed=gains['CENSORED_LOSS_LORA']>=1 and gains['NAIVE_LORA']>0 and cp<cb and biasp<biasb and unc<=.5
    verdict='PASS' if passed else 'WEAK' if gains[best]>0 and all(v>0 for b,v in gains.items() if b!='F0') else 'FAIL'
    status=dict(verdict=verdict,fit_count=len(arms)*2,stream_count=0,problem_gate='PASS',novelty_collision=False,strongest_baseline=best,proposed=proposed,proposed_primary=pval,f0_primary=f0,baseline_primary=primary(best),gain_percent_f0=gains[best],gains_percent_f0=gains,diagnostics=diagnostics,round2_recommended=passed)
    write_json(out/'status.json',status)
    integ=json.loads((out/'integrity.json').read_text());integ.update(metric_replay_max_abs=max(replay_errors),selection_seal_before_e=True,no_future_context=True,saved_predictions=len(rows));write_json(out/'integrity.json',integ)
