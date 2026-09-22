"""CPU analysis of sealed predictions; never trains, selects, or changes a model."""
import sys,csv,math,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import *
from evaluation import metric,apply_cal,calibrate,score,path,loadpred

def write_csv(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def read_csv(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))

def gain(candidate,baseline):return 100*(1-float(candidate)/float(baseline))

def oracle(z,p,y,sigma,values):
    errors=[]
    for i,h in [(0,0),(len(z)//2,z.shape[1]//2),(-1,-1)]:
        zz=z[i,h].astype(float);pp=np.broadcast_to(p,z.shape)[i,h].astype(float);pp/=pp.sum()
        direct=pp@np.abs(zz-y[i,h])-.5*np.sum(pp[:,None]*pp[None,:]*np.abs(zz[:,None]-zz[None,:]))
        errors.append(abs(direct/sigma[i]-values['crps'][i,h]))
        ix=np.argsort(zz,kind='stable');cdf=np.cumsum(pp[ix]);levels=np.arange(1,10)/10
        q=np.array([zz[ix[min(np.searchsorted(cdf,l,side='left'),len(zz)-1)]] for l in levels])
        e=y[i,h]-q;pin=np.mean(2*np.maximum(levels*e,(levels-1)*e))/sigma[i]
        assert np.isclose(pin,values['pinball'][i,h],atol=1e-10)
        assert np.isclose(abs(y[i,h]-q[4])/sigma[i],values['nmae'][i,h],atol=1e-10)
    assert max(errors)<1e-9
    return max(errors)

def budget_audit(c):
    r=RESULTS/c;state=read(r/'BUDGET_STATE.json');assert state['main_fits']==12 and state['main_updates']==3072 and state['smoke_updates']==12
    events=[json.loads(x) for x in (r/'UPDATE_LEDGER.jsonl').read_text().splitlines()]
    intents=[(x['key'],x['step']) for x in events if x['kind']=='update_intent'];commits=[(x['key'],x['step']) for x in events if x['kind']=='update_commit']
    assert intents==commits and len(set(intents))==len(intents)==3084
    fits=[];limits=[];selection=read(r/'MODEL_SELECTION.json')
    for key,entry in state['fits'].items():
        assert entry['intent']==entry['commit']==entry['planned'] and entry['status']=='COMPLETE'
        report=read(r/'fits'/f'{key}.json');assert report['frozen_unchanged'] and report['initial_sha']!=report['final_sha']
        row=dict(key=key,phase=entry['phase'],updates=entry['commit'],fit_seconds=report['fit_seconds'],wall_seconds=report['wall_seconds'],trainable_parameters=report['trainable_parameters'],peak_allocated=report['peak_allocated'],peak_reserved=report['peak_reserved'])
        if entry['phase']=='main':
            arm,seed=key.rsplit('_',1);s=selection[seed][arm];expected=min([0,128,256],key=lambda k:(report['curves'][str(k)],k));assert s['checkpoint']==expected
            row['selected_checkpoint']=expected
            for step in [0,128,256]:
                val=score(loadpred(path(c,int(seed),arm,'VAL',step)),c)
                assert np.isclose(val,report['curves'][str(step)],atol=1e-12)
                row[f'raw_val_{step}']=val
        fits.append(row)
    for arm in (ARMS_A if c=='A' else ARMS_B):
        means=[np.mean([selection[str(seed)][arm]['raw_validation'][str(k)] for seed in SEEDS]) for k in [0,128,256]]
        if means[0]>means[1]>means[2]:limits.append(arm)
    write_csv(r/'FIT_LEDGER.csv',fits)
    return dict(main_fits=12,main_updates=3072,smoke_updates=12,unique_intent_commit_pairs=3084,frozen_all_unchanged=True,optimization_limit_arms=limits)

def evaluate(c):
    r=RESULTS/c;arms=ARMS_A if c=='A' else ARMS_B;refs=['F0_NATIVE','F0_MEDIAN','CHRONOS2'] if c=='A' else ['F0','NAIVE','GBQR']
    cal=read(r/'CALIBRATION.json');baseline=read(r/'BASELINE_SELECTION.json')['baseline'];candidate='CONTEXT3' if c=='A' else 'SCENARIO3'
    allrows=[];leads=[];originrows=[];grouprows=[];scores={};origin_scores={};errors=[];cal_count=0;cal_seconds=0
    if c=='B':
        from bdata_adapter import Wind
        issues=Wind().raw['issue'].astype(str)
    for arm in arms+refs:
        for seed in (SEEDS if arm in arms else [0]):
            pred=loadpred(path(c,seed,arm,'TEST'));coeff=cal[str(seed)][arm]
            tick=time.perf_counter();recal=calibrate(loadpred(path(c,seed,arm,'CAL')),c);cal_seconds+=time.perf_counter()-tick
            assert [(x['alpha'],x['beta']) for x in recal]==[(x['alpha'],x['beta']) for x in coeff]
            cal_count+=sum(x['grid_evaluations'] for x in coeff)
            for mode in ['raw','cal']:
                z=pred['z'] if mode=='raw' else apply_cal(pred,coeff)
                nq=pred.get('native_q')
                if nq is not None and mode=='cal':
                    from evaluation import quantile
                    nq=nq.astype(float).copy();med=quantile(pred['z'],pred['p'])
                    for item in coeff:
                        lo,hi=item['span'];nq[:,lo:hi]=med[:,lo:hi,None]+item['beta']*pred['sigma'][:,None,None]+item['alpha']*(nq[:,lo:hi]-med[:,lo:hi,None])
                values=metric(z,pred['p'],pred['y'],pred['sigma'],nq);errors.append(oracle(z,pred['p'],pred['y'],pred['sigma'],values))
                regions={'first64':slice(0,64),'tail64':slice(64,128),'full128':slice(0,128)} if c=='A' else {'all8':slice(0,8)}
                for region,span in regions.items():
                    row=dict(candidate=c,arm=arm,seed=seed,mode=mode,region=region,examples=len(z),**{k:float(v[:,span].mean()) for k,v in values.items()});allrows.append(row)
                    if region==('tail64' if c=='A' else 'all8'):scores[arm,seed,mode]=row
                for lead in range(z.shape[1]):leads.append(dict(arm=arm,seed=seed,mode=mode,lead=lead+1 if c=='A' else (lead+1)*3,**{k:float(v[:,lead].mean()) for k,v in values.items()}))
                primary=slice(64,128) if c=='A' else slice(0,8);perrow={k:v[:,primary].mean(1) for k,v in values.items()}
                for i,pair in enumerate(pred['pairs']):originrows.append(dict(arm=arm,seed=seed,mode=mode,series=int(pair[0]),origin=int(pair[1]),**{k:float(v[i]) for k,v in perrow.items()}))
                unique=np.unique(pred['pairs'][:,1]);origin_scores[arm,seed,mode]=np.array([perrow['crps'][pred['pairs'][:,1]==o].mean() for o in unique])
                groups=pred['pairs'][:,0].astype(str) if c=='A' else np.array([issues[int(i)][:7] for i in pred['pairs'][:,1]])
                for group in np.unique(groups):grouprows.append(dict(arm=arm,seed=seed,mode=mode,group=group,**{k:float(v[groups==group].mean()) for k,v in perrow.items()}))
    write_csv(r/'SCORES.csv',allrows);write_csv(r/'LEAD_SCORES.csv',leads);write_csv(r/'ORIGIN_SCORES.csv',originrows);write_csv(r/('SERIES_SCORES.csv' if c=='A' else 'MONTH_SCORES.csv'),grouprows)
    def get(arm,seed,mode='cal',metric_name='crps'):return scores[arm,seed if arm in arms else 0,mode][metric_name]
    def avg(arm,mode='cal',metric_name='crps'):return float(np.mean([get(arm,s,mode,metric_name) for s in SEEDS]))
    def effect(a,b,mode='cal',metric_name='crps'):return gain(avg(a,mode,metric_name),avg(b,mode,metric_name))
    pairs=[(candidate,b) for b in arms+refs if b!=candidate]
    if c=='A':pairs += [('FULL9','F0_NATIVE'),('FULL9','F0_MEDIAN')]+[(a,'FULL9') for a in arms[1:-1]]
    else:pairs += [('TARGET','F0')]+[(a,'TARGET') for a in ['CONTROL','MOMENTS','SET','MEMBER','GBQR']]+[(a,'CONTROL') for a in ['MOMENTS','SET','MEMBER','GBQR']]
    effects=[];means=[];bootstrap=[]
    for a,b in pairs:
        for mode in ['raw','cal']:
            for seed in SEEDS:
                for metric_name in ['crps','pinball','nmae']:
                    av=get(a,seed,mode,metric_name);bv=get(b,seed,mode,metric_name)
                    effects.append(dict(candidate=a,baseline=b,seed=seed,mode=mode,metric=metric_name,candidate_score=av,baseline_score=bv,gain_pct=gain(av,bv)))
            for metric_name in ['crps','pinball','nmae']:
                means.append(dict(candidate=a,baseline=b,mode=mode,metric=metric_name,candidate_score=avg(a,mode,metric_name),baseline_score=avg(b,mode,metric_name),gain_pct=effect(a,b,mode,metric_name)))
            av=np.mean([origin_scores[a,s if a in arms else 0,mode] for s in SEEDS],axis=0);bv=np.mean([origin_scores[b,s if b in arms else 0,mode] for s in SEEDS],axis=0)
            n=len(av);rng=np.random.default_rng(92300);starts=rng.integers(0,n-14+1,size=(2000,math.ceil(n/14)));indices=(starts[:,:,None]+np.arange(14)).reshape(2000,-1)[:,:n]
            draws=100*(1-av[indices].mean(1)/bv[indices].mean(1));lo,hi=np.quantile(draws,[.025,.975])
            bootstrap.append(dict(candidate=a,baseline=b,mode=mode,gain_pct=effect(a,b,mode),lower95=float(lo),upper95=float(hi),replicates=2000,block_origins=14,origins=n,seed_resampling=False,series_resampling=False))
    write_csv(r/'SEED_EFFECTS.csv',effects);write_csv(r/'MEAN_EFFECTS.csv',means);write_csv(r/'BOOTSTRAP.csv',bootstrap)
    resource=read_csv(r/'RESOURCES_FAIR.csv');lat={x['arm']:float(x['median_seconds']) for x in resource if x['batch']=='8'}
    g=effect(candidate,baseline);seedg=[gain(get(candidate,s),get(baseline,s)) for s in SEEDS];point=effect(candidate,baseline,metric_name='nmae')
    decision=dict(candidate=c,proposal=candidate,baseline=baseline,mean_gain_pct=g,seed_gain_pct=seedg,point_gain_pct=point,latency_batch8_ms={a:v*1000 for a,v in lat.items()},novelty_claim=False,automatic_followup=False)
    if c=='A':
        saving=100*(1-lat[candidate]/lat['FULL9']);teacher_weak=avg('FULL9')>min(avg('F0_NATIVE'),avg('F0_MEDIAN'))
        checks=dict(gain_at_least_1pct=g>=1,both_seeds_positive=min(seedg)>0,full9_crps_loss_at_most_1pct=effect(candidate,'FULL9')>=-1,full9_pinball_loss_at_most_1pct=effect(candidate,'FULL9',metric_name='pinball')>=-1,latency_saving_at_least_15pct=saving>=15,point_loss_at_most_1pct=point>=-1,teacher_not_weaker_than_f0=not teacher_weak)
        compression=[a for a in arms[1:-1] if effect(a,'FULL9')>=-1 and effect(a,'FULL9',metric_name='pinball')>=-1 and 100*(1-lat[a]/lat['FULL9'])>=15]
        status='GO_A_METHOD_SCREEN' if all(checks.values()) else ('GO_A_COMPRESSION_ONLY' if compression and g<1 else ('NO_GO_A_CURRENT' if g<=0 and lat[candidate]>=lat[baseline] else 'HOLD_NO_AUTO_RESCUE'))
        decision.update(checks=checks,simple_compression_arms=compression,full9_gain_pct=effect(candidate,'FULL9'),full9_pinball_gain_pct=effect(candidate,'FULL9',metric_name='pinball'),latency_saving_vs_full9_pct=saving,teacher_weak_veto=teacher_weak)
    else:
        increase=100*(lat[candidate]/lat['SET']-1)
        checks=dict(gain_at_least_1pct=g>=1,both_seeds_positive=min(seedg)>0,point_loss_at_most_1pct=point>=-1,latency_increase_vs_set_at_most_25pct=increase<=25)
        information=[a for a in ['CONTROL','MOMENTS','SET','MEMBER','GBQR'] if effect(a,'TARGET')>=1 and all(get(a,s)<get('TARGET',s) for s in SEEDS)]
        accuracy=checks['gain_at_least_1pct'] and checks['both_seeds_positive'] and checks['point_loss_at_most_1pct']
        status='GO_B_METHOD_SCREEN' if all(checks.values()) else ('TRADEOFF_HOLD' if accuracy else ('GO_B_INFORMATION_ONLY' if information else ('NO_GO_B_CURRENT' if g<=0 and lat[candidate]>=lat[baseline] else 'HOLD_NO_AUTO_RESCUE')))
        decision.update(checks=checks,information_useful_arms=information,latency_increase_vs_set_pct=increase,release_time='RELEASE_TIME_UNVERIFIED',operational_go=False)
    decision['status']=status
    save(r/'FINAL_DECISION.json',decision)
    verification=budget_audit(c);verification.update(status='PASS',max_pairwise_crps_absolute_error=max(errors),cal_grid_evaluations_recomputed=cal_count,cal_verification_seconds=cal_seconds,original_cal_separate_wall_time_available=False,cal_coefficients_reproduced=True,checkpoint_selection_reproduced=True,bootstrap='2000 paired non-circular moving blocks, 14 origins; seeds and series kept jointly; fixed RNG92300; two training seeds only')
    baseline_values=read(r/'BASELINE_SELECTION.json')['validation_scores']
    for arm,val in baseline_values.items():
        vv=np.mean([score(loadpred(path(c,s if arm in arms else 0,arm,'VAL')),c,cal[str(s if arm in arms else 0)][arm]) for s in SEEDS]);assert np.isclose(val,vv,atol=1e-12)
    assert baseline==min(baseline_values,key=baseline_values.get);verification['baseline_selection_reproduced']=True
    save(r/'VERIFICATION.json',verification)
    if c=='A':
        zero=[]
        for seed in SEEDS:
            for arm in arms:
                p=loadpred(path(c,seed,arm,'TEST',0));v=metric(p['z'],p['p'],p['y'],p['sigma'])
                zero.append(dict(arm=arm,seed=seed,checkpoint=0,mode='raw',region='tail64',**{k:float(x[:,64:].mean()) for k,x in v.items()}))
        write_csv(r/'CHECKPOINT_ZERO_DIAGNOSTICS.csv',zero)
    return decision

def main():
    assert read(RESULTS/'RUN_COMPLETE.json')['status']=='COMPLETE'
    assert read(RESULTS/'ALL_DIAGNOSTIC_PREDICTIONS_COMPLETE.json')['status']=='COMPLETE'
    assert read(RESULTS/'FAIR_RUNTIME_COMPLETE.json')['status']=='COMPLETE'
    from runner import check_selection
    check_selection()
    for c in read(RESULTS/'EXECUTION_PLAN.json')['completed_candidates']:
        for name,record in read(RESULTS/c/'PREDICTIONS_MANIFEST.json')['files'].items():assert sha(ROOT/name)==record['sha256']
    if not (RESULTS/'ANALYSIS_STARTED.json').exists():save(RESULTS/'ANALYSIS_STARTED.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),all_predictions_precede_scoring=True,script_sha=sha(__file__)))
    decisions=[evaluate(c) for c in read(RESULTS/'EXECUTION_PLAN.json')['completed_candidates']]
    save(RESULTS/'ANALYSIS_COMPLETE.json',dict(decisions=decisions,additional_optimizer_updates=0))
    print(json.dumps(decisions,indent=2))

if __name__=='__main__':main()
