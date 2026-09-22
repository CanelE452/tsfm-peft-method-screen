"""Score only after both tracks have saved all TEST forecasts. No optimizer/GPU."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from data import Data,scale
import numpy as np
import pandas as pd

def csv(p,rows):pd.DataFrame(rows).to_csv(p,index=False)
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def metrics(q,y,s):
    e=y[...,None].astype(float)-q.astype(float);levels=np.arange(1,10)/10;pb=(2*np.maximum(levels*e,(levels-1)*e)).mean(-1)
    return dict(pinball=pb/s,raw_pinball=pb,nmae=np.abs(e[...,4])/s,raw_mae=np.abs(e[...,4]),coverage=((y>=q[...,0])&(y<=q[...,-1])).astype(float),width=(q[...,-1]-q[...,0])/s,crossing=(np.diff(q,axis=-1)<0).mean(-1))
def gain(a,b):return float(100*(1-np.mean(a)/np.mean(b)))
def interval(a,b,track,mode='time'):
    assert a.shape==b.shape and a.shape[0:2]==(2,8)
    rng=np.random.default_rng(92403);out=[];n=a.shape[-1]
    for _ in range(2000):
        if track=='A':
            ix=rng.integers(0,8,8);aa=[];bb=[]
            for s in ix:
                starts=rng.integers(0,n-3+1,(n+2)//3);ti=np.concatenate([np.arange(x,x+3) for x in starts])[:n];aa.append(a[:,s,ti]);bb.append(b[:,s,ti])
            out.append(gain(aa,bb))
        elif mode=='series':
            ix=rng.integers(0,8,8);out.append(gain(a[:,ix,:],b[:,ix,:]))
        else:
            starts=rng.integers(0,n-8+1,(n+7)//8);ti=np.concatenate([np.arange(x,x+8) for x in starts])[:n];out.append(gain(a[:,:,ti],b[:,:,ti]))
    return np.quantile(out,[.025,.975]).tolist()
def loadq(track,role,arm,seed,tag,pairs):
    folder=CACHE/track/'predictions'/role/arm/f'{seed}_{tag}'
    return np.stack([np.load(folder/f'{s}_{t}.npz')['q'] for s,t in pairs])

def main():
    assert read(RESULTS/'ALL_MAIN_COMPLETE.json')['status']=='COMPLETE'
    for name,h in read(RESULTS/'SOURCE_SEAL.json')['files'].items():assert sha(EXP/name)==h,name
    for name,h in read(RESULTS/'SOURCE_SEAL.json')['data'].items():assert sha(RESULTS/name)==h,name
    d=Data();updates=lines(RESULTS/'UPDATE_LEDGER.jsonl');intents=[(x['phase'],x['key'],x['step']) for x in updates if x['kind']=='intent'];commits=[(x['phase'],x['key'],x['step']) for x in updates if x['kind']=='commit']
    assert intents==commits and len(intents)==len(set(intents));counts={p:sum(x[0]==p for x in intents) for p in ['offline','A_local','B_online','smoke']}
    assert counts==dict(offline=8192,A_local=6144,B_online=12288,smoke=14),counts
    ref=lines(RESULTS/'REFERENCE_FORECASTS.jsonl');keys=set();refmanifest=[]
    for x in ref:
        key=(x['seed'],x['series'],x['issue']);assert key not in keys;keys.add(key);p=ROOT/x['file'];assert sha(p)==x['sha256'];assert x['issue']>=d.edges[1]
        with np.load(p) as z:
            context=d.asof(x['series'],x['issue'])[-256:];assert hashlib.sha256(context.tobytes()).hexdigest()==str(z['context_hash'])==x['context_sha'];assert str(z['bank_hash'])==x['bank_sha']
        refmanifest.append({k:x[k] for k in ['seed','series','issue','file','sha256','seconds','bytes']})
    save(RESULTS/'REFERENCE_FORECAST_MANIFEST.json',dict(status='PASS',forecasts=refmanifest,total_calls=len(ref),seconds=sum(x['seconds'] for x in ref),bytes=sum(x['bytes'] for x in ref),local_series_index=True))
    event('test_scoring_started',both_tracks_complete=True)
    overall={}
    for track in ['A','B']:
        out=RESULTS/track;assert read(out/'ALL_TEST_SAVED.json')['status']=='COMPLETE'
        for name,h in read(out/'SELECTION_SEAL.json')['files'].items():assert sha(out/name)==h
        selection=read(out/'MODEL_SELECTION.json');bs=read(out/'BASELINE_SELECTION.json');issued=lines(out/'ISSUED_FORECASTS.jsonl');seen=set();predmanifest=[]
        for x in issued:
            key=(x['role'],x['arm'],x['seed'],x['tag'],x['series'],x['issue']);assert key not in seen;seen.add(key);assert x['target_start']==x['issue']==x['visible_before'];assert sha(ROOT/x['file'])==x['sha256'];predmanifest.append(dict(file=x['file'],sha256=x['sha256']))
        save(out/'PREDICTION_MANIFEST.json',dict(issued=len(issued),files=predmanifest,all_hashes_verified=True,immutable_keys=True))
        raw=[];lead=[];curves=[];cubes={};q_metrics={}
        pairs=d.episodes(track,'TEST');n=len(pairs)//8
        for arm,tags in bs['tags'].items():
            vals=[]
            for seed in SEEDS:
                q=loadq(track,'TEST',arm,seed,tags[str(seed)],pairs);mm=[]
                for (s,t),qq in zip(pairs,q):
                    met=metrics(qq,d.scorer_targets(np.array([[s,t]]),'TEST')[0],scale(d.asof(s,t)[-256:]));mm.append(met)
                    raw.append(dict(arm=arm,seed=seed,series_local=int(s),series_original=d.ids['TEST'][int(s)-40],issue=int(t),tag=tags[str(seed)],**{k:float(v.mean()) for k,v in met.items()},early32=float(met['pinball'][:32].mean()),late32=float(met['pinball'][32:].mean()),early16=float(met['pinball'][:16].mean()),late48=float(met['pinball'][16:].mean())))
                bymetric={k:np.stack([m[k] for m in mm]) for k in mm[0]};q_metrics[arm,seed]=bymetric
                vals.append(bymetric['pinball'].mean(1).reshape(8,n))
                for h in range(64):lead.append(dict(arm=arm,seed=seed,lead=h+1,**{k:float(v[:,h].mean()) for k,v in bymetric.items()}))
            cubes[arm]=np.stack(vals)
        df=pd.DataFrame(raw);csv(out/'RAW_SCORES.csv',raw);csv(out/'LEAD_SCORES.csv',lead)
        aggregate=df.groupby(['arm','seed'],sort=False)[['pinball','raw_pinball','nmae','raw_mae','coverage','width','crossing','early32','late32','early16','late48']].mean().reset_index();aggregate.to_csv(out/'SEED_SCORES.csv',index=False)
        comparisons=[(track+'_TIME',arm) for arm in cubes if arm!=track+'_TIME']+[(a,'STATIC') for a in cubes if a not in ['STATIC',track+'_TIME','G0']]+[('STATIC','G0')]
        effects=[];se=[];series_effects=[]
        for a,b in comparisons:
            ca,cb=cubes[a],cubes[b];lo,hi=interval(ca,cb,track);slo,shi=interval(ca,cb,track,'series') if track=='B' else (lo,hi)
            effects.append(dict(candidate=a,control=b,candidate_score=float(ca.mean()),control_score=float(cb.mean()),improvement_pct=gain(ca,cb),ci_low=lo,ci_high=hi,series_ci_low=slo,series_ci_high=shi))
            for i,seed in enumerate(SEEDS):se.append(dict(candidate=a,control=b,seed=seed,candidate_score=float(ca[i].mean()),control_score=float(cb[i].mean()),improvement_pct=gain(ca[i],cb[i])))
            for j,s in enumerate(d.ids['TEST']):series_effects.append(dict(candidate=a,control=b,series_original=s,improvement_pct=gain(ca[:,j],cb[:,j])))
        csv(out/'EFFECTS.csv',effects);csv(out/'SEED_EFFECTS.csv',se);csv(out/'SERIES_EFFECTS.csv',series_effects)
        if track=='A':
            for role in ['DEV','TEST']:
                ps=d.episodes(track,role)
                for arm in ['A_LOCAL','A_COEFF']:
                    for seed in SEEDS:
                        for k in [0,1,4,8]:
                            q=loadq(track,role,arm,seed,k,ps);scores=[metrics(qq,d.scorer_targets(np.array([[s,t]]),role)[0],scale(d.asof(s,t)[-256:]))['pinball'].mean() for (s,t),qq in zip(ps,q)]
                            curves.append(dict(role=role,arm=arm,seed=seed,updates_per_episode=k,pinball=float(np.mean(scores)),primary=k==bs['kselected'][arm]))
        else:
            for arm in cubes:
                for i,seed in enumerate(SEEDS):
                    for lo in [0,16,32]:curves.append(dict(role='TEST',arm=arm,seed=seed,tick_start=lo,tick_end=lo+15,pinball=float(cubes[arm][i,:,lo:lo+16].mean())))
        csv(out/'ADAPTATION_BUDGET_CURVES.csv',curves)
        res=[];resource_events=lines(out/'RESOURCE_EVENTS.jsonl');bench=read(out/'FORWARD_BENCHMARK.json')['rows'];amort=[]
        for x in resource_events:
            if x['kind']=='local':
                for k in [0,1,4,8]:
                    times=[c['seconds'] for c in x['costs'] if c['k']==k];res.append({k0:v for k0,v in x.items() if k0!='costs'}|dict(budget=k,seconds=sum(times),mean_episode_seconds=float(np.mean(times)),peak_scope='Process high-water since last offline fit reset; not isolated method memory'))
            else:res.append({k:v for k,v in x.items() if k!='coefficient_stats'}|dict(peak_scope='Process high-water since last offline fit reset; not isolated method memory'))
        csv(out/'RESOURCES.csv',res)
        for arm in selection['checkpoints']:
            if arm=='STATIC':continue
            for seed in SEEDS:
                f=read(RESULTS/'fits'/f'{arm}_{seed}.json');g0=read(RESULTS/'fits'/f'G0_{seed}.json');static=read(RESULTS/'fits'/f'STATIC_{seed}.json')
                qr=next(b for b in bench if b['arm']=='G0_SUPPORT4' and b['seed']==seed);qf=next(b for b in bench if b['arm']==arm and b['seed']==seed)
                actual=next(x for x in resource_events if x['kind']=='fixed' and x['role']=='TEST' and x['arm']==arm and x['seed']==seed and x['tag']==selection['checkpoints'][arm][str(seed)])
                feature_query_mean=actual['compute_seconds']/actual['query_count']
                localbase=bs['local_baseline'] if track=='A' else bs['baseline'];cold=feature_query_mean+qr['median'];basecost=None
                if track=='A':
                    ev=next(e for e in resource_events if e['kind']=='local' and e['role']=='TEST' and e['arm']==localbase and e['seed']==seed);k=bs['kselected'][localbase];basecost=float(np.mean([c['seconds'] for c in ev['costs'] if c['k']==k]))
                else:
                    bb=[b for b in bench if b['arm']==localbase and b['seed']==seed]
                    if bb:
                        actualbase=next(x for x in resource_events if x['kind']=='fixed' and x['role']=='TEST' and x['arm']==localbase and x['seed']==seed)
                        basecost=actualbase['compute_seconds']/actualbase['query_count']+(qr['median'] if bb[0].get('needs_G0_support') else 0)
                    else:
                        rr=[e for e in resource_events if e['kind']=='online' and e['role']=='TEST' and e['arm']==localbase and e['seed']==seed];basecost=float(np.mean([e['compute_seconds']/48 for e in rr]))
                # Standalone preparation estimate charges uncached diagnostic forecasts to each generator.
                packets=d.meta_schedule(seed).reshape(-1,2);offs=[256,192,128,64] if track=='A' else [64,32,16,8]
                required={(int(s),int(t-o)) for s,t in packets for o in offs};rtime={ (x['series'],x['issue']):x['seconds'] for x in ref if x['seed']==seed};coldprep=sum(rtime[key] for key in required)
                prep=g0['wall_seconds']+f['wall_seconds']-f['reference_new_seconds']+coldprep
                baseprep=g0['wall_seconds']+static['wall_seconds']
                if localbase=='G0':baseprep=g0['wall_seconds']
                if localbase.endswith('COEFF') or localbase in selection['checkpoints'] and localbase!='STATIC':
                    bankarm=track+'_TIME' if localbase.endswith('COEFF') else localbase
                    cf=read(RESULTS/'fits'/f'{bankarm}_{seed}.json');baseprep=g0['wall_seconds']+cf['wall_seconds']-cf['reference_new_seconds']+coldprep
                extra=max(0,prep-baseprep);saving=basecost-cold;break_even=int(np.ceil(extra/saving)) if saving>0 else None
                amort.append(dict(arm=arm,seed=seed,baseline=localbase,g0_wall_seconds=g0['wall_seconds'],generator_fit_wall_seconds=f['wall_seconds'],uncached_reference_prep_seconds=coldprep,standalone_prep_estimate_seconds=prep,baseline_prep_estimate_seconds=baseprep,additional_prep_seconds=extra,prepared_record_forward_median_seconds=qf['median'],feature_generator_query_mean_seconds=feature_query_mean,support4_cold_seconds=qr['median'],field_cold_seconds=cold,baseline_field_seconds=basecost,field_saving_pct=100*(1-cold/basecost),break_even_episodes=break_even,status='NEVER_RECOVERED' if saving<=0 else 'ESTIMATE_ONLY',timing_limitation='Field estimate includes actual feature construction and query mean plus cold support median. Optimizer trajectories include journal IO. Read-only forward benchmark uses prepared records. Not an isolated efficiency proof.'))
        csv(out/'AMORTIZATION_COSTS.csv',amort)
        prop=track+'_TIME';controls=[bs['baseline'],track+'_SET',track+('_NOERROR' if track=='A' else '_FULLGEN')];controls=list(dict.fromkeys(controls))
        ee=[next(e for e in effects if e['candidate']==prop and e['control']==b) for b in controls]
        ss=[e for e in se if e['candidate']==prop and e['control'] in controls]
        if track=='A':
            simple=min(['STATIC','STATIC_LONG512','A_AFFINE'],key=lambda a:bs['dev_scores'][a])
            direct_controls=['A_SET','A_NOERROR']
            direct_effects=[e for e in effects if e['candidate']==prop and e['control'] in direct_controls]
            simple_effect=next(e for e in effects if e['candidate']==prop and e['control']==simple)
            specific=all(e['improvement_pct']>=.5 for e in direct_effects) and simple_effect['improvement_pct']>0 and all(e['improvement_pct']>0 for e in se if e['candidate']==prop and e['control'] in direct_controls+[simple])
        else:
            simple=None;specific=all(e['improvement_pct']>=.5 for e in ee) and all(e['improvement_pct']>0 for e in ss)
        late={b:gain(np.stack([q_metrics[prop,s]['pinball'][:,16:] for s in SEEDS]),np.stack([q_metrics[b,s]['pinball'][:,16:] for s in SEEDS])) for b in controls}
        end_only=all(k==512 for perseed in selection['checkpoints'].values() for k in perseed.values())
        efficiency=None
        if track=='A':
            target=bs['local_baseline'];quality=all(gain(cubes[prop][i],cubes[target][i])>=-1 for i in range(2));costrows=[x for x in amort if x['arm']==prop];timing=all(x['field_saving_pct']>=20 for x in costrows)
            efficiency=dict(same_quality=quality,operational_measured_saving20=timing,isolated_compute_efficiency_confirmed=False,reason='Journal overhead differs between optimizer and forward-only timings; conservatively do not award GO from this timing alone')
        vsbase=next(e for e in effects if e['candidate']==prop and e['control']==bs['baseline'])
        useful=[e for e in effects if e['control']=='STATIC' and e['candidate'] not in ['STATIC_LONG512','STATIC_LONG320'] and e['improvement_pct']>=.5]
        repeated=any(all(x['improvement_pct']>0 for x in se if x['candidate']==e['candidate'] and x['control']=='STATIC') for e in useful)
        if end_only:label='HOLD_NO_AUTO_TRAIN'
        elif specific and track=='B':label='GO_NEXT_METHOD_CHECK'
        elif specific:label='HOLD_NO_AUTO_TRAIN'
        elif repeated:label='GO_GENERIC_ONLY'
        elif vsbase['improvement_pct']<0 and all(x['improvement_pct']<0 for x in se if x['candidate']==prop and x['control']==bs['baseline']):label='NO_GO_CURRENT_RECIPE'
        else:label='HOLD_NO_AUTO_TRAIN'
        result=dict(label=label,baseline=bs['baseline'],simple_baseline_from_sealed_dev=simple,primary=ee,seed_effects=ss,late48_effects=late,specific_signal=specific,all_selected_final_checkpoint=end_only,efficiency=efficiency,classification='SCIENTIFIC_SCREEN_RESULT',no_novelty_claim=True,automatic_followup=False)
        save(out/'DECISION.json',result);overall[track]=result
        save(out/'VERIFICATION.json',dict(status='PASS',counts=counts,issued_predictions=len(issued),issued_unique_keys=True,prediction_hashes_verified=True,reference_hashes_verified=True,selection_seal_unchanged=True,source_seal_unchanged=True,all_test_predictions_saved_before_scoring=True,series_equal_weight=True,test_online_prefix_access_not_unopened_test_claim=True,bootstrap_repeats=2000,bootstrap_seed=92403,primary_baseline_fixed_before_test=True))
    save(RESULTS/'SUMMARY_DECISION.json',overall);save(RESULTS/'VERIFICATION.json',dict(status='PASS',counts=counts,offline_fits=16,unmatched_intents=0,duplicate_updates=0,reference_forecasts=len(ref),automatic_followup=False))
    print(json.dumps(overall,indent=2))

if __name__=='__main__':main()
