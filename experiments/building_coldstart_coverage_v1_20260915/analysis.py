"""Fixed screen gates and prediction-only rules, no GPU fits."""
from core import *
from engine import prediction

def records(role):return [r for r in read(OUT/'prediction_manifest.json') if r['role']==role and r['method'] in ['F0','STANDARD','AFFINE']]
def mean_metric(rows,method,h=None):
    rr=[r['scaled_RMSE'] for r in rows if r['method']==method and (h is None or r['history_days']==h)];assert rr;return float(np.mean(rr))
def screen_gate():
    rr=records('dev');assert len(rr)==48;csvwrite(OUT/'stageA_metrics.csv',rr);lookup={(r['building_id'],r['history_days'],r['forecast_type'],r['method']):r['scaled_RMSE'] for r in rr};ints=[]
    for b in read(OUT/'building_split.json')['dev']:
        g={(h,d):gain(lookup[b,h,d,'F0'],lookup[b,h,d,'STANDARD']) for h in [3,14] for d in ['Wednesday','Saturday']}
        d3=g[3,'Wednesday']-g[3,'Saturday'];d14=g[14,'Wednesday']-g[14,'Saturday']
        ints.append(dict(building_id=b,D3=d3,D14=d14,I=d3-d14,H3_Wednesday_gain=g[3,'Wednesday'],H3_Saturday_gain=g[3,'Saturday'],H14_Wednesday_gain=g[14,'Wednesday'],H14_Saturday_gain=g[14,'Saturday'],H3_Saturday_affine_gain_vs_F0=gain(lookup[b,3,'Saturday','F0'],lookup[b,3,'Saturday','AFFINE'])))
    csvwrite(OUT/'stageA_interaction.csv',ints);positive=sum(r['I']>0 for r in ints);mi=float(np.mean([r['I'] for r in ints]));passed=positive>=3 and mi>=.5
    result=dict(tag='확인',status='COVERAGE_PROBLEM_SIGNAL' if passed else 'NO_COVERAGE_PROBLEM_SIGNAL',positive_buildings=positive,total_buildings=4,mean_interaction_pp=mi,rule='>=3/4 I>0 and mean(I)>=0.5pp',cause='Operational development gate, not significance or paper PASS')
    save(OUT/'stageA_gate.json',result);print(result,flush=True);return result

def rule_rows(role,alphas,taus,store=False):
    base=records(role);out=list(base);es={e['id']:e for e in read(OUT/'episodes.json') if e['role']==role}
    for eid,e in es.items():
        r0=next(r for r in base if r['episode']==eid and r['method']=='F0');rl=next(r for r in base if r['episode']==eid and r['method']=='STANDARD')
        with np.load(ROOT/r0['path']) as d:f=d['q'];y=d['target'];h=d['history']
        with np.load(ROOT/rl['path']) as d:l=d['q']
        choices=[('BINARY_FALLBACK',float(e['coverage_count']>0))]+[(f'UNIFORM_{a}',a) for a in alphas]+[(f'COVERAGE_{t}',e['coverage_count']/(e['coverage_count']+t)) for t in taus]
        for method,a in choices:
            q=interpolate(f,l,a)
            if store:r=prediction(e,method,q,q,y,h,extra=dict(alpha=a))
            else:r=dict(episode=eid,building_id=e['building_id'],role=role,history_days=e['history_days'],forecast_type=e['forecast_type'],method=method,alpha=a,**metrics(q,y,h))
            out.append(r)
    return out

def rules():
    assert read(OUT/'stageA_gate.json')['status']=='COVERAGE_PROBLEM_SIGNAL'
    if (OUT/'stageB_selection.json').exists() and read(OUT/'stageB_selection.json')['status']!='NOT_RUN':return read(OUT/'stageB_selection.json')
    rr=rule_rows('dev',[0,.25,.5,.75,1],[.5,1,2,4]);a=min([0,.25,.5,.75,1],key=lambda a:(mean_metric(rr,f'UNIFORM_{a}'),a));t=min([.5,1,2,4],key=lambda t:(mean_metric(rr,f'COVERAGE_{t}'),t))
    cov=f'COVERAGE_{t}';uniform=f'UNIFORM_{a}';simple=min([uniform,'BINARY_FALLBACK','AFFINE'],key=lambda m:(mean_metric(rr,m,3),m));gstd=gain(mean_metric(rr,'STANDARD',3),mean_metric(rr,cov,3));gsimple=gain(mean_metric(rr,simple,3),mean_metric(rr,cov,3));degradation=-gain(mean_metric(rr,'STANDARD',14),mean_metric(rr,cov,14))
    pos=0
    for b in read(OUT/'building_split.json')['dev']:
        def value(m):return next(r['scaled_RMSE'] for r in rr if r['building_id']==b and r['history_days']==3 and r['forecast_type']=='Saturday' and r['method']==m)
        pos+=value(cov)<value(simple)
    result=dict(tag='확인',status='METHOD_SIGNAL' if gstd>=.3 and gsimple>=.3 and pos>=3 and degradation<=.2 else 'METHOD_SIGNAL_FAIL',alpha=a,tau=t,simple_baseline=simple,h3_gain_vs_standard_percent=gstd,h3_gain_vs_simple_percent=gsimple,h3_saturday_positive_buildings=pos,h14_degradation_percent=degradation,selection_scope='Dev4 buildings all16 episodes macro; baseline for gate is fixed best H3 macro simple rule',at=__import__('time').time())
    csvwrite(OUT/'stageB_rules.csv',rr);save(OUT/'stageB_selection.json',result);print(result,flush=True);return result

def heldout_summary():
    if (OUT/'heldout_result.json').exists():return read(OUT/'heldout_result.json')
    sel=read(OUT/'stageB_selection.json');assert sel['status']=='METHOD_SIGNAL';a=sel['alpha'];t=sel['tau'];rr=rule_rows('heldout',[a],[t],store=True);csvwrite(OUT/'heldout_metrics.csv',rr)
    cov=f'COVERAGE_{t}';simple=min([f'UNIFORM_{a}','BINARY_FALLBACK','AFFINE'],key=lambda m:(mean_metric(rr,m),m));gstd=gain(mean_metric(rr,'STANDARD'),mean_metric(rr,cov));gsimple=gain(mean_metric(rr,simple),mean_metric(rr,cov));degradation=-gain(mean_metric(rr,'STANDARD',14),mean_metric(rr,cov,14));bids=read(OUT/'building_split.json')['heldout'];bg=[];pairs=[]
    for b in bids:
        sub=[r for r in rr if r['building_id']==b];s=mean_metric(sub,simple);v=mean_metric(sub,cov);pairs.append([s,v]);bg.append(dict(building_id=b,baseline=simple,baseline_primary=s,coverage_primary=v,gain_percent=gain(s,v)))
    pos=sum(r['gain_percent']>0 for r in bg);rng=np.random.default_rng(61620);arr=np.array(pairs);boot=[]
    for _ in range(2000):
        sample=arr[rng.integers(0,6,6)];s,v=sample.mean(0);boot.append(gain(s,v))
    ci=np.quantile(boot,[.025,.975]);result=dict(tag='확인',status='HELDOUT_METHOD_SIGNAL' if gstd>0 and gsimple>=.3 and pos>=4 and degradation<=.5 else 'HELDOUT_METHOD_SIGNAL_FAIL',simple_baseline=simple,gain_vs_standard_percent=gstd,gain_vs_simple_percent=gsimple,positive_buildings=pos,h14_degradation_percent=degradation,paired_building_bootstrap_95ci=ci.tolist(),bootstrap_reps=2000,bootstrap_seed=61620,macro={m:mean_metric(rr,m) for m in sorted({r['method'] for r in rr})})
    csvwrite(OUT/'heldout_building_gains.csv',bg);save(OUT/'heldout_result.json',result);return result

def scalar_metrics(q,y,h):
    # Independent scalar loops, including independent population std.
    hv=[float(x) for x in h];mu=sum(hv)/len(hv);scale=max(math.sqrt(sum((x-mu)**2 for x in hv)/len(hv)),1e-6)
    errors=[float(q[10,i])-float(y[i]) for i in range(24)];rmse=math.sqrt(sum(x*x for x in errors)/24);pin=[]
    for k,level in enumerate(QUANTILES):
        for i in range(24):
            d=float(y[i])-float(q[k,i]);pin.append(2*(level*d if d>=0 else (level-1)*d))
    return dict(scaled_RMSE=rmse/scale,raw_RMSE=rmse,raw_MAE=sum(abs(x) for x in errors)/24,scaled_2pinball=sum(pin)/len(pin)/scale)

def verify():
    hist=read(OUT/'historical_hashes.json')
    for p,h in hist.items():assert sha(ROOT/p)==h,('OLD_RESULT_CHANGED',p)
    if (OUT/'prepare_seal.json').exists():check_seal()
    manifest=read(OUT/'prediction_manifest.json') if (OUT/'prediction_manifest.json').exists() else [];maxerror=0.;replays=[]
    for r in manifest:
        p=ROOT/r['path'];assert sha(p)==r['sha256']
        with np.load(p) as d:
            v=scalar_metrics(d['q'],d['target'],d['history']);assert np.all(np.diff(d['q'],axis=0)>=0)
        for k,x in v.items():
            err=abs(x-r[k]);maxerror=max(maxerror,err);assert math.isclose(x,r[k],rel_tol=1e-10,abs_tol=1e-10),(r['id'],k,x,r[k])
        replays.append(dict(id=r['id'],**v))
    fits=read(OUT/'fit_attempts.json') if (OUT/'fit_attempts.json').exists() else [];assert len(fits)<=48
    steps=[]
    for f in fits:
        rr=csvread(OUT/(f['fit']+'_trajectory.csv'));assert len(rr)==f['actual_updates'];steps+=rr
        if f['status']=='COMPLETE':assert f['frozen_unchanged'] and f['reload_max_abs']==0 and sha(ROOT/f['checkpoint'])==f['checkpoint_sha256']
    csvwrite(OUT/'train_trajectories.csv',steps);csvwrite(OUT/'resource_usage.csv',[{k:v for k,v in f.items() if k not in ['recipe_records','traceback','affine']} for f in fits])
    result=dict(tag='확인',status='VERIFIED',prediction_records=len(manifest),max_metric_abs_error=maxerror,scalar_tolerance=dict(relative=1e-10,absolute=1e-10),fits_attempted=len(fits),fits_completed=sum(f['status']=='COMPLETE' for f in fits),optimizer_updates=sum(f['actual_updates'] for f in fits),historical_files_preserved=len(hist),heldout_evaluated=any(r['role']=='heldout' for r in manifest),new_gpu_updates=0,replays=replays)
    save(OUT/'independent_verification.json',result);return result
