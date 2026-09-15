"""Read-only experiment audit, independent scalar scoring and finite-run reporting.
Does not launch fits or change any sealed experimental source or selection.
"""
import argparse, csv, hashlib, json, math, statistics, sys, time, traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN='building_peft_topic_decision_20260916'
OUT=ROOT/'results'/RUN
EXP=ROOT/'experiments'/RUN
METRICS=['primary','raw_RMSE','raw_MAE','scaled_2pinball']
METHODS=['STD','SIMPLE','CANDIDATE']
POLICIES=['ZERO','EPOCH1','EPOCH4','EPOCH16','FIXED120']
def read(p): return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p); t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def csvwrite(p,rows):
    if not rows:return
    with open(p,'w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator="\n");w.writeheader();w.writerows(rows)
def avg(x): return math.fsum(float(v) for v in x)/len(x)
def at_step(policy,n):return {'ZERO':0,'EPOCH1':n,'EPOCH4':4*n,'EPOCH16':16*n,'FIXED120':120}[policy]
def gain(base,cand):return None if base==0 else 100*(base-cand)/base

def scalar_metric(q,y,h,taus):
    """No runtime.metric / vectorized reductions. Python doubles and scalar loops."""
    mi=list(taus).index(.5); hh=[float(v) for v in h]; mu=avg(hh)
    sd=max(math.sqrt(avg([(v-mu)**2 for v in hh])),1e-6)
    e=[float(q[mi,j])-float(y[j]) for j in range(24)]
    rmse=math.sqrt(avg([v*v for v in e])); pins=[]
    for k,tau in enumerate(taus):
        for j in range(24):
            d=float(y[j])-float(q[k,j]);pins.append(2*(tau*d if d>=0 else (tau-1)*d))
    return dict(primary=rmse/sd,raw_RMSE=rmse,raw_MAE=avg([abs(v) for v in e]),scaled_2pinball=avg(pins)/sd,history_std=sd)

def assert_near(a,b):
    # Fixed before LOCKED scores; numerical equivalence only, never a score threshold.
    assert math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-10),(a,b)

def seals():
    s=read(OUT/'std_seal.json')
    for p,h in s['code'].items():assert sha(ROOT/p)==h,('STD_SOURCE_CHANGED',p)
    assert sha(OUT/'episodes.json')==s['episodes_sha256']
    for p,h in s['model'].items():assert sha(p)==h,('MODEL_CHANGED',p)
    t=read(OUT/'topic_seal.json')
    for p,h in t['sources'].items():assert sha(ROOT/p)==h,('TOPIC_SOURCE_CHANGED',p)
    assert sha(OUT/'STD_selection.json')==t['std_selection_sha256']
    if (OUT/'selection_seal.json').exists():
        x=read(OUT/'selection_seal.json');assert x['topic_seal_sha256']==sha(OUT/'topic_seal.json')
        for p,h in x['selection_sources'].items():assert sha(ROOT/p)==h
    environment=read(OUT/'verification_protocol.json')
    for p,h in environment['chronos_python_sources'].items():assert sha(ROOT/p)==h,('INSTALLED_MODEL_CODE_CHANGED',p)
    old=read(OUT/'historical_hashes.json')
    for p,h in old.items():assert sha(ROOT/p)==h,('HISTORICAL_CHANGED',p)
    return dict(old_files_unchanged=len(old),std_seal=sha(OUT/'std_seal.json'),topic_seal=sha(OUT/'topic_seal.json'),model_files=len(s['model']))

def independent_selection(rows,method):
    choices=[]
    for lr in [3e-5,1e-4]:
        for policy in POLICIES:
            rr=[r for r in rows if r['role']=='DISCOVERY' and r['method']==method and r['lr']==lr and r['step']==at_step(policy,r['days']-1)]
            assert len(rr)==8
            choices.append(dict(lr=lr,policy=policy,primary=avg([r['primary'] for r in rr]),mean_updates=avg([r['step'] for r in rr])))
    old=read(OUT/f'{method}_selection.json')
    for a,b in zip(choices,old['choices']):
        assert (a['lr'],a['policy'])==(b['lr'],b['policy']);assert_near(a['primary'],b['primary'])
    best=min(choices,key=lambda r:(r['primary'],r['mean_updates'],r['lr']))
    assert (best['lr'],best['policy'])==(old['selected']['lr'],old['selected']['policy'])
    return best

def independent_references(rows,episodes,arrays):
    records=[];mi=read(OUT/'parameter_audit.json')['median_index']
    for row in rows:
        if row['method']!='AFFINE':continue
        eid=row['episode'];h=arrays[eid][0]
        standard=next(r for r in rows if r['episode']==eid and r['method']=='STD' and r['step']==0)
        initial=np.load(ROOT/standard['prediction_path']);past_q=initial['q'][1:,mi,:]
        ys=np.stack([h[j:j+24] for j in range(24,len(h)-23,24)])
        x=[float(v) for rr in past_q for v in rr];y=[float(v) for rr in ys for v in rr]
        mx=avg(x);my=avg(y);var=avg([(v-mx)**2 for v in x]);fallback=bool(var<=np.finfo(float).eps*max(1,avg([v*v for v in x])))
        if fallback:a,b=1.,0.
        else:a=max(1e-6,avg([(v-mx)*(z-my) for v,z in zip(x,y)])/var);b=my-a*mx
        assert row['affine']['fallback']==fallback;assert_near(a,row['affine']['a']);assert_near(b,row['affine']['b'])
        for method in ['F0','AFFINE','SEASONAL24']:
            reference=next(r for r in rows if r['episode']==eid and r['method']==method);z=np.load(ROOT/reference['prediction_path'])
            for kind in ['q','raw']:
                expected=initial[kind][0] if method=='F0' else a*initial[kind][0]+b if method=='AFFINE' else np.tile(h[-24:],(len(initial['q'][0]),1))
                assert np.allclose(z[kind],expected,atol=1e-10,rtol=1e-10),(eid,method,kind)
        records.append(dict(episode=eid,role=row['role'],a=a,b=b,fallback=fallback,past_supervision_points=len(y),status='PASS'))
    return records


def verify_cpu():
    evidence=seals(); fits=read(OUT/'fits.json'); rows=read(OUT/'scores.json')
    topic_time=read(OUT/'topic_seal.json')['at']
    for fit in fits:
        if fit['method']!='STD':assert fit['started_at']>=topic_time
        if fit['role']=='LOCKED':assert fit['started_at']>=read(OUT/'selection_seal.json')['at']
    assert not any(r['status']=='RUNNING' for r in fits),'Worker still running'
    assert len({r['id'] for r in fits})==len(fits)<=120
    assert sum(r['updates'] for r in fits)<=19680
    smoke=sum(r['updates'] for r in read(OUT/'smoke_ledger.json'));assert smoke<=24
    assert read(OUT/'std_smoke.json')['status']=='PASS' and read(OUT/'candidate_smoke.json')['status']=='PASS'
    episodes={e['id']:e for e in read(OUT/'episodes.json')}
    arrays={};seen=set();manifest=[];maximum=0.;checks=0
    for r in rows:
        e=episodes[r['episode']]
        if e['role']=='LOCKED':assert (OUT/'selection_seal.json').exists()
        if e['id'] not in arrays:
            for d in [e['history'],e['target']]:assert sha(ROOT/d['path'])==d['sha256']
            arrays[e['id']]=(np.load(ROOT/e['history']['path']),np.load(ROOT/e['target']['path']))
        h,y=arrays[e['id']];path=ROOT/r['prediction_path'];assert sha(path)==r['prediction_sha256'];z=np.load(path)
        q,raw=z['q'],z['raw'];assert np.array_equal(np.sort(raw,axis=-2),q),'SORT_MISMATCH'
        assert np.isfinite(q).all() and np.isfinite(raw).all()
        if q.ndim==3:q=q[0]
        vals=scalar_metric(q,y,h,read(OUT/'parameter_audit.json')['quantiles'])
        for k,v in vals.items():assert_near(v,r[k]);maximum=max(maximum,abs(v-r[k]));checks+=1
        if str(path) not in seen:
            seen.add(str(path));manifest.append(dict(path=r['prediction_path'],sha256=r['prediction_sha256'],fit=r['fit'],step=r['step'],episode=e['id'],history_sha256=e['history']['sha256'],target_sha256=e['target']['sha256'],shape=list(z['q'].shape),contents='raw unsorted and common sorted quantiles; fit arrays contain final forecast then history-window forecasts'))
    references=independent_references(rows,episodes,arrays)
    save(OUT/'independent_references.json',dict(status='PASS',episodes=len(references),affine_fallbacks=sum(r['fallback'] for r in references),records=references))
    losses=[]; checkpoints=0
    for f in fits:
        assert len(f['losses'])==f['updates']
        assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in f['losses'])
        if f['status']=='COMPLETE':
            assert f['frozen_unchanged'] and f['buffers_unchanged'] and f['parameter_updated'] and f['trainable']==147456
            assert sorted(map(int,f['checkpoints']))==sorted({at_step(p,f['days']-1) for p in POLICIES})
            assert f['updates']==max(at_step(p,f['days']-1) for p in POLICIES)
        for k,d in f['checkpoints'].items():
            assert sha(ROOT/d['weights'])==d['weights_sha256'];checkpoints+=1
            matches=[r for r in rows if r['fit']==f['id'] and r['step']==int(k)]
            if f['status']=='COMPLETE':assert len(matches)==1
        for l in f['losses']:losses.append(dict(fit=f['id'],**l))
    selections={m:independent_selection(rows,m) for m in METHODS if (OUT/f'{m}_selection.json').exists()}
    csvwrite(OUT/'loss_curve.csv',losses);save(OUT/'prediction_manifest.json',manifest)
    evidence.update(status='PASS',rows=len(rows),scalar_values=checks,max_abs_difference=maximum,checkpoints=checkpoints,main_attempts=len(fits),main_updates=sum(f['updates'] for f in fits),smoke_updates=smoke,selection_recomputed=selections,verifier_sha256=sha(__file__),at=time.time())
    save(OUT/'independent_cpu_verification.json',evidence);print(json.dumps(evidence,ensure_ascii=False),flush=True)
    return evidence

def verify_calibration(rt,m,h,w):
    import torch
    hh=hashlib.sha256(np.asarray(h,dtype=np.float64).tobytes()).hexdigest()
    out=OUT/'independent_calibration.json';done=read(out) if out.exists() else {}
    if hh in done:return
    metas=[read(OUT/'calibration'/(hh+'_'+method+'.json')) for method in ['SIMPLE','CANDIDATE'] if (OUT/'calibration'/(hh+'_'+method+'.json')).exists()]
    if not metas:return
    x,y=rt.windows(h);residual=[];losses=[];mi=list(m.chronos_config.quantiles).index(.5)
    with torch.no_grad(),rt.disabled(m):
        for xx,yy in zip(x,y):
            w.boundary();v=rt.native(m,xx,yy);_,(_,scale)=m.instance_norm(rt.tensor(xx).reshape(1,24));sigma=float(scale.reshape(-1)[0])
            q=v.quantile_preds[0,mi,:24].double().cpu().numpy()
            residual.append([(float(q[j])-float(yy[j]))/sigma for j in range(24)]);losses.append(float(v.loss))
    n=len(residual);b=[avg(row) for row in residual];mb=avg(b);shape=[[row[j]-b[i] for j in range(24)] for i,row in enumerate(residual)];ms=[avg([row[j] for row in shape]) for j in range(24)]
    sl=mb**2;nl=math.fsum((v-mb)**2 for v in b)/(n*(n-1));ss=avg([v*v for v in ms]);ns=avg([math.fsum((row[j]-ms[j])**2 for row in shape)/(n*(n-1)) for j in range(24)])
    ul=(nl+1e-6)/(sl+nl+1e-6);us=(ns+1e-6)/(ss+ns+1e-6);dimension=len(m.chronos_config.quantiles)*24;c=(ul+(dimension-1)*us)/dimension;wl=ul/c;ws=us/c
    energy=avg([v*v for row in residual for v in row]);base=avg(losses);lam=base/(energy+1e-6)
    for meta in metas:
        for k,val in dict(w_level=wl,w_shape=ws,signal_level=sl,noise_level=nl,signal_shape=ss,noise_shape=ns,u_level=ul,u_shape=us).items():assert_near(meta['geometry'][k],val)
        assert_near(meta['lambda_value'],lam);assert_near(meta['f0_native_loss'],base);assert_near(meta['f0_median_residual_energy'],energy)
        assert_near(meta['effective_w_level'],1 if meta['method']=='SIMPLE' else wl);assert_near(meta['effective_w_shape'],1 if meta['method']=='SIMPLE' else ws)
    done[hh]=dict(status='PASS',windows=n,methods=[r['method'] for r in metas],w_level=wl,w_shape=ws,lambda_value=lam,source='Independent scalar statistics from frozen native F0 and exposed H only')
    save(out,done)


def verify_gpu():
    sys.path.insert(0,str(EXP));import runtime as rt
    import torch
    cpu=read(OUT/'independent_cpu_verification.json');assert cpu['status']=='PASS'
    assert not any(f['status']=='RUNNING' for f in read(OUT/'fits.json'))
    prior=read(OUT/'checkpoint_replay.json') if (OUT/'checkpoint_replay.json').exists() else []
    done={(r['fit'],r['step']):r for r in prior};episodes={e['id']:e for e in read(OUT/'episodes.json')}
    w=rt.Watch(OUT,'independent_replay',wall_cap=14400);start=time.monotonic()
    try:
        w.boundary(startup=True)
        for f in read(OUT/'fits.json'):
            if f['status']!='COMPLETE':continue
            if all((f['id'],int(k)) in done for k in f['checkpoints']):continue
            rt.setup(f['seed']);m=rt.make(f['seed']);matches=[e for e in episodes.values() if (e['building'],e['days'],e['origin'])==(f['building'],f['days'],f['origin'])];assert len(matches)==1
            ep=matches[0];h=rt.load_history(ep['history']);x,_=rt.windows(h)
            verify_calibration(rt,m,h,w)
            for k,d in sorted(f['checkpoints'].items(),key=lambda z:int(z[0])):
                if (f['id'],int(k)) in done:continue
                assert sha(ROOT/d['weights'])==d['weights_sha256']
                rt.restore(m,torch.load(ROOT/d['weights'],map_location='cpu',weights_only=True));z=np.load(ROOT/d['path']);largest=0.
                for j,xx in enumerate([h[-24:]]+list(x)):
                    raw=rt.predict(m,xx,w)[1];largest=max(largest,float(np.max(abs(raw-z['raw'][j]))))
                    assert np.array_equal(raw,z['raw'][j]),('RELOAD_NOT_EXACT',f['id'],k,j,largest)
                if int(k)==0:
                    with rt.disabled(m):base=rt.predict(m,h[-24:],w)[1]
                    assert np.array_equal(base,z['raw'][0])
                row=dict(fit=f['id'],step=int(k),weights_sha256=d['weights_sha256'],prediction_sha256=d['sha256'],queries_replayed=len(x)+1,max_abs_difference=largest,exact=True)
                prior.append(row);done[(f['id'],int(k))]=row;save(OUT/'checkpoint_replay.json',prior)
            del m;rt.cleanup()
            print('REPLAY_FIT',f['id'],flush=True)
        save(OUT/'independent_gpu_verification.json',dict(status='PASS',checkpoints=len(prior),queries=sum(r['queries_replayed'] for r in prior),max_abs_difference=max((r['max_abs_difference'] for r in prior),default=0.),fresh_model_per_fit=True,all_history_and_final_forecasts_replayed=True,optimizer_updates=0,wall_seconds=time.monotonic()-start,verifier_sha256=sha(__file__)))
    finally:w.close()

def paired_effect(base,cand,indices=None):
    """One element per physical building, already averaged over H and seed."""
    b=np.asarray(base,dtype=float);c=np.asarray(cand,dtype=float);assert b.shape==c.shape and b.ndim==1
    if indices is None:indices=np.random.default_rng(61690).integers(0,len(b),(2000,len(b)))
    observed=gain(avg(b),avg(c));g=[];a=[]
    for ids in indices:
        bb=avg(b[ids]);cc=avg(c[ids]);v=gain(bb,cc)
        if v is not None:g.append(v)
        a.append(bb-cc)
    lo=[]
    for i in range(len(b)):
        v=gain(avg(np.delete(b,i)),avg(np.delete(c,i)))
        if v is not None:lo.append(v)
    return dict(gain_percent=observed,absolute_difference=avg(b)-avg(c),ci95_percent=np.percentile(g,[2.5,97.5]).tolist() if len(g)==len(indices) else None,ci95_absolute=np.percentile(a,[2.5,97.5]).tolist(),zero_denominator_resamples=len(indices)-len(g),leave_one_out_range_percent=[min(lo),max(lo)] if len(lo)==len(b) else None,bootstrap_unit='physical building; H and seeds kept together',bootstrap_samples=len(indices),bootstrap_seed=61690)

def summarize():
    rows=read(OUT/'scores.json');episodes=[e for e in read(OUT/'episodes.json') if e['role']=='LOCKED'];selection=read(OUT/'selection_seal.json')['selected']
    out=[]
    for e in episodes:
        for seed in [61680,61681]:
            for method in METHODS+['F0','AFFINE','SEASONAL24']:
                labels=[method,method+'_FIXED120'] if method in METHODS else [method]
                for label in labels:
                    lr=selection[method]['lr'] if method in METHODS else 0
                    step=at_step(selection[method]['policy'],e['days']-1) if method in METHODS and label==method else 120 if method in METHODS else 0
                    ss=seed if method in METHODS else 0
                    matches=[r for r in rows if r['episode']==e['id'] and r['method']==method and r['lr']==lr and r['step']==step and r['seed']==ss]
                    assert len(matches)==1,('MISSING_LOCKED_CELL',e['id'],label,seed)
                    r=matches[0];out.append(dict(episode=e['id'],building=e['building'],days=e['days'],seed=seed,arm=label,physical_fit=r['fit'],reference_replicated=method not in METHODS,step=step,train_seconds=r['train_seconds'],**{k:r[k] for k in METRICS}))
    csvwrite(OUT/'locked_selected_and_fixed_scores.csv',out)
    buildings=sorted({r['building'] for r in out});assert len(buildings)==6
    arms=list(dict.fromkeys(r['arm'] for r in out));br=[];macro=[];vectors={}
    for arm in arms:
        ar=[r for r in out if r['arm']==arm];vector=[]
        for b in buildings:
            rr=[r for r in ar if r['building']==b];assert len(rr)==4
            bb=dict(building=b,arm=arm,**{k:avg([r[k] for r in rr]) for k in METRICS});br.append(bb);vector.append(bb['primary'])
        vectors[arm]=vector
        macro.append(dict(arm=arm,**{k:avg([r[k] for r in br if r['arm']==arm]) for k in METRICS},H3=avg([r['primary'] for r in ar if r['days']==3]),H14=avg([r['primary'] for r in ar if r['days']==14]),seed61680=avg([r['primary'] for r in ar if r['seed']==61680]),seed61681=avg([r['primary'] for r in ar if r['seed']==61681]),selected_mean_updates=avg([r['step'] for r in ar]),selected_gradient_seconds_mean=avg([r['train_seconds'] for r in ar]),worst_building=buildings[int(np.argmax(vector))],worst_building_primary=max(vector)))
    lookup={r['arm']:r for r in macro};candidate=lookup['CANDIDATE'];effects={}
    ids=np.random.default_rng(61690).integers(0,6,(2000,6))
    for arm in arms:
        if arm=='CANDIDATE':continue
        effects[arm]=paired_effect(vectors[arm],vectors['CANDIDATE'],ids)
        effects[arm]['by_H']={str(h):gain(lookup[arm]['H'+str(h)],candidate['H'+str(h)]) for h in [3,14]}
        effects[arm]['by_seed']={str(s):gain(lookup[arm]['seed'+str(s)],candidate['seed'+str(s)]) for s in [61680,61681]}
    fixed=['STD_FIXED120','SIMPLE_FIXED120','F0','AFFINE','SEASONAL24']
    strongest=min(fixed,key=lambda a:lookup[a]['primary'])
    gates={}
    for arm in ['STD','SIMPLE']:
        e=effects[arm];g=e['gain_percent'];ci=e['ci95_percent']
        gates[arm+'_gain_ge_1pct']=g is not None and g>=1
        gates[arm+'_each_seed_positive']=all(v is not None and v>0 for v in e['by_seed'].values())
        gates[arm+'_CI_lower_positive']=ci is not None and ci[0]>0
        gates[arm+'_neither_H_worse_over_1pct']=all(v is not None and v>=-1 for v in e['by_H'].values())
    gates['no_fixed_control_better']=candidate['primary']<=lookup[strongest]['primary']
    gates['not_worse_than_F0']=candidate['primary']<=lookup['F0']['primary']
    # No scientific success inferred from execution success or a favorable secondary metric.
    if all(gates.values()):predictive='SCREEN_PASS'
    elif any(lookup[a]['primary']<=candidate['primary'] for a in ['STD','SIMPLE']+fixed):
        positive=all(effects[a]['gain_percent'] is not None and effects[a]['gain_percent']>0 for a in ['STD','SIMPLE'])
        predictive='TRADEOFF_ONLY' if positive else 'NO_ADDED_VALUE_IN_THIS_PILOT'
    else:predictive='POSITIVE_BUT_UNCERTAIN'
    novelty='UNRESOLVED'
    decision='PROMISING_EFFECT_TOPIC_NOT_CONFIRMED' if predictive=='SCREEN_PASS' else 'TOPIC_UNRESOLVED' if predictive in ['POSITIVE_BUT_UNCERTAIN','TRADEOFF_ONLY'] else 'NO_METHOD_TOPIC_THIS_RUN'
    csvwrite(OUT/'building_summary.csv',br);csvwrite(OUT/'macro_scores.csv',macro);save(OUT/'paired_building_effects.json',effects)
    result=dict(predictive=predictive,novelty=novelty,topic_decision=decision,gates=gates,strongest_fixed_control=strongest,macro=macro,effects=effects,buildings=buildings,selected=selection)
    save(OUT/'scientific_decision.json',result);return result

def inventory():
    fits=read(OUT/'fits.json');aliases=read(OUT/'fit_aliases.json') if (OUT/'fit_aliases.json').exists() else []
    records={r['id']:r for r in fits};aliasmap={r['id']:r['physical_fit'] for r in aliases};expected=[]
    selected=read(OUT/'selection_seal.json')['selected'] if (OUT/'selection_seal.json').exists() else {}
    for e in read(OUT/'episodes.json'):
        for method in METHODS:
            seeds=[61680] if e['role']=='DISCOVERY' else [61680,61681]
            lrs=[3e-5,1e-4] if e['role']=='DISCOVERY' else [selected[method]['lr']] if method in selected else [None]
            for seed in seeds:
                for lr in lrs:
                    fid=f"{method}_{e['id']}_lr{lr}_seed{seed}"
                    physical=aliasmap.get(fid,fid);r=records.get(physical)
                    expected.append(dict(id=fid,physical_fit=physical if r else None,role=e['role'],method=method,building=e['building'],days=e['days'],seed=seed,lr=lr,status=r['status'] if r else 'NOT_RUN',updates=r['updates'] if r else 0,alias=fid in aliasmap))
    assert len(expected)==120
    if selected:assert set(records)<=set(r['id'] for r in expected)
    errors=[{k:r[k] for k in ['id','status','updates','error'] if k in r} for r in fits if r['status'] not in ['COMPLETE','RUNNING']]
    unrun=[r for r in expected if r['status']!='COMPLETE']
    save(OUT/'execution_errors.json',dict(fit_errors=errors,controller_status=read(OUT/'status.json'),preparation_repair=read(OUT/'preparation_repair.json')))
    save(OUT/'execution_inventory.json',dict(main_attempts=len(fits),complete_physical_fits=sum(r['status']=='COMPLETE' for r in fits),main_updates=sum(r['updates'] for r in fits),smoke_updates=sum(r['updates'] for r in read(OUT/'smoke_ledger.json')),aliases=aliases,expected_logical_fits=120,unexecuted_or_incomplete=len(unrun),cells=expected))
    (OUT/'UNEXECUTED.md').write_text('# 미실행 및 오류 범위\n\n'+('계약의 120개 논리 비교 셀을 모두 완료했다. 동일 알고리즘 alias는 실행 원장에서 물리 학습과 분리한다.\n' if not unrun else '\n'.join(f"- {r['id']}: {r['status']}, {r['updates']} updates" for r in unrun)+'\n')+'\n새 후보, Censor, Query, source bank, 다른 rank/학습률 추가 실험은 이번 실행 범위에 포함하지 않았다.\n')
    return read(OUT/'execution_inventory.json')

def resources():
    fits=read(OUT/'fits.json');result=[]
    for role in ['DISCOVERY','LOCKED']:
        for method in METHODS:
            rr=[r for r in fits if r['role']==role and r['method']==method and r['status']=='COMPLETE']
            if not rr:continue
            result.append(dict(role=role,method=method,fits=len(rr),updates=sum(r['updates'] for r in rr),training_seconds=sum(r['training_seconds'] for r in rr),fit_wall_seconds=sum(r['wall_seconds'] for r in rr),median_peak_allocated_MiB=statistics.median(r['peak_allocated']/2**20 for r in rr),maximum_peak_allocated_MiB=max(r['peak_allocated']/2**20 for r in rr),median_peak_reserved_MiB=statistics.median(r['peak_reserved']/2**20 for r in rr),trainable=147456))
    csvwrite(OUT/'resources.csv',result);samples=[]
    for p in OUT.glob('gpu_*.jsonl'):
        for line in p.read_text().splitlines():
            if line.strip():samples.append(read_line(line,p))
    monitor=dict(samples=len(samples),minimum_free_MiB=min((r['free_mib'] for r in samples),default=None),unapproved_external_compute_samples=sum(any(not a['own'] and a['name']!='/usr/share/rustdesk/rustdesk' for a in r['apps']) for r in samples),below_1GiB_samples=sum(r['free_mib']<1024 for r in samples),rustdesk_exception_samples=sum(any(a['name']=='/usr/share/rustdesk/rustdesk' for a in r['apps']) for r in samples),budget=read(OUT/'gpu_budget.json'))
    calibration=[read(p) for p in sorted((OUT/'calibration').glob('*.json'))]
    csvwrite(OUT/'calibration_summary.csv',[dict(history_sha256=r['history_sha256'],method=r['method'],lambda_value=r['lambda_value'],effective_w_level=r['effective_w_level'],effective_w_shape=r['effective_w_shape'],calibration_first_call_seconds=r['calibration_seconds'],**r['geometry']) for r in calibration])
    save(OUT/'resource_summary.json',dict(groups=result,monitor=monitor,calibration_timing_scope='Each history/method metadata retains first call only; full repeated calibration is included in fit wall, not gradient seconds. No claim that first-call sum is all calibration cost.'))
    return result,monitor

def read_line(line,path):
    try:return json.loads(line)
    except Exception as e:raise ValueError(f'Invalid GPU record in {path}') from e

def fnum(v):return 'N/A' if v is None else f'{v:.6f}'
def interval(v):return 'N/A' if v is None else f'[{v[0]:.3f}, {v[1]:.3f}]'

def report(valid=True,error=None):
    inv=inventory();res,monitor=resources();complete=inv['unexecuted_or_incomplete']==0
    verified=valid and (OUT/'independent_gpu_verification.json').exists() and read(OUT/'independent_gpu_verification.json')['status']=='PASS'
    execution='COMPLETE' if complete and verified else 'INVALID' if not valid else 'PARTIAL'
    d=summarize() if complete and verified else None
    predictive=d['predictive'] if d else 'INCONCLUSIVE_EXECUTION';novelty='UNRESOLVED'
    decision=d['topic_decision'] if d else 'TOPIC_UNRESOLVED'
    table='';effecttext='독립 비교 판정에 필요한 검산 또는 평가가 완료되지 않았다.'
    if d:
        table='| 방법 | Primary | Raw RMSE | MAE | scaled 2-pinball | H3 primary | H14 primary |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for r in d['macro']:table+='| '+r['arm']+' | '+' | '.join(fnum(r[k]) for k in METRICS+['H3','H14'])+' |\n'
        effecttext='\n'.join(f"- {a} 대비 후보: {fnum(d['effects'][a]['gain_percent'])}%, 건물 paired bootstrap 95% {interval(d['effects'][a]['ci95_percent'])}%, leave-one-building-out {interval(d['effects'][a]['leave_one_out_range_percent'])}%." for a in ['STD','SIMPLE',d['strongest_fixed_control']])
    firsttopic='이번 실행에서 새 방법론 주제 미확보' if not d or predictive!='SCREEN_PASS' else '예측 개선 신호가 있으나 새 방법론 주제는 신규성 검토 전까지 미확정'
    nextdecision='종료' if predictive=='NO_ADDED_VALUE_IN_THIS_PILOT' else '보류'
    strongest=d['strongest_fixed_control'] if d else '미판정'
    if d:
        mm={r['arm']:r for r in d['macro']}
        strongest+=f" (primary {mm[d['strongest_fixed_control']]['primary']:.9f}; 후보 {mm['CANDIDATE']['primary']:.9f})"
    counts=f"본학습 {inv['complete_physical_fits']}/{inv['main_attempts']} fits 완료, {inv['main_updates']} optimizer updates. Smoke {inv['smoke_updates']} updates 별도. 미완료 논리 셀 {inv['unexecuted_or_incomplete']}/120."
    final=f"1) {firsttopic}.\n\n2) 타깃 과거 잔차의 일관성에 따라 같은 trace의 레벨·형상 보존 가중치를 배분했다. 균일 functional anchoring을 직접 대조했다. 비등방 regularization 자체는 알려져 있으며 구체 수식의 신규성은 미확인이다.\n\n3) 가장 강한 사전 고정 단순 대조: {strongest}.\n{effecttext}\n\n4) {counts}\n\n5) EXECUTION={execution} / PREDICTIVE_EVIDENCE={predictive} / NOVELTY={novelty}. 최종 판정={decision}.\n\n6) 다음 결정: {nextdecision}. 이 실행에서는 추가 후보나 학습을 자동 실행하지 않는다.\n"
    (OUT/'FINAL_DECISION.md').write_text(final)
    selected='\n'.join(f"- {m}: LR={v['lr']}, {v['policy']}, DISCOVERY primary={v['primary']:.9f}." for m,v in (d['selected'] if d else {}).items())
    costs='| 범위 | 방법 | Fits | Updates | Gradient 초 | Fit wall 초 | Median peak allocated MiB |\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in res:costs+=f"| {r['role']} | {r['method']} | {r['fits']} | {r['updates']} | {r['training_seconds']:.2f} | {r['fit_wall_seconds']:.2f} | {r['median_peak_allocated_MiB']:.2f} |\n"
    validation=read(OUT/'independent_cpu_verification.json') if (OUT/'independent_cpu_verification.json').exists() else {}
    replay=read(OUT/'independent_gpu_verification.json') if (OUT/'independent_gpu_verification.json').exists() else {}
    zero='선택 ZERO는 정확히 F0 반환이다. ZERO가 선택된 방법의 차이를 학습 효과라고 해석하지 않는다.'
    gates='\n'.join('- '+k+': '+str(v) for k,v in d['gates'].items()) if d else '미판정'
    text=f"""# 짧은 이력 건물 PEFT — 한 후보의 직접 비교

[확인] **{execution} / {predictive} / {novelty}**. {firsttopic}. 이 문서는 실행 성공과 방법 발견을 분리한다. {counts}

## 문제와 변경의 이유

[설계] 새 건물의 3일/14일 관측만으로 다음 24시간을 예측할 때, 타깃만 학습하는 rank1 LoRA에 잔차 기반 보존 방향 배분을 추가하면 표준 LoRA와 균일 보존보다 도움이 되는지 물었다. 기존 결과에서는 학습량과 건물에 따라 수준 편향과 형상 오차가 서로 다르게 움직였다. 그 분해는 사후 진단이며 인과 증명이나 미래 입력이 아니다. [EVIDENCE.md](EVIDENCE.md)에 반례와 과거 원점수를 보존했다.

[설계] STD는 native supervised loss, SIMPLE은 균일한 F0 출력 보존항, CANDIDATE는 같은 보존항의 레벨·형상 가중치만 달리한다. lambda 척도와 파라미터·학습 기회는 동일하다. Frozen F0의 이력 내부 잔차로 가중치를 계산하며, 평가 정답으로 보정 크기를 맞추지 않는다. 정확한 식과 동치는 [MECHANISM_SPEC.md](MECHANISM_SPEC.md)에 있다.

## 정보와 비교의 공정성

[확인] DISCOVERY는 이미 노출된 기존 tune4, LOCKED는 노출 감사에서 성능 사용이 발견되지 않은 기존 heldout6이다. 새로운 건물 교체 없이 canonical ID 순으로 수요일/토요일 origin을 고정했다. 한 건물당 origin 하나이며 두 H와 두 seed를 독립 건물로 세지 않았다. 모든 worker는 H와 모델만 받는다. LOCKED 채점은 선택 봉인 뒤에 수행했다.

[설계] native Chronos-2 revision29ec3766d36d6f73f0696f85560a422f50e8498c, FP32/batch1, rank1/alpha2, 96 projections/147456 trainable scalars. LR2개×budget5종의 전역 선택 기회가 세 방법에 동일하다. 각 fit은 H3 120/H14 208 updates까지 실행했다. 선택 checkpoint와 fixed120은 같은 trajectory에서 읽었다.

{selected}

[확인] {zero} 후보 개발 점수의 부호를 LOCKED 입장 gate로 사용하지 않았다. 초기 CPU 수식 prototype 작성은 STD 종료 직전이었다는 준비 순서 차이를 [RESEARCH_REVIEW.md](RESEARCH_REVIEW.md)에 기록했다. 후보 GPU 비교는 topic seal 후 하나의 고정 구현으로 진행했다.

## 실제 원점수와 추가 가치

Primary는 episode median RMSE / H의 population std(floor1e-6)를 건물 내 H·seed 평균한 뒤 건물 평균한 값이다. Raw RMSE/MAE와 scaled 2-pinball도 같은 셀에서 보고한다. 분위수 산술평균을 조건부 평균으로 부르지 않는다.

{table}

{effecttext}

사전 SCREEN 조건 각각의 결과:

{gates}

[확인] F0→STD는 타깃 학습 자체, STD→SIMPLE은 균일 출력 보존, SIMPLE→CANDIDATE는 잔차 기반 방향 배분의 추가 가치를 비교한다. 선택형과 fixed120형을 함께 공개했으며 평가 뒤 더 좋은 checkpoint로 배포 설정을 바꾸지 않았다. 전체 building/seed/H 점수는 [locked_selected_and_fixed_scores.csv](locked_selected_and_fixed_scores.csv), 건물별 값은 [building_summary.csv](building_summary.csv), 효과·CI·최악 건물은 [paired_building_effects.json](paired_building_effects.json)과 [macro_scores.csv](macro_scores.csv)에 있다.

## 자원과 실제 실행

{costs}

[확인] 모든 군의 trainable 수는147456으로 파라미터 절감은0%다. 추가 fit당 실제 wall에는 모델 로딩·보존 통계·예측·파일 저장이 포함되며 gradient 시간과 다르다. 선택 checkpoint까지의 gradient 시간은 배포 adaptation의 일부 비용이지 초기화·통계·최종 추론을 모두 포함한 지연이 아니다. 선택 비용은 macro_scores에, 전체 연구 비용은 resources.csv에 있다. Calibration metadata의 시간은 history/method별 첫 호출만 보존하므로 전체 반복 통계 비용이라고 합산하지 않았다. 후보가 더 빠르거나 메모리가 적다고 사전 가정하지 않는다.

GPU monitor {monitor['samples']} samples, 최소 free {monitor['minimum_free_MiB']} MiB, 비승인 외부 compute {monitor['unapproved_external_compute_samples']} samples, free<1GiB {monitor['below_1GiB_samples']} samples. RustDesk만 기존 사용자 승인 예외다. 원시 monitor는 gpu_*.jsonl에 보존한다. 준비의 syntax 수정1회는 본학습 전이며 preparation_repair.json에 기록했다. 실제 오류·미실행은 [execution_errors.json](execution_errors.json), [UNEXECUTED.md](UNEXECUTED.md)에 있다.

## 독립 검산과 재현 범위

[확인] 독립 scalar 재계산 {validation.get('scalar_values','미완료')}개 값, 최대 절대차 {validation.get('max_abs_difference','미완료')}. 이전 {validation.get('old_files_unchanged','미완료')}개 파일 hash 불변. 실제 checkpoint {replay.get('checkpoints','미완료')}개, 이력·최종 예측 {replay.get('queries','미완료')}개를 fit마다 새 모델에서 복원해 exact equality 검사했다. GPU 검산 optimizer updates=0. CPU 수식/gradient 검사, 실제모델 smoke, 동결/finite/step0/poison 검사와 데이터·소스·모델 seal을 함께 검증했다. 검산 오류: {error or '없음'}.

[확인] GitHub에는 코드·프로토콜·점수·hash manifest를 공개한다. 원자료, npz 예측, pt 가중치는 ignored 로컬 cache에 남아 있으므로 GitHub만으로 수치 replay가 가능한 완전 배포물이 아니다. 새 검산 코드는 실험 소스와 분리했으며 최종 hash를 검산 기록에 남겼다.

## 신규성 한계와 남은 구멍

[확인] 비등방 정규화와 사전학습점 보존은 알려진 원리다. [L2-SP](https://proceedings.mlr.press/v80/li18a.html), [TILDE-Q](https://arxiv.org/abs/2210.15050), [Meta-LoRA](https://arxiv.org/html/2608.12389v1)와의 경계 및 나머지 직접 읽은 선행은 [LITERATURE_BOUNDARY.md](LITERATURE_BOUNDARY.md)에 정리했다. 특정 잔차 배분 식이 동일하다는 확인과 최초성 입증 모두 현재 없다. 성능이 좋아도 UNRESOLVED를 임의 승격하지 않는다.

[추정] H3의 두 과거 창에서 잔차 일관성을 추정하는 것은 불안정할 수 있다. 다음날 운영이 바뀌면 과거 일관성이 미래 편향 수정 방향을 보장하지 않는다. 이 실행은 경쟁 설명을 모두 인과 분리하지 못한다. 건물6개/원점1개/공개 원천1개, 동일 site 가능성, 전체기간 연속 관측 필터 재사용, foundation pretraining 중복 미확인은 일반화 한계다. Bootstrap은 건물 단위2000회이며 작은 표본의 정교한 모집단 추정으로 해석하지 않는다.

[확인] 최종 투자 판정은 **{decision}**, 다음 결정은 **{nextdecision}**. 계약의 판정 기준·학습률·데이터·후보를 결과에 맞춰 바꾸지 않았다. 이번 run에서 새 후보를 자동 생성하거나 추가 학습하지 않는다.
"""
    (OUT/'REPORT.md').write_text(text)
    save(OUT/'final_status.json',dict(execution=execution,predictive=predictive,novelty=novelty,topic_decision=decision,actual_main_fits=inv['main_attempts'],actual_main_updates=inv['main_updates'],unexecuted=inv['unexecuted_or_incomplete'],error=error))
    print((OUT/'FINAL_DECISION.md').read_text(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['cpu','gpu','report','all']);a=p.parse_args()
    try:
        if a.command in ['cpu','all']:verify_cpu()
        if a.command in ['gpu','all']:verify_gpu()
        if a.command in ['report','all']:report()
    except BaseException as exc:
        save(OUT/'finalization_error.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
        if a.command=='all':report(valid=False,error=repr(exc))
        raise
