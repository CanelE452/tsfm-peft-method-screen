"""TEST labels are opened only after prediction sealing. Independent scalar verification."""
import math,subprocess,collections
import pandas as pd
from .common import *
from .data import label,InputsInvalid

def score_all():
 seal=read(OUT/'test_prediction_seal.json');assert sha(OUT/'prediction_manifest.json')==seal['manifest_sha256']
 ev=read(OUT/'evaluation_seal.json');assert sha(OUT/'evaluation_seal.json')==seal['evaluation_sha256']
 assert seal['at']>=ev['at'];assert sha(OUT/'calibration_parameters.json')==ev['calibration_sha256']
 for p,h in ev['source_files'].items():assert sha(p)==h,('SOURCE_CHANGED_BEFORE_SCORING',p)
 ds=read(OUT/'data_seal.json');scales=read(OUT/'train_scaling.json');preds=read(OUT/'prediction_manifest.json');cals={r['id']:np.array(r['delta']) for r in read(OUT/'calibration_parameters.json')};jobs={j['id']:j for j in ds['jobs']};ys={};valid=[]
 for t in ds['targets']:
  # This is the first evaluation label extraction, AFTER all model predictions have hashes.
  load=pd.read_parquet(ROOT/t['load_path'],columns=['timestamp','load']);load['timestamp']=pd.to_datetime(load.timestamp,utc=True)
  for j in ds['jobs']:
   if j['target']!=t['target'] or j['role']!='TEST':continue
   try:
    y=label(load,pd.Timestamp(j['origin']));ys[j['id']]=y;p=CACHE/'test_labels'/f"{j['id']}.npy";p.parent.mkdir(exist_ok=True);np.save(p,y);valid.append(dict(id=j['id'],target=j['target'],origin=j['origin'],valid=True,path=str(p.relative_to(ROOT)),sha256=sha(p)))
   except InputsInvalid:valid.append(dict(id=j['id'],target=j['target'],origin=j['origin'],valid=False,reason='TARGET_MISSING_COMMON_EXCLUSION'))
 save(OUT/'test_label_manifest.json',dict(at=time.time(),prediction_seal_sha256=sha(OUT/'test_prediction_seal.json'),labels=valid))
 rows=[];leadrows=[]
 for r in ev['roles']:
  p=next(p for p in preds if p['role']=='TEST' and p['target']==r['target'] and p['weight_key']==r['weight_key'] and p['history']==r['history']);assert sha(ROOT/p['path'])==p['sha256'];z=np.load(ROOT/p['path']);sigma=scales[r['target']]['sigma_y']
  for variant in (['RAW','CALIBRATED'] if r['policy']=='SELECTED' else ['RAW']):
   leads={k:[] for k in range(4)}
   for i,jid in enumerate(z['ids']):
    if jid not in ys:continue
    q=np.sort(z['raw'][i],axis=-2)
    if variant=='CALIBRATED':q=apply_cal(q,cals[r['id']])
    y=ys[jid];j=jobs[jid]
    for k in range(4):
     met=metric(q[k],y,sigma);rows.append(dict(target=r['target'],arm=r['arm'],seed=r['seed'],policy=r['policy'],variant=variant,case=f'S{k}',id=jid,date=j['date'],month=int(j['date'][5:7]),prediction=p['path'],prediction_index=i,**met))
     e=y[None]-q[k];leads[k].append((2*np.maximum(TAUS[:,None]*e,(TAUS[:,None]-1)*e)).mean(0)/sigma)
   for k,l in leads.items():
    if l:
     for h,v in enumerate(np.mean(l,axis=0)):leadrows.append(dict(target=r['target'],arm=r['arm'],seed=r['seed'],policy=r['policy'],variant=variant,case=f'S{k}',hour=h,primary=float(v)))
 # Last-day is point-only; it is never silently treated as 21-quantile probabilistic output.
 last=[]
 for jid,y in ys.items():
  x=np.load(ROOT/jobs[jid]['input']);d=x['context'][-24:].astype(float)-y
  last.append(dict(target=jobs[jid]['target'],id=jid,date=jobs[jid]['date'],RMSE=float(np.sqrt(np.mean(d*d))),MAE=float(np.mean(abs(d)))))
 csvwrite(OUT/'last_day_scores.csv',last);csvwrite(OUT/'scores_by_origin.csv',rows);csvwrite(OUT/'per_lead_hour.csv',leadrows)
 df=pd.DataFrame(rows);keys=['target','arm','seed','policy','variant','case'];agg=[]
 for key,g in df.groupby(keys,sort=False):
  agg.append(dict(zip(keys,key),n_origins=len(g),primary=float(g.primary.mean()),raw_2pinball=float(g.raw_2pinball.mean()),RMSE=float(np.sqrt(np.mean(g.RMSE**2))),MAE=float(g.MAE.mean()),coverage80=float(g.coverage80.mean()),width80=float(g.width80.mean())))
 csvwrite(OUT/'scores_by_target_seed_case.csv',agg)
 csvwrite(OUT/'raw_and_calibrated.csv',[r for r in agg if r['policy']=='SELECTED'])
 csvwrite(OUT/'monthly_scores.csv',[dict(zip(keys+['month'],key),primary=float(g.primary.mean()),n_origins=len(g)) for key,g in df.groupby(keys+['month'],sort=False)])
 save(OUT/'scoring_status.json',dict(at=time.time(),score_rows=len(rows),test_labels_read_after_prediction_seal=True,valid_by_target={t['target']:sum(r['valid'] and r['target']==t['target'] for r in valid) for t in ds['targets']},invalid_labels=sum(not r['valid'] for r in valid)))
 return df

def contrasts(df):
 dates=pd.date_range('2024-10-01','2024-12-30',freq='D');N=len(dates);datemap={d.date().isoformat():i for i,d in enumerate(dates)}
 # A single calendar resampling schedule across all simultaneous targets, arms, seeds and cases.
 blocks=[np.arange(i,min(i+7,N)) for i in range(0,N,7)];rng=np.random.default_rng(61916);counts=[]
 for _ in range(2000):
  ix=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),size=math.ceil(N/7)+1)])[:N];counts.append(np.bincount(ix,minlength=N))
 counts=np.array(counts,dtype=float);np.savez_compressed(CACHE/'bootstrap_calendar.npz',counts=counts,dates=np.array(list(datemap)))
 targets=sorted(df.target.unique());series={}
 def value(arm,policy,variant,target,seed):
  sd=-1 if arm.startswith('FROZEN') else seed;po='SELECTED' if arm.startswith('FROZEN') else policy
  key=(arm,po,variant,target,sd)
  if key not in series:
   sub=df[(df.arm==arm)&(df.policy==po)&(df.variant==variant)&(df.target==target)&(df.seed==sd)];a=np.full((4,N),np.nan)
   for r in sub.itertuples():a[int(r.case[1]),datemap[r.date]]=r.primary
   assert np.isfinite(a).any();mask=np.isfinite(a);num=np.nan_to_num(a)@counts.T;den=mask.astype(float)@counts.T
   assert (den>0).all();series[key]=(np.nanmean(a,axis=1),num/den)
  return series[key]
 comparisons=[('PATH','LATEST'),('PATH','DROP'),('PATH','POINT')]+[(a,'FROZEN_WEATHER') for a in ARMS]
 scopes={**{t:[t] for t in targets},'ALL':targets,'ADDITIONAL':[t for t in targets if t!='T0']}
 out=[];seedout=[]
 for policy,variant in [('SELECTED','RAW'),('SELECTED','CALIBRATED'),('FIXED512','RAW')]:
  for a,b in comparisons:
   for scope,ts in scopes.items():
    if not ts:continue
    for case in ['S0','S1','S2','S3','MEAN','WORST']:
     vals=[];boots=[]
     for arm in [a,b]:
      tv=[];tb=[]
      for t in ts:
       sv=[];sb=[]
       for seed in SEEDS:
        v,boot=value(arm,policy,variant,t,seed)
        if case=='MEAN':v=v.mean();boot=boot.mean(0)
        elif case=='WORST':pass
        else:k=int(case[1]);v=v[k];boot=boot[k]
        sv.append(v);sb.append(boot)
       
       if case=='WORST':tv.append(np.mean(sv,axis=0).max());tb.append(np.mean(sb,axis=0).max(0))
       else:tv.append(np.mean(sv));tb.append(np.mean(sb,axis=0))
      vals.append(float(np.mean(tv)));boots.append(np.mean(tb,axis=0))
     gain=100*(vals[1]-vals[0])/vals[1];bs=100*(boots[1]-boots[0])/boots[1]
     out.append(dict(scope=scope,policy=policy,variant=variant,case=case,method=a,baseline=b,method_score=vals[0],baseline_score=vals[1],gain_percent=gain,ci_low=float(np.quantile(bs,.025)),ci_high=float(np.quantile(bs,.975)),bootstrap_seed=61916,replicates=2000))
   for t in targets:
    for seed in SEEDS:
     va,_=value(a,policy,variant,t,seed);vb,_=value(b,policy,variant,t,seed)
     for case in ['S0','S1','S2','S3','MEAN','WORST']:
      fun=(lambda x:x.mean()) if case=='MEAN' else (lambda x:x.max()) if case=='WORST' else (lambda x:x[int(case[1])])
      aa=float(fun(va));bb=float(fun(vb));seedout.append(dict(target=t,seed=seed,policy=policy,variant=variant,case=case,method=a,baseline=b,method_score=aa,baseline_score=bb,gain_percent=100*(bb-aa)/bb))
 csvwrite(OUT/'contrasts.csv',out);csvwrite(OUT/'uncertainty.csv',out);csvwrite(OUT/'contrasts_by_seed.csv',seedout)
 save(OUT/'bootstrap_manifest.json',dict(path=str((CACHE/'bootstrap_calendar.npz').relative_to(ROOT)),sha256=sha(CACHE/'bootstrap_calendar.npz'),blocks=[b.tolist() for b in blocks],replicates=2000,seed=61916,rule='shared nonoverlapping 7-day calendar blocks; random blocks concatenate/truncate to calendar length; missing dates masked with per-target/seed denominators',scope='descriptive conditional on observed targets/seeds; not population uncertainty or multiple-test correction'))
 return out

def verify():
 df=pd.read_csv(OUT/'scores_by_origin.csv');ds=read(OUT/'data_seal.json');scales=read(OUT/'train_scaling.json');jobs={j['id']:j for j in ds['jobs']};labels={r['id']:r for r in read(OUT/'test_label_manifest.json')['labels'] if r['valid']};cal={(r['target'],r['arm'],r['seed']):np.array(r['delta']) for r in read(OUT/'calibration_parameters.json')};predcache={};ycache={};maxerr=collections.defaultdict(float);checks=0
 for r in df.itertuples():
  if r.prediction not in predcache:predcache[r.prediction]=np.load(ROOT/r.prediction)['raw']
  if r.id not in ycache:
   lr=labels[r.id];assert sha(ROOT/lr['path'])==lr['sha256'];ycache[r.id]=np.load(ROOT/lr['path'])
  q=np.sort(predcache[r.prediction][r.prediction_index,int(r.case[1])],axis=0)
  if r.variant=='CALIBRATED':q=np.sort(q+cal[(r.target,r.arm,r.seed)][:,None],axis=0)
  y=ycache[r.id];sigma=scales[r.target]['sigma_y'];loss=[]
  for qi,tau in enumerate(TAUS):
   for h in range(24):
    e=float(y[h])-float(q[qi,h]);loss.append(2*(tau*e if e>=0 else (tau-1)*e))
  pin=math.fsum(loss)/(21*24);med=int(np.flatnonzero(TAUS==.5)[0]);d=[float(q[med,h])-float(y[h]) for h in range(24)];lo=int(np.flatnonzero(TAUS==.1)[0]);hi=int(np.flatnonzero(TAUS==.9)[0])
  scalar=dict(primary=pin/sigma,raw_2pinball=pin,RMSE=math.sqrt(math.fsum(v*v for v in d)/24),MAE=math.fsum(abs(v) for v in d)/24,coverage80=sum(float(q[lo,h])<=float(y[h])<=float(q[hi,h]) for h in range(24))/24,width80=math.fsum(float(q[hi,h])-float(q[lo,h]) for h in range(24))/24)
  for k,v in scalar.items():
   old=float(getattr(r,k));assert math.isclose(v,old,rel_tol=1e-10,abs_tol=1e-10),(r.id,k,v,old);maxerr[k]=max(maxerr[k],abs(v-old));checks+=1
 # Recompute every V score and selection from saved predictions, not from fit log values.
 fitchecks=0;fits=read(OUT/'fits.json');sele=read(OUT/'selection_seal.json')['selections'];vmap={}
 for f in fits:
  cpvals=[]
  for cp in f['checkpoints']:
   assert sha(ROOT/cp['path'])==cp['sha256'];z=np.load(ROOT/cp['prediction']);scores=[]
   for i,jid in enumerate(z['ids']):
    j=jobs[jid];assert j['role']=='V_SELECT';y=np.load(ROOT/j['label']);sigma=scales[j['target']]['sigma_y']
    for k in range(4):
     q=np.sort(z['raw'][i,k],axis=0);pieces=[]
     for qi,tau in enumerate(TAUS):
      for h in range(24):
       d=float(y[h])-float(q[qi,h]);pieces.append(2*(tau*d if d>=0 else (tau-1)*d))
     scores.append(math.fsum(pieces)/(504*sigma))
   mean=math.fsum(scores)/len(scores);assert math.isclose(mean,cp['validation_primary'],rel_tol=1e-10,abs_tol=1e-10);cpvals.append((mean,cp['step']));fitchecks+=1
  vmap[f['id']]=min(cpvals)
 for sel in sele:
  relevant=[f for f in fits if f['target']==sel['target'] and f['arm']==sel['arm']];lrmean={lr:math.fsum(vmap[f['id']][0] for f in relevant if f['lr']==lr)/2 for lr in LRS};lr=min(LRS,key=lambda x:(lrmean[x],x));assert lr==sel['lr'] and sel['selected']['step']==vmap[sel['id']][1]
 # Recompute 21 empirical quantiles with explicit sorted order/interpolation.
 calchecks=0
 for c in read(OUT/'calibration_parameters.json'):
  z=np.load(ROOT/c['prediction']['path']);q=np.sort(z['raw'],axis=-2);y=np.stack([np.load(ROOT/jobs[jid]['label']) for jid in z['ids']]);assert all(jobs[jid]['role']=='V_CALIBRATE' for jid in z['ids'])
  for qi,tau in enumerate(TAUS):
   vals=sorted(float(y[i,h])-float(q[i,k,qi,h]) for i in range(len(y)) for k in range(4) for h in range(24));idx=(len(vals)-1)*tau;lower=math.floor(idx);upper=math.ceil(idx);v=vals[lower]+(vals[upper]-vals[lower])*(idx-lower)
   assert math.isclose(v,c['delta'][qi],rel_tol=1e-10,abs_tol=1e-10);calchecks+=1
 for p in read(OUT/'prediction_manifest.json'):assert sha(ROOT/p['path'])==p['sha256']
 history=read(OUT/'historical_hashes.json')
 for p,h in history.items():assert sha(ROOT/p)==h,('HISTORICAL_CHANGE',p)
 logs=[json.loads(l) for l in (OUT/'update_log.jsonl').read_text().splitlines()];main=[r for r in logs if not r['smoke']];smoke=[r for r in logs if r['smoke']]
 assert len(main)==len(fits)*512 and len(smoke)==len(ds['targets'])*8
 for f in fits:
  rows=[r for r in main if r['id']==f['id']];assert [r['step'] for r in rows]==list(range(1,513));assert len({r['origin_id'] for r in rows})==64
  counts=collections.Counter(r['origin_id'] for r in rows);assert set(counts.values())=={8}
 # Common initial parameters and per-step origin stream within target/seed; LRs do not alter stream.
 for t in ds['targets']:
  for seed in SEEDS:
   group=[f for f in fits if f['target']==t['target'] and f['seed']==seed];assert len(set(f['initial_hash'] for f in group))==1
   streams=[[r['origin_id'] for r in main if r['id']==f['id']] for f in group];assert all(a==streams[0] for a in streams)
 csvwrite(OUT/'train_curves.csv',logs)
 csvwrite(OUT/'resource_usage.csv',[dict(id=f['id'],target=f['target'],arm=f['arm'],seed=f['seed'],lr=f['lr'],updates=f['updates'],optimizer_seconds=f['optimizer_seconds'],fit_seconds=f['seconds'],peak_allocated_bytes=f['peak_allocated_bytes'],peak_reserved_bytes=f['peak_reserved_bytes'],trainable_parameters=f['trainable_parameters']) for f in fits])
 save(OUT/'checkpoint_manifest.json',[dict(id=f['id'],checkpoints=f['checkpoints'],resume_path=str((CACHE/'resume'/f"{f['id']}.pt").relative_to(ROOT)),resume_sha256=sha(CACHE/'resume'/f"{f['id']}.pt")) for f in fits])
 out=dict(status='PASS',scalar_metrics_checked=checks,max_absolute_errors=dict(maxerr),validation_checkpoint_checks=fitchecks,selection_checks=len(sele),calibration_scalar_checks=calchecks,historical_files_preserved=len(history),main_updates=len(main),smoke_updates=len(smoke),new_optimizer_updates=len(main)+len(smoke),all_initializations_matched=True,all_origin_streams_matched=True,all_targets_seen_eight_times=True,contaminated_updates=sum(r['contaminated'] for r in logs),normalized_restore_tolerance=1e-5,scalar_rtol=1e-10,scalar_atol=1e-10)
 save(OUT/'independent_verification.json',out);return out
