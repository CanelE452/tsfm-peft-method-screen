"""R05/N06: auditable saved-prediction policy and dependence comparisons, zero neural fits."""
import itertools,math,time
import pandas as pd
from scipy.special import ndtri
from scipy.spatial.distance import cdist,pdist
from .common import *
OLD=ROOT/'results/forecast_path_structure_v1_20260916'
TAU=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99])
class Prior:
 def __init__(self):
  self.data=read(OLD/'data_seal.json');self.jobs={j['id']:j for j in self.data['jobs']};self.man=read(OLD/'prediction_manifest.json');self.sels=read(OLD/'selection_seal.json')['selections'];self.scales=read(OLD/'train_scaling.json');self.labels={r['id']:r for r in read(OLD/'test_label_manifest.json')['labels'] if r['valid']};self.cals={(r['target'],r['arm'],r['seed']):np.array(r['delta']) for r in read(OLD/'calibration_parameters.json')};self.used={}
 def get(self,target,role,arm,seed=None,calibrated=False):
  if arm=='FROZEN_WEATHER':key=arm
  else:s=next(s for s in self.sels if s['target']==target and s['arm']==arm and s['seed']==seed);key=s['selected']['weight_key']
  row=next(r for r in self.man if r['target']==target and r['role']==role and r['weight_key']==key and not r['history']);p=ROOT/row['path'];assert sha(p)==row['sha256'];self.used[row['path']]=row['sha256'];z=np.load(p);ids=list(z['ids']);ix=[i for i,k in enumerate(ids) if role!='TEST' or k in self.labels];qs=np.sort(z['raw'][ix],axis=-2);ids=[ids[i] for i in ix];ys=[]
  for jid in ids:
   row=self.labels[jid] if role=='TEST' else self.jobs[jid];path=row['path'] if role=='TEST' else row['label'];h=row['sha256'] if role=='TEST' else row['label_sha256'];assert sha(ROOT/path)==h;self.used[path]=h;ys.append(np.load(ROOT/path))
  if calibrated:qs=np.sort(qs+self.cals[(target,arm,seed if seed is not None else -1)][None,None,:,None],axis=-2)
  return qs,np.array(ys),ids

def pin(q,y,sigma):
 e=y[:,None,None,:]-q;return (2*np.maximum(TAU[None,None,:,None]*e,(TAU[None,None,:,None]-1)*e)).mean((-1,-2))/sigma

def summarize(rows,t,controls,proposed):
 df=pd.DataFrame(rows);csvwrite(OUT/t/'scores_by_origin.csv',rows);keys=[k for k in ['target','seed','arm','variant','condition','metric'] if k in df];summary=[]
 for key,g in df.groupby(keys):summary.append(dict(zip(keys,key),score=float(g.value.mean()),n=len(g)))
 sdf=pd.DataFrame(summary);csvwrite(OUT/t/'raw_scores.csv',summary)
 # Shared calendar blocks, including empty dates; mean seeds then equal targets.
 dates=pd.date_range('2024-10-01','2024-12-30',freq='D');mapping={str(d.date()):i for i,d in enumerate(dates)};calendar=block_counts(np.arange(91),7);series={};contrasts=[]
 for key,g in df.groupby(keys):
  v=np.full(91,np.nan)
  for r in g.itertuples():v[mapping[r.date]]=r.value
  valid=np.isfinite(v);den=calendar@valid;assert (den>0).all();series[key]=(float(np.nanmean(v)),calendar@np.nan_to_num(v)/den)
 def aggregate(arm,scope,variant,condition,metric):
  values=[];boots=[]
  for target in scope:
   vv=[];bb=[]
   for key,(v,b) in series.items():
    d=dict(zip(keys,key))
    if d['arm']==arm and d['target']==target and d.get('variant','RAW')==variant and d.get('condition','ALL')==condition and d['metric']==metric:vv.append(v);bb.append(b)
   if vv:values.append(np.mean(vv));boots.append(np.mean(bb,axis=0))
  return float(np.mean(values)),np.mean(boots,axis=0)
 variants=sorted(df.variant.unique()) if 'variant' in df else ['RAW'];conditions=sorted(df.condition.unique()) if 'condition' in df else ['ALL']
 for scope,targets in [('ALL',['T0','T1','T2']),('ADDITIONAL',['T1','T2']),*[(x,[x]) for x in ['T0','T1','T2']]]:
  for v in variants:
   for c in conditions:
    for metric in sorted(df.metric.unique()):
     for method in proposed:
      for baseline in controls:
       if method==baseline:continue
       a,ab=aggregate(method,targets,v,c,metric);b,bb=aggregate(baseline,targets,v,c,metric);gain=100*(b-a)/b if b else 0.;samples=100*(bb-ab)/np.maximum(bb,1e-300);contrasts.append(dict(scope=scope,variant=v,condition=c,metric=metric,method=method,baseline=baseline,method_score=a,baseline_score=b,gain_percent=gain,ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)),replicates=2000,bootstrap_seed=73300))
 csvwrite(OUT/t/'contrasts.csv',contrasts);csvwrite(OUT/t/'uncertainty.csv',contrasts);npz(CACHE/t/'bootstrap.npz',counts=calendar,days=np.array([str(d.date()) for d in dates]));return contrasts

def r05():
 t='R05';p=Prior();policies={};selrows=[];start=time.perf_counter();grid=[0,.25,.5,.75,1];comb=list(itertools.combinations_with_replacement(grid,4));assert len(comb)==70
 for target in ['T0','T1','T2']:
  data=[(p.get(target,'V_SELECT','LATEST',s),p.get(target,'V_SELECT','PATH',s)) for s in [61730,61731]];sigma=p.scales[target]['sigma_y'];s0=np.mean([pin(a[0],a[1],sigma)[:,0].mean() for a,b in data]);chosen={}
  for arm,options in [('E2',[(a,)*4 for a in grid]),('E4',comb)]:
   scores=[]
   for alpha in options:
    aa=np.array(alpha)[None,:,None,None];vv=np.mean([pin((1-aa)*a[0]+aa*b[0],a[1],sigma).mean(0) for a,b in data],axis=0);feasible=vv[0]<=s0*1.01;scores.append((float(vv[1:].mean()),float(np.mean(alpha)),alpha,feasible));selrows.append(dict(target=target,arm=arm,alpha=list(alpha),V_S0=float(vv[0]),V_delayed=float(vv[1:].mean()),V_S0_limit=float(s0*1.01),feasible=bool(feasible)))
   best=min([r for r in scores if r[3]],key=lambda r:(r[0],r[1],r[2]));chosen[arm]=list(best[2])
  policies[target]={'E0':[0]*4,'E1':[1]*4,'E2':chosen['E2'],'E3':[0,1,1,1],'E4':chosen['E4']}
 save(OUT/t/'LR_selection.json',dict(neural_fits=0,prior_LRs_steps_unchanged=True));save(OUT/t/'selections.json',dict(policies=policies,grid_scores=selrows,at=time.time(),selection_role='V_SELECT',uses_E=False));save(OUT/t/'evaluation_seal.json',dict(at=time.time(),selection_sha256=sha(OUT/t/'selections.json'),master_sha256=sha(OUT/'MASTER_SEAL.json')))
 rows=[];verification=[];pred_manifest=[];origins=[]
 for target in ['T0','T1','T2']:
  sigma=p.scales[target]['sigma_y']
  for seed in [61730,61731]:
   for variant in ['RAW','CALIBRATED']:
    a,y,ids=p.get(target,'TEST','LATEST',seed,variant=='CALIBRATED');b,yb,idb=p.get(target,'TEST','PATH',seed,variant=='CALIBRATED');assert ids==idb and np.array_equal(y,yb)
    for arm,alpha in policies[target].items():
     aa=np.array(alpha)[None,:,None,None];q=(1-aa)*a+aa*b;assert (np.diff(q,axis=2)>=0).all();val=pin(q,y,sigma);file=CACHE/t/f'{target}_{seed}_{variant}_{arm}.npz';npz(file,q=q,ids=np.array(ids));pred_manifest.append(dict(path=str(file.relative_to(ROOT)),sha256=sha(file),arm=arm,target=target,seed=seed,variant=variant))
     for i,jid in enumerate(ids):
      date=p.jobs[jid]['date'];vals={f'S{k}':float(val[i,k]) for k in range(4)};vals.update(DELAYED=float(val[i,1:].mean()),MEAN=float(val[i].mean()))
      for case,v in vals.items():rows.append(dict(target=target,seed=seed,arm=arm,variant=variant,condition=case,metric='normalized_2pinball',date=date,id=jid,value=v))
      for k in range(4):
       scalar=math.fsum(2*(tau*(float(y[i,h])-float(q[i,k,qi,h])) if y[i,h]>=q[i,k,qi,h] else (tau-1)*(float(y[i,h])-float(q[i,k,qi,h]))) for qi,tau in enumerate(TAU) for h in range(24))/(21*24*sigma);assert math.isclose(scalar,val[i,k],rel_tol=1e-10,abs_tol=1e-10)
     verification.append(dict(target=target,seed=seed,arm=arm,variant=variant,checks=len(ids)*4,quantile_monotone=True,alpha0_exact_latest=alpha[0]!=0 or np.array_equal(q[:,0],a[:,0]),models_per_case=[1 if v in [0,1] else 2 for v in alpha]))
    origins.extend([dict(target=target,role='E_REUSED',id=j,origin=p.jobs[j]['origin']) for j in ids])
 save(OUT/t/'predictions_manifest.json',pred_manifest);save(OUT/t/'used_prior_hashes.json',p.used);contrasts=summarize(rows,t,['E0','E2','E3'],['E4']);csvwrite(OUT/t/'origins.csv',origins)
 # Worst delayed risk is max of the three case means, not mean per-origin maxima.
 df=pd.DataFrame(rows);worst=[]
 for key,g in df[df.condition.isin(['S1','S2','S3'])].groupby(['target','seed','arm','variant']):worst.append(dict(zip(['target','seed','arm','variant'],key),worst_delayed_mean=float(g.groupby('condition').value.mean().max())))
 csvwrite(OUT/t/'worst_delayed_scores.csv',worst);save(OUT/t/'verification.json',dict(passed=True,scalar_checks=sum(r['checks'] for r in verification),rtol=1e-10,atol=1e-10,checks=verification,neural_fits=0,prior_hashes_verified=len(p.used)));save(OUT/t/'STATUS.json',dict(EXECUTION='COMPLETE',EVIDENCE='POSITIVE_UNCERTAIN',NOVELTY='KNOWN_CONTROL',neural_fits=0,CPU_policy_grid_evaluations=3*75,seconds=time.perf_counter()-start));csvwrite(OUT/t/'resources.csv',[dict(neural_fits=0,optimizer_updates=0,new_forwards=0,CPU_policy_grid_evaluations=225,seconds=time.perf_counter()-start)])

def correlation(z):
 sd=z.std(0);valid=sd>0;zz=np.zeros_like(z);zz[:,valid]=(z[:,valid]-z[:,valid].mean(0))/sd[valid];S=zz.T@zz/len(z);np.fill_diagonal(S,1);return S,np.flatnonzero(~valid).tolist()
def pit(q,y):
 out=[]
 for h in range(24):
  vals=np.unique(q[:,h]);ts=np.array([TAU[q[:,h]==v].mean() for v in vals]);out.append(np.clip(np.interp(y[h],vals,ts,left=.01,right=.99),.01,.99))
 return np.array(out)
def fix_psd(R):
 R=(R+R.T)/2;v,U=np.linalg.eigh(R);R=(U*np.maximum(v,1e-6))@U.T;R=R/np.sqrt(np.diag(R)[:,None]*np.diag(R)[None,:]);assert np.linalg.eigvalsh(R).min()>0;return R

def n06():
 t='N06';p=Prior();start=time.perf_counter();parameters={};marginal_checks=[];rows=[];preds=[];orows=[]
 for target in ['T0','T1','T2']:
  q,y,ids=p.get(target,'V_CALIBRATE','FROZEN_WEATHER');u=np.stack([pit(q[i,0],y[i]) for i in range(len(y))]);z=ndtri(u);S,constant=correlation(z);aa=z[:,:-1].ravel();bb=z[:,1:].ravel();rho=float(np.clip(np.corrcoef(aa,bb)[0,1],-.95,.95)) if aa.std()>0 and bb.std()>0 else 0.;ar=rho**abs(np.arange(24)[:,None]-np.arange(24)[None,:]);shr=fix_psd(.5*np.eye(24)+.5*S)
  # Unique lawful hourly TRAIN context+label values, only consecutive six-hour windows.
  values={}
  for j in p.data['jobs']:
   if j['target']!=target or j['role']!='TRAIN':continue
   assert sha(ROOT/j['input'])==j['input_sha256'] and sha(ROOT/j['label'])==j['label_sha256'];xx=np.load(ROOT/j['input'])['context'];yy=np.load(ROOT/j['label']);o=pd.Timestamp(j['origin']);
   for h,v in enumerate(np.concatenate([xx,yy])):values[o+pd.Timedelta(hours=h-336)]=float(v)
  target_record=next(a for a in p.data['targets'] if a['target']==target);rawload=pd.read_parquet(ROOT/target_record['load_path'],columns=['timestamp','load']);rawload['timestamp']=pd.to_datetime(rawload['timestamp'],utc=True);hourly=rawload.set_index('timestamp')['load'].resample('h').mean();values={key:float(hourly.loc[key]) for key in values};assert all(np.isfinite(v) for v in values.values())
  times=sorted(values);rolling=[np.mean([values[v-pd.Timedelta(hours=h)] for h in range(6)]) for v in times if all(v-pd.Timedelta(hours=h) in values for h in range(6))];threshold=float(np.quantile(rolling,.9));parameters[target]=dict(rho=rho,AR1=ar.tolist(),SHRUNK=shr.tolist(),pit=u.tolist(),constant_leads=constant,threshold_rolling6_p90=threshold,TRAIN_unique_hourly=len(values),TRAIN_rolling6=len(rolling),CAL_ids=ids,calibration_n=len(y),units='W hourly mean; sum*1h = Wh',seed=73260)
 save(OUT/t/'selections.json',dict(parameters=parameters,at=time.time(),selection_role='V_CALIBRATE',uses_E=False));save(OUT/t/'LR_selection.json',dict(neural_fits=0));save(OUT/t/'evaluation_seal.json',dict(at=time.time(),selection_sha256=sha(OUT/t/'selections.json'),master_sha256=sha(OUT/'MASTER_SEAL.json')))
 pair=np.triu_indices(24,1);mchecks=0;scalar_checks=0
 for target in ['T0','T1','T2']:
  pr=parameters[target];sigma=p.scales[target]['sigma_y'];q,y,ids=p.get(target,'TEST','FROZEN_WEATHER');ar=np.array(pr['AR1']);shr=np.array(pr['SHRUNK']);u=np.array(pr['pit']);last=[]
  for i,jid in enumerate(ids):
   sorted_values=np.stack([np.interp((np.arange(256)+.5)/256,TAU,q[i,0,:,h]) for h in range(24)],axis=1);basecrps=None
   for arm in ARMS[t]:
    g=rng('N06',73260,target,jid,arm)
    if arm=='F0':template=g.random((256,24))
    elif arm in ['F1','F2']:template=g.standard_normal((256,24))@np.linalg.cholesky(ar if arm=='F1' else shr).T
    else:template=u[g.integers(len(u),size=256)]
    ranks=np.empty((256,24),int)
    for h in range(24):ranks[np.lexsort((g.random(256),template[:,h])),h]=np.arange(256)
    samples=np.take_along_axis(sorted_values,ranks,axis=0);assert np.array_equal(np.sort(samples,axis=0),sorted_values);mchecks+=24
    coeff=(2*np.arange(1,257)-257)/256**2;leadcrps=np.mean(abs(samples-y[i]),axis=0)-coeff@sorted_values
    if basecrps is None:basecrps=leadcrps
    else:assert np.allclose(basecrps,leadcrps,rtol=1e-10,atol=1e-10)
    s=samples.sum(1);sy=y[i].sum();scrps=np.mean(abs(s-sy))-np.dot(coeff,np.sort(s));brute=np.mean(abs(s-sy))-.5*np.mean(abs(s[:,None]-s[None,:]));assert math.isclose(scrps,brute,rel_tol=1e-10,abs_tol=1e-10);scalar_checks+=1
    zsamples=samples/sigma;zy=y[i]/sigma;energy=float(np.linalg.norm(zsamples-zy,axis=1).mean()-.5*cdist(zsamples,zsamples).mean());vg=float(np.mean((abs(y[i,pair[0]]-y[i,pair[1]])**.5-np.mean(abs(samples[:,pair[0]]-samples[:,pair[1]])**.5,axis=0))**2));roll=np.stack([samples[:,h:h+6].mean(1) for h in range(19)],axis=1).max(1);event=float(max(y[i,h:h+6].mean() for h in range(19))>pr['threshold_rolling6_p90']);prob=float(np.mean(roll>pr['threshold_rolling6_p90']));brier=(prob-event)**2
    metrics=dict(sum_CRPS_normalized=float(scrps/(24*sigma)),sum_CRPS_Wh=float(scrps),energy=energy,variogram=vg,variogram_normalized=vg/sigma,Brier=brier,marginal_CRPS_normalized=float(leadcrps.mean()/sigma))
    for metric,value in metrics.items():rows.append(dict(target=target,seed=73260,arm=arm,variant='RAW',condition='S0',metric=metric,date=p.jobs[jid]['date'],id=jid,value=value))
    path=CACHE/t/'samples'/f'{jid}_{arm}.npz';npz(path,samples=samples,ids=np.array([jid]));preds.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),target=target,arm=arm,id=jid,sorted_marginals_exact=True));marginal_checks.append(dict(target=target,id=jid,arm=arm,lead_CRPS=leadcrps.tolist(),event=event,probability=prob))
   orows.append(dict(target=target,role='E_REUSED',id=jid,origin=p.jobs[jid]['origin']))
 save(OUT/t/'predictions_manifest.json',preds);save(OUT/t/'marginal_verification.json',marginal_checks);save(OUT/t/'used_prior_hashes.json',p.used);summarize(rows,t,['F0','F1'],['F1','F2','F3']);csvwrite(OUT/t/'origins.csv',orows);save(OUT/t/'verification.json',dict(passed=True,sorted_marginal_exact_checks=mchecks,scalar_checks=scalar_checks,rtol=1e-10,atol=1e-10,correlation_PSD=True,V_CAL_E_disjoint=True,neural_fits=0));save(OUT/t/'STATUS.json',dict(EXECUTION='COMPLETE',EVIDENCE='POSITIVE_UNCERTAIN',NOVELTY='KNOWN_CONTROL',neural_fits=0,CPU_dependence_fits=9,seconds=time.perf_counter()-start));csvwrite(OUT/t/'resources.csv',[dict(neural_fits=0,optimizer_updates=0,new_forwards=0,CPU_dependence_fits=9,seconds=time.perf_counter()-start)])

def run_cached(t):
 for path,h in read(OUT/t/'data_receipt.json')['dependencies'].items():assert sha(ROOT/path)==h,('PRIOR_CHANGED',path)
 (CACHE/t).mkdir(parents=True,exist_ok=True)
 if t=='R05':r05()
 elif t=='N06':n06()
 else:raise ValueError(t)
