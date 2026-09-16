"""Frozen-selection evaluation, condition tables, paired block uncertainty and scalar replay."""
import math,time
import pandas as pd
from .common import *
from .runtime import primary

def spectral_ridge(data,role):
 coef=np.array(data.aux['ridge_coefficients']);out=[]
 for x in data.inputs[role]['context']:
  z=(x-data.mu[:,None])/data.sigma[:,None];p=np.fft.fft(z[:,-48:],norm='ortho');q=np.fft.fft(z[:,-96:-48],norm='ortho');f=np.zeros((4,48),complex)
  for c in range(4):
   for k in range(25):
    features=np.array([1,p[c,k].real,p[c,k].imag,q[c,k].real,q[c,k].imag]);v=features@coef[c,k];f[c,k]=v[0]+1j*v[1]
    if k not in [0,24]:f[c,48-k]=f[c,k].conjugate()
  ff=np.fft.ifft(f,norm='ortho');assert abs(ff.imag).max()<1e-10;out.append(ff.real*data.sigma[:,None]+data.mu[:,None])
 return np.array(out)

def fixed_lead(data,role,base):
 out=[]
 for i,x in enumerate(data.inputs[role]['context']):
  z=(base[i]-data.mu[:,None])/data.sigma[:,None];ctx=(x-data.mu[:,None])/data.sigma[:,None];gg=[]
  for c in range(4):
   b=data.aux['ridge'][c];g=np.full(48,b[0])
   for donor,lag,beta in zip(data.aux['donors'][c],data.aux['lags'][c],b[1:]):g+=beta*np.r_[ctx[donor,-lag:],z[donor,:48-lag]]
   gg.append(g)
  out.append(np.array(gg)*data.sigma[:,None]+data.mu[:,None])
 return np.array(out)

def select_cpu_controls(t,data,selections,frozen):
 rows=[]
 if t not in ['N02','R04','R08']:save(OUT/t/'CPU_selection.json',dict(neural_fits=0,choices=[],ridge_fixed=t=='N07'));return
 basearm={'N02':'B0','R04':'D0','R08':'H0'}[t]
 for s in selections:
  if s['arm']!=basearm:continue
  pp=np.load(ROOT/s['selected']['prediction']);p=pp['pred'];yy=data.labels['V_SELECT'];choices=[];conds=list(pp['conditions'])
  if t=='N02':simple=data.inputs['V_SELECT']['retrieval_future'].reshape(32,4,2,48).mean(2);simple=simple[:,None]
  if t=='R08':simple=fixed_lead(data,'V_SELECT',frozen['V_SELECT'])[:,None]
  for alpha in [0,.25,.5,.75,1]:
   if t=='R04':mixed=p.copy();mixed[:,0,1,:,:24]=(1-alpha)*p[:,0,1,:,:24]+alpha*p[:,0,0,:,24:]
   else:mixed=(1-alpha)*p+alpha*simple
   score=primary(t,mixed,yy,data.sigma,conds);choices.append(dict(alpha=alpha,**score))
  if t=='R04':good=[r for r in choices if r['accuracy']<=s['selected']['accuracy']*1.01];chosen=min(good,key=lambda r:(r['revision'],r['alpha']))
  else:chosen=min(choices,key=lambda r:(r['accuracy'],r['alpha']))
  rows.append(dict(seed=s['seed'],arm='BLEND',base=basearm,alpha=chosen['alpha'],scores=choices))
 save(OUT/t/'CPU_selection.json',dict(neural_fits=0,choices=rows,at=time.time(),selection_role='V_SELECT',uses_E=False))

def score_track(t,data,selections,predictions,frozen,aliases):
 ev=read(OUT/t/'evaluation_seal.json');assert sha(OUT/t/'selections.json')==ev['selections_sha256'];assert sha(OUT/t/'CPU_selection.json')==ev['CPU_sha256'];assert ev['at']<=read(OUT/t/'predictions_manifest.json')['at'];
 # E labels opened only after every selected/fixed prediction is committed to the manifest.
 y=np.load(CACHE/t/'E_DISCOVERY_labels.npz')['y'];origins=data.inputs['E_DISCOVERY']['origins'];bundle={};meta={};conds=None
 for r in predictions:
  assert sha(ROOT/r['path'])==r['sha256'];z=np.load(ROOT/r['path']);assert np.array_equal(z['origins'],origins);conds=list(z['conditions']);key=(r['arm'],r['seed'],r['policy']);bundle[key]=z['pred'];meta[key]=r
 for alias,base in aliases.items():
  for (arm,seed,policy),p in list(bundle.items()):
   if arm==base:bundle[(alias,seed,policy)]=p;meta[(alias,seed,policy)]=dict(alias_of=base)
 cpu=read(OUT/t/'CPU_selection.json')
 for r in cpu['choices']:
  key=(r['base'],r['seed'],'selected')
  if key not in bundle:continue
  p=bundle[key];alpha=r['alpha']
  if t=='N02':simple=data.inputs['E_DISCOVERY']['retrieval_future'].reshape(64,4,2,48).mean(2)[:,None];blend=(1-alpha)*p+alpha*simple
  elif t=='R08':simple=fixed_lead(data,'E_DISCOVERY',frozen['E_DISCOVERY'])[:,None];blend=(1-alpha)*p+alpha*simple
  else:simple=None;blend=p.copy();blend[:,0,1,:,:24]=(1-alpha)*p[:,0,1,:,:24]+alpha*p[:,0,0,:,24:]
  bundle[('BLEND',r['seed'],'selected')]=blend
  if simple is not None:bundle[('RETRIEVAL_MEAN' if t=='N02' else 'FROZEN_RIDGE',r['seed'],'selected')]=simple
 if t=='N07':
  simple=spectral_ridge(data,'E_DISCOVERY')[:,None]
  for seed in [73101,73102]:bundle[('SPECTRAL_RIDGE',seed,'selected')]=simple
 if t in ['R04','R09','R08']:
  p=frozen['E_DISCOVERY'];p=np.stack([p,frozen['E_DISCOVERY_late']],axis=1) if t=='R04' else p
  for seed in [73101,73102]:bundle[('FROZEN',seed,'selected')]=p[:,None]
 bpath=CACHE/t/'evaluation_bundle.npz';npz(bpath,**{'__'.join(map(str,k)):v for k,v in bundle.items()},y=y,origins=origins,conditions=np.array(conds));save(OUT/t/'evaluation_bundle_manifest.json',dict(path=str(bpath.relative_to(ROOT)),sha256=sha(bpath),at=time.time()))
 raw=[];originrows=[];summary=[];checks=0;series={};secondary=[]
 counts=block_counts(origins,672 if t=='N03' else 168);den=counts.sum(1);assert (den>0).all();npz(CACHE/t/'bootstrap.npz',counts=counts,origins=origins)
 for (arm,seed,policy),p in bundle.items():
  main=primary(t,p,y,data.sigma,conds);summary.append(dict(arm=arm,seed=seed,policy=policy,condition='PRIMARY',score=main['accuracy'],revision=main.get('revision')))
  if t=='R04':yy=np.stack([y[:,:,:48],y[:,:,24:72]],1);e=p[:,0]-yy;arr=(e/data.sigma[None,None,:,None])**2;per=arr.mean((1,3));mse=per.mean(0);boot=np.sqrt(counts@per/den[:,None]).mean(1);rev=((p[:,0,1,:,:24]-p[:,0,0,:,24:])/data.sigma[None,:,None])**2;rvper=rev.mean(-1);rb=np.sqrt(counts@rvper/den[:,None]).mean(1);series[(arm,seed,policy,'PRIMARY')]=(main['accuracy'],boot);series[(arm,seed,policy,'REVISION')]=(main['revision'],rb)
  else:
   primv=[];primb=[]
   for ci,cond in enumerate(conds):
    chs=[int(cond.split('_c')[1])] if t=='N01' and cond!='all24' else list(range(4));e=p[:,ci,chs]-y[:,chs];s=data.sigma[chs];per=((e/s[None,:,None])**2).mean(-1);score=float(np.sqrt(per.mean(0)).mean());boot=np.sqrt(counts@per/den[:,None]).mean(1);series[(arm,seed,policy,cond)]=(score,boot);summary.append(dict(arm=arm,seed=seed,policy=policy,condition=cond,score=score,revision=None));use=(cond.startswith(('d6_','d24_')) if t=='N01' else cond in ['delta2','irregular'] if t=='N03' else True)
    if use:primv.append(score);primb.append(boot)
    for cc,c in enumerate(chs):
     rmse=float(np.sqrt(np.mean(e[:,cc]**2)));mae=float(np.mean(abs(e[:,cc])));raw.append(dict(arm=arm,seed=seed,policy=policy,condition=cond,channel=c,normalized_RMSE=rmse/data.sigma[c],raw_RMSE=rmse,raw_MAE=mae))
     # Independent Python scalar accumulation over every origin and horizon.
     d=[float(p[i,ci,c,h])-float(y[i,c,h]) for i in range(len(y)) for h in range(y.shape[-1])];sr=math.sqrt(math.fsum(v*v for v in d)/len(d));sm=math.fsum(abs(v) for v in d)/len(d);assert math.isclose(sr,rmse,rel_tol=1e-10,abs_tol=1e-10) and math.isclose(sm,mae,rel_tol=1e-10,abs_tol=1e-10);checks+=2
     for i,o in enumerate(origins):originrows.append(dict(arm=arm,seed=seed,policy=policy,condition=cond,channel=c,origin=int(o),normalized_MSE=float(per[i,cc]),raw_MAE=float(abs(e[i,cc]).mean())))
   series[(arm,seed,policy,'PRIMARY')]=(float(np.mean(primv)),np.mean(primb,0));assert math.isclose(np.mean(primv),main['accuracy'],rel_tol=1e-10,abs_tol=1e-10)
  if t=='R04':
   for side in range(2):
    for c in range(4):
     d=(p[:,0,side,c]-yy[:,side,c]).reshape(-1);raw.append(dict(arm=arm,seed=seed,policy=policy,condition=['early','late'][side],channel=c,normalized_RMSE=float(np.sqrt(np.mean(d*d))/data.sigma[c]),raw_RMSE=float(np.sqrt(np.mean(d*d))),raw_MAE=float(np.mean(abs(d)))));assert math.isclose(math.sqrt(math.fsum(float(v)**2 for v in d)/len(d)),np.sqrt(np.mean(d*d)),rel_tol=1e-10,abs_tol=1e-10);checks+=1
   trainfixed=read(OUT/t/'frozen_parameters.json');v=np.mean(((data.inputs['E_DISCOVERY']['innovation_observed']-frozen['E_DISCOVERY'][:,:,:24])/data.sigma[None,:,None])**2,-1);high=v>=trainfixed['innovation_upper_quartile'];rv=rev.mean(-1);secondary.append(dict(arm=arm,seed=seed,policy=policy,metric='high_innovation_revision_RMS',value=float(np.sqrt(rv[high].mean())),n_pairs_channels=int(high.sum())))
  if t=='R09':
   q=p[:,0];dm=(q.mean(-1)-y.mean(-1));pat=((q-q.mean(-1,keepdims=True))-(y-y.mean(-1,keepdims=True)))/data.sigma[None,:,None];secondary.extend([dict(arm=arm,seed=seed,policy=policy,metric='daily_mean_RMSE_normalized',value=float(np.sqrt(np.mean((dm/data.sigma)**2,0)).mean())),dict(arm=arm,seed=seed,policy=policy,metric='daily_sum_RMSE_raw',value=float(np.sqrt(np.mean((24*dm)**2,0)).mean())),dict(arm=arm,seed=seed,policy=policy,metric='pattern_RMSE_normalized',value=float(np.sqrt(np.mean(pat**2,(0,2))).mean()))])
  if t=='R08':
   avail=np.array([[sum(abs(b)*int(h<l) for b,l in zip(data.aux['ridge'][c][1:],data.aux['lags'][c]))/max(sum(abs(b) for b in data.aux['ridge'][c][1:]),1e-8) for h in range(48)] for c in range(4)])
   for name,mask in [('observed_donor',avail>0),('predicted_only',avail==0)]:secondary.append(dict(arm=arm,seed=seed,policy=policy,metric=name+'_normalized_RMSE',value=float(np.sqrt(np.mean((((p[:,0]-y)/data.sigma[None,:,None])**2)[:,mask])))))
  if t=='N07':
   err=np.fft.fft((p[:,0]-y)/data.sigma[None,:,None],norm='ortho');energy=np.array(data.aux['energy']);skill=np.array(data.aux['skill']);mask=(energy<=np.median(energy[:,1:24],1)[:,None])&(skill>0);mask[:,[0,24]]=False
   for name,m in [('low_predictable',mask),('remaining',~mask)]:
    if m.any():secondary.append(dict(arm=arm,seed=seed,policy=policy,metric=name+'_spectral_MSE',value=float(np.mean(abs(err[:,m])**2)),bins=int(m.sum())))
 csvwrite(OUT/t/'scores_summary.csv',summary);csvwrite(OUT/t/'raw_scores.csv',raw);csvwrite(OUT/t/'scores_by_origin.csv',originrows);csvwrite(OUT/t/'secondary_scores.csv',secondary)
 contrasts=[];allconditions=sorted(set(k[3] for k in series));prop=PROPOSED[t]
 for policy in ['selected','fixed512']:
  for condition in allconditions:
   for baseline in CONTRASTS[t]:
    seedrows=[]
    for seed in [73101,73102]:
     if (prop,seed,policy,condition) not in series or (baseline,seed,policy,condition) not in series:continue
     a,ab=series[(prop,seed,policy,condition)];b,bb=series[(baseline,seed,policy,condition)];g=100*(b-a)/b;bs=100*(bb-ab)/bb;row=dict(method=prop,baseline=baseline,policy=policy,condition=condition,seed=str(seed),method_score=a,baseline_score=b,gain_percent=g,ci_low=float(np.quantile(bs,.025)),ci_high=float(np.quantile(bs,.975)),replicates=2000);contrasts.append(row);seedrows.append((a,b,ab,bb))
    if len(seedrows)==2:
     a,b=np.mean([(v[0],v[1]) for v in seedrows],0);ab=np.mean([v[2] for v in seedrows],0);bb=np.mean([v[3] for v in seedrows],0);bs=100*(bb-ab)/bb;contrasts.append(dict(method=prop,baseline=baseline,policy=policy,condition=condition,seed='MEAN',method_score=float(a),baseline_score=float(b),gain_percent=float(100*(b-a)/b),ci_low=float(np.quantile(bs,.025)),ci_high=float(np.quantile(bs,.975)),replicates=2000))
 csvwrite(OUT/t/'contrasts.csv',contrasts);csvwrite(OUT/t/'uncertainty.csv',contrasts)
 if t=='R04':
  frontier=[]
  for (arm,seed,policy,condition),(score,boot) in series.items():
   if condition!='PRIMARY':continue
   ref=series.get(('D0',seed,policy,'PRIMARY'))
   if ref:frontier.append(dict(arm=arm,seed=seed,policy=policy,normalized_RMSE=score,raw_revision_RMS=series[(arm,seed,policy,'REVISION')][0],E_accuracy_protected=score<=1.01*ref[0],accuracy_ratio=score/ref[0]))
  csvwrite(OUT/t/'accuracy_revision_frontier.csv',frontier)
 fits=read(OUT/t/'fits.json');updates=sum(f['updates'] for f in fits);complete=sum(f['status']=='COMPLETE' for f in fits);save(OUT/t/'verification.json',dict(passed=True,scalar_metric_checks=checks,rtol=1e-10,atol=1e-10,selection_sealed_before_E=True,metric_definition='sqrt after mean_origin,h; then equal channels and designated conditions; then equal two seeds',bootstrap_seed=73300,bootstrap_replicates=2000,all_origins_retained=len(y),aliases=aliases));csvwrite(OUT/t/'fit_manifest.csv',[{k:v for k,v in f.items() if k not in ['checkpoints','extra_parameters']} for f in fits]);csvwrite(OUT/t/'resources.csv',[{k:f.get(k) for k in ['id','arm','seed','updates','optimizer_seconds','seconds','trainable_parameters','peak_allocated_bytes','peak_reserved_bytes','status']} for f in fits]);save(OUT/t/'STATUS.json',dict(EXECUTION='COMPLETE' if complete==4*(len(ARMS[t])-len(aliases)) else 'PARTIAL',EVIDENCE='POSITIVE_UNCERTAIN',NOVELTY='UNVERIFIED_VARIANT',neural_fits=complete,attempted_fits=len(fits),main_updates=updates,aliases=aliases,selected_INIT=sum(s['selected']['step']==0 for s in selections),repeated_models=len(selections)))
