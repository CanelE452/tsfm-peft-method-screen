"""Independent fixed-grid CPU-control selection replay, no neural calls or new fitting."""
import sys,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.condition_sampling_repair_v1_20260917.common import *

def scalar_scores(t,p,y,sigma):
 n=len(y);s=[];rev=[]
 for c in range(4):
  if t=='R04':
   e=[(float(p[i,0,k,c,h])-float(y[i,c,h+24*k]))/sigma[c] for i in range(n) for k in range(2) for h in range(48)];d=[(float(p[i,0,1,c,h])-float(p[i,0,0,c,h+24]))/sigma[c] for i in range(n) for h in range(24)];rev.append(math.sqrt(math.fsum(v*v for v in d)/len(d)))
  else:e=[(float(p[i,0,c,h])-float(y[i,c,h]))/sigma[c] for i in range(n) for h in range(48)]
  s.append(math.sqrt(math.fsum(v*v for v in e)/len(e)))
 out=dict(accuracy=math.fsum(s)/4)
 if rev:out['revision']=math.fsum(rev)/4
 return out

def verify_train_scales():
 t='R04';fixed=read(OUT/t/'frozen_parameters.json');z=dict(np.load(CACHE/t/'frozen.npz'));inp=dict(np.load(CACHE/t/'TRAIN_inputs.npz'));y=np.load(CACHE/t/'TRAIN_labels.npz')['y'];sigma=read(OUT/t/'train_statistics.json')['sigma'];task=[];revision=[];innovation=[]
 for i in range(64):
  task.append(math.fsum(((float(z['TRAIN'][i,c,h])-float(y[i,c,h]))/sigma[c])**2+((float(z['TRAIN_late'][i,c,h])-float(y[i,c,h+24]))/sigma[c])**2 for c in range(4) for h in range(48))/(2*4*48))
  revision.append(math.fsum(((float(z['TRAIN_late'][i,c,h])-float(z['TRAIN'][i,c,h+24]))/sigma[c])**2 for c in range(4) for h in range(24))/(4*24))
  innovation.extend([math.fsum(((float(inp['innovation_observed'][i,c,h])-float(z['TRAIN'][i,c,h]))/sigma[c])**2 for h in range(24))/24 for c in range(4)])
 values=dict(lambda_value=float(np.clip(.05*np.median(task[:16])/max(np.median(revision[:16]),1e-8),1e-4,1)),mean_w=math.fsum(1/(1+v) for v in innovation)/len(innovation),innovation_upper_quartile=float(np.quantile(innovation,.75)))
 for k,v in values.items():assert math.isclose(v,fixed['lambda' if k=='lambda_value' else k],rel_tol=1e-10,abs_tol=1e-10)
 save(OUT/t/'independent_TRAIN_scale_verification.json',dict(passed=True,TRAIN_pairs_for_lambda=16,TRAIN_pairs_for_mean_weight=64,**values,rtol=1e-10,atol=1e-10,new_model_calls=0,optimizer_updates=0))

def run():
 checks=[]
 for t in ['N02','R04','R08']:
  if read(OUT/t/'STATUS.json')['EXECUTION']!='COMPLETE':continue
  if t=='R04':verify_train_scales()
  sel=read(OUT/t/'selections.json')['selections'];cpu=read(OUT/t/'CPU_selection.json');sigma=np.array(read(OUT/t/'train_statistics.json')['sigma']);mu=np.array(read(OUT/t/'train_statistics.json')['mu']);inputs=np.load(CACHE/t/'V_SELECT_inputs.npz');y=np.load(CACHE/t/'V_SELECT_labels.npz')['y']
  if t=='N02':simple=inputs['retrieval_future'].reshape(32,4,2,48).mean(2)[:,None]
  if t=='R08':
   q0=np.load(CACHE/t/'frozen.npz')['V_SELECT'];aux=read(OUT/t/'feature_or_transform_manifest.json');simple=np.empty((32,1,4,48))
   for i in range(32):
    for c in range(4):
     for h in range(48):
      z=aux['ridge'][c][0]
      for d,l,b in zip(aux['donors'][c],aux['lags'][c],aux['ridge'][c][1:]):
       value=inputs['context'][i,d,336+h-l] if h<l else q0[i,d,h-l];z+=b*(value-mu[d])/sigma[d]
      simple[i,0,c,h]=z*sigma[c]+mu[c]
  for choice in cpu['choices']:
   s=next(x for x in sel if x['seed']==choice['seed'] and x['arm']==choice['base']);p=np.load(ROOT/s['selected']['prediction'])['pred'];rows=[]
   for old in choice['scores']:
    a=old['alpha']
    if t=='R04':mixed=p.copy();mixed[:,0,1,:,:24]=(1-a)*p[:,0,1,:,:24]+a*p[:,0,0,:,24:]
    else:mixed=(1-a)*p+a*simple
    sc=scalar_scores(t,mixed,y,sigma)
    for k,v in sc.items():assert math.isclose(v,old[k],rel_tol=1e-10,abs_tol=1e-10),(t,k,v,old[k])
    rows.append(dict(alpha=a,**sc))
   if t=='R04':rows=[r for r in rows if r['accuracy']<=s['selected']['accuracy']*1.01];best=min(rows,key=lambda r:(r['revision'],r['alpha']))
   else:best=min(rows,key=lambda r:(r['accuracy'],r['alpha']))
   assert best['alpha']==choice['alpha'];checks.append(dict(track=t,seed=choice['seed'],base=choice['base'],alpha=best['alpha'],passed=True,scalar_grid_checks=5))
  save(OUT/t/'independent_CPU_control_verification.json',dict(passed=True,checks=[r for r in checks if r['track']==t],new_fits=0,optimizer_updates=0,new_model_calls=0,uses_E=False,rtol=1e-10,atol=1e-10))
 save(OUT/'independent_CPU_control_verification.json',dict(passed=True,checks=checks));print('CPU_CONTROL_SCALAR_VERIFIED',len(checks))
if __name__=='__main__':run()
