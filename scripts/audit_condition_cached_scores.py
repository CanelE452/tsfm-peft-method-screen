"""Independent saved-prediction policy / sample-score audit, no neural execution."""
import sys,time,math,itertools
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.condition_studies_v1_20260916.common import *
from experiments.condition_studies_v1_20260916.cached import Prior,TAU,pin

def audit_r05():
 p=Prior();sels=read(OUT/'R05/selections.json')['policies'];checks=[]
 for target in sels:
  ys=[];latest=[];path=[];sigma=p.scales[target]['sigma_y']
  for seed in [61730,61731]:
   q,y,ids=p.get(target,'V_SELECT','LATEST',seed);r,yy,ii=p.get(target,'V_SELECT','PATH',seed);assert ids==ii and np.array_equal(y,yy);ys.append(y);latest.append(q);path.append(r)
  base=np.mean([pin(a,y,sigma)[:,0].mean() for a,y in zip(latest,ys)])
  for arm,grid in [('E2',[(x,)*4 for x in [0,.25,.5,.75,1]]),('E4',list(itertools.combinations_with_replacement([0,.25,.5,.75,1],4)))]:
   rows=[]
   for alpha in grid:
    a=np.array(alpha)[None,:,None,None];v=np.mean([pin((1-a)*q+a*r,y,sigma).mean(0) for q,r,y in zip(latest,path,ys)],0)
    if v[0]<=1.01*base:rows.append((float(v[1:].mean()),np.mean(alpha),alpha))
   chosen=min(rows,key=lambda r:(r[0],r[1],r[2]))[2];assert list(chosen)==sels[target][arm];checks.append(dict(target=target,arm=arm,grid=len(grid),policy=list(chosen),exact=True))
 aliases={t:{a:next((b for b in ARMS['R05'] if b<a and v==sels[t][b]),None) for a,v in s.items()} for t,s in sels.items()};save(OUT/'R05/independent_policy_verification.json',dict(passed=True,checks=checks,aliases=aliases,neural_fits=0,optimizer_updates=0,new_policy_fits=0,scope='Exact audit replay of prescribed V grids; selections unchanged'))
 # Worst delayed case mean, computed per seed/target then compared with shared calendar resamples.
 df=pd.read_csv(OUT/'R05/scores_by_origin.csv');df=df[(df.variant=='RAW')&df.condition.isin(['S1','S2','S3'])];calendar=np.load(CACHE/'R05/bootstrap.npz');counts=calendar['counts'];dates=list(calendar['days']);series={}
 for (t,s,a),g in df.groupby(['target','seed','arm']):
  vv=np.full((91,3),np.nan)
  for r in g.itertuples():vv[dates.index(r.date),int(r.condition[1])-1]=r.value
  valid=np.isfinite(vv);boot=(counts@np.nan_to_num(vv))/(counts@valid);series[t,s,a]=(float(np.nanmean(vv,axis=0).max()),boot.max(1))
 rows=[]
 for scope,targets in [('ALL',['T0','T1','T2']),('ADDITIONAL',['T1','T2']),*[(t,[t]) for t in ['T0','T1','T2']]]:
  for baseline in ['E0','E2','E3']:
   vals=[];boots=[]
   for arm in ['E4',baseline]:vals.append(np.mean([series[t,s,arm][0] for t in targets for s in [61730,61731]]));boots.append(np.mean([series[t,s,arm][1] for t in targets for s in [61730,61731]],0))
   gain=100*(vals[1]-vals[0])/vals[1];bs=100*(boots[1]-boots[0])/boots[1];rows.append(dict(scope=scope,method='E4',baseline=baseline,method_score=vals[0],baseline_score=vals[1],gain_percent=gain,ci_low=float(np.quantile(bs,.025)),ci_high=float(np.quantile(bs,.975)),rule='max delayed case mean per seed/target, then equal average seeds/targets'))
 csvwrite(OUT/'R05/worst_delayed_contrasts.csv',rows);print('R05_POLICY_VERIFIED')

def audit_n06():
 p=Prior();pars=read(OUT/'N06/selections.json')['parameters'];df=pd.read_csv(OUT/'N06/scores_by_origin.csv');manifest=read(OUT/'N06/predictions_manifest.json');rows=[]
 for target in ['T0','T1','T2']:
  ids=sorted({r['id'] for r in manifest if r['target']==target});jid=ids[0];label=p.labels[jid];assert sha(ROOT/label['path'])==label['sha256'];y=np.load(ROOT/label['path']);sigma=p.scales[target]['sigma_y'];threshold=pars[target]['threshold_rolling6_p90']
  for arm in ARMS['N06']:
   r=next(r for r in manifest if r['id']==jid and r['arm']==arm);assert sha(ROOT/r['path'])==r['sha256'];x=np.load(ROOT/r['path'])['samples'];N=len(x);s=[math.fsum(map(float,row)) for row in x];truth=math.fsum(map(float,y));crps=math.fsum(abs(v-truth) for v in s)/N-math.fsum(abs(s[i]-s[j]) for i in range(N) for j in range(i))/N**2
   z=x/sigma;zy=y/sigma;distance=lambda a,b:math.sqrt(math.fsum((float(u)-float(v))**2 for u,v in zip(a,b)));energy=math.fsum(distance(a,zy) for a in z)/N-math.fsum(distance(z[i],z[j]) for i in range(N) for j in range(i))/N**2;terms=[]
   for h in range(24):
    for k in range(h):terms.append((math.sqrt(abs(float(y[h])-float(y[k])))-math.fsum(math.sqrt(abs(float(a[h])-float(a[k]))) for a in x)/N)**2)
   vg=math.fsum(terms)/len(terms);event=lambda a:max(math.fsum(map(float,a[h:h+6]))/6 for h in range(19))>threshold;prob=sum(event(a) for a in x)/N;brier=(prob-int(event(y)))**2;metrics=dict(sum_CRPS_normalized=crps/(24*sigma),sum_CRPS_Wh=crps,energy=energy,variogram=vg,variogram_normalized=vg/sigma,Brier=brier)
   for name,value in metrics.items():
    old=float(df[(df.target==target)&(df.arm==arm)&(df.id==jid)&(df.metric==name)].value.iloc[0]);assert math.isclose(value,old,rel_tol=1e-10,abs_tol=1e-10),(target,arm,name,value,old);rows.append(dict(target=target,arm=arm,id=jid,metric=name,scalar=value,recorded=old,absolute_error=abs(value-old)))
 csvwrite(OUT/'N06/independent_scalar_scores.csv',rows);save(OUT/'N06/independent_scalar_verification.json',dict(passed=True,scope='Predefined first E origin per target, all4couplings, all6scalar scores; full900primary fast/brute checks recorded separately',checks=len(rows),rtol=1e-10,atol=1e-10,optimizer_updates=0,new_model_calls=0));print('N06_SCALAR_VERIFIED',len(rows))
if __name__=='__main__':
 if read(OUT/'R05/STATUS.json')['EXECUTION']=='COMPLETE':audit_r05()
 if read(OUT/'N06/STATUS.json')['EXECUTION']=='COMPLETE':audit_n06()
