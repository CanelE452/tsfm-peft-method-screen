#!/usr/bin/env python3
"""Requested monthly uncertainty; reuse the original sealed bootstrap calendar."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from experiments.forecast_path_structure_v1_20260916.common import *
def main():
 df=pd.read_csv(OUT/'scores_by_origin.csv');b=np.load(CACHE/'bootstrap_calendar.npz');counts=b['counts'];dates=b['dates'];dmap={d:i for i,d in enumerate(dates)};targets=sorted(df.target.unique());scopes={t:[t] for t in targets};scopes.update(ALL=targets,ADDITIONAL=[t for t in targets if t!='T0']);cache={};rows=[]
 def values(arm,policy,variant,t,seed,month):
  sd=-1 if arm.startswith('FROZEN') else seed;policy='SELECTED' if arm.startswith('FROZEN') else policy;key=(arm,policy,variant,t,sd,month)
  if key not in cache:
   sub=df[(df.target==t)&(df.arm==arm)&(df.policy==policy)&(df.variant==variant)&(df.seed==sd)&(df.month==month)];a=np.full((4,len(dates)),np.nan)
   for r in sub.itertuples():a[int(r.case[1]),dmap[r.date]]=r.primary
   den=np.isfinite(a).astype(float)@counts.T;num=np.nan_to_num(a)@counts.T;boot=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
   cache[key]=(np.nanmean(a,axis=1),boot,sub.date.nunique())
  return cache[key]
 for month in [10,11,12]:
  for policy,variant in [('SELECTED','RAW'),('SELECTED','CALIBRATED'),('FIXED512','RAW')]:
   for method,base in [('PATH',x) for x in ['LATEST','DROP','POINT','FROZEN_WEATHER']]:
    for scope,ts in scopes.items():
     for case in ['S0','S1','S2','S3','MEAN','WORST']:
      vals=[];boots=[];nn=[]
      for arm in [method,base]:
       tvals=[];tboots=[]
       for t in ts:
        sv=[];sb=[]
        for seed in SEEDS:
         v,bo,n=values(arm,policy,variant,t,seed,month);sv.append(v);sb.append(bo);nn.append(n)
        v=np.mean(sv,axis=0);bo=np.mean(sb,axis=0)
        if case=='MEAN':v=v.mean();bo=bo.mean(0)
        elif case=='WORST':v=v.max();bo=bo.max(0)
        else:k=int(case[1]);v=v[k];bo=bo[k]
        tvals.append(v);tboots.append(bo)
       vals.append(float(np.mean(tvals)));boots.append(np.mean(tboots,axis=0))
      finite=np.isfinite(boots[0])&np.isfinite(boots[1]);g=100*(boots[1][finite]-boots[0][finite])/boots[1][finite]
      rows.append(dict(month=month,scope=scope,policy=policy,variant=variant,case=case,method=method,baseline=base,method_score=vals[0],baseline_score=vals[1],gain_percent=100*(vals[1]-vals[0])/vals[1],ci_low=float(np.quantile(g,.025)),ci_high=float(np.quantile(g,.975)),original_resamples=2000,defined_month_resamples=int(finite.sum()),undefined_empty_month_resamples=int((~finite).sum()),origins_per_target=min(nn)))
 csvwrite(OUT/'monthly_contrasts.csv',rows)
 save(OUT/'monthly_uncertainty_manifest.json',dict(bootstrap_calendar_sha256=sha(CACHE/'bootstrap_calendar.npz'),method='Use exactly the same 2000 global calendar resamples, restricting each statistic to the named month. Zero-observation monthly resamples are undefined, counted explicitly, and omitted only from that descriptive interval; no observed dates, targets or seeds removed. Main contrasts and CIs are unchanged.',new_model_calls=0,new_fits=0,new_selections=0,rows=len(rows)))
 print('MONTHLY_DESCRIPTIVE_ROWS',len(rows))
if __name__=='__main__':main()
