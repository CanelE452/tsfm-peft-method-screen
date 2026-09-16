"""Independent Python scalar audit of paired revision and aggregate/pattern scores."""
import sys,math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.condition_studies_v1_20260916.common import *

def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
def mean(v):return math.fsum(v)/len(v)
def rms(v):return math.sqrt(mean([x*x for x in v]))
def run():
 total=[]
 for t in ['R04','R09']:
  if read(OUT/t/'STATUS.json')['EXECUTION']!='COMPLETE':continue
  b=dict(np.load(CACHE/t/'evaluation_bundle.npz'));sigma=read(OUT/t/'train_statistics.json')['sigma'];summary=pd.read_csv(OUT/t/'scores_summary.csv');raw=pd.read_csv(OUT/t/'raw_scores.csv');secondary=pd.read_csv(OUT/t/'secondary_scores.csv');rows=[]
  for key,values in b.items():
   if key in ['y','origins','conditions']:continue
   arm,seed,policy=key.split('__');seed=int(seed);p=values[:,0];y=b['y']
   if t=='R04':
    rev=[]
    for c in range(4):
     for side in range(2):
      errors=[float(p[i,side,c,h])-float(y[i,c,h+24*side]) for i in range(len(y)) for h in range(48)];record=raw[(raw.arm==arm)&(raw.seed==seed)&(raw.policy==policy)&(raw.condition==['early','late'][side])&(raw.channel==c)].iloc[0]
      for metric,value in [('raw_RMSE',rms(errors)),('raw_MAE',mean([abs(v) for v in errors]))]:close(value,record[metric]);rows.append(dict(arm=arm,seed=seed,policy=policy,metric=metric,channel=c,condition=['early','late'][side],scalar=value,recorded=float(record[metric])))
     rev.append(rms([(float(p[i,1,c,h])-float(p[i,0,c,h+24]))/sigma[c] for i in range(len(y)) for h in range(24)]))
    value=mean(rev);expected=float(summary[(summary.arm==arm)&(summary.seed==seed)&(summary.policy==policy)&(summary.condition=='PRIMARY')].revision.iloc[0]);close(value,expected);rows.append(dict(arm=arm,seed=seed,policy=policy,metric='revision_RMS_normalized',scalar=value,recorded=expected))
   else:
    dm=[];ds=[];pattern=[]
    for c in range(4):
     means=[];errors=[]
     for i in range(len(y)):
      pm=mean([float(v) for v in p[i,c]]);ym=mean([float(v) for v in y[i,c]]);means.append(pm-ym);errors.extend([(float(p[i,c,h])-pm-float(y[i,c,h])+ym)/sigma[c] for h in range(24)])
     dm.append(rms([x/sigma[c] for x in means]));ds.append(rms([24*x for x in means]));pattern.append(rms(errors))
    for metric,value in [('daily_mean_RMSE_normalized',mean(dm)),('daily_sum_RMSE_raw',mean(ds)),('pattern_RMSE_normalized',mean(pattern))]:
     expected=float(secondary[(secondary.arm==arm)&(secondary.seed==seed)&(secondary.policy==policy)&(secondary.metric==metric)].value.iloc[0]);close(value,expected);rows.append(dict(arm=arm,seed=seed,policy=policy,metric=metric,scalar=value,recorded=expected))
  csvwrite(OUT/t/'independent_secondary_scores.csv',rows);receipt=dict(passed=True,track=t,checks=len(rows),scope='all saved repeat/selected/fixed512/control predictions; Python fsum scalar audit',rtol=1e-10,atol=1e-10,new_model_calls=0,optimizer_updates=0);save(OUT/t/'independent_secondary_verification.json',receipt);total.append(receipt)
 save(OUT/'independent_secondary_verification.json',dict(passed=True,tracks=total));print('SECONDARY_SCALAR_VERIFIED',[(r['track'],r['checks']) for r in total])
if __name__=='__main__':run()
