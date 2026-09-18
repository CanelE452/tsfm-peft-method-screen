import itertools
import pandas as pd
from .common import *

def analyze():
 inventory={r['fit']:r for r in read(OUT/'MOMENT_INVENTORY.json')};states={};blocked=[]
 for row in read(PRIOR/'GRID.json'):
  info=inventory[row['fit']]
  if not info['available']:blocked.append(dict(fit=row['fit'],status='BLOCKED_OLD_OPTIMIZER_STATE'));continue
  s=torch.load(ROOT/info['resume'],map_location='cpu',weights_only=False);assert s['step']==1024
  group=s['optimizer']['param_groups'];assert len(group)==1;ids=group[0]['params'];assert len(ids)==4
  expected=[(8,512),(8,),(512,8),(512,)];assert [tuple(s['optimizer']['state'][j]['exp_avg'].shape) for j in ids]==expected
  vectors={n:torch.cat([s['optimizer']['state'][j][n].flatten() for j in ids]).numpy().astype(float) for n in ['exp_avg','exp_avg_sq']};vectors['weights']=torch.cat([v.flatten() for v in s['model'].values()]).numpy().astype(float);states[row['fit']]=vectors
 rows=[]
 for source,b,i,arm in itertools.product(SOURCES,LEVELS,LEVELS,ARMS):
  pair=sorted([r for r in read(PRIOR/'GRID.json') if r['source']==source and r['seed'][:2]==[b,i] and r['arm']==arm],key=lambda r:r['seed'][2]);assert len(pair)==2
  if any(r['fit'] not in states for r in pair):continue
  a,d=[states[r['fit']] for r in pair];meta=dict(source=source,b=b,i=i,arm=arm)
  for n in ['weights','exp_avg','exp_avg_sq']:meta[n+'_cosine']=cos(a[n],d[n]);meta[n+'_relative_difference']=rel(d[n],a[n])
  rows.append(meta)
 pd.DataFrame(rows).to_csv(OUT/'ORDER_FINAL_MOMENTS.csv',index=False);save(OUT/'MOMENT_CHECKS.json',dict(available=len(states),blocked=blocked,optimizer_updates=0));print('MOMENTS',len(states),'available',flush=True)
