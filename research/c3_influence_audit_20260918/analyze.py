"""CPU-only descriptive sensitivity; no fitting, inference or causal ranking."""
from pathlib import Path
import hashlib, json, itertools, sys
import numpy as np
import pandas as pd
import torch
P=Path(__file__).resolve().parent; ROOT=P.parents[1]; sys.path.insert(0,str(ROOT))
from experiments.outlier_signal_peft_v1_20260917.reference_core import rng
D=ROOT/'results/c3_magnitude_diagnostic_20260918'; W=ROOT/'results/c3_weakness_controls_20260918'
seen={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def source(p):seen[str(p.relative_to(ROOT))]=sha(p);return p
def read(p):return json.loads(source(p).read_text())
def dump(n,x):(P/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def variance_parts(x):
 center=x-x.mean(); parts={}; rows=[]; total=np.mean(center**2)
 for size in range(1,4):
  for axes in itertools.combinations(range(3),size):
   y=center.mean(axis=tuple(a for a in range(3) if a not in axes),keepdims=True)
   for small,v in parts.items():
    if set(small)<set(axes):y=y-v
   parts[axes]=y; v=float(np.mean(y*y)); rows.append((axes,v,100*v/total if total else 0.))
 np.testing.assert_allclose(sum(parts.values()),center,rtol=1e-9,atol=1e-14)
 np.testing.assert_allclose(sum(r[1] for r in rows),total,rtol=1e-9,atol=1e-16)
 return rows
# Known additive and interaction signals validate the decomposition independently.
a=np.array([-1.,1.])[:,None,None]; b=np.array([-1.,0.,1.])[None,:,None]; c=np.array([-2.,0.,2.])[None,None,:]
r=dict((k,v) for k,v,_ in variance_parts(5+a+2*b+3*c))
np.testing.assert_allclose([r[(0,)],r[(1,)],r[(2,)]],[1,8/3,24])
assert sum(v for k,v in r.items() if len(k)>1)<1e-20
r=dict((k,v) for k,v,_ in variance_parts(np.broadcast_to(a*b,(2,3,3))))
np.testing.assert_allclose(r[(0,1)],2/3);assert sum(v for k,v in r.items() if k!=(0,1))<1e-20
source(P/'PROTOCOL.md'); source(P/'analyze.py')
path=source(W/'ALL_ORIGIN_SCORES.csv.gz'); frames=[]
for f in pd.read_csv(path,usecols=['panel','kind','condition','arm','seed','origin','channel','nmae'],chunksize=150000,dtype={'channel':str}):
 frames.append(f[f.arm.isin(['C3','MAG_ONLY'])])
f=pd.concat(frames,ignore_index=True); keys=['panel','kind','condition','seed','origin','channel'];assert not f.duplicated(keys+['arm']).any()
z=f.pivot(index=keys,columns='arm',values='nmae').reset_index();assert not z.isna().any().any()
fault=z[(z.kind=='standard')&z.condition.str.startswith(('POINT','BURST'))]
assert fault.condition.nunique()==6
f=fault.groupby([k for k in keys if k!='condition'],as_index=False)[['C3','MAG_ONLY']].mean();f['condition']='FAULT';z=pd.concat([z,f],ignore_index=True)
parent=pd.read_csv(source(D/'DECOMPOSITION.csv'));parent=parent[parent.family=='ALL']
orig=pd.read_csv(source(D/'ORIGIN_ERRORS.csv.gz')); seedparent=pd.read_csv(source(D/'SEED_DECOMPOSITION.csv'));seedparent=seedparent[seedparent.family=='ALL']
parts=[];sens=[];summary=[];seedrows=[]
for (panel,kind,cond),g in z.groupby(['panel','kind','condition'],sort=True):
 meta=dict(panel=panel,kind=kind,condition=cond);s=sorted(g.seed.unique());o=sorted(g.origin.unique());c=sorted(g.channel.unique());assert len(s)==3
 idx=pd.MultiIndex.from_product([s,o,c],names=['seed','origin','channel']);v=g.set_index(['seed','origin','channel']).reindex(idx);assert not v[['C3','MAG_ONLY']].isna().any().any()
 aa=v.C3.to_numpy().reshape(len(s),len(o),len(c));dd=v.MAG_ONLY.to_numpy().reshape(aa.shape);diff=aa-dd
 p=parent[(parent.panel==panel)&(parent.kind==kind)&(parent.condition==cond)];assert len(p)==1
 np.testing.assert_allclose([aa.mean(),dd.mean()],[p.iloc[0].A,p.iloc[0].D],rtol=1e-10,atol=1e-12)
 q=orig[(orig.panel==panel)&(orig.kind==kind)&(orig.condition==cond)].set_index(['seed','origin']).reindex(pd.MultiIndex.from_product([s,o]))
 np.testing.assert_allclose(aa.mean(2).ravel(),q.A,rtol=1e-10,atol=1e-12);np.testing.assert_allclose(dd.mean(2).ravel(),q.D,rtol=1e-10,atol=1e-12)
 for axes,var,share in variance_parts(diff):parts.append(dict(**meta,component=':'.join(['seed','origin','channel'][i] for i in axes),variance=var,share_pct=share))
 full=100*diff.mean()/aa.mean();summary.append(dict(**meta,origins=len(o),channels=len(c),seeds=3,C3=aa.mean(),MAG=dd.mean(),gain_pct=full))
 for i,seed in enumerate(s):
  q=seedparent[(seedparent.panel==panel)&(seedparent.kind==kind)&(seedparent.condition==cond)&(seedparent.seed==seed)];assert len(q)==1
  np.testing.assert_allclose(diff[i].mean(),q.iloc[0].total,rtol=1e-9,atol=1e-12)
  seedrows.append(dict(**meta,seed=int(seed),difference=diff[i].mean(),gain_pct=100*diff[i].mean()/aa[i].mean(),signed_mean_share_pct=100*diff[i].mean()/diff.mean()/3 if abs(diff.mean())>1e-15 else None))
 for axis,labels in [(0,np.array(s)),(2,np.array(c)),(1,np.array(o)//(7*(24 if panel.startswith('electricity') else 96)))]:
  factor={0:'seed',1:'index_week',2:'channel'}[axis]
  for value in np.unique(labels):
   mask=labels!=value;at=np.compress(mask,aa,axis=axis);dt=np.compress(mask,diff,axis=axis);gain=100*dt.mean()/at.mean()
   sens.append(dict(**meta,factor=factor,excluded=str(value),removed_cells=int(aa.size-at.size),gain_pct=gain,change_pp=gain-full))
assert len(summary)==60
for name,rows in [('VARIANCE_COMPONENTS.csv',parts),('LEAVE_ONE_GROUP_OUT.csv',sens),('SUMMARY.csv',summary),('SEED_CONTRIBUTIONS.csv',seedrows)]:pd.DataFrame(rows).to_csv(P/name,index=False)
# Training provenance: checkpoint state equality (CPU only), raw hashes, shared cache.
models=read(D/'MODELS.json'); bases=read(W/'BASELINE_MANIFEST.json');provenance=[];initials={}
for m in models:
 run=Path(m['checkpoint']).parts[1];receipt=read(ROOT/'results'/run/'fits'/m['fit']/'receipt.json');base=bases[f"{m['source']}_{m['seed']}"]
 initial=next(r for r in receipt['checkpoints'] if r['step']==0)
 for row in [m,base,initial]:assert sha(source(ROOT/row['checkpoint']))==row['sha256']
 state=torch.load(ROOT/initial['checkpoint'],map_location='cpu',weights_only=True);initials[(m['source'],m['seed'],m['arm'])]=state
 order=np.stack([rng(84100,m['source'],m['seed'],epoch).permutation(1024) for epoch in range(32)])
 paths=[ROOT/'.cache'/run/'conditions'/m['source']/(n+'.npy') for n in ['train_x','train_y','train_sigma']]
 for p in paths:assert p.exists()
 provenance.append(dict(source=m['source'],seed=m['seed'],arm=m['arm'],lr=m['lr'],selected_step=m['step'],B0_step=base['step'],B0_sha256=base['sha256'],initial_sha256=initial['sha256'],order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),frozen_sha256=receipt['frozen_sha256'],microbatch=receipt['microbatch'],train_paths=[str(p.resolve().relative_to(ROOT)) for p in paths]))
for source_name in ['electricity','ettm1']:
 rows=[r for r in provenance if r['source']==source_name];assert len({r['B0_sha256'] for r in rows})==3;assert len({r['order_sha256'] for r in rows})==3
 assert len({tuple(r['train_paths']) for r in rows})==1
 for seed in [81551,81552,81553]:
  x=initials[(source_name,seed,'C3')];y=initials[(source_name,seed,'MAG_ONLY')];assert x.keys()==y.keys();assert all(torch.equal(x[k],y[k]) for k in x)
  u,v=[r for r in rows if r['seed']==seed]
  for k in ['B0_sha256','order_sha256','frozen_sha256','lr','microbatch']:assert u[k]==v[k],(source_name,seed,k)
 assert all(any(not torch.equal(initials[(source_name,s,'C3')][k],initials[(source_name,t,'C3')][k]) for k in initials[(source_name,s,'C3')]) for s,t in itertools.combinations([81551,81552,81553],2))
for run in ['additive_b0_adapter_v1_20260917','additive_persistence_validation_v1_20260917','c3_weakness_controls_20260918']:
 for n in ['train.py','model.py','checks.py','common.py']:source(ROOT/'experiments'/run/n)
# Hash shared arrays once, without evaluating any model.
for paths in {tuple(r['train_paths']) for r in provenance}:
 for p in paths:source(ROOT/p)
dump('TRAINING_PROVENANCE.json',provenance)
assert all(sha(ROOT/p)==h for p,h in seen.items())
dump('AUDIT.json',dict(status='VERIFIED',new_fits=0,optimizer_updates=0,new_inference=0,new_bootstrap=0,conditions=60,variance_rows=len(parts),mean_seed_and_origin_replay=True,balanced_complete=True,synthetic_math_checks=True,variance_additivity=True,paired_initial_weights_equal=True,paired_B0_order_frozen_lr_microbatch_equal=True,three_distinct_B0_initialization_and_orders=True,input_hashes=seen,causal_ranking_identified=False))
print(pd.DataFrame(parts).query("panel=='electricity_transfer' and condition=='SHIFT8'").to_string(index=False))
for factor in ['seed','channel','index_week']:
 q=pd.DataFrame(sens).query("panel=='electricity_transfer' and condition=='SHIFT8' and factor==@factor")
 print(factor,'range',q.gain_pct.min(),q.gain_pct.max(),'max_abs_change_pp',q.change_pp.abs().max())
print('VERIFIED',len(summary),'conditions; zero training/inference')
