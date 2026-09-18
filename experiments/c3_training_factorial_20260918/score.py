import pandas as pd
from .common import *
from .analysis_math import *
from experiments.persistence_evidence_extension_20260918 import common as ext
from experiments.persistence_evidence_extension_20260918.score import metric_arrays
from experiments.outlier_signal_peft_v1_20260917.reference_core import pinball_scalar
PANELS=['electricity','electricity_transfer','ettm1']

def score():
 check_seal();manifest=read(OUT/'PREDICTIONS.json');marker=read(OUT/'ALL_PREDICTIONS_SAVED.json');assert sha(OUT/'PREDICTIONS.json')==marker['manifest_sha256'];assert read(OUT/'EVALUATION_SEAL.json')['at']<marker['at']
 rows=[];channels=[];scalar=0;data=read(OUT/'DATA_MANIFEST.json')
 for key,r in manifest.items():
  panel,kind=r['panel'],r['kind'];d=np.load(ext.data_path(panel)/'E_DISCOVERY_inputs.npz');y0=np.load(ext.data_path(panel)/'E_DISCOVERY_labels.npz')['y'];nc=y0.shape[1]
  if kind=='standard':names=ext.STATES;ids=np.arange(128);offset=np.load(ext.panel_path(panel,kind)/'E_DISCOVERY_offset.npy')
  else:
   meta=read(ext.panel_path(panel,kind)/'manifest.json');names=meta['states'];ids=np.array(meta['origin_indices']);offset=np.load(ext.panel_path(panel,kind)/'offset.npy')
  assert sha(ROOT/r['path'])==r['sha256'];p=np.load(ROOT/r['path'],mmap_mode='r');n=len(ids);y=np.tile(y0[ids].reshape(-1,64),(2*len(names),1))+offset[:,None];s=np.tile(d['sigma'],len(y)//nc)
  m=np.concatenate([metric_arrays(p[i:i+256],y[i:i+256],s[i:i+256]) for i in range(0,len(y),256)]).reshape(len(names),2,n,nc,5).mean(1)
  for ci,cond in enumerate(names):
   common=dict(fit=r['fit'],panel=panel,kind=kind,stage=r['stage'],condition=cond,arm=r['arm'],b=r['seed'][0],i=r['seed'][1],o=r['seed'][2],step=r['step'])
   for oi,origin in enumerate(d['origins'][ids]):rows.append(dict(**common,origin=int(origin),nmae=m[ci,oi,:,0].mean(),mae=m[ci,oi,:,1].mean(),pinball=m[ci,oi,:,3].mean()))
   for ch,cid in enumerate(data[panel]['selected_columns']):channels.append(dict(**common,channel=str(cid),nmae=m[ci,:,ch,0].mean(),mae=m[ci,:,ch,1].mean(),pinball=m[ci,:,ch,3].mean()))
   for oi in [0,n-1]:
    for ch in [0,nc-1]:
     ii=[((ci*2+z)*n+oi)*nc+ch for z in range(2)];pp=p[ii].astype(float);yy=y[ii];ss=s[ii]
     aa=sum(abs(float(pp[z,4,h])-float(yy[z,h]))/float(ss[z]) for z in range(2) for h in range(64))/128;bb=pinball_scalar(pp,yy,ss,np.arange(1,10)/10)
     np.testing.assert_allclose(m[ci,oi,ch,[0,3]],[aa,bb],rtol=1e-10,atol=1e-12);scalar+=1
 f=pd.DataFrame(rows);f.to_csv(OUT/'ORIGIN_SCORES.csv.gz',index=False,compression='gzip');pd.DataFrame(channels).to_csv(OUT/'CHANNEL_SCORES.csv.gz',index=False,compression='gzip')
 group=['panel','kind','stage','arm','b','i','o','step','origin'];fault=f[f.condition.str.startswith(('POINT','BURST'))].groupby(group,as_index=False)[['nmae','mae','pinball']].mean();fault['condition']='FAULT';f=pd.concat([f,fault],ignore_index=True)
 raw=f.groupby([k for k in group if k not in ['origin']]+['condition'],as_index=False)[['nmae','mae','pinball']].mean();raw.to_csv(OUT/'RAW_SCORES.csv',index=False)
 effects=[];cells=[];paired=[]
 for (panel,kind,stage,cond),g in f.groupby(['panel','kind','stage','condition']):
  origins=sorted(g.origin.unique());idx=pd.MultiIndex.from_product([origins,LEVELS,LEVELS,LEVELS,ARMS],names=['origin','b','i','o','arm'])
  q=g.set_index(['origin','b','i','o','arm']).reindex(idx);assert not q.nmae.isna().any();cube=q.nmae.to_numpy().reshape(len(origins),8,2);diff=cube[:,:,0]-cube[:,:,1];mean=cube.mean(0);dmean=diff.mean(0);denom=mean[:,0].mean()
  period=24 if panel.startswith('electricity') else 96;allorig=np.load(ext.data_path(panel)/'E_DISCOVERY_inputs.npz')['origins'];blocks=np.unique(allorig//(7*period));random=np.random.default_rng(89418+PANELS.index(panel));cnt=np.stack([np.bincount(random.integers(0,len(blocks),len(blocks)),minlength=len(blocks)) for _ in range(2000)]);w=cnt[:,np.searchsorted(blocks,np.array(origins)//(7*period))].astype(float);assert (w.sum(1)>0).all();w/=w.sum(1,keepdims=True)
  observed=contrasts(dmean);boot=contrasts(w@diff);c3=contrasts(mean[:,0]);mag=contrasts(mean[:,1]);np.testing.assert_allclose(observed,c3-mag,rtol=1e-9,atol=1e-12)
  np.testing.assert_allclose(dmean,dmean.mean()+DESIGN@(observed/2),rtol=1e-9,atol=1e-12)
  var=float(np.var(dmean))
  for j,name in enumerate(NAMES):
   lo,hi=np.quantile(boot[:,j],[.025,.975]);bl,bh=np.quantile(boot[:,j],[.05/14,1-.05/14]);effects.append(dict(panel=panel,kind=kind,stage=stage,condition=cond,factor=name,effect_nmae=observed[j],effect_pct_common_C3=100*observed[j]/denom,C3_effect=c3[j],MAG_effect=mag[j],ci_low=lo,ci_high=hi,bonferroni7_low=bl,bonferroni7_high=bh,cell_variance_share_pct=100*(observed[j]/2)**2/var if var>1e-25 else 0,common_C3=denom,conditional_on_two_factor_levels=True))
  for j,triple in enumerate(COMBINATIONS):
   cells.append(dict(panel=panel,kind=kind,stage=stage,condition=cond,b=triple[0],i=triple[1],o=triple[2],C3=mean[j,0],MAG=mean[j,1],difference=dmean[j],MAG_gain_pct=100*dmean[j]/mean[j,0]))
   for k,origin in enumerate(origins):paired.append(dict(panel=panel,kind=kind,stage=stage,condition=cond,b=triple[0],i=triple[1],o=triple[2],origin=origin,difference=diff[k,j]))
 pd.DataFrame(effects).to_csv(OUT/'FACTORIAL_EFFECTS.csv',index=False);pd.DataFrame(cells).to_csv(OUT/'CELL_EFFECTS.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'PAIRED_ORIGIN_DIFFERENCES.csv.gz',index=False,compression='gzip')
 assert len(effects)==840 and len(cells)==960
 # All selected diagonal means exactly replay the existing study (including negative conditions).
 parent=pd.read_csv(ROOT/'results/c3_magnitude_diagnostic_20260918/SEED_DECOMPOSITION.csv');parent=parent[parent.family=='ALL'];replayed=0
 for r in cells:
  if r['stage']!='selected' or len({r['b'],r['i'],r['o']})!=1:continue
  q=parent[(parent.panel==r['panel'])&(parent.kind==r['kind'])&(parent.condition==r['condition'])&(parent.seed==r['b'])];assert len(q)==1;np.testing.assert_allclose([r['C3'],r['MAG']],[q.iloc[0].A,q.iloc[0].D],rtol=1e-10,atol=1e-12);replayed+=1
 check_seal();save(OUT/'SCORE_VERIFICATION.json',dict(status='VERIFIED',predictions=192,conditions=120,scalar_metric_checks=scalar*2,parent_diagonal_replay=replayed,factorial_contrasts=840,cells=960,all_predictions_saved_before_scoring=True,exact_factorial_reconstruction=True));print('SCORE_VERIFIED',len(effects),flush=True)
