import pandas as pd
from .common import *
from experiments.persistence_evidence_extension_20260918 import common as ext
from experiments.persistence_evidence_extension_20260918.score import metric_arrays
from experiments.outlier_signal_peft_v1_20260917.reference_core import pinball_scalar
PANELS=['electricity','electricity_transfer','ettm1']

def score():
 check_seal();marker=read(OUT/'ALL_PREDICTIONS_SAVED.json');assert marker['manifest_sha256']==sha(OUT/'PREDICTIONS.json');assert read(OUT/'EVALUATION_SEAL.json')['at']<marker['at'];rows=[];scalar=0
 for key,r in read(OUT/'PREDICTIONS.json').items():
  panel,kind=r['panel'],r['kind'];d=np.load(ext.data_path(panel)/'E_DISCOVERY_inputs.npz');y0=np.load(ext.data_path(panel)/'E_DISCOVERY_labels.npz')['y'];nc=y0.shape[1]
  if kind=='standard':names=ext.STATES;ids=np.arange(128);offset=np.load(ext.panel_path(panel,kind)/'E_DISCOVERY_offset.npy')
  else:
   meta=read(ext.panel_path(panel,kind)/'manifest.json');names=meta['states'];ids=np.array(meta['origin_indices']);offset=np.load(ext.panel_path(panel,kind)/'offset.npy')
  assert sha(ROOT/r['path'])==r['sha256'];p=np.load(ROOT/r['path'],mmap_mode='r');n=len(ids);y=np.tile(y0[ids].reshape(-1,64),(2*len(names),1))+offset[:,None];s=np.tile(d['sigma'],len(y)//nc)
  metrics=np.concatenate([metric_arrays(p[k:k+256],y[k:k+256],s[k:k+256]) for k in range(0,len(y),256)]).reshape(len(names),2,n,nc,5).mean(1)
  for ci,condition in enumerate(names):
   for oi,origin in enumerate(d['origins'][ids]):rows.append(dict(panel=panel,kind=kind,condition=condition,arm=r['arm'],trained_b=r['trained_b'],received_b=r['received_b'],i=r['i'],o=r['o'],origin=int(origin),nmae=metrics[ci,oi,:,0].mean(),pinball=metrics[ci,oi,:,3].mean()))
   for oi in [0,n-1]:
    for ch in [0,nc-1]:
     ii=[((ci*2+z)*n+oi)*nc+ch for z in range(2)];pp=p[ii].astype(float);yy=y[ii];ss=s[ii];mae=sum(abs(float(pp[z,4,h])-float(yy[z,h]))/float(ss[z]) for z in range(2) for h in range(64))/128;pin=pinball_scalar(pp,yy,ss,np.arange(1,10)/10);np.testing.assert_allclose(metrics[ci,oi,ch,[0,3]],[mae,pin],rtol=1e-10,atol=1e-12);scalar+=2
 f=pd.DataFrame(rows);f.to_csv(OUT/'ORIGIN_SCORES.csv.gz',index=False,compression='gzip');keys=['panel','kind','arm','trained_b','received_b','i','o','origin'];fault=f[f.condition.str.startswith(('POINT','BURST'))].groupby(keys,as_index=False)[['nmae','pinball']].mean();fault['condition']='FAULT';f=pd.concat([f,fault],ignore_index=True);raw=f.groupby([k for k in keys if k!='origin']+['condition'],as_index=False)[['nmae','pinball']].mean();raw.to_csv(OUT/'RAW_SCORES.csv',index=False)
 parent_raw=pd.read_csv(PRIOR/'RAW_SCORES.csv');parent_raw=parent_raw[parent_raw.stage=='fixed1024'];replayed=0
 for r in raw[raw.trained_b==raw.received_b].itertuples():
  q=parent_raw[(parent_raw.panel==r.panel)&(parent_raw.kind==r.kind)&(parent_raw.condition==r.condition)&(parent_raw.arm==r.arm)&(parent_raw.b==r.trained_b)&(parent_raw.i==r.i)&(parent_raw.o==r.o)];assert len(q)==1;np.testing.assert_allclose([r.nmae,r.pinball],[q.iloc[0].nmae,q.iloc[0].pinball],rtol=1e-10,atol=1e-12);replayed+=1
 effects=[];paired=[]
 f['matched']=f.trained_b==f.received_b
 for (panel,kind,condition,arm),g in f.groupby(['panel','kind','condition','arm']):
  origins=sorted(g.origin.unique());v=g.groupby(['origin','matched']).nmae.mean().unstack().reindex(origins);assert v.shape==(len(origins),2);match=v[True].to_numpy();swap=v[False].to_numpy();diff=swap-match
  period=24 if panel.startswith('electricity') else 96;allorig=np.load(ext.data_path(panel)/'E_DISCOVERY_inputs.npz')['origins'];blocks=np.unique(allorig//(7*period));random=np.random.default_rng(90418+PANELS.index(panel));cnt=np.stack([np.bincount(random.integers(0,len(blocks),len(blocks)),minlength=len(blocks)) for _ in range(2000)]);w=cnt[:,np.searchsorted(blocks,np.array(origins)//(7*period))].astype(float);assert (w.sum(1)>0).all();w/=w.sum(1,keepdims=True);boot=w@diff;lo,hi=np.quantile(boot,[.025,.975]);bl,bh=np.quantile(boot,[.0125,.9875]);effects.append(dict(panel=panel,kind=kind,condition=condition,arm=arm,matched_nmae=match.mean(),swapped_nmae=swap.mean(),pairing_penalty=diff.mean(),penalty_pct=100*diff.mean()/match.mean(),ci_low=lo,ci_high=hi,bonferroni2_low=bl,bonferroni2_high=bh))
  for o,x in zip(origins,diff):paired.append(dict(panel=panel,kind=kind,condition=condition,arm=arm,origin=o,pairing_penalty=x))
 pd.DataFrame(effects).to_csv(OUT/'PAIRING_EFFECTS.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'PAIRING_BY_ORIGIN.csv.gz',index=False,compression='gzip');assert len(effects)==120
 save(OUT/'SCORE_VERIFICATION.json',dict(status='VERIFIED',scalar_checks=scalar,original_rows_replayed=replayed,effects=120,all_predictions_saved_before_scoring=True));print('SCORE_VERIFIED',flush=True)
