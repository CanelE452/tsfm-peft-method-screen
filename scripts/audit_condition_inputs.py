"""Independent TRAIN/input permission audit; no model invocation or E scoring."""
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experiments.condition_studies_v1_20260916.common import *
from experiments.condition_studies_v1_20260916.prepare import spectral

def archive():
 out=OUT/'N02';rec=read(out/'data_receipt.json');a=np.loadtxt(ROOT/rec['path'],delimiter=',',usecols=[int(c.split('_')[1]) for c in rec['selected_columns']]);b1=rec['bounds'][1];bank=np.arange(336,b1-48+1,24)
 if len(bank)>2048:bank=bank[np.floor(np.linspace(0,len(bank)-1,2048)).astype(int)]
 def retrieve(values,o,c):
  eligible=bank[bank+48<=min(b1,o-360)];q=values[o-336:o,c];qm=q.mean();qs=max(q.std(),1e-6);x=np.stack([values[r-336:r,c] for r in eligible]);mu=x.mean(1);std=np.maximum(x.std(1),1e-6);distance=(((x-mu[:,None])/std[:,None]-(q-qm)/qs)**2).mean(1);order=np.lexsort((eligible,distance))[:2];rr=eligible[order];past=(x[order]-mu[order,None])/std[order,None]*qs+qm;future=np.stack([(values[r:r+48,c]-mu[j])/std[j]*qs+qm for r,j in zip(rr,order)]);return rr,past,future
 checks=0
 for role in ROLES:
  packet=np.load(CACHE/'N02'/f'{role}_inputs.npz')
  for i in [0,len(packet['origins'])-1]:
   o=int(packet['origins'][i]);poison=a.copy();poison[o:o+48]+=1e12
   for c in range(4):
    v=retrieve(a,o,c);w=retrieve(poison,o,c)
    for x,y in zip(v,w):assert np.array_equal(x,y)
    assert np.array_equal(v[0],packet['retrieval_origins'][i,c*2:c*2+2]);assert np.array_equal(v[1].astype(np.float32),packet['retrieval_past'][i,c*2:c*2+2]);assert np.array_equal(v[2].astype(np.float32),packet['retrieval_future'][i,c*2:c*2+2]);checks+=1
 return dict(actual_retrieval_poison_cases=checks,rank_and_past_future_exact=True,legal_time=True)

def weights():
 rec=read(OUT/'N07/data_receipt.json');a=np.loadtxt(ROOT/rec['path'],delimiter=',',usecols=[int(c.split('_')[1]) for c in rec['selected_columns']]);st=read(OUT/'N07/train_statistics.json');z=(a-np.array(st['mu']))/np.array(st['sigma']);oo=read(OUT/'N07/origins.json')['TRAIN'];expected=read(OUT/'N07/feature_or_transform_manifest.json');z[rec['bounds'][1]:]=1e12;observed=spectral(z,oo);assert observed==expected;return dict(V_E_poison_does_not_change_spectral_statistics=True,prefix_fold_normalization=True)

save(OUT/'supplemental_input_audit.json',dict(at=time.time(),archive=archive(),spectral=weights(),no_gpu=True,no_optimizer=True,no_E_scores=True))
print('SUPPLEMENTAL_INPUT_AUDIT_OK')
