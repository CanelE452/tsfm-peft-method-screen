import gzip
import numpy as np
import pandas as pd
from common import *
class Data:
    def __init__(self,source):
        self.source=source;old=read(ROOT/'results/rollout_uncertainty_peft_v1_20260921/DATA_AND_SPLIT_AUDIT.json')['sources'][source]
        raw=ROOT/old['raw_path'];assert sha(raw)==old['raw_sha256'];prepared=ROOT/old['prepared_npz']['path'];assert sha(prepared)==old['prepared_npz']['sha256']
        with np.load(prepared,allow_pickle=True) as f:cached=f['values'];ids=f['ids'].astype(str)
        if source=='Electricity':
            with gzip.open(raw,'rt') as f:matrix=pd.read_csv(f,header=None).to_numpy(dtype=np.float32)
            ids=ids[:8];values=matrix[:,ids.astype(int)];assert np.array_equal(values,cached[:,:8])
        else:
            frame=pd.read_csv(raw);times=pd.to_datetime(frame['date']);assert ((times.diff().dropna())==pd.Timedelta(hours=1)).all() and not times.duplicated().any()
            values=frame[list(ids)].to_numpy(dtype=np.float32);assert np.array_equal(values,cached)
        self.values=values;self.ids=ids;self.edges=[0,int(len(values)*.6),int(len(values)*.7),int(len(values)*.8),len(values)]
        assert np.isfinite(values).all();self.sigma=values[:self.edges[1]].astype(float).std(0);assert (self.sigma>0).all()
        self.pairs={}
        for role,lo,hi in zip(['TRAIN','CAL','VAL','TEST'],self.edges[:-1],self.edges[1:]):
            origins=np.arange(max(lo,256),hi-64+1,1 if role=='TRAIN' else 64)
            self.pairs[role]=np.array([(s,o) for o in origins for s in range(len(ids))],dtype=np.int64)
            assert len(origins)>7
        # No overlapping contexts within a series; selection ignores targets and performance.
        choices=np.arange(256,self.edges[1]+1,256);basis=[]
        for i in range(256):
            s=i%len(ids);j=i//len(ids);count=(255-s)//len(ids)+1;ix=int(np.floor(j*len(choices)/count));basis.append((s,int(choices[ix])))
        self.basis_pairs=np.array(basis,dtype=np.int64)
        for s in range(len(ids)):assert (np.diff(self.basis_pairs[self.basis_pairs[:,0]==s,1])>=256).all()
        self.audit=dict(status='PASS',upstream=old['upstream'],raw_path=raw,raw_sha256=sha(raw),prepared_input_sha256=sha(prepared),selected_ids=ids.tolist(),shape=list(values.shape),edges=self.edges,
            sigma=self.sigma.tolist(),roles={k:dict(examples=len(v),first=v[0].tolist(),last=v[-1].tolist()) for k,v in self.pairs.items()},basis_contexts=256,basis_contexts_disjoint_per_series=True,context=256,horizon=64,prior_exposed_development_data=True)
    def batch(self,pairs):
        return dict(x=np.stack([self.values[o-256:o,s] for s,o in pairs]),y=np.stack([self.values[o:o+64,s] for s,o in pairs]),sigma=self.sigma[pairs[:,0]].astype(np.float32),pairs=pairs)
    def basis_x(self):return np.stack([self.values[o-256:o,s] for s,o in self.basis_pairs])
    def schedule(self,seed):
        rng=np.random.default_rng(seed+(0 if self.source=='Electricity' else 10000));p=self.pairs['TRAIN'];return p[rng.integers(0,len(p),size=(512,8))]
