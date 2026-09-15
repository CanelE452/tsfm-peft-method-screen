import math
import numpy as np
import torch
from common import ROOT,read,sha

def stage(cache,allow_existing=False):
    old=read(ROOT/'results/priority12_20260915/channel_data_manifest.json');data={}
    for name in ['electricity','traffic']:
        r=old[name];path=ROOT/r['raw_path'] if not str(r['raw_path']).startswith('/') else r['raw_path'];assert sha(path)==r['raw_sha256']
        if name=='traffic':assert r['raw_sha256']=='c7be5a00519d344a5ec0eabdbfec5ea0c7dd1eed5f9b1a3843a93bb88086a56d'
        raw=np.loadtxt(path,delimiter=',',dtype=np.float64);n,columns=raw.shape;assert n=={'electricity':26304,'traffic':17544}[name],'BLOCKED_DATA row count'
        t1=int(.7*n);t2=int(.8*n);eligible=np.flatnonzero(np.isfinite(raw[:t1]).all(0)&(raw[:t1].std(0)>1e-6));assert len(eligible)>=32
        ix=eligible[:32];x=raw[:,ix];mean=x[:t1].mean(0);std=x[:t1].std(0);values=((x-mean)/std).astype(np.float32)
        train=np.arange(96,t1-96+1,24)
        if len(train)>512:train=train[np.linspace(0,len(train)-1,512,dtype=int)]
        origins=dict(train=train.tolist(),validation=list(range(t1,t2-96+1,96)),evaluation=list(range(t2,n-96+1,96)))
        assert len(train)>=128 and len(origins['validation'])>=8 and len(origins['evaluation'])>=8
        assert [len(origins[k]) for k in ['train','validation','evaluation']]=={'electricity':[512,27,54],'traffic':[504,18,36]}[name]
        files={}
        for split,end in [('train',t1),('development',t2),('evaluation',n)]:
            p=cache/f'{name}_{split}.npz'
            if p.exists():
                assert allow_existing
                with np.load(p) as saved:
                    assert np.array_equal(saved['values'],values[:end],equal_nan=True) and np.array_equal(saved['mean'],mean) and np.array_equal(saved['std'],std)
            else:np.savez_compressed(p,values=values[:end],mean=mean,std=std)
            files[str(p.relative_to(ROOT))]=sha(p)
        data[name]=dict(raw_path=str(path),raw_sha256=r['raw_sha256'],rows=n,raw_columns=columns,indices=ix.tolist(),channel_ids=[f'column_{i:03}' for i in ix],t1=t1,t2=t2,mean=mean.tolist(),std=std.tolist(),origins=origins,staged=files,exposure='Existing Electricity/Traffic sources and E periods, including previous 64-channel experiment; not independent confirmation',missing={k:int((~np.isfinite(x[a:b])).sum()) for k,a,b in [('train',0,t1),('validation',t1,t2),('evaluation',t2,n)]})
    return data

def load(cache,name,split):
    with np.load(cache/f'{name}_{split}.npz') as z:return z['values'],z['std']
def batch(values,origins,device='cuda'):
    x=np.stack([values[o-96:o].T for o in origins]);y=np.stack([values[o:o+96].T for o in origins]);assert x.shape[1:]==(32,96)
    # Each context is filled causally from its own previous observations, then train mean=0.
    x=x.copy()
    for t in range(96):x[:,:,t]=np.where(np.isfinite(x[:,:,t]),x[:,:,t],x[:,:,t-1] if t else 0.)
    return torch.tensor(x,device=device),torch.tensor(y,device=device)
def loss(pred,y,counts=None):
    valid=torch.isfinite(y);count=valid.sum((0,2)) if counts is None else counts
    eligible=count>0;err=torch.where(valid,pred.float()-y,0.)
    return ((err.square().sum((0,2))/count.clamp_min(1))[eligible]).mean()
def metrics(pred,y,std):
    p=np.asarray(pred,dtype=np.float64);y=np.asarray(y,dtype=np.float64);valid=np.isfinite(y);assert p.shape==y.shape and np.isfinite(p).all()
    counts=valid.sum((0,2));err=np.where(valid,p-y,0.);good=counts>0
    mse=np.divide((err**2).sum((0,2)),counts,out=np.full(len(counts),np.nan),where=good);mae=np.divide(abs(err).sum((0,2)),counts,out=np.full(len(counts),np.nan),where=good)
    clean=lambda a:[float(v) if np.isfinite(v) else None for v in a]
    return dict(mse=float(mse[good].mean()) if good.any() else None,mae=float(mae[good].mean()) if good.any() else None,channel_mse=clean(mse),channel_mae=clean(mae),channel_raw_mae=clean(mae*std),valid_counts=counts.tolist(),eligible_channels=int(good.sum()))
def independent(p,y):
    vals=[]
    for c in range(y.shape[1]):
        pairs=[(float(a),float(b)) for a,b in zip(p[:,c].flat,y[:,c].flat) if np.isfinite(b)]
        if pairs:vals.append(math.fsum((a-b)**2 for a,b in pairs)/len(pairs))
    return math.fsum(vals)/len(vals) if vals else None
