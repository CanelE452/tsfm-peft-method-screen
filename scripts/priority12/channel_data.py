import numpy as np
import torch
from .common import ROOT,read,save,sha

def stage(cache):
    meta=read(ROOT/'data/processed/electricity/manifest.json');e=list(meta['sources'].items());assert len(e)==1
    t=read(ROOT/'research/overnight_20260913/data_receipt.json')['datasets']['traffic']
    paths={'electricity':(e[0][0],e[0][1]),'traffic':(str(ROOT/t['file']),t['sha256'])};data={}
    for name,(path,h) in paths.items():
        assert sha(path)==h
        raw=np.loadtxt(path,delimiter=',',dtype=np.float64);n,columns=raw.shape;te=int(.6*n);ve=int(.8*n)
        train=raw[:te];eligible=np.flatnonzero(np.isfinite(train).all(0)&(np.std(train,axis=0)>1e-6));assert len(eligible)>=64,'BLOCKED_CHANNEL_SUPPORT'
        selected=eligible[:64];x=raw[:,selected];mean=x[:te].mean(0);std=x[:te].std(0);values=((x-mean)/std).astype(np.float32)
        oo=dict(train=list(range(512,te-96+1,24)),validation=list(range(te,ve-96+1,96)),evaluation=list(range(ve,n-96+1,96)))
        assert len(oo['validation'])>=16 and len(oo['evaluation'])>=16
        assert all(o+96<=te for o in oo['train']) and all(te<=o and o+96<=ve for o in oo['validation'])
        ids=[f'column_{i:03d}' for i in selected];files={}
        for split,end in [('train',te),('development',ve),('evaluation',n)]:
            p=cache/f'{name}_{split}.npz';np.savez_compressed(p,values=values[:end],mean=mean,std=std);files[str(p.relative_to(ROOT))]=sha(p)
        data[name]=dict(raw_path=path,raw_sha256=h,rows=n,raw_columns=columns,selected_indices=selected.tolist(),channel_ids=ids,train_end=te,val_end=ve,mean=mean.tolist(),std=std.tolist(),origins=oo,staged=files,
            missing=dict(train=int((~np.isfinite(x[:te])).sum()),validation=int((~np.isfinite(x[te:ve])).sum()),evaluation=int((~np.isfinite(x[ve:])).sum())),
            preprocessing='Train-only eligibility and mean/std; mechanical E staging only, no model scoring',input_mask_api='MOMENT mask [B,T], no per-channel mask; missing inputs filled train standardized zero; missing targets masked in scoring')
    return data

def load(cache,name,split):
    with np.load(cache/f'{name}_{split}.npz') as z:return z['values'],z['mean'],z['std']
def batch(values,origins,device='cuda'):
    x=np.stack([values[o-512:o].T for o in origins]);y=np.stack([values[o:o+96].T for o in origins]);assert x.shape[1:]==(64,512) and y.shape[1:]==(64,96)
    return torch.tensor(np.nan_to_num(x,nan=0.,posinf=0.,neginf=0.),device=device),torch.tensor(y,device=device)
def metrics(pred,target,std):
    pred=np.asarray(pred,dtype=np.float64);target=np.asarray(target,dtype=np.float64);valid=np.isfinite(target);assert pred.shape==target.shape and np.isfinite(pred).all()
    error=np.where(valid,pred-target,0.);count=valid.sum(axis=(0,2));assert (count>0).all()
    mse=(error**2).sum(axis=(0,2))/count;mae=np.abs(error).sum(axis=(0,2))/count
    return dict(mse=float(mse.mean()),mae=float(mae.mean()),channel_standardized_mse=mse.tolist(),channel_raw_mse=(mse*np.asarray(std)**2).tolist(),channel_raw_mae=(mae*np.asarray(std)).tolist())
def independent(pred,target):
    p=np.asarray(pred,dtype=np.float64);y=np.asarray(target,dtype=np.float64);answers=[]
    # Separate scalar accumulation, equal channel weighting.
    import math
    for c in range(p.shape[1]):
        pairs=[(float(a),float(b)) for a,b in zip(p[:,c].flat,y[:,c].flat) if np.isfinite(b)]
        assert pairs;answers.append(math.fsum((a-b)**2 for a,b in pairs)/len(pairs))
    return math.fsum(answers)/len(answers)
