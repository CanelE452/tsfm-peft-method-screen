import time
import numpy as np
import torch
from common import *
from engine import tensors,distribution

def metric(z,p,y,sigma,native_q=None):
    z=np.asarray(z,dtype=float);p=np.broadcast_to(np.asarray(p,dtype=float),z.shape).copy();p/=p.sum(-1,keepdims=True)
    order=np.argsort(z,axis=-1,kind='stable');zs=np.take_along_axis(z,order,-1);ps=np.take_along_axis(p,order,-1)
    mass=np.cumsum(ps,-1);value=np.cumsum(ps*zs,-1)
    half=(ps*(zs*(mass-ps)-(value-ps*zs))).sum(-1)
    crps=(p*np.abs(z-y[...,None])).sum(-1)-half
    levels=np.arange(1,10)/10
    q=np.stack([np.take_along_axis(zs,np.minimum((mass<t).sum(-1),zs.shape[-1]-1)[...,None],-1)[...,0] for t in levels],-1)
    e=y[...,None]-q
    mae=np.abs(y-q[...,4]);pin=(2*np.maximum(levels*e,(levels-1)*e)).mean(-1)
    out=dict(crps=crps/sigma[:,None],pinball=pin/sigma[:,None],nmae=mae/sigma[:,None],raw_mae=mae,
             coverage=((y>=q[...,0])&(y<=q[...,-1])).astype(float),width=(q[...,-1]-q[...,0])/sigma[:,None],negative_mass=(p*(z<0)).sum(-1))
    if native_q is not None:
        ee=y[...,None]-native_q
        out['native_pinball']=(2*np.maximum(levels*ee,(levels-1)*ee)).mean(-1)/sigma[:,None]
    return out

def quantile(z,p,level=.5):
    order=np.argsort(z,axis=-1,kind='stable');zs=np.take_along_axis(z,order,-1);ps=np.take_along_axis(np.broadcast_to(p,z.shape),order,-1)
    mass=ps.cumsum(-1)/ps.sum(-1,keepdims=True)
    ix=np.minimum((mass<level).sum(-1),z.shape[-1]-1)
    return np.take_along_axis(zs,ix[...,None],-1)[...,0]

def apply_cal(pred,coefficients):
    z=pred['z'].astype(float).copy();median=quantile(z,pred['p'])
    for row in coefficients:
        lo,hi=row['span'];a=row['alpha'];b=row['beta']
        z[:,lo:hi]=median[:,lo:hi,None]+b*pred['sigma'][:,None,None]+a*(z[:,lo:hi]-median[:,lo:hi,None])
    return z

def calibrate(pred,c):
    spans=[(0,64),(64,128)] if c=='A' else [(i,i+1) for i in range(8)];rows=[]
    for lo,hi in spans:
        z=pred['z'][:,lo:hi].astype(float);p=pred['p'][:,lo:hi];median=quantile(z,p)
        p=p.astype(float);p=p/p.sum(-1,keepdims=True)
        order=np.argsort(z,axis=-1);zs=np.take_along_axis(z,order,-1);ps=np.take_along_axis(p,order,-1)
        before=ps.cumsum(-1)-ps;before_value=(ps*zs).cumsum(-1)-ps*zs
        half=(ps*(zs*before-before_value)).sum(-1)
        choices=[]
        for a in [.5,.75,1,1.25,1.5,2,3]:
            for b in [-.5,-.25,0,.25,.5]:
                zz=median[...,None]+b*pred['sigma'][:,None,None]+a*(z-median[...,None])
                loss=(((p*np.abs(zz-pred['y'][:,lo:hi,None])).sum(-1)-a*half)/pred['sigma'][:,None]).mean()
                choices.append((float(loss),(a-1)**2+b*b,a,abs(b),b))
        best=min(choices);rows.append(dict(span=[lo,hi],alpha=best[2],beta=best[4],cal_score=best[0],grid_evaluations=35,
                                         tie_rule='CRPS, squared Euclidean distance to identity, alpha, abs(beta), beta'))
    return rows

def score(pred,c,cal=None):
    z=pred['z'] if cal is None else apply_cal(pred,cal)
    values=metric(z,pred['p'],pred['y'],pred['sigma'])['crps']
    return float(values[:,64:].mean() if c=='A' else values.mean())

def path(c,seed,arm,role,step='selected'):return CACHE/c/'predictions'/str(seed)/arm/f'{role}_{step}.npz'

def loadpred(p):
    assert sha(p)==read(p.with_suffix('.json'))['sha256']
    with np.load(p) as z:return {k:z[k] for k in z.files}

def predict(model,packet,c,p=None,batch=8):
    if p is not None and p.exists():return loadpred(p)
    arrays=[];weights=[];start=time.perf_counter()
    with torch.no_grad():
        for i in range(0,len(packet['x']),batch):
            b=tensors({k:v[i:i+batch] for k,v in packet.items()});z,w,_=distribution(model,b,c)
            arrays.append(z.cpu().numpy());weights.append(w.cpu().numpy())
    out={k:packet[k] for k in ['y','sigma','pairs']};out.update(z=np.concatenate(arrays),p=np.concatenate(weights))
    if c=='A' and getattr(model,'arm',None)=='F0_NATIVE':
        first=out['z'][:,:64,:9]
        tail=np.quantile(out['z'][:,64:],np.arange(1,10)/10,axis=-1).transpose(1,2,0)
        out['native_q']=np.concatenate([first,tail],axis=1)
    assert np.isfinite(out['z']).all()
    if p is not None:
        p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**out)
        save(p.with_suffix('.json'),dict(sha256=sha(p),seconds=time.perf_counter()-start,examples=len(out['z']),role=p.stem.split('_')[0],bytes=p.stat().st_size))
    return out
