import numpy as np
import torch
from common import *
def metrics(q,y,sigma):
    e=y[...,None].astype(float)-q.astype(float);levels=np.arange(1,10)/10
    return dict(pinball=(2*np.maximum(levels*e,(levels-1)*e)).mean(-1)/sigma[:,None],nmae=np.abs(e[:,:,4])/sigma[:,None],raw_mae=np.abs(e[:,:,4]),coverage=((y>=q[:,:,0])&(y<=q[:,:,-1])).astype(float),width=(q[:,:,-1]-q[:,:,0])/sigma[:,None],crossing=(np.diff(q,axis=-1)<0).mean(-1))
def predpath(source,seed,arm,role,step):return CACHE/source/'predictions'/f'{arm}_{seed}_{role}_{step}.npz'
def predict(m,d,role,p):
    assert not p.exists(),'Predictions cannot silently be reused'
    packet=d.batch(d.pairs[role]);qs=[]
    with torch.no_grad():
        for i in range(0,len(packet['x']),32):qs.append(m(torch.as_tensor(packet['x'][i:i+32],device='cuda')).cpu().numpy())
    out={k:packet[k] for k in ['y','sigma','pairs']};out['q']=np.concatenate(qs);assert np.isfinite(out['q']).all();p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**out)
    save(p.with_suffix('.json'),dict(sha256=sha(p),examples=len(out['q'])));return out
def load(p):
    assert sha(p)==read(p.with_suffix('.json'))['sha256']
    with np.load(p) as f:return {k:f[k] for k in f.files}
def score(pred):return float(metrics(pred['q'],pred['y'],pred['sigma'])['pinball'].mean())
def adjust(p,c):return p['q'][:,:,4,None]+c['beta']*p['sigma'][:,None,None]+c['alpha']*(p['q']-p['q'][:,:,4,None])
def calibrate(p):
    choices=[]
    for alpha in [.5,.75,1,1.25,1.5,2,3]:
        for beta in [-.5,-.25,0,.25,.5]:
            c=dict(alpha=alpha,beta=beta);loss=score({**p,'q':adjust(p,c)});choices.append((loss,(alpha-1)**2+beta*beta,alpha,abs(beta),beta))
    b=min(choices);return dict(alpha=b[2],beta=b[4],cal_pinball=b[0],grid_evaluations=35)
