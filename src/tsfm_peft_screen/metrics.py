"""Float64 equal-channel 2-pinball and independent scalar replay."""
import numpy as np
from .backbone import QUANTILES

def score(p,y,scale,mask=None):
    p=np.sort(np.asarray(p,dtype=np.float64),axis=2);y=np.asarray(y,dtype=np.float64);s=np.asarray(scale,dtype=np.float64)
    assert p.ndim==4 and y.shape==p.shape[:2]+p.shape[3:] and s.shape==(p.shape[1],)
    assert np.isfinite(p).all() and np.isfinite(s).all() and (s>0).all()
    valid=np.isfinite(y) if mask is None else np.isfinite(y)&mask
    n=valid.sum((0,2));active=n>0
    if not active.any():raise ValueError('No targets')
    e=y[:,:,None,:]-p;q=np.array(QUANTILES)[None,None,:,None]
    pin=np.where(valid[:,:,None,:],2*np.maximum(q*e,(q-1)*e),0.)
    per=(pin.sum((0,3))[active]/n[active,None]/s[active,None]).mean(1)
    def mean(a):return float((np.where(valid,a,0).sum((0,2))[active]/n[active]).mean())
    med=p[:,:,10];lo=p[:,:,2];hi=p[:,:,18]
    return dict(scaled_2pinball=float(per.mean()),median_mae=mean(abs(y-med)),qmean_mse=mean((y-p.mean(2))**2),interval80_coverage=mean((y>=lo)&(y<=hi)),interval80_width=mean(hi-lo))

def independent(p,y,scale):
    # Separate channel/quantile loops and asymmetric residual branches.
    totals=[]
    for c in range(y.shape[1]):
        v=np.isfinite(y[:,c]);obs=y[:,c][v].astype(np.float64)
        pc=np.sort(p[:,c].astype(np.float64),axis=1)
        for k,q in enumerate(QUANTILES):
            residual=obs-pc[:,k][v]
            losses=np.empty_like(residual);positive=residual>=0
            losses[positive]=residual[positive]*(2*q)
            losses[~positive]=(-residual[~positive])*(2*(1-q))
            totals.append(losses.sum()/len(obs)/scale[c])
    return float(np.mean(totals))
def replay(path):
    with np.load(path,allow_pickle=False) as z:
        a=score(z['prediction'],z['target'],z['scale'])['scaled_2pinball'];b=independent(z['prediction'],z['target'],z['scale'])
    assert abs(a-b)<=1e-10
    return abs(a-b)
