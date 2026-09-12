import numpy as np
import torch

def corrupt(x,kind):
    a=np.array(x,copy=True);observed=np.isfinite(a)
    if kind.startswith('block'):
        n=int(kind[5:]);observed[:,-n:]=False
    elif kind.startswith('refresh'):
        n=int(kind[7:]);observed &= (np.arange(a.shape[-1])[None,:]%n==0)
    elif kind=='stale':observed[:,-24:]=False
    elif kind!='clean':raise ValueError(kind)
    a[~observed]=np.nan
    # Explicit stale carry differs from a missing mask: gate still knows age.
    if kind=='stale':
        for t in range(1,a.shape[-1]):a[:,t]=np.where(np.isfinite(a[:,t]),a[:,t],a[:,t-1])
    age=np.zeros_like(a);recent=np.zeros_like(a)
    for t in range(a.shape[-1]):
        age[:,t]=np.where(observed[:,t],0,1+(age[:,t-1] if t else 0))
        recent[:,t]=observed[:,max(0,t-23):t+1].mean(1)
    state=np.stack([observed.astype(float),age/336,recent],-1).astype(np.float32)
    return a,state

def token_state(r,reg=True):
    # Patch means are computed only from the 336-step context. Future/REG
    # tokens use the last observed state, never a future observation mask.
    n=r.shape[0];patch=r.reshape(n,21,16,3).mean(2)
    tail=r[:,-1:,:].expand(n,3+int(reg),3)
    return torch.cat([patch,tail],1)
