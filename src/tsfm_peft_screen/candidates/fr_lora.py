import torch
from torch.nn import functional as F

def aligned(origin,delta=24,horizon=48):
    assert 0<delta<horizon
    t0=torch.arange(origin,origin+horizon)[delta:];t1=torch.arange(origin+delta,origin+delta+horizon)[:horizon-delta]
    assert torch.equal(t0,t1)
    return slice(delta,horizon),slice(0,horizon-delta)
def regularizer(arm,raw,base,scale):
    # batch ordered [early channel rows, late channel rows] per pair.
    n=len(raw)//2;assert len(raw)==2*n
    d=(raw-base.detach())/scale[:,None,None];a,b=aligned(0)
    if arm=='FR_LORA':v=d[:n,...,a]-d[n:,...,b]
    elif arm=='RAW_STABILITY_LORA':
        v=raw[:n,...,a]/scale[:n,None,None]-raw[n:,...,b]/scale[n:,None,None]
    elif arm=='F0_ANCHOR_LORA':v=d
    else:return raw.sum()*0
    return F.smooth_l1_loss(v,torch.zeros_like(v))
def objective(task,reg,lam):return task if lam==0 else task+lam*reg
