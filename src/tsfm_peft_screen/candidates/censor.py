import torch
from torch.nn import functional as F
from ..backbone import QUANTILES,native_loss

def cdf(pred,value):
    # piecewise monotone interpolation; linear tails extended one adjacent
    # knot spacing to probabilities 0/1. Duplicate knots use rightmost CDF.
    p=pred.sort(dim=1).values.transpose(1,2).contiguous();q=torch.tensor(QUANTILES,device=p.device,dtype=p.dtype)
    delta_lo=(p[...,1]-p[...,0]).clamp_min(1e-4);delta_hi=(p[...,-1]-p[...,-2]).clamp_min(1e-4)
    knots=torch.cat([(p[...,0]-delta_lo)[...,None],p,(p[...,-1]+delta_hi)[...,None]],-1)
    probs=torch.cat([q.new_zeros(1),q,q.new_ones(1)])
    ix=torch.searchsorted(knots.contiguous(),value[...,None].contiguous(),right=True).squeeze(-1).clamp(1,len(probs)-1)
    lo=knots.gather(-1,(ix-1)[...,None]).squeeze(-1);hi=knots.gather(-1,ix[...,None]).squeeze(-1)
    frac=((value-lo)/(hi-lo).clamp_min(1e-6)).clamp(0,1)
    return probs[ix-1]+frac*(probs[ix]-probs[ix-1])

def objective(arm,z,raw,sale,censored,loc,scale,base,train_scale,lc,lp):
    if arm=='NAIVE_LORA':return native_loss(z,sale,loc,scale)
    uncensored=sale.masked_fill(censored,float('nan'));task=native_loss(z,uncensored,loc,scale)
    if arm=='DROP_CENSORED_LORA':return task
    surv=-(1-cdf(raw,sale)).clamp_min(1e-6).log();c=(surv*censored).mean()
    out=task+lc*c
    if arm=='CENSOR_PRESERVE_LORA':
        d=(raw-base.detach())/train_scale[:,None,None];v=(~censored)[:,None,:].expand_as(d)
        out=out+lp*F.smooth_l1_loss(d[v],torch.zeros_like(d[v]))
    return out
