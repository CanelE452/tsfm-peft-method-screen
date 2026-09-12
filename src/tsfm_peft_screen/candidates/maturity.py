import torch
from torch import nn
from torch.nn import functional as F

def matured_mask(issued_origin,now,horizon=48):
    assert now>=issued_origin
    return torch.arange(issued_origin,issued_origin+horizon)<now

def preservation(pred,pre,available,scale):
    unknown=(~available)[None,None,:].expand_as(pred)
    d=(pred-pre.detach())/scale[:,None,None]
    return F.smooth_l1_loss(d[unknown],torch.zeros_like(d[unknown])) if unknown.any() else pred.sum()*0

class Calibration(nn.Module):
    """TAFAS equation (3), per-channel temporal input and output GCM.
    Fixed stride24 replaces PAAS for budget fairness; no retrospective PA.
    """
    def __init__(self,channels):
        super().__init__();self.c=channels
        self.w_in=nn.Parameter(torch.zeros(channels,336,336));self.b_in=nn.Parameter(torch.zeros(channels,336));self.g_in=nn.Parameter(torch.ones(channels)*.01)
        self.w_out=nn.Parameter(torch.zeros(channels,48,48));self.b_out=nn.Parameter(torch.zeros(channels,48));self.g_out=nn.Parameter(torch.ones(channels)*.01)
    def input(self,x):return x+torch.tanh(self.g_in)[:,None]*(torch.einsum('cij,cj->ci',self.w_in,x)+self.b_in)
    def output(self,p):return p+torch.tanh(self.g_out)[:,None,None]*(torch.einsum('cij,cqj->cqi',self.w_out,p)+self.b_out[:,None,:])
