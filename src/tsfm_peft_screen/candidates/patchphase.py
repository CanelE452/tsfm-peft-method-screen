"""Boundary change via masked padding; actual times and values remain fixed."""
import torch
from torch import nn
from torch.nn import functional as F

def patch_phase(x,phase):
    assert phase in (0,4,8,12) and x.shape[-1]==336
    right=0 if phase==0 else 16-phase
    padded=F.pad(x,(phase,right),value=float('nan'))
    return padded.reshape(*x.shape[:-1],-1,16),phase

def unpatch(value):
    p,phase=value
    return p.flatten(-2)[...,phase:phase+336]

def prepare_context(m,x,mask,phase):
    # Normalize the exact original 336 values, then pad normalized data,
    # explicit observation mask and absolute-time encoding together.
    norm,locscale=m.instance_norm(x);obs=(torch.isfinite(x) if mask is None else mask.bool())
    p,_=patch_phase(norm,phase);pm,_=patch_phase(obs.to(x.dtype),phase);pm=torch.nan_to_num(pm,nan=0.)
    p=torch.where(pm>0,p,0.)
    times=torch.arange(-336,0,device=x.device,dtype=x.dtype).expand_as(x)
    pt,_=patch_phase(times,phase);pt=torch.nan_to_num(pt,nan=0.)/m.chronos_config.time_encoding_scale
    return torch.cat([pt,p,pm],-1).to(m.dtype),pm.sum(-1)>0,locscale

class PhaseAdapter(nn.Module):
    def __init__(self,width,conditioned):
        super().__init__();self.down=nn.Linear(width,8,bias=False);self.up=nn.Linear(8,width,bias=False);nn.init.zeros_(self.up.weight)
        self.gate=nn.Linear(2,8) if conditioned else None
        if self.gate is not None:nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
    def forward(self,h,phase):
        a=torch.nn.functional.silu(self.down(h))
        if self.gate is not None:
            angle=h.new_tensor(phase*2*torch.pi/16);r=torch.stack([angle.sin(),angle.cos()]);a=a*(2*self.gate(r).sigmoid())
        return h+self.up(a)
