"""Support-complete phase experiment; original pilot code remains immutable."""
import torch
from torch import nn
from torch.nn import functional as F
from ..backbone import load_base
from ..lora import attach
from .patchphase import PhaseAdapter


def patch_phase(x,phase):
    if not isinstance(phase,int) or not 0<=phase<16 or x.shape[-1]!=336:
        raise ValueError('Expected integer phase 0..15 and 336 observations')
    return F.pad(x,(phase,(-phase)%16),value=float('nan')).reshape(*x.shape[:-1],-1,16),phase


def unpatch(value):
    p,phase=value
    return p.flatten(-2)[...,phase:phase+336]


def prepare_context(m,x,mask,phase):
    norm,locscale=m.instance_norm(x)
    observed=torch.isfinite(x) if mask is None else mask.bool() & torch.isfinite(x)
    p,_=patch_phase(norm,phase);pm,_=patch_phase(observed.to(x.dtype),phase)
    pm=torch.nan_to_num(pm,nan=0.);p=torch.where(pm>0,p,0.)
    t=torch.arange(-336,0,device=x.device,dtype=x.dtype).expand_as(x)
    pt,_=patch_phase(t,phase);pt=torch.nan_to_num(pt,nan=0.)/m.chronos_config.time_encoding_scale
    return torch.cat([pt,p,pm],-1).to(m.dtype),pm.sum(-1)>0,locscale


class PhaseModel(nn.Module):
    def __init__(self,arm,seed):
        super().__init__();self.arm=arm;self.core=load_base();attach(self.core,seed)
        torch.manual_seed(seed+2000)
        self.adapter=None if arm=='standard' else PhaseAdapter(self.core.config.d_model,arm=='conditioned').cuda()
    def forward(self,x,g,phase):
        previous=self.core._prepare_patched_context
        self.core._prepare_patched_context=lambda context,context_mask=None:prepare_context(self.core,context,context_mask,phase)
        try:
            enc,(loc,scale),_,_=self.core.encode(context=x,group_ids=g,num_output_patches=3)
        finally:self.core._prepare_patched_context=previous
        h=enc.last_hidden_state[:,-3:]
        if self.adapter is not None:
            a=self.adapter(h,phase);h=torch.empty_strided(h.size(),h.stride(),device=h.device,dtype=h.dtype).copy_(a)
        z=self.core.output_patch_embedding(h).reshape(len(x),3,21,16).permute(0,2,1,3).reshape(len(x),21,48).float()
        return z,z.sinh()*scale[:,None,:]+loc[:,None,:],loc,scale
