"""Observed-context-only restoration; reuses immutable v1 Bolt/LoRA adapter."""
import torch
from torch import nn
from experiments.outlier_signal_peft_v1_20260917.model import (
    ForecastModel as OldModel, attach_lora, robust_scale, loss_2pinball, preprocess)

def coordinates(x, sigma):
    m,r=robust_scale(x,sigma)
    clip=torch.maximum(torch.minimum(x,m+6*r),m-6*r)
    d=(x-m)/r; excess=(x-clip).abs()/r; extreme=d.abs()>3
    # Trailing truncated window: all terms refer only to observed past.
    pos=(extreme & (d>0)).to(x);neg=(extreme & (d<0)).to(x)
    def trailing(v):
        c=nn.functional.pad(v.cumsum(-1),(1,0))
        end=torch.arange(1,x.shape[-1]+1,device=x.device);start=(end-8).clamp_min(0)
        return (c[:,end]-c[:,start])/(end-start)
    p=torch.where(d>0,trailing(pos),trailing(neg))*extreme
    return clip,excess,extreme.to(x),p

def restore_input(x, clip, gate):
    return clip+gate*(x-clip)

class ForecastModel(OldModel):
    def __init__(self,base,arm,seed):
        super().__init__(base,'A5' if arm=='B2' else 'A1',seed)
        self.new_arm=arm
        if arm in ['B4','B5']:
            self.gate=nn.Parameter(torch.tensor([-4.,0.,0.] if arm=='B4' else [-4.,0.,0.,0.]))

    def transform(self,observed,sigma,persistence_mode='normal'):
        if self.new_arm=='B0':return observed,None,None,None
        clip,e,indicator,p=coordinates(observed,sigma)
        if self.new_arm in ['B1','B2']:g=torch.zeros_like(p)
        elif self.new_arm=='B3':g=p
        else:
            z=torch.log1p(e)
            if self.new_arm=='B4':logit=self.gate[0]+self.gate[1]*z+self.gate[2]*indicator
            else:
                if persistence_mode=='zero':p=torch.zeros_like(p)
                elif persistence_mode=='permute':
                    order=torch.randperm(512,generator=torch.Generator().manual_seed(83400)).to(p.device)
                    p=p[:,order]
                else:assert persistence_mode=='normal'
                logit=self.gate[0]+self.gate[1]*z+self.gate[2]*p+self.gate[3]*z*p
            g=torch.sigmoid(logit)
        return restore_input(observed,clip,g),g,p,e

    def forward(self,observed,sigma,residual_mode='normal',persistence_mode='normal'):
        if self.new_arm=='B2':
            return super().forward(observed,sigma,residual_mode=residual_mode)
        x,_,_,_=self.transform(observed,sigma,persistence_mode)
        return super().forward(x,sigma)
