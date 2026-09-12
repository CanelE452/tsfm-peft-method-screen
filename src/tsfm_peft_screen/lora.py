"""Independent rank-8 LoRA with optional availability gate or additive features."""
import math
from contextlib import contextmanager
import torch
from torch import nn
MODULES=[f'encoder.block.{b}.layer.{a}.self_attention.{p}' for b in range(12) for a in (0,1) for p in ('q','k','v','o')]
class LowRank(nn.Module):
    def __init__(self,base,mode='standard',group_attention=False):
        super().__init__();self.base=base;self.mode=mode;self.group_attention=group_attention;self.enabled=True;self.state=None
        self.lora_A=nn.Parameter(torch.empty(8,base.in_features,device=base.weight.device));nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,8,device=base.weight.device))
        if mode in ('feature','freshness'):
            self.condition=nn.Linear(3,8,device=base.weight.device);nn.init.zeros_(self.condition.weight);nn.init.zeros_(self.condition.bias)
    def forward(self,x):
        out=self.base(x)
        if not self.enabled:return out
        a=nn.functional.linear(x,self.lora_A)
        if self.mode in ('feature','freshness'):
            assert self.state is not None
            s=self.state.transpose(0,1) if self.group_attention else self.state
            assert s.shape[:2]==x.shape[:2],(s.shape,x.shape)
            r=self.condition(s)
            a=a*(2*torch.sigmoid(r)) if self.mode=='freshness' else a+r
        return out+2*nn.functional.linear(a,self.lora_B)

def attach(m,seed=30000,mode='standard'):
    torch.manual_seed(seed+1000)
    for index,name in enumerate(MODULES):
        torch.manual_seed(seed+1000+index)
        parent,leaf=name.rsplit('.',1);p=m.get_submodule(parent)
        setattr(p,leaf,LowRank(getattr(p,leaf),mode,'.layer.1.' in name))
    audit(m,standard=mode=='standard');return m

def audit(m,standard=True):
    layers={n:v for n,v in m.named_modules() if isinstance(v,LowRank)}
    params={n:p for n,p in m.named_parameters() if p.requires_grad}
    assert set(layers)==set(MODULES)
    assert all('lora_' in n or '.condition.' in n for n in params)
    assert not any(p.requires_grad for p in m.output_patch_embedding.parameters())
    count=sum(p.numel() for p in params.values())
    if standard:assert count==1179648 and len(params)==192
    return dict(modules=96,trainable_count=count,trainable_tensors=len(params),head_frozen=True,base_frozen=True)

def set_state(m,state):
    for v in m.modules():
        if isinstance(v,LowRank):v.state=state
@contextmanager
def disabled(m):
    layers=[v for v in m.modules() if isinstance(v,LowRank)];states=[v.enabled for v in layers]
    try:
        for v in layers:v.enabled=False
        yield
    finally:
        for v,s in zip(layers,states):v.enabled=s

def snapshot(*models):
    return {f'{i}:{n}':p.detach().cpu().clone() for i,m in enumerate(models) if m is not None for n,p in m.named_parameters() if p.requires_grad}
def restore(state,*models):
    params={f'{i}:{n}':p for i,m in enumerate(models) if m is not None for n,p in m.named_parameters() if p.requires_grad}
    assert params.keys()==state.keys()
    with torch.no_grad():
        for n,p in params.items():p.copy_(state[n].to(p))
