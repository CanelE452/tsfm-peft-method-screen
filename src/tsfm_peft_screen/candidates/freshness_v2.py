"""Causal asynchronous corruption and clean-centered rank-space affine LoRA."""
import hashlib
import numpy as np
import torch
from torch import nn
from ..lora import LowRank,MODULES,attach,audit
from .freshness import token_state
TRAIN_VARIANTS=['clean','refresh_train','block_train','combined_train']
E_VARIANTS=['refresh_shift1','refresh_shift2','block_shift1','block_shift2','clean']
RULES={
 'refresh_train':dict(periods=[2,3,4,6]),
 'block_train':dict(lengths=[3,6,9,12],ends=[0,3,6,9]),
 'combined_train':dict(periods=[2,3,4,6],lengths=[3,6,9,12],ends=[0,3,6,9]),
 'refresh_shift1':dict(periods=[5,7,9,11]),
 'refresh_shift2':dict(periods=[7,9,11,13]),
 'block_shift1':dict(lengths=[12,18,24,30],ends=[0,6,12,18]),
 'block_shift2':dict(lengths=[18,24,30,36],ends=[0,9,18,27]),
}

def corrupt(x,kind,origin):
    a=np.asarray(x,dtype=np.float32).copy()
    assert a.shape==(4,336) and origin>=336
    observed=np.isfinite(a)
    # Hash only known origin and declared variant, independent of labels/seed.
    key=int.from_bytes(hashlib.sha256(f'freshness-v2:{kind}:{origin}'.encode()).digest()[:8],'little')
    rng=np.random.default_rng(key);order=rng.permutation(4)
    if kind!='clean':
        rule=RULES[kind]
        if 'periods' in rule:
            periods=np.array(rule['periods'])[order]
            offsets=np.array([rng.integers(0,int(n)) for n in periods])
            observed &= ((np.arange(origin-336,origin)[None,:]-offsets[:,None])%periods[:,None]==0)
        if 'lengths' in rule:
            lengths=np.array(rule['lengths'])[order];ends=np.array(rule['ends'])[order]
            for c in range(4):
                end=336-int(ends[c]);observed[c,end-int(lengths[c]):end]=False
    a[~observed]=np.nan
    age=np.zeros_like(a);recent=np.zeros_like(a)
    for t in range(336):
        age[:,t]=np.where(observed[:,t],0,1+(age[:,t-1] if t else 0))
        recent[:,t]=observed[:,max(0,t-23):t+1].mean(1)
    state=np.stack([observed,age/336,recent],-1).astype(np.float32)
    return a,state

class AffineLowRank(LowRank):
    def __init__(self,base,group_attention=False):
        super().__init__(base,'standard',group_attention)
        self.mode='affine_v2'
        # Centering an affine linear map cancels its bias: omit dead biases.
        self.condition=nn.Linear(3,16,bias=False,device=base.weight.device)
        nn.init.zeros_(self.condition.weight)
    def modulation(self,state):
        clean=state.new_tensor([1.,0.,1.])
        r=self.condition(state);ref=self.condition(clean)
        # a(r)-a(clean), a=.5*tanh(linear); multiplier is bounded in (0,2).
        mult=1+.5*(r[...,:8].tanh()-ref[:8].tanh())
        shift=r[...,8:]-ref[8:]
        return mult,shift
    def forward(self,x):
        out=self.base(x)
        if not self.enabled:return out
        assert self.state is not None
        state=self.state.transpose(0,1) if self.group_attention else self.state
        assert state.shape[:2]==x.shape[:2]
        a=nn.functional.linear(x,self.lora_A);mult,shift=self.modulation(state)
        return out+2*nn.functional.linear(mult*a+shift,self.lora_B)

def attach_v2(m,arm,seed):
    attach(m,seed,'feature' if arm=='FEATURE_LORA' else 'standard')
    if arm=='AFFINE_V2':
        for i,name in enumerate(MODULES):
            old=m.get_submodule(name);torch.manual_seed(seed+1000+i)
            new=AffineLowRank(old.base,old.group_attention)
            with torch.no_grad():new.lora_A.copy_(old.lora_A);new.lora_B.copy_(old.lora_B)
            parent,leaf=name.rsplit('.',1);setattr(m.get_submodule(parent),leaf,new)
    return audit(m,standard=arm=='STANDARD_LORA')
