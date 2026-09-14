"""Storage and origin-microbatch variants; historical model functions unchanged."""
from contextlib import contextmanager, nullcontext
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from .equal_time import EqualTimeModel, TrainingClock, adapted_side
from ..backbone import native_loss


def block_indices(k):
    if k not in (0,3,6,9,12):raise ValueError(k)
    return np.linspace(0,11,k,dtype=int).tolist() if k else []


@contextmanager
def partial_checkpoint(model,k):
    saved=[]
    try:
        for i in block_indices(k):
            block=model.encoder.block[i];fn=block.forward;saved.append((block,fn))
            def wrapped(*args,_fn=fn,**kwargs):
                return checkpoint(_fn,*args,use_reentrant=False,**kwargs)
            block.forward=wrapped
        yield
    finally:
        for block,fn in saved:block.forward=fn


class BudgetModel(EqualTimeModel):
    def __init__(self,arm,seed,cp=0):
        super().__init__(arm,seed);self.set_storage(cp)
    def set_storage(self,cp):
        if cp not in ((0,3,6,9,12) if self.arm=='standard' else (0,12)):raise ValueError(cp)
        self.cp_blocks=cp;self.checkpoint_enabled=bool(cp) and self.arm!='standard'
    def forward(self,x,g):
        if self.arm=='standard':
            with self.phase('standard_encoder'),partial_checkpoint(self.base,self.cp_blocks):
                enc,(loc,scale),_,n=self.base.encode(context=x,group_ids=g,num_output_patches=3)
                assert n==(x.shape[-1]+15)//16;h=enc.last_hidden_state[:,-3:]
        elif self.arm=='query':return super().forward(x,g)
        else:
            with self.phase('frozen_encoder_and_cache'):c,loc,scale=self.encode_frozen(x,g)
            with self.phase('side_branch'):
                h=adapted_side(self.adapter,c['features'],c['time_mask'],c['group_mask'],c['base'],self.checkpoint_enabled)
        with self.phase('output_head'):
            z=self.base.output_patch_embedding(h).reshape(len(x),3,21,16).permute(0,2,1,3).reshape(len(x),21,48).float()
            raw=z.sinh()*scale[:,None,:]+loc[:,None,:]
        return z,raw,loc,scale


def micro_origins(origins,size):
    if len(origins)!=2 or size not in (1,2):raise ValueError('Two origins; microbatch size 1 or 2')
    return [(origins[i:i+size],size/2) for i in range(0,2,size)]


def batch(values,origins,device='cuda'):
    assert min(origins)>=4096 and max(origins)+48<=len(values)
    x=np.concatenate([values[o-4096:o,:4].T for o in origins]).astype(np.float32)
    y=np.concatenate([values[o:o+48,:4].T for o in origins]).astype(np.float32)
    return torch.tensor(x,device=device),torch.tensor(y,device=device),torch.arange(len(origins),device=device).repeat_interleave(4)


def choose_option(records,budget):
    feasible=[r for r in records if r['valid'] and r['peak_allocated']<=.95*budget]
    if not feasible:return None
    best=min(r['block1_seconds'] for r in feasible)
    tied=[r for r in feasible if r['block1_seconds']<=1.02*best]
    return min(tied,key=lambda r:(r['peak_allocated'],r['cp'], -r['micro']))


def delta(reference,actual):
    a=reference.double().reshape(-1);b=actual.double().reshape(-1);norm=float(a.norm())
    return dict(max_absolute=float((a-b).abs().max()),relative_l2=float((a-b).norm()/norm) if norm>=1e-12 else None,reference_norm=norm)


def check_parity(reference,actual,precision,micro=False,scale=None):
    keys=['z','raw','loss','raw_gradient','clipped_gradient','update','adam']
    metrics={k:delta(reference[k],actual[k]) for k in keys}
    tol=1e-5 if precision=='fp32' else 1e-4
    if micro and precision=='bf16':
        assert scale is not None
        sc=torch.as_tensor(np.tile(scale,2),dtype=torch.float64)[:,None,None]
        metrics['raw_train_scaled']=delta(reference['raw']/sc,actual['raw']/sc)
        gated=['z','raw_train_scaled','raw_gradient','update']
        ok=all(v['max_absolute']<=1e-6 if v['reference_norm']<1e-12 else v['relative_l2']<=1e-2 for v in (metrics[k] for k in gated))
        investigate=metrics['raw_train_scaled']['max_absolute']>2e-2
        ok=ok and not investigate
    else:
        gated=['z','raw','raw_gradient','update'] if micro else keys
        ok=all(v['max_absolute']<=tol and v['relative_l2'] is not None and v['relative_l2']<=tol
               if v['reference_norm']>=1e-12 else v['max_absolute']<=1e-6 for v in (metrics[k] for k in gated))
        investigate=False
    rng_equal=reference['rng']==actual['rng']
    return dict(passed=bool(ok and rng_equal),precision=precision,microbatch_comparison=micro,metrics=metrics,rng_equal=rng_equal,requires_raw_error_investigation=investigate)


def seal_valid(seal,contract_hash,expected):
    from ..reproducibility import digest
    content={k:v for k,v in seal.items() if k!='seal_hash'}
    return (seal.get('seal_hash')==digest(content) and seal.get('contract_hash')==contract_hash
        and len(seal.get('selections',[]))==len(expected)
        and {(r['dataset'],r['seed'],r['arm']) for r in seal['selections']}==set(expected))
