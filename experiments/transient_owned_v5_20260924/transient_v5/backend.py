"""Pinned local Chronos backend and a clearly labelled, small interface test double.

The test double is NOT Chronos and its test results never certify native Chronos.
No production run is allowed to fall back to it.
"""
from __future__ import annotations
from types import SimpleNamespace
from pathlib import Path
import importlib.metadata as meta
import inspect
import torch
from torch import nn
from .util import require, file_hash

class TinyMHA(nn.Module):
    def __init__(self,d):
        super().__init__()
        for name in ('q','k','v','o'):
            setattr(self,name,nn.Linear(d,d,bias=False))
    def forward(self,x,mask):
        q,k,v=self.q(x),self.k(x),self.v(x)
        logits=(q@k.transpose(-2,-1))/(x.shape[-1]**0.5)
        logits=logits.masked_fill(~mask, -1e4)
        return self.o(logits.softmax(-1)@v)

class TinyTime(nn.Module):
    def __init__(self,d):
        super().__init__(); self.self_attention=TinyMHA(d)
    def forward(self,x,valid,gids):
        mask=valid[:,None,:].expand(-1,x.shape[1],-1)
        return x+self.self_attention(x,mask)

class TinyGroup(nn.Module):
    def __init__(self,d):
        super().__init__(); self.self_attention=TinyMHA(d)
    def forward(self,x,valid,gids):
        trans=x.transpose(0,1)
        mask=(gids[:,None]==gids[None,:])[None,:,:] & valid.T[:,None,:]
        return x+self.self_attention(trans,mask).transpose(0,1)

class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(num_layers=2,d_model=16)
        self.chronos_config=SimpleNamespace(input_patch_size=4,input_patch_stride=4,output_patch_size=4,
                                            context_length=256,max_output_patches=16,use_reg_token=True)
        self.input_patch_embedding=nn.Linear(12,16)
        self.reg=nn.Parameter(torch.randn(1,1,16)*0.01)
        self.layers=nn.ModuleList([nn.ModuleList([TinyTime(16),TinyGroup(16),nn.LayerNorm(16)]) for _ in range(2)])
        self.output_patch_embedding=nn.Linear(16,12)
        self.register_buffer('quantiles',torch.tensor([0.1,0.5,0.9]))
    def forward(self, context,context_mask,group_ids,future_covariates,future_covariates_mask,
                future_target,future_target_mask,num_output_patches):
        n,length=context.shape; p=4; h=num_output_patches*p
        valid=context_mask.bool()
        safe=torch.where(valid,context,torch.zeros_like(context))
        count=valid.sum(-1,keepdim=True).clamp_min(1)
        loc=safe.sum(-1,keepdim=True)/count
        scale=(((safe-loc).square()*valid).sum(-1,keepdim=True)/count+1e-3).sqrt()
        normal=torch.where(valid,(safe-loc)/scale,torch.zeros_like(context))
        pad=(-length)%p
        normal=nn.functional.pad(normal,(pad,0)); mask=nn.functional.pad(context_mask,(pad,0))
        cp=normal.reshape(n,-1,p); cm=mask.reshape(n,-1,p)
        ct=torch.arange(-cp.shape[1]*p,0,device=context.device,dtype=context.dtype).reshape(1,-1,p).expand(n,-1,-1)/256
        ctx_emb=self.input_patch_embedding(torch.cat([ct,cp,cm],-1))
        fval=(future_covariates-loc)/scale
        fval=torch.where(future_covariates_mask.bool(),fval,torch.zeros_like(fval))
        fval=nn.functional.pad(fval,(0,h-fval.shape[-1])); fm=nn.functional.pad(future_covariates_mask,(0,h-future_covariates_mask.shape[-1]))
        fp=fval.reshape(n,-1,p); fmp=fm.reshape(n,-1,p)
        ft=torch.arange(h,device=context.device,dtype=context.dtype).reshape(1,-1,p).expand(n,-1,-1)/256
        fe=self.input_patch_embedding(torch.cat([ft,fp,fmp],-1))
        x=torch.cat([ctx_emb,self.reg.expand(n,1,-1),fe],1)
        token_valid=torch.cat([cm.sum(-1)>0,torch.ones(n,1,device=context.device,dtype=torch.bool),
                               torch.ones(n,num_output_patches,device=context.device,dtype=torch.bool)],-1)
        for t,g,norm in self.layers:
            x=norm(g(t(x,token_valid,group_ids),token_valid,group_ids))
        pred=self.output_patch_embedding(x[:,-num_output_patches:]).reshape(n,num_output_patches,3,p).permute(0,2,1,3).reshape(n,3,h)
        target=(future_target-loc)/scale
        target=nn.functional.pad(target,(0,h-target.shape[-1]))
        maskt=nn.functional.pad(future_target_mask,(0,h-future_target_mask.shape[-1]))*(1-fm)
        q=self.quantiles[None,:,None]
        loss=(2*((target[:,None,:]-pred)*((target[:,None,:]<=pred).float()-q)).abs()*maskt[:,None,:]).mean(-1).sum(-1).mean()
        return SimpleNamespace(loss=loss,quantile_preds=pred*scale[:,None,:]+loc[:,None,:])


def resolve_backend(kind: str) -> dict:
    if kind == 'tiny':
        return {'kind':'tiny','base_seed':813,'native_verified':False,
                'note':'Interface test double; not pretrained Chronos, not physical simulation.'}
    require(kind=='chronos','Unknown backend')
    require(meta.version('chronos-forecasting')=='2.3.2','Expected installed chronos-forecasting==2.3.2; do not install/upgrade')
    require(meta.version('peft')=='0.18.1','Expected installed peft==0.18.1 for equivalence check')
    from huggingface_hub import snapshot_download
    path=Path(snapshot_download('amazon/chronos-2',local_files_only=True)).resolve()
    require(path.is_dir(),'Cached native checkpoint unavailable')
    import chronos.chronos2.model as cm
    import chronos.chronos2.layers as cl
    info={'kind':'chronos','model_path':str(path),'snapshot':path.name,'native_verified':True,
          'versions':{m:meta.version(m) for m in ('torch','chronos-forecasting','peft','transformers')},
          'source_sha256':{m.__name__:file_hash(Path(inspect.getfile(m))) for m in (cm,cl)}}
    return info


def load_base(info: dict):
    if info['kind']=='tiny':
        with torch.random.fork_rng():
            torch.manual_seed(info['base_seed']); base=TinyBase()
        return base.eval(),TinyTime,TinyGroup
    from chronos import Chronos2Pipeline
    from chronos.chronos2.layers import TimeSelfAttention,GroupSelfAttention
    # Always OFFLINE, CPU/FP32. The resolved snapshot, rather than a mutable remote name, is used on replay.
    pipe=Chronos2Pipeline.from_pretrained(info['model_path'],device_map='cpu',dtype=torch.float32,local_files_only=True)
    base=pipe.inner_model
    base.config._attn_implementation='eager'
    return base.float().eval(),TimeSelfAttention,GroupSelfAttention
