import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch import nn
from types import MethodType
from chronos import ChronosBoltPipeline
from peft import LoraConfig,get_peft_model
from common import *
from contract.reference_core import PathRouter,ScenarioPool,medoids3,equal_atoms,weighted_crps,cdf_l2,discrete_quantiles

def setup():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)

def tensor_hash(items):
    h=hashlib.sha256()
    for n,v in sorted(items):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def embeddings(self):return self.shared

def fast_medoids(paths):
    from itertools import combinations
    choices=torch.tensor(list(combinations(range(9),3)),device=paths.device)
    distance=(paths[:,:,None]-paths[:,None]).square().mean(-1)
    costs=distance[:,:,choices].min(-1).values.sum(1)
    indices=choices[costs.argmin(-1)]
    centers=paths[torch.arange(len(paths),device=paths.device)[:,None],indices]
    assignments=(paths[:,:,None]-centers[:,None]).square().mean(-1).argmin(-1)
    weights=torch.nn.functional.one_hot(assignments,3).to(paths.dtype).mean(1)
    return centers,weights,indices

class Base(nn.Module):
    def __init__(self,seed=92301,lora=True):
        super().__init__()
        info=read(ROOT/'results/continuation_lora_v1_20260922/SOURCE_MANIFEST.json')['models']['amazon/chronos-bolt-small']
        self.pipeline=ChronosBoltPipeline.from_pretrained(info['path'],device_map='cuda',torch_dtype=torch.float32,local_files_only=True)
        base=self.pipeline.model;base.requires_grad_(False);base.get_input_embeddings=MethodType(embeddings,base)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        self.network=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q','v'],bias='none')) if lora else base
        self.has_lora=lora;self.pipeline.model=self.network.get_base_model() if lora else base;self.eval()
    def conditional(self,x):return self.network(context=x).quantile_preds
    def learned(self):return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}
    def load_learned(self,state):
        params=dict(self.named_parameters());assert set(state)=={n for n,p in params.items() if p.requires_grad}
        with torch.no_grad():
            for n,v in state.items():params[n].copy_(v)
    def frozen_hash(self):return tensor_hash([(n,p) for n,p in self.named_parameters() if not p.requires_grad]+list(self.named_buffers()))
    def load_lora(self,state):
        params=dict(self.named_parameters())
        with torch.no_grad():
            for n,v in state.items():
                if n.startswith('network.'):params[n].copy_(v)

class ModelA(Base):
    def __init__(self,arm,seed=92301,lora=True):
        super().__init__(seed,lora);self.arm=arm
        if arm in ['GLOBAL3','CONTEXT3']:
            torch.manual_seed(seed+700);self.router=PathRouter(arm=='CONTEXT3').cuda()
        self.eval()
    @torch.no_grad()
    def first(self,x):
        if self.has_lora:
            with self.network.disable_adapter():return self.conditional(x).detach()
        return self.conditional(x).detach()
    def distribution(self,x,first=None):
        first=self.first(x) if first is None else first.detach()
        if self.arm in ['FULL9','F0_NATIVE']:
            paths=first;weights=x.new_full((len(x),9),1/9)
        elif self.arm in ['MEDIAN1','F0_MEDIAN']:
            paths=first[:,4:5];weights=x.new_ones((len(x),1))
        elif self.arm=='FIXED3':paths=first[:,[1,4,7]];weights=x.new_full((len(x),3),1/3)
        elif self.arm=='MEDOID3':
            scale=x.std(-1,unbiased=False).clamp_min(1e-5)
            _,weights,indices=fast_medoids(first/scale[:,None,None])
            paths=first[torch.arange(len(x),device=x.device)[:,None],indices]
        else:
            normalized=(first-x.mean(-1)[:,None,None])/x.std(-1,unbiased=False).clamp_min(1e-5)[:,None,None]
            _,weights,assignment=self.router(normalized)
            paths=torch.einsum('bkm,bmh->bkh',assignment,first)
        ctx=torch.cat([x[:,None].expand(-1,paths.shape[1],-1),paths],dim=-1).flatten(0,1)
        second=self.conditional(ctx).reshape(len(x),paths.shape[1],9,64).permute(0,3,1,2).flatten(2)
        p=weights[:,None,:,None].expand(-1,64,-1,9).flatten(2)/9
        # Repeat first-block atoms only to make support count uniform; its measure is unchanged.
        k=paths.shape[1];first_atoms=first.transpose(1,2).repeat(1,1,k)
        z=torch.cat([first_atoms,second],1);p=torch.cat([equal_atoms(first_atoms),p],1)
        return z,p,first

def encoder(dim):return nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,16))
def decoder(hidden=64,out=9):
    m=nn.Sequential(nn.Linear(34,hidden),nn.GELU(),nn.Linear(hidden,out));nn.init.zeros_(m[-1].weight);nn.init.zeros_(m[-1].bias);return m

class ModelB(Base):
    def __init__(self,arm,path_dim,seed=92301,lora=True):
        super().__init__(seed,lora);self.arm=arm;self.path_dim=path_dim
        torch.manual_seed(seed+800)
        if arm not in ['TARGET','F0']:
            if arm=='SCENARIO3':
                self.pool=ScenarioPool(path_dim);torch.manual_seed(seed+801);self.condition=decoder()
            else:
                self.encoder=encoder(path_dim*2 if arm=='MOMENTS' else path_dim)
                if arm=='SET':
                    # Exact count-only matching to E + query + mix + shared decoder.
                    reference=sum(p.numel() for p in ScenarioPool(path_dim).parameters())+sum(p.numel() for p in decoder().parameters())
                    width=next(h for h in range(64,513) if sum(p.numel() for p in self.encoder.parameters())+(34*h+h)+(h*30+30)>=reference)
                    torch.manual_seed(seed+801);self.condition=decoder(width,30);self.set_width=width
                else:
                    torch.manual_seed(seed+801);self.condition=decoder()
            self.cuda()
        self.eval()
    def distribution(self,x,weather,control,sigma):
        q0=self.conditional(x)[:,:,1:9].transpose(1,2)
        if self.arm in ['TARGET','F0']:return q0,equal_atoms(q0),q0
        history=x[:,-8:]/sigma[:,None]
        r=torch.cat([q0/sigma[:,None,None],history[:,None].expand(-1,8,-1),torch.arange(1,9,device=x.device)[None,:,None].expand(len(x),-1,-1)/8],-1)
        context=torch.cat([q0.flatten(1)/sigma[:,None],history],-1)
        paths=weather.flatten(2) if weather is not None else None
        if self.arm=='CONTROL':z=self.encoder(control.flatten(1))[:,None];mix=x.new_ones((len(x),1))
        elif self.arm=='MOMENTS':
            z=self.encoder(torch.cat([paths.mean(1),paths.std(1,unbiased=False)],-1))[:,None];mix=x.new_ones((len(x),1))
        elif self.arm=='MEMBER':z=self.encoder(paths);mix=x.new_full((len(x),50),1/50)
        elif self.arm=='SCENARIO3':
            prototypes,mix,assignment=self.pool(paths,context);z=self.pool.encoder(prototypes)
        elif self.arm=='SET':
            z=self.encoder(paths).mean(1)[:,None].expand(-1,8,-1)
            raw=self.condition(torch.cat([r,z],-1));atoms=q0[:,:,None,:]+sigma[:,None,None,None]*raw[:,:,:27].reshape(-1,8,3,9)
            weights=raw[:,:,27:].softmax(-1)[:,:,:,None].expand(-1,-1,-1,9)/9
            return atoms.flatten(2),weights.flatten(2),q0
        k=z.shape[1]
        inputs=torch.cat([r[:,:,None].expand(-1,-1,k,-1),z[:,None].expand(-1,8,-1,-1)],-1)
        atoms=q0[:,:,None,:]+sigma[:,None,None,None]*self.condition(inputs)
        weights=mix[:,None,:,None].expand(-1,8,-1,9)/9
        return atoms.flatten(2),weights.flatten(2),q0
