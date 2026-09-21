import hashlib
from types import MethodType
import numpy as np
import torch
from torch import nn
from chronos import ChronosBoltPipeline, Chronos2Pipeline
from peft import LoraConfig, get_peft_model
from common import RESULTS, read_json
from contract.reference_core import MetadataAdapter

ARMS = ['R_ROLLOUT_LORA', 'S_STATE_ADAPTER', 'U_UNCERTAINTY_ADAPTER']

def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def tensor_hash(items):
    h=hashlib.sha256()
    for name, value in sorted(items):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def shared_embeddings(self):
    return self.shared

class RolloutModel(nn.Module):
    def __init__(self, arm, seed, lora=True):
        super().__init__()
        info=read_json(RESULTS/'SOURCE_AND_MODEL_MANIFEST.json')['models']['amazon/chronos-bolt-small']
        self.pipeline=ChronosBoltPipeline.from_pretrained(info['path'], device_map='cuda', torch_dtype=torch.float32, local_files_only=True)
        base=self.pipeline.model
        base.requires_grad_(False)
        # PEFT embedding inspection in transformers 5 needs the existing shared embedding.
        base.get_input_embeddings=MethodType(shared_embeddings,base)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        self.network=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q','v'],bias='none')) if lora else base
        self.pipeline.model=self.network.get_base_model() if lora else base
        self.arm=arm
        self.adapter=None
        if arm in ARMS[1:]:
            torch.manual_seed(seed+100000); torch.cuda.manual_seed_all(seed+100000)
            self.adapter=MetadataAdapter().to('cuda',torch.float32)
        self.eval()

    def learned(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def load_learned(self, state):
        params=dict(self.named_parameters())
        assert set(state)=={n for n,p in params.items() if p.requires_grad}
        with torch.no_grad():
            for n,v in state.items(): params[n].copy_(v)

    def frozen_hash(self):
        return tensor_hash([(n,p) for n,p in self.named_parameters() if not p.requires_grad]+list(self.named_buffers()))

    def conditional(self, values, metadata=None, adapter_on=True):
        hook=None
        if self.adapter is not None and adapter_on:
            assert metadata is not None
            hook=self.pipeline.model.input_patch_embedding.register_forward_hook(lambda _m,_a,h: self.adapter(h,metadata))
        try:
            return self.network(context=values).quantile_preds.transpose(1,2)
        finally:
            if hook is not None: hook.remove()

def initial_state(x,sigma):
    zeros=torch.zeros_like(x)
    s0=torch.maximum(x.std(dim=1,correction=0),sigma*1e-6)
    return [x, zeros, zeros.clone(), zeros.clone(),s0,0]

def metadata(state,arm):
    values,m,lead,width,s0,k=state
    a=lead/256
    last=m*a.square() if arm=='S_STATE_ADAPTER' else m*torch.log1p(width/s0[:,None])
    z=torch.stack([m,m*a,m*k/3,last],dim=-1)
    return z.reshape(len(values),-1,16,4).mean(dim=2)

def append(state,q):
    values,m,lead,width,s0,k=state
    med=q[:,:,4].detach()
    w=(q[:,:,-1]-q[:,:,0]).clamp_min(0).detach()
    newlead=torch.arange(k*64+1,(k+1)*64+1,device=values.device,dtype=values.dtype).expand_as(med)
    return [torch.cat([values,med],1),torch.cat([m,torch.ones_like(med)],1),torch.cat([lead,newlead],1),torch.cat([width,w],1),s0,k+1]

def pinball(y,q,sigma):
    taus=torch.arange(1,10,device=q.device,dtype=q.dtype)/10
    e=y[:,:,None]-q
    return (2*torch.maximum(taus*e,(taus-1)*e)/sigma[:,None,None]).mean()

def rollout(model,x,sigma,adapter_on=True,trace=None):
    state=initial_state(x,sigma)
    raw=[]
    for k in range(4):
        z=metadata(state,model.arm)
        q=model.conditional(state[0],z,adapter_on)
        if trace is not None:
            trace.append({'length':state[0].shape[1],'metadata':z.detach().cpu(),'values':state[0].detach().cpu(),'width_requires_grad':state[3].requires_grad,'value_requires_grad':state[0].requires_grad})
        raw.append(q)
        state=append(state,q.sort(dim=-1).values)
    raw=torch.cat(raw,dim=1)
    return raw.sort(dim=-1).values,raw

def train_update(model,optimizer,x,y,sigma):
    model.eval(); optimizer.zero_grad(set_to_none=True)
    state=initial_state(x,sigma)
    losses=[]; crossings=[]
    for k in range(4):
        q=model.conditional(state[0],metadata(state,model.arm))
        crossings.append((q[:,:,1:]<q[:,:,:-1]).float().mean().detach())
        ordered=q.sort(dim=-1).values
        loss=pinball(y[:,k*64:(k+1)*64],ordered,sigma)/4
        loss.backward()
        losses.append(loss.detach())
        state=append(state,ordered)
        assert not state[0].requires_grad and not state[3].requires_grad
    params=[p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    norm=torch.nn.utils.clip_grad_norm_(params,1.)
    assert torch.isfinite(norm)
    optimizer.step()
    return {'loss':sum(losses).item(),'grad_norm':norm.item(),'conditional_raw_crossing':torch.stack(crossings).mean().item(),'conditional_calls':4}

@torch.no_grad()
def native(model,x,trace=None):
    hook=None
    if trace is not None:
        def record(_m,_a,kwargs,out):
            q=out.quantile_preds
            trace.append({'batch':q.shape[0],'length':kwargs['context'].shape[-1],'crossing':float((q[:,1:]<q[:,:-1]).float().mean()),'quantiles':q.detach().cpu()})
        hook=model.pipeline.model.register_forward_hook(record,with_kwargs=True)
    try:
        return model.pipeline.predict(x,prediction_length=256).transpose(1,2)
    finally:
        if hook is not None: hook.remove()

def inverse_cdf(q,u):
    position=(u.clamp(.1,.9)*10-1).clamp(0,8)
    lo=position.floor().long(); hi=(lo+1).clamp_max(8)
    return q.gather(-1,lo[...,None]).squeeze(-1)+(position-lo)*(q.gather(-1,hi[...,None]).squeeze(-1)-q.gather(-1,lo[...,None]).squeeze(-1))

@torch.no_grad()
def mc16(model,x,uniforms,chunk=64):
    # Uniforms are precomputed per example, block, particle, lead; microbatch invariant.
    first_raw=model.conditional(x)
    first=first_raw.sort(dim=-1).values
    particle_q=first[:,None].expand(-1,16,-1,-1)
    sample=inverse_cdf(particle_q,uniforms[:,0])
    contexts=torch.cat([x[:,None].expand(-1,16,-1),sample],dim=-1)
    out=[first]
    raw_cross=[float((first_raw[...,1:]<first_raw[...,:-1]).float().mean())]
    for k in range(1,4):
        flat=contexts.flatten(0,1)
        raw=torch.cat([model.conditional(flat[j:j+chunk]) for j in range(0,len(flat),chunk)],0).reshape(len(x),16,64,9)
        raw_cross.append(float((raw[...,1:]<raw[...,:-1]).float().mean()))
        sample=inverse_cdf(raw.sort(dim=-1).values,uniforms[:,k])
        out.append(torch.quantile(sample,torch.arange(1,10,device=x.device,dtype=x.dtype)/10,dim=1,interpolation='linear').permute(1,2,0))
        contexts=torch.cat([contexts,sample],dim=-1)
    return torch.cat(out,dim=1),raw_cross

def load_direct():
    info=read_json(RESULTS/'SOURCE_AND_MODEL_MANIFEST.json')['models']['amazon/chronos-2']
    return Chronos2Pipeline.from_pretrained(info['path'],device_map='cuda',torch_dtype=torch.float32,local_files_only=True)

@torch.no_grad()
def direct(pipeline,x):
    # Each list member is one independent univariate group; no cross-learning.
    groups=[row[None,:].cpu() for row in x]
    q,_=pipeline.predict_quantiles(groups,prediction_length=256,quantile_levels=[i/10 for i in range(1,10)],batch_size=len(groups),context_length=512,cross_learning=False)
    return torch.cat(q,dim=0)
