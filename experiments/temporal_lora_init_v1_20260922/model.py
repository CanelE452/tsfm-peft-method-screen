import os
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import time,hashlib
import numpy as np
import torch
from types import MethodType
from chronos import ChronosBoltPipeline
from peft import LoraConfig,get_peft_model
from common import *
def setup():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def hash_tensors(items):
    h=hashlib.sha256()
    for n,v in sorted(items):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def embed(self):return self.shared
class Model(torch.nn.Module):
    def __init__(self,seed=92341):
        super().__init__();info=read(ROOT/'results/ab_path_ensemble_screen_v1_20260922/A/SOURCE_AND_MODEL_MANIFEST.json')['models']['amazon/chronos-bolt-small']
        self.pipeline=ChronosBoltPipeline.from_pretrained(info['path'],device_map='cuda',torch_dtype=torch.float32,local_files_only=True)
        base=self.pipeline.model;base.requires_grad_(False);base.get_input_embeddings=MethodType(embed,base);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        self.network=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q','v'],bias='none'));self.pipeline.model=self.network.get_base_model();self.eval()
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==294912
    def forward(self,x):return self.network(context=x).quantile_preds.transpose(1,2)
    def state(self):return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}
    def restore(self,state):
        params=dict(self.named_parameters());assert set(state)=={n for n,p in params.items() if p.requires_grad}
        with torch.no_grad():
            for n,v in state.items():params[n].copy_(v)
    def frozen(self):return hash_tensors([(n,p) for n,p in self.named_parameters() if not p.requires_grad]+list(self.named_buffers()))
    def structured(self,basis):
        with torch.no_grad():
            for n,p in self.named_parameters():
                if '.encoder.block.' in n and '.lora_A.' in n:
                    layer=int(n.split('.encoder.block.')[1].split('.')[0]);b=torch.as_tensor(basis[layer],device=p.device,dtype=p.dtype)
                    p.copy_(b*p.norm(dim=1,keepdim=True))
def pinball(q,y,sigma):
    levels=torch.arange(1,10,device=q.device)/10;e=y[...,None]-q
    return (2*torch.maximum(levels*e,(levels-1)*e)/sigma[:,None,None]).mean()
def eigbasis(c):
    vals,vec=torch.linalg.eigh(torch.as_tensor(c,dtype=torch.float64));basis=vec[:,-8:].flip(1).T
    signs=basis[torch.arange(8),basis.abs().argmax(1)].sign();basis=basis*signs[:,None]
    assert torch.allclose(basis@basis.T,torch.eye(8,dtype=torch.float64),atol=1e-8)
    return basis.float().numpy(),vals[-8:].numpy()
def estimate(z,shuffle=False,mode='both'):
    z=z.astype(np.float64);z-=z.mean(1,keepdims=True);flat=z.reshape(-1,z.shape[-1])
    c0=flat.T@flat/len(flat) if mode in ['both','PCA'] else None
    if mode=='PCA':return c0,None
    a=z[:,:-1].copy();b=z[:,1:].copy()
    if shuffle:
        rng=np.random.default_rng(92340)
        for i in range(len(b)):b[i]=b[i,rng.permutation(b.shape[1])]
    a=a.reshape(-1,z.shape[-1]);b=b.reshape(-1,z.shape[-1]);cross=a.T@b/len(a)
    return c0,(cross+cross.T)/2
def build_bases(d):
    model=Model();collected={i:[] for i in range(6)};handles=[]
    for i,block in enumerate(model.pipeline.model.encoder.block):
        q=block.layer[0].SelfAttention.q;v=block.layer[0].SelfAttention.v;last={}
        def hook(module,inputs,i=i,last=last):
            z=inputs[0];assert z.shape[1:]==(17,512);last['q']=z.detach().clone();collected[i].append(z[:,:16].detach().cpu().numpy())
        def vhook(module,inputs,last=last):assert torch.equal(last['q'],inputs[0])
        handles.extend([q.register_forward_pre_hook(hook),v.register_forward_pre_hook(vhook)])
    x=d.basis_x();torch.cuda.synchronize();start=time.perf_counter()
    with torch.no_grad():
        for i in range(0,len(x),16):model(torch.as_tensor(x[i:i+16],device='cuda'))
    torch.cuda.synchronize();feature_seconds=time.perf_counter()-start
    for h in handles:h.remove()
    del model;torch.cuda.empty_cache();acts={i:np.concatenate(z) for i,z in collected.items()};out={};times={};diagnostic={}
    for arm in ['PCA','TEMP','SHUFFLE']:
        start=time.perf_counter();out[arm]={};diagnostic[arm]={}
        for layer,z in acts.items():
            c0,c1=estimate(z,arm=='SHUFFLE',arm);b,vals=eigbasis(c0 if arm=='PCA' else c1);out[arm][layer]=b;diagnostic[arm][layer]=dict(eigenvalues=vals.tolist(),basis_sha=hashlib.sha256(b.tobytes()).hexdigest())
        times[arm]=feature_seconds+time.perf_counter()-start
    times['RANDOM']=0.
    return out,dict(feature_seconds=feature_seconds,charged_prep_seconds=times,basis_diagnostics=diagnostic,q_v_input_exact_parity=True,initialization_target_access=False,basis_input_sha=hashlib.sha256(x.tobytes()).hexdigest())
