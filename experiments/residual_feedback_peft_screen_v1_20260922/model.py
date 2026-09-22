import os
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import torch,numpy as np
from torch.nn import functional as F
from types import MethodType
from chronos import ChronosBoltPipeline
from peft import LoraConfig,get_peft_model
from common import *
from contract.reference_core import CoefficientGenerator,safe_masked_pinball
def setup():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.enabled=False
def hash_tensors(items):
    h=hashlib.sha256()
    for n,v in sorted(items):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def embed(self):return self.shared
def functional_forward(self,x,*args,**kwargs):
    c=self.task_coefficient
    if c is None:return self.standard_forward(x,*args,**kwargs)
    assert x.ndim==3 and c.shape==(x.shape[0],8)
    return self.base_layer(x)+self.scaling['default']*F.linear(F.linear(x,self.lora_A['default'].weight)*c[:,None,:],self.lora_B['default'].weight)
class Model(torch.nn.Module):
    def __init__(self,seed=92401):
        super().__init__();info=read(ROOT/'results/temporal_lora_init_v1_20260922/MODEL_AND_ENVIRONMENT.json')['models']['amazon/chronos-bolt-small']
        self.pipeline=ChronosBoltPipeline.from_pretrained(info['path'],device_map='cuda',torch_dtype=torch.float32,local_files_only=True)
        base=self.pipeline.model;base.requires_grad_(False);base.get_input_embeddings=MethodType(embed,base);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        self.network=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q','v'],bias='none'));self.pipeline.model=self.network.get_base_model();self.eval()
        self.targets=[];self.names=[]
        for n,m in self.network.named_modules():
            if hasattr(m,'lora_A'):
                self.names.append(n);self.targets.append(m);m.task_coefficient=None;m.standard_forward=m.forward;m.forward=MethodType(functional_forward,m)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==294912
    def forward(self,x,c=None):
        if c is not None:assert c.shape==(len(x),len(self.targets),8)
        for i,m in enumerate(self.targets):m.task_coefficient=None if c is None else c[:,i,:]
        try:return self.network(context=x).quantile_preds.transpose(1,2)
        finally:
            for m in self.targets:m.task_coefficient=None
    def bank(self):return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if 'lora_' in n}
    def restore(self,state):
        params=dict(self.named_parameters());assert set(state)==set(self.bank())
        with torch.no_grad():
            for n,v in state.items():params[n].copy_(v)
    def bank_grad(self,enabled):
        for n,p in self.named_parameters():p.requires_grad_(enabled and 'lora_' in n)
    def frozen(self):return hash_tensors([(n,p) for n,p in self.named_parameters() if 'lora_' not in n]+list(self.named_buffers()))
def generator(seed,n,arm):
    torch.manual_seed(seed);g=CoefficientGenerator(n,mode='set' if arm.endswith('SET') else 'time').cuda();g.eval();return g
def loss(q,y,s):return safe_masked_pinball(q,y,torch.ones_like(y,dtype=torch.bool),s)
def opt(params,lr=1e-4):return torch.optim.AdamW(params,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.)
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device='cuda')
