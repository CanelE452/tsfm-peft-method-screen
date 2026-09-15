import sys,math,time,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];RUN='covariate_lora_controls_20260916';OUT=ROOT/'results'/RUN;CACHE=ROOT/'.cache'/RUN
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experiments.peft_rank12_20260915.common import Watch,save,read,sha,parameters,cpu_state,restore,tensor_hash,frozen_hash,cleanup,csvwrite
from tsfm_peft_screen.backbone import load_base,QUANTILES
from tsfm_peft_screen.lora import MODULES
from chronos.chronos2.preprocess import from_list_of_dicts
FEATURES=['temperature_2m','wind_speed_10m','shortwave_radiation'];SEEDS=[61710,61711];ARMS=['STD','EXODROP','VINTAGE'];CHECKPOINTS=[0,40,80,120]
class LoRA(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base;self.a=nn.Parameter(torch.empty(1,base.in_features,device=base.weight.device));nn.init.kaiming_uniform_(self.a,a=math.sqrt(5));self.b=nn.Parameter(torch.zeros(base.out_features,1,device=base.weight.device))
 def forward(self,x):return self.base(x)+2*nn.functional.linear(nn.functional.linear(x,self.a),self.b)
def make(seed):
 m=load_base()
 for i,name in enumerate(MODULES):
  torch.manual_seed(seed+1000+i);parent,leaf=name.rsplit('.',1);p=m.get_submodule(parent);setattr(p,leaf,LoRA(getattr(p,leaf)))
 ps=parameters(m);assert len(ps)==192 and sum(p.numel() for p in ps.values())==147456
 assert all(n.endswith('.a') or n.endswith('.b') for n in ps);return m

def payload(job,path=0,mask=None,raw=False):
 with np.load(ROOT/job['input']) as x:
  a=x['past'].copy();b=x['paths'][path].copy();target=x['target'].copy()
 if not raw:
  s=read(OUT/'scaling.json');mu=np.array(s['mean'])[:,None];sd=np.array(s['std'])[:,None];a=((a-mu)/sd).astype('float32');b=((b-mu)/sd).astype('float32')
 if mask is not None:a=a*mask[:,None];b=b*mask[:,None]
 return dict(target=target,past_covariates={k:a[i] for i,k in enumerate(FEATURES)},future_covariates={k:b[i] for i,k in enumerate(FEATURES)})
def tensors(item,y=None):
 p=from_list_of_dicts([item],prediction_length=24)[0];c=p['context'].to('cuda');f=p['future_covariates'].to('cuda');assert c.shape==(4,336) and f.shape==(4,24)
 labels=None
 if y is not None:labels=torch.full_like(f,float('nan'));labels[0]=torch.as_tensor(y,dtype=torch.float32,device='cuda')
 return dict(context=c,group_ids=torch.zeros(4,dtype=torch.long,device='cuda'),future_covariates=f,num_output_patches=2,future_target=labels)
def forward(m,item,y=None):
 v=m(**tensors(item,y));assert v.quantile_preds.shape==(4,21,32) and torch.isfinite(v.quantile_preds).all()
 if y is not None:assert torch.isfinite(v.loss)
 return v
def predict(m,item,w):
 w.boundary()
 with torch.no_grad():q=forward(m,item).quantile_preds[0,:,:24].double().cpu().numpy()
 return np.sort(q,axis=0),q

def score(q,y,sd):
 q=np.asarray(q,dtype=float);y=np.asarray(y,dtype=float);taus=np.array(QUANTILES);e=y[None]-q;pin=float((2*np.maximum(taus[:,None]*e,(taus[:,None]-1)*e)).mean());d=q[10]-y
 return dict(primary=pin/sd,raw_2pinball=pin,raw_RMSE=float(np.sqrt(np.mean(d*d))),raw_MAE=float(np.mean(abs(d))),coverage80=float(np.mean((y>=q[2])&(y<=q[18]))),width80=float((q[18]-q[2]).mean()))
def configure():
 torch.set_num_threads(4);torch.manual_seed(61710);np.random.seed(61710);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
def augmentation(arm,seed,k):
 path=(k//8)%4 if arm=='VINTAGE' else 0
 mask=(np.random.default_rng(seed+50000+k).random(3)>=.3).astype('float32')/.7 if arm=='EXODROP' else None
 return path,mask
