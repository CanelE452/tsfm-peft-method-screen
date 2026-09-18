from experiments.c3_training_factorial_20260918.common import *
from experiments.c3_training_factorial_20260918 import common as parent
from experiments.additive_persistence_validation_v1_20260917.checks import arrays,batch
from experiments.outlier_signal_peft_v1_20260917.model import loss_2pinball
from experiments.outlier_signal_peft_v1_20260917.reference_core import rng
from experiments.additive_persistence_validation_v1_20260917.model import correction_gate,ResidualAdapter
from experiments.outlier_signal_peft_v1_20260917.model import robust_scale
NAME='c3_internal_mechanism_20260918';OUT=ROOT/'results'/NAME;CACHE=ROOT/'.cache'/NAME;EXP=ROOT/'experiments'/NAME
PRIOR=parent.OUT;STEPS=[0,256,512,768,1024];GRAD_CAP=1280;COUNTS={'autograd_calls':0,'diagnostic_forward_calls':0,'optimizer_updates':0,'new_fits':0}
class Watch(parent.Watch):
 def __init__(self):
  self.last_disk=0;OriginalWatch.__init__(self,OUT,'mechanism',wall_cap=7200)
def grad(*a,**kw):
 COUNTS['autograd_calls']+=1;assert COUNTS['autograd_calls']<=GRAD_CAP
 return torch.autograd.grad(*a,**kw)
def check_seal():
 for p,h in read(OUT/'SEAL.json')['hashes'].items():assert sha(ROOT/p)==h,('SEALED_CHANGED',p)
def load(row,b=None,step=None):
 t=list(row['seed']);t[0]=t[0] if b is None else b;m=build(row['arm'],t,row['source']);rec=read(PRIOR/'fits'/row['fit']/'receipt.json');c=next(c for c in rec['checkpoints'] if c['step']==(1024 if step is None else step));assert sha(ROOT/c['checkpoint'])==c['sha256'];restore(m,torch.load(ROOT/c['checkpoint'],weights_only=True,map_location='cpu'));return m
class Capture:
 def __init__(self,m):
  self.m=m;self.data={};self.hook=m.adapter.register_forward_hook(self.capture)
 def capture(self,module,args,out):self.data=dict(h=args[0],g=args[1],u=out)
 def close(self):self.hook.remove()
def forward(m,x,s):COUNTS['diagnostic_forward_calls']+=1;return m(x,s)
def flat(gs):return torch.cat([x.reshape(-1) for x in gs])
def cos(a,b):
 a=np.asarray(a).ravel().astype(float);b=np.asarray(b).ravel().astype(float);n=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/n) if n>1e-25 else None
def rel(a,b):return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b),1e-20))
def save_counts():save(OUT/'COUNTS.json',COUNTS)
def capof(h):return h.square().mean(-1).sqrt().quantile(.5,dim=-1,keepdim=True).clamp_min(1e-6).detach()[...,None]
def gateof(x,s,arm):
 if arm=='C3':return correction_gate(x,s)
 m,r=robust_scale(x,s);return 1-(((x-m)/r).abs()>3).to(x).reshape(len(x),32,16).mean(-1)
def formula(h,g,r,adapter):
 z=torch.nn.functional.gelu(adapter.down(h));t=capof(h)*g[...,None]*r
 return torch.cat([torch.einsum('ntd,ntk->dk',t,z).reshape(-1),t.sum((0,1))]),z
