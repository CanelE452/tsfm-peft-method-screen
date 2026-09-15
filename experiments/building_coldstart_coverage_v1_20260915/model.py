"""Pinned native Chronos-2, rank8 standard LoRA, frozen head, batch1 FP32."""
import sys,time,contextlib
from core import *
sys.path.insert(0,str(ROOT/'scripts'))
import torch
from tsfm_peft_screen.backbone import load_base
from tsfm_peft_screen.lora import attach,audit,disabled
from priority12.common import Watch as BaseWatch,parameters,cpu_state,tensor_hash,frozen_hash,restore,cleanup
class Watch(BaseWatch):
    def sample(self):
        r=super().sample()
        for a in r['apps']:a['allowed_desktop']=a['name']=='/usr/share/rustdesk/rustdesk'
        r['busy']=any(not a['own'] and not a['allowed_desktop'] for a in r['apps']) or r['free_mib']<1024
        return r

def setup():
    torch.set_num_threads(4);torch.manual_seed(61600);np.random.seed(61600)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)

def make():
    m=attach(load_base(),seed=read(CONFIG)['lora_seed']);m.eval();audit(m);return m

def tensor(x):return torch.as_tensor(np.array(x,copy=True),dtype=torch.float32,device='cuda')
def forward(m,x,y=None):
    x=tensor(x).reshape(1,24);y=None if y is None else tensor(y).reshape(1,24)
    r=m(context=x,group_ids=torch.zeros(1,dtype=torch.long,device='cuda'),num_output_patches=2,future_target=y)
    assert r.quantile_preds.shape==(1,21,32)
    if not torch.isfinite(r.quantile_preds).all():raise FloatingPointError('NONFINITE_OUTPUT')
    if y is not None and not torch.isfinite(r.loss):raise FloatingPointError('NONFINITE_LOSS')
    return r

def predict(m,x,f0=False):
    with torch.no_grad(),disabled(m) if f0 else contextlib.nullcontext():
        raw=forward(m,x).quantile_preds[0,:,:24].double().cpu().numpy()
    q=np.sort(raw,axis=0);assert np.all(np.diff(q,axis=0)>=0)
    return q,raw

def update(m,opt,x,y,w):
    w.boundary();torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True)
    r=forward(m,x,y);r.loss.backward();ps=list(parameters(m).values())
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps):raise FloatingPointError('NONFINITE_OR_MISSING_GRADIENT')
    norm=torch.nn.utils.clip_grad_norm_(ps,1.0,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
    return dict(loss=float(r.loss.detach()),gradient_norm=float(norm),seconds=time.monotonic()-t,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
