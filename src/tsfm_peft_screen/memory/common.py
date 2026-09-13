import gc,json,time
from collections import defaultdict
import numpy as np
import torch
from ..backbone import load_base,forecast,native_loss
from ..lora import attach,snapshot,restore
from ..reproducibility import ROOT,seed_all,guard
OUT=ROOT/'results/memory_feasibility'
CACHE=ROOT/'.cache/memory_feasibility'
DATASETS=['ettm2','electricity']
LENGTHS=[336,1024,2048,4096]
ORIGINS=[4352,4608]

def build():
    seed_all(30000);m=load_base();attach(m);return m

def batch(name,length,origins=ORIGINS):
    root=ROOT/'data/processed'/name
    with np.load(root/'fit.npz') as f:values=f['values'];scale=f['scale'][:4]
    meta=json.loads((root/'manifest.json').read_text())
    assert max(origins)+48<=meta['bounds']['validation_start']
    assert min(origins)>=length
    x=np.concatenate([values[o-length:o,:4].T for o in origins]).astype(np.float32)
    y=np.concatenate([values[o:o+48,:4].T for o in origins]).astype(np.float32)
    return torch.tensor(x,device='cuda'),torch.tensor(y,device='cuda'),torch.arange(len(origins),device='cuda').repeat_interleave(4)

def forward(m,x,groups):
    assert x.shape[-1]<=m.chronos_config.context_length
    enc,(loc,scale),_,count=m.encode(context=x,group_ids=groups,num_output_patches=3)
    assert count==(x.shape[-1]+15)//16,'Silent context truncation'
    h=enc.last_hidden_state[:,-3:]
    z=m.output_patch_embedding(h).reshape(len(x),3,21,16).permute(0,2,1,3).reshape(len(x),21,48).float()
    p=z.sinh()*scale[:,None,:]+loc[:,None,:]
    assert torch.isfinite(p).all()
    return z,p,loc,scale

def gradient(m):return {n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad and p.grad is not None}
def params(m):return [p for p in m.parameters() if p.requires_grad]

def zero(m):m.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()

def backward(m,x,y,g):
    z,p,l,s=forward(m,x,g);loss=native_loss(z,y,l,s);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in params(m) if v.grad is not None)
    return float(loss.detach())

class SavedInventory:
    def __init__(self,m):
        self.m=m;self.stack=[];self.handles=[];self.records={};self.calls=0
        self.parameters={p.untyped_storage().data_ptr() for p in list(m.parameters())+list(m.buffers())}
    def enter(self,name):
        def fn(module,args):self.stack.append(name)
        return fn
    def leave(self,module,args,out):self.stack.pop()
    def __enter__(self):
        for name,module in self.m.named_modules():
            self.handles.append(module.register_forward_pre_hook(self.enter(name)))
            self.handles.append(module.register_forward_hook(self.leave))
        self.ctx=torch.autograd.graph.saved_tensors_hooks(self.pack,lambda x:x);self.ctx.__enter__();return self
    def pack(self,t):
        if t.device.type!='cuda':return t
        self.calls+=1;storage=t.untyped_storage();key=storage.data_ptr()
        r=self.records.setdefault(key,dict(bytes=storage.nbytes(),parameter_storage=key in self.parameters,views=set(),owners=set()))
        r['views'].add((tuple(t.shape),tuple(t.stride()),t.storage_offset(),str(t.dtype)))
        r['owners'].add(self.stack[-1] if self.stack else '<outside_model>')
        return t
    def __exit__(self,*args):
        self.ctx.__exit__(*args)
        for h in self.handles:h.remove()
    def result(self):
        rows=[]
        for r in self.records.values():
            rows.append(dict(bytes=r['bytes'],parameter_storage=r['parameter_storage'],views=[dict(shape=v[0],stride=v[1],offset=v[2],dtype=v[3]) for v in sorted(r['views'])],owners=sorted(r['owners'])))
        return dict(saved_calls=self.calls,unique_nonparameter_storage_bytes=sum(r['bytes'] for r in rows if not r['parameter_storage']),unique_parameter_storage_bytes=sum(r['bytes'] for r in rows if r['parameter_storage']),storages=sorted(rows,key=lambda r:-r['bytes']))
