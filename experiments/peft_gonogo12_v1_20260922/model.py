import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
from torch import nn
from types import MethodType
import hashlib
from peft import LoraConfig, get_peft_model
from chronos import ChronosBoltPipeline
from common import ROOT, read


def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def embeddings(self):
    return self.shared


def tensor_hash(items):
    h=hashlib.sha256()
    for name,value in sorted(items):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ForecastModel(nn.Module):
    def __init__(self, lora=False, seed=0):
        super().__init__()
        info = read(ROOT / 'results/rollout_uncertainty_peft_v1_20260921/SOURCE_AND_MODEL_MANIFEST.json')['models']['amazon/chronos-bolt-small']
        self.pipeline = ChronosBoltPipeline.from_pretrained(info['path'], device_map='cuda', torch_dtype=torch.float32, local_files_only=True)
        self.network = self.pipeline.model
        self.network.requires_grad_(False)
        if lora:
            self.network.get_input_embeddings=MethodType(embeddings,self.network)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            self.network=get_peft_model(self.network,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q','v'],bias='none'))
            self.pipeline.model=self.network.get_base_model()
        self.eval()

    def forward(self, x):
        return self.network(context=x).quantile_preds[:, :, :24].transpose(1, 2)

    @torch.no_grad()
    def predict(self, xs, batch_size=16):
        qs = []
        for j in range(0, len(xs), batch_size):
            x = torch.as_tensor(xs[j:j+batch_size], device='cuda', dtype=torch.float32)
            q = self(x)
            assert torch.isfinite(q).all()
            qs.append(q.cpu())
        return torch.cat(qs).numpy()

    def learned(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def load_learned(self,state):
        params=dict(self.named_parameters())
        assert set(state)=={n for n,p in params.items() if p.requires_grad}
        with torch.no_grad():
            for name,value in state.items():
                params[name].copy_(value)

    def frozen_hash(self):
        return tensor_hash([(n,p) for n,p in self.named_parameters() if not p.requires_grad]+list(self.named_buffers()))


def loss(q,y,sigma):
    levels=torch.arange(1,10,device=q.device,dtype=q.dtype)/10
    error=y[...,None]-q
    return ((2*torch.maximum(levels*error,(levels-1)*error)).mean((-1,-2))/sigma).mean()


@torch.no_grad()
def extract_basis(lora_model):
    basis={}
    for name,module in lora_model.network.get_base_model().named_modules():
        if not hasattr(module,'lora_A'):
            continue
        a=module.lora_A['default'].weight;b=module.lora_B['default'].weight
        qb,rb=torch.linalg.qr(b,mode='reduced');qa,ra=torch.linalg.qr(a.T,mode='reduced')
        u,s,vt=torch.linalg.svd((rb@ra.T)*module.scaling['default'],full_matrices=False)
        basis[name]=dict(u=(qb@u).cpu(),s=s.cpu(),v=(qa@vt.T).cpu(),c=(torch.linalg.vector_norm(s)/8**.5).cpu())
    assert len(basis)==36, 'IMPLEMENTATION_FAILURE: unexpected q/v module count'
    assert any(float(v['c'])>0 for v in basis.values()), 'IMPLEMENTATION_FAILURE: zero shared adaptation basis'
    return basis


class CoefficientLinear(nn.Module):
    def __init__(self,base,basis):
        super().__init__();self.base=base
        for key in ['u','s','v','c']:
            self.register_buffer(key,basis[key].to(base.weight.device))
        self.d=nn.Parameter(torch.zeros_like(self.s))

    def forward(self,x):
        return self.base(x)+((x@self.v)*(self.s+self.c*self.d))@self.u.T


class CoefficientModel(ForecastModel):
    def __init__(self,basis):
        super().__init__()
        for name,b in basis.items():
            parent_name,attr=name.rsplit('.',1)
            parent=self.network.get_submodule(parent_name)
            setattr(parent,attr,CoefficientLinear(getattr(parent,attr),b))
        self.eval()


def bias_features(x,arm):
    # Inputs contain NaN at hidden observations; no target or hidden truth is accepted.
    observed=torch.isfinite(x).reshape(len(x),12,16)
    dtype=x.dtype;device=x.device
    grid=torch.arange(192,device=device,dtype=dtype).reshape(12,16)
    count=observed.sum(-1);valid=count>0;full=count==16
    denom=count.clamp_min(1).to(dtype)
    centroid=(grid[None]*observed).sum(-1)/denom
    original=grid.mean(-1)
    taus=[16.,64.,192.]
    if arm=='SET_BIAS':
        distance=grid[:,None,:,None]-grid[None,:,None,:]
        support=observed[:,:,None,:,None]&observed[:,None,:,None,:]
        features=[]
        for tau in taus:
            logw=-.5*(distance/tau)**2
            logbase=logw.flatten(-2).logsumexp(-1)-torch.log(torch.tensor(256.,device=device))
            logobs=logw[None].expand(len(x),-1,-1,-1,-1).masked_fill(~support,-torch.inf).flatten(-2).logsumexp(-1)
            logobs=logobs-denom.log()[:,:,None]-denom.log()[:,None,:]
            f=torch.where(valid[:,:,None]&valid[:,None,:],logobs-logbase,0.)
            f=torch.where(full[:,:,None]&full[:,None,:],0.,f)
            features.append(f)
        result=torch.stack(features,1)
    elif arm=='CENTROID_BIAS':
        dist=centroid[:,:,None]-centroid[:,None,:]
        base=original[:,None]-original[None,:]
        result=torch.stack([torch.exp(-.5*(dist/t)**2)-torch.exp(-.5*(base/t)**2)[None] for t in taus],1)
    elif arm=='KEY_BIAS':
        last=torch.where(observed,grid[None],-torch.inf).max(-1).values
        key=torch.stack([count.to(dtype)/16-1,(centroid-original)/16,(last-grid[:,-1])/16],1)
        key=torch.where(valid[:,None,:],key,0.)
        result=key[:,:,None,:].expand(-1,-1,12,-1)
    elif arm=='GENERIC_BIAS':
        dist=original[:,None]-original[None,:]
        result=torch.stack([torch.exp(-.5*(dist/t)**2) for t in taus],0)[None].expand(len(x),-1,-1,-1)
        result=result*(~observed.all((-1,-2)))[:,None,None,None]
    else:
        raise ValueError(arm)
    result=torch.where((valid[:,:,None]&valid[:,None,:])[:,None],result,0.)
    # Native REG token has no timestamp. Its row and column are not modified.
    return torch.nn.functional.pad(result,(0,1,0,1))


class BiasModel(ForecastModel):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        self.alpha=nn.Parameter(torch.zeros((8,3),device='cuda'))
        self.features=None
        def add_bias(module,args,kwargs):
            assert kwargs['position_bias'] is not None
            delta=torch.einsum('hk,bkij->bhij',self.alpha,self.features)
            kwargs['position_bias']=kwargs['position_bias']+delta
            return args,kwargs
        target=self.network.encoder.block[-1].layer[0].SelfAttention
        self.hook=target.register_forward_pre_hook(add_bias,with_kwargs=True)

    def forward(self,x):
        with torch.no_grad():
            self.features=bias_features(x,self.arm)
        return super().forward(x)
