"""Local A-gradient codecs. Exact forward, B gradient and input propagation.

CARE/PRAC implementations below are scoped primitives, not official reproductions.
No reconstructed full X is used by the residual or subspace backward.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math,weakref
import torch
from torch import nn
from torch.multiprocessing.reductions import StorageWeakRef
from ..lora import LowRank

@dataclass
class LocalPayload:
    kind: str
    parts: tuple
    shape: tuple
    group: bool
    weak: object=None

    def grad(self,incoming):
        u=incoming.transpose(0,1) if self.group else incoming
        b,t,r=u.shape;d=self.shape[-1]
        if self.kind in ('exact','fp16'):
            return u.reshape(-1,r).T@self.parts[0].to(u.dtype).reshape(-1,d)
        if self.kind=='care':
            z,decoder=self.parts
            return (incoming.reshape(-1,r).T@z.reshape(-1,r))@decoder
        if self.kind=='prac':
            zp,zq,vp,vq=self.parts
            n=d-vp.shape[1]
            flat=u.reshape(-1,r)
            return (flat.T@zp)@vp.T+(n/vq.shape[1])*(flat.T@zq)@vq.T
        mean,special,detail,index,invprob=self.parts
        uc=(u[:,:-4:2]+u[:,1:-4:2])/2
        result=2*uc.reshape(-1,r).T@mean.reshape(-1,d)
        result+=u[:,-4:].reshape(-1,r).T@special.reshape(-1,d)
        du=u[:,:-4:2]-u[:,1:-4:2]
        if self.kind=='all_details':
            result+=.5*du.reshape(-1,r).T@detail.reshape(-1,d)
        else:
            selected=du.gather(1,index[:,:,None].expand(-1,-1,r))
            result+=(.5/index.shape[1])*(selected*invprob[:,:,None]).reshape(-1,r).T@detail.reshape(-1,d)
        return result

    def nbytes(self):return sum(v.numel()*v.element_size() for v in self.parts if isinstance(v,torch.Tensor))

class LocalCodecs:
    def __init__(self,mode,seed=51000):
        assert mode in ('exact','fp16','care','prac','residual','all_details')
        self.mode=mode;self.seed=seed;self.cache=weakref.WeakValueDictionary();self.generators={}
        self.stats=dict(payload_bytes=0,unique_payloads=0,shared_hits=0,source_bytes=0)
    def generator(self,device):
        key=str(device)
        if key not in self.generators:self.generators[key]=torch.Generator(device=device).manual_seed(self.seed)
        return self.generators[key]
    def pack(self,x,z,group):
        storage=x.untyped_storage();key=(storage._cdata,x._version,tuple(x.shape),tuple(x.stride()),group)
        old=self.cache.get(key) if self.mode!='care' else None
        if old is not None:self.stats['shared_hits']+=1;return old
        v=x.transpose(0,1) if group else x
        b,t,d=v.shape;mode=self.mode
        if mode in ('exact','fp16'):parts=(v.clone() if mode=='exact' else v.half(),)
        elif mode=='care':
            zz=z.reshape(-1,z.shape[-1]);flat=x.reshape(-1,d)
            gram=zz.T@zz;ridge=(gram.diag().mean()*1e-4+1e-8)
            decoder=torch.linalg.solve(gram+ridge*torch.eye(gram.shape[0],device=x.device,dtype=x.dtype),zz.T@flat)
            parts=(z,decoder)
        elif mode=='prac':
            flat=v.reshape(-1,d);k=min(8,d//3);gen=self.generator(x.device)
            # One randomized subspace iteration, no SVD; count all costs per pass.
            omega=torch.randn(d,k,device=x.device,dtype=x.dtype,generator=gen)
            q=torch.linalg.qr(flat@omega,mode='reduced').Q
            vp=torch.linalg.qr(flat.T@q,mode='reduced').Q
            tail=torch.randn(d,k,device=x.device,dtype=x.dtype,generator=gen)
            tail-=vp@(vp.T@tail)
            vq=torch.linalg.qr(tail,mode='reduced').Q
            parts=(flat@vp,flat@vq,vp,vq)
        else:
            assert t>=6 and (t-4)%2==0
            left=v[:,:-4:2];right=v[:,1:-4:2]
            mean=(left+right)/2;delta=left-right;special=v[:,-4:].clone();n=mean.shape[1]
            if mode=='all_details':parts=(mean,special,delta,None,None)
            else:
                norms=delta.norm(dim=-1);total=norms.sum(-1,keepdim=True)
                weights=.9*norms/total.clamp_min(1e-20)+.1/n
                weights=torch.where(total>0,weights,torch.full_like(weights,1/n))
                m=max(1,math.ceil(n/4))
                index=torch.multinomial(weights,m,replacement=True,generator=self.generator(x.device))
                prob=weights.gather(1,index)
                details=delta.gather(1,index[:,:,None].expand(-1,-1,d))
                parts=(mean,special,details,index,1/prob)
        payload=LocalPayload(mode,parts,tuple(v.shape),group,StorageWeakRef(storage))
        if mode!='care':self.cache[key]=payload
        self.stats['payload_bytes']+=payload.nbytes();self.stats['unique_payloads']+=1;self.stats['source_bytes']+=x.numel()*x.element_size()
        return payload

class LocalA(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,a,codecs,group):
        z=nn.functional.linear(x,a)
        ctx.save_for_backward(a);ctx.payload=codecs.pack(x,z,group)
        return z
    @staticmethod
    def backward(ctx,incoming):
        (a,)=ctx.saved_tensors
        payload=ctx.payload;del ctx.payload
        da=payload.grad(incoming)
        dx=incoming@a
        return dx,da,None,None

@contextmanager
def local_backward(model,mode,seed=51000):
    codecs=LocalCodecs(mode,seed);originals=[]
    for layer in model.modules():
        if not isinstance(layer,LowRank):continue
        assert layer.mode=='standard'
        original=layer.forward;originals.append((layer,original))
        def forward(x,_layer=layer):
            out=_layer.base(x)
            if not _layer.enabled:return out
            z=LocalA.apply(x,_layer.lora_A,codecs,_layer.group_attention)
            return out+2*nn.functional.linear(z,_layer.lora_B)
        layer.forward=forward
    try:yield codecs
    finally:
        for layer,original in originals:layer.forward=original
