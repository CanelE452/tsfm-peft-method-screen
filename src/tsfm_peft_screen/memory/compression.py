"""Saved-storage primitives, not official reproductions of published methods."""
from contextlib import contextmanager
from dataclasses import dataclass
import weakref
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils.checkpoint import checkpoint

@dataclass
class Payload:
    weak: object
    kind: str
    data: torch.Tensor
    extra: object = None
    special: object = None
    layout: str = ''
    shape: tuple = ()

    def unpack(self):
        if self.kind == 'fp16': return self.data.float()
        if self.kind == 'temporal':
            context = self.data.float().repeat_interleave(2, dim=1)
        else:
            context = self.data.float() * self.extra
        value = torch.cat((context, self.special), dim=1)
        if self.layout == 'tbc': value = value.transpose(0, 1).contiguous()
        return value.reshape(-1)

    def nbytes(self):
        return sum(t.numel()*t.element_size() for t in (self.data,self.extra,self.special) if isinstance(t,torch.Tensor))

class CompressedSaved:
    """Deduplicate aliases without retaining original or reconstructed GPU storage.

    StorageWeakRef holds the storage identity alive, not its data allocation.
    Weak payload cache releases compressed tensors when autograd releases them.
    Every unpack reconstructs a fresh backing allocation and its original view.
    """
    def __init__(self,m,mode,batch_size,tokens):
        assert mode in ('fp16','int8','temporal')
        self.m=m;self.mode=mode;self.b=batch_size;self.t=tokens
        self.excluded={p.untyped_storage()._cdata for p in list(m.parameters())+list(m.buffers())}
        self.cache=weakref.WeakValueDictionary();self.stack=[];self.handles=[]
        self.stats=dict(unique_compressed=0,original_bytes=0,payload_bytes=0,hidden_original_bytes=0,alias_hits=0)

    def layout(self,t):
        if t.untyped_storage().nbytes()!=self.b*self.t*768*4 or t.storage_offset()!=0:return None
        if t.ndim==2 and tuple(t.shape)==(self.b*self.t,768) and t.is_contiguous():
            return 'tbc' if '.layer.1.' in (self.stack[-1] if self.stack else '') else 'btc'
        if t.ndim==3:
            if tuple(t.shape)==(self.b,self.t,768) and t.is_contiguous():return 'btc'
            if tuple(t.shape)==(self.t,self.b,768):
                if t.is_contiguous():return 'tbc'
                if tuple(t.stride())==(768,self.t*768,1):return 'btc'
        if tuple(t.shape)==(self.b,12,self.t,64) and tuple(t.stride())==(self.t*768,64,768,1):return 'btc'
        return None

    def pack(self,t):
        s=t.untyped_storage()
        if t.dtype!=torch.float32 or s._cdata in self.excluded or s.nbytes()<65536:return t
        key=(s._cdata,t._version)
        payload=self.cache.get(key)
        if payload is None:
            flat=t.as_strided((s.nbytes()//4,),(1,),0)
            layout=self.layout(t) if self.mode!='fp16' else None
            if layout:
                v=flat.view(self.b,self.t,768) if layout=='btc' else flat.view(self.t,self.b,768).transpose(0,1)
                context=v[:,:-4];special=v[:,-4:].clone();assert context.shape[1]%2==0
                if self.mode=='temporal':
                    data=context.reshape(self.b,-1,2,768).mean(2).half()
                    # Explicit padding equalizes retained bytes with INT8 scales.
                    extra=torch.zeros((self.b,1,768),device=t.device,dtype=torch.float32)
                else:
                    extra=context.abs().amax(1,keepdim=True).clamp_min(1e-20)/127
                    data=(context/extra).round().clamp(-127,127).to(torch.int8)
                payload=Payload(StorageWeakRef(s),self.mode,data,extra,special,layout,tuple(v.shape))
                self.stats['hidden_original_bytes']+=s.nbytes()
            else:
                data=flat.half()
                # Preserve sign/zero status for saved ReLU outputs, including underflow.
                data=torch.where((flat!=0)&(data==0),torch.copysign(torch.full_like(data,2**-24),data),data)
                payload=Payload(StorageWeakRef(s),'fp16',data)
            self.cache[key]=payload
            self.stats['unique_compressed']+=1;self.stats['original_bytes']+=s.nbytes();self.stats['payload_bytes']+=payload.nbytes()
        else:self.stats['alias_hits']+=1
        return (payload,tuple(t.shape),tuple(t.stride()),t.storage_offset())

    @staticmethod
    def unpack(packed):
        if isinstance(packed,torch.Tensor):return packed
        payload,shape,stride,offset=packed
        return payload.unpack().as_strided(shape,stride,offset)

    def leave(self,module,args,out):
        self.stack.pop()

    def __enter__(self):
        for name,module in self.m.named_modules():
            self.handles.append(module.register_forward_pre_hook(lambda mod,args,n=name:self.stack.append(n)))
            self.handles.append(module.register_forward_hook(self.leave))
        self.ctx=torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack);self.ctx.__enter__();return self

    def __exit__(self,*args):
        self.ctx.__exit__(*args)
        for h in self.handles:h.remove()

@contextmanager
def checkpoint_blocks(m):
    originals=[]
    for block in m.encoder.block:
        original=block.forward;originals.append((block,original))
        def wrapped(*args,_original=original,**kwargs):
            return checkpoint(_original,*args,use_reentrant=False,**kwargs)
        block.forward=wrapped
    try:yield
    finally:
        for block,original in originals:block.forward=original
