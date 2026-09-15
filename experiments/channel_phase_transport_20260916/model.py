"""Input-conditioned same-phase transport inside a frozen-encoder residual adapter."""
import sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/peft_rank12_20260915'))
from rank2_model import make as original_make
from priority12.channel_model import up
ARMS=['POINTWISE','UNIFORM','CONDITIONED']

class Transport(nn.Module):
    def __init__(self,arm,d=512,r=16,seed=41000):
        super().__init__();assert arm in ARMS;self.arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+202)
            self.down=nn.Linear(d,r,bias=False);self.up=nn.Linear(r,d,bias=False)
            nn.init.zeros_(self.up.weight)
    def weights(self,patches):
        n=patches.shape[-2];assert n==12 and patches.shape[-1]==8
        dtype=torch.float64 if patches.dtype==torch.float64 else torch.float32
        with torch.autocast(device_type=patches.device.type,enabled=False):
            x=patches.to(dtype);idx=torch.arange(n,device=x.device)
            mask=(idx[:,None]-idx[None,:]).remainder(3)==0
            if self.arm=='POINTWISE':return torch.eye(n,device=x.device,dtype=dtype).expand(*x.shape[:-2],n,n)
            if self.arm=='UNIFORM':return (mask.to(dtype)/4).expand(*x.shape[:-2],n,n)
            distance=(x.unsqueeze(-2)-x.unsqueeze(-3)).square().mean(-1)
            return (-distance).masked_fill(~mask,-torch.inf).softmax(-1)
    def forward(self,h,patches):
        z=self.down(h);t=self.weights(patches)
        z=torch.matmul(t.to(z.dtype),z)
        return h+self.up(torch.nn.functional.gelu(z))

class Model(nn.Module):
    def __init__(self,base,arm,seed):
        super().__init__()
        for key in ['normalizer','tokenizer','patch_embedding','encoder','head']:setattr(self,key,getattr(base,key))
        self.requires_grad_(False);self.head.requires_grad_(True)
        self.adapter=Transport(arm,seed=seed)
    def forward(self,x_enc,input_mask=None):
        b,c,t=x_enc.shape;assert t==96
        if input_mask is None:input_mask=torch.ones(b,t,device=x_enc.device,dtype=torch.bool)
        x=self.normalizer(x=x_enc,mask=input_mask,mode='norm');x=torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
        rawpatch=self.tokenizer(x);patches=self.patch_embedding(rawpatch,mask=input_mask)
        b,c,n,d=patches.shape;assert (n,d)==(12,512)
        mask=up.Masking.convert_seq_to_patch_view(input_mask,self.patch_embedding.patch_len).repeat_interleave(c,dim=0)
        hidden=self.encoder(inputs_embeds=patches.reshape(b*c,n,d),attention_mask=mask).last_hidden_state.reshape(b,c,n,d)
        hidden=self.adapter(hidden,rawpatch)
        return up.TimeseriesOutputs(input_mask=input_mask,forecast=self.normalizer(x=self.head(hidden),mode='denorm'))

def make(arm,ids,seed,device='cpu'):
    base=original_make('LH',ids,seed,'cpu')
    return Model(base,arm,seed).to(device)
