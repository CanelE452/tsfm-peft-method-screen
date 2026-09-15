"""Matched side attention with or without a frozen backbone attention prior."""
import math
import sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/peft_rank12_20260915'))
from rank2_model import make as original_make
from priority12.channel_model import up
ARMS=['SIDE','PRIOR']

class SideBlock(nn.Module):
    def __init__(self,arm,d=512,r=10):
        super().__init__();assert arm in ARMS;self.arm=arm;self.r=r
        self.norm=nn.LayerNorm(d,eps=1e-5)
        self.q=nn.Linear(d,r,bias=False);self.k=nn.Linear(d,r,bias=False)
        self.v=nn.Linear(d,r,bias=False);self.up=nn.Linear(r,d,bias=False)
        nn.init.zeros_(self.up.weight)
    def forward(self,h,side,log_prior):
        x=self.norm(h+side);q=self.q(x);k=self.k(x);v=self.v(x)
        dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
        with torch.autocast(device_type=q.device.type,enabled=False):
            score=(q.to(dtype)@k.to(dtype).transpose(-1,-2))/math.sqrt(self.r)
            if self.arm=='PRIOR':score=score+log_prior.to(dtype)
            weight=score.softmax(-1)
        return side+self.up(weight.to(v.dtype)@v)

class Model(nn.Module):
    def __init__(self,base,arm,seed):
        super().__init__();self.arm=arm
        for key in ['normalizer','tokenizer','patch_embedding','encoder','head']:setattr(self,key,getattr(base,key))
        self.requires_grad_(False);self.head.requires_grad_(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+9019)
            self.side=nn.ModuleList([SideBlock(arm) for _ in range(8)])
        self._snapshots=[];self._handles=[]
        blocks=self.encoder.base_model.model.block;assert len(blocks)==8
        for i,block in enumerate(blocks):
            attn=block.layer[0].SelfAttention
            def capture_q(module,args,output,i=i):self._snapshots[i]['q']=output.detach()
            def capture_k(module,args,output,i=i):self._snapshots[i]['k']=output.detach()
            def capture_prior(module,args,output,i=i):
                record=self._snapshots[i];q=record.pop('q');k=record.pop('k')
                b,n,_=q.shape;heads=module.n_heads;dim=module.key_value_proj_dim
                q=q.reshape(b,n,heads,dim).transpose(1,2);k=k.reshape(b,n,heads,dim).transpose(1,2)
                # Match the native BF16 score matmul; softmax mixture in stable log space.
                scores=q@k.transpose(-1,-2);scores=scores+output[2]
                with torch.autocast(device_type=q.device.type,enabled=False):
                    logp=torch.logsumexp(scores.float().log_softmax(-1),dim=1)-math.log(heads)
                assert torch.isfinite(logp).all();record['log_prior']=logp.detach()
            self._handles += [attn.q.register_forward_hook(capture_q),attn.k.register_forward_hook(capture_k),attn.register_forward_hook(capture_prior)]
    def forward(self,x_enc,input_mask=None):
        b,c,t=x_enc.shape;assert t==96
        if input_mask is None:input_mask=torch.ones(b,t,device=x_enc.device,dtype=torch.bool)
        assert input_mask.all(), 'This pilot uses fully present context masks'
        self._snapshots=[{} for _ in range(8)]
        with torch.no_grad():
            x=self.normalizer(x=x_enc,mask=input_mask,mode='norm');x=torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
            patches=self.patch_embedding(self.tokenizer(x),mask=input_mask)
            b,c,n,d=patches.shape;assert (n,d)==(12,512)
            mask=up.Masking.convert_seq_to_patch_view(input_mask,self.patch_embedding.patch_len).repeat_interleave(c,dim=0)
            encoded=self.encoder(inputs_embeds=patches.reshape(b*c,n,d),attention_mask=mask,output_hidden_states=True)
        states=encoded.hidden_states[1:];assert len(states)==8
        side=torch.zeros_like(states[0])
        for layer,h,record in zip(self.side,states,self._snapshots):
            assert not h.requires_grad and not record['log_prior'].requires_grad
            side=layer(h,side,record['log_prior'])
        hidden=(encoded.last_hidden_state+side).reshape(b,c,n,d)
        self._snapshots=[]
        return up.TimeseriesOutputs(input_mask=input_mask,forecast=self.normalizer(x=self.head(hidden),mode='denorm'))

def make(arm,ids,seed,device='cpu'):
    return Model(original_make('LH',ids,seed,'cpu'),arm,seed).to(device)
