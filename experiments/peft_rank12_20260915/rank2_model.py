"""Known affine-sharing parameterizations on the pinned Time-PEFT path."""
import math,copy
import torch
from torch import nn
from common import ROOT,read,parameters
from priority12.channel_model import up,seed_all,inventory
ARMS=['LH','INDIV_REF','INDIV_BUDGET','SHARED_BUDGET','FACTOR_BUDGET','BASIS_BUDGET']
WIDTH={'INDIV_REF':256,'INDIV_BUDGET':16,'SHARED_BUDGET':191,'FACTOR_BUDGET':95,'BASIS_BUDGET':95}
def count(arm,d=512,c=32):
    if arm=='LH':return 0
    h=WIDTH[arm]
    if arm.startswith('INDIV'):return (2*d+1+c*d)*h+c*d+2*d
    if arm=='SHARED_BUDGET':return (3*d+1)*h+3*d
    if arm=='FACTOR_BUDGET':return (2*d+1)*h+c*h*51+51*d+c*d+2*d
    return (2*d+1+4*d)*h+4*d+c*3+2*d
class Channel(nn.Module):
    def __init__(self,arm,ids,d=512,seed=41000,width=None,q=51):
        super().__init__();self.arm=arm;self.ids=tuple(ids);self.lookup={v:i for i,v in enumerate(ids)};self.d=d;self.h=WIDTH[arm] if width is None else width;self.q=q
        assert len(self.lookup)==len(ids)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+100)
            self.down_projection=nn.Linear(2*d,self.h);self.activation=nn.ReLU();self.dropout=nn.Dropout(.1);self.layer_norm=nn.LayerNorm(d)
            if arm=='FACTOR_BUDGET':
                common=torch.randn(self.h,q)/math.sqrt(self.h)
                self.left=nn.Parameter(common.repeat(len(ids),1,1));self.right=nn.Parameter(torch.randn(q,d)/math.sqrt(3*q));self.bias=nn.Parameter(torch.zeros(len(ids),d))
            else:
                n=len(ids) if arm.startswith('INDIV') else 4 if arm=='BASIS_BUDGET' else 1
                u=nn.Linear(self.h,d);self.up_projections=nn.ModuleList(copy.deepcopy(u) for _ in range(n))
                for layer in self.up_projections:nn.init.zeros_(layer.bias)
                if arm=='BASIS_BUDGET':
                    for layer in self.up_projections[1:]:nn.init.zeros_(layer.weight)
                    self.coefficients=nn.Parameter(torch.randn(len(ids),3)*.1)
    def indices(self,ids,device):return torch.tensor([self.lookup[v] for v in (self.ids if ids is None else ids)],device=device)
    def affine(self,z,ids,merged=False):
        if self.arm=='FACTOR_BUDGET':return torch.einsum('bcnh,chq,qd->bcnd',z,self.left[ids],self.right)+self.bias[ids][None,:,None,:]
        if self.arm=='SHARED_BUDGET':return self.up_projections[0](z)
        if self.arm.startswith('INDIV'):return torch.stack([self.up_projections[int(i)](z[:,j]) for j,i in enumerate(ids)],1)
        if merged:
            u0=self.up_projections[0];weights=torch.stack([u.weight for u in self.up_projections[1:]]);bias=torch.stack([u.bias for u in self.up_projections[1:]])
            w=u0.weight[None]+torch.einsum('ck,kdh->cdh',self.coefficients[ids],weights);b=u0.bias[None]+self.coefficients[ids]@bias
            return torch.einsum('bcnh,cdh->bcnd',z,w)+b[None,:,None,:]
        return self.up_projections[0](z)+torch.einsum('bcnkd,ck->bcnd',torch.stack([u(z) for u in self.up_projections[1:]],-2),self.coefficients[ids])
    def forward(self,h,f,channel_ids=None,merged=False):
        ids=self.indices(channel_ids,h.device);assert len(ids)==h.shape[1]
        z=self.dropout(self.activation(self.down_projection(torch.cat([h,f],-1))))
        return self.layer_norm(self.affine(z,ids,merged))
class Model(nn.Module):
    def __init__(self,base,arm,ids,seed):
        super().__init__();self.arm=arm;self.channel_ids=tuple(ids)
        for key in ['normalizer','tokenizer','patch_embedding','encoder','head']:setattr(self,key,getattr(base,key))
        if arm!='LH':
            with torch.random.fork_rng(devices=[]):torch.manual_seed(seed+50);self.frequency_adapter=up.FrequencyAdapter(3,512)
            self.channel_adapter=Channel(arm,ids,seed=seed)
        self.requires_grad_(False);self.head.requires_grad_(True)
        if arm!='LH':self.frequency_adapter.requires_grad_(True);self.channel_adapter.requires_grad_(True)
        for n,p in self.encoder.named_parameters():
            if 'lora_' in n:p.requires_grad_(True)
    def forward(self,x_enc,input_mask=None,channel_ids=None,merged=False):
        b,c,t=x_enc.shape;assert t==96
        if input_mask is None:input_mask=torch.ones(b,t,device=x_enc.device,dtype=torch.bool)
        x=self.normalizer(x=x_enc,mask=input_mask,mode='norm');x=torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
        patches=self.patch_embedding(self.tokenizer(x),mask=input_mask);b,c,n,d=patches.shape;assert (n,d)==(12,512)
        mask=up.Masking.convert_seq_to_patch_view(input_mask,self.patch_embedding.patch_len).repeat_interleave(c,dim=0)
        hidden=self.encoder(inputs_embeds=patches.reshape(b*c,n,d),attention_mask=mask).last_hidden_state.reshape(b,c,n,d)
        if self.arm!='LH':
            # FFT and frequency projection explicitly FP32; surrounding BF16 is common to every arm.
            with torch.autocast(device_type=x.device.type,enabled=False):freq=self.frequency_adapter(hidden.float())
            hidden=self.channel_adapter(hidden,freq,channel_ids,merged)
        return up.TimeseriesOutputs(input_mask=input_mask,forecast=self.normalizer(x=self.head(hidden),mode='denorm'))
def make(arm,ids,seed,device='cpu'):
    seed_all(seed);receipt=read(ROOT/'results/priority12_20260915/moment_source.json')
    base=up.MOMENTPipeline.from_pretrained(receipt['snapshot'],model_kwargs=dict(task_name='forecasting',forecast_horizon=96,head_dropout=.1,weight_decay=0,freeze_encoder=True,freeze_embedder=True,freeze_head=False,seq_len=96));base.init()
    assert base.encoder.config.d_model==512 and base.patch_embedding.patch_len==8 and base.head.linear.in_features==6144,'CONTRACT_MISMATCH'
    base.encoder=up.get_peft_model(base.encoder,up.LoraConfig(task_type=up.TaskType.FEATURE_EXTRACTION,r=8,lora_alpha=32,target_modules=['q','k','v']))
    base.encoder.gradient_checkpointing_disable();m=Model(base,arm,ids,seed).to(device)
    assert sum(p.numel() for p in m.channel_adapter.parameters())==count(arm) if arm!='LH' else True
    return m
