"""Controlled channel sharing atop the pinned Time-PEFT public forecasting path."""
from pathlib import Path
import sys,importlib.util,hashlib,random,json
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'sources/timepeft_ea4e7e1'
sys.path.insert(0,str(UPSTREAM))
spec=importlib.util.spec_from_file_location('timepeft_upstream_pinned',UPSTREAM/'run.py')
up=importlib.util.module_from_spec(spec);spec.loader.exec_module(up)
ARMS=['LORA_HEAD','SPECIFIC','SHARED','SHARED_WIDE','GROUP4','BASIS4']

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def groups(channel_ids,seed=60010):
    order=sorted(range(len(channel_ids)),key=lambda i:hashlib.sha256(f'{seed}:{channel_ids[i]}'.encode()).hexdigest())
    result=[0]*len(order)
    for rank,i in enumerate(order):result[i]=rank%4
    return result

def channel_count(arm,d=512,c=64,r=256,k=4):
    if arm=='LORA_HEAD':return 0
    if arm in ['SHARED','SHARED_WIDE']:
        s=513 if arm=='SHARED_WIDE' else r
        return (3*d+1)*s+3*d
    return (2*d+1)*r+(c if arm=='SPECIFIC' else k)*d*(r+1)+2*d+(c*k if arm=='BASIS4' else 0)

class ChannelBlock(nn.Module):
    def __init__(self,arm,channel_ids,d=512,r=256,k=4,seed=40000,wide=None):
        super().__init__();assert arm in ARMS[1:]
        self.arm=arm;self.channel_ids=tuple(channel_ids);self.id_to_index={v:i for i,v in enumerate(channel_ids)};assert len(self.id_to_index)==len(channel_ids)
        self.d=d;self.r=r;self.k=k;self.width=(513 if wide is None else wide) if arm=='SHARED_WIDE' else r
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+100)
            down=nn.Linear(2*d,r);up0=nn.Linear(r,d)
            self.down_projection=nn.Linear(2*d,self.width);self.activation=nn.ReLU();self.dropout=nn.Dropout(.1);self.layer_norm=nn.LayerNorm(d)
            count=len(channel_ids) if arm=='SPECIFIC' else 1 if arm in ['SHARED','SHARED_WIDE'] else k
            self.up_projections=nn.ModuleList(nn.Linear(self.width,d) for _ in range(count))
            with torch.no_grad():
                self.down_projection.weight[:r].copy_(down.weight);self.down_projection.bias[:r].copy_(down.bias)
                for i,u in enumerate(self.up_projections):
                    if arm!='BASIS4' or i==0:
                        u.weight[:,:r].copy_(up0.weight);u.bias.copy_(up0.bias)
                    if self.width>r:u.weight[:,r:].zero_()
            if arm=='BASIS4':
                self.coefficients=nn.Parameter(torch.zeros(len(channel_ids),k));self.coefficients.data[:,0]=1
        self.register_buffer('group_indices',torch.tensor(groups(channel_ids)))
    def indices(self,ids,device):
        ids=self.channel_ids if ids is None else tuple(ids)
        return torch.tensor([self.id_to_index[v] for v in ids],device=device)
    def forward(self,h,f,channel_ids=None):
        assert h.shape==f.shape and h.shape[-1]==self.d
        ids=self.indices(channel_ids,h.device);assert len(ids)==h.shape[1]
        u=self.dropout(self.activation(self.down_projection(torch.cat([h,f],-1))))
        if self.arm in ['SHARED','SHARED_WIDE']:out=self.up_projections[0](u)
        elif self.arm=='BASIS4':
            terms=torch.stack([layer(u) for layer in self.up_projections],dim=-2)
            out=torch.einsum('bcnkd,ck->bcnd',terms,self.coefficients[ids])
        else:
            pick=ids if self.arm=='SPECIFIC' else self.group_indices[ids]
            out=torch.stack([self.up_projections[int(i)](u[:,j]) for j,i in enumerate(pick)],dim=1)
        return self.layer_norm(out)

class ChannelPipeline(nn.Module):
    def __init__(self,base,arm,channel_ids,seed):
        super().__init__();self.arm=arm;self.channel_ids=tuple(channel_ids)
        for key in ['normalizer','tokenizer','patch_embedding','encoder','head']:setattr(self,key,getattr(base,key))
        if arm!='LORA_HEAD':
            with torch.random.fork_rng(devices=[]):torch.manual_seed(seed+50);self.frequency_adapter=up.FrequencyAdapter(3,512)
            self.channel_adapter=ChannelBlock(arm,channel_ids,seed=seed)
        self.requires_grad_(False);self.head.requires_grad_(True)
        if arm!='LORA_HEAD':self.frequency_adapter.requires_grad_(True);self.channel_adapter.requires_grad_(True)
        for n,p in self.encoder.named_parameters():
            if 'lora_' in n:p.requires_grad_(True)
    def forward(self,x_enc,input_mask=None,channel_ids=None):
        b,c,t=x_enc.shape
        if input_mask is None:input_mask=torch.ones(b,t,device=x_enc.device,dtype=torch.bool)
        x=self.normalizer(x=x_enc,mask=input_mask,mode='norm');x=torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
        patches=self.patch_embedding(self.tokenizer(x),mask=input_mask);b,c,n,d=patches.shape
        mask=up.Masking.convert_seq_to_patch_view(input_mask,self.patch_embedding.patch_len).repeat_interleave(c,dim=0)
        hidden=self.encoder(inputs_embeds=patches.reshape(b*c,n,d),attention_mask=mask).last_hidden_state.reshape(b,c,n,d)
        if self.arm!='LORA_HEAD':
            assert hidden.dtype==torch.float32,'Fourier path must be FP32'
            hidden=self.channel_adapter(hidden,self.frequency_adapter(hidden),channel_ids)
        pred=self.normalizer(x=self.head(hidden),mode='denorm')
        return up.TimeseriesOutputs(input_mask=input_mask,forecast=pred)

def make_model(arm,channel_ids,seed,checkpoint=False,device='cpu'):
    seed_all(seed)
    receipt=json.loads((ROOT/'results/priority12_20260915/moment_source.json').read_text())
    base=up.MOMENTPipeline.from_pretrained(receipt['snapshot'],model_kwargs=dict(task_name='forecasting',forecast_horizon=96,head_dropout=.1,weight_decay=0,freeze_encoder=True,freeze_embedder=True,freeze_head=False,seq_len=512))
    base.init();assert base.encoder.config.d_model==512 and base.patch_embedding.patch_len==8 and base.head.linear.in_features==32768
    base.encoder=up.get_peft_model(base.encoder,up.LoraConfig(task_type=up.TaskType.FEATURE_EXTRACTION,r=8,lora_alpha=32,target_modules=['q','k','v']))
    if checkpoint:base.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    else:base.encoder.gradient_checkpointing_disable()
    model=ChannelPipeline(base,arm,channel_ids,seed).to(device)
    actual=sum(p.numel() for p in model.channel_adapter.parameters()) if arm!='LORA_HEAD' else 0
    assert actual==channel_count(arm,c=len(channel_ids))
    return model

def inventory(model):
    params={n:dict(shape=list(p.shape),numel=p.numel(),trainable=p.requires_grad) for n,p in model.named_parameters()}
    groups={g:sum(v['numel'] for n,v in params.items() if v['trainable'] and (('lora_' in n) if g=='lora' else n.startswith(g+'.'))) for g in ['lora','head','frequency_adapter','channel_adapter']}
    total=sum(v['numel'] for v in params.values() if v['trainable']);assert sum(groups.values())==total
    return dict(arm=model.arm,groups=groups,trainable=total,total=sum(v['numel'] for v in params.values()),parameters=params)
