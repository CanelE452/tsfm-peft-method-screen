"""Preserve the trained raw-input B0; add zero-output patch residuals only."""
import torch
from torch import nn
from experiments.outlier_signal_peft_v1_20260917.model import loss_2pinball
from experiments.outlier_signal_followup_v2_20260917.model import coordinates

def correction_gate(observed,sigma,mode='normal'):
    _,_,_,p=coordinates(observed,sigma)
    if mode=='zero':p=torch.zeros_like(p)
    elif mode=='permute':
        order=torch.randperm(512,generator=torch.Generator().manual_seed(84400)).to(p.device);p=p[:,order]
    else:assert mode=='normal'
    # p from observed past only; no clipping is applied to model input.
    return 1-p.reshape(len(p),32,16).mean(-1)

class ResidualAdapter(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+200000)
            self.down=nn.Linear(512,8);self.up=nn.Linear(8,512)
            nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,embeds,gate):
        cap=embeds.square().mean(-1).sqrt().quantile(.5,dim=-1,keepdim=True).clamp_min(1e-6).detach()[...,None]
        delta=cap*torch.tanh(self.up(nn.functional.gelu(self.down(embeds))))
        return embeds+gate[...,None]*delta

class ForecastModel(nn.Module):
    def __init__(self,base,arm,seed):
        super().__init__();self.base=base;self.arm=arm
        self.adapter=ResidualAdapter(seed) if arm in ['C2','C3'] else None
    def forward(self,observed,sigma,residual_mode='normal',persistence_mode='normal'):
        assert observed.ndim==2 and observed.shape[-1]==512 and sigma.shape==observed.shape[:1]
        assert torch.isfinite(observed).all() and (sigma>0).all()
        b=self.base;normalized,loc_scale=b.instance_norm(observed)
        z=b.patch(normalized.to(b.dtype));patch_mask=b.patch(torch.ones_like(observed))
        embeds=b.input_patch_embedding(torch.cat([z,patch_mask],dim=-1))
        if self.adapter is not None and residual_mode!='off':
            assert residual_mode=='normal'
            gate=correction_gate(observed,sigma,persistence_mode) if self.arm=='C3' else torch.ones_like(z[:,:,0])
            embeds=self.adapter(embeds,gate)
        attention_mask=(patch_mask.sum(dim=-1)>0).to(b.dtype)
        if b.chronos_config.use_reg_token:
            reg=torch.full((len(observed),1),b.config.reg_token_id,device=observed.device,dtype=torch.long)
            embeds=torch.cat([embeds,b.shared(reg)],dim=1)
            attention_mask=torch.cat([attention_mask,torch.ones_like(reg,dtype=b.dtype)],dim=1)
        hidden=b.encoder(attention_mask=attention_mask,inputs_embeds=embeds)[0]
        output=b.decode(embeds,attention_mask,hidden)
        quantiles=b.output_patch_embedding(output).reshape(len(observed),9,64)
        return b.instance_norm.inverse(quantiles.flatten(1),loc_scale).reshape(len(observed),9,64)
