import torch
from torch import nn
from experiments.additive_persistence_validation_v1_20260917.model import ResidualAdapter,ForecastModel,loss_2pinball
from experiments.outlier_signal_peft_v1_20260917.model import robust_scale
class ControlModel(ForecastModel):
    def __init__(self,base,arm,seed):
        super().__init__(base,'C2' if arm!='OUTPUT_CONTEXT' else 'C0',seed);self.arm=arm
        if arm=='POS_ONLY':self.position_logits=nn.Parameter(torch.zeros(32))
        if arm=='OUTPUT_CONTEXT':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+300000);self.output_linear=nn.Linear(72,64);nn.init.normal_(self.output_linear.weight,std=.01);nn.init.zeros_(self.output_linear.bias)
            self.gamma=nn.Parameter(torch.zeros(()))
    def gate(self,x,s):
        if self.arm=='POS_ONLY':return self.position_logits.sigmoid()[None,:].expand(len(x),-1)
        assert self.arm=='MAG_ONLY';m,r=robust_scale(x,s);return 1-(((x-m)/r).abs()>3).to(x).reshape(len(x),32,16).mean(-1)
    def forward(self,observed,sigma,residual_mode='normal',persistence_mode='normal'):
        assert residual_mode in ['normal','off'] and persistence_mode=='normal'
        assert observed.ndim==2 and observed.shape[-1]==512 and sigma.shape==observed.shape[:1]
        assert torch.isfinite(observed).all() and (sigma>0).all()
        if self.arm=='OUTPUT_CONTEXT':
            with torch.no_grad():
                q=ForecastModel.forward(self,observed,sigma,residual_mode='off')
                normalized,loc_scale=self.base.instance_norm(observed)
                assert not self.base.instance_norm.use_arcsinh
                qnorm,_=self.base.instance_norm(q[:,4],loc_scale)
                features=torch.cat([qnorm,normalized.reshape(len(observed),8,64).mean(-1)],-1)
            if residual_mode=='off':return q
            correction=loc_scale[1]*torch.tanh(self.gamma)*self.output_linear(features)
            return q+correction[:,None,:]
        b=self.base;normalized,loc_scale=b.instance_norm(observed)
        z=b.patch(normalized.to(b.dtype));patch_mask=b.patch(torch.ones_like(observed))
        embeds=b.input_patch_embedding(torch.cat([z,patch_mask],dim=-1))
        if residual_mode!='off':embeds=self.adapter(embeds,self.gate(observed,sigma))
        attention_mask=(patch_mask.sum(dim=-1)>0).to(b.dtype)
        if b.chronos_config.use_reg_token:
            reg=torch.full((len(observed),1),b.config.reg_token_id,device=observed.device,dtype=torch.long)
            embeds=torch.cat([embeds,b.shared(reg)],dim=1);attention_mask=torch.cat([attention_mask,torch.ones_like(reg,dtype=b.dtype)],dim=1)
        hidden=b.encoder(attention_mask=attention_mask,inputs_embeds=embeds)[0];output=b.decode(embeds,attention_mask,hidden)
        quantiles=b.output_patch_embedding(output).reshape(len(observed),9,64)
        return b.instance_norm.inverse(quantiles.flatten(1),loc_scale).reshape(len(observed),9,64)
