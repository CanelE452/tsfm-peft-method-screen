"""Mean versus learned convex pooling of native historical patch embeddings.
Real native time embeddings and mean original patch positions are retained.
No pseudo-future, archive lookup, custom backward, or backbone parameter change.
"""
import torch
from torch import nn
from .common import *
from experiments.condition_sampling_repair_v1_20260917.model import Adapted,configure
class Model(Adapted):
 def __init__(self,arm,seed,device='cuda'):
  super().__init__('N02','B1' if arm=='LONG' else 'B0',seed,{},device=device)
  self.kind=arm
  if arm.startswith('LEARN'):
   torch.manual_seed(seed+9000)
   self.pool_v=nn.Parameter(torch.empty(8,self.base.model_dim,device=device));nn.init.xavier_uniform_(self.pool_v)
   self.pool_u=nn.Parameter(torch.zeros(8,device=device))
  self.last_weights=None
 def pooled(self,h):
  assert h.shape[1]==84
  old=h[:,:63].reshape(4,21,3,-1);mean=old.mean(2,keepdim=True)
  if self.kind.startswith('LEARN'):
   logits=torch.einsum('ctkr,r->ctk',torch.tanh(torch.nn.functional.linear(old-mean,self.pool_v)),self.pool_u)
   weights=logits.softmax(-1)
  else:weights=torch.full(old.shape[:3],1/3,device=h.device,dtype=h.dtype)
  self.last_weights=weights.detach()
  return torch.cat([(old*weights[...,None]).sum(2),h[:,63:]],1)
 def forward(self,p,manual_full=False):
  if self.kind in ['SHORT','LONG'] and not manual_full:return super().forward(p)
  b=self.base;x=self.T(p['x']);patched,mask,locscale=b._prepare_patched_context(x)
  assert patched.shape[1]==84 and mask.all()
  h=b.input_patch_embedding(patched)
  if manual_full:pos=torch.arange(84,device=x.device)
  elif self.kind=='STATS_SHORT':h=h[:,63:];pos=torch.arange(63,84,device=x.device)
  else:h=self.pooled(h);pos=torch.cat([torch.arange(1,63,3,device=x.device),torch.arange(63,84,device=x.device)])
  assert b.chronos_config.use_reg_token
  reg=b.shared(torch.full((4,1),b.config.reg_token_id,device=x.device))
  future,_=b._prepare_patched_future(future_covariates=None,future_covariates_mask=None,loc_scale=locscale,num_output_patches=3,batch_size=4)
  h=torch.cat([h,reg,b.input_patch_embedding(future)],1)
  pos=torch.cat([pos,torch.arange(84,88,device=x.device)])[None]
  enc=b.encoder(inputs_embeds=h,attention_mask=torch.ones(h.shape[:2],device=x.device),position_ids=pos,group_ids=torch.zeros(4,dtype=torch.long,device=x.device))
  q=b.output_patch_embedding(enc.last_hidden_state[:,-3:]).reshape(4,3,21,16).permute(0,2,1,3).reshape(4,21,48)
  q=b.instance_norm.inverse(q.reshape(4,-1),locscale).reshape(4,21,48)[:,self.mid]
  if not torch.isfinite(q).all():raise FloatingPointError('NONFINITE_PREDICTION')
  return q
 def objective(self,p,y,teacher):
  q=self(p);s=self.T(p['sigma'])[:,None];task=((q-self.T(y))/s).square().mean()
  kd=((q-self.T(teacher))/s).square().mean() if self.kind.endswith('_KD') else q.new_zeros(())
  return task+.25*kd,dict(task=float(task.detach()),distillation=float(kd.detach()))
