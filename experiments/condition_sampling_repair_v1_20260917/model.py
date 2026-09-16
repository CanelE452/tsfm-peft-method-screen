"""Differentiable input/output transforms connected to the inspected native model API."""
import math,sys
import torch
from torch import nn
from .common import *
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from tsfm_peft_screen.backbone import load_base
from tsfm_peft_screen.lora import attach,disabled
from experiments.peft_rank12_20260915.common import parameters,cpu_state,restore,tensor_hash,frozen_hash,cleanup

def configure():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)

def conditions(t,role):
 if t=='N01':return [f'd{d}_c{c}' for d in ([0,6,24] if role=='V_SELECT' else [0,6,24,12]) for c in range(4)]+(['all24'] if role!='V_SELECT' else [])
 if t=='N03':return ['delta1','delta4'] if role=='V_SELECT' else ['delta2','irregular','delta1','delta4']
 return ['paired' if t=='R04' else 'standard']

def mask_age(x,mask,mu):
 C=x.shape[-1];ix=np.arange(C);last=np.maximum.accumulate(np.where(mask,ix,-1),axis=1);age=np.where(last>=0,ix-last,C);hold=np.where(last>=0,np.take_along_axis(np.where(mask,x,mu[:,None]),np.maximum(last,0),axis=1),mu[:,None]);return age,hold

def clock_mask(origin,epoch,condition):
 M=np.zeros((4,336),bool)
 for c in range(4):
  rr=rng('CLOCK',int(origin),c,int(epoch))
  if condition=='irregular':
   at=int(rr.integers(4));steps=[1,3] if rr.integers(2)==0 else [3,1];k=0
   while at<336:M[c,at]=True;at+=steps[k%2];k+=1
  else:
   d=int(condition[5:]);phase=int(rr.integers(d));M[c,phase::d]=True
 return M

class Packets:
 def __init__(self,t):
  self.t=t;self.spec=read(OUT/t/'PROTOCOL.json');self.stats=read(OUT/t/'train_statistics.json');self.mu=np.array(self.stats['mu']);self.sigma=np.array(self.stats['sigma']);self.aux=read(OUT/t/'feature_or_transform_manifest.json');self.inputs={r:dict(np.load(CACHE/t/f'{r}_inputs.npz')) for r in ROLES};self.labels={r:np.load(CACHE/t/f'{r}_labels.npz')['y'] for r in ['TRAIN','V_SELECT','V_CAL']};self.fixed={};self.prepared={}
 def label(self,role,i):
  assert role!='E_DISCOVERY','Evaluation labels not accessible in model packet API'
  return self.labels[role][i]
 def packet(self,role,i,epoch=-1,condition=None,late=False):
  key=(role,i,epoch,condition,late)
  if key in self.prepared:return self.prepared[key]
  p=self.inputs[role];x=p['late_context' if late else 'context'][i].copy();o=int(p['origins'][i]);v=dict(x=x,origin=o,mu=self.mu,sigma=self.sigma,H=self.spec['H'])
  if self.t=='N01':
   if condition is None:c=i%4;d=[0,6,24][epoch%3];all_delayed=False
   elif condition=='all24':c=0;d=24;all_delayed=True
   else:a,b=condition.split('_');d=int(a[1:]);c=int(b[1:]);all_delayed=False
   M=np.ones_like(x,bool)
   if d:M[:, -d:]=False if all_delayed else M[:,-d:];M[c,-d:]=False
   # Explicit branch avoids Python negative-zero slices for clean data.
   if d==0:M[:]=True
   age,hold=mask_age(x,M,self.mu);safe=np.where(M,x,self.mu[:,None]);z=(safe-self.mu[:,None])/self.sigma[:,None];lh=(hold-self.mu[:,None])/self.sigma[:,None];rz=[];phi=[]
   for cc in range(4):
    ds=[j for j in range(4) if j!=cc];coef=np.array(self.aux['ridge'][cc]);r=coef[0]+np.array(coef[1:])@lh[ds];rz.append(r);phi.append(np.stack([np.ones(336),lh[cc],r,*[z[j]*M[j] for j in ds],*[M[j].astype(float) for j in ds]],-1))
   v.update(x=safe,M=M,age=age,hold=lh,ridge=np.array(rz),phi=np.array(phi),target=c,condition=f'd{d}_c{c}' if not all_delayed else 'all24')
  if self.t=='N03':
   cond=condition or ('delta1' if epoch%2==0 else 'delta4');M=clock_mask(o,epoch,cond);age,hold=mask_age(x,M,self.mu);safe=np.where(M,x,self.mu[:,None]);inds=[];dist=[]
   for c in range(4):
    obs=np.flatnonzero(M[c]);D=abs(np.arange(336)[:,None]-obs[None,:]);j=np.argsort(D,axis=1,kind='stable')[:,:4];inds.append(obs[j]);dist.append(np.take_along_axis(D,j,1))
   v.update(x=safe,M=M,age=age,hold=hold,neighbors=np.array(inds),distances=np.array(dist),condition=cond)
  if self.t=='N02':v.update(retrieval_past=p['retrieval_past'][i],retrieval_future=p['retrieval_future'][i])
  if self.t=='R09':v.update(resolution=p['resolution'][i],detailed=bool(p['detailed'][i]))
  self.prepared[key]=v;return v

class Adapted(nn.Module):
 def __init__(self,t,arm,seed,aux,device='cuda',frozen=False):
  super().__init__();self.t=t;self.arm=arm;self.aux=aux;self.base=load_base(device=device)
  if not frozen:attach(self.base,seed=seed)
  self.extra=nn.ParameterDict();shapes={'A2':{'theta':(4,9)},'A3':{'theta':(4,9),'eta':(4,)},'B3':{'a':(4,),'b':(4,)},'C3':{'theta':(4,)},'H1':{'theta':(4,)},'H2':{'u':(4,),'v':(4,)},'H3':{'u':(4,),'v':(4,)}}
  for name,shape in shapes.get(arm,{}).items():
   value=math.log((3.5/7.5)/(1-3.5/7.5)) if arm=='C3' else 0.;self.extra[name]=nn.Parameter(torch.full(shape,value,dtype=torch.float32,device=device),requires_grad=not frozen)
  self.mid=list(self.base.chronos_config.quantiles).index(.5);assert self.base.chronos_config.context_length>=1344
 def T(self,x):return torch.as_tensor(x,dtype=torch.float32,device=next(self.base.parameters()).device)
 def tensors(self,p):
  x=self.T(p['x']);mu=self.T(p['mu'])[:,None];sigma=self.T(p['sigma'])[:,None];future=None;arm=self.arm
  if self.t=='N01':
   mask=torch.as_tensor(p['M'],device=x.device);r=self.T(p['ridge']);a=self.T(p['age']/24).clamp(max=2)
   if arm=='A0':x=torch.where(mask,x,torch.full_like(x,float('nan')))
   else:
    fill=r
    if arm in ['A2','A3']:fill=fill+torch.einsum('ctf,cf->ct',self.T(p['phi']),self.extra['theta'])
    if arm=='A3':fill=fill+self.extra['eta'][:,None]*a*(r-self.T(p['hold']))
    x=torch.where(mask,x,fill*sigma+mu)
   x=torch.cat([x,mask.float(),a],0)
  if self.t=='N03':
   mask=torch.as_tensor(p['M'],device=x.device)
   if arm=='C0':x=torch.where(mask,x,torch.full_like(x,float('nan')))
   elif arm=='C1':x=torch.where(mask,x,self.T(p['hold']))
   else:
    tau=(.5+7.5*torch.sigmoid(self.extra['theta']))[:,None,None] if arm=='C3' else 4.
    w=torch.softmax(-self.T(p['distances'])/tau,dim=-1);idx=torch.as_tensor(p['neighbors'],device=x.device);vals=x[torch.arange(4,device=x.device)[:,None,None],idx];x=torch.where(mask,x,(w*vals).sum(-1))
   x=torch.cat([x,mask.float(),self.T(p['age']/4)],0)
  if self.t=='N02':
   if arm!='B1':x=x[:,-336:]
   if arm in ['B2','B3']:
    past=self.T(p['retrieval_past']);f=self.T(p['retrieval_future']);
    if arm=='B3':
     qs=x.std(-1,correction=0).clamp(min=1e-6).repeat_interleave(2)[:,None];last=past[:,-1:];f=f+(self.extra['a'].clamp(-2,2).exp()-1).repeat_interleave(2)[:,None]*(f-last)+self.extra['b'].repeat_interleave(2)[:,None]*qs
    x=torch.cat([x,past],0);future=torch.cat([torch.full((4,48),float('nan'),device=x.device),f],0)
  if self.t=='R09':x=torch.cat([x,self.T(p['resolution'])[None]],0)
  return dict(context=x,group_ids=torch.zeros(len(x),dtype=torch.long,device=x.device),future_covariates=future,num_output_patches=math.ceil(p['H']/16))
 def forward(self,p,apply_output=True):
  v=self.base(**self.tensors(p));q=v.quantile_preds[:4,self.mid,:p['H']]
  if not torch.isfinite(q).all():raise FloatingPointError('NONFINITE_PREDICTION')
  if self.t=='R08' and self.arm!='H0' and apply_output:
   mu=self.T(p['mu'])[:,None];sigma=self.T(p['sigma'])[:,None];z=(q-mu)/sigma;g,a=self.lead(z,p);h=torch.arange(48,device=q.device)/47
   if self.arm=='H1':factor=self.extra['theta'].tanh()[:,None]
   elif self.arm=='H2':factor=(1-h)*self.extra['u'].tanh()[:,None]+h*self.extra['v'].tanh()[:,None]
   else:factor=a*self.extra['u'].tanh()[:,None]+(1-a)*self.extra['v'].tanh()[:,None]
   q=(z+factor*(g-z))*sigma+mu
  return q
 def lead(self,z,p):
  ctx=(self.T(p['x'])-self.T(p['mu'])[:,None])/self.T(p['sigma'])[:,None];gg=[];aa=[]
  for c in range(4):
   coef=self.aux['ridge'][c];g=torch.full((48,),coef[0],device=z.device);avail=torch.zeros_like(g);den=max(sum(abs(b) for b in coef[1:]),1e-8)
   for j,l,b in zip(self.aux['donors'][c],self.aux['lags'][c],coef[1:]):
    # All corrections read the same uncorrected z, never an already-corrected donor.
    s=torch.cat([ctx[j,-l:],z[j,:48-l]],0);assert len(s)==48;g=g+b*s;avail=avail+abs(b)/den*(torch.arange(48,device=z.device)<l)
   gg.append(g);aa.append(avail)
  return torch.stack(gg),torch.stack(aa)

def task_loss(m,p,y,q0=None,late_packet=None,late_q0=None,fixed=None):
 q=m(p);yy=m.T(y);s=m.T(p['sigma'])[:,None];metrics={}
 if m.t=='N01':c=p['target'];loss=(((q[c]-yy[c])/s[c])**2).mean()
 elif m.t=='R04':
  qlate=m(late_packet);early_y=yy[:,:48];late_y=yy[:,24:72];task=.5*(((q-early_y)/s)**2).mean()+.5*(((qlate-late_y)/s)**2).mean();raw=(qlate[:,:24]-q[:,24:])/s;corr=((qlate-late_q0)[:,:24]-(q-q0)[:,24:])/s;reg=raw.square().mean() if m.arm=='D1' else corr.square().mean()
  if m.arm=='D3':
   v=(((m.T(p['innovation_observed'])-q0[:,:24])/s)**2).mean(-1);w=(1/(1+v))/fixed['mean_w'];reg=(w[:,None]*corr.square()).mean()
  loss=task+(fixed['lambda']*reg if m.arm!='D0' else 0);metrics=dict(task=float(task.detach()),penalty=float(reg.detach()),penalty_coefficient=0 if m.arm=='D0' else fixed['lambda'])
 elif m.t=='N07':
  e=(q-yy)/s;w=m.T(m.aux['weights'][m.arm]);loss=.5*e.square().mean()+.5*(w*torch.fft.fft(e,norm='ortho').abs().square()).mean()
 elif m.t=='R09':
  delta=(q-q0)/s;agg=((q.mean(-1)-yy.mean(-1))/s[:,0]).square().mean();n=delta-delta.mean(-1,keepdim=True);fine=((q-yy)/s).square().mean();reg=torch.zeros((),device=q.device)
  if p['detailed'] or m.arm=='I0':loss=fine
  elif m.arm=='I2':pseudo=q0+(yy.mean(-1,keepdim=True)-q0.mean(-1,keepdim=True));loss=((q-pseudo)/s).square().mean()
  else:
   if m.arm=='I3':reg=delta.square().mean()
   if m.arm=='I4':reg=n.square().mean()
   loss=agg+.1*reg
  metrics=dict(observation=float((fine if p['detailed'] else agg).detach()),penalty=float(reg.detach()))
 else:loss=(((q-yy)/s)**2).mean()
 if not torch.isfinite(loss):raise FloatingPointError('NONFINITE_LOSS')
 return loss,metrics
