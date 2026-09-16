"""Required task/regularizer gradient decomposition, zero optimizer steps."""
import sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experiments.condition_studies_v1_20260916.common import *
from experiments.condition_studies_v1_20260916.model import *
from experiments.condition_studies_v1_20260916.runtime import Controller,Guard

def norm(gs):return float(torch.sqrt(sum(g.double().square().sum() for g in gs)))
def run():
 assert read(OUT/'controller_state.json')['status']=='FINISHED','Main queue must finish first'
 if (OUT/'gradient_contribution_verification.json').exists():print('ALREADY_VERIFIED_NO_DUPLICATE');return
 schedule=read(OUT/'verification_schedule.json')['gradient_verification'];configure();ctrl=Controller(resume=True);ctrl.guard=Guard(OUT,'gradient_verify',wall_cap=86400);ctrl.phase='verify';start=ctrl.state['forwards']['verify'];rows=[]
 try:
  ctrl.guard.boundary(startup=True)
  for t in schedule['tracks']:
   if read(OUT/t/'STATUS.json')['EXECUTION']!='COMPLETE':continue
   data=Packets(t);frozen=dict(np.load(CACHE/t/'frozen.npz'));fixed=read(OUT/t/'frozen_parameters.json');sels=read(OUT/t/'selections.json')['selections'];cases=list(range(4)) if t=='R04' else np.flatnonzero(~data.inputs['TRAIN']['detailed'])[:4].tolist()
   for sel in sels:
    if sel['arm'] not in schedule['arms'][t]:continue
    cp=sel['selected'];assert sha(ROOT/cp['path'])==cp['sha256'];m=ctrl.make(t,sel['arm'],sel['seed'],data.aux);restore(m,torch.load(ROOT/cp['path'],map_location='cpu',weights_only=True));before=tensor_hash(cpu_state(m));fb=frozen_hash(m);ps=list(parameters(m).values())
    for i in cases:
     ctrl.guard.boundary();p=data.packet('TRAIN',i,epoch=0);q=m(p);s=m.T(data.sigma)[:,None];y=m.T(data.labels['TRAIN'][i]);q0=m.T(frozen['TRAIN'][i])
     if t=='R04':
      late=m(data.packet('TRAIN',i,epoch=0,late=True));late0=m.T(frozen['TRAIN_late'][i]);task=.5*((q-y[:,:48])/s).square().mean()+.5*((late-y[:,24:72])/s).square().mean();raw=(late[:,:24]-q[:,24:])/s;corr=((late-late0)[:,:24]-(q-q0)[:,24:])/s;reg=raw.square().mean() if sel['arm']=='D1' else corr.square().mean()
      if sel['arm']=='D3':v=((m.T(data.inputs['TRAIN']['innovation_observed'][i])-q0[:,:24])/s).square().mean(-1);reg=((1/(1+v)/fixed['mean_w'])[:,None]*corr.square()).mean()
      coefficient=fixed['lambda']
     else:
      task=((q.mean(-1)-y.mean(-1))/s[:,0]).square().mean();delta=(q-q0)/s;null=delta-delta.mean(-1,keepdim=True);reg=delta.square().mean() if sel['arm']=='I3' else null.square().mean();coefficient=.1
     ga=torch.autograd.grad(task,ps,retain_graph=True);gb=torch.autograd.grad(coefficient*reg,ps);na=norm(ga);nb=norm(gb);dot=float(sum((a.double()*b.double()).sum() for a,b in zip(ga,gb)));total=norm([a+b for a,b in zip(ga,gb)]);rows.append(dict(track=t,arm=sel['arm'],seed=sel['seed'],checkpoint_step=cp['step'],origin_ordinal=i,origin=int(data.inputs['TRAIN']['origins'][i]),task_loss=float(task),raw_penalty=float(reg),coefficient=coefficient,task_parameter_gradient_norm=na,weighted_penalty_parameter_gradient_norm=nb,weighted_penalty_over_task=nb/na if na else None,gradient_cosine=dot/(na*nb) if na and nb else None,total_gradient_norm=total,would_clip=total>1,optimizer_updates=0));del q,task,reg,ga,gb
     ctrl.persist()
    assert tensor_hash(cpu_state(m))==before and frozen_hash(m)==fb;del m;cleanup()
  calls=ctrl.state['forwards']['verify']-start;assert calls<=schedule['native_forward_cap'];csvwrite(OUT/'gradient_contributions.csv',rows)
  for t in schedule['tracks']:csvwrite(OUT/t/'gradient_contributions.csv',[r for r in rows if r['track']==t])
  save(OUT/'gradient_contribution_verification.json',dict(passed=True,at=time.time(),native_forwards=calls,cases=len(rows),optimizer_updates=0,frozen_and_trainable_weights_unchanged=True,scope='Selected checkpoints at predeclared first4 TRAIN cases; diagnostic gradient probes, not gradients at every historical optimizer update'))
 finally:ctrl.guard.close();ctrl.persist()
 print('GRADIENT_AUDIT',len(rows),calls)
if __name__=='__main__':run()
