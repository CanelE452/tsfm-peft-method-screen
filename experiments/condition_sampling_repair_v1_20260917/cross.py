"""Zero-update transport of selected old/new weights, after the global selection seal."""
import math,time
import pandas as pd
import torch
from .common import *
from .model import *
from .runtime import primary
from experiments.condition_studies_v1_20260916.model import Packets as OldPackets

def scalar_score(p,y,s,conds,t):
 vals=[]
 for j,c in enumerate(conds):
  if t=='N03' and c not in ['delta2','irregular']:continue
  ch=[]
  for k in range(4):
   d=[(float(p[i,j,k,h])-float(y[i,k,h]))/s[k] for i in range(len(y)) for h in range(48)];ch.append(math.sqrt(math.fsum(v*v for v in d)/len(d)))
  vals.append(math.fsum(ch)/4)
 return math.fsum(vals)/len(vals)

def summarize(t,direction,bundle,y,origins,sigma,conds):
 counts=block_counts(origins,672 if t=='N03' else 168);den=counts.sum(1);valid=den>0;den=np.where(valid,den,np.nan);series={};raw=[]
 for (a,seed),p in bundle.items():
  cp=[];cb=[]
  for j,c in enumerate(conds):
   if t=='N03' and c not in ['delta2','irregular']:continue
   per=(((p[:,j]-y)/sigma[None,:,None])**2).mean(-1);cp.append(float(np.sqrt(per.mean(0)).mean()));cb.append(np.sqrt(counts@per/den[:,None]).mean(1))
  score=float(np.mean(cp));sc=scalar_score(p,y,sigma,conds,t);assert math.isclose(score,sc,rel_tol=1e-10,abs_tol=1e-10);series[(a,seed)]=(score,np.mean(cb,0));raw.append(dict(direction=direction,arm=a,seed=seed,normalized_RMSE=score,scalar_verified=True))
 contrasts=[]
 for base in CONTRASTS[t]:
  ss=[]
  for seed in [73101,73102]:
   if (PROPOSED[t],seed) not in series or (base,seed) not in series:continue
   a,ab=series[(PROPOSED[t],seed)];b,bb=series[(base,seed)];bs=100*(bb-ab)/bb;ss.append((a,b,ab,bb));contrasts.append(dict(direction=direction,method=PROPOSED[t],baseline=base,seed=str(seed),method_score=a,baseline_score=b,gain_percent=100*(b-a)/b,ci_low=float(np.nanquantile(bs,.025)),ci_high=float(np.nanquantile(bs,.975))))
  if len(ss)==2:
   a,b=np.mean([(x[0],x[1]) for x in ss],0);ab=np.mean([x[2] for x in ss],0);bb=np.mean([x[3] for x in ss],0);bs=100*(bb-ab)/bb;contrasts.append(dict(direction=direction,method=PROPOSED[t],baseline=base,seed='MEAN',method_score=float(a),baseline_score=float(b),gain_percent=float(100*(b-a)/b),ci_low=float(np.nanquantile(bs,.025)),ci_high=float(np.nanquantile(bs,.975))))
 return raw,contrasts,dict(defined=int(valid.sum()),empty=int((~valid).sum()),represented_blocks=len(np.unique(origins//(672 if t=='N03' else 168))))

def run_cross(ctrl,t,newdata):
 out=OUT/t
 if (out/'cross_verification.json').exists():assert read(out/'cross_verification.json')['passed'];return
 # Preflight makes only N02/N03 eligible in this contract; no GPU work for diversity-blocked tracks.
 assert t in ['N02','N03'],'Unsupported eligible cross track; must be implemented before any result-dependent choices'
 assert (OUT/'EVALUATION_SEAL.json').exists();olddata=OldPackets(t);records=[];allraw=[];allcontrasts=[];blocks={};start=time.time();before_updates=ctrl.state['main_updates'];plan=[]
 for direction,selected,source in [('OLD_MODEL_NEW_E',read(OLD/t/'selections.json')['selections'],'old'),('NEW_MODEL_OLD_E',read(out/'selections.json')['selections'],'new')]:
  for s in selected:
   cp=s['selected'];path=ROOT/cp['path'];valid=path.exists() and sha(path)==cp['sha256'];plan.append(dict(direction=direction,arm=s['arm'],seed=s['seed'],checkpoint=cp['path'],checkpoint_sha256=cp['sha256'],status='READY' if valid else 'BLOCKED_OLD_CHECKPOINT'))
 save(out/'cross_evaluation_seal.json',dict(at=time.time(),plan=plan,selection_fixed=True,optimizer_updates=0))
 for direction,data,weightout,selection in [('OLD_MODEL_NEW_E',newdata,OLD,read(OLD/t/'selections.json')['selections']),('NEW_MODEL_OLD_E',olddata,OUT,read(out/'selections.json')['selections'])]:
  local=[]
  for s in selection:
   cp=s['selected'];path=ROOT/cp['path']
   if not path.exists() or sha(path)!=cp['sha256']:records.append(dict(direction=direction,arm=s['arm'],seed=s['seed'],status='BLOCKED_OLD_CHECKPOINT'));continue
   m=ctrl.make(t,s['arm'],s['seed'],data.aux);restore(m,torch.load(path,map_location='cpu',weights_only=True));fh=frozen_hash(m);h=tensor_hash(cpu_state(m));file=ctrl.predict(m,data,'E_DISCOVERY',direction+'_'+s['arm']+'_'+cp['parameter_hash']);assert frozen_hash(m)==fh and tensor_hash(cpu_state(m))==h;local.append(dict(direction=direction,arm=s['arm'],seed=s['seed'],checkpoint=cp['path'],checkpoint_sha256=cp['sha256'],path=str(file.relative_to(ROOT)),sha256=sha(file),status='COMPLETE'));del m;cleanup()
  records.extend(local)
  save(out/'cross_evaluation_manifest.json',dict(at=time.time(),records=records,optimizer_updates=0,current_direction_predictions_complete=True,current_direction_labels_opened=False))
  # Only now read that direction's labels. TRAIN/V/CPU selection is already sealed.
  labelroot=CACHE if direction=='OLD_MODEL_NEW_E' else OLDCACHE;y=np.load(labelroot/t/'E_DISCOVERY_labels.npz')['y'];bundle={};conds=None
  for r in local:
   z=np.load(ROOT/r['path']);assert sha(ROOT/r['path'])==r['sha256'];assert np.array_equal(z['origins'],data.inputs['E_DISCOVERY']['origins']);bundle[(r['arm'],r['seed'])]=z['pred'];conds=list(z['conditions'])
  if t=='N02':
   for r in read(weightout/t/'CPU_selection.json')['choices']:
    key=('B0',r['seed'])
    if key in bundle:
     simple=data.inputs['E_DISCOVERY']['retrieval_future'].reshape(64,4,2,48).mean(2)[:,None];bundle[('BLEND',r['seed'])]=(1-r['alpha'])*bundle[key]+r['alpha']*simple
  if bundle:
   raw,con,bs=summarize(t,direction,bundle,y,data.inputs['E_DISCOVERY']['origins'],data.sigma,conds);allraw.extend(raw);allcontrasts.extend(con);blocks[direction]=bs
 assert ctrl.state['main_updates']==before_updates
 csvwrite(out/'cross_raw_scores.csv',allraw);csvwrite(out/'cross_contrasts.csv',allcontrasts);save(out/'cross_verification.json',dict(passed=True,at=time.time(),optimizer_updates=0,scalar_metric_checks=len(allraw),rtol=1e-10,atol=1e-10,weights_unchanged=True,bootstrap=blocks,seconds=time.time()-start,scope='all available old-selected/new-E and new-selected/old-E, target labels read only after corresponding prediction manifest, never retrain missing old weights'))
