"""Independent scalar V metrics and checkpoint/LR policy replay; no training or E selection."""
import sys,json,math,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experiments.condition_sampling_repair_v1_20260917.common import *

def rms(a):return math.sqrt(math.fsum(float(v)**2 for v in np.ravel(a))/np.size(a))
def scalar(t,p,y,sigma,conditions):
 if t=='R04':
  yy=np.stack([y[:,:,:48],y[:,:,24:72]],1);d=(p[:,0]-yy)/sigma[None,None,:,None];rev=(p[:,0,1,:,:24]-p[:,0,0,:,24:])/sigma[None,:,None];return dict(accuracy=math.fsum(rms(d[:,:,c]) for c in range(4))/4,revision=math.fsum(rms(rev[:,c]) for c in range(4))/4)
 scores=[]
 for j,condition in enumerate(conditions):
  if t=='N01':
   if not condition.startswith(('d6_','d24_')):continue
   c=int(condition.split('_c')[1]);scores.append(rms((p[:,j,c]-y[:,c])/sigma[c]))
  else:scores.append(math.fsum(rms((p[:,j,c]-y[:,c])/sigma[c]) for c in range(4))/4)
 return dict(accuracy=math.fsum(scores)/len(scores))

def run():
 checks=[]
 for t in SOURCES:
  if not (OUT/t/'selections.json').exists() or not (OUT/t/'fits.json').exists():continue
  if not all(f['status']=='COMPLETE' for f in read(OUT/t/'fits.json')):continue
  fits=read(OUT/t/'fits.json');sels=read(OUT/t/'selections.json')['selections'];lrsel=read(OUT/t/'LR_selection.json');sigma=np.array(read(OUT/t/'train_statistics.json')['sigma']);y=np.load(CACHE/t/'V_SELECT_labels.npz')['y'];n=0;cache={}
  for f in fits:
   for cp in f['checkpoints']:
    p=ROOT/cp['prediction'];meta=read(p.with_suffix('.json'));assert sha(p)==meta['sha256']
    if str(p) not in cache:
     z=np.load(p);cache[str(p)]=scalar(t,z['pred'],y,sigma,list(z['conditions']))
    values=cache[str(p)]
    for k,v in values.items():assert math.isclose(v,cp[k],rel_tol=1e-10,abs_tol=1e-10),(t,f['id'],cp['step'],k,v,cp[k]);n+=1
  replay=[]
  for seed in SEEDS:
   reference=None
   for arm in ARMS[t]:
    ff=[f for f in fits if f['seed']==seed and f['arm']==arm and f['status']=='COMPLETE']
    if not ff:continue
    candidates=[(f,cp) for f in ff for cp in f['checkpoints']]
    if t=='R04' and arm!='D0':
     eligible=[(f,cp) for f,cp in candidates if cp['accuracy']<=1.01*reference]
     if eligible:f,cp=min(eligible,key=lambda v:(v[1]['revision'],v[0]['lr'],v[1]['step']))
     else:f,cp=min(candidates,key=lambda v:(v[1]['accuracy'],v[0]['lr'],v[1]['step']))
    else:f,cp=min(candidates,key=lambda v:(v[1]['accuracy'],v[0]['lr'],v[1]['step']))
    expected=next(v for v in (lrsel['selections'] if seed==73100 else sels) if v['arm']==arm and v['seed']==seed);assert f['id']==expected['id'] and cp['step']==expected['selected']['step']
    if arm=='D0':reference=cp['accuracy']
    if seed!=73100:assert f['lr']==lrsel['learning_rates'][arm]
    replay.append(dict(arm=arm,seed=seed,lr=f['lr'],step=cp['step'],matches=True))
  record=dict(passed=True,scalar_values=n,selected_models=len(replay),replay=replay,rtol=1e-10,atol=1e-10,TRAIN_or_V_only=True,no_new_optimization=True);save(OUT/t/'independent_selection_verification.json',record);checks.append(dict(track=t,scalar_values=n,models=len(replay)))
 save(OUT/'independent_selection_verification.json',dict(at=time.time(),completed_tracks_only=True,checks=checks));print('SELECTION_AUDIT',checks)
if __name__=='__main__':run()
