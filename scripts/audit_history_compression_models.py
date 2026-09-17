"""After completion: fresh-model E replay, old teacher replay and pooling behavior.
Zero optimizer updates; bounded extra forward calls counted separately.
"""
import os,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('HF_HUB_OFFLINE','1')
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.history_compression_v1_20260917.common import *
from experiments.history_compression_v1_20260917.model import Model,configure
assert read(OUT/'state.json')['status']=='FINISHED';validate_seal();configure();d=Data()
path=OUT/'independent_model_verification.json'
if path.exists():
 assert read(path)['passed'];print('Independent model audit already complete');raise SystemExit(0)
ss=read(OUT/'selections.json')['selections'];reuse=read(OUT/'REUSE_RECEIPT.json');ss+=[dict(s,arm={'B0':'SHORT','B1':'LONG'}[s['arm']]) for s in reuse['selections']]
man=read(OUT/'predictions_manifest.json')+[dict(r,arm={'B0':'SHORT','B1':'LONG'}[r['arm']]) for r in reuse['predictions']]
checks=[];weights=[];calls=0;guard=Watch(OUT,'independent_replay',wall_cap=14400)
try:
 guard.boundary(startup=True)
 for s in ss:
  guard.boundary();m=Model(s['arm'],s['seed']);cp=s['selected'];assert sha(ROOT/cp['path'])==cp['sha256'];restore(m,torch.load(ROOT/cp['path'],map_location='cpu',weights_only=True))
  record=next(r for r in man if r['arm']==s['arm'] and r['seed']==s['seed'] and r['policy']=='selected');arr=np.load(ROOT/record['path'])['pred'];arr=arr[:,0] if arr.ndim==4 else arr
  fh=frozen_hash(m)
  for i in [0,63]:
   guard.boundary()
   with torch.no_grad():q=m(d.packet('E_DISCOVERY',i)).double().cpu().numpy()
   calls+=1;assert np.array_equal(q,arr[i]),(s['arm'],s['seed'],i)
   checks.append(dict(arm=s['arm'],seed=s['seed'],role='E_DISCOVERY',origin_ordinal=i,bitwise=True))
  if s['arm'].startswith('LEARN'):
   # Inspect full V, not E, for the actual learned pooling weights. No intervention/selection.
   ww=[]
   for i in range(32):
    guard.boundary()
    with torch.no_grad():m(d.packet('V_SELECT',i));ww.append(m.last_weights.cpu().numpy())
    calls+=1
   w=np.array(ww);assert np.allclose(w.sum(-1),1,atol=1e-6) and (w>=0).all()
   weights.append(dict(arm=s['arm'],seed=s['seed'],V_origins=32,mean_abs_deviation_from_uniform=float(abs(w-1/3).mean()),maximum_abs_deviation=float(abs(w-1/3).max()),mean_entropy=float(-(w*np.log(w)).sum(-1).mean()),uniform_entropy=float(np.log(3))))
  assert frozen_hash(m)==fh;del m;cleanup()
 teacher=reuse['teacher'];m=Model('LONG',teacher['seed']);restore(m,torch.load(ROOT/teacher['selected']['path'],map_location='cpu',weights_only=True));p=np.load(ROOT/teacher['selected']['prediction'])['pred'];p=p[:,0];train=np.load(ROOT/read(OUT/'teacher.json')['path'])['pred']
 for role,arr in [('V_SELECT',p),('TRAIN',train)]:
  for i in [0,1]:
   guard.boundary()
   with torch.no_grad():q=m(d.packet(role,i)).double().cpu().numpy()
   calls+=1;assert np.array_equal(q,arr[i]);checks.append(dict(arm='FIXED_TEACHER',seed=73100,role=role,origin_ordinal=i,bitwise=True))
 del m;cleanup()
 assert read(OUT/'state.json')['forwards']+calls<=40000
 save(path,dict(passed=True,checks=checks,pooling_behavior=weights,extra_forward_calls=calls,total_including_controller=read(OUT/'state.json')['forwards']+calls,extra_optimizer_updates=0,all_restore_bitwise=True,old_teacher_prediction_exact=True,at=time.time()))
 csvwrite(OUT/'pooling_behavior.csv',weights)
finally:guard.close()
print('Fresh model replay complete',calls,'forwards; zero updates')
