"""CPU-only regression: crash after resume512 save, before V512 publication.
No model inference or optimizer step is performed. Production loop is exercised.
"""
import sys,tempfile,json
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from experiments.forecast_path_structure_v1_20260916 import runtime as r
class Dummy(torch.nn.Module):
 def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
class Opt:
 def load_state_dict(self,s):pass
class Harness:
 def __init__(self):
  self.targets=[{'target':'T0'}];self.data={'planned_fits':16};self.jobs=[];self.state={'phase':'TRAIN','completed_fits':15};self.fits=[];self.checkpoint_calls=[]
  for seed in r.SEEDS:
   for lr in r.LRS:
    for arm in r.ARMS:
     fid=f'T0_{arm}_{seed}_{lr:.0e}';self.fits.append(dict(id=fid,target='T0',arm=arm,seed=seed,lr=lr,status='COMPLETE',updates=512,checkpoints=[{'step':0},{'step':256}],optimizer_seconds=1.))
  self.fits[0]['status']='RUNNING'
 def persist(self):pass
 def counted(self,m):return m
 def optimizer(self,m,lr):return Opt()
 def checkpoint(self,m,f,step):
  assert step==512 and torch.equal(m.weight.detach(),torch.ones(1));self.checkpoint_calls.append(step);f['checkpoints'].append({'step':step})
def main():
 h=Harness();fid=h.fits[0]['id']
 with tempfile.TemporaryDirectory(prefix='forecast-resume-test-') as folder,ExitStack() as st:
  p=Path(folder);(p/'resume').mkdir();(p/'resume'/f'{fid}.pt').touch();(p/'update_log.jsonl').write_text(''.join(json.dumps(dict(id=fid,step=i))+'\n' for i in range(1,513)))
  resume=dict(step=512,parameters={'weight':torch.ones(1)},optimizer={},rng={})
  for name,value in [('CACHE',p),('OUT',p),('make',lambda seed:Dummy()),('load_rng',lambda x:None),('cleanup',lambda:None)]:st.enter_context(patch.object(r,name,value))
  st.enter_context(patch.object(torch,'load',lambda *a,**k:resume))
  st.enter_context(patch.object(torch.cuda,'reset_peak_memory_stats',lambda:None));st.enter_context(patch.object(torch.cuda,'max_memory_allocated',lambda:0));st.enter_context(patch.object(torch.cuda,'max_memory_reserved',lambda:0))
  r.Controller.train(h)
 assert h.checkpoint_calls==[512] and h.state['completed_fits']==16 and h.fits[0]['status']=='COMPLETE'
 print('PASS: missing V512 checkpoint finalized after exact resume; optimizer calls=0; GPU calls=0')
if __name__=='__main__':main()
