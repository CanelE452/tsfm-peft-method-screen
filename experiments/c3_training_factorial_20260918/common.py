"""Independent B0/init/order interventions; parent algorithms remain unchanged."""
from experiments.additive_persistence_validation_v1_20260917.common import *
from experiments.additive_persistence_validation_v1_20260917 import common as old
from experiments.c3_weakness_controls_20260918.model import ControlModel
from experiments.additive_persistence_validation_v1_20260917.model import ForecastModel
NAME='c3_training_factorial_20260918';OUT=ROOT/'results'/NAME;CACHE=ROOT/'.cache'/NAME;EXP=ROOT/'experiments'/NAME
SOURCES=['electricity','ettm1'];ARMS=['C3','MAG_ONLY'];LEVELS=[81551,81552];MAIN_CAP=24576
class Watch(old.Watch):
 def __init__(self):
  self.last_disk=0;OriginalWatch.__init__(self,OUT,'factorial',wall_cap=14400)
def baseline_row(source,seed):return old.baseline_row(source,int(seed[0]))
def build(arm,seed,source,device='cuda'):
 b,i,o=map(int,seed);base=old.build('C0',b,source,device='cpu');base.requires_grad_(False)
 if arm=='C0':return base.to(device)
 model=ForecastModel(base.base,'C3',i) if arm=='C3' else ControlModel(base.base,'MAG_ONLY',i)
 model=model.eval().to(device);assert sum(p.numel() for p in parameters(model).values())==8712
 assert not any(isinstance(m,torch.nn.Dropout) and m.p for m in model.modules())
 return model
def optimizer_for(model,lr):return torch.optim.AdamW(parameters(model).values(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
def check_seal():
 d=read(OUT/'SEAL.json')
 for p,h in d['hashes'].items():assert sha(ROOT/p)==h,('SEALED_CHANGED',p)
 return d
def status(**extra):
 r=[read(p) for p in (OUT/'fits').glob('*/receipt.json')];journal=OUT/'UPDATE_LEDGER.jsonl';updates=sum(1 for _ in open(journal)) if journal.exists() else 0
 save(OUT/'status.json',dict(execution='RUNNING',completed_paths=len(r),new_fits=sum(not x.get('reused',False) for x in r),reused_fits=sum(x.get('reused',False) for x in r),main_updates=updates,smoke_updates=sum(sum(1 for _ in open(p)) for p in OUT.glob('smoke_*.jsonl')),automatic_successor=False,**extra))
