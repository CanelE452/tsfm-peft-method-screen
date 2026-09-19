"""Isolated protocol; historical experiments are read-only dependencies."""
from experiments.additive_persistence_validation_v1_20260917.common import *
from experiments.additive_persistence_validation_v1_20260917 import common as old
from experiments.additive_persistence_validation_v1_20260917.checks import arrays,batch
from experiments.additive_persistence_validation_v1_20260917.train import validation,predict
from experiments.additive_persistence_validation_v1_20260917.model import loss_2pinball
from .core import ARMS,probes,penalty
NAME='temporal_response_peft_20260919';EXP=ROOT/'experiments'/NAME;OUT=ROOT/'results'/NAME;CACHE=ROOT/'.cache'/NAME
SOURCES=['electricity','ettm1'];SEEDS=[81551,81552];NEW_ARMS=[a for a in ARMS if a!='PLAIN'];MAIN_CAP=16384;SMOKE_CAP=16
LR=3e-4;MICRO=32
class Watch(old.Watch):
    def __init__(self):
        self.last_disk=0;OriginalWatch.__init__(self,OUT,'trp',wall_cap=86400)
    def limits(self):
        import shutil
        OriginalWatch.limits(self)
        if shutil.disk_usage(ROOT).free<10*2**30:raise ResourceError('DISK_BELOW_10GIB')
        if time.monotonic()-self.last_disk>60:
            if sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())>30*2**30:raise ResourceError('CACHE_CAP_30GIB')
            self.last_disk=time.monotonic()
def build(arm,seed,source,device='cuda'):
    assert arm in ARMS+['B0','C3','MAG_ONLY']
    if arm=='MAG_ONLY':
        from experiments.c3_training_factorial_20260918.common import build as fb
        return fb(arm,[seed,seed,seed],source,device)
    return old.build('C0' if arm=='B0' else 'C3' if arm=='C3' else 'C2',seed,source,device=device)
def optimizer_for(model,lr=LR):return torch.optim.AdamW(parameters(model).values(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
def fit_id(source,arm,seed):return f'{source}_{arm}_s{seed}_lr{LR:g}'
def check_seal():
    d=read(OUT/'SEAL.json')
    for p,h in d['hashes'].items():assert sha(ROOT/p)==h,('SEALED_CHANGED',p)
    return d
def status(**kw):
    receipts=[read(p) for p in (OUT/'fits').glob('*/receipt.json')]
    ledger=OUT/'UPDATE_LEDGER.jsonl'
    save(OUT/'status.json',dict(at=time.time(),new_fits=sum(not r.get('reused') for r in receipts),reused_fits=sum(bool(r.get('reused')) for r in receipts),main_updates=sum(1 for _ in open(ledger)) if ledger.exists() else 0,main_cap=MAIN_CAP,smoke_updates=sum(sum(1 for _ in open(p)) for p in OUT.glob('smoke_*.jsonl')),automatic_successor=False,**kw))
def backward(model,x,y,s,arm,source,epoch,idx):
    u,v,a,_=probes(source,epoch,idx,s.detach().cpu().numpy(),device=x.device)
    t=x+(v if arm=='SHUFFLE' else u)
    with torch.no_grad():b=model(x,s,residual_mode='off');bt=model(t,s,residual_mode='off')
    p=model(x,s);pt=model(t,s)
    sup=loss_2pinball(p,y,s,model.base.quantiles);reg=penalty(arm,p,pt,b,bt,s,a)
    loss=sup+reg;assert torch.isfinite(loss),'NONFINITE_LOSS';loss.backward()
    gs=[p.grad for p in parameters(model).values()];assert all(g is not None and torch.isfinite(g).all() for g in gs),'NONFINITE_GRADIENT'
    return dict(loss=float(loss.detach()),supervised=float(sup.detach()),regularizer=float(reg.detach()),labelled_examples=len(x),adapted_forwards=2,teacher_forwards=2)
