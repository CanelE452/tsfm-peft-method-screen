"""One bounded extension: three explanation controls, no C3 modification."""
from experiments.additive_persistence_validation_v1_20260917.common import *
from experiments.additive_persistence_validation_v1_20260917 import common as old
NAME='c3_weakness_controls_20260918';OUT=ROOT/'results'/NAME;CACHE=ROOT/'.cache'/NAME;EXP=ROOT/'experiments'/NAME
PRIOR=old.OUT;SOURCES=['electricity','ettm1'];ARMS=['POS_ONLY','MAG_ONLY','OUTPUT_CONTEXT'];CONTROLS=ARMS
class Watch(old.Watch):
    def __init__(self):
        self.last_disk=0;OriginalWatch.__init__(self,OUT,'run',wall_cap=14400)
def check_seal():
    d=read(OUT/'SEAL.json')
    for p,h in d['hashes'].items():assert sha(ROOT/p)==h,('SEALED_CHANGED',p)
    return d
def build(arm,seed,source,device='cuda',**unused):
    if arm=='C0':return old.build('C0',seed,source,device)
    from .model import ControlModel
    baseline=old.build('C0',seed,source,device='cpu');baseline.requires_grad_(False)
    m=ControlModel(baseline.base,arm,seed).eval().to(device)
    assert sum(p.numel() for p in parameters(m).values())=={'POS_ONLY':8744,'MAG_ONLY':8712,'OUTPUT_CONTEXT':4673}[arm]
    return m
def optimizer_for(model,lr):
    groups=[dict(params=[p for n,p in parameters(model).items() if n!='position_logits'],lr=lr)]
    if model.arm=='POS_ONLY':groups.append(dict(params=[model.position_logits],lr=.01))
    return torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=0)
def status(**extra):
    fits=[read(p) for p in (OUT/'fits').glob('*/receipt.json')];n=sum(1 for _ in open(OUT/'UPDATE_LEDGER.jsonl')) if (OUT/'UPDATE_LEDGER.jsonl').exists() else 0
    save(OUT/'status.json',dict(execution='RUNNING',completed_fits=len(fits),main_updates=n,smoke_updates=sum(sum(1 for _ in open(p)) for p in OUT.glob('smoke_*.jsonl')),**extra))
