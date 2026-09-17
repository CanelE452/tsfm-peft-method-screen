"""Only generic GPU guard, hash, journal and state helpers are reused."""
import os, sys, time, json, gc, random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from priority12.common import save,read,sha,csvwrite,parameters,cpu_state,restore,frozen_hash,tensor_hash,ResourceError
from priority12.common import Watch as OriginalWatch
NAME='outlier_signal_peft_v1_20260917'
OUT=ROOT/'results'/NAME
CACHE=ROOT/'.cache'/NAME
EXP=ROOT/'experiments'/NAME
ARMS=['A0','A1','A2','A3','A4','A5']
SOURCES=['electricity','ettm1']
def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    random.seed(81500);np.random.seed(81500);torch.manual_seed(81500)

class Watch(OriginalWatch):
    def __init__(self,phase):
        self.last_disk=0
        super().__init__(OUT,phase,wall_cap=86400)
    def sample(self):
        r=super().sample()
        for a in r['apps']: a['allowed_desktop']=a['name']=='/usr/share/rustdesk/rustdesk'
        r['busy']=any(not a['own'] and not a['allowed_desktop'] for a in r['apps']) or r['free_mib']<1024
        return r
    def limits(self):
        super().limits()
        if time.monotonic()-self.last_disk>60:
            used=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file() and not p.is_symlink())
            if used>50*2**30:raise ResourceError('CACHE_LIMIT_50GIB')
            self.last_disk=time.monotonic()

def append(path,row):
    with open(path,'a') as f:
        f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def build(arm,seed,device='cuda'):
    from chronos import ChronosBoltPipeline
    from .model import attach_lora,ForecastModel
    receipt=read(OUT/'download_receipts.json')['amazon/chronos-bolt-small']
    base=ChronosBoltPipeline.from_pretrained(str(ROOT/receipt['snapshot']),device_map='cpu',torch_dtype=torch.float32).model
    base.eval().requires_grad_(False)
    for m in base.modules():
        if isinstance(m,torch.nn.Dropout):m.p=0.
        elif isinstance(getattr(m,'dropout',None),float):m.dropout=0.
    if arm in ARMS:attach_lora(base,seed)
    model=ForecastModel(base,arm,seed).eval().to(device)
    count=sum(p.numel() for p in parameters(model).values())
    assert count==(299784 if arm in ['A4','A5'] else 294912 if arm in ARMS else 0)
    return model

def cleanup():
    gc.collect();torch.cuda.empty_cache()

def check_seal():
    seal=read(OUT/'EXECUTION_SEAL.json')
    for path,h in seal['source_hashes'].items():assert sha(ROOT/path)==h,('SEALED_CODE_CHANGED',path)
    for path,h in seal['data_hashes'].items():assert sha(ROOT/path)==h,('SEALED_INPUT_CHANGED',path)
    return seal

def sync():
    torch.cuda.synchronize()

def status(**extra):
    runs=[read(p) for p in (OUT/'fits').glob('*/receipt.json')]
    smoke=read(OUT/'smoke_checks.json') if (OUT/'smoke_checks.json').exists() else {}
    journal=OUT/'optimizer.jsonl'
    n=sum(1 for _ in open(journal)) if journal.exists() else 0
    result=dict(execution='RUNNING',completed_fits=sum(r['status']=='COMPLETE' for r in runs),main_optimizer_updates=n,
                smoke_optimizer_updates=sum(v.get('updates',0) for v in smoke.values()))
    result.update(extra)
    save(OUT/'status.json',result)
    return result
