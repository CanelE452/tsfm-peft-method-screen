"""Only generic GPU guard, hash, journal and state helpers are reused."""
import os, sys, time, json, gc, random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from priority12.common import save,read,sha,csvwrite,parameters,cpu_state,restore,frozen_hash,tensor_hash,ResourceError
from priority12.common import Watch as OriginalWatch
NAME='additive_b0_adapter_v1_20260917'
OUT=ROOT/'results'/NAME
CACHE=ROOT/'.cache'/NAME
EXP=ROOT/'experiments'/NAME
ARMS=['C1','C2','C3']
SOURCES=['electricity','ettm1']
def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    random.seed(81550);np.random.seed(81550);torch.manual_seed(81550)

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
        import shutil
        if shutil.disk_usage(ROOT).free<10*2**30:raise ResourceError('DISK_BELOW_10GIB')
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

def build(arm,seed,source,device='cuda'):
    from experiments.outlier_signal_followup_v2_20260917.common import build as old_build
    from .model import ForecastModel
    old=old_build('B0',seed,device='cpu')
    row=read(OUT/'BASELINE_MANIFEST.json')['models'][f'{source}_{seed}']
    assert sha(ROOT/row['checkpoint'])==row['sha256']
    restore(old,torch.load(ROOT/row['checkpoint'],weights_only=True,map_location='cpu'))
    if arm!='C1':old.requires_grad_(False)
    model=ForecastModel(old.base,arm,seed).eval().to(device)
    expected={'C0':0,'C1':294912,'C2':8712,'C3':8712}
    assert sum(p.numel() for p in parameters(model).values())==expected[arm]
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
