"""Fixed, bounded development pilot; old studies are read-only dependencies."""
import sys, time, json, hashlib, csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experiments.condition_sampling_repair_v1_20260917.common import sha,read,save,csvwrite,npz,rng,nrms,block_counts
from experiments.peft_rank12_20260915.common import parameters,cpu_state,restore,tensor_hash,frozen_hash,cleanup,Watch,ResourceError
NAME='history_compression_v1_20260917';OUT=ROOT/'results'/NAME;CACHE=ROOT/'.cache'/NAME;EXP=ROOT/'experiments'/NAME
OLD=ROOT/'results/condition_sampling_repair_v1_20260917';OC=ROOT/'.cache/condition_sampling_repair_v1_20260917/N02'
ARMS=['STATS_SHORT','POOL','POOL_KD','LEARN','LEARN_KD'];LRS=[1e-4,3e-5];SEEDS=[73100,73101,73102]
ROLES=['TRAIN','V_SELECT','E_DISCOVERY']
def write(p,text):Path(p).write_text(text)
def atomic_torch(p,x):
 import torch
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(p)
def all_source_hashes():
 paths=list(EXP.glob('*.py'))+[ROOT/'scripts/run_history_compression.py',ROOT/'src/tsfm_peft_screen/backbone.py',ROOT/'src/tsfm_peft_screen/lora.py']
 paths+=list((ROOT/'experiments/condition_sampling_repair_v1_20260917').glob('*.py'))+[ROOT/'experiments/peft_rank12_20260915/common.py',ROOT/'scripts/priority12/common.py']
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
def validate_seal():
 s=read(OUT/'SEAL.json')
 for p,h in {**s['sources'],**s['inputs']}.items():assert sha(ROOT/p)==h,('SEALED_FILE_CHANGED',p)
 return s
class Data:
 def __init__(self):
  self.stats=read(OLD/'N02/train_statistics.json');self.sigma=np.array(self.stats['sigma']);self.mu=np.array(self.stats['mu'])
  self.x={r:dict(np.load(OC/f'{r}_inputs.npz')) for r in ROLES}
  self.y={r:np.load(OC/f'{r}_labels.npz')['y'] for r in ['TRAIN','V_SELECT']}
 def packet(self,role,i):return dict(x=self.x[role]['context'][i],mu=self.mu,sigma=self.sigma,H=48,origin=int(self.x[role]['origins'][i]))
