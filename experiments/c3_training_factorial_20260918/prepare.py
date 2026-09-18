import itertools
from .common import *
from .train import fit_id
from experiments.outlier_signal_peft_v1_20260917.reference_core import rng
from experiments.persistence_evidence_extension_20260918 import common as ext

def prepare():
 OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
 if (OUT/'SEAL.json').exists():check_seal();return
 hashes={}
 def include(p):hashes[str(p.relative_to(ROOT))]=sha(p)
 for run in [EXP,ROOT/'experiments/additive_persistence_validation_v1_20260917',ROOT/'experiments/additive_b0_adapter_v1_20260917',ROOT/'experiments/c3_weakness_controls_20260918',ROOT/'experiments/outlier_signal_followup_v2_20260917',ROOT/'experiments/outlier_signal_peft_v1_20260917']:
  for p in run.glob('*.py'):include(p)
 include(OUT/'PROTOCOL.md');include(ROOT/'scripts/priority12/common.py')
 for source in SOURCES:
  d=CACHE/'conditions'/source;d.parent.mkdir(exist_ok=True)
  if not d.exists():d.symlink_to((old.CACHE/'conditions'/source).resolve(),target_is_directory=True)
  for name in ['train_x','train_y','train_sigma','V_SELECT_x','V_SELECT_y','V_SELECT_sigma']:include(d/(name+'.npy'))
 for panel in ['electricity','electricity_transfer','ettm1']:
  for kind in ['standard','shape']:
   f=ext.panel_path(panel,kind)
   for p in f.glob('*'):
    if p.is_file() and (kind=='shape' or p.name.startswith('E_DISCOVERY')):include(p)
  for p in ext.data_path(panel).glob('E_DISCOVERY*.npz'):include(p)
 data=read(old.OUT/'DATA_MANIFEST.json');save(OUT/'DATA_MANIFEST.json',data)
 bases={f'{s}_{b}':old.baseline_row(s,b) for s in SOURCES for b in LEVELS}
 for r in bases.values():assert sha(ROOT/r['checkpoint'])==r['sha256'];include(ROOT/r['checkpoint'])
 save(OUT/'BASELINES.json',bases)
 models=read(ROOT/'results/c3_magnitude_diagnostic_20260918/MODELS.json');grid=[];reuse=[]
 for source in SOURCES:
  lr=.0003 if source=='electricity' else .0001
  for triple in itertools.product(LEVELS,repeat=3):
   for arm in ARMS:
    fid=fit_id(source,arm,triple,lr);row=dict(fit=fid,source=source,arm=arm,seed=list(triple),lr=lr,reused=len(set(triple))==1);grid.append(row)
    if row['reused']:
     m=next(x for x in models if x['source']==source and x['arm']==arm and x['seed']==triple[0]);folder=Path(m['checkpoint']).parts[1];p=ROOT/'results'/folder/'fits'/m['fit']/'receipt.json';receipt=read(p);include(p)
     assert receipt['updates']==1024 and receipt['microbatch']==32 and receipt['lr']==lr and receipt['frozen_unchanged']
     assert [r['step'] for r in receipt['checkpoints']]==[0,256,512,768,1024]
     for r in receipt['checkpoints']:assert sha(ROOT/r['checkpoint'])==r['sha256'];include(ROOT/r['checkpoint'])
     out=dict(receipt,**row,original_fit=m['fit'],original_receipt=str(p.relative_to(ROOT)),new_updates=0)
     save(OUT/'fits'/fid/'receipt.json',out);reuse.append(out)
 assert len(grid)==32 and len(reuse)==8
 save(OUT/'GRID.json',grid);include(OUT/'GRID.json');save(OUT/'REUSED_FITS.json',reuse)
 # Three factors really are independently assigned, paired arms share everything else.
 checks=[]
 for source in SOURCES:
  for triple in itertools.product(LEVELS,repeat=3):
   order=np.stack([rng(84100,source,triple[2],e).permutation(1024) for e in range(32)])
   checks.append(dict(source=source,seed=list(triple),B0_sha256=bases[f'{source}_{triple[0]}']['sha256'],order_sha256=sha_bytes(order.tobytes())))
 save(OUT/'ASSIGNMENT_CHECKS.json',checks)
 save(OUT/'microbatch.json',{s:dict(microbatch=32) for s in SOURCES})
 save(OUT/'SEAL.json',dict(at=time.time(),hashes=hashes,main_update_cap=MAIN_CAP,smoke_cap=12,new_fit_cap=24,paths=32,reused=8,posthoc=True))
 print('SEALED',len(hashes),'inputs;32 paths,8 reused,24 new',flush=True)
def sha_bytes(b):
 import hashlib
 return hashlib.sha256(b).hexdigest()
if __name__=='__main__':prepare()
