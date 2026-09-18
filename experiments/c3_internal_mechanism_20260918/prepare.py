from .common import *
from experiments.persistence_evidence_extension_20260918 import common as ext

def prepare():
 OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
 if (OUT/'SEAL.json').exists():check_seal();return
 parent.check_seal();hs={}
 def include(p):hs[str(p.relative_to(ROOT))]=sha(p)
 for p in EXP.glob('*.py'):include(p)
 for folder in ['c3_training_factorial_20260918','additive_persistence_validation_v1_20260917','c3_weakness_controls_20260918','outlier_signal_followup_v2_20260917','outlier_signal_peft_v1_20260917']:
  for p in (ROOT/'experiments'/folder).glob('*.py'):include(p)
 for name in ['GRID.json','MODEL_SELECTION.json','PREDICTIONS.json','RAW_SCORES.csv','CELL_EFFECTS.csv','INDEPENDENT_AUDIT.json']:include(PRIOR/name)
 include(OUT/'PROTOCOL.md');include(ROOT/'scripts/priority12/common.py')
 grid=read(PRIOR/'GRID.json');inventory=[]
 for row in grid:
  rec=read(PRIOR/'fits'/row['fit']/'receipt.json');include(PRIOR/'fits'/row['fit']/'receipt.json')
  for c in rec['checkpoints']:assert sha(ROOT/c['checkpoint'])==c['sha256'];include(ROOT/c['checkpoint'])
  resume=(ROOT/rec['checkpoints'][-1]['checkpoint']).parent/'resume.pt';ok=resume.exists()
  if ok:include(resume)
  inventory.append(dict(fit=row['fit'],resume=str(resume.relative_to(ROOT)),available=ok))
 for source in SOURCES:
  for n in ['train_x','train_y','train_sigma','train_generator_audit']:include(old.CACHE/'conditions'/source/(n+('.json' if n=='train_generator_audit' else '.npy')))
 for panel in ['electricity','electricity_transfer','ettm1']:
  for kind in ['standard','shape']:
   for p in ext.panel_path(panel,kind).glob('*'):
    if p.is_file() and (kind=='shape' or p.name.startswith('E_DISCOVERY')):include(p)
  for p in ext.data_path(panel).glob('E_DISCOVERY*.npz'):include(p)
 original={k:r for k,r in read(PRIOR/'PREDICTIONS.json').items() if r['stage']=='fixed1024'};assert len(original)==96
 for r in original.values():assert sha(ROOT/r['path'])==r['sha256'];include(ROOT/r['path'])
 save(OUT/'ORIGINAL_PREDICTIONS.json',original);save(OUT/'MOMENT_INVENTORY.json',inventory)
 probe=[dict(epoch=e,indices=(np.linspace(0,255,8,dtype=int)[:,None]*4+np.arange(4)[None,:]).ravel().tolist()) for e in [0,5,18,23]];save(OUT/'PROBE.json',probe);include(OUT/'PROBE.json')
 save(OUT/'SEAL.json',dict(at=time.time(),hashes=hs,new_fit_cap=0,optimizer_update_cap=0,autograd_cap=1280,new_E_views_cap=96,posthoc=True))
 print('SEALED',len(hs),'inputs;0 optimizer updates',flush=True)
if __name__=='__main__':prepare()
