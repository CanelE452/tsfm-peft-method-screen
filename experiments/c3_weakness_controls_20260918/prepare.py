import subprocess,inspect
from .common import *
from experiments.persistence_evidence_extension_20260918 import common as ext
from experiments.paper_readiness_20260918 import common as ready
from chronos.chronos_bolt import InstanceNorm

def prepare():
    OUT.mkdir(exist_ok=True,parents=True);CACHE.mkdir(exist_ok=True,parents=True)
    if (OUT/'SEAL.json').exists():check_seal();return
    old.check_seal();ready.check_seal()
    for name in ['conditions','data']:
        p=CACHE/name
        if not p.exists():p.symlink_to(old.CACHE/name,target_is_directory=True)
    save(OUT/'DATA_MANIFEST.json',read(PRIOR/'DATA_MANIFEST.json'))
    save(OUT/'microbatch.json',{s:dict(microbatch=32,source='verified parent same source/model/batch; current smoke required') for s in SOURCES})
    bases={f'{s}_{seed}':old.baseline_row(s,seed) for s in SOURCES for seed in [81550,81551,81552,81553]};save(OUT/'BASELINE_MANIFEST.json',bases)
    (OUT/'INSTANCE_NORM_SOURCE.txt').write_text(inspect.getsource(InstanceNorm))
    plan=dict(sources=SOURCES,arms=ARMS,selection_seed=81550,lr_grid=[.0001,.0003],repeat_seeds=[81551,81552,81553],fit_cap=30,main_update_cap=30720,smoke_update_cap=12,total_update_cap=30732,checkpoints=[0,256,512,768,1024],panels=['electricity','electricity_transfer','ettm1'],kinds=['standard','shape'],new_prediction_view_cap=54,automatic_successor=False);save(OUT/'PLAN.json',plan)
    hashes=dict(read(ready.OUT/'SEAL.json')['hashes'])
    for root in [PRIOR,ext.OUT,ready.OUT,EXP]:
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:hashes[str(p.relative_to(ROOT))]=sha(p)
    for r in bases.values():assert sha(ROOT/r['checkpoint'])==r['sha256'];hashes[r['checkpoint']]=r['sha256']
    # Both train and V arrays are inherited unchanged; no new training generator.
    for source in SOURCES:
        for p in (old.CACHE/'conditions'/source).glob('*.npy'):hashes[str(p.relative_to(ROOT))]=sha(p)
    for p in OUT.glob('*'):
        if p.is_file():hashes[str(p.relative_to(ROOT))]=sha(p)
    save(OUT/'SEAL.json',dict(at=time.time(),initial_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),hashes=hashes,contract_sha256=sha(OUT/'EXECUTION_CONTRACT.md'),new_fits_cap=30,total_updates_cap=30732,no_new_E_scoring=True))
    check_seal();print('SEALED 30 fits / 30732 total updates',flush=True)
if __name__=='__main__':prepare()
