"""Explicit new paths; reuse tested helpers without rebinding historical globals."""
import sys,time,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.covariate_lora_controls_20260916 import common as base
np,torch=base.np,base.torch
save,read,sha=base.save,base.read,base.sha
RUN='covariate_order_control_20260916';OUT=ROOT/'results'/RUN;CACHE=ROOT/'.cache'/RUN;OLD=ROOT/'results/covariate_lora_controls_20260916'
ARMS=['REVERSE','SHUFFLE'];SEEDS=[61710,61711]

def prepare():
 assert not (OUT/'seal.json').exists(),'Existing preparation'
 old=read(OLD/'seal.json');assert read(OLD/'status.json')['status']=='COMPLETE'
 for p,h in {**old['files'],**old['model_files']}.items():assert sha(p)==h,p
 for p,h in read(OLD/'publication_audit.json')['artifacts'].items():assert sha(ROOT/p)==h,p
 for j in old['jobs']:
  for k in ['input','target']:assert sha(ROOT/j[k])==j[k+'_sha256']
 for p in read(OLD/'predictions.json'):assert sha(ROOT/p['path'])==p['sha256']
 train=sorted([j for j in old['jobs'] if j['split']=='TRAIN'],key=lambda j:j['id']);schedule=[];exposure=[]
 old_schedule=read(OLD/'schedule.json');sequence=np.array([0,1,2,3]*3+[0,1,2])
 for seed in SEEDS:
  reference=[r for r in old_schedule if r['arm']=='VINTAGE' and r['seed']==seed]
  for arm in ARMS:
   versions={j['id']:(sequence[::-1] if arm=='REVERSE' else np.random.default_rng(seed+60000+i).permutation(sequence)) for i,j in enumerate(train)}
   for r in reference:schedule.append(dict(arm=arm,seed=seed,step=r['step'],epoch=r['epoch'],id=r['id'],path=int(versions[r['id']][r['epoch']])))
   for j in train:
    paths=versions[j['id']].tolist();assert [paths.count(k) for k in range(4)]==[4,4,4,3]
    exposure.append(dict(arm=arm,seed=seed,id=j['id'],paths=paths,counts=[4,4,4,3]))
 assert len(schedule)==480;save(OUT/'schedule.json',schedule);save(OUT/'exposure.json',exposure)
 oldpaths=subprocess.check_output(['git','ls-files','results','research'],cwd=ROOT,text=True).splitlines();save(OUT/'historical_hashes.json',{p:sha(ROOT/p) for p in oldpaths})
 files=list(Path(__file__).parent.glob('*.py'))+[OUT/'PROTOCOL.md',OUT/'schedule.json',OUT/'exposure.json',Path(base.__file__),OLD/'scaling.json',OLD/'fits.json',OLD/'scores.csv',OLD/'predictions.json',OLD/'preflight.json']
 # Source formatting is checked before the immutable execution receipt is made.
 for p in files:
  if p.suffix=='.py':assert all(line==line.rstrip() for line in p.read_text().splitlines()),('TRAILING_WHITESPACE',str(p))
 save(OUT/'seal.json',dict(at=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files={str(p):sha(p) for p in files},reference_seal=str((OLD/'seal.json').relative_to(ROOT)),reference_seal_sha256=sha(OLD/'seal.json'),reference_files=old['files'],model_files=old['model_files'],jobs=old['jobs'],planned_fits=4,planned_updates=480,planned_smoke_updates=4))
 print('PREPARED: 4 fits, 480 updates; equal per-origin vintage multisets',flush=True)

def verified():
 s=read(OUT/'seal.json')
 for p,h in {**s['files'],**s['reference_files'],**s['model_files']}.items():assert sha(p)==h,p
 assert sha(ROOT/s['reference_seal'])==s['reference_seal_sha256']
 for j in s['jobs']:
  for k in ['input','target']:assert sha(ROOT/j[k])==j[k+'_sha256']
 return s
