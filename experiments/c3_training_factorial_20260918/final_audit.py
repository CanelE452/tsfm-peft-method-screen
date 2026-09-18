"""Independent saved-score marginal contrasts and actual factor matching audit."""
import itertools
import pandas as pd
from .common import *

def audit():
 check_seal();grid=read(OUT/'GRID.json');assert len(grid)==32
 receipts=[read(OUT/'fits'/r['fit']/'receipt.json') for r in grid];new=[r for r in receipts if not r.get('reused',False)]
 assert len(new)==24 and len(receipts)-len(new)==8
 ledger=[json.loads(l) for l in open(OUT/'UPDATE_LEDGER.jsonl')];assert len(ledger)==24576
 for r in new:
  rows=[x for x in ledger if x['fit']==r['fit']];assert [x['step'] for x in rows]==list(range(1,1025));assert all(np.isfinite(x['loss']) and np.isfinite(x['gradient_norm']) for x in rows)
  assert r['frozen_unchanged'] and r['microbatch']==32
  assert r['selected']==min(r['checkpoints'],key=lambda c:(c['objective'],c['step']))
  assert read(OUT/'fits'/r['fit']/'initial_parity.json')['exact']
 assert sum(sum(1 for _ in open(p)) for p in OUT.glob('smoke_*.jsonl'))==12
 for source in SOURCES:
  for b in LEVELS:
   rows=[r for r in receipts if r['source']==source and r['seed'][0]==b];assert len({r['frozen_sha256'] for r in rows})==1
  for i in LEVELS:
   rows=[r for r in receipts if r['source']==source and r['seed'][1]==i];hashes=[]
   for r in rows:
    zero=next(c for c in r['checkpoints'] if c['step']==0);assert sha(ROOT/zero['checkpoint'])==zero['sha256'];hashes.append(tensor_hash(torch.load(ROOT/zero['checkpoint'],map_location='cpu',weights_only=True)))
   assert len(set(hashes))==1
 f=pd.read_csv(OUT/'ORIGIN_SCORES.csv.gz');group=['panel','kind','stage','arm','b','i','o','origin'];fault=f[f.condition.str.startswith(('POINT','BURST'))].groupby(group,as_index=False).nmae.mean();fault['condition']='FAULT';f=pd.concat([f,fault],ignore_index=True)
 means=f.groupby(['panel','kind','stage','condition','arm','b','i','o']).nmae.mean();effects=pd.read_csv(OUT/'FACTORIAL_EFFECTS.csv');checked=0
 # No call to the vectorized factorial implementation: marginal grouping and loop sums.
 for r in effects.itertuples():
  axes=r.factor.split(':');aa=[];bb=[]
  for b,i,o in itertools.product(LEVELS,repeat=3):
   levels={'B0':b,'INIT':i,'ORDER':o};sign=np.prod([1 if levels[k]==81552 else -1 for k in axes]);a=float(means.loc[(r.panel,r.kind,r.stage,r.condition,'C3',b,i,o)]);d=float(means.loc[(r.panel,r.kind,r.stage,r.condition,'MAG_ONLY',b,i,o)])
   aa.append(sign*a);bb.append(sign*d)
  a=sum(aa)/4;b=sum(bb)/4
  np.testing.assert_allclose([a,b,a-b],[r.C3_effect,r.MAG_effect,r.effect_nmae],rtol=1e-8,atol=1e-12);checked+=1
 cells=pd.read_csv(OUT/'CELL_EFFECTS.csv');assert len(cells)==960 and len(effects)==840
 for r in cells.itertuples():
  a=float(means.loc[(r.panel,r.kind,r.stage,r.condition,'C3',r.b,r.i,r.o)]);b=float(means.loc[(r.panel,r.kind,r.stage,r.condition,'MAG_ONLY',r.b,r.i,r.o)])
  np.testing.assert_allclose([a,b,a-b,100*(a-b)/a],[r.C3,r.MAG,r.difference,r.MAG_gain_pct],rtol=1e-8,atol=1e-12)
 manifest=read(OUT/'PREDICTIONS.json');marker=read(OUT/'ALL_PREDICTIONS_SAVED.json');assert sha(OUT/'PREDICTIONS.json')==marker['manifest_sha256']
 for r in manifest.values():assert sha(ROOT/r['path'])==r['sha256']
 for r in read(OUT/'MODEL_SELECTION.json'):assert sha(ROOT/r['checkpoint'])==r['sha256']
 checks=read(OUT/'SMOKE.json');assert len(checks)==4 and all(r['frozen_unchanged'] and r['restore_exact'] and r['off_B0_exact'] for r in checks.values())
 gpu=[json.loads(l) for l in open(OUT/'gpu_factorial.jsonl')];unsafe=[r for r in gpu if any(not a['own'] and a['name']!='/usr/share/rustdesk/rustdesk' for a in r['apps'])]
 save(OUT/'INDEPENDENT_AUDIT.json',dict(status='VERIFIED',completed_paths=32,new_fits=24,reused_fits=8,new_main_updates=24576,smoke_updates=12,independent_contrasts=checked,independent_cell_replay=960,factor_initialization_match=True,B0_frozen_match=True,checkpoint_selection_V_only=True,all_model_and_prediction_hashes=True,source_seal_unchanged=True,GPU_min_free_mib=min(r['free_mib'] for r in gpu),GPU_external_training_samples=len(unsafe),new_prediction_views=sum(not r['reused_prediction'] for r in manifest.values()),reused_prediction_views=sum(r['reused_prediction'] for r in manifest.values()),posthoc_two_level_scope=True))
 print('INDEPENDENT_AUDIT_VERIFIED',checked,flush=True)
if __name__=='__main__':audit()
