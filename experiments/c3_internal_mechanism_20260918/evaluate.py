from .common import *
from experiments.c3_training_factorial_20260918.train import predict
from experiments.persistence_evidence_extension_20260918 import common as ext

def evaluate(watch):
 check_seal();original=read(OUT/'ORIGINAL_PREDICTIONS.json');manifest={}
 for key,r in original.items():
  b,i,o=r['seed'];manifest['original__'+key]=dict(r,trained_b=b,received_b=b,i=i,o=o,reused_prediction=True)
 if (OUT/'PREDICTIONS.json').exists():manifest=read(OUT/'PREDICTIONS.json')
 plan=read(PRIOR/'GRID.json');save(OUT/'EVALUATION_SEAL.json',dict(at=time.time(),plan_sha256=sha(PRIOR/'GRID.json'),source_seal=sha(OUT/'SEAL.json'),step=1024,optimizer_updates=0))
 for row in plan:
  b,i,o=row['seed'];received=81551 if b==81552 else 81552;rec=read(PRIOR/'fits'/row['fit']/'receipt.json');cp=next(c for c in rec['checkpoints'] if c['step']==1024)
  keys=[(panel,kind,f"swapped__{row['fit']}__r{received}__{panel}__{kind}") for panel in [row['source']]+(['electricity_transfer'] if row['source']=='electricity' else []) for kind in ['standard','shape']]
  if all(k in manifest for _,_,k in keys):continue
  watch.boundary();m=load(row,b=received);m2=load(row,b=received);base=build('C0',(received,i,o),row['source']);before=tensor_hash(m.state_dict())
  for panel,kind,key in keys:
   if key in manifest:assert sha(ROOT/manifest[key]['path'])==manifest[key]['sha256'];continue
   x,s=ext.inputs(panel,kind);torch.cuda.reset_peak_memory_stats();t=time.perf_counter();p=predict(m,x,s,32,watch);elapsed=time.perf_counter()-t;assert np.isfinite(p).all()
   replay=predict(m2,x[:32],s[:32],32,watch);assert np.array_equal(p[:32],replay)
   xx=torch.tensor(np.array(x[:32]),device='cuda');ss=torch.tensor(np.array(s[:32]),device='cuda')
   with torch.no_grad():assert torch.equal(m2(xx,ss,residual_mode='off'),base(xx,ss))
   path=CACHE/'predictions'/f'{key}.npy';path.parent.mkdir(exist_ok=True);np.save(path,p)
   manifest[key]=dict(fit=row['fit'],source=row['source'],arm=row['arm'],seed=row['seed'],trained_b=b,received_b=received,i=i,o=o,panel=panel,kind=kind,stage='fixed1024',step=1024,checkpoint_sha256=cp['sha256'],B0_sha256=parent.baseline_row(row['source'],(received,i,o))['sha256'],path=str(path.relative_to(ROOT)),sha256=sha(path),shape=list(p.shape),reused_prediction=False,inference_seconds=elapsed,peak_allocated=torch.cuda.max_memory_allocated(),restore_exact=True,off_B0_exact=True)
   assert tensor_hash(m.state_dict())==before;COUNTS['E_small_forward_calls']=COUNTS.get('E_small_forward_calls',0)+3;save_counts();save(OUT/'PREDICTIONS.json',manifest);print('PREDICTION',len(manifest),key,flush=True)
  del m,m2,base,p;cleanup()
 assert len(manifest)==192 and sum(not r['reused_prediction'] for r in manifest.values())==96
 for r in manifest.values():assert sha(ROOT/r['path'])==r['sha256']
 check_seal();save(OUT/'ALL_PREDICTIONS_SAVED.json',dict(at=time.time(),count=192,manifest_sha256=sha(OUT/'PREDICTIONS.json'),new_E_scored=False))
