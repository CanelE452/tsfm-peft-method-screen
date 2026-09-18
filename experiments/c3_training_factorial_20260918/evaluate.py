from .common import *
from .train import predict
from experiments.persistence_evidence_extension_20260918 import common as ext

def load(row):
 assert sha(ROOT/row['checkpoint'])==row['sha256'];m=build(row['arm'],row['seed'],row['source']);restore(m,torch.load(ROOT/row['checkpoint'],weights_only=True,map_location='cpu'));return m

def evaluate(watch):
 check_seal();models=read(OUT/'MODEL_SELECTION.json');assert len(models)==64
 assert sum(1 for _ in open(OUT/'UPDATE_LEDGER.jsonl'))==MAIN_CAP
 selection=dict(at=time.time(),models_sha256=sha(OUT/'MODEL_SELECTION.json'),main_updates=MAIN_CAP,smoke_updates=12)
 if (OUT/'EVALUATION_SEAL.json').exists():assert read(OUT/'EVALUATION_SEAL.json')['models_sha256']==selection['models_sha256']
 else:save(OUT/'EVALUATION_SEAL.json',selection)
 prior=read(ROOT/'results/c3_magnitude_diagnostic_20260918/REUSED_PREDICTIONS.json')
 raw=read(old.OUT/'PREDICTIONS_MANIFEST.json');prior+=list(raw.values()) if isinstance(raw,dict) else raw
 manifest=read(OUT/'PREDICTIONS.json') if (OUT/'PREDICTIONS.json').exists() else {}
 for row in models:
  for panel in [row['source']]+(['electricity_transfer'] if row['source']=='electricity' else []):
   for kind in ['standard','shape']:
    key=f"{row['fit']}__{row['stage']}__{panel}__{kind}"
    if key in manifest:assert sha(ROOT/manifest[key]['path'])==manifest[key]['sha256'];continue
    meta=dict(fit=row['fit'],source=row['source'],arm=row['arm'],seed=row['seed'],panel=panel,kind=kind,stage=row['stage'],checkpoint_sha256=row['sha256'],step=row['step'])
    matches=[r for r in manifest.values() if r['panel']==panel and r['kind']==kind and r['arm']==row['arm'] and r['seed'][0]==row['seed'][0] and r['checkpoint_sha256']==row['sha256']]
    if len(set(row['seed']))==1:
     matches += [r for r in prior if r.get('panel')==panel and r.get('kind')==kind and r.get('arm')==row['arm'] and r.get('seed')==row['seed'][0] and r.get('checkpoint_sha256')==row['sha256']]
    if matches:
     r=matches[0];assert sha(ROOT/r['path'])==r['sha256'];manifest[key]=dict(**meta,path=r['path'],sha256=r['sha256'],shape=r['shape'],reused_prediction=True,inference_seconds=0.)
    else:
     watch.boundary();m=load(row);before=tensor_hash(m.state_dict());x,s=ext.inputs(panel,kind);t=time.perf_counter();torch.cuda.reset_peak_memory_stats();p=predict(m,x,s,32,watch);elapsed=time.perf_counter()-t
     assert before==tensor_hash(m.state_dict());cost=dict(inference_seconds=elapsed,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
     m2=load(row);replay=predict(m2,x[:32],s[:32],32,watch);assert np.array_equal(replay,p[:32]),'RESTORE_NOT_EXACT'
     base=build('C0',row['seed'],row['source']);xx=torch.tensor(np.array(x[:32]),device='cuda');ss=torch.tensor(np.array(s[:32]),device='cuda')
     with torch.no_grad():assert torch.equal(m2(xx,ss,residual_mode='off'),base(xx,ss))
     path=CACHE/'predictions'/f'{key}.npy';path.parent.mkdir(exist_ok=True);np.save(path,p)
     manifest[key]=dict(**meta,path=str(path.relative_to(ROOT)),sha256=sha(path),shape=list(p.shape),reused_prediction=False,restore_exact=True,off_B0_exact=True,state_unchanged=True,**cost)
     del m,m2,base,p;cleanup()
    save(OUT/'PREDICTIONS.json',manifest);print('PREDICTION',len(manifest),key,flush=True)
 assert len(manifest)==192;check_seal();save(OUT/'ALL_PREDICTIONS_SAVED.json',dict(at=time.time(),count=192,manifest_sha256=sha(OUT/'PREDICTIONS.json'),new_E_scored=False))
