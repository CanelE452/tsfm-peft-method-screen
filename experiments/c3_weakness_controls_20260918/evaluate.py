from .common import *
from .train import predict
from experiments.persistence_evidence_extension_20260918.common import inputs
from experiments.additive_persistence_validation_v1_20260917.evaluate import profile

def load(row):
    assert sha(ROOT/row['checkpoint'])==row['sha256'];m=build(row['arm'],row['seed'],row['source']);restore(m,torch.load(ROOT/row['checkpoint'],weights_only=True,map_location='cpu'));return m

def evaluate(watch):
    check_seal();models=read(OUT/'MODEL_SELECTION.json');assert len(models)==18
    assert sum(1 for _ in open(OUT/'UPDATE_LEDGER.jsonl'))==30720
    fits=[read(p) for p in (OUT/'fits').glob('*/receipt.json')];assert len(fits)==30 and all(r['status']=='COMPLETE' for r in fits)
    selection=dict(at=time.time(),model_selection_sha256=sha(OUT/'MODEL_SELECTION.json'),lr_selection_sha256=sha(OUT/'LR_SELECTION.json'),main_updates=30720,smoke_updates=12)
    if (OUT/'EVALUATION_SEAL.json').exists():
        prior=read(OUT/'EVALUATION_SEAL.json');assert prior['model_selection_sha256']==selection['model_selection_sha256'] and prior['lr_selection_sha256']==selection['lr_selection_sha256']
    else:save(OUT/'EVALUATION_SEAL.json',selection)
    manifest=read(OUT/'PREDICTIONS.json') if (OUT/'PREDICTIONS.json').exists() else {};checks=read(OUT/'MODEL_REPLAY.json') if (OUT/'MODEL_REPLAY.json').exists() else {};resources=read(OUT/'RESOURCES.json') if (OUT/'RESOURCES.json').exists() else {}
    for row in models:
        for panel in [row['source']]+(['electricity_transfer'] if row['source']=='electricity' else []):
            for kind in ['standard','shape']:
                key=f"{panel}__{kind}__{row['arm']}__{row['seed']}"
                if key in manifest:assert sha(ROOT/manifest[key]['path'])==manifest[key]['sha256'];continue
                watch.boundary();m=load(row);before=tensor_hash(m.state_dict());x,s=inputs(panel,kind);t=time.perf_counter();torch.cuda.reset_peak_memory_stats();p=predict(m,x,s,32,watch);elapsed=time.perf_counter()-t
                cost=dict(inference_seconds=elapsed,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
                if kind=='standard':cost.update(profile(m,x,s,32,watch))
                assert before==tensor_hash(m.state_dict());m2=load(row);replay=predict(m2,x[:32],s[:32],32,watch);np.testing.assert_allclose(replay,p[:32],rtol=1e-4,atol=0)
                b=build('C0',row['seed'],row['source']);xx=torch.tensor(np.array(x[:32]),device='cuda');ss=torch.tensor(np.array(s[:32]),device='cuda')
                with torch.no_grad():assert torch.equal(m2(xx,ss,residual_mode='off'),b(xx,ss))
                path=CACHE/'predictions'/f'{key}.npy';path.parent.mkdir(exist_ok=True);np.save(path,p);manifest[key]=dict(panel=panel,kind=kind,arm=row['arm'],seed=row['seed'],source=row['source'],path=str(path.relative_to(ROOT)),sha256=sha(path),shape=list(p.shape),checkpoint_sha256=row['sha256']);checks[key]=dict(restore=True,off_B0_exact=True,state_unchanged=True);resources[key]=cost
                save(OUT/'PREDICTIONS.json',manifest);save(OUT/'MODEL_REPLAY.json',checks);save(OUT/'RESOURCES.json',resources);print('PREDICTION',len(manifest),key,flush=True)
                del m,m2,b,p;cleanup()
    assert len(manifest)==54;check_seal();save(OUT/'ALL_PREDICTIONS_SAVED.json',dict(at=time.time(),count=54,manifest_sha256=sha(OUT/'PREDICTIONS.json'),new_E_scored=False))
