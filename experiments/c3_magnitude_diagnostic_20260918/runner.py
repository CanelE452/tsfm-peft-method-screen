"""36 fixed-weight cross-gate predictions, no training or selection."""
from .common import *
from experiments.additive_persistence_validation_v1_20260917.train import predict

def prepare():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    if (OUT/'SEAL.json').exists():return check_seal()
    parent.check_seal()
    hashes=dict(read(parent.OUT/'SEAL.json')['hashes'])
    for p in parent.OUT.rglob('*'):
        if p.is_file():hashes[str(p.relative_to(ROOT))]=sha(p)
    for r in models()+records():
        p=ROOT/(r['checkpoint'] if 'checkpoint' in r else r['path']);assert sha(p)==r['sha256'];hashes[str(p.relative_to(ROOT))]=r['sha256']
    for p in list(EXP.glob('*.py'))+[OUT/'PROTOCOL.md']:
        hashes[str(p.relative_to(ROOT))]=sha(p)
    save(OUT/'SEAL.json',dict(at=time.time(),posthoc=True,max_new_prediction_views=36,new_fits=0,optimizer_updates=0,hashes=hashes))
    save(OUT/'MODELS.json',models());save(OUT/'REUSED_PREDICTIONS.json',records())
    return check_seal()

def run():
    prepare();old.setup()
    from .features import make
    if not (OUT/'FEATURE_VERIFICATION.json').exists():make()
    watch=Watch();manifest=read(OUT/'PREDICTIONS.json') if (OUT/'PREDICTIONS.json').exists() else {};checks=read(OUT/'MODEL_CHECKS.json') if (OUT/'MODEL_CHECKS.json').exists() else {}
    try:
        watch.boundary(startup=True)
        for row in models():
            source=row['source'];gate='MAG_ONLY' if row['arm']=='C3' else 'C3'
            for panel in [source]+(['electricity_transfer'] if source=='electricity' else []):
                for kind in ['standard','shape']:
                    key=f"{panel}__{kind}__W{row['arm']}__G{gate}__{row['seed']}"
                    if key in manifest:assert sha(ROOT/manifest[key]['path'])==manifest[key]['sha256'];continue
                    watch.boundary();x,s=ext.inputs(panel,kind);idx=np.linspace(0,len(x)-1,32,dtype=int);model=load(row,row['arm']);before=tensor_hash(model.state_dict())
                    r=next(r for r in records() if (r['panel'],r['kind'],r['arm'],r['seed'])==(panel,kind,row['arm'],row['seed']))
                    pp=predict(model,x[idx],s[idx],32,watch);expected=np.load(ROOT/r['path'],mmap_mode='r')[idx];np.testing.assert_allclose(pp,expected,rtol=1e-4,atol=1e-5)
                    b=parent.build('C0',row['seed'],source);xx=torch.tensor(np.array(x[idx]),device='cuda');ss=torch.tensor(np.array(s[idx]),device='cuda')
                    with torch.no_grad():assert torch.equal(model(xx,ss,residual_mode='off'),b(xx,ss))
                    del b;model.gate=(lambda x,s:correction_gate(x,s)) if gate=='C3' else model.__class__.gate.__get__(model)
                    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();p=predict(model,x,s,32,watch);cost=dict(seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated());assert before==tensor_hash(model.state_dict())
                    fresh=load(row,gate);replay=predict(fresh,x[idx],s[idx],32,watch);np.testing.assert_allclose(replay,p[idx],rtol=1e-4,atol=1e-5)
                    path=CACHE/(key+'.npy');np.save(path,p);manifest[key]=dict(panel=panel,kind=kind,arm='C3_W_MAG_G' if row['arm']=='C3' else 'MAG_W_C3_G',weights=row['arm'],gate=gate,source=source,seed=row['seed'],path=str(path.relative_to(ROOT)),sha256=sha(path),shape=list(p.shape),checkpoint_sha256=row['sha256'],**cost)
                    checks[key]=dict(diagonal_replay=True,off_B0_exact=True,weights_and_buffers_unchanged=True,fresh_restore_replay=True,all_parameters_frozen=not any(t.requires_grad for t in model.parameters()))
                    save(OUT/'PREDICTIONS.json',manifest);save(OUT/'MODEL_CHECKS.json',checks);save(OUT/'status.json',dict(execution='PREDICTING',completed_views=len(manifest),new_fits=0,optimizer_updates=0));print('SWAP',len(manifest),key,flush=True)
                    del model,fresh,p;old.cleanup()
        assert len(manifest)==36;check_seal();save(OUT/'ALL_PREDICTIONS_SAVED.json',dict(at=time.time(),count=36,manifest_sha256=sha(OUT/'PREDICTIONS.json')))
    except BaseException as e:
        import traceback
        save(OUT/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
    finally:watch.close()
    from .analyze import analyze
    analyze()
if __name__=='__main__':run()
