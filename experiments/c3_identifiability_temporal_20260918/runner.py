"""Inference-only run. No optimizer, backward, LR/checkpoint selection or calibration."""
import traceback
from .common import *

def run():
    check_seal();old.setup();watch=Watch();manifest=read(OUT/'PREDICTIONS.json') if (OUT/'PREDICTIONS.json').exists() else {};checks=read(OUT/'MODEL_CHECKS.json') if (OUT/'MODEL_CHECKS.json').exists() else {}
    parents=list(read(PRIOR/'PREDICTIONS_MANIFEST.json').values());t0=time.time()
    try:
        watch.boundary(startup=True)
        for j in read(OUT/'PLAN.json'):
            key=j['id']
            if key in manifest:assert sha(ROOT/manifest[key]['path'])==manifest[key]['sha256'];continue
            x,s=inputs(NEW,j['kind']);started=time.perf_counter();peak=None
            if j['arm'] in ['PERSISTENCE','SEASONAL']:
                a,b=ready.naive(np.asarray(x),24);p=np.repeat((a if j['arm']=='PERSISTENCE' else b)[:,None,:],9,axis=1).copy();checks[key]=dict(observed_only=True,optimizer_updates=0)
            else:
                watch.boundary()
                def build():
                    if j['row']:m=load_model(j['row'])
                    else:
                        from experiments.outlier_signal_peft_v1_20260917.common import build as foundation
                        m=foundation('FROZEN_RAW',0)
                    m.requires_grad_(False);return m
                m=build();before=tensor_hash(m.state_dict());assert not any(q.requires_grad for q in m.parameters())
                if j['row']:
                    xx,ss=ext.inputs('electricity','standard');ref=next(r for r in parents if r['panel']=='electricity' and r['kind']=='standard' and r['arm']==j['trained_arm'] and r['seed']==j['seed'] and r.get('stage')=='selected')
                    assert sha(ROOT/ref['path'])==ref['sha256'];assert np.array_equal(predict(m,xx[:32],ss[:32],32,watch),np.load(ROOT/ref['path'],mmap_mode='r')[:32]),'PARENT_FORWARD_CHANGED'
                    m.arm=j['gate_arm']
                torch.cuda.reset_peak_memory_stats();p=predict(m,x,s,32,watch);peak=torch.cuda.max_memory_allocated();assert before==tensor_hash(m.state_dict())
                m2=build()
                if j['row']:m2.arm=j['gate_arm']
                assert before==tensor_hash(m2.state_dict());np.testing.assert_allclose(predict(m2,x[:32],s[:32],32,watch),p[:32],rtol=1e-4,atol=0)
                checks[key]=dict(all_weights_frozen=True,state_unchanged=True,restore_forward_verified=True,parent_exact_forward=bool(j['row']),state_sha256=before,optimizer_updates=0)
                del m,m2;old.cleanup()
            assert np.isfinite(p).all();path=CACHE/'predictions'/f'{key}.npy';path.parent.mkdir(exist_ok=True);np.save(path,p)
            manifest[key]={**{k:j[k] for k in ['panel','kind','arm','trained_arm','gate_arm','seed']},'path':str(path.relative_to(ROOT)),'sha256':sha(path),'shape':list(p.shape),'seconds':time.perf_counter()-started,'peak_allocated_bytes':peak,'checkpoint_sha256':j['row']['sha256'] if j['row'] else None};del p
            save(OUT/'PREDICTIONS.json',manifest);save(OUT/'MODEL_CHECKS.json',checks);save(OUT/'status.json',dict(execution='RUNNING',views=len(manifest),cap=60,fits=0,updates=0));print('SAVED',len(manifest),key,flush=True)
        assert len(manifest)==60;check_seal();save(OUT/'ALL_PREDICTIONS_SAVED.json',dict(at=time.time(),manifest_sha256=sha(OUT/'PREDICTIONS.json'),views=60,new_scores_computed=False,updates=0));save(OUT/'status.json',dict(execution='PREDICTIONS_COMPLETE',views=60,fits=0,updates=0));save(OUT/'COST.json',dict(wall_seconds=time.time()-t0,view_seconds=sum(r['seconds'] for r in manifest.values()),new_fits=0,optimizer_updates=0,GPU_views=56,CPU_views=4,peak_allocated_bytes=max(r['peak_allocated_bytes'] or 0 for r in manifest.values())))
    except BaseException as e:
        save(OUT/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));save(OUT/'status.json',dict(execution='RESOURCE_BLOCK' if isinstance(e,ResourceError) else 'EXECUTION_ERROR',views=len(manifest),error=str(e),updates=0));raise
    finally:watch.close()
if __name__=='__main__':run()
