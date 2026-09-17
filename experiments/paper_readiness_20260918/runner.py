import argparse,traceback
from .common import *

def prepare():
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True);ext.check_seal()
    if (OUT/'SEAL.json').exists():check_seal();return
    rows=read(PRIOR/'MODEL_SELECTION.json');plan=[]
    for panel in ['electricity','electricity_transfer','neso_2025']:
        for kind in ['standard','shape']:
            for seed in [81551,81552,81553]:
                for trained in ['C2','C3']:
                    original=next(r for r in rows if (r['source'],r['arm'],r['seed'])==('electricity',trained,seed));assert original['lr']==.0003
                    row=dict(original,**checkpoint_row(original,'fixed1024'));assert row['step']==1024
                    for gate in ['C2','C3']:
                        arm=f'FIX_{trained}_G{gate}'
                        plan.append(dict(panel=panel,kind=kind,seed=seed,arm=arm,row=row,gate=gate,trained=trained))
    for kind in ['standard','shape']:
        for seed in [81551,81552,81553]:
            for trained,gate in [('C3','M_RECENCY'),('M_RECENCY','C3')]:
                row=next(r for r in rows if (r['source'],r['arm'],r['seed'])==('electricity',trained,seed))
                plan.append(dict(panel='neso_2025',kind=kind,seed=seed,arm=f'SEL_{trained}_G{gate}',row=row,gate=gate,trained=trained))
    for panel in PANELS:
        for kind in ['standard','shape']:
            for arm in ['F0','PERSISTENCE','SEASONAL']:plan.append(dict(panel=panel,kind=kind,seed=0,arm=arm,row=None,gate=None,trained=arm))
    assert len(plan)==114
    for j in plan:j['id']='__'.join(str(j[k]) for k in ['panel','kind','arm','seed'])
    save(OUT/'PLAN.json',plan)
    hashes=dict(read(ext.OUT/'SEAL.json')['hashes'])
    # Preserve all published parent results, and the actual input arrays behind symlinks.
    for base in [PRIOR,ext.OUT]:
        for p in base.rglob('*'):
            if p.is_file():hashes[str(p.relative_to(ROOT))]=sha(p)
    for panel in PANELS:
        for kind in ['standard','shape']:
            for p in panel_path(panel,kind).glob('*'):
                if p.is_file():hashes[str(p.relative_to(ROOT))]=sha(p)
    for j in plan:
        if j['row']:hashes[j['row']['checkpoint']]=j['row']['sha256']
    for p in list(EXP.glob('*.py'))+[OUT/'PROTOCOL.md',OUT/'PLAN.json']:hashes[str(p.relative_to(ROOT))]=sha(p)
    save(OUT/'SEAL.json',dict(at=time.time(),hashes=hashes,logical_views_cap=114,new_fits_cap=0,optimizer_updates_cap=0,posthoc_development_diagnostic=True))
    print('SEALED',len(plan),flush=True)

def load(j):
    if j['arm']=='F0':
        from experiments.outlier_signal_peft_v1_20260917.common import build
        m=build('FROZEN_RAW',0)
    else:m=load_model(j['row'])
    m.requires_grad_(False);return m

def run():
    prepare();check_seal();old.setup();watch=Watch();manifest=read(OUT/'PREDICTIONS.json') if (OUT/'PREDICTIONS.json').exists() else {};checks=read(OUT/'MODEL_CHECKS.json') if (OUT/'MODEL_CHECKS.json').exists() else {}
    oldpreds=list(read(PRIOR/'PREDICTIONS_MANIFEST.json').values())+list(read(ext.OUT/'PREDICTIONS.json').values())
    try:
        watch.boundary(startup=True)
        for j in read(OUT/'PLAN.json'):
            key=j['id']
            if key in manifest:assert sha(ROOT/manifest[key]['path'])==manifest[key]['sha256'];continue
            x,s=inputs(j['panel'],j['kind']);alias=None
            if j['row'] and j['gate']==j['trained']:
                alias=next((r for r in oldpreds if r['panel']==j['panel'] and r['kind']==j['kind'] and r['arm']==j['trained'] and r['seed']==j['seed'] and r.get('checkpoint_sha256')==j['row']['sha256']),None)
            record={k:j[k] for k in ['panel','kind','arm','seed','gate','trained']};t=time.perf_counter()
            if alias:
                assert sha(ROOT/alias['path'])==alias['sha256'];record.update(path=alias['path'],sha256=alias['sha256'],reused=True,reuse_reason='same input/weight/gate/seed; parent verified')
                checks[key]=dict(reused_parent_verified=True,optimizer_updates=0)
            elif j['arm'] in ['PERSISTENCE','SEASONAL']:
                a,b=naive(np.asarray(x),read(ext.OUT/'DATA_MANIFEST.json')[j['panel']]['period']);p=np.repeat((a if j['arm']=='PERSISTENCE' else b)[:,None,:],9,axis=1).copy()
                checks[key]=dict(point_baseline=True,observed_only=True,optimizer_updates=0)
            else:
                watch.boundary();m=load(j);before=tensor_hash(m.state_dict());assert not any(p.requires_grad for p in m.parameters())
                if j['row']:
                    xx,ss=inputs('electricity','standard');ref=predict(m,xx[:32],ss[:32],32,watch)
                    matching=next(r for r in oldpreds if r['panel']=='electricity' and r['kind']=='standard' and r['arm']==j['trained'] and r['seed']==j['seed'] and r.get('checkpoint_sha256')==j['row']['sha256'])
                    assert np.array_equal(ref,np.load(ROOT/matching['path'],mmap_mode='r')[:32]),'PARENT_PARITY'
                    m.arm=j['gate']
                p=predict(m,x,s,32,watch);assert tensor_hash(m.state_dict())==before
                m2=load(j)
                if j['row']:m2.arm=j['gate']
                assert tensor_hash(m2.state_dict())==before
                np.testing.assert_allclose(predict(m2,x[:32],s[:32],32,watch),p[:32],rtol=1e-4,atol=0)
                checks[key]=dict(frozen=True,state_unchanged=True,restore=True,parent_forward_exact=bool(j['row']),optimizer_updates=0,weight_sha256=before)
                del m,m2;old.cleanup()
            if not alias:
                assert np.isfinite(p).all();path=CACHE/'predictions'/f'{key}.npy';path.parent.mkdir(exist_ok=True);np.save(path,p);record.update(path=str(path.relative_to(ROOT)),sha256=sha(path),reused=False);del p
            record.update(seconds=time.perf_counter()-t);manifest[key]=record;save(OUT/'PREDICTIONS.json',manifest);save(OUT/'MODEL_CHECKS.json',checks)
            save(OUT/'status.json',dict(execution='RUNNING',completed_views=len(manifest),cap=114,new_fits=0,optimizer_updates=0));print('VIEW',len(manifest),key,'reuse',bool(alias),flush=True)
        assert len(manifest)==114;check_seal();save(OUT/'ALL_PREDICTIONS_SAVED.json',dict(at=time.time(),manifest_sha256=sha(OUT/'PREDICTIONS.json'),logical_views=114,optimizer_updates=0))
        from .score import score
        score();save(OUT/'status.json',dict(execution='VERIFIED',completed_views=114,new_views=sum(not r['reused'] for r in manifest.values()),reused_views=sum(r['reused'] for r in manifest.values()),new_fits=0,optimizer_updates=0,automatic_successor=False))
    except BaseException as e:
        save(OUT/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));save(OUT/'status.json',dict(execution='RESOURCE_BLOCK' if isinstance(e,ResourceError) else 'EXECUTION_ERROR',completed_views=len(manifest),error=str(e),optimizer_updates=0));raise
    finally:watch.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args();prepare() if a.action=='prepare' else run()
