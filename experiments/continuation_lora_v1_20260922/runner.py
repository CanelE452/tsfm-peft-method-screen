import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import gc
import time
import traceback
import warnings
import numpy as np
import torch
from common import *
from model import BranchModel, setup, tensor_hash, update, load_direct, direct
from preflight import gpu_batch
from metrics import point_loss, calibrate
import data

CHECKPOINTS=(0,128,256,512)

def targets(d,role):
    return np.stack([d['values'][o:o+128].T for o in d['origins'][role]])

@torch.no_grad()
def predict(m,d,role,key,direct_model=False,save_components=True):
    if role=='TEST':
        seal=read(RESULTS/'SELECTION_SEAL.json')
        assert seal['selections_sha']==sha(RESULTS/'SELECTIONS.json')
        assert seal['calibration_sha']==sha(RESULTS/'CALIBRATION.json')
    folder=CACHE/'predictions'/role
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{key}.npz'
    receipt=folder/f'{key}.json'
    if receipt.exists():
        info=read(receipt)
        assert info['sha256']==sha(path)
        with np.load(path) as z: return {n:z[n] for n in z.files}
    xs=np.stack([d['values'][o-512:o].T for o in d['origins'][role]]).reshape(-1,512)
    qs=[]; cs=[]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); start=time.perf_counter()
    for j in range(0,len(xs),8):
        x=torch.as_tensor(xs[j:j+8],device='cuda')
        if direct_model: q=direct(m,x)
        else:
            q,c=m.forecast(x)
            if save_components: cs.append(c.cpu().numpy())
        assert torch.isfinite(q).all()
        qs.append(q.cpu().numpy())
    torch.cuda.synchronize()
    info=dict(key=key,role=role,seconds=time.perf_counter()-start,examples=len(xs),batch=8,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    out={'q':np.concatenate(qs).reshape(len(d['origins'][role]),len(d['ids']),128,9)}
    if cs: out['components']=np.concatenate(cs).reshape(len(d['origins'][role]),len(d['ids']),9,64,9)
    np.savez_compressed(path,**out)
    save(receipt,dict(**info,path=str(path.relative_to(ROOT)),sha256=sha(path),shapes={k:list(v.shape) for k,v in out.items()}))
    event('predictions',**info)
    return out

def atomic_torch(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    torch.save(obj,tmp)
    os.replace(tmp,path)

def fit(arm,seed,d):
    key=f'{arm}_s{seed}'
    receipt=RESULTS/'fits'/f'{key}.json'
    if receipt.exists(): return read(receipt)
    folder=CACHE/'fits'/key
    folder.mkdir(parents=True,exist_ok=True)
    latest=folder/'latest.pt'
    pending=folder/'pending.json'
    m=BranchModel(seed,mode=arm)
    frozen=m.frozen_hash()
    initial=tensor_hash(m.learned().items())
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    packet=data.schedule(seed)
    packet_sha=__import__('hashlib').sha256(packet.tobytes()).hexdigest()
    vals={}; step=0; training_seconds=0.; peak=0; records=[]
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        assert state['source_hashes']==source_hashes()
        assert state['initial_hash']==initial and state['schedule_sha']==packet_sha
        m.load_learned(state['learned']); opt.load_state_dict(state['optimizer'])
        step=state['step']; vals=state['validation']; training_seconds=state['training_seconds']; peak=state['peak_allocated_bytes']; records=state['records']
    if pending.exists() and read(pending)['step']>step:
        raise RuntimeError('RESOURCE_BLOCK: in-flight update uncommitted, do not replay automatically')
    def state():
        return dict(step=step,learned=m.learned(),optimizer=opt.state_dict(),source_hashes=source_hashes(),
                    initial_hash=initial,schedule_sha=packet_sha,validation=vals,training_seconds=training_seconds,
                    peak_allocated_bytes=peak,records=records)
    def checkpoint():
        p=folder/f'step_{step:04d}.pt'
        if not p.exists(): atomic_torch(p,dict(step=step,learned=m.learned()))
        if str(step) not in vals:
            out=predict(m,d,'VALIDATION',f'{key}_step{step}',save_components=False)
            score=float(point_loss(targets(d,'VALIDATION')[:,:,64:],np.sort(out['q'][:,:,64:],axis=-1),d['sigma']).mean())
            vals[str(step)]=dict(score=score,path=str(p.relative_to(ROOT)),sha256=sha(p))
            event('validation',key=key,step=step,score=score)
        atomic_torch(latest,state())
    if step in CHECKPOINTS:
        checkpoint()
    while step<512:
        x,y,sigma=gpu_batch(d,packet[step])
        save(pending,dict(step=step+1,arm=arm,seed=seed))
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t=time.perf_counter()
        record=update(m,opt,x,y,sigma,arm)
        torch.cuda.synchronize()
        training_seconds+=time.perf_counter()-t
        peak=max(peak,torch.cuda.max_memory_allocated())
        step+=1
        records.append(dict(step=step,**record))
        atomic_torch(latest,state())
        save(RESULTS/'progress.json',dict(stage='training',key=key,step=step,total=512,training_seconds=training_seconds))
        if step%32==0: event('training',key=key,step=step,loss=record['loss'],training_seconds=training_seconds)
        if step in CHECKPOINTS: checkpoint()
    assert frozen==m.frozen_hash()
    selected=min(vals,key=lambda s:(vals[s]['score'],int(s)))
    row=dict(key=key,arm=arm,seed=seed,updates=512,initial_hash=initial,schedule_sha=packet_sha,
             validation=vals,selected_step=int(selected),checkpoint=vals[selected],
             fixed512=vals['512'],trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
             frozen_unchanged=True,training_seconds=training_seconds,peak_allocated_bytes=peak,
             batch_conditional_forward_calls=1024,batch_backward_calls=1024 if arm=='SHARED' else 512,
             series_context_forward_evaluations=20480,series_context_backward_evaluations=20480 if arm=='SHARED' else 18432,
             loss_records=records)
    save(receipt,row)
    del m,opt
    gc.collect(); torch.cuda.empty_cache()
    return row

def training():
    assert read(RESULTS/'PREFLIGHT.json')['status']=='PASS'
    assert read(RESULTS/'ATTENUATION_DIAGNOSTIC.json')['complete']
    d=data.load()
    seal=RESULTS/'TRAINING_SEAL.json'
    payload=dict(protocol_sha=sha(EXP/'PROTOCOL.md'),source_hashes=source_hashes(),data_sha=sha(RESULTS/'DATA_AUDIT.json'),
                 attenuation_sha=sha(RESULTS/'ATTENUATION_DIAGNOSTIC.json'),
                 seeds=SEEDS,arms=ARMS,main_cap=2048,fit_cap=4,smoke_cap=4)
    if seal.exists(): assert read(seal)==json_roundtrip(payload)
    else: save(seal,payload)
    fits=[]
    for seed in SEEDS:
        for arm in ARMS: fits.append(fit(arm,seed,d))
    for seed in SEEDS:
        pair=[r for r in fits if r['seed']==seed]
        assert len({r['initial_hash'] for r in pair})==1 and len({r['schedule_sha'] for r in pair})==1
    assert sum(r['updates'] for r in fits)==2048 and len(fits)==4
    save(RESULTS/'SELECTIONS.json',fits)
    save(RESULTS/'OPTIMIZER_LEDGER.json',dict(status='COMPLETE',fits=4,main_updates=2048,smoke_updates=4,
         main_conditional_batch_forwards=4096,main_batch_backwards=3072,
         main_series_context_forward_evaluations=81920,main_series_context_backward_evaluations=77824,
         selections_sha=sha(RESULTS/'SELECTIONS.json')))

def json_roundtrip(obj):
    import json
    return json.loads(json.dumps(obj))

def specs():
    yield 'F0_NATIVE',None,False,False
    yield 'CHRONOS2_DIRECT',None,True,False
    for row in read(RESULTS/'SELECTIONS.json'):
        yield row['key'],row,False,False
        yield row['key']+'_fixed512',row,False,True

def open_model(row,is_direct,fixed):
    if is_direct: return load_direct()
    if row is None: return BranchModel(lora=False)
    m=BranchModel(row['seed'],mode=row['arm'])
    rec=row['fixed512'] if fixed else row['checkpoint']
    assert sha(ROOT/rec['path'])==rec['sha256']
    m.load_learned(torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)['learned'])
    return m

def calibration():
    d=data.load(); params={}
    for key,row,is_direct,fixed in specs():
        if fixed: continue
        m=open_model(row,is_direct,fixed)
        out=predict(m,d,'CALIBRATION',key,is_direct,False)
        params[key]=calibrate(targets(d,'CALIBRATION'),np.sort(out['q'],axis=-1),d['sigma'])
        if row is not None and row['arm']=='CONTINUATION':
            # Identical first-block output must have the identical first-block calibration.
            assert params[key][0] == params['F0_NATIVE'][0]
        del m
        gc.collect(); torch.cuda.empty_cache()
    save(RESULTS/'CALIBRATION.json',params)
    save(RESULTS/'SELECTION_SEAL.json',dict(selections_sha=sha(RESULTS/'SELECTIONS.json'),calibration_sha=sha(RESULTS/'CALIBRATION.json'),
          training_seal_sha=sha(RESULTS/'TRAINING_SEAL.json'),test_scored=False))

def test_predictions():
    d=data.load()
    for key,row,is_direct,fixed in specs():
        m=open_model(row,is_direct,fixed)
        predict(m,d,'TEST',key,is_direct,True)
        del m
        gc.collect(); torch.cuda.empty_cache()
    files=[read(p) for p in sorted((CACHE/'predictions').rglob('*.json'))]
    assert sum(r['role']=='TEST' for r in files)==10
    save(RESULTS/'PREDICTIONS_MANIFEST.json',dict(selection_seal_sha=sha(RESULTS/'SELECTION_SEAL.json'),test_scored=False,files=files))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['train','calibrate','predict','all'])
    args=p.parse_args()
    setup(); warnings.filterwarnings('ignore',message='We recommend keeping prediction length')
    try:
        if args.stage in ('all','train'): training()
        if args.stage in ('all','calibrate'): calibration()
        if args.stage in ('all','predict'): test_predictions()
    except Exception as exc:
        kind='RESOURCE_BLOCK' if isinstance(exc,torch.cuda.OutOfMemoryError) or 'RESOURCE_BLOCK' in str(exc) else 'IMPLEMENTATION_FAILURE'
        save(RESULTS/'FAILURE.json',dict(kind=kind,error=repr(exc),traceback=traceback.format_exc()))
        event('failure',classification=kind,error=repr(exc))
        raise
