import time
import numpy as np
import torch
from common import *
from model import loss, tensor_hash
from ledger import Ledger
import data


def torch_batch(d,pairs,datagroup,seed=None,step=0,missing=False):
    x,y,sigma,_=d.batch(pairs,group=datagroup,role='TRAIN')
    if missing:
        for j,(sid,origin) in enumerate(pairs):
            if j%4!=0:
                observed=data.mask_for(str(sid),int(origin),'TRAIN','IID48',repeat=step,seed=seed)
                x[j,~observed]=np.nan
    return tuple(torch.as_tensor(v,device='cuda',dtype=torch.float32) for v in [x,y,sigma])


def fit_model(model,d,pairs,key,candidate,datagroup,lr,seed,ridge=False,missing=False,phase='main',timer=None):
    steps=len(pairs);ledger=Ledger();ledger.start(key,candidate,phase,steps)
    folder=CACHE/'fits'/key;folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'complete.json').exists()
    np.savez_compressed(folder/'schedule.npz',pairs=np.asarray(pairs).astype(str))
    initial=model.learned();frozen=model.frozen_hash()
    parameters=[p for p in model.parameters() if p.requires_grad]
    assert parameters
    optimizer=torch.optim.AdamW(parameters,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    checkpoints={0:initial};losses=[]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    for step,pair in enumerate(pairs,1):
        if timer is not None:timer.check()
        model.eval();optimizer.zero_grad(set_to_none=True)
        x,y,sigma=torch_batch(d,pair,datagroup,seed,step,missing)
        objective=loss(model(x),y,sigma)
        if ridge:objective=objective+.01*torch.cat([p.flatten() for p in parameters]).square().mean()
        assert torch.isfinite(objective), 'IMPLEMENTATION_FAILURE: nonfinite loss'
        objective.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        assert torch.isfinite(norm) and norm>0, 'IMPLEMENTATION_FAILURE: no finite learning gradient'
        ledger.reserve(key);optimizer.step();ledger.completed(key)
        losses.append(dict(step=step,loss=float(objective.detach()),grad_norm=float(norm)))
        if step in [steps//2,steps]:
            checkpoints[step]=model.learned()
            torch.save(dict(learned=checkpoints[step],step=step,optimizer=optimizer.state_dict()),folder/f'step_{step}.pt')
        if step%128==0:event('fit_progress',key=key,step=step,total=steps)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    assert frozen==model.frozen_hash(), 'IMPLEMENTATION_FAILURE: frozen weights changed'
    assert tensor_hash(initial.items())!=tensor_hash(model.learned().items())
    ledger.finish(key)
    report=dict(key=key,candidate=candidate,phase=phase,steps=steps,seconds=seconds,
                trainable_parameters=sum(p.numel() for p in parameters),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),frozen_unchanged=True,frozen_sha256=frozen,seed=seed,
                schedule_sha256=sha(folder/'schedule.npz'),initial_sha256=tensor_hash(initial.items()),
                final_sha256=tensor_hash(model.learned().items()),losses=losses)
    save(RESULTS/'fits'/f'{key}.json',report);save(folder/'complete.json',report)
    torch.save(checkpoints,folder/'checkpoints.pt')
    return checkpoints,report


def context_packet(d,group,part,condition='CLEAN',repeat=0):
    pairs=[(sid,int(o)) for sid in d.group_ids(group) for o in d.origins(group,part,24)]
    x,y,sigma,masks=d.batch(pairs,group=group,role=f'{group}_{part}',condition=condition,repeat=repeat)
    return dict(x=x,y=y,sigma=sigma,masks=masks,pairs=np.asarray(pairs).astype(str))


def predict_packet(model,packet,path):
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    q=model.predict(packet['x']);torch.cuda.synchronize()
    info=dict(seconds=time.perf_counter()-start,examples=len(q),peak_allocated_bytes=torch.cuda.max_memory_allocated())
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,q=q,y=packet['y'],sigma=packet['sigma'],pairs=packet['pairs'],masks=packet['masks'])
    save(path.with_suffix('.json'),dict(**info,sha256=sha(path)))
    return q,info
