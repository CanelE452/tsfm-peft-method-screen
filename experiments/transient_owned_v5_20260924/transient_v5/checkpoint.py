"""Exact-key compact PyTorch checkpoint; the training model is the save source.

All trainable tensors and fixed response buffers are stored. Original pretrained
weights are referenced by pinned local snapshot and checked by a full frozen-
parameter hash. No pickle model object, permissive strict=False load, or separate
untrained state generator is involved. This is inference roundtrip, NOT resume.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import torch
from .util import require,write_json,read_json,file_hash,state_hash
from .model import TransientModel,ModelSpec
from .backend import load_base
from .fixture import serialize_tasks,restore_tasks


def reference(model,tasks,output):
    d=model.last_debug
    return {'predictions':{str(k):v.detach().cpu().clone() for k,v in output['predictions'].items()},
            'states':{str(k):v for k,v in d['states'].items()},'c':d['c'],
            'valid':d['valid'],'rows':d['rows'], 'native_loss':output['native_loss'].detach().cpu(),
            'corrected_loss':output['loss'].detach().cpu()}


def save_trained(model,backend,tasks,output,path: Path,steps: int):
    require(not path.exists(),'Checkpoint destination already exists')
    path.mkdir(parents=True)
    compact=model.compact_state()
    torch.save(compact,path/'owned_state.pt')
    torch.save(serialize_tasks(tasks),path/'fixture.pt')
    torch.save(reference(model,tasks,output),path/'reference.pt')
    metadata={'spec':asdict(model.spec),'backend':backend,'steps_from_optimizer':steps,
              'frozen_sha256':model.frozen_state_hash(),'compact_sha256':state_hash(compact),
              'scope':model.scope,'inventory':model.inventory(),'training_resume_claimed':False,
              'files':{n:file_hash(path/n) for n in ('owned_state.pt','fixture.pt','reference.pt')}}
    write_json(path/'metadata.json',metadata)
    return metadata


def load_trained(path: Path):
    metadata=read_json(path/'metadata.json')
    for rel,sha in metadata['files'].items():
        require(file_hash(path/rel)==sha,f'Checkpoint file corruption: {rel}')
    base,tt,gt=load_base(metadata['backend'])
    spec=metadata['spec'];spec['tau_init']=tuple(spec['tau_init'])
    model=TransientModel(base,ModelSpec(**spec),tt,gt)
    require(model.frozen_state_hash()==metadata['frozen_sha256'],'Frozen pretrained weights changed')
    require(model.scope==metadata['scope'],'Module scope changed')
    require(model.inventory()==metadata['inventory'],'Parameter ownership changed')
    state=torch.load(path/'owned_state.pt',map_location='cpu',weights_only=True)
    model.load_compact(state)
    require(state_hash(model.compact_state())==metadata['compact_sha256'],'Restored state mismatch')
    tasks=restore_tasks(torch.load(path/'fixture.pt',map_location='cpu',weights_only=True))
    return model,tasks,metadata


def compare_reference(actual,expected):
    results={}
    for group in ('predictions','states'):
        require(set(actual[group])==set(expected[group]),f'Roundtrip {group} key mismatch')
        for key in actual[group]:
            a,b=actual[group][key],expected[group][key]
            diff=float((a-b).abs().max())
            require(a.shape==b.shape and torch.allclose(a,b,atol=1e-5,rtol=1e-4),f'Roundtrip {group}/{key} mismatch')
            results[group+'/'+key]={'exact':torch.equal(a,b),'max_abs':diff}
    for key in ('c','valid'):
        a,b=actual[key],expected[key]
        require((a is None)==(b is None),f'Roundtrip {key} missing')
        if a is not None:
            require(torch.equal(a,b),f'Roundtrip {key} changed')
            results[key]={'exact':True}
    require(actual['rows']==expected['rows'],'Roundtrip row ownership mismatch')
    for key in ('native_loss','corrected_loss'):
        require(torch.allclose(actual[key],expected[key],atol=1e-6,rtol=1e-5),f'Roundtrip {key} changed')
    return results
