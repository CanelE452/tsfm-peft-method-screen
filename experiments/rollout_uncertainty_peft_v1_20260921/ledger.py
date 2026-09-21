import os
import random
import shutil
import time
import numpy as np
import torch
from common import *

def save_checkpoint(path,model,optimizer,step,extra=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    state={'learned':model.learned(),'optimizer':optimizer.state_dict(),'step':step,'schedule_index':step,
           'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()},'extra':extra or {}}
    tmp=path.with_suffix('.tmp'); torch.save(state,tmp); os.replace(tmp,path)

def restore(path,model,optimizer=None):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model.load_learned(state['learned'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        for v in optimizer.state.values():
            for k,x in v.items():
                if isinstance(x,torch.Tensor) and k!='step': v[k]=x.to('cuda')
        torch.set_rng_state(state['rng']['torch']); torch.cuda.set_rng_state_all(state['rng']['cuda'])
        np.random.set_state(state['rng']['numpy']); random.setstate(state['rng']['python'])
    return state

class Ledger:
    def __init__(self,fit_id,kind,model,optimizer,extra=None):
        self.fit_id=fit_id; self.kind=kind; self.model=model; self.optimizer=optimizer
        self.folder=CACHE/'fits'/fit_id; self.folder.mkdir(parents=True,exist_ok=True)
        self.latest=self.folder/'latest.pt'; self.intent=self.folder/'pending.json'
        self.extra=extra or {}
        log=RESULTS/'UPDATE_LEDGER.jsonl'
        entries=[json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        existing={r['step'] for r in entries if r['fit_id']==fit_id and r['status'].startswith('committed')}
        if existing and not self.latest.exists():
            raise RuntimeError('BLOCKED_IMPLEMENTATION: missing checkpoint for spent updates '+fit_id)
        if self.latest.exists():
            self.step=restore(self.latest,model,optimizer)['step']
        else:
            self.step=0; save_checkpoint(self.latest,model,optimizer,0,self.extra)
        if self.intent.exists():
            pending=read_json(self.intent)
            if self.step!=pending['step']:
                raise RuntimeError('BLOCKED_IMPLEMENTATION: ambiguous spent optimizer update; do not repeat '+fit_id)
            self._append({**pending,'status':'committed_recovered','checkpoint_sha256':sha(self.latest)})
            self.intent.unlink()
            existing.add(pending['step'])
        assert self.step==max(existing,default=0), 'Checkpoint/ledger mismatch; cannot repeat updates'
        assert existing==set(range(1,self.step+1)), 'Noncontiguous committed ledger'

    def _append(self,record):
        log=RESULTS/'UPDATE_LEDGER.jsonl'
        with log.open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,default=json_default,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())

    def update(self,fn):
        assert self.step < (2 if self.kind=='smoke' else 512)
        log=RESULTS/'UPDATE_LEDGER.jsonl'
        entries=[json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        completed={(r['fit_id'],r['step']) for r in entries if r['kind']==self.kind and r['status'].startswith('committed')}
        assert (self.fit_id,self.step+1) not in completed, 'Refusing duplicate optimizer update'
        assert len(completed)<(12 if self.kind=='smoke' else 12288)
        info={'fit_id':self.fit_id,'kind':self.kind,'step':self.step+1,'status':'intent','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        save_json(self.intent,info)
        start=time.perf_counter(); metrics=fn(); torch.cuda.synchronize()
        self.step+=1
        save_checkpoint(self.latest,self.model,self.optimizer,self.step,self.extra)
        info.update(status='committed',seconds=time.perf_counter()-start,checkpoint_sha256=sha(self.latest),**metrics)
        self._append(info); self.intent.unlink()
        return info

    def checkpoint(self):
        target=self.folder/f'step_{self.step:04d}.pt'
        if not target.exists(): shutil.copyfile(self.latest,target)
        return target
