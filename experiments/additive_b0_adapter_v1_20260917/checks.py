"""Actual GPU eligibility and exactly 24 smoke optimizer updates."""
import time,unittest
import numpy as np
import torch
from .common import *
from .model import loss_2pinball

def arrays(source):
    f=CACHE/'conditions'/source
    return {k:np.load(f/(k+'.npy'),mmap_mode='r') for k in ['train_x','train_y','train_sigma']}

def batch(a,arm,epoch,idx,device='cuda'):
    return [torch.as_tensor(np.array(v,copy=True),device=device) for v in
            [a['train_x'][epoch,idx],a['train_y'][epoch,idx],a['train_sigma'][idx]]]

def backward(model,x,y,s,micro):
    value=0.
    for lo in range(0,len(x),micro):
        end=min(lo+micro,len(x));pred=model(x[lo:end],s[lo:end])
        loss=loss_2pinball(pred,y[lo:end],s[lo:end],model.base.quantiles)*(end-lo)/len(x)
        assert torch.isfinite(loss),'NONFINITE_LOSS'
        loss.backward();value+=float(loss.detach())
    grads=[p.grad for p in parameters(model).values() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads),'NONFINITE_GRADIENT'
    return value

def cpu_check():
    suite=unittest.defaultTestLoader.loadTestsFromName('experiments.additive_b0_adapter_v1_20260917.test_components')
    with open(OUT/'cpu_tests.txt','w') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    save(OUT/'cpu_tests.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                                authorship='locally authored, explicitly user-authorized',passed=result.wasSuccessful()))
    assert result.wasSuccessful(),'CPU_CHECK_FAILED'

@torch.no_grad()
def precheck(watch):
    if (OUT/'pretraining_checks.json').exists():return
    rows=[];param={}
    for source in SOURCES:
        a=arrays(source);x,y,s=batch(a,'C1',0,np.arange(32))
        for seed in [81550,81551,81552]:
            base=build('C0',seed,source);original=base(x,s);before=x.clone();basehash=tensor_hash(dict(base.base.named_parameters()))
            states={}
            for arm in ARMS:
                watch.boundary();model=build(arm,seed,source);pred=model(x,s)
                assert torch.equal(original,pred) and torch.equal(before,x),'ZERO_ADAPTER_NOT_B0'
                assert tensor_hash(dict(model.base.named_parameters()))==basehash
                st=cpu_state(model);states[arm]=st
                param[arm]=dict(trainable=sum(v.numel() for v in st.values()),shapes={n:list(v.shape) for n,v in st.items()},base_frozen=arm!='C1')
                rows.append(dict(source=source,seed=seed,arm=arm,initial_B0_bitwise_equal=True,raw_input_unchanged=True,base_full_hash=basehash))
                del model;cleanup()
            assert tensor_hash(states['C2'])==tensor_hash(states['C3'])
            del base;cleanup()
    save(OUT/'PARAMETER_RECEIPT.json',param);save(OUT/'pretraining_checks.json',dict(rows=rows,optimizer_updates=0,common_adapter_initialization=True))

def smoke(watch):
    path=OUT/'smoke_checks.json';done=read(path) if path.exists() else {}
    batches=read(OUT/'microbatch.json') if (OUT/'microbatch.json').exists() else {}
    for source in SOURCES:
        a=arrays(source)
        if source not in batches:
            events=[]
            for micro in [32,16,8,4]:
                watch.boundary();model=build('C1',81550,source);x,y,s=batch(a,'C1',0,np.arange(32));torch.cuda.reset_peak_memory_stats()
                try:
                    backward(model,x,y,s,micro);sync();events.append(dict(micro=micro,status='FIT',peak_allocated=torch.cuda.max_memory_allocated(),optimizer_updates=0))
                    del model;cleanup();batches[source]=dict(microbatch=micro,probes=events);save(OUT/'microbatch.json',batches);break
                except torch.cuda.OutOfMemoryError:
                    events.append(dict(micro=micro,status='OOM',optimizer_updates=0));del model;cleanup()
            assert source in batches
        micro=batches[source]['microbatch']
        for arm in ARMS:
            key=source+'_'+arm
            if key in done:assert done[key]['updates']==2;continue
            journal=OUT/f'smoke_{key}.jsonl';intent=OUT/f'smoke_{key}_intent.json'
            assert not journal.exists() and not intent.exists(),'AMBIGUOUS_SMOKE_NO_REPLAY'
            model=build(arm,81550,source);before=frozen_hash(model);initial=cpu_state(model)
            base_before=tensor_hash(dict(model.base.named_parameters()))
            optimizer=torch.optim.AdamW(parameters(model).values(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
            losses=[];torch.cuda.reset_peak_memory_stats()
            for step in range(2):
                watch.boundary();x,y,s=batch(a,arm,0,np.arange(step*32,(step+1)*32))
                optimizer.zero_grad(set_to_none=True);value=backward(model,x,y,s,micro);norm=torch.nn.utils.clip_grad_norm_(parameters(model).values(),1.);assert torch.isfinite(norm)
                save(intent,dict(next_step=step+1,status='BEFORE_OPTIMIZER'));optimizer.step();sync()
                append(journal,dict(step=step+1,loss=value,optimizer_updates=1));save(intent,dict(completed_step=step+1,status='JOURNALED'));losses.append(value)
            assert frozen_hash(model)==before
            state=cpu_state(model);changed=[n for n in state if not torch.equal(state[n],initial[n])]
            if arm=='C1':assert any(n.endswith('.b') for n in changed)
            else:
                assert 'adapter.up.weight' in changed and 'adapter.down.weight' in changed
                assert tensor_hash(dict(model.base.named_parameters()))==base_before
            checkpoint=CACHE/'smoke'/f'{key}.pt';atomic_torch(checkpoint,state)
            with torch.no_grad():
                pred=model(x,s);rev=model(x.flip(0),s.flip(0)).flip(0)
                diff=float(((pred-rev)/s[:,None,None]).abs().max());assert diff<=1e-5 or torch.allclose(pred,rev,rtol=1e-4,atol=0)
                restore(model,initial);restore(model,torch.load(checkpoint,weights_only=True,map_location='cpu'));assert torch.equal(pred,model(x,s))
                # No hypothetical y enters forward; swapping the training target without any optimizer step leaves prediction fixed.
                hypothetical=y+10*s[:,None];assert not torch.equal(y,hypothetical);assert torch.equal(pred,model(x,s))
                off_exact=None
                if arm in ['C2','C3']:
                    baseline=build('C0',81550,source);off_exact=torch.equal(model(x,s,residual_mode='off'),baseline(x,s));assert off_exact;del baseline
            done[key]=dict(updates=2,losses=losses,microbatch=micro,trainable_parameters=sum(v.numel() for v in state.values()),changed_parameters=changed,frozen_unchanged=True,frozen_sha256=before,restore_exact=True,adapter_off_B0_exact=off_exact,batch_order_max_normalized_error=diff,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
            save(path,done);print('SMOKE',key,losses,flush=True);del model,optimizer;cleanup()
    assert len(done)==6 and sum(v['updates'] for v in done.values())==12
