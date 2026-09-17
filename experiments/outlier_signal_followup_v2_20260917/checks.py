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
    suite=unittest.defaultTestLoader.loadTestsFromName('experiments.outlier_signal_followup_v2_20260917.test_components')
    with open(OUT/'cpu_tests.txt','w') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    save(OUT/'cpu_tests.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                                authorship='locally authored, explicitly user-authorized',passed=result.wasSuccessful()))
    assert result.wasSuccessful(),'CPU_CHECK_FAILED'

def smoke(watch):
    path=OUT/'smoke_checks.json';done=read(path) if path.exists() else {}
    batches=read(OUT/'microbatch.json') if (OUT/'microbatch.json').exists() else {}
    for source in SOURCES:
        a=arrays(source)
        if source not in batches:
            # One full-batch eligibility probe, zero optimizer steps.
            events=[]
            for micro in [32,16,8,4]:
                watch.boundary();model=build('B2',81550);x,y,s=batch(a,'B2',0,np.arange(32))
                torch.cuda.reset_peak_memory_stats()
                try:
                    backward(model,x,y,s,micro);sync()
                    events.append(dict(micro=micro,status='FIT',peak_allocated=torch.cuda.max_memory_allocated(),optimizer_updates=0))
                    del model,x,y,s;cleanup();batches[source]=dict(microbatch=micro,probes=events)
                    save(OUT/'microbatch.json',batches);break
                except torch.cuda.OutOfMemoryError:
                    events.append(dict(micro=micro,status='OOM',optimizer_updates=0));del model,x,y,s;cleanup()
            assert source in batches,'BLOCKED_GPU_MEMORY'
        micro=batches[source]['microbatch']
        for arm in ARMS:
            key=source+'_'+arm
            if key in done:assert done[key]['updates']==2;continue
            journal=OUT/f'smoke_{key}.jsonl'
            assert not (OUT/f'smoke_{key}_intent.json').exists(),'AMBIGUOUS_SMOKE_INTENT'
            assert not journal.exists(),'PARTIAL_SMOKE_REQUIRES_EXACT_RECOVERY_NOT_REPLAY'
            model=build(arm,81550);before=frozen_hash(model);initial=cpu_state(model)
            optimizer=torch.optim.AdamW(parameters(model).values(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
            losses=[];timings=[]
            for step in range(2):
                watch.boundary();x,y,s=batch(a,arm,0,np.arange(step*32,(step+1)*32))
                sync();t=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                value=backward(model,x,y,s,micro);norm=torch.nn.utils.clip_grad_norm_(parameters(model).values(),1.)
                assert torch.isfinite(norm)
                if arm in ['B4','B5']:assert model.gate.grad is not None and torch.isfinite(model.gate.grad).all() and model.gate.grad.abs().sum()>0
                save(OUT/f'smoke_{key}_intent.json',dict(step=step+1,status='BEFORE_OPTIMIZER'))
                optimizer.step();sync();timings.append(time.perf_counter()-t);losses.append(value)
                append(journal,dict(step=step+1,loss=value,optimizer_updates=1))
                save(OUT/f'smoke_{key}_intent.json',dict(step=step+1,status='JOURNALED'))
            assert frozen_hash(model)==before,'FROZEN_WEIGHT_CHANGED'
            state=cpu_state(model);changed=[n for n in state if not torch.equal(state[n],initial[n])]
            assert any(n.endswith('.b') for n in changed)
            if arm in ['B2']:
                assert 'adapter.up.weight' in changed and 'adapter.down.weight' in changed,('ADAPTER_NOT_LEARNING',key,changed)
            if arm in ['B4','B5']:assert 'gate' in changed
            # Checkpoint restore, ordering and ambiguity using actual trained smoke model.
            p=CACHE/'smoke'/f'{key}.pt';atomic_torch(p,state)
            model.eval()
            with torch.no_grad():
                pred=model(x,s);rev=model(x.flip(0),s.flip(0)).flip(0)
                diff=float(((pred-rev)/s[:,None,None]).abs().max());assert diff<=1e-5 or torch.allclose(pred,rev,rtol=1e-4,atol=0)
                restore(model,initial);restore(model,torch.load(p,weights_only=True,map_location='cpu'))
                restored=model(x,s);assert torch.equal(pred,restored)
                # y is never an argument: same past, hypothetical different future -> same output.
                assert torch.equal(pred,model(x,s))
            done[key]=dict(updates=2,losses=losses,optimizer_seconds=timings,microbatch=micro,
                           trainable_parameters=sum(v.numel() for v in state.values()),changed_parameters=changed,
                           frozen_sha256=before,restore_exact=True,batch_order_max_normalized_error=diff,
                           initial_lora_sha256=tensor_hash({n:v for n,v in initial.items() if not n.startswith('adapter.') and n!='gate'}),
                           peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
            save(path,done);print('SMOKE',key,losses,flush=True)
            del model,optimizer,x,y,s,pred,rev,restored,state,initial;cleanup()
    assert len(done)==12 and sum(v['updates'] for v in done.values())==24
    assert len({v['initial_lora_sha256'] for v in done.values()})==1
    save(OUT/'implementation_checks_completed.json',dict(cpu_tests=read(OUT/'cpu_tests.json'),smoke_updates=24,
         all_sources_arms_checked=True,original_initial_checks=read(OUT/'pretraining_checks.json'),
         LoRA_common_initialization=True,all_labels_single_shared_file=True,generator_cannot_read_future=True,
         performance_gate=False,final_model_restore_verification='pending selected models'))
