"""Actual GPU eligibility and exactly 36 smoke optimizer updates."""
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
    from . import reference_checks
    suite=unittest.defaultTestLoader.loadTestsFromModule(reference_checks)
    import io
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (OUT/'reference_checks.txt').write_text(stream.getvalue());assert r.wasSuccessful() and r.testsRun==20
    save(OUT/'CPU_REFERENCE_CHECKS.json',dict(tests=r.testsRun,status='PASS',authorship='locally authored from sole TXT; attachment reference_checks.py not found; not recovered original tests',actual_model_run=False))

@torch.no_grad()
def precheck(watch):
    from experiments.additive_b0_adapter_v1_20260917.common import build as previous_build
    rows=[]
    for source in SOURCES:
        seed=85550 if source=='ettm2' else 81550;proxy=source=='ettm2'
        a=arrays(source);x,y,s=batch(a,'C1',0,np.arange(32))
        base=build('C0',seed,source,smoke_proxy=proxy);pred=base(x,s);initial={};xcopy=x.clone();ones_outputs={}
        for arm in CORE+(CONTROLS if source!='ettm2' else []):
            watch.boundary();m=build(arm,seed,source,smoke_proxy=proxy);assert torch.equal(pred,m(x,s));initial[arm]=cpu_state(m)
            rows.append(dict(source=source,arm=arm,initial_B0_equal=True,ETTm2_untrained_wiring_proxy=proxy,actual_trained_B0_checked_before_each_main_fit=True))
            assert torch.equal(x,xcopy)
            if m.adapter is not None:
                m.adapter.up.weight.fill_(.01);m.adapter.up.bias.fill_(.005)
                ones_outputs[arm]=m(torch.zeros_like(x),s).cpu()
                assert torch.equal(m(x,s,residual_mode='off'),pred)
                # Restore is not needed: this optimizer-zero probe model is discarded.
            del m;cleanup()
        for arm in ['C3']+(CONTROLS if source!='ettm2' else []):
            assert tensor_hash(initial['C2'])==tensor_hash(initial[arm])
            assert torch.equal(ones_outputs['C2'],ones_outputs[arm]),'GATE_ONE_NOT_C2'
        del base;cleanup()
        if source!='ettm2':
            for seed in [81551,81552]:
                row=next(r for r in read(OLD/'MODEL_SELECTION.json') if r['source']==source and r['arm']=='C3' and r['seed']==seed)
                m=build('C3',seed,source);prev=previous_build('C3',seed,source)
                weights=torch.load(ROOT/row['checkpoint'],weights_only=True,map_location='cpu');restore(m,weights);restore(prev,weights)
                assert torch.equal(m(x,s),prev(x,s));rows.append(dict(source=source,seed=seed,trained_C3_previous_forward_exact=True))
                del m,prev;cleanup()
    save(OUT/'INITIAL_AND_REUSED_PARITY.json',rows)

def smoke(watch):
    path=OUT/'smoke_checks.json';done=read(path) if path.exists() else {}
    batches=read(OUT/'microbatch.json') if (OUT/'microbatch.json').exists() else {}
    for source in SOURCES:
        a=arrays(source)
        if source not in batches:
            events=[]
            for micro in [32,16,8,4]:
                watch.boundary();model=build('C1',85550 if source=='ettm2' else 81550,source,smoke_proxy=source=='ettm2');x,y,s=batch(a,'C1',0,np.arange(32));torch.cuda.reset_peak_memory_stats()
                try:
                    backward(model,x,y,s,micro);sync();events.append(dict(micro=micro,status='FIT',peak_allocated=torch.cuda.max_memory_allocated(),optimizer_updates=0))
                    del model;cleanup();batches[source]=dict(microbatch=micro,probes=events);save(OUT/'microbatch.json',batches);break
                except torch.cuda.OutOfMemoryError:
                    events.append(dict(micro=micro,status='OOM',optimizer_updates=0));del model;cleanup()
            assert source in batches
        micro=batches[source]['microbatch']
        for arm in ['B0']+CORE+(CONTROLS if source!='ettm2' else []):
            key=source+'_'+arm
            if key in done:assert done[key]['updates']==2;continue
            journal=OUT/f'smoke_{key}.jsonl';intent=OUT/f'smoke_{key}_intent.json'
            assert not journal.exists() and not intent.exists(),'AMBIGUOUS_SMOKE_NO_REPLAY'
            model=build(arm,85550 if source=='ettm2' else 81550,source,smoke_proxy=source=='ettm2' and arm!='B0');before=frozen_hash(model);buffers=tensor_hash(dict(model.named_buffers()));initial=cpu_state(model)
            base_before=tensor_hash(dict(model.base.named_parameters()))
            optimizer=torch.optim.AdamW(parameters(model).values(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
            losses=[];torch.cuda.reset_peak_memory_stats()
            for step in range(2):
                watch.boundary();x,y,s=batch(a,arm,0,np.arange(step*32,(step+1)*32))
                optimizer.zero_grad(set_to_none=True);value=backward(model,x,y,s,micro);norm=torch.nn.utils.clip_grad_norm_(parameters(model).values(),1.);assert torch.isfinite(norm)
                assert sum(sum(1 for _ in open(p)) for p in OUT.glob('smoke_*.jsonl'))<36,'SMOKE_CAP';save(intent,dict(next_step=step+1,status='BEFORE_OPTIMIZER'));optimizer.step();sync()
                append(journal,dict(step=step+1,loss=value,optimizer_updates=1));save(intent,dict(completed_step=step+1,status='JOURNALED'));losses.append(value)
            assert frozen_hash(model)==before and buffers==tensor_hash(dict(model.named_buffers()))
            state=cpu_state(model);changed=[n for n in state if not torch.equal(state[n],initial[n])]
            if arm in ['B0','C1']:assert any(n.endswith('.b') for n in changed)
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
                if arm in ['C2','C3']+CONTROLS:
                    baseline=build('C0',85550 if source=='ettm2' else 81550,source,smoke_proxy=source=='ettm2');off_exact=torch.equal(model(x,s,residual_mode='off'),baseline(x,s));assert off_exact;del baseline
            done[key]=dict(updates=2,losses=losses,microbatch=micro,trainable_parameters=sum(v.numel() for v in state.values()),changed_parameters=changed,frozen_unchanged=True,frozen_sha256=before,restore_exact=True,adapter_off_B0_exact=off_exact,batch_order_max_normalized_error=diff,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
            save(path,done);print('SMOKE',key,losses,flush=True);del model,optimizer;cleanup()
    assert len(done)==18 and sum(v['updates'] for v in done.values())==36
