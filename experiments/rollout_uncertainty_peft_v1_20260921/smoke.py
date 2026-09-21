import gc
import inspect
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from common import *
from model import *
from ledger import Ledger,restore
from contract.reference_core import RolloutState

def batch(d,tuples):
    pairs=np.asarray(tuples)
    x=np.stack([d['values'][o-512:o,s] for s,o in pairs])
    y=np.stack([d['values'][o:o+256,s] for s,o in pairs])
    sigma=d['sigma'][pairs[:,0]]
    return tuple(torch.as_tensor(a,device='cuda',dtype=torch.float32) for a in [x,y,sigma])

def close(a,b,sigma=None):
    err=(a-b).abs()
    scale=max(1.,float(a.abs().max())) if sigma is None else max(1.,float(sigma.max()))
    assert float(err.max())<=2e-5*scale, (float(err.max()),scale)
    return {'max_abs':float(err.max()),'normalized_max':float(err.max())/scale,'atol_scale':2e-5}

def run():
    from data import load,schedule
    setup()
    report=read_json(RESULTS/'MODEL_AND_SMOKE_AUDIT.json') if (RESULTS/'MODEL_AND_SMOKE_AUDIT.json').exists() else {'status':'RUNNING','sources':{},'smoke_budget':12}
    # Audit and serialization before any optimizer updates. Reuse saved completed smoke checks.
    for source in ['Electricity','ETTh1']:
        d=load(source); packets=schedule(source,92120); x,y,sigma=batch(d,packets[0])
        if source in report['sources'] and report['sources'][source].get('complete'): continue
        audit={}; initial={}; init_hash={}; adapter_hash={}
        base=RolloutModel('F0',92120,lora=False)
        with torch.no_grad():
            raw=base.conditional(x)
            official=base.pipeline.predict(x,prediction_length=64).transpose(1,2).to('cuda')
            audit['native64_parity']=close(raw,official)
            f0=rollout(base,x,sigma)[0]
            native_trace=[]; p=native(base,x,native_trace)
            p2=base.pipeline.predict(x,prediction_length=256).transpose(1,2)
            audit['native256_api_parity']=close(p,p2)
            audit['native_calls']=[{k:v for k,v in t.items() if k!='quantiles'} for t in native_trace]
            assert [(t['batch'],t['length']) for t in native_trace]==[(8,512),(72,576),(72,640),(72,704)]
            audit['native_quantiles_sha256']=tensor_hash([(str(i),t['quantiles']) for i,t in enumerate(native_trace)])
        del base; gc.collect(); torch.cuda.empty_cache()
        for arm in ARMS:
            m=RolloutModel(arm,92120)
            init_hash[arm]=tensor_hash([(n,p) for n,p in m.named_parameters() if p.requires_grad and not n.startswith('adapter.')])
            counts={'lora':sum(p.numel() for n,p in m.named_parameters() if p.requires_grad and not n.startswith('adapter.')),'adapter':sum(p.numel() for n,p in m.named_parameters() if n.startswith('adapter.'))}
            assert counts['lora']==294912
            assert counts['adapter']==(0 if arm==ARMS[0] else 8744)
            if m.adapter is not None: adapter_hash[arm]=tensor_hash(m.adapter.state_dict().items())
            with torch.no_grad():
                trace=[]; pred,_=rollout(m,x,sigma,trace=trace)
                initial[arm]=pred.cpu()
                checks={'init_f0_parity':close(pred,f0),'counts':counts}
                assert [t['length'] for t in trace]==[512,576,640,704]
                assert [t['metadata'].shape[1] for t in trace]==[32,36,40,44]
                assert all(not t['width_requires_grad'] and not t['value_requires_grad'] for t in trace)
                state=RolloutState.initial(x.cpu().numpy(),sigma.cpu().numpy())
                for k,t in enumerate(trace):
                    mode='STATE' if arm==ARMS[1] else 'UNCERTAINTY'
                    np.testing.assert_allclose(t['metadata'].numpy(),state.metadata(mode),rtol=2e-5,atol=2e-5)
                    assert torch.count_nonzero(t['metadata'][:,:32])==0
                    state=state.append(pred[:,k*64:(k+1)*64].cpu().numpy())
                poisoned={**d,'values':d['values'].copy()}
                series,origin=packets[0,0]
                poisoned['values'][origin:origin+256,series]=1234567.
                px,py,ps=batch(poisoned,packets[0,:1])
                assert torch.equal(px,x[:1]) and not torch.equal(py,y[:1])
                clean_trace=[]; clean,_=rollout(m,x[:1],sigma[:1],trace=clean_trace)
                poison_trace=[]; pp,_=rollout(m,px,ps,trace=poison_trace)
                checks['future_poison_parity']=close(clean,pp)
                assert all(torch.equal(a['metadata'],b['metadata']) for a,b in zip(clean_trace,poison_trace))
                assert 'y' not in inspect.signature(rollout).parameters and 'target' not in inspect.signature(m.conditional).parameters
                checks['batch_split_parity']=close(pred,torch.cat([rollout(m,x[j:j+1],sigma[j:j+1])[0] for j in range(8)]))
                checks['order_parity']=close(pred,rollout(m,x.flip(0),sigma.flip(0))[0].flip(0))
            frozen=m.frozen_hash(); learned=m.learned()
            opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
            ledger=Ledger(f'smoke_{source}_{arm}','smoke',m,opt)
            while ledger.step<2:
                bx,by,bs=batch(d,packets[ledger.step])
                r=ledger.update(lambda:train_update(m,opt,bx,by,bs)); event('smoke_update',record=r)
            assert m.frozen_hash()==frozen
            after=m.learned()
            assert any(not torch.equal(v,after[n]) for n,v in learned.items() if not n.startswith('adapter.'))
            if m.adapter is not None:
                assert all(not torch.equal(v,after[n]) for n,v in learned.items() if n.startswith('adapter.'))
            with torch.no_grad():
                trained=rollout(m,x,sigma)[0]
                for p in m.parameters():
                    if p.requires_grad: p.add_(.01)
                assert not torch.equal(trained,rollout(m,x,sigma)[0])
                restore(ledger.latest,m)
                checks['save_restore']=close(trained,rollout(m,x,sigma)[0])
                off=rollout(m,x,sigma,adapter_on=False)[0]
                if m.adapter is not None:
                    assert not torch.equal(off,trained)
                checks['adapter_off_change']=float((off-trained).abs().max())
                if arm==ARMS[0]:
                    checks['trained_R_native_api']=close(native(m,x),m.pipeline.predict(x,prediction_length=256).transpose(1,2))
                    u=torch.from_numpy(np.random.default_rng(1701).random((8,4,16,64)).astype('float32')).cuda()
                    mc,_=mc16(m,x,u)
                    checks['MC_batch_split']=close(mc,torch.cat([mc16(m,x[j:j+1],u[j:j+1])[0] for j in range(8)]))
            checks.update(frozen_weights_buffers_unchanged=True,all_trainable_grad_finite=True,smoke_updates=2,rollout_lengths=[512,576,640,704],feedback_detached=True)
            audit[arm]=checks
            save_json(RESULTS/'MODEL_AND_SMOKE_AUDIT.json',{**report,'partial_source':source,'partial_checks':audit})
            del m,opt,ledger; gc.collect(); torch.cuda.empty_cache()
        assert len(set(init_hash.values()))==1 and len(set(adapter_hash.values()))==1
        audit.update(lora_init_hashes=init_hash,adapter_init_hashes=adapter_hash,complete=True)
        report['sources'][source]=audit
        save_json(RESULTS/'MODEL_AND_SMOKE_AUDIT.json',report)
    pipeline=load_direct()
    with torch.no_grad():
        q=direct(pipeline,x[:8]); qs=torch.cat([direct(pipeline,x[j:j+1]) for j in range(8)])
        assert q.shape==(8,256,9)
        report['chronos2_independent_group_parity']=close(q,qs)
        report['chronos2_reorder_parity']=close(q,direct(pipeline,x[:8].flip(0)).flip(0))
    report['status']='PASS'; save_json(RESULTS/'MODEL_AND_SMOKE_AUDIT.json',report)
    event('model_smoke_complete',status='PASS')

if __name__=='__main__': run()
