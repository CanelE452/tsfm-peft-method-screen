from .common import *

def smoke(watch):
    path=OUT/'SMOKE.json';done=read(path) if path.exists() else {}
    for source in SOURCES:
        data=arrays(source)
        for arm in NEW_ARMS:
            key=f'{source}_{arm}'
            if key in done:assert done[key]['updates']==2;continue
            journal=OUT/f'smoke_{key}.jsonl';intent=OUT/f'smoke_{key}_intent.json'
            assert not journal.exists() and not intent.exists(),'AMBIGUOUS_SMOKE_NO_REPLAY'
            watch.boundary();m=build(arm,81551,source);initial=cpu_state(m);frozen=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()));opt=optimizer_for(m)
            x,y,s=batch(data,arm,0,np.arange(32));x0=x.clone()
            with torch.no_grad():
                before=m(x,s);teacher=m(x,s,residual_mode='off');assert torch.equal(before,teacher)
                # Reused PLAIN starts from exactly this adapter initialization.
                r=read(OUT/'fits'/fit_id(source,'PLAIN',81551)/'receipt.json');st=torch.load(ROOT/r['checkpoints'][0]['checkpoint'],map_location='cpu',weights_only=True)
                assert tensor_hash(st)==tensor_hash(initial)
            losses=[];torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            for k in range(2):
                watch.boundary();idx=np.arange(k*32,(k+1)*32);x,y,s=batch(data,arm,0,idx);opt.zero_grad(set_to_none=True)
                values=backward(m,x,y,s,arm,source,0,idx);norm=torch.nn.utils.clip_grad_norm_(parameters(m).values(),1.);assert torch.isfinite(norm)
                n=sum(sum(1 for _ in open(p)) for p in OUT.glob('smoke_*.jsonl'));assert n<SMOKE_CAP
                save(intent,dict(status='BEFORE_OPTIMIZER',next_step=k+1));opt.step();sync();append(journal,dict(step=k+1,gradient_norm=float(norm),**values));save(intent,dict(status='JOURNALED',completed_step=k+1));losses.append(values)
            assert frozen_hash(m)==frozen and tensor_hash(dict(m.named_buffers()))==buffers
            st=cpu_state(m);changed=[n for n in st if not torch.equal(st[n],initial[n])];assert 'adapter.up.weight' in changed and 'adapter.down.weight' in changed
            p=CACHE/'smoke'/f'{key}.pt';atomic_torch(p,st)
            with torch.no_grad():
                pred=m(x,s);off=m(x,s,residual_mode='off');restore(m,initial);restore(m,torch.load(p,weights_only=True,map_location='cpu'));assert torch.equal(pred,m(x,s));assert torch.equal(off,m(x,s,residual_mode='off'))
                xr,_,sr=batch(data,arm,0,np.arange(32));assert torch.equal(x0,xr) and torch.equal(before,m(xr,sr,residual_mode='off'))
            done[key]=dict(updates=2,losses=losses,initial_B0_exact=True,reused_initial_exact=True,frozen_unchanged=True,buffers_unchanged=True,restore_exact=True,off_B0_exact=True,input_unchanged=True,changed=changed,trainable=8712,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),seconds=time.perf_counter()-start)
            save(path,done);print('SMOKE_COMPLETE',key,done[key]['seconds'],flush=True);del m,opt;cleanup()
    assert len(done)==8 and sum(r['updates'] for r in done.values())==16
