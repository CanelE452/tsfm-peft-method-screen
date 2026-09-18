from .common import *
from .model import loss_2pinball
from experiments.additive_persistence_validation_v1_20260917.checks import arrays,batch,backward

def smoke(watch):
    done=read(OUT/'SMOKE.json') if (OUT/'SMOKE.json').exists() else {}
    for source in SOURCES:
        a=arrays(source);x,y,s=batch(a,'C2',0,np.arange(32));base=build('C0',81550,source)
        with torch.no_grad():ref=base(x,s)
        for arm in ARMS:
            key=source+'_'+arm
            if key in done:assert done[key]['updates']==2;continue
            journal=OUT/f'smoke_{key}.jsonl';intent=OUT/f'smoke_{key}_intent.json';assert not journal.exists() and not intent.exists(),'AMBIGUOUS_SMOKE'
            watch.boundary();m=build(arm,81550,source);before=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()));initial=cpu_state(m);xx=x.clone()
            with torch.no_grad():assert torch.equal(m(x,s),ref) and torch.equal(m(x,s,residual_mode='off'),ref) and torch.equal(x,xx)
            if arm in ['POS_ONLY','MAG_ONLY']:
                c2=old.build('C2',81550,source);assert tensor_hash(m.adapter.state_dict())==tensor_hash(c2.adapter.state_dict());del c2
            if arm=='POS_ONLY':assert torch.equal(m.gate(x,s),m.gate(x.flip(-1)*3,s))
            micro=32;optimizer=optimizer_for(m,1e-4);losses=[];grads=[]
            for step in range(2):
                watch.boundary();optimizer.zero_grad(set_to_none=True);xx,yy,ss=batch(a,arm,0,np.arange(step*32,(step+1)*32));loss=backward(m,xx,yy,ss,micro);norm=torch.nn.utils.clip_grad_norm_(parameters(m).values(),1.);assert torch.isfinite(norm)
                grad={n:float(p.grad.norm()) for n,p in parameters(m).items() if p.grad is not None};assert sum(sum(1 for _ in open(p)) for p in OUT.glob('smoke_*.jsonl'))<12
                save(intent,dict(status='BEFORE_OPTIMIZER',next_step=step+1));optimizer.step();sync();append(journal,dict(step=step+1,loss=loss,gradients=grad));save(intent,dict(status='JOURNALED',completed_step=step+1));losses.append(loss);grads.append(grad)
            assert frozen_hash(m)==before and buffers==tensor_hash(dict(m.named_buffers()));state=cpu_state(m);changed=[n for n in state if not torch.equal(state[n],initial[n])]
            needed=['output_linear.weight','gamma'] if arm=='OUTPUT_CONTEXT' else ['adapter.up.weight','adapter.down.weight']
            if arm=='POS_ONLY':needed+=['position_logits']
            assert all(n in changed for n in needed),(arm,changed)
            path=CACHE/'smoke'/f'{key}.pt';atomic_torch(path,state);restored=build(arm,81550,source);restore(restored,torch.load(path,weights_only=True,map_location='cpu'))
            with torch.no_grad():
                p=m(x,s);assert torch.equal(p,restored(x,s));assert torch.equal(m(x,s,residual_mode='off'),ref)
                if arm=='OUTPUT_CONTEXT':torch.testing.assert_close(torch.diff(p,dim=1),torch.diff(ref,dim=1),rtol=1e-4,atol=1e-5*float(s.max()))
            done[key]=dict(updates=2,losses=losses,changed=changed,gradients=grads,initial_B0_exact=True,off_B0_exact=True,frozen_unchanged=True,restore_exact=True,parameters=sum(p.numel() for p in state.values()),microbatch=micro)
            save(OUT/'SMOKE.json',done);print('SMOKE',key,changed,flush=True);del m,restored,optimizer;cleanup()
        del base;cleanup()
    assert len(done)==6
