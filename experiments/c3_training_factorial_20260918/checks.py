from .common import *
from experiments.additive_persistence_validation_v1_20260917.checks import arrays,batch,backward
from .analysis_math import contrasts,SIGNS

def cpu_checks():
 # Exact known main and interaction effects; paired row-order independence.
 values=2+3*SIGNS[:,0]-4*SIGNS[:,1]+5*SIGNS[:,0]*SIGNS[:,2]
 out=contrasts(values);np.testing.assert_allclose(out,[6,-8,0,0,10,0,0],atol=1e-12)
 save(OUT/'CPU_CHECKS.json',dict(status='VERIFIED',known_factorial_effects=True,main_updates=0))

def smoke(watch):
 done=read(OUT/'SMOKE.json') if (OUT/'SMOKE.json').exists() else {}
 for source in SOURCES:
  triple=(81551,81552,81551);a=arrays(source);x,y,s=batch(a,'C3',0,np.arange(32));xx=x.clone();base=build('C0',triple,source)
  with torch.no_grad():ref=base(x,s)
  states=[]
  for arm in ARMS:
   key=source+'_'+arm
   if key in done:continue
   journal=OUT/f'smoke_{key}.jsonl';intent=OUT/f'smoke_{key}_intent.json';assert not journal.exists() and not intent.exists(),'AMBIGUOUS_SMOKE'
   watch.boundary();m=build(arm,triple,source);before=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()));initial=cpu_state(m);states.append(tensor_hash(initial))
   with torch.no_grad():assert torch.equal(m(x,s),ref) and torch.equal(m(x,s,residual_mode='off'),ref) and torch.equal(x,xx)
   opt=optimizer_for(m,.0003 if source=='electricity' else .0001)
   for step in range(3):
    watch.boundary();opt.zero_grad(set_to_none=True);bx,by,bs=batch(a,arm,0,np.arange(step*32,(step+1)*32));loss=backward(m,bx,by,bs,32);norm=torch.nn.utils.clip_grad_norm_(parameters(m).values(),1.);assert torch.isfinite(norm)
    assert sum(sum(1 for _ in open(p)) for p in OUT.glob('smoke_*.jsonl'))<12
    save(intent,dict(status='BEFORE_OPTIMIZER',next_step=step+1));opt.step();sync();append(journal,dict(step=step+1,loss=loss,gradient_norm=float(norm)));save(intent,dict(status='JOURNALED',completed_step=step+1))
   assert frozen_hash(m)==before and buffers==tensor_hash(dict(m.named_buffers()))
   state=cpu_state(m);assert all(not torch.equal(state[k],initial[k]) for k in ['adapter.up.weight','adapter.down.weight'])
   path=CACHE/'smoke'/f'{key}.pt';atomic_torch(path,state);restored=build(arm,triple,source);restore(restored,torch.load(path,weights_only=True,map_location='cpu'))
   with torch.no_grad():assert torch.equal(m(x,s),restored(x,s)) and torch.equal(m(x,s,residual_mode='off'),ref)
   done[key]=dict(updates=3,initial_B0_exact=True,off_B0_exact=True,frozen_unchanged=True,buffers_unchanged=True,restore_exact=True,input_unchanged=True,adapter_changed=True,initial_adapter_sha256=tensor_hash(initial))
   save(OUT/'SMOKE.json',done);print('SMOKE',key,flush=True);del m,opt,restored;cleanup()
  assert done[source+'_C3']['initial_adapter_sha256']==done[source+'_MAG_ONLY']['initial_adapter_sha256']
  del base;cleanup()
 assert len(done)==4
