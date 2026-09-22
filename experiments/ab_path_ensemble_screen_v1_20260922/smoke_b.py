import gc
import numpy as np
import torch
from common import *
from model import *
from engine import fit,tensors,distribution
from bdata_adapter import Wind

def main():
    setup();d=Wind();r=RESULTS/'B'
    assert not (r/'MODEL_SMOKE_AUDIT.json').exists()
    packets=[d.batch(v) for v in d.schedule(SEEDS[0])[:2]];b=tensors(packets[0]);records={};gradient={}
    for arm in ARMS_B:
        m=ModelB(arm,d.dim);calls=[]
        hook=m.network.register_forward_hook(lambda *args:calls.append(1))
        with torch.no_grad():
            z,p,q0=distribution(m,b,'B');assert len(calls)==1
            native=m.pipeline.predict(b['x'],prediction_length=9).to('cuda')[:,:,1:9].transpose(1,2)
            torch.testing.assert_close(q0,native,rtol=0,atol=0)
            torch.testing.assert_close(z,q0[:,:,None].expand(-1,-1,z.shape[-1]//9,-1).flatten(2),rtol=0,atol=0)
            # Target replacement cannot affect either weather routing or base input.
            replaced={**b,'y':torch.ones_like(b['y'])*-12345}
            torch.testing.assert_close(distribution(m,replaced,'B')[0],z,rtol=0,atol=0)
        hook.remove()
        cps,receipt=fit(m,'B','smoke_'+arm,packets,phase='smoke')
        with torch.no_grad():
            z,p,_=distribution(m,b,'B');m.load_learned(cps[0]);m.load_learned(cps[2]);restored=distribution(m,b,'B')
            torch.testing.assert_close(z,restored[0],rtol=0,atol=0)
            reorder={k:v.flip(0) for k,v in b.items()};torch.testing.assert_close(z,distribution(m,reorder,'B')[0].flip(0),rtol=1e-5,atol=1e-4)
            if arm in ['SET','MEMBER','SCENARIO3']:
                permuted={**b,'weather':b['weather'].flip(1)};zp,pp,_=distribution(m,permuted,'B')
                torch.testing.assert_close(weighted_crps(z,p,b['y'])/b['sigma'][:,None],weighted_crps(zp,pp,b['y'])/b['sigma'][:,None],rtol=1e-5,atol=1e-5)
            if arm=='SCENARIO3':
                paths=b['weather'].flatten(2);ctx=torch.zeros((len(paths),80),device='cuda')
                identical=paths[:,0:1].expand(-1,50,-1);prototype,mix,assignment=m.pool(identical,ctx)
                torch.testing.assert_close(prototype,paths[:,0:1].expand(-1,3,-1),rtol=1e-5,atol=1e-5)
                assert all(receipt['last_gradients'][n]>0 for n in ['pool.queries.weight','pool.mix.weight','pool.encoder.0.weight'])
                gradient={n:receipt['last_gradients'][n] for n in ['pool.queries.weight','pool.mix.weight','pool.encoder.0.weight']}
        records[arm]=dict(trainable=receipt['trainable_parameters'],small_module_parameters=sum(p.numel() for n,p in m.named_parameters() if p.requires_grad and not n.startswith('network.')),
            initial_trainable_sha=receipt['initial_sha'],final_trainable_sha=receipt['final_sha'],target_forward_calls_per_distribution=1,set_hidden=getattr(m,'set_width',None))
        del m;gc.collect();torch.cuda.empty_cache()
    assert records['SET']['small_module_parameters']>=records['SCENARIO3']['small_module_parameters']
    save(r/'MODEL_SMOKE_AUDIT.json',dict(status='PASS',smoke_updates=12,forecast_slice='native indices 1:9, +3..+24h after last observation issue-3h',
        native_initial_parity=True,zero_condition_decoder_parity=True,future_target_replacement=True,save_restore=True,batch_reorder=True,
        set_member_scenario_permutation=True,identical_member_reduction=True,scenario_second_update_gradients=gradient,arms=records))

if __name__=='__main__':
    with executor_lock():main()
