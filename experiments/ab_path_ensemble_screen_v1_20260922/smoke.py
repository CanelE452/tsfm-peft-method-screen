import gc,sys
import numpy as np
import torch
from common import *
from model import *
from engine import fit,tensors,distribution,objective
from A.data import Solar

def close(m):del m;gc.collect();torch.cuda.empty_cache()

def smoke_a():
    c='A';r=RESULTS/c
    assert not (r/'MODEL_SMOKE_AUDIT.json').exists()
    d=Solar();packet=d.batch(d.schedule(SEEDS[0])[0]);x=torch.as_tensor(packet['x'][:2],device='cuda')
    m=ModelA('FULL9');f=m.first(x)
    with torch.no_grad():
        z,p,_=m.distribution(x);native=m.pipeline.predict(x,prediction_length=128).to('cuda')
        linear=torch.cat([f,torch.quantile(z[:,64:],torch.arange(1,10,device='cuda')/10,dim=-1).permute(1,0,2)],-1)
        torch.testing.assert_close(native,linear,rtol=1e-5,atol=1e-5)
    initial_native=float((native-linear).abs().max());del m;gc.collect();torch.cuda.empty_cache()
    teacher=None;reports={};initial_router=None;route_grad={}
    for arm in ARMS_A:
        m=ModelA(arm)
        if teacher is not None:m.load_lora(teacher)
        packets=[]
        for j in range(2):
            b=d.batch(d.schedule(SEEDS[0])[j]);packets.append(b)
        if arm!='FULL9':
            # Same disposable smoke teacher distribution for every reduced arm.
            for b,t in zip(packets,teacher_targets):b.update(teacher_z=t[0],teacher_p=t[1])
        if arm in ['GLOBAL3','CONTEXT3']:
            with torch.no_grad():a=tuple(v.detach().cpu() for v in m.distribution(x)[:2])
            if initial_router is None:initial_router=a
            else:
                for aa,bb in zip(initial_router,a):torch.testing.assert_close(aa,bb,rtol=0,atol=0)
            loss=objective(m,tensors(packets[0]),c)
            parameter=m.router.logits if arm=='GLOBAL3' else m.router.up.bias
            grad=torch.autograd.grad(loss,parameter)[0];assert grad[:27].norm()>0 and grad[27:].norm()>0
            route_grad[arm]=dict(prototype_logits_norm=float(grad[:27].norm()),mixture_logits_norm=float(grad[27:].norm()))
        cps,report=fit(m,c,'smoke_'+arm,packets,phase='smoke');reports[arm]=report
        with torch.no_grad():
            after=m.distribution(x)
            torch.testing.assert_close(after[2],f,rtol=0,atol=0)
            m.load_learned(cps[0]);m.load_learned(cps[2]);restored=m.distribution(x)
            torch.testing.assert_close(after[0],restored[0],rtol=0,atol=0)
            reversed_result=m.distribution(x.flip(0))[0].flip(0)
            torch.testing.assert_close(after[0],reversed_result,rtol=1e-5,atol=1e-5)
            if arm=='FULL9':
                teacher=cps[2];teacher_targets=[]
                for b in packets:
                    zz,pp,_=m.distribution(tensors(b)['x']);teacher_targets.append((zz[:,64:].cpu().numpy(),pp[:,64:].cpu().numpy()))
                ctx=torch.cat([x[:,None].expand(-1,9,-1),f],-1).flatten(0,1)
                official=m.pipeline.predict(ctx,prediction_length=64).to('cuda')
                torch.testing.assert_close(m.conditional(ctx),official,rtol=1e-5,atol=1e-5)
        if arm=='CONTEXT3':assert report['last_gradients']['router.down.weight']>0
        del m;gc.collect();torch.cuda.empty_cache()
    save(r/'ROUTER_GRADIENT_AUDIT.json',dict(status='PASS',gradients=route_grad,first_paths_detached=True,prototypes_detached=False,teacher_detached=True,down_changes_on_second_update=True))
    save(r/'MODEL_SMOKE_AUDIT.json',dict(status='PASS',smoke_updates=12,native_initial_max_abs=initial_native,trained_native_conditional_parity=True,
         native_full128_note='Native all-LoRA first block differs by design after learning; hybrid first-F0 + native trained conditional verified separately',
         initial_global_context_parity=True,first64_preserved=True,save_restore=True,batch_reorder=True,arms=list(reports)))

if __name__=='__main__':
    setup()
    with executor_lock():smoke_a()
