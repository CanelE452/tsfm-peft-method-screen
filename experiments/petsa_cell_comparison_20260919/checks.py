"""Official-cell parity and actual Chronos CPU wiring; no optimizer updates."""
import ast
import math
from .common import *
from .model import GCM

def cpu_check(actual_model=True):
    setup();OUT.mkdir(parents=True,exist_ok=True)
    receipt=read(ROOT/'research/method_baseline_compatibility_20260919/PETSA_CODE_RECEIPTS.json')[0]
    path=ROOT/receipt['local'];assert sha(path)==receipt['sha256']
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GCM')
    ns={'torch':torch,'nn':torch.nn,'math':math}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    parity=[]
    for length in [64,512]:
        torch.manual_seed(91981)
        official=ns['GCM'](length,n_var=1,low_rank=16).double()
        ours=GCM(length).double();ours.load_state_dict(official.state_dict())
        x=torch.randn(3,length,1,dtype=torch.float64,requires_grad=True)
        assert torch.equal(ours(x),x)
        # Nonzero fixed synthetic weights test the full cell, not only identity.
        with torch.no_grad():
            ours.lora_B.copy_(torch.randn_like(ours.lora_B)*.01)
            ours.bias.copy_(torch.randn_like(ours.bias)*.01)
        official.load_state_dict(ours.state_dict())
        a,b=official(x),ours(x);assert torch.equal(a,b)
        ga=torch.autograd.grad(a.square().sum(),[x,*official.parameters()])
        gb=torch.autograd.grad(b.square().sum(),[x,*ours.parameters()])
        assert all(torch.equal(u,v) for u,v in zip(ga,gb))
        torch.testing.assert_close(ours(x.flip(0)).flip(0),b,rtol=0,atol=0)
        parity.append(dict(length=length,nonzero_output_exact=True,input_and_all_parameter_gradients_exact=True,initial_identity=True))
    rows=[]
    if actual_model:
        for source in SOURCES:
            m=build(ARMS[0],81550,source,device='cpu');a=arrays(source)
            x,y,s=batch(a,ARMS[0],0,np.arange(4),device='cpu');original_x=x.clone()
            frozen=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()));initial=cpu_state(m)
            with torch.no_grad():
                base=m(x,s,off=True);initial_pred=m(x,s)
            assert torch.equal(base,initial_pred) and torch.equal(x,original_x)
            # Probe only. Discard these manually perturbed states after the check.
            with torch.no_grad():
                m.in_cali.lora_B.fill_(.001);m.out_cali.lora_B.fill_(.001)
            pred=m(x,s);loss=loss_2pinball(pred,y,s,m.base.quantiles)
            assert torch.isfinite(loss);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters(m).values())
            assert all(p.grad is None for p in m.b0.parameters())
            assert m.in_cali.lora_B.grad.abs().sum()>0 and m.out_cali.lora_B.grad.abs().sum()>0
            assert not torch.equal(pred,base)
            state=cpu_state(m)
            fresh=build(ARMS[0],81550,source,device='cpu');restore(fresh,state)
            with torch.no_grad():assert torch.equal(fresh(x,s),pred)
            restore(m,initial)
            with torch.no_grad():assert torch.equal(m(x,s),base)
            assert frozen==frozen_hash(m) and buffers==tensor_hash(dict(m.named_buffers()))
            rows.append(dict(source=source,seed=81550,TRAIN_examples=4,initial_B0_exact=True,adapter_off_B0_exact=True,nonzero_input_and_output_gradient=True,all_trainable_gradients_finite=True,frozen_parameters_and_buffers_unchanged=True,fresh_restore_exact=True,input_not_mutated=True,optimizer_updates=0))
            del m,fresh;cleanup()
    result=dict(status='CPU_ACTUAL_MODEL_WIRING_VERIFIED' if actual_model else 'CPU_CELL_PARITY_VERIFIED',reference=receipt,cell_cases=parity,actual_Chronos_cases=rows,trainable_parameters=19010,optimizer_updates=0,evaluation_samples_read=0,synthetic_parameter_perturbation_not_training=True,GPU_smoke_complete=False,main_training_complete=False)
    save(OUT/'CPU_CHECKS.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':cpu_check()
