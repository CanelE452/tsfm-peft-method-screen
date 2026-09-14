import numpy as np
import pytest
import torch
from torch import nn
from types import SimpleNamespace
from tsfm_peft_screen.forecast_query.budget import block_indices,partial_checkpoint,micro_origins,batch,choose_option,TrainingClock,seal_valid,check_parity
from tsfm_peft_screen.backbone import native_loss
from tsfm_peft_screen.reproducibility import digest


def test_bound_checkpoint_layers_and_restore():
    torch.manual_seed(5)
    blocks=nn.ModuleList([nn.Linear(4,4) for _ in range(12)])
    model=SimpleNamespace(encoder=SimpleNamespace(block=blocks))
    x=torch.randn(3,4,requires_grad=True)
    def forward():
        v=x
        for b in blocks:v=b(v).tanh()
        return v
    y=forward();g=torch.autograd.grad(y.sum(),tuple(blocks.parameters()))
    for k in [0,3,6,9,12]:
        assert block_indices(k)==(np.linspace(0,11,k,dtype=int).tolist() if k else [])
        with partial_checkpoint(model,k):z=forward()
        gg=torch.autograd.grad(z.sum(),tuple(blocks.parameters()))
        torch.testing.assert_close(y,z,rtol=0,atol=0)
        for a,b in zip(g,gg):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_microbatch_group_loss_gradient_adam():
    values=np.arange(5000*4,dtype=np.float32).reshape(5000,4)/1000
    x,y,g=batch(values,[4200,4400],device='cpu')
    assert g.tolist()==[0,0,0,0,1,1,1,1]
    outputs=[]
    for size in [1,2]:
        p=nn.Parameter(torch.full((21,48),.2));opt=torch.optim.AdamW([p],lr=1e-4,weight_decay=0)
        total=0
        for oo,w in micro_origins([4200,4400],size):
            xx,yy,gg=batch(values,oo,device='cpu');assert len(xx)==4*size
            # Coupling inside each four-channel origin is preserved under partition.
            context=xx[:,-1].reshape(size,4).mean(1).repeat_interleave(4)[:,None]
            pred=p[None].expand(len(xx),21,48)+context[:,None,:]*.001
            target=yy.clone();target[0,0]=float('nan')
            # Missing positions must be the same under either partition.
            target=yy.clone();target[::4,0]=float('nan')
            loss=native_loss(pred,target,torch.zeros(len(xx),1),torch.ones(len(xx),1))*w
            loss.backward();total+=float(loss.detach())
        grad=p.grad.clone();torch.nn.utils.clip_grad_norm_([p],1.);opt.step();outputs.append((total,grad,p.detach().clone()))
    assert outputs[0][0]==pytest.approx(outputs[1][0],rel=1e-6)
    for a,b in zip(outputs[0][1:],outputs[1][1:]):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)


def test_budget_and_tiebreak():
    r=[dict(valid=True,peak_allocated=90,block1_seconds=1.,cp=0,micro=2),dict(valid=True,peak_allocated=80,block1_seconds=1.019,cp=3,micro=1),dict(valid=True,peak_allocated=96,block1_seconds=.5,cp=0,micro=2)]
    assert choose_option(r,100)==r[1]
    assert choose_option(r,50) is None
    assert choose_option(r,1000)==r[2]
    for v in r:v['valid']=False
    assert choose_option(r,1000) is None


def test_clock_whole_updates_and_cap():
    c=TrainingClock((30.,60.,120.),1.,8192)
    assert c.add(29.9) is None and c.add(.2)==30
    assert c.add(30.)==60 and c.add(60.)==120 and c.complete
    with pytest.raises(RuntimeError):c.add(.1)
    with pytest.raises(RuntimeError):TrainingClock((30.,),1.,8192).add(31.01)
    with pytest.raises(RuntimeError):TrainingClock((30.,),1.,1).add(.1)


def test_seal_exact_cells_and_tampering():
    cells=[('ettm2',39000,'query'),('electricity',39000,'query')]
    s=dict(contract_hash='a',selections=[dict(dataset=d,seed=z,arm=a) for d,z,a in cells]);s['seal_hash']=digest(s)
    assert seal_valid(s,'a',cells)
    assert not seal_valid(s,'b',cells)
    s['selections'][0]['arm']='side';assert not seal_valid(s,'a',cells)


def test_micro_tolerance_is_not_checkpoint_tolerance():
    a={k:torch.ones(8,21,48) if k in ['z','raw'] else torch.ones(5) for k in ['z','raw','loss','raw_gradient','clipped_gradient','update','adam']};a['rng']='same'
    b={k:v+.001 if isinstance(v,torch.Tensor) else v for k,v in a.items()}
    assert not check_parity(a,b,'bf16')['passed']
    assert check_parity(a,b,'bf16',micro=True,scale=np.ones(4))['passed']
    b['rng']='different';assert not check_parity(a,b,'bf16',micro=True,scale=np.ones(4))['passed']


def test_report_gain_sign():
    import importlib.util
    from pathlib import Path
    p=Path(__file__).parents[1]/'scripts/finalize_query_budget_pilot.py'
    spec=importlib.util.spec_from_file_location('query_finalizer',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.gain(.5,.5)==0
    assert m.gain(.5,.45)==pytest.approx(10)
    assert m.gain(.5,.55)==pytest.approx(-10)
    assert m.gain(0,1) is None
