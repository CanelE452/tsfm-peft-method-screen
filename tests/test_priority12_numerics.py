import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import numpy as np
import torch
from priority12.numerics import check_parity,path_check

def test_fp64_partition_mask_and_loss_gradient():
    torch.manual_seed(60200);x=torch.randn(8,6,dtype=torch.float64);y=torch.randn(8,3,5,dtype=torch.float64)
    valid=torch.ones_like(y,dtype=torch.bool);valid[0,:,1]=False;valid[7,:,3:]=False;q=torch.tensor([.1,.5,.9],dtype=torch.float64)[None,:,None]
    answers=[]
    for micro in [8,4]:
        p=torch.ones(6,15,dtype=torch.float64,requires_grad=True);total=0.
        for i in range(0,8,micro):
            z=(x[i:i+micro]@p).reshape(micro,3,5);e=y[i:i+micro]-z
            loss=(2*torch.maximum(q*e,(q-1)*e)*valid[i:i+micro]).mean(-1).sum(-1).mean()*(micro/8)
            loss.backward();total+=loss.detach()
        answers.append((total,p.grad))
    for a,b in zip(*answers):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)

def test_v2_units_and_fixed_thresholds():
    a={k:torch.ones(8,21,48) if k in ['z','raw'] else torch.ones(8) for k in ['z','raw','loss','raw_gradient','clipped_gradient','update','adam']};a['loss']=torch.ones(1);a['rng']='x'
    b={k:v.clone() if torch.is_tensor(v) else v for k,v in a.items()};a['raw']*=1000;b['raw']*=1000;b['raw']+=.001
    assert check_parity(a,b,'fp32',micro=True,scale=np.full(4,1000))['passed']
    b['update']+=.0002;assert not check_parity(a,b,'fp32',micro=True,scale=np.full(4,1000))['passed']
    assert check_parity(a,b,'bf16',micro=True,scale=np.full(4,1000))['passed']
    b['rng']='bad';assert not check_parity(a,b,'bf16',micro=True,scale=np.full(4,1000))['passed']

def test_path_gate_does_not_reuse_single_step_gate():
    a={'p':torch.zeros(3)};b={'p':torch.ones(3)};c={'p':torch.ones(3)*1.0005};raw=torch.ones(8,21,48)
    assert path_check(a,b,c,raw,raw,np.ones(4),'fp32')['passed']
    c={'p':torch.ones(3)*1.002};assert not path_check(a,b,c,raw,raw,np.ones(4),'fp32')['passed']
