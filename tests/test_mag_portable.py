import pytest
import torch
from tsfm_peft_screen.mag import MagnitudeResidual, magnitude_gate

def test_constant_context():
    x=torch.ones(2,512);s=torch.ones(2)
    assert torch.equal(magnitude_gate(x,s),torch.ones(2,32))

def test_one_extreme_patch_and_boundary():
    x=torch.zeros(1,512);x[:,:16]=.4
    g=magnitude_gate(x,torch.ones(1))
    assert g[0,0]==0 and torch.equal(g[0,1:],torch.ones(31))
    x[:,:16]=.3
    assert torch.equal(magnitude_gate(x,torch.ones(1)),torch.ones(1,32))

def test_affine_gate_with_scaled_train_sigma():
    gen=torch.Generator().manual_seed(9);x=torch.randn(4,512,generator=gen);x[:,480:]+=8;s=torch.ones(4)
    assert torch.equal(magnitude_gate(x,s),magnitude_gate(x*2+3,s*2))

@pytest.mark.parametrize('kind',['nan','inf_sigma','zero_sigma','length','dtype'])
def test_invalid_input_rejected(kind):
    x=torch.zeros(1,512);s=torch.ones(1)
    if kind=='nan':x[0,0]=float('nan')
    elif kind=='inf_sigma':s.fill_(float('inf'))
    elif kind=='zero_sigma':s.zero_()
    elif kind=='length':x=x[:,:500]
    else:x=x.double()
    with pytest.raises(ValueError):magnitude_gate(x,s)

def test_identity_rng_and_parameter_budget():
    torch.manual_seed(99);before=torch.random.get_rng_state().clone();a=MagnitudeResidual(81551)
    assert torch.equal(before,torch.random.get_rng_state())
    assert sum(p.numel() for p in a.parameters())==8712
    h=torch.randn(2,32,512);g=torch.rand(2,32)
    with torch.inference_mode():assert torch.equal(a(h,g),h)

def test_nonzero_residual_obeys_local_bound_and_zero_gate():
    a=MagnitudeResidual(81551)
    with torch.no_grad():a.up.weight.fill_(.2);a.up.bias.fill_(.5)
    h=torch.randn(2,32,512);g=torch.rand(2,32);g[:,0]=0
    with torch.inference_mode():
        out=a(h,g);bound=h.square().mean(-1).sqrt().quantile(.5,dim=-1,keepdim=True).clamp_min(1e-6)*g
    assert torch.equal(out[:,0],h[:,0])
    assert ((out-h).abs()<=bound[...,None]+1e-6).all()
