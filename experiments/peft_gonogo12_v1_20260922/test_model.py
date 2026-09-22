import numpy as np
import torch
from model import CoefficientLinear, bias_features


def test_bias_zero_clean_and_aligned_set_and_special_tokens():
    clean=torch.arange(192,dtype=torch.float32)[None].repeat(2,1)
    aligned=clean.clone();aligned[:,-48:]=torch.nan
    for arm in ['SET_BIAS','CENTROID_BIAS','KEY_BIAS','GENERIC_BIAS']:
        torch.testing.assert_close(bias_features(clean,arm),torch.zeros(2,3,13,13),atol=0,rtol=0)
        features=bias_features(aligned,arm)
        assert torch.isfinite(features).all()
        assert torch.count_nonzero(features[:,:,-1,:])==0
        assert torch.count_nonzero(features[:,:,:,-1])==0
        if arm=='SET_BIAS':
            assert torch.count_nonzero(features)==0


def test_set_bias_matches_scalar_observed_support_oracle():
    x=torch.arange(192,dtype=torch.float32)[None]
    x[:,[1,2,5,17,18,29]]=torch.nan
    features=bias_features(x,'SET_BIAS')
    for k,tau in enumerate([16.,64.,192.]):
        a=np.array([j for j in range(16) if j not in [1,2,5]])
        b=np.array([j for j in range(16,32) if j not in [17,18,29]])
        observed=np.mean([np.exp(-.5*((u-v)/tau)**2) for u in a for v in b])
        full=np.mean([np.exp(-.5*((u-v)/tau)**2) for u in range(16) for v in range(16,32)])
        np.testing.assert_allclose(float(features[0,k,0,1]),np.log(observed/full),atol=1e-6)


def test_coefficient_linear_equals_dense_weight_update_and_frozen_base():
    torch.manual_seed(22)
    base=torch.nn.Linear(16,16,bias=False);base.requires_grad_(False)
    u=torch.linalg.qr(torch.randn(16,8)).Q;v=torch.linalg.qr(torch.randn(16,8)).Q
    s=torch.arange(1,9,dtype=torch.float32)/10;c=s.norm()/8**.5
    model=CoefficientLinear(base,dict(u=u,v=v,s=s,c=c))
    with torch.no_grad():model.d.copy_(torch.linspace(-.2,.3,8))
    x=torch.randn(3,4,16)
    expected=x@(base.weight+u@torch.diag(s+c*model.d)@v.T).T
    torch.testing.assert_close(model(x),expected,atol=1e-6,rtol=1e-5)
    model(x).square().mean().backward()
    assert model.d.grad is not None and model.d.grad.norm()>0 and base.weight.grad is None
