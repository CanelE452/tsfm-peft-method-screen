import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from tsfm_peft_screen.metrics import score

spec=importlib.util.spec_from_file_location('gradient_probe',Path(__file__).resolve().parents[1]/'scripts/run_temporal_transfer_gradient_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_masked_equal_channel_metric_and_gradient():
    rng=np.random.default_rng(61)
    p=torch.tensor(rng.normal(size=(2,4,21,5)),requires_grad=True)
    y=torch.tensor(rng.normal(size=(2,4,5)))
    y[0,0,:]=float('nan');y[1,2,0]=float('nan')
    scale=torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
    loss=m.eval_loss(p,y,scale)
    assert abs(float(loss.detach())-score(p.detach().numpy(),y.numpy(),scale.numpy())['scaled_2pinball'])<1e-12
    g=torch.autograd.grad(loss,p)[0]
    assert torch.isfinite(g).all() and (g[0,0]==0).all()
    d=torch.tensor(rng.normal(size=p.shape)); eps=1e-7
    delta=(m.eval_loss(p.detach()+eps*d,y,scale)-m.eval_loss(p.detach()-eps*d,y,scale))/(2*eps)
    assert float(delta)==pytest.approx(float((g*d).sum()),abs=1e-8)


def test_inactive_channel_and_no_targets():
    p=torch.zeros(2,4,21,5,dtype=torch.float64);y=torch.ones(2,4,5);y[:,2]=float('nan');s=torch.ones(4)
    assert float(m.eval_loss(p,y,s))==pytest.approx(score(p.numpy(),y.numpy(),s.numpy())['scaled_2pinball'])
    with pytest.raises(ValueError):m.eval_loss(p,torch.full_like(y,float('nan')),s)


def test_exact_restoration_after_perturbation_exception():
    params={'a':torch.nn.Parameter(torch.tensor([1.,-3.])), 'b':torch.nn.Parameter(torch.tensor([[4.]]))}
    saved={k:p.detach().clone() for k,p in params.items()};h=m.tensor_hash(params)
    try:
        m.put_vector(params,m.flatten(params.values())+1e-3)
        assert m.tensor_hash(params)!=h
        raise RuntimeError('synthetic forward failure')
    except RuntimeError:
        pass
    finally:
        m.restore(params,saved)
    assert m.tensor_hash(params)==h


def test_zero_gradient_is_undefined():
    assert m.geometry(torch.zeros(3),torch.ones(3))[-1]=='UNDEFINED_ZERO_GRADIENT'
    assert m.geometry(torch.ones(3),-torch.ones(3))[-1]==pytest.approx(-1.)
