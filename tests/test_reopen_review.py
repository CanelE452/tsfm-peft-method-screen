import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_reopen_query_resources import blocks,partial_cp,dominates
from tsfm_peft_screen.candidates.censor import cdf


def test_partial_blocks_and_exact_multistep_gradients():
    for k in [0,3,6,9,12]:assert len(set(blocks(k)))==k and all(0<=i<12 for i in blocks(k))
    from types import SimpleNamespace
    layers=torch.nn.ModuleList([torch.nn.Linear(3,3) for _ in range(12)])
    m=SimpleNamespace(encoder=SimpleNamespace(block=layers));x=torch.randn(2,3)
    def f():
        h=x
        for b in layers:h=torch.tanh(b(h))
        return h
    for k in [0,3,6,9,12]:
        ref=f();a=torch.autograd.grad(ref.sum(),tuple(layers.parameters()))
        with partial_cp(m,k):out=f()
        b=torch.autograd.grad(out.sum(),tuple(layers.parameters()))
        torch.testing.assert_close(out,ref,rtol=0,atol=0)
        for xg,yg in zip(a,b):torch.testing.assert_close(xg,yg,rtol=0,atol=0)


def test_three_axis_dominance_requires_no_worse_axis():
    p=dict(loss=1.,memory=2.,time=3.)
    assert not dominates(p,p)
    assert dominates(dict(loss=.9,memory=2.,time=3.),p)
    assert not dominates(dict(loss=.9,memory=3.,time=3.),p)


def test_censor_extreme_tail_has_zero_gradient():
    p=torch.arange(21,dtype=torch.float32).reshape(1,21,1).requires_grad_()
    loss=-(1-cdf(p,torch.tensor([[100.]]))).clamp_min(1e-6).log()
    g=torch.autograd.grad(loss.sum(),p)[0]
    assert float(loss.detach())>10 and torch.count_nonzero(g)==0
