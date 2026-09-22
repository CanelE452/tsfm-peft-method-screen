import numpy as np
import torch
from model import empirical_crps, continuation_score, branch_context

def test_empirical_score_and_gradient_match_pairwise():
    torch.manual_seed(10)
    for m in (9,81):
        z = torch.randn(3,4,m,dtype=torch.double,requires_grad=True)
        y = torch.randn(3,4,dtype=torch.double)
        ref = (z-y[...,None]).abs().mean(-1) - .5*(z[..., :,None]-z[...,None,:]).abs().mean((-1,-2))
        fast = empirical_crps(z,y)
        torch.testing.assert_close(fast,ref,rtol=1e-12,atol=1e-12)
        a, = torch.autograd.grad(fast.sum(),z,retain_graph=True)
        b, = torch.autograd.grad(ref.sum(),z)
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

def test_component_mixture_identity_by_cdf_integration():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(1,9,3,9))
    y = torch.tensor(rng.normal(size=(1,3)))
    t = torch.tensor(z)
    difference = (continuation_score(t,y,'COMPONENT')-continuation_score(t,y,'MIXTURE')).numpy()[0]
    for h in range(3):
        points = np.sort(np.unique(z[0,:,h,:]))
        mid = (points[1:]+points[:-1])/2
        cdfs = (z[0,:,h,:,None] <= mid).mean(1)
        integral = (cdfs.var(0)*np.diff(points)).sum()
        np.testing.assert_allclose(difference[h],integral,rtol=1e-12,atol=1e-12)
    assert (difference>=0).all()

def test_degenerate_components_and_translation():
    z = torch.randn(2,1,4,9,dtype=torch.double).repeat(1,9,1,1)
    y = torch.randn(2,4,dtype=torch.double)
    torch.testing.assert_close(continuation_score(z,y,'COMPONENT'),continuation_score(z,y,'MIXTURE'))
    torch.testing.assert_close(continuation_score(z+17,y+17,'MIXTURE'),continuation_score(z,y,'MIXTURE'))

def test_branch_context_detached_no_target_and_no_sort():
    x = torch.arange(2*512,dtype=torch.float32).reshape(2,512)
    first = torch.randn(2,9,64,requires_grad=True)
    ctx = branch_context(x,first).reshape(2,9,576)
    assert not ctx.requires_grad
    torch.testing.assert_close(ctx[:,:,:512],x[:,None].expand(-1,9,-1))
    torch.testing.assert_close(ctx[:,:,512:],first.detach())
