"""Distribution, gradient, reduction, optimizer and information-boundary contracts."""
import math
import numpy as np
import pytest
import torch
from tsfm_peft_screen.backbone import QUANTILES, native_loss
from tsfm_peft_screen.candidates.censor import cdf
from tsfm_peft_screen.candidates.censor_tail_v1 import log_survival_from_quantiles as logsf, objective, survival_loss


def fixture(dtype=torch.float64):
    return torch.arange(21, dtype=dtype)[None, :, None], torch.tensor(QUANTILES, dtype=torch.float64), torch.ones(1, dtype=dtype)


def scalar(p, y, tau, scale):
    p = sorted(p); x = [p[0]]
    for a, b in zip(p, p[1:]): x.append(x[-1] + max(b-a, 1e-4*max(1, scale)))
    if y <= x[0]:
        b=(x[1]-x[0])*tau[0]/(tau[1]-tau[0])
        return math.log1p(-math.exp(math.log(tau[0])+(y-x[0])/b))
    if y >= x[-1]:
        b=(x[-1]-x[-2])*(1-tau[-1])/(tau[-1]-tau[-2])
        return math.log1p(-tau[-1])-(y-x[-1])/b
    j=next(i for i in range(len(x)-1) if x[i]<=y<x[i+1])
    f=tau[j]+(tau[j+1]-tau[j])*(y-x[j])/(x[j+1]-x[j])
    return math.log1p(-f)


def test_distribution_and_scalar():
    p,t,s=fixture()
    for j in range(21):
        assert abs(float(1-logsf(p,torch.tensor([[float(j)]]),t,s).exp())-float(t[j])) < 1e-12
    ys=torch.linspace(-100,100,2001,dtype=torch.float64)[None,:]
    v=logsf(p.expand(1,21,2001),ys,t,s)
    assert torch.isfinite(v).all() and (v<=0).all() and (v[:,1:]<=v[:,:-1]).all()
    f=1-v.exp(); assert float(f[0,0])<1e-10 and float(f[0,-1])>1-1e-10
    for y in [-12.,0.,.37,6.5,19.9,20.,1000000.]:
        actual=float(logsf(p,torch.tensor([[y]],dtype=torch.float64),t,s))
        assert actual == pytest.approx(scalar(p.flatten().tolist(),y,t.tolist(),1.),abs=1e-10,rel=1e-8)
    for y in [0.,20.]:
        a=logsf(p,torch.tensor([[y-1e-8]],dtype=torch.float64),t,s)
        b=logsf(p,torch.tensor([[y+1e-8]],dtype=torch.float64),t,s)
        assert float(abs(a-b))<1e-6


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('variant',['plain','tied','crossed'])
def test_upper_gradient(dtype,variant):
    p,t,s=fixture(dtype)
    if variant=='tied':p.zero_()
    if variant=='crossed':p=p.flip(1)
    shift=torch.tensor(0.,dtype=dtype,requires_grad=True)
    losses=[-logsf(p+shift,torch.tensor([[y]],dtype=dtype),t,s).sum() for y in [1e5,1e6]]
    grad=torch.autograd.grad(losses[-1],shift)[0]
    assert torch.isfinite(grad) and grad<0 and losses[1]>losses[0]
    old=(p.clone().requires_grad_())
    loss=-(1-cdf(old,torch.tensor([[1e6]],dtype=dtype))).clamp_min(1e-6).log().sum()
    assert torch.autograd.grad(loss,old)[0].abs().sum()==0


def test_gradcheck():
    p,t,s=fixture();p.requires_grad_()
    assert torch.autograd.gradcheck(lambda v:logsf(v,torch.tensor([[21.3]],dtype=torch.float64),t,s),(p,),eps=1e-6,atol=1e-5,rtol=1e-4)
    assert torch.autograd.gradcheck(lambda v:logsf(v,torch.tensor([[6.37]],dtype=torch.float64),t,s),(p,),eps=1e-6,atol=1e-5,rtol=1e-4)


def update(arm,mask,lc):
    z=torch.nn.Parameter(torch.linspace(-1.,1.,21)[:,None].expand(21,4)[None].clone())
    opt=torch.optim.AdamW([z],lr=1e-4,weight_decay=0)
    y=torch.tensor([[0.,1.,2.,float('nan')]])
    loss,task,tail=objective(arm,z,z.sinh(),y,mask,torch.zeros(1,1),torch.ones(1,1),torch.ones(1),lc)
    loss.backward();g=z.grad.clone();opt.step()
    return loss.detach(),g,z.detach()


def test_zero_lambda_and_no_censor_updates():
    mask=torch.tensor([[False,True,True,False]])
    for a,b in zip(update('DROP',mask,1),update('TAIL',mask,0)): assert torch.equal(a,b)
    mask.zero_()
    for arm in ['DROP','TAIL']:
        for a,b in zip(update('NAIVE',mask,1),update(arm,mask,1)):assert torch.equal(a,b)


def test_all_censored_missing_and_normalization():
    p,t,s=fixture(); p=p.expand(1,21,3).clone().requires_grad_()
    sale=torch.tensor([[21.,22.,float('nan')]],dtype=torch.float64)
    m=torch.tensor([[True,True,False]])
    loss=survival_loss(p,sale,m,s)
    expect=-logsf(p[:,:,:2],sale[:,:2],t,s).mean()
    assert torch.equal(loss,expect)
    assert torch.isfinite(torch.autograd.grad(loss,p)[0]).all()
    with pytest.raises(ValueError):survival_loss(p,sale,torch.ones_like(m),s)
    with pytest.raises(ValueError):survival_loss(p,torch.full_like(sale,float('nan')),torch.zeros_like(m),s)
    loss,task,_=objective('TAIL',p,p,sale,m,torch.zeros(1,1),torch.ones(1,1),s,1)
    assert task==0 and loss>0
    # No-censor does not inspect invalid raw at missing positions.
    raw=p.detach().clone();raw[:,:,2]=float('nan');raw.requires_grad_()
    assert survival_loss(raw,sale,torch.zeros_like(m),s)==0


def test_hidden_excess_invariance_and_gain():
    caps=np.array([1.,3.]); original=np.array([[0.,5.],[2.,1.],[8.,7.]])
    mask=original>caps; altered=original.copy();altered[mask]+=1000
    for v in [altered]:
        assert np.array_equal(np.minimum(original,caps),np.minimum(v,caps))
        assert np.array_equal(mask,v>caps)
        assert np.array_equal(np.std(np.minimum(original,caps),axis=0),np.std(np.minimum(v,caps),axis=0))
    from importlib.util import spec_from_file_location,module_from_spec
    from pathlib import Path
    spec=spec_from_file_location('censor_runner',Path(__file__).parents[1]/'scripts/run_censor_tail_controlled_v1.py')
    runner=module_from_spec(spec);spec.loader.exec_module(runner)
    for candidate,expected in [(.5,0),(.45,10),(.55,-10)]:assert runner.gain(.5,candidate)==pytest.approx(expected)
    assert runner.gain(0,1) is None


def test_subgroup_contributions_and_bias():
    from importlib.util import spec_from_file_location,module_from_spec
    from pathlib import Path
    spec=spec_from_file_location('censor_metrics',Path(__file__).parents[1]/'scripts/run_censor_tail_controlled_v1.py')
    runner=module_from_spec(spec);spec.loader.exec_module(runner)
    y=np.array([[[0.,4.],[1.,8.]],[[2.,0.],[0.,1.]]])
    p=np.zeros((2,2,21,2));scale=np.array([1.,2.]);caps=np.array([1.,3.])
    r,v=runner.metric_vectors(p,y,scale,caps)
    assert r['all']['primary']==pytest.approx((y/scale[None,:,None]).mean())
    assert r['all']['primary']==pytest.approx(r['censored']['contribution']+r['uncensored']['contribution'])
    assert r['all']['signed_bias']==pytest.approx(-y.mean())
    assert r['censored']['positions']==3 and r['uncensored']['positions']==5
