"""Explicit FP64 formula, gradient and prior-neutrality checks."""
import math
import torch
from model import SideBlock,ARMS

def verify_mechanism():
    torch.manual_seed(9019)
    h=torch.randn(2,4,6,dtype=torch.float64,requires_grad=True)
    side=torch.randn(2,4,6,dtype=torch.float64,requires_grad=True)
    logp=torch.randn(2,4,4,dtype=torch.float64).log_softmax(-1)
    records=[];outputs={};state=None
    for arm in ARMS:
        m=SideBlock(arm,d=6,r=3).double()
        if state is not None:m.load_state_dict(state)
        assert torch.equal(m(h,side,logp),side)
        with torch.no_grad():m.up.weight.copy_(torch.arange(18,dtype=torch.float64).reshape(6,3)/19-.4)
        if state is None:
            state={k:v.clone() for k,v in m.state_dict().items()};state['up.weight'].zero_()
        y=m(h,side,logp);x=h+side
        z=(x-x.mean(-1,keepdim=True))/torch.sqrt(x.var(-1,unbiased=False,keepdim=True)+1e-5)
        z=z*m.norm.weight+m.norm.bias
        q=z@m.q.weight.T;k=z@m.k.weight.T;v=z@m.v.weight.T
        rows=[]
        for b in range(2):
            tokens=[]
            for i in range(4):
                scores=torch.stack([torch.dot(q[b,i],k[b,j])/math.sqrt(3)+(logp[b,i,j] if arm=='PRIOR' else 0) for j in range(4)])
                p=scores.softmax(0)
                tokens.append(side[b,i]+m.up.weight@sum(p[j]*v[b,j] for j in range(4)))
            rows.append(torch.stack(tokens))
        ref=torch.stack(rows);params=tuple(m.parameters())+(h,side)
        g=torch.autograd.grad(y.square().sum(),params,retain_graph=True)
        gg=torch.autograd.grad(ref.square().sum(),params)
        e=float((y-ref).abs().max().detach());ge=max(float((a-b).abs().max()) for a,b in zip(g,gg))
        assert e<1e-10 and ge<1e-10
        outputs[arm]=y.detach();records.append(dict(arm=arm,output_error=e,gradient_error=ge,zero_up_identity=True))
    assert not torch.allclose(outputs['SIDE'],outputs['PRIOR'])
    a=SideBlock('SIDE',d=6,r=3).double();b=SideBlock('PRIOR',d=6,r=3).double();b.load_state_dict(a.state_dict())
    with torch.no_grad():a.up.weight.fill_(.2);b.up.weight.copy_(a.up.weight)
    neutral=torch.full_like(logp,-math.log(4))
    err=float((a(h,side,neutral)-b(h,side,neutral)).abs().max().detach());assert err<1e-12
    return dict(fp64=records,same_weights_nonuniform_prior_changes_output=True,uniform_prior_reduces_to_SIDE_error=err)
