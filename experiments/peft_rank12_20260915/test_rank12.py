import copy,io
import numpy as np
import torch
from rank2_model import Channel,ARMS,count
from rank2_data import metrics,independent,loss

def test_budget_counts():
    expected={'INDIV_REF':4474112,'INDIV_BUDGET':295952,'SHARED_BUDGET':295103,'FACTOR_BUDGET':295935,'BASIS_BUDGET':295103}
    for a,n in expected.items():
        m=Channel(a,list(range(32)));assert sum(p.numel() for p in m.parameters())==n==count(a)
        if a!='INDIV_REF':assert abs(n-295952)/295952<=.005

def test_basis_merge_output_gradient_shared_and_ids():
    ids=['a','b','c'];m=Channel('BASIS_BUDGET',ids,d=7,width=4,seed=2).double().eval()
    with torch.no_grad():
        for layer in m.up_projections[1:]:layer.weight.normal_(0,.1);layer.bias.normal_(0,.1)
    h=torch.randn(2,3,5,7,dtype=torch.float64);f=torch.randn_like(h)
    a=m(h,f);ga=torch.autograd.grad(a.square().sum(),tuple(m.parameters()))
    b=m(h,f,merged=True);gb=torch.autograd.grad(b.square().sum(),tuple(m.parameters()))
    torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-9)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-9)
    perm=[2,0,1];torch.testing.assert_close(m(h[:,perm],f[:,perm],[ids[i] for i in perm]),m(h,f)[:,perm],atol=1e-11,rtol=1e-9)
    with torch.no_grad():m.coefficients.zero_()
    z=m.dropout(m.activation(m.down_projection(torch.cat([h,f],-1))));torch.testing.assert_close(m(h,f),m.layer_norm(m.up_projections[0](z)),atol=1e-11,rtol=1e-9)
    buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0);restored=Channel('BASIS_BUDGET',ids,d=7,width=4,seed=4).double().eval();restored.load_state_dict(torch.load(buf,weights_only=True));torch.testing.assert_close(restored(h,f),m(h,f),atol=0,rtol=0)

def test_all_channel_permutations():
    ids=['x','y','z'];h=torch.randn(2,3,5,7,dtype=torch.float64);f=torch.randn_like(h);perm=[2,0,1]
    for a in ARMS[1:]:
        m=Channel(a,ids,d=7,width=4,q=3).double().eval();torch.testing.assert_close(m(h[:,perm],f[:,perm],[ids[i] for i in perm]),m(h,f)[:,perm],atol=1e-11,rtol=1e-9)

def test_mask_denominators_and_partial_batch():
    rng=np.random.default_rng(7);p=rng.normal(size=(9,3,4));y=rng.normal(size=p.shape);y[0,0]=np.nan;y[8,1,2:]=np.nan
    m=metrics(p,y,np.ones(3));assert abs(m['mse']-independent(p,y))<1e-12
    pp=torch.tensor(p,dtype=torch.float32,requires_grad=True);yy=torch.tensor(y,dtype=torch.float32);counts=torch.isfinite(yy).sum((0,2))
    whole=loss(pp,yy);split=sum(loss(pp[i:i+4],yy[i:i+4],counts) for i in range(0,9,4));torch.testing.assert_close(whole,split)
    torch.testing.assert_close(torch.autograd.grad(whole,pp)[0],torch.autograd.grad(split,pp)[0])
