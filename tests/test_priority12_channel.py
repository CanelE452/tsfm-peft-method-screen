import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pytest
# Optional isolated environment: main Q environment must remain unchanged.
pytest.importorskip('momentfm')
import torch
from priority12.channel_model import ChannelBlock,channel_count,groups

def test_counts_and_no_parameter_alias():
    expected={'SPECIFIC':8684800,'SHARED':395008,'SHARED_WIDE':790017,'GROUP4':789760,'BASIS4':790016}
    for a,n in expected.items():
        m=ChannelBlock(a,list(range(64)));assert sum(p.numel() for p in m.parameters())==n==channel_count(a)
        if a=='SPECIFIC':assert len({p.weight.data_ptr() for p in m.up_projections})==64
    assert [groups(list(range(64))).count(i) for i in range(4)]==[16]*4

def test_basis_effective_matrix_reductions_and_channel_ids():
    ids=['a','b','c','d'];m=ChannelBlock('BASIS4',ids,d=5,r=3,k=4).double().eval();h=torch.randn(2,4,3,5,dtype=torch.float64);f=torch.randn_like(h)
    with torch.no_grad():m.coefficients.copy_(torch.randn_like(m.coefficients))
    u=m.activation(m.down_projection(torch.cat([h,f],-1)))
    w=torch.stack([p.weight for p in m.up_projections]);b=torch.stack([p.bias for p in m.up_projections])
    ew=torch.einsum('ck,kdr->cdr',m.coefficients,w);eb=m.coefficients@b
    manual=m.layer_norm(torch.einsum('bcnr,cdr->bcnd',u,ew)+eb[None,:,None,:])
    torch.testing.assert_close(m(h,f),manual,atol=1e-10,rtol=1e-10)
    perm=[2,0,3,1];torch.testing.assert_close(m(h[:,perm],f[:,perm],[ids[i] for i in perm]),m(h,f)[:,perm],atol=1e-10,rtol=1e-10)
    with torch.no_grad():m.coefficients.zero_();m.coefficients[:,0]=1
    torch.testing.assert_close(m(h,f),m.layer_norm(m.up_projections[0](u)),atol=1e-10,rtol=1e-10)
    with torch.no_grad():m.coefficients.copy_(torch.eye(4,dtype=torch.float64))
    expected=m.layer_norm(torch.stack([m.up_projections[i](u[:,i]) for i in range(4)],1))
    torch.testing.assert_close(m(h,f),expected,atol=1e-10,rtol=1e-10)
    k1=ChannelBlock('BASIS4',ids,d=5,r=3,k=1).double().eval()
    u1=k1.activation(k1.down_projection(torch.cat([h,f],-1)))
    torch.testing.assert_close(k1(h,f),k1.layer_norm(k1.up_projections[0](u1)),atol=1e-10,rtol=1e-10)

def test_initial_functions_and_eventual_active_paths():
    ids=list(range(4));h=torch.randn(2,4,3,5,dtype=torch.float64);f=torch.randn_like(h);ys=[]
    for a in ['SPECIFIC','SHARED','SHARED_WIDE','GROUP4','BASIS4']:
        m=ChannelBlock(a,ids,d=5,r=3,k=4,wide=7).double().eval();ys.append(m(h,f))
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        for step in range(3):
            opt.zero_grad();(m(h,f)-torch.arange(5,dtype=torch.float64)).square().mean().backward();opt.step()
        if a=='BASIS4':assert m.up_projections[1].weight.grad.abs().sum()>0
        if a=='SHARED_WIDE':assert m.down_projection.weight.grad[3:].abs().sum()>0
    for y in ys:torch.testing.assert_close(y,ys[0],atol=1e-10,rtol=1e-10)
