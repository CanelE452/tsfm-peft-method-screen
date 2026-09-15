"""Small explicit FP64 operator/gradient and mechanism distinction checks."""
import torch
from model import Transport,ARMS

def verify_mechanism():
    torch.manual_seed(9016);h=torch.randn(1,2,12,6,dtype=torch.float64);x=torch.randn(1,2,12,8,dtype=torch.float64)
    rows=[];outputs={}
    for arm in ARMS:
        m=Transport(arm,d=6,r=3,seed=9016).double()
        assert torch.equal(m(h,x),h)
        with torch.no_grad():m.up.weight.copy_(torch.arange(m.up.weight.numel(),dtype=torch.float64).reshape_as(m.up.weight)/17-.4)
        y=m(h,x);terms=[]
        for b in range(1):
            channels=[]
            for c in range(2):
                patches=[]
                for i in range(12):
                    js=[i] if arm=='POINTWISE' else [j for j in range(12) if (j-i)%3==0]
                    logits=torch.stack([-((x[b,c,i]-x[b,c,j])**2).mean() if arm=='CONDITIONED' else x.new_tensor(0.) for j in js])
                    weights=torch.softmax(logits,0)
                    mixed=sum(w*(m.down.weight@h[b,c,j]) for w,j in zip(weights,js))
                    patches.append(h[b,c,i]+m.up.weight@torch.nn.functional.gelu(mixed))
                channels.append(torch.stack(patches))
            terms.append(torch.stack(channels))
        ref=torch.stack(terms)
        g=torch.autograd.grad(y.square().sum(),tuple(m.parameters()),retain_graph=True)
        gg=torch.autograd.grad(ref.square().sum(),tuple(m.parameters()))
        output_diff=float((y-ref).abs().max());gradient_diff=max(float((a-b).abs().max()) for a,b in zip(g,gg))
        assert output_diff<1e-10 and gradient_diff<1e-10
        t=m.weights(x);assert torch.allclose(t.sum(-1),torch.ones_like(t.sum(-1)),rtol=0,atol=1e-12)
        outputs[arm]=y.detach();rows.append(dict(arm=arm,initial_identity_exact=True,max_output_diff=output_diff,max_gradient_diff=gradient_diff))
    assert not torch.allclose(outputs['POINTWISE'],outputs['UNIFORM'])
    assert not torch.allclose(outputs['CONDITIONED'],outputs['UNIFORM'])
    repeated=x[:,:,:3].repeat(1,1,4,1)
    a=Transport('CONDITIONED',d=6,r=3);b=Transport('UNIFORM',d=6,r=3)
    assert torch.equal(a.weights(repeated),b.weights(repeated))
    # Moving channel order moves outputs; no channel ID or cross-channel input.
    m=Transport('CONDITIONED',d=6,r=3).double()
    with torch.no_grad():m.up.weight.fill_(.1)
    assert torch.equal(m(h.flip(1),x.flip(1)),m(h,x).flip(1))
    return dict(fp64=rows,conditioned_differs_from_uniform=True,uniform_differs_from_pointwise=True,repeated_cycle_reduces_to_uniform=True,channel_permutation_equivariant=True)
