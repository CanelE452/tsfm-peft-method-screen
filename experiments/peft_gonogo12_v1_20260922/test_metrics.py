import numpy as np
from metrics import affine, point_loss


def test_pinball_matches_scalar_oracle_and_scale():
    rng=np.random.default_rng(22)
    q=rng.normal(size=(3,24,9));y=rng.normal(size=(3,24));sigma=np.array([.5,1.,2.])
    expected=[]
    for n in range(3):
        total=0.
        for h in range(24):
            for k,p in enumerate(sorted(q[n,h])):
                error=y[n,h]-p;tau=(k+1)/10
                total+=2*(tau*error if error>=0 else (tau-1)*error)
        expected.append(total/(24*9*sigma[n]))
    np.testing.assert_allclose(point_loss(q,y,sigma),expected,rtol=1e-14)
    np.testing.assert_allclose(point_loss(7*q,7*y,7*sigma),expected,rtol=1e-14)


def test_affine_has_positive_bounded_scale_and_no_hidden_weights():
    m=np.arange(48,dtype=float).reshape(2,24)
    q=np.repeat(m[...,None],9,axis=-1)
    a,b=affine(q,1.2*m+3)
    np.testing.assert_allclose([a,b],[1.2,3],atol=1e-12)
    a,b=affine(q,4*m+3)
    assert a==1.5
    np.testing.assert_allclose(b,np.mean(4*m+3)-1.5*m.mean())
    a,b=affine(np.zeros_like(q),np.full((2,24),5.))
    np.testing.assert_allclose([a,b],[1.,5.],atol=1e-14)


def test_h2_calibration_condition_weight_is_not_repeat_count():
    m=np.array([np.arange(24),np.arange(24),np.arange(24)],dtype=float)
    q=np.repeat(m[...,None],9,axis=-1)
    y=m+np.array([0.,10.,10.])[:,None]
    a,b=affine(q,y,weights=np.array([.5,.25,.25]))
    np.testing.assert_allclose([a,b],[1.,5.],atol=1e-12)
