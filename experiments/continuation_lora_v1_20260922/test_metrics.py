import numpy as np
from metrics import point_loss, empirical, calibrate, apply, bootstrap

def test_pinball_scalar_and_empirical_pairwise():
    rng=np.random.default_rng(77)
    y=rng.normal(size=(2,3,128))
    q=rng.normal(size=(2,3,128,9))
    sigma=np.array([.4,1.,3.])
    score=point_loss(y,q,sigma)
    for i,s,h in ((0,0,0),(1,1,63),(1,2,127)):
        scalar=0.
        for k,tau in enumerate(np.arange(1,10)/10):
            e=y[i,s,h]-q[i,s,h,k]
            scalar+=2*(tau*e if e>=0 else (tau-1)*e)/9/sigma[s]
        np.testing.assert_allclose(score[i,s,h],scalar,atol=1e-12)
    z=rng.normal(size=(2,81))
    target=np.array([.1,.4])
    ref=np.abs(z-target[:,None]).mean(-1)-.5*np.abs(z[:,:,None]-z[:,None,:]).mean((1,2))
    np.testing.assert_allclose(empirical(z,target),ref,atol=1e-12)

def test_affine_grid_and_application():
    q=np.tile(np.arange(-4,5),(8,2,128,1)).astype(float)
    y=np.zeros((8,2,128))
    sigma=np.array([1.,2.])
    params=calibrate(y,q,sigma)
    assert all(p['alpha']==.5 and p['beta']==0 for p in params)
    np.testing.assert_allclose(apply(q,sigma,params),q*.5)
    assert all(len(p['grid'])==35 for p in params)

def test_bootstrap_pairing_and_constant_effect():
    a=np.tile(np.arange(30)+2.,(2,1))
    b=a-1.
    result=bootstrap(a,b)
    assert result['effect']==result['ci95_low']==result['ci95_high']==1.
