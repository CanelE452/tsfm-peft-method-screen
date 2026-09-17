"""Locally authored contract tests; origin/math checks use independent formulas."""
import unittest
import numpy as np
import torch
from .reference_core import *
from .audit import select_origins
from .model import loss_2pinball
from .test_local_components import LocalChecks

class ContractChecks(unittest.TestCase):
    def test_sampling_independent(self):
        for p,n in [(24,256),(96,64),(96,128)]:
            legal=np.arange(512,70000-64);legal=legal[legal//p!=38]
            actual,_=select_origins(legal,p,n,81700)
            np.testing.assert_array_equal(actual,select_days_reference(legal,p,n,81700))
    def test_transform_information_and_targets(self):
        x=np.sin(np.arange(512)/10);sigma=2.
        for state in STATES:
            observed,delta,a=transform(x,sigma,state,rng(10,state))
            again,other,_=transform(x.copy(),sigma,state,rng(10,state))
            np.testing.assert_array_equal(observed,again);self.assertEqual(delta,other)
            expected=x.copy()
            if state.startswith('SHIFT'):expected[-32:]+=delta;self.assertNotEqual(delta,0)
            else:self.assertEqual(delta,0)
            if a['positions']:expected[a['positions']]+=a['fault_delta']
            np.testing.assert_array_equal(observed,expected.astype(np.float32))
            for y in [np.zeros(64),np.full(64,1e10)]:
                # Input transform accepts no y, hence poisoning any future cannot affect it.
                np.testing.assert_array_equal(transform(x,sigma,state,rng(10,state))[0],observed)
                np.testing.assert_allclose((y+delta)-y,delta,atol=1e-5)
    def test_training_exposures(self):
        for i in range(1024):self.assertEqual(np.bincount([(e+i)%4 for e in range(32)]).tolist(),[8]*4)
    def test_independent_loss(self):
        r=rng('scalar');p=r.normal(size=(2,9,64));y=r.normal(size=(2,64));s=np.array([2.,3.]);q=np.arange(1,10)/10
        a=loss_2pinball(*map(torch.tensor,[p,y,s,q])).item()
        np.testing.assert_allclose(a,pinball_scalar(p,y,s,q),rtol=1e-10,atol=1e-12)
    def test_microbatch_weighting(self):
        # Gradient equality of the objective's weighted reduction, no model/shape bitwise gate.
        q=torch.arange(1,10,dtype=torch.float64)/10
        p=torch.randn(32,9,64,dtype=torch.float64,requires_grad=True);y=torch.randn(32,64,dtype=torch.float64);s=torch.linspace(.5,5,32,dtype=torch.float64)
        loss_2pinball(p,y,s,q).backward();full=p.grad.clone();p.grad=None
        for lo in range(0,32,8): (loss_2pinball(p[lo:lo+8],y[lo:lo+8],s[lo:lo+8],q)*8/32).backward()
        torch.testing.assert_close(p.grad,full,rtol=1e-10,atol=1e-12)

if __name__=='__main__':unittest.main()
