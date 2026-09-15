import unittest
import numpy as np
import torch
from candidate import geometry,penalty
class CandidateContracts(unittest.TestCase):
    def test_explicit_matrix_value_gradient_fp64(self):
        z=torch.arange(12,dtype=torch.float64).reshape(3,4).requires_grad_();p=torch.ones(12,12,dtype=torch.float64)/12;mat=.2*p+1.3*(torch.eye(12,dtype=torch.float64)-p)
        a=penalty(z,.2,1.3);b=z.flatten()@mat@z.flatten()/12
        ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(b,z)[0]
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12);torch.testing.assert_close(ga,gb,atol=1e-12,rtol=1e-12)
    def test_legal_difference_not_random_init(self):
        residual=np.array([[2,1,3,2],[2,3,1,2],[2,1,3,2]],float);g=geometry(residual,12)
        z=torch.ones((3,4),dtype=torch.float64);self.assertLess(penalty(z,g['w_level'],g['w_shape']).item(),penalty(z,1,1).item())
    def test_equal_precision_and_uniform_equivalence(self):
        g=geometry(np.zeros((2,24)),504);self.assertEqual(g['w_level'],1.);self.assertEqual(g['w_shape'],1.)
        d=torch.randn(3,4,dtype=torch.float64);torch.testing.assert_close(penalty(d,1,1),d.square().mean())
    def test_trace_normalization_no_weaker_uniform_shortcut(self):
        for n in [2,13]:
            g=geometry(np.random.default_rng(n).normal(size=(n,24)),504);self.assertAlmostEqual((g['w_level']+503*g['w_shape'])/504,1)
    def test_zero_update_penalty_gradient(self):
        d=torch.zeros((21,24),dtype=torch.float64,requires_grad=True);v=penalty(d,.2,1.1);v.backward();self.assertEqual(v.item(),0);self.assertEqual(d.grad.abs().sum().item(),0)
if __name__=='__main__':unittest.main()
