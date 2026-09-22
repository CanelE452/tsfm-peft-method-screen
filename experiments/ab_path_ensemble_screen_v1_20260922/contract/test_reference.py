import unittest
import torch
from reference_core import (weighted_crps, cdf_l2, discrete_quantiles,
                            equal_atoms, medoids3, PathRouter, ScenarioPool)


def pairwise(z, p, y):
    p = p.expand_as(z); p = p / p.sum(-1, keepdim=True)
    return (p*(z-y[...,None]).abs()).sum(-1) - 0.5*(
        p[..., :, None]*p[..., None, :]*(z[..., :, None]-z[..., None, :]).abs()).sum((-2,-1))


class Tests(unittest.TestCase):
    def setUp(self): torch.manual_seed(712)
    def test_single_atom(self):
        self.assertEqual(weighted_crps(torch.tensor([3.]),torch.tensor([1.]),torch.tensor(1.)).item(),2.)
    def test_known_two_atoms(self):
        self.assertAlmostEqual(weighted_crps(torch.tensor([0.,2.]),torch.tensor([.5,.5]),torch.tensor(1.)).item(),.5)
    def test_sorted_matches_pairwise(self):
        z=torch.randn(3,8,27,dtype=torch.float64);p=torch.rand_like(z);y=torch.randn(3,8,dtype=torch.float64)
        torch.testing.assert_close(weighted_crps(z,p,y),pairwise(z,p,y))
    def test_450_supports(self):
        z=torch.randn(1,2,450,dtype=torch.float64);p=torch.rand_like(z);y=torch.randn(1,2,dtype=torch.float64)
        torch.testing.assert_close(weighted_crps(z,p,y),pairwise(z,p,y))
    def test_score_permutation(self):
        z=torch.randn(2,9);p=torch.rand_like(z);y=torch.randn(2);ix=torch.randperm(9)
        torch.testing.assert_close(weighted_crps(z,p,y),weighted_crps(z[:,ix],p[:,ix],y))
    def test_location_equivariance(self):
        z=torch.randn(2,9);p=equal_atoms(z);y=torch.randn(2)
        torch.testing.assert_close(weighted_crps(z,p,y),weighted_crps(z+4,p,y+4))
    def test_scale_equivariance(self):
        z=torch.randn(2,9);p=equal_atoms(z);y=torch.randn(2)
        torch.testing.assert_close(3*weighted_crps(z,p,y),weighted_crps(3*z,p,3*y))
    def test_duplicate_distribution(self):
        z=torch.randn(2,9);p=equal_atoms(z);y=torch.randn(2)
        torch.testing.assert_close(weighted_crps(z,p,y),weighted_crps(z.repeat_interleave(2,-1),p.repeat_interleave(2,-1)/2,y))
    def test_invalid_weights(self):
        with self.assertRaises(ValueError): weighted_crps(torch.tensor([1.,2.]),torch.tensor([-1.,2.]),torch.tensor(1.))
    def test_zero_mass(self):
        with self.assertRaises(ValueError): weighted_crps(torch.tensor([1.,2.]),torch.zeros(2),torch.tensor(1.))
    def test_energy_zero(self):
        z=torch.randn(2,9,dtype=torch.float64);p=equal_atoms(z)
        torch.testing.assert_close(cdf_l2(z,p,z,p),torch.zeros(2,dtype=torch.float64),atol=1e-12,rtol=0)
    def test_energy_positive(self):
        z=torch.randn(4,9);t=torch.randn(4,27)
        self.assertTrue((cdf_l2(z,equal_atoms(z),t,equal_atoms(t))>=-1e-6).all())
    def test_weights_receive_gradient(self):
        z=torch.tensor([[0.,1.,3.]],requires_grad=True);l=torch.tensor([[.2,-.2,.1]],requires_grad=True)
        weighted_crps(z,l.softmax(-1),torch.tensor([2.])).sum().backward()
        self.assertGreater(l.grad.abs().sum().item(),0);self.assertGreater(z.grad.abs().sum().item(),0)
    def test_discrete_quantiles(self):
        z=torch.tensor([[0.,10.,20.]]);p=torch.tensor([[.2,.5,.3]])
        q=discrete_quantiles(z,p,torch.tensor([.1,.5,.9]))
        torch.testing.assert_close(q,torch.tensor([[0.,10.,20.]]))
    def test_medoid_three_clusters(self):
        x=torch.tensor([0.,0.,0.,10.,10.,10.,20.,20.,20.])[None,:,None].expand(1,9,64)
        z,w,ids=medoids3(x)
        torch.testing.assert_close(z[:, :, 0],torch.tensor([[0.,10.,20.]]))
        torch.testing.assert_close(w,torch.full((1,3),1/3));self.assertEqual(ids.tolist(),[[0,3,6]])
    def test_router_initial_parity(self):
        a=PathRouter(False);b=PathRouter(True);x=torch.randn(2,9,64)
        for av,bv in zip(a(x),b(x)): torch.testing.assert_close(av,bv)
    def test_router_count(self):
        self.assertEqual(sum(p.numel() for p in PathRouter(True).parameters()),4886)
        self.assertEqual(sum(p.numel() for p in PathRouter(False).parameters()),30)
    def test_router_convexity(self):
        x=torch.randn(2,9,64);z,w,a=PathRouter(True)(x)
        self.assertTrue((z>=x.min(1).values[:,None]-1e-6).all())
        self.assertTrue((z<=x.max(1).values[:,None]+1e-6).all())
        torch.testing.assert_close(w.sum(-1),torch.ones(2));torch.testing.assert_close(a.sum(-1),torch.ones(2,3))
    def test_router_context_gradient(self):
        m=PathRouter(True);opt=torch.optim.AdamW(m.parameters(),lr=.001);x=torch.randn(2,9,64)
        for _ in range(2):
            opt.zero_grad();z,w,_=m(x);loss=(w[:,:,None]*z.square()).mean();loss.backward();opt.step()
        self.assertGreater(m.down.weight.grad.abs().sum().item(),0)
        self.assertGreater(m.up.weight.grad.abs().sum().item(),0)
    def test_scenarios_permutation(self):
        m=ScenarioPool(40);x=torch.randn(3,50,40);c=torch.randn(3,80);ix=torch.randperm(50)
        z,w,_=m(x,c);zz,ww,_=m(x[:,ix],c)
        torch.testing.assert_close(z,zz);torch.testing.assert_close(w,ww)
    def test_scenarios_gradient(self):
        m=ScenarioPool(40);x=torch.randn(2,50,40);c=torch.randn(2,80)
        z,w,_=m(x,c);(w[:,:,None]*z.square()).mean().backward()
        self.assertGreater(m.queries.weight.grad.abs().sum().item(),0)
        self.assertGreater(m.encoder[0].weight.grad.abs().sum().item(),0)
    def test_scenarios_same_path(self):
        m=ScenarioPool(40);x=torch.randn(2,1,40).expand(2,50,40);z,w,_=m(x,torch.randn(2,80))
        torch.testing.assert_close(z,x[:,:3])

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
