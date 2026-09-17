import unittest
import torch,numpy as np
from .model import ResidualAdapter,correction_gate,loss_2pinball
from .reference_core import pinball_scalar

class Components(unittest.TestCase):
    def test_persistence_independent(self):
        rr=np.random.default_rng(84600);x=rr.normal(size=(3,512)).astype(np.float32);x[0,-32:]+=8;x[1,[30,80,300]]-=12
        s=np.ones(3,np.float32);m=np.median(x,axis=-1);r=np.maximum(1.4826*np.median(abs(x-m[:,None]),axis=-1),.1*s);d=(x-m[:,None])/r[:,None];p=np.zeros_like(x)
        for b in range(3):
            for i in range(512):
                w=d[b,max(0,i-7):i+1]
                if abs(d[b,i])>3:p[b,i]=np.mean((abs(w)>3)&(np.sign(w)==np.sign(d[b,i])))
        actual=correction_gate(torch.tensor(x),torch.tensor(s)).numpy()
        np.testing.assert_allclose(actual,1-p.reshape(3,32,16).mean(-1),rtol=1e-6,atol=1e-7)
        self.assertTrue(((actual>=0)&(actual<=1)).all())
    def test_identity_and_matching(self):
        e=torch.randn(4,32,512,generator=torch.Generator().manual_seed(3));g=torch.rand(4,32)
        a=ResidualAdapter(81550);b=ResidualAdapter(81550)
        self.assertEqual(sum(p.numel() for p in a.parameters()),8712)
        self.assertTrue(torch.equal(a(e,g),e))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))
        with torch.no_grad():a.up.weight.fill_(.001)
        self.assertTrue(torch.equal(a(e,torch.zeros_like(g)),e))
        self.assertFalse(torch.equal(a(e,torch.ones_like(g)),e))
    def test_scalar_pinball(self):
        rr=np.random.default_rng(9);p=rr.normal(size=(2,9,64));y=rr.normal(size=(2,64));s=np.array([1.,2.]);q=np.arange(1,10)/10
        actual=loss_2pinball(torch.tensor(p),torch.tensor(y),torch.tensor(s),torch.tensor(q))
        self.assertAlmostEqual(float(actual),pinball_scalar(p,y,s,q),places=12)
    def test_information_signature(self):
        import inspect
        from .model import ForecastModel
        self.assertEqual(list(inspect.signature(ForecastModel.forward).parameters),['self','observed','sigma','residual_mode','persistence_mode'])

if __name__=='__main__':unittest.main()
