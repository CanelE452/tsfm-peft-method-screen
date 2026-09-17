import unittest
import numpy as np
import torch
from .model import coordinates,restore_input,ForecastModel,loss_2pinball,preprocess
from .reference_core import pinball_scalar

class Components(unittest.TestCase):
    def test_trailing_persistence_scalar(self):
        r=np.random.default_rng(83600);x=r.normal(size=(3,512)).astype(np.float32);x[0,-32:]+=10;x[1,[40,140,240]]-=20
        t=torch.tensor(x);s=torch.ones(3);clip,e,I,p=coordinates(t,s)
        m=np.median(x,axis=1);scale=np.maximum(1.4826*np.median(abs(x-m[:,None]),axis=1),.1);d=(x-m[:,None])/scale[:,None]
        expected=np.zeros_like(x)
        for b in range(3):
            for j in range(512):
                if abs(d[b,j])>3:
                    w=d[b,max(0,j-7):j+1];expected[b,j]=np.mean((abs(w)>3)&(np.sign(w)==np.sign(d[b,j])))
        np.testing.assert_allclose(p.numpy(),expected,rtol=1e-6,atol=1e-7)
        self.assertTrue(torch.equal(restore_input(t,clip,torch.zeros_like(p)),clip))
        torch.testing.assert_close(restore_input(t,clip,torch.ones_like(p)),t,rtol=1e-6,atol=1e-6)
        self.assertTrue(torch.equal(clip,preprocess(t,s,'A5')))
    def test_gate_formula_and_grad(self):
        x=torch.randn(2,512,generator=torch.Generator().manual_seed(1));x[:,-32:]+=20;s=torch.ones(2)
        for arm in ['B4','B5']:
            model=ForecastModel(torch.nn.Identity(),arm,81550)
            a,g,p,e=model.transform(x,s);self.assertTrue(((g>=0)&(g<=1)).all())
            self.assertTrue(torch.equal(a,model.transform(x,s)[0]))
            a.square().mean().backward();self.assertTrue(torch.isfinite(model.gate.grad).all());self.assertGreater(float(model.gate.grad.abs().sum()),0)
            self.assertEqual(model.gate.numel(),3 if arm=='B4' else 4)
    def test_pinball_independent(self):
        rr=np.random.default_rng(2);p=rr.normal(size=(3,9,64));y=rr.normal(size=(3,64));s=np.array([1.,2.,3.]);q=np.arange(1,10)/10
        expected=pinball_scalar(p,y,s,q);v=loss_2pinball(torch.tensor(p),torch.tensor(y),torch.tensor(s),torch.tensor(q))
        self.assertAlmostEqual(float(v),expected,places=12)
    def test_no_metadata_forward_signature(self):
        import inspect
        self.assertEqual(list(inspect.signature(ForecastModel.forward).parameters),['self','observed','sigma','residual_mode','persistence_mode'])
        self.assertEqual(list(inspect.signature(ForecastModel.transform).parameters),['self','observed','sigma','persistence_mode'])

if __name__=='__main__':unittest.main()
