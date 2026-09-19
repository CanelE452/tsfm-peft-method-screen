import unittest
import numpy as np
import torch
from .core import probes,penalty
class CoreTests(unittest.TestCase):
    def test_probe_contract_and_reordering(self):
        ids=np.arange(64);s=np.linspace(.1,100,64).astype('float32')
        u,v,a,d=probes('electricity',2,ids,s)
        self.assertTrue(torch.equal(u.sort(-1).values,v.sort(-1).values))
        self.assertTrue(np.array_equal((u!=0).sum(-1).numpy(),d))
        self.assertTrue(torch.equal(u[:,-1],a));self.assertTrue(set(d)<=set([16,32,64,128]))
        self.assertTrue(torch.all(a.abs()/torch.tensor(s)>=2));self.assertTrue(torch.all(a.abs()/torch.tensor(s)<=8))
        ur,vr,ar,dr=probes('electricity',2,ids[::-1],s[::-1].copy())
        self.assertTrue(torch.equal(u,ur.flip(0)));self.assertTrue(torch.equal(v,vr.flip(0)))
        self.assertFalse(torch.equal(u,v))
    def test_response_constant_residual_not_anchoring(self):
        b=torch.zeros(2,9,64);bt=b+4;s=torch.ones(2);a=s*4;p=b+3;pt=bt+3
        self.assertEqual(float(penalty('TRP',p,pt,b,bt,s,a)),0)
        self.assertEqual(float(penalty('ANCHOR',p,pt,b,bt,s,a)),3)
        self.assertEqual(float(penalty('TRP',p,pt+1,b,bt,s,a)),1)
    def test_ideal_is_different_from_teacher_matching(self):
        b=torch.zeros(2,9,64);bt=b+2;s=torch.ones(2);a=s*4
        self.assertEqual(float(penalty('TRP',b,bt,b,bt,s,a)),0)
        self.assertEqual(float(penalty('IDEAL',b,bt,b,bt,s,a)),2)
    def test_teacher_detached_both_adapted_paths_differentiable(self):
        b=torch.zeros(2,9,64,requires_grad=True);bt=torch.ones(2,9,64,requires_grad=True)
        p=torch.full((2,9,64),2.,requires_grad=True);pt=torch.full((2,9,64),4.,requires_grad=True)
        r=penalty('TRP',p,pt,b,bt,torch.ones(2),torch.ones(2));r.backward()
        self.assertIsNone(b.grad);self.assertIsNone(bt.grad)
        self.assertTrue(torch.equal(p.grad,-pt.grad));self.assertGreater(float(p.grad.abs().sum()),0)
    def test_scale_and_zero_output_invariants(self):
        b=torch.arange(18.).reshape(2,9,1).expand(2,9,64);bt=b+2;s=torch.ones(2);a=s*3
        for arm in ['ANCHOR','TRP','SHUFFLE']:
            self.assertEqual(float(penalty(arm,b,bt,b,bt,s,a)),0)
        r=penalty('TRP',b+1,bt+2,b,bt,s,a)
        self.assertEqual(float(r),float(penalty('TRP',(b+1)*5,(bt+2)*5,b*5,bt*5,s*5,a*5)))
if __name__=='__main__':unittest.main()
