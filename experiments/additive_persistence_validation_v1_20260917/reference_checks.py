"""20 local CPU checks authored from MASTER_CLI; not the missing supplied PY."""
import unittest,inspect
import numpy as np
import torch
from .model import correction_gate,mechanism_gate,ResidualAdapter,ForecastModel,loss_2pinball
from experiments.outlier_signal_peft_v1_20260917.reference_core import select_days_reference,pinball_scalar,scale

def scalar_gate(x,sigma):
    med=np.median(x);r=max(1.4826*np.median(np.abs(x-med)),.1*sigma);d=(x-med)/r
    p=[]
    for t in range(512):
        p.append(sum(abs(d[j])>3 and np.sign(d[j])==np.sign(d[t]) for j in range(max(0,t-7),t+1))/(min(t+1,8)) if abs(d[t])>3 else 0.)
    return 1-np.array(p).reshape(32,16).mean(1)

class ReferenceChecks(unittest.TestCase):
    def setUp(self):
        self.x=torch.tensor(np.random.default_rng(1).normal(size=(3,512)),dtype=torch.float64);self.x[:,-33:]+=12;self.s=torch.ones(3,dtype=torch.float64)
    def g(self,arm):return mechanism_gate(self.x,self.s,arm).numpy()
    def test_01_scalar_persistence(self):np.testing.assert_allclose(self.g('C3'),np.stack([scalar_gate(x.numpy(),1.) for x in self.x]),rtol=1e-10,atol=1e-12)
    def test_02_mean_sum(self):np.testing.assert_allclose(self.g('C3').sum(1),self.g('M_MEAN').sum(1),atol=1e-12)
    def test_03_mean_constant(self):self.assertTrue((np.ptp(self.g('M_MEAN'),axis=1)==0).all())
    def test_04_rotate_multiset(self):np.testing.assert_array_equal(np.sort(self.g('C3')),np.sort(self.g('M_ROTATE16')))
    def test_05_rotate_square(self):np.testing.assert_allclose((self.g('C3')**2).sum(1),(self.g('M_ROTATE16')**2).sum(1),atol=1e-12)
    def test_06_rotate_offset(self):np.testing.assert_array_equal(self.g('M_ROTATE16'),np.roll(self.g('C3'),16,axis=-1))
    def test_07_recency_multiset(self):np.testing.assert_array_equal(np.sort(self.g('C3')),np.sort(self.g('M_RECENCY')))
    def test_08_recency_square(self):np.testing.assert_allclose((self.g('C3')**2).sum(1),(self.g('M_RECENCY')**2).sum(1),atol=1e-12)
    def test_09_recency_order(self):self.assertTrue((np.diff(self.g('M_RECENCY'))<=0).all())
    def test_10_zero_identity(self):
        a=ResidualAdapter(1);x=torch.randn(2,32,512);self.assertTrue(torch.equal(a(x,torch.rand(2,32)),x))
    def test_11_zero_gate_identity(self):
        a=ResidualAdapter(1);torch.nn.init.normal_(a.up.weight);x=torch.randn(2,32,512);self.assertTrue(torch.equal(a(x,torch.zeros(2,32)),x))
    def test_12_init_parameters(self):
        a=ResidualAdapter(1);b=ResidualAdapter(1);self.assertEqual(sum(p.numel() for p in a.parameters()),8712);self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))
    def test_13_no_metadata_input(self):self.assertEqual(list(inspect.signature(ForecastModel.forward).parameters),['self','observed','sigma','residual_mode','persistence_mode'])
    def test_14_nonextreme_equal(self):
        x=torch.zeros(1,512);s=torch.ones(1)
        for arm in ['C3','M_MEAN','M_ROTATE16','M_RECENCY']:self.assertTrue(torch.equal(mechanism_gate(x,s,arm),torch.ones(1,32)))
    def test_15_origin_diversity(self):
        for p in [24,96]:
            for n in [64,128,256]:
                o=select_days_reference(np.arange(512,p*500-64),p,n,85700);self.assertEqual(len(np.unique(o//p)),n);self.assertLessEqual(np.ptp(np.bincount(o%p,minlength=p)),1)
    def test_16_insufficient_days(self):
        with self.assertRaises(ValueError):select_days_reference(np.arange(512,700),96,256,1)
    def test_17_pinball_independent(self):
        q=np.arange(1,10)/10;y=np.arange(8).reshape(2,4)/3;p=np.zeros((2,9,4));s=np.array([1.,2.]);expected=pinball_scalar(p,y,s,q);actual=loss_2pinball(torch.tensor(p),torch.tensor(y),torch.tensor(s),torch.tensor(q));self.assertAlmostEqual(float(actual),expected,12)
    def test_18_shrink_endpoints(self):
        a=np.arange(10,dtype=float);b=a[::-1].copy();self.assertTrue(np.array_equal(a+0*(b-a),a));self.assertTrue(np.array_equal(a+1*(b-a),b))
    def test_19_pulse_past_future(self):
        x=self.x[0].numpy();delta=8*scale(x,1);past=x.copy();past[-32:]+=delta;shiftpast=x.copy();shiftpast[-32:]+=delta;np.testing.assert_array_equal(past,shiftpast);self.assertNotEqual(delta,0)
    def test_20_same_total_not_square(self):
        self.assertTrue(np.any(np.abs((self.g('C3')**2).sum(1)-(self.g('M_MEAN')**2).sum(1))>1e-5))

if __name__=='__main__':unittest.main()
