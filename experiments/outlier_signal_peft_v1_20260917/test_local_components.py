"""Independent local tests, explicitly NOT the user's missing test_reference.py."""
import unittest
import numpy as np
import torch
from .audit import CACHE, select_origins
from .model import BoundedAdapter, loss_2pinball, preprocess, robust_scale


class LocalChecks(unittest.TestCase):
    def test_full_days_and_balanced_phases(self):
        for p,n in [(24,256),(96,64),(96,128)]:
            legal = np.arange(513,p*600+11)
            origins, full = select_origins(legal,p,n,81700)
            self.assertEqual(len(set(origins//p)), n)
            self.assertTrue(np.isin(origins,legal).all())
            self.assertTrue(all(np.isin(np.arange(d*p,(d+1)*p),legal).all() for d in full))
            counts = np.bincount(origins%p,minlength=p)
            self.assertLessEqual(counts.max()-counts.min(),1)
            np.testing.assert_array_equal(origins, select_origins(legal,p,n,81700)[0])

    def test_partial_days_do_not_fill_deficit(self):
        with self.assertRaisesRegex(ValueError,'INSUFFICIENT_FULL_DAYS'):
            select_origins(np.arange(1,24*8),24,8,10)

    def test_eight_real_input_preprocessors_numpy(self):
        d = np.load(CACHE/'data/electricity/TRAIN_inputs.npz')
        x = d['x'][:2].reshape(8,512).astype(np.float64)
        sigma = np.tile(d['sigma'],2)
        # Add past-only stress so the comparison is not restricted to no-op cases.
        x[:,[1,32,510]] += np.array([1,-1,1])*20*sigma[:,None]
        med = np.median(x,axis=-1,keepdims=True)
        scale = np.maximum(1.4826*np.median(np.abs(x-med),axis=-1,keepdims=True), .1*sigma[:,None])
        clip = np.clip(x,med-6*scale,med+6*scale)
        hampel = x.copy()
        for i in range(512):
            w = x[:,max(0,i-12):min(512,i+13)]
            m = np.median(w,axis=-1)
            r = np.maximum(1.4826*np.median(np.abs(w-m[:,None]),axis=-1),.1*sigma)
            hampel[:,i] = np.where(np.abs(x[:,i]-m)>4*r,m,x[:,i])
        xt,st = torch.tensor(x),torch.tensor(sigma)
        np.testing.assert_allclose(preprocess(xt,st,'A2').numpy(),clip,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(preprocess(xt,st,'A3').numpy(),hampel,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(robust_scale(xt,st)[1],scale,rtol=1e-10,atol=1e-12)
        z = (clip-clip.mean(-1,keepdims=True))/clip.std(-1,keepdims=True)
        discarded = ((x-clip)/scale).reshape(8,32,16)
        generic = np.concatenate([np.clip(np.arcsinh(z.reshape(8,32,16)),-6,6), np.tanh(z.reshape(8,32,16))],axis=-1)
        residual = np.concatenate([generic[:,:,:16],np.clip(np.arcsinh(discarded),-6,6)],axis=-1)
        zt = (torch.tensor(clip)-torch.tensor(clip).mean(-1,keepdim=True))/torch.tensor(clip).std(-1,keepdim=True,correction=0)
        first = torch.asinh(zt.reshape(8,32,16)).clamp(-6,6)
        for actual,expected in [(torch.cat([first,torch.tanh(zt.reshape(8,32,16))],-1),generic),
                                (torch.cat([first,torch.asinh(((xt-torch.tensor(clip))/robust_scale(xt,st)[1]).reshape(8,32,16)).clamp(-6,6)],-1),residual)]:
            np.testing.assert_allclose(actual.numpy(),expected,rtol=1e-10,atol=1e-12)

    def test_scalar_pinball_mean_and_units(self):
        r = np.random.default_rng(13)
        p,y,s = r.normal(size=(3,9,64)),r.normal(size=(3,64)),np.array([.2,1,7])
        q = np.arange(1,10)/10
        terms = [2*max(q[j]*(y[i,h]-p[i,j,h]),(q[j]-1)*(y[i,h]-p[i,j,h]))/s[i]
                 for i in range(3) for j in range(9) for h in range(64)]
        actual = loss_2pinball(torch.tensor(p),torch.tensor(y),torch.tensor(s),torch.tensor(q)).item()
        np.testing.assert_allclose(actual,np.mean(terms),rtol=1e-10,atol=1e-12)
        scaled = loss_2pinball(torch.tensor(p*17),torch.tensor(y*17),torch.tensor(s*17),torch.tensor(q)).item()
        np.testing.assert_allclose(actual,scaled,rtol=1e-10,atol=1e-12)

    def test_adapter_bound_and_zero_initialization(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(81500)
            e,v = torch.randn(8,32,512),torch.randn(8,32,32)*100
            a = BoundedAdapter(81500)
            torch.testing.assert_close(a(e,v),e,rtol=0,atol=0)
            with torch.no_grad():
                a.up.weight.normal_()
                a.up.bias.normal_()
            delta = a(e,v)-e
            cap = e.square().mean(-1).sqrt().quantile(.5,dim=-1)[:,None]
            self.assertTrue((delta.square().mean(-1).sqrt() <= cap+1e-6).all())
            self.assertTrue((delta.abs().amax(-1) <= cap+1e-6).all())


if __name__ == '__main__': unittest.main()
