import unittest
from unittest.mock import patch
import numpy as np
import runtime as r
class RuntimeContracts(unittest.TestCase):
    def test_cap(self):
        self.assertEqual(r.checkpoints(2),[0,2,8,32,120]);self.assertEqual(r.checkpoints(13),[0,13,52,120,208]);self.assertEqual(4*2*3*(120+208)+6*3*2*(120+208),19680)
    def test_history_boundary(self):
        for n in [72,336]:
            x,y=r.windows(np.arange(n));self.assertEqual(len(x),n//24-1);self.assertEqual(y[-1,-1],n-1);self.assertTrue(np.all(x[:,-1]+1==y[:,0]))
    def test_metric_median_from_config_and_scale(self):
        with patch.object(r,'read',return_value={'quantiles':[.1,.5,.9],'median_index':1}):
            q=np.array([[0.]*24,[2.]*24,[3.]*24]);y=np.ones(24);h=np.arange(72,dtype=float);m=r.metric(q,y,h);self.assertAlmostEqual(m['raw_RMSE'],1.);self.assertAlmostEqual(m['primary'],1/h.std())
    def test_affine_fallback_is_f0(self):
        with patch.object(r,'read',return_value={'median_index':1}):
            a=r.affine(np.ones((2,3,24)),np.ones((2,24))*7);self.assertTrue(a['fallback']);self.assertEqual((a['a'],a['b']),(1.,0.))
    def test_worker_does_not_accept_target_handle(self):
        job={'id':'x','target_file':'forbidden'}
        with self.assertRaises(AssertionError):r.fit(job,np.arange(72),None)
if __name__=='__main__':unittest.main()
