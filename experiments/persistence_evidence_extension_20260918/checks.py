"""Focused synthetic checks before any new inference, optimizer updates always zero."""
import unittest
import numpy as np,pandas as pd
from .score import metric_arrays
from .prepare import plan
from .common import pinball_scalar,OUT,save
class Checks(unittest.TestCase):
    def test_scalar_metrics(self):
        r=np.random.default_rng(88300);p=r.normal(size=(5,9,64));y=r.normal(size=(5,64));s=np.arange(1,6);m=metric_arrays(p,y,s)
        np.testing.assert_allclose(m[:,3].mean(),pinball_scalar(p,y,s,np.arange(1,10)/10),rtol=1e-12)
        self.assertAlmostEqual(m[:,0].mean(),sum(abs(p[i,4,j]-y[i,j])/s[i] for i in range(5) for j in range(64))/(5*64))
    def test_identical_predictions(self):
        y=np.arange(128).reshape(2,64);p=np.tile(y[:,None,:],(1,9,1));self.assertEqual(float(metric_arrays(p,y,np.ones(2)).sum()),0.)
    def test_dst(self):
        for day,expected in [('2025-03-30',46),('2025-10-26',50)]:
            t=pd.Timestamp(day,tz='Europe/London');self.assertEqual(int(((t+pd.DateOffset(days=1)).tz_convert('UTC')-t.tz_convert('UTC')).total_seconds()/1800),expected)
    def test_plan_and_intervention(self):
        p=plan();self.assertEqual(len(p),105);self.assertEqual(sum(x['arm'] in ['C3_W_RECENCY_G','RECENCY_W_C3_G'] for x in p),36);self.assertTrue(all(x['row']['seed']==x['seed'] for x in p))
    def test_factorial_identity(self):
        A,B,C,D=.4,.7,.6,.8;g=.5*((B-A)+(D-C));w=.5*((C-A)+(D-B));self.assertAlmostEqual(g+w,D-A)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));save(OUT/'CPU_CHECKS.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,optimizer_updates=0));assert result.wasSuccessful()
