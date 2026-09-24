import unittest, numpy as np
from difficulty_screen import Ridge,spearman,slope
class Tests(unittest.TestCase):
 def test_ridge(self):
  x=np.arange(30)[:,None].astype(float);y=np.c_[2*x[:,0]+1,-x[:,0]+3];m=Ridge(1e-6).fit(x,y);self.assertLess(np.abs(m.predict(x)-y).mean(),1e-4)
 def test_history_signal(self):
  rng=np.random.default_rng(0);n=200;x=rng.normal(size=(n,2));h=rng.normal(size=(n,1));y=(x[:,0]+2*h[:,0])[:,None];a=Ridge(1).fit(x[:150],y[:150]);b=Ridge(1).fit(np.c_[x[:150],h[:150]],y[:150]);ea=np.abs(a.predict(x[150:])-y[150:]).mean();eb=np.abs(b.predict(np.c_[x[150:],h[150:]])-y[150:]).mean();self.assertLess(eb,ea*.3)
 def test_spearman(self):self.assertGreater(spearman([1,2,3],[2,4,8]),.99)
 def test_slope(self):self.assertAlmostEqual(slope(np.array([0.,1.,2.]),1.),1.0,places=10)
