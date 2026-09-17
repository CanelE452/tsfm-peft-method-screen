import unittest
from .common import *
class Checks(unittest.TestCase):
    def test_naive(self):
        x=np.arange(512,dtype=float)[None];a,b=naive(x,24);np.testing.assert_array_equal(a,np.full((1,64),511.));np.testing.assert_array_equal(b[0,:25],list(range(488,512))+[488])
        _,b=naive(x,96);np.testing.assert_array_equal(b[0],np.arange(416,480))
    def test_factorial(self):
        a,b,c,d=.2,.3,.25,.4
        self.assertAlmostEqual(((b-a)+(d-c))/2+((c-a)+(d-b))/2,d-a)
    def test_plan_pairing(self):
        plan=read(OUT/'PLAN.json');self.assertEqual(len(plan),114)
        fixed=[j for j in plan if j['arm'].startswith('FIX_')]
        self.assertEqual(len(fixed),72)
        self.assertTrue(all(j['row']['lr']==.0003 and j['row']['step']==1024 for j in fixed))
        self.assertEqual(len([j for j in plan if j['seed']==0]),30)
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));save(OUT/'CPU_CHECKS.json',dict(passed=r.wasSuccessful(),tests=r.testsRun));assert r.wasSuccessful()
