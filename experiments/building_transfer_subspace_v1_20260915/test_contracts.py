import unittest
import numpy as np
from run import steps,windows,metrics,affine,transfer_gate,POLICIES
class Contracts(unittest.TestCase):
    def test_budget_ceiling(self):
        tune=4*2*(120+208);warm_tune=4*(120+208);dev=8*(120+208)
        self.assertEqual(tune+warm_tune+2640+dev+dev,11824)
        self.assertEqual(11824+2640+warm_tune+dev,18400)
        self.assertEqual([steps(p,2) for p in POLICIES],[2,8,32,120])
        self.assertEqual([steps(p,13) for p in POLICIES],[13,52,208,120])
    def test_no_target_in_windows(self):
        for n in (72,336):
            h=np.arange(n);x,y,dates,_=windows(h,'2016-03-12')
            self.assertEqual(len(x),n//24-1);self.assertEqual(y[-1,-1],n-1)
            self.assertTrue(np.all(x[:,-1]+1==y[:,0]))
            self.assertTrue(all(str(d)<'2016-03-12' for d in dates))
    def test_metric_scale_history_only(self):
        h=np.arange(72,dtype=float);y=np.ones(24);q=np.zeros((21,24))
        a=metrics(q,y,h);b=metrics(q,10*y,h)
        self.assertAlmostEqual(a['history_std'],b['history_std']);self.assertAlmostEqual(b['scaled_RMSE'],10*a['scaled_RMSE'])
    def test_affine_and_ill_condition(self):
        q=np.tile(np.arange(48).reshape(2,1,24),(1,21,1));y=2*q[:,10]+3
        a=affine(q,y);self.assertAlmostEqual(a['a'],2);self.assertAlmostEqual(a['b'],3)
        with self.assertRaises(ValueError):affine(np.ones_like(q),y)
    def rows(self,wins):
        rows=[]
        for i in range(8):
            for h in (3,14):
                for arm,value in [('S',1.),('T',.8 if i<wins else 1.1)]:rows.append(dict(role='dev',episode=f'{i}_H{h}',building=str(i),days=h,arm=arm,scaled_RMSE=value))
        return rows
    def test_one_worse_episode_not_automatic_fail(self):self.assertTrue(transfer_gate(self.rows(7),'S','T')['pass'])
    def test_macro_alone_not_sufficient(self):
        r=transfer_gate(self.rows(5),'S','T');self.assertGreater(r['macro_gain_percent'],1);self.assertFalse(r['pass'])
if __name__=='__main__':unittest.main()
