import unittest
from core import *
from data import episodes,history,target
from analysis import scalar_metrics
class Correctness(unittest.TestCase):
    def test_history_boundary(self):
        full=np.arange(1000,dtype=float);o=720;h=full[o-72:o].copy();a=windows(h,'2020-01-08')[0:2];full[:o-72]=-999;full[o:]=999;b=windows(full[o-72:o],'2020-01-08')[0:2]
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    def test_future_not_in_samples(self):
        h=np.arange(336.);x,y,ds,c=windows(h,'2020-01-08');self.assertLess(y.max(),336);self.assertEqual(y[-1,-1],335)
    def test_disjoint_daily_targets(self):
        for n in [72,336]:
            x,y,ds,c=windows(np.arange(n),'2020-01-08');self.assertEqual(len(y),n//24-1);self.assertEqual(len(np.unique(y)),y.size);self.assertTrue((x>=0).all());self.assertTrue((y<n).all())
    def test_coverage(self):
        for origin,expected in [('2020-01-08',2),('2020-01-11',0)]:
            self.assertEqual(windows(np.arange(72),origin)[3],expected);self.assertGreater(windows(np.arange(336),origin)[3],0)
    def test_building_split(self):
        s=read(OUT/'building_split.json');ids=s['recipe']+s['dev']+s['heldout'];self.assertEqual(len(set(ids)),14);self.assertEqual([len(s[k]) for k in ['recipe','dev','heldout']],[4,4,6])
    def test_physical_groups(self):
        e=read(OUT/'episodes.json');groups={}
        for x in e:groups.setdefault(x['physical_group'],set()).add(x['role'])
        self.assertEqual(len(groups),14);self.assertTrue(all(len(g)==1 for g in groups.values()))
    def test_coverage_load_independent(self):
        for n in [72,336]:self.assertEqual(windows(np.arange(n),'2020-01-11')[3],windows(np.ones(n)*999,'2020-01-11')[3])
    def test_quantile_interpolation(self):
        rng=np.random.default_rng(90);f=np.sort(rng.normal(size=(21,24)),axis=0);l=np.sort(rng.normal(size=(21,24)),axis=0)
        for a in [0,.25,.5,.75,1]:self.assertTrue((np.diff(interpolate(f,l,a),axis=0)>=0).all())
    def test_independent_metrics(self):
        rng=np.random.default_rng(91);q=np.sort(rng.normal(size=(21,24)),axis=0);y=rng.normal(size=24);h=rng.normal(size=72)
        a=metrics(q,y,h);b=scalar_metrics(q,y,h)
        for k in b:self.assertAlmostEqual(a[k],b[k],places=12)
    def test_exact_endpoints(self):
        rng=np.random.default_rng(92);f=rng.normal(size=(21,24));l=rng.normal(size=(21,24));self.assertTrue(np.array_equal(interpolate(f,l,0),f));self.assertTrue(np.array_equal(interpolate(f,l,1),l))
    def test_affine(self):
        q=np.tile(np.arange(24.)[None,None,:],(2,21,1));y=2*q[:,10,:]+3;r=affine(q,y);self.assertAlmostEqual(r['a'],2);self.assertAlmostEqual(r['b'],3)
        with self.assertRaises(ValueError):affine(np.ones((2,21,24)),np.ones((2,24)))
    def test_actual_episodes(self):
        for role in ['recipe','dev','heldout']:
            for e in episodes(role):
                h=history(e);x,y,d,c=windows(h,e['origin']);self.assertEqual(c,e['coverage_count']);self.assertEqual(len(x),e['adaptation_windows']);self.assertEqual(len(h),e['history_days']*24)
    def test_heldout_and_recipe_future_closed(self):
        with self.assertRaises(AssertionError):target(episodes('recipe')[0])
        if not (OUT/'heldout_seal.json').exists():
            with self.assertRaises(AssertionError):target(episodes('heldout')[0])
if __name__=='__main__':unittest.main()
