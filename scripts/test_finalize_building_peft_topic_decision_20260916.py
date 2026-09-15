"""Tests of metric independence, paired building inference, and honest zero handling."""
import unittest, math, tempfile, json
from pathlib import Path
import numpy as np
import finalize_building_peft_topic_decision_20260916 as f

class AuditTest(unittest.TestCase):
    def test_scalar_metric_known_values(self):
        h=np.tile([-1.,1.],36);y=np.ones(24);q=np.zeros((3,24))
        r=f.scalar_metric(q,y,h,[.1,.5,.9])
        for k in f.METRICS:self.assertEqual(r[k],1.)
    def test_scalar_uses_actual_median_position(self):
        q=np.array([[-2.]*24,[0.]*24,[3.]*24,[4.]*24]);y=np.zeros(24)
        self.assertEqual(f.scalar_metric(q,y,np.tile([-1.,1.],36),[.1,.5,.7,.9])['primary'],0.)
    def test_zero_denominator_never_epsilon_gain(self):
        self.assertIsNone(f.gain(0,1));v=f.paired_effect([0,0,0],[1,2,3])
        self.assertIsNone(v['gain_percent']);self.assertIsNone(v['ci95_percent']);self.assertEqual(v['zero_denominator_resamples'],2000);self.assertLess(v['absolute_difference'],0)
    def test_building_pair_constant_gain(self):
        b=np.array([1.,2.,4.,8.,16.,32.]);v=f.paired_effect(b,b*.9)
        self.assertAlmostEqual(v['gain_percent'],10.)
        for n in ['ci95_percent','leave_one_out_range_percent']:
            for x in v[n]:self.assertAlmostEqual(x,10.)
    def test_building_replication_does_not_create_independent_hours(self):
        b=[1,2,3,4,5,6];c=[6,5,4,3,2,1];v=f.paired_effect(b,c)
        self.assertEqual(v['gain_percent'],0);self.assertLess(v['ci95_percent'][0],0);self.assertGreater(v['ci95_percent'][1],0)
        self.assertEqual(v,f.paired_effect(b,c))
    def test_policies_and_h14_fixed120(self):
        self.assertEqual([f.at_step(p,2) for p in f.POLICIES],[0,2,8,32,120])
        self.assertEqual([f.at_step(p,13) for p in f.POLICIES],[0,13,52,208,120])
    def test_macro_gain_is_not_mean_percentage(self):
        self.assertAlmostEqual(f.gain(f.avg([1,9]),f.avg([.5,9.5])),0.)
        self.assertNotEqual(f.avg([f.gain(1,.5),f.gain(9,9.5)]),0.)
class DecisionTest(unittest.TestCase):
    def run_case(self, values):
        with tempfile.TemporaryDirectory() as td:
            previous=f.OUT;f.OUT=Path(td)
            try:
                eps=[];rows=[];selected={m:dict(lr=3e-5,policy='EPOCH1',primary=1.) for m in f.METHODS}
                for b in range(6):
                    for days in [3,14]:
                        eid=f'b{b}_H{days}';eps.append(dict(id=eid,building=f'b{b}',days=days,role='LOCKED'))
                        for method in f.METHODS+['F0','AFFINE','SEASONAL24']:
                            for seed in ([61680,61681] if method in f.METHODS else [0]):
                                for step in ([days-1,120] if method in f.METHODS else [0]):
                                    arm=method+('_FIXED120' if step==120 else '');value=values[arm]*(b+1)
                                    rows.append(dict(episode=eid,building=f'b{b}',role='LOCKED',method=method,lr=3e-5 if method in f.METHODS else 0,step=step,seed=seed,fit=f'{eid}_{method}_{seed}',train_seconds=float(step),**{k:value for k in f.METRICS}))
                f.save(f.OUT/'episodes.json',eps);f.save(f.OUT/'scores.json',rows);f.save(f.OUT/'selection_seal.json',dict(selected=selected))
                return f.summarize()
            finally:f.OUT=previous
    def defaults(self):return dict(STD=1.,SIMPLE=1.02,CANDIDATE=.8,STD_FIXED120=1.,SIMPLE_FIXED120=1.,CANDIDATE_FIXED120=.7,F0=.98,AFFINE=.97,SEASONAL24=1.2)
    def test_all_preregistered_conditions_pass_but_novelty_not_promoted(self):
        d=self.run_case(self.defaults());self.assertEqual(d['predictive'],'SCREEN_PASS');self.assertEqual(d['novelty'],'UNRESOLVED');self.assertNotEqual(d['topic_decision'],'TOPIC_TO_ADVANCE')
    def test_strong_fixed_control_blocks_pass(self):
        v=self.defaults();v['AFFINE']=.6;d=self.run_case(v);self.assertNotEqual(d['predictive'],'SCREEN_PASS');self.assertFalse(d['gates']['no_fixed_control_better'])
    def test_equal_f0_not_added_value(self):
        v={k:1. for k in self.defaults()};d=self.run_case(v);self.assertEqual(d['predictive'],'NO_ADDED_VALUE_IN_THIS_PILOT')

if __name__=='__main__':unittest.main()
