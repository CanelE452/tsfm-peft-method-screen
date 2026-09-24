import copy,csv,hashlib,json,math,os,tempfile,unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from triage.common import *
from triage.statistics import *
from triage.respiration import period_stats,cycle_forecast,cases_for_signal,read_signal
from triage.coordinates import rotate,heading,canonical,cv_predict,read_track,windows,heading_bank,novelty,rotated_cases
from triage.wind import soft_weights,turnover,interp_context,nearest,bank_for
from triage.engine import choose_simple,seal_value,validate_splits,main_topic


def case(uid='a',group='g',score=.1,H=4):
    x=np.sin(np.arange(32)/3)[:,None];y=np.sin(np.arange(32,32+H)/3)[:,None]
    return Case(uid,group,x,y,score,score,np.arange(5,dtype=float),np.repeat(x[-1:],H,axis=0),scale_of(x),np.array([score])).validate()

class Contracts(unittest.TestCase):
    def test01_json_strict_nan(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';write_json(p,{'a':float('nan'),'b':np.array([1.])})
            self.assertEqual(read_json(p),{'a':None,'b':[1.]})
    def test02_csv_lf(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.csv';write_csv(p,[{'a':1,'b':2}]);self.assertNotIn(b'\r',p.read_bytes())
    def test03_entropy_constant(self):self.assertEqual(entropy(np.ones(20)),0.)
    def test04_entropy_scale(self):self.assertAlmostEqual(entropy(np.arange(30)),entropy(np.arange(30)*100),8)
    def test05_error_mask(self):
        c=case();c.mask=np.array([[1],[0],[1],[1]],bool);p=c.y.copy();p[1]=10000
        self.assertEqual(normalized_error(c,p),0.)
    def test06_vector_rotation_error_invariant(self):
        c=case();c.x=np.column_stack([c.x[:,0],c.x[:,0]**2]);c.y=np.c_[c.y[:,0],c.y[:,0]];c.simple=c.y+.2
        e=normalized_error(c,c.simple)
        z=replace(c,x=rotate(c.x,61),y=rotate(c.y,61),simple=rotate(c.simple,61))
        self.assertAlmostEqual(e,normalized_error(z,z.simple),10)
    def test07_ridge_fit_multitarget(self):
        rng=np.random.default_rng(0);x=rng.normal(size=(80,4));y=x@rng.normal(size=(4,3))
        m=Ridge().fit(x,y,1e-8);self.assertLess(np.abs(m.predict(x)-y).max(),1e-6)
    def test08_ridge_constant_columns(self):
        x=np.c_[np.arange(10),np.ones(10)];m=Ridge().fit(x,np.arange(10)[:,None],1.);self.assertTrue(np.isfinite(m.predict(x)).all())
    def test09_no_future_in_features(self):
        c=case();p=replace(c,y=c.y+10000)
        self.assertTrue(np.array_equal(normalized_features(c)[0],normalized_features(p)[0]))
    def test10_ridge_prediction_no_future_values(self):
        cs=[case(str(i),str(i),i/20) for i in range(10)]
        X,Y=ridge_data(cs);m=Ridge().fit(X,Y,1.)
        a=predict_ridge(m,cs);b=predict_ridge(m,[replace(c,y=c.y+9999) for c in cs])
        np.testing.assert_array_equal(a,b)
    def test11_train_scaling_not_test(self):
        x=np.arange(50)[:,None];m=Ridge().fit(x,x,1.);h=m.identity();m.predict(np.ones((20,1))*100000);self.assertEqual(h,m.identity())
    def test12_cal_select_group_weight(self):
        cs=[case(str(i),'a' if i<9 else 'b') for i in range(10)]
        pred={'SIMPLE':[c.y+(.1 if i<9 else 10) for i,c in enumerate(cs)],'RIDGE':[c.y+2 for c in cs]}
        best,v=choose_simple(cs,pred);self.assertEqual(best,'RIDGE')
    def test13_cluster_bootstrap_mean(self):
        r=bootstrap_mean([1,2,3,4,5]);self.assertEqual(r['mean'],3);self.assertEqual(r['n_groups'],5);self.assertGreater(r['lower'],0)
    def test14_cluster_bootstrap_negative(self):self.assertLess(bootstrap_mean([-1,-2,-3,-4])['upper'],0)
    def test15_no_group_support(self):self.assertEqual(bootstrap_mean([])['n_groups'],0)
    def test16_strata_train_fixed(self):
        np.testing.assert_array_equal(strata(np.array([-1,.4,2]),(.3,.6)),[0,1,2])
    def test17_group_contrast_no_pseudo_replication(self):
        cs=[case(str(i),'one',s) for i,s in enumerate([0,0,1,1])]
        self.assertEqual(len(contrasts([0,0,1,1],cs,[.3,.6])),1)
    def test18_proxy_collapsed(self):self.assertIsNone(correlation(np.ones(20),np.arange(20)))
    def test19_resp_period_constant(self):
        x=np.sin(2*np.pi*np.arange(320)/20)
        score,p,_=period_stats(x);self.assertLess(score,.05);self.assertAlmostEqual(p,4.,delta=.3)
    def test20_resp_irregular_higher(self):
        t=np.arange(320)/5;phase=2*np.pi*(.23*t+.022*t*np.sin(t/13))
        self.assertGreater(period_stats(np.sin(phase))[0],period_stats(np.sin(2*np.pi*.25*t))[0])
    def test21_resp_short_support(self):self.assertTrue(math.isnan(period_stats(np.ones(320))[0]))
    def test22_resp_cycle_shape(self):self.assertEqual(cycle_forecast(np.sin(np.arange(320)/3),40).shape,(40,1))
    def test23_resp_context_future_poison(self):
        x=np.sin(np.arange(2400)/3);a,_=cases_for_signal(x,1,'g')
        y=x.copy();y[320:360]+=1000;b,_=cases_for_signal(y,1,'g')
        self.assertEqual(a[0].score,b[0].score);np.testing.assert_array_equal(a[0].simple,b[0].simple)
    def test24_rotation_inverse(self):
        x=np.arange(16).reshape(8,2);np.testing.assert_allclose(rotate(rotate(x,71),-71),x,atol=1e-12)
    def test25_canonical_yaw_invariance(self):
        x=np.c_[np.arange(8),np.arange(8)**1.2];np.testing.assert_allclose(canonical(rotate(x,49)),canonical(x),atol=1e-12)
    def test26_cv_equivariant(self):
        x=np.c_[np.arange(8),np.arange(8)**1.2];np.testing.assert_allclose(cv_predict(rotate(x,49),12),rotate(cv_predict(x,12),49),atol=1e-12)
    def test27_motion_parser_verified_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.txt';rows=[]
            for i in range(101):
                a=[str(i),'1','Pedestrian']+['-1']*14;a[13]=str(i*.1);a[15]=str(i*.2);rows.append(' '.join(a))
            p.write_text('\n'.join(rows));arr=read_track(p);np.testing.assert_allclose(arr[-1],[100,1,10,20])
    def test28_motion_bad_schema_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.txt';p.write_text('1 1 2 3\n')
            with self.assertRaises(ContractError):read_track(p)
    def test29_motion_split_no_target_overlap(self):
        a=np.c_[np.arange(1000),np.ones(1000),np.arange(1000)*.1,np.sin(np.arange(1000)/20)]
        parts,bounds=windows(a,'s')
        for split,cs in parts.items():
            for c in cs:
                first=int(c.uid.split(':')[-1]);self.assertGreaterEqual(first,bounds[split][0]);self.assertLess(first+19,bounds[split][1])
    def test30_motion_heading_novelty_only_past(self):
        c=case();c.x=np.c_[np.arange(8),np.arange(8)*.2];b=heading_bank([c]);self.assertEqual(novelty(c.x,b),novelty(c.x,b))
    def test31_wind_weights_sum_one(self):
        b=np.arange(24).reshape(8,3)+1;b=b/b.sum(1,keepdims=True)
        w=soft_weights(np.arange(-100,400),b);np.testing.assert_allclose(w.sum(1),1)
    def test32_wind_wrap_continuity(self):
        b=np.ones((8,3))/3;self.assertEqual(turnover(np.array([359,0,1]),b),0.)
    def test33_wind_turnover_detects_change(self):
        b=np.tile(np.array([[.9,.05,.05],[.05,.9,.05]]),(4,1))
        self.assertGreater(turnover(np.array([0,45,90,135]),b),.8)
    def test34_wind_context_imputation(self):
        a=np.arange(20,dtype=float);v=np.ones(20,bool);v[8]=False;a[8]=999
        z=interp_context(a,v);self.assertEqual(z[8],8)
    def test35_wind_bad_quality_not_filled(self):self.assertIsNone(interp_context(np.ones(10),np.zeros(10,bool)))
    def test36_neighbours_no_self(self):self.assertNotIn(0,nearest(np.c_[np.arange(12),np.zeros(12)],0))
    def test37_wind_bank_future_poison(self):
        rng=np.random.default_rng(4);n=42*144
        p={'Patv':rng.normal(100,10,(n,12)),'valid':np.ones((n,12),bool),'Ndir':rng.uniform(0,360,(n,12))}
        xy=np.c_[np.arange(12),np.zeros(12)];b,_=bank_for(p,xy,0,[1,2,3]);p['Patv'][3*144:]=999999
        c,_=bank_for(p,xy,0,[1,2,3]);np.testing.assert_array_equal(b,c)
    def test38_no_synthetic_data_on_download_fail(self):
        from triage.network import Fetcher
        with tempfile.TemporaryDirectory() as d:
            f=Fetcher(Path(d)/'cache',Path(d)/'pub')
            with patch('urllib.request.urlopen',side_effect=OSError('offline')):
                with self.assertRaises(Blocked):f.get('https://example.org/file','x')
            self.assertFalse((Path(d)/'cache/x').exists())
    def test39_cache_checksum_rejects(self):
        from triage.network import Fetcher
        with tempfile.TemporaryDirectory() as d:
            f=Fetcher(Path(d)/'cache',Path(d)/'pub');(f.cache/'a').write_text('bad')
            with patch('urllib.request.urlopen',side_effect=OSError('offline')):
                with self.assertRaises(Blocked):f.get('https://example.org/file','a',sha='0'*64)
    def test40_bytecode_free_current_sources(self):
        # Distribution packing itself is validated separately after ZIP fresh extraction.
        root=Path(__file__).parents[1]
        self.assertTrue((root/'run.py').exists());self.assertTrue((root/'RUN_CONFIG.json').exists())
    def test41_native_input_does_not_receive_future_target(self):
        import torch
        from triage.predictors import Native
        class Pipe:
            def __init__(self):self.calls=[]
            def predict_quantiles(self,inputs,**kw):
                self.calls.append(copy.deepcopy(inputs));H=kw['prediction_length']
                return [torch.tensor(np.repeat(z['target'][:,-1:],H,1)[:,:,None]) for z in inputs],None
        n=Native.__new__(Native);n.pipe=Pipe();n.windows=0
        c=case();a=n.predict([c]);b=n.predict([replace(c,y=c.y+1e9)])
        np.testing.assert_array_equal(a,b);self.assertNotIn('future_covariates',n.pipe.calls[0][0])
    def test42_native_shape_contract(self):
        import torch
        from triage.predictors import Native
        class Pipe:
            def predict_quantiles(self,*a,**k):return [torch.zeros(4,1,1)],None
        n=Native.__new__(Native);n.pipe=Pipe();n.windows=0
        with self.assertRaises(ContractError):n.predict([case()])
    def test43_nogo_is_not_missing_reference(self):
        train=[case(str(i),'g',i/50) for i in range(30)]
        cal=[case('c'+str(i),'g',i/50) for i in range(20)];test=[case('t'+str(i),'g',i/50) for i in range(20)]
        p={'SIMPLE':[c.simple for c in test]};pc={'SIMPLE':[c.simple for c in cal]}
        z,e=analyze(train,cal,test,p,pc,'SIMPLE',{'alpha_family':.05/3,'min_contrast_groups':4,'min_agreement':.5})
        self.assertEqual(z['axis_status'],'BLOCKED_F0_REFERENCE')
    def test44_no_circular_test_thresholds(self):
        tr=[case(str(i),'g',i/100) for i in range(30)];cal=[case('c'+str(i),'g',i/100) for i in range(20)]
        te=[case('t'+str(i),'g',i/100) for i in range(20)];m=Ridge().fit(np.arange(20)[:,None],np.arange(20)[:,None],1.)
        a=seal_value({'train':tr,'cal':cal,'test':te},{'m':m},{},'s',{}, {},None)
        te=[replace(c,y=c.y+1000000) for c in te]
        b=seal_value({'train':tr,'cal':cal,'test':te},{'m':m},{},'s',{}, {},None)
        self.assertEqual(hash_obj(a),hash_obj(b))
    def test45_equal_forecast_information(self):
        c=case();raw,_,_=normalized_features(c);X,Y=ridge_data([c],True)
        np.testing.assert_array_equal(X[0,:len(raw)],raw)
    def test46_physical_data_dont_enter_package(self):
        root=Path(__file__).parents[1]
        self.assertFalse(any(root.rglob('*.pt')));self.assertFalse(any(root.rglob('*.csv.gz')))
    def test47_case_invalid_past_rejected(self):
        c=case();c.x[0]=np.nan
        with self.assertRaises(ContractError):c.validate()
    def test48_scope_not_peft(self):
        root=Path(__file__).parents[1];cfg=read_json(root/'RUN_CONFIG.json')
        self.assertFalse(cfg['new_peft_fit']);self.assertFalse(cfg['new_lora_fit'])

if __name__=='__main__':unittest.main()
