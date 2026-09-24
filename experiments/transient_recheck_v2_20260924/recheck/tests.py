"""Synthetic and local-bare-Git tests only; never reads the user's study data."""
from __future__ import annotations
import copy,csv,gzip,hashlib,io,json,os,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .common import ContractError,require,read_json,write_json,file_hash,canonical_hash,Ledger
from .data import (Episode,PinnedSource,ARMS,regenerate_schedule,schedule_hash,plan_for_endpoints,
                   make_view,features,label_panels,prepare)
from .ridge import ridge_path,choose_alpha,RidgeModel
from .engine import run_engine,assert_temporal
from .evaluate import export_evaluation,MODEL_NAMES,summarize
from .gitops import git,publish_exact,empty_index

ROOT=Path(__file__).resolve().parents[1]

def config():return read_json(ROOT/'RUN_CONFIG.json')

def synthetic_episode(cfg,index=0):
    start=(cfg['source_data'][str(index)]['start_day']-1)*86400.
    schedule=regenerate_schedule(cfg['schedule'],index);step=cfg['schedule']['control_step_s']
    dt=30.;t=start+np.arange(len(schedule)*step/dt+1)*dt
    command=np.r_[schedule[0],plan_for_endpoints(schedule,t[1:]-start,step)]
    hours=(t-start)/3600
    # Test-only deterministic signal, not a physical simulation or proposed generator.
    zone=294+.2*index+np.sin(hours*.4)+.02*hours+.05*np.cumsum(command)/120
    supply=300+.3*index+1.4*np.sin(hours)+2*command+.03*hours
    fields={'time':t,'reaTZon_y':zone,'reaTSup_y':supply,'oveHeaPumY_u':command,
            'ovePum_u':(command>0).astype(float),'oveFan_u':(command>0).astype(float)}
    return Episode(index,t,fields,schedule,start,step)

def synthetic_source(repo,cfg):
    cfg=copy.deepcopy(cfg)
    for i in range(7):
        ep=synthetic_episode(cfg,i);p=Path(repo)/cfg['source_data'][str(i)]['relative_path'];p.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(p,'wt',encoding='utf-8',newline='') as f:
            w=csv.writer(f);cols=list(ep.fields);w.writerow(cols)
            for j in range(len(ep.time)):w.writerow([ep.fields[k][j] for k in cols])
        cfg['source_data'][str(i)]['sha256']=file_hash(p)
    return cfg

def tiny_config():
    cfg=config()
    for cc in cfg['candidates']:
        cc.update(sample_s=300,context_s=1800,horizon_s=900,origin_stride_s=3600)
        cfg['recent_window_s'][cc['name']]=900
    return cfg

class DataContracts(unittest.TestCase):
    def setUp(self):
        self.cfg=config();self.ep=synthetic_episode(self.cfg);self.cc=self.cfg['candidates'][0]
        self.origin=self.ep.start_time+12*3600

    def test_01_schedule_matches_published_hashes(self):
        for i in range(7):self.assertEqual(schedule_hash(regenerate_schedule(self.cfg['schedule'],i)),self.cfg['source_data'][str(i)]['schedule_sha256'])

    def test_02_boundary_uses_preceding_interval(self):
        np.testing.assert_array_equal(plan_for_endpoints([.35,.7],np.array([1,30,300,330,600]),300),[.35,.35,.35,.7,.7])

    def test_03_no_plan_for_t0_or_beyond_end(self):
        for x in ([0],[601]):
            with self.assertRaises(ContractError):plan_for_endpoints([.35,.7],x,300)

    def test_04_context_endpoints_inclusive(self):
        v=make_view(self.ep,self.cc,self.origin)
        self.assertEqual(len(v.target_context),241);self.assertEqual(v.time_context[0],self.origin-7200)
        self.assertEqual(v.time_context[-1],self.origin)

    def test_05_older_target_cannot_enter_features(self):
        old=features(make_view(self.ep,self.cc,self.origin),self.cc,self.cfg)[0]
        ep=copy.deepcopy(self.ep);ep.fields[self.cc['target']][ep.time<self.origin-7200]+=999
        new=features(make_view(ep,self.cc,self.origin),self.cc,self.cfg)[0]
        for arm in ARMS:np.testing.assert_array_equal(old[arm],new[arm])

    def test_06_older_command_cannot_enter_features_or_age(self):
        old=features(make_view(self.ep,self.cc,self.origin),self.cc,self.cfg)[0]
        ep=copy.deepcopy(self.ep);ep.fields['oveHeaPumY_u'][ep.time<self.origin-7200]+=99
        new=features(make_view(ep,self.cc,self.origin),self.cc,self.cfg)[0]
        for arm in ARMS:np.testing.assert_array_equal(old[arm],new[arm])

    def test_07_future_targets_not_in_features(self):
        old=features(make_view(self.ep,self.cc,self.origin),self.cc,self.cfg)[0]
        ep=copy.deepcopy(self.ep);ep.fields[self.cc['target']][ep.time>self.origin]=-100000
        new=features(make_view(ep,self.cc,self.origin),self.cc,self.cfg)[0]
        for arm in ARMS:np.testing.assert_array_equal(old[arm],new[arm])

    def test_08_future_realized_command_not_used_as_plan(self):
        ep=copy.deepcopy(self.ep);ep.fields['oveHeaPumY_u'][ep.time>self.origin]=10000
        a=make_view(self.ep,self.cc,self.origin);b=make_view(ep,self.cc,self.origin)
        np.testing.assert_array_equal(a.command_plan,b.command_plan)

    def test_09_plan_changes_every_arm(self):
        a=features(make_view(self.ep,self.cc,self.origin),self.cc,self.cfg)[0]
        ep=copy.deepcopy(self.ep);k=int((self.origin-ep.start_time)/ep.step_s);ep.schedule[k:k+3]=.123
        b=features(make_view(ep,self.cc,self.origin),self.cc,self.cfg)[0]
        for arm in ARMS:self.assertGreater(np.abs(a[arm]-b[arm]).max(),0)

    def test_10_common_plan_prefix_identical(self):
        f,s,a=features(make_view(self.ep,self.cc,self.origin),self.cc,self.cfg)
        n=a['shared_prefix_features']
        np.testing.assert_array_equal(f[ARMS[0]],f[ARMS[1]][:n]);np.testing.assert_array_equal(f[ARMS[0]],f[ARMS[2]][:n])

    def test_11_no_out_of_context_named_lags(self):
        _,s,_=features(make_view(self.ep,self.cc,self.origin),self.cc,self.cfg)
        self.assertNotIn('y_lag_28800s',s[ARMS[0]])
        self.assertIn('y_lag_7200s',s[ARMS[0]])
        self.assertFalse(any('u_21600s' in x for x in s[ARMS[1]]))

    def test_12_censored_age_not_full_episode_age(self):
        v=make_view(self.ep,self.cc,self.origin);v.command_context[:]=.35
        a,s,_=features(v,self.cc,self.cfg);names=s[ARMS[1]]
        self.assertEqual(a[ARMS[1]][names.index('command_age_capped_at_context_s')],7200)
        self.assertEqual(a[ARMS[1]][names.index('command_age_left_censored')],1)

    def test_13_future_target_does_not_change_panels(self):
        v=make_view(self.ep,self.cc,self.origin);a=label_panels(v,self.cc,self.cfg)
        v.target_context[:]=np.arange(len(v.target_context))*100
        self.assertEqual(a,label_panels(v,self.cc,self.cfg))

    def test_14_upcoming_change_not_called_settled(self):
        v=make_view(self.ep,self.cc,self.origin);v.command_context[:]=0;v.command_plan[:]=1
        f,_=label_panels(v,self.cc,self.cfg)
        self.assertTrue(f['CHANGE_IN_PLAN']);self.assertFalse(f['NO_RECENT_OR_PLANNED_CHANGE'])
        self.assertFalse(f['RECENT_PAST_CHANGE'])

    def test_15_exact_future_horizon_and_grid(self):
        v=make_view(self.ep,self.cc,self.origin)
        self.assertEqual(v.plan_times[-1]-self.origin,900);self.assertEqual(v.plan_times[0]-self.origin,30)
        bad=copy.deepcopy(self.cc);bad['sample_s']=31
        with self.assertRaises(ContractError):make_view(self.ep,bad,self.origin)

    def test_16_reserve_refused_before_open(self):
        with tempfile.TemporaryDirectory() as d:
            s=PinnedSource(d,self.cfg,Path(d)/'out')
            with self.assertRaisesRegex(ContractError,'RESERVE'):s.load(7)
            self.assertEqual(s.opened_episodes,[])

    def test_17_dev_refused_before_seal(self):
        with tempfile.TemporaryDirectory() as d:
            s=PinnedSource(d,self.cfg,Path(d)/'out')
            with self.assertRaisesRegex(ContractError,'BEFORE_SEAL'):s.load(5)

    def test_18_hash_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/self.cfg['source_data']['0']['relative_path'];p.parent.mkdir(parents=True);p.write_bytes(b'bad')
            with self.assertRaisesRegex(ContractError,'HASH'):PinnedSource(d,self.cfg,Path(d)/'out').load(0)

    def test_19_symlink_source_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/self.cfg['source_data']['0']['relative_path'];p.parent.mkdir(parents=True)
            other=Path(d)/'other';other.write_text('bad');p.symlink_to(other)
            with self.assertRaises(ContractError):PinnedSource(d,self.cfg,Path(d)/'out').load(0)

    def test_20_all_five_dimensions_and_labels(self):
        for cc in self.cfg['candidates']:
            p=prepare(self.ep,cc,self.cfg)
            self.assertTrue(np.isfinite(p.Y).all());self.assertEqual(p.Y.shape[1],cc['horizon_s']//cc['sample_s'])
            self.assertTrue(all(len(m)==len(p.Y) for m in p.masks.values()))

class RidgeContracts(unittest.TestCase):
    def test_21_svd_matches_normal_equations(self):
        rng=np.random.default_rng(24);X=rng.normal(size=(50,8));Y=rng.normal(size=(50,3))
        for a,m in ridge_path(X,Y,[.01,1,100]).items():
            Z=(X-X.mean(0))/X.std(0)
            b=np.linalg.solve(Z.T@Z+a*np.eye(8),Z.T@(Y-Y.mean(0)))
            np.testing.assert_allclose(m.coef,b,atol=1e-10,rtol=1e-9)

    def test_22_rank_deficient_more_columns_than_rows(self):
        rng=np.random.default_rng(2);X=rng.normal(size=(10,20));X[:,1]=X[:,0];Y=rng.normal(size=(10,5))
        for m in ridge_path(X,Y,[.01,1,100]).values():self.assertTrue(np.isfinite(m.predict(X)).all())

    def test_23_standardization_training_only(self):
        X=np.arange(20.).reshape(10,2);Y=X[:,:1]
        m=ridge_path(X,Y,[1])[1.];old=m.mean_x.copy();m.predict(np.full((2,2),999))
        np.testing.assert_array_equal(old,m.mean_x)

    def test_24_zero_variance_and_constant_target(self):
        X=np.ones((10,3));Y=np.ones((10,2))*294.5
        p=ridge_path(X,Y,[1])[1.].predict(X)
        np.testing.assert_array_equal(p,Y)

    def test_25_alpha_tie_prefers_stronger(self):
        a,_=choose_alpha([{.1:2.,1.:2.,10.:2.}],[.1,1.,10.]);self.assertEqual(a,10)

    def test_26_exact_model_save_restore(self):
        X=np.arange(30.).reshape(10,3);Y=np.sin(X[:,:2]);m=ridge_path(X,Y,[1])[1.]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.npz';m.save(p);r=RidgeModel.load(p)
            np.testing.assert_array_equal(m.predict(X),r.predict(X))

    def test_27_sklearn_parity_if_installed(self):
        try:from sklearn.linear_model import Ridge
        except ImportError:self.skipTest('scikit-learn not installed; no installation attempted')
        rng=np.random.default_rng(4);X=rng.normal(size=(40,7));Y=rng.normal(size=(40,2));X[:,2]=3
        for a,m in ridge_path(X,Y,[.01,1.,100.]).items():
            Z=(X-m.mean_x)/m.scale_x
            ref=Ridge(alpha=a,solver='svd').fit(Z,Y).predict(Z)
            np.testing.assert_allclose(m.predict(X),ref,atol=1e-10,rtol=1e-8)

    def test_28_invalid_values_raise(self):
        with self.assertRaises(ContractError):ridge_path(np.full((4,2),np.nan),np.ones((4,1)),[1])
        with self.assertRaises(ContractError):ridge_path(np.ones((4,2)),np.ones((4,1)),[0])

class EndToEndContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name)
        cls.cfg=synthetic_source(cls.path/'repo',tiny_config())
        cls.public=cls.path/'public';cls.private=cls.path/'private'
        # Full engine, small numerical dimensions, real production IO/phase guards.
        cls.result=run_engine(cls.path/'repo',ROOT,cls.public,cls.private,config=cls.cfg)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_29_engine_completed_without_gate(self):
        self.assertEqual(self.result['status'],'COMPLETED_BOUNDED_RECHECK')
        self.assertEqual(self.result['research_verdict'],'NOT_AUTOMATICALLY_DECIDED')

    def test_30_reserve_not_opened_and_dev_after_seal(self):
        a=read_json(self.public/'DATA_ACCESS.json');self.assertEqual(a['opened_episodes'],list(range(7)))
        phases=[x['phase'] for x in a['records'] if 'episode' in x]
        self.assertEqual(phases[:5],['TRAIN_ONLY']*5);self.assertEqual(phases[5:],['AFTER_PRE_DEV_SEAL']*2)

    def test_31_nested_selection_never_uses_outer_holdout(self):
        for p in (self.public/'selection').glob('*SELECTION.json'):
            for r in read_json(p):
                if r['purpose']=='OUTER_TRAIN':self.assertLess(max(r['tuning_validation_episodes']),r['evaluation_episode'])
                self.assertFalse(any(x in [5,6,7] for x in r['tuning_validation_episodes']))

    def test_32_no_candidate_selection_all_tasks(self):
        s=read_json(self.public/'PRE_DEV_SEAL.json')
        self.assertEqual(s['target_selection'],'NONE_ALL_FIVE_RETAINED');self.assertEqual(len(s['model_states']),5)

    def test_33_predictions_independently_replay(self):
        ps=list((self.public/'replays').glob('*.json'));self.assertEqual(len(ps),25)
        for p in ps:self.assertLessEqual(read_json(p)['max_origin_MAE_diff'],1e-10)

    def test_34_budget_true_attempt_counts(self):
        c=read_json(self.public/'RUN_MANIFEST.json')['actual_counts']
        self.assertLessEqual(c['ridge_fit_attempts'],540);self.assertGreaterEqual(c['ridge_fit_attempts'],480)
        self.assertEqual(c['ridge_fit_attempts'],c['ridge_fit_completed']);self.assertEqual(c['neural_fit'],0)

    def test_35_empty_panels_no_nan_no_fake_zero(self):
        with (self.public/'SCORES.csv').open() as f:rows=list(csv.DictReader(f))
        empty=[r for r in rows if r['n_origins']=='0'];self.assertTrue(empty)
        for r in empty:self.assertEqual(r['mae_K'],'');self.assertEqual(r['status'],'NO_PANEL_SUPPORT')
        self.assertNotIn('NaN',(self.public/'EFFECT_SUMMARY.json').read_text())

    def test_36_invalid_seal_cannot_authorize_dev(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'seal.json';write_json(p,{'seal_sha256':'bad','target_selection':'NONE_ALL_FIVE_RETAINED'})
            s=PinnedSource(self.path/'repo',self.cfg,Path(d))
            with self.assertRaises(ContractError):s.enable_dev(p)

    def test_37_frozen_model_hashes_match_seal(self):
        seal=read_json(self.public/'PRE_DEV_SEAL.json')
        for models in seal['model_states'].values():
            for m in models.values():self.assertEqual(file_hash(self.private/m['file']),m['sha256'])

    def test_38_all_negative_is_not_information_absence(self):
        self.assertNotIn('NO_HISTORY_SIGNAL',self.result['status'])
        self.assertFalse(self.result['neural_evaluation']);self.assertFalse(self.result['TMD_computed'])

    def test_39_fixed_alpha_comparison_retained(self):
        with (self.public/'SCORES.csv').open() as f:rows=list(csv.DictReader(f))
        models=set(r['model'] for r in rows);self.assertEqual(models,set(MODEL_NAMES))
        self.assertEqual(set(r['alpha'] for r in rows if r['model'].endswith('_A1')),{'1.0'})

    def test_40_temporal_overlap_rejected(self):
        cfg=self.cfg;ep=synthetic_episode(cfg);p=prepare(ep,cfg['candidates'][0],cfg)
        parts={0:p,1:p}
        with self.assertRaises(ContractError):assert_temporal(parts,[0],1)

class PublicationContracts(unittest.TestCase):
    def test_41_local_bare_push_exact_scope_preserves_unrelated(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);repo=d/'repo';bare=d/'bare.git';repo.mkdir()
            subprocess.run(['git','init','--bare',str(bare)],capture_output=True,check=True)
            subprocess.run(['git','init','-b','main',str(repo)],capture_output=True,check=True)
            git(repo,'config','user.name','Package selftest');git(repo,'config','user.email','fixture@example.invalid')
            (repo/'base').write_text('base\n');(repo/'.gitignore').write_text('*.log\n*.npz\n');git(repo,'add','base','.gitignore');git(repo,'commit','-m','fixture base')
            git(repo,'remote','add','origin',str(bare));git(repo,'push','-u','origin','main')
            head=git(repo,'rev-parse','HEAD').stdout.strip()
            (repo/'base').write_text('unrelated working change\n');(repo/'other').write_text('not staged\n')
            p=repo/'results/new/FAILURE.json';write_json(p,{'status':'FAILED_SYNTHETIC_FIXTURE'})
            (repo/'results/new/worker.log').write_text('fixture failure log\n')
            receipt=publish_exact(repo,head,'main',['results/new/FAILURE.json','results/new/worker.log'],'preserve fixture failure')
            self.assertEqual(receipt['status'],'PUSH_VERIFIED')
            self.assertEqual((repo/'base').read_text(),'unrelated working change\n')
            self.assertEqual((repo/'other').read_text(),'not staged\n')
            self.assertEqual(set(git(repo,'diff-tree','--no-commit-id','--name-only','-r','HEAD').stdout.split()),{'results/new/FAILURE.json','results/new/worker.log'})
            self.assertEqual(git(repo,'show','HEAD:base').stdout,'base\n')

    def test_42_existing_staged_change_refused(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(['git','init','-b','main',d],capture_output=True,check=True)
            p=Path(d)/'user.txt';p.write_text('user');git(d,'add','user.txt')
            with self.assertRaisesRegex(ContractError,'STAGED'):empty_index(d)
            self.assertIn('user.txt',git(d,'diff','--cached','--name-only').stdout)

if __name__=='__main__':unittest.main()
