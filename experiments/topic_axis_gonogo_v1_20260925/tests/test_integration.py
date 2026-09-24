import copy,hashlib,json,os,subprocess,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from triage.common import Case,write_json,write_csv,read_json,hash_file,ContractError
from triage.engine import main_topic
from triage.gitops import git,publish_exact
from triage.wind import load_panel,make_cases
from triage.respiration import read_signal

CFG={'alphas':[.01,.1,1.,10.,100.],'device':'auto','alpha_family':.05/3,'min_contrast_groups':4,'min_agreement':.5,
     'download_bytes_per_topic':1024**2,'download_seconds':1}

def fake_data(vector=False):
    rng=np.random.default_rng(515);out={}
    for split in ('train','cal','test'):
        cs=[]
        for g in range(5):
            for j in range(12):
                t=np.arange(32);x=(.15*np.sin(t/(3+g/4))+.1*t)[:,None]
                if vector:x=np.c_[x[:,0],.02*t**1.15];x=x-x[-1]
                h=4;v=(x[-1]-x[-4])/3;y=x[-1]+np.arange(1,h+1)[:,None]*v
                score=j/11;noise=np.sin(np.arange(h)[:,None])*score*.1
                y=y+noise
                cs.append(Case(split+str(g)+':'+str(j),'group'+str(g),x,y,score,score,
                     np.array([1.,g,.1,.2,.3]),y-noise,1.,np.array([score])))
        out[split]=cs
    return out

class DummyNative:
    """Strictly test-only. Production never falls back to this class."""
    def __init__(self,*a):self.windows=0;self.forwards=0
    def predict(self,cases,canonicalize=False):
        self.windows+=len(cases);self.forwards+=1
        preds=[]
        for c in cases:
            v=np.mean(np.diff(c.x[-4:],axis=0),axis=0)[:c.y.shape[1]]
            p=c.x[-1,:c.y.shape[1]]+np.arange(1,len(c.y)+1)[:,None]*v
            preds.append(p+(0. if canonicalize else c.score*.25))
        return preds
    def close(self):pass

class Integration(unittest.TestCase):
    def test49_engine_complete_single_topic(self):
        with tempfile.TemporaryDirectory() as d, patch('triage.respiration.prepare',return_value=fake_data()),patch('triage.engine.Native',DummyNative):
            p=Path(d);r=main_topic('respiration',p,p/'cache',p/'out',CFG)
            self.assertTrue((p/'out/SCORE_ROWS.csv').is_file());self.assertIn('axis_status',r)
            self.assertFalse(r['neural_adaptation_tested']);self.assertEqual(r['ridge_coefficient_fits'],12)
    def test50_engine_coordinate_stress(self):
        with tempfile.TemporaryDirectory() as d,patch('triage.coordinates.prepare',return_value=fake_data(True)),patch('triage.engine.Native',DummyNative):
            p=Path(d);r=main_topic('coordinates',p,p/'cache',p/'out',CFG)
            self.assertIn('coordinate_stress',r);self.assertFalse(r['coordinate_stress']['rotations_are_independent_samples'])
    def test51_engine_missing_native_not_nogo(self):
        from triage.common import Blocked
        with tempfile.TemporaryDirectory() as d,patch('triage.respiration.prepare',return_value=fake_data()),patch('triage.engine.Native',side_effect=Blocked('none')):
            p=Path(d);r=main_topic('respiration',p,p/'cache',p/'out',CFG)
            self.assertEqual(r['axis_status'],'BLOCKED_F0_REFERENCE')
    def test52_wind_csv_loader_timestamps_and_masks(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);rows=[]
            for t in range(144):
                for i in range(1,13):
                    rows.append({'TurbID':i,'Day':1,'Tmstamp':f'{t//6:02d}:{t%6*10:02d}','Wspd':5.,'Wdir':0.,'Ndir':t*2.,'Pab1':0.,'Pab2':0.,'Pab3':0.,'Patv':float(t+i)})
            write_csv(p/'raw.csv',rows);write_csv(p/'loc.csv',[{'TurbID':i,'x':i*82.,'y':(i%3)*82.} for i in range(1,13)])
            panel,xy,info=load_panel(('csv',p/'raw.csv',p/'loc.csv'))
            self.assertEqual(panel['Patv'][5,0],6.);self.assertTrue(panel['valid'][5,0]);self.assertFalse(panel['valid'][145,0])
    def test53_wind_archive_does_not_extract_paths(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);write_csv(p/'raw.csv',[{'TurbID':1,'Day':1,'Tmstamp':'00:00','Wspd':3,'Wdir':0,'Ndir':0,'Pab1':0,'Pab2':0,'Pab3':0,'Patv':1}])
            write_csv(p/'loc.csv',[{'TurbID':i,'x':i,'y':0} for i in range(1,13)])
            z=p/'d.zip'
            with zipfile.ZipFile(z,'w') as f:f.write(p/'raw.csv','../evil.csv');f.write(p/'loc.csv','location.csv')
            panel,_,_=load_panel(('zip',z,('../evil.csv','location.csv')))
            self.assertFalse((p.parent/'evil.csv').exists());self.assertEqual(panel['Patv'][0,0],1)
    def test54_resp_loader_causal_block_mean(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'resp.csv'
            # Clinical source is not copied: synthetic data in its verified schema.
            n=50125;t=np.arange(n)/125;x=np.arange(n)*.001
            with p.open('w') as f:
                f.write('Time [s], RESP\n')
                for a,b in zip(t,x):f.write(f'{a:.3f},{b:.3f}\n')
            y=read_signal(p);self.assertAlmostEqual(y[0],.012);self.assertAlmostEqual(y[1],.037)
    def test55_git_failed_report_and_unrelated_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);remote=p/'bare';repo=p/'work'
            subprocess.run(['git','init','--bare',str(remote)],capture_output=True,check=True)
            subprocess.run(['git','init','-b','main',str(repo)],capture_output=True,check=True)
            git(repo,'config','user.email','author-test@example.invalid');git(repo,'config','user.name','LocalTest')
            (repo/'unrelated.txt').write_text('before\n');git(repo,'add','unrelated.txt');git(repo,'commit','-m','base');git(repo,'remote','add','origin',str(remote));git(repo,'push','-u','origin','main')
            initial={'head':git(repo,'rev-parse','HEAD').stdout.strip()}
            (repo/'unrelated.txt').write_text('keep my dirty change\n')
            src=repo/'experiments/test/a.py';src.parent.mkdir(parents=True);src.write_text('x = 1\n')
            out=repo/'results/test/r';out.mkdir(parents=True);write_json(out/'RESULT.json',{'status':'FAILED_FOR_TEST_ONLY'})
            # Generated console trailing spaces and CSV CRLF must not break publication.
            (out/'CONSOLE.txt').write_bytes(b'progress  \r\n');(out/'SCORES.csv').write_bytes(b'a,b\r\n1,2\r\n')
            rec=publish_exact(repo,initial,[src],list(out.iterdir()),'publish diagnostic failure')
            self.assertEqual(rec['status'],'PUSH_VERIFIED')
            self.assertIn('unrelated.txt',git(repo,'status','--porcelain').stdout)
            changed=git(repo,'diff-tree','--no-commit-id','--name-only','-r','HEAD').stdout
            self.assertNotIn('unrelated.txt',changed)
    def test56_git_existing_index_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);subprocess.run(['git','init','-b','main',str(p)],capture_output=True,check=True)
            git(p,'config','user.email','test@invalid');git(p,'config','user.name','Test')
            (p/'a').write_text('a');git(p,'add','a');git(p,'commit','-m','init')
            head=git(p,'rev-parse','HEAD').stdout.strip();(p/'b').write_text('b');git(p,'add','b')
            with self.assertRaises(ContractError):publish_exact(p,{'head':head},[],[],'do not commit')
    def test57_run_entrypoint_help_is_stdlib(self):
        root=Path(__file__).parents[1]
        q=subprocess.run([os.sys.executable,'-B',str(root/'run.py'),'--help'],capture_output=True)
        self.assertEqual(q.returncode,0)
    def test58_numeric_bank_no_test_labels(self):
        from triage.wind import bank_for
        rng=np.random.default_rng(2);n=42*144
        pp={'Patv':rng.normal(100,10,(n,12)),'valid':np.ones((n,12),bool),'Ndir':rng.uniform(0,360,(n,12))}
        xy=np.c_[np.arange(12),np.zeros(12)];b,_=bank_for(pp,xy,0,range(1,21))
        pp['Patv'][20*144:]=-1e12;pp['valid'][20*144:]=False
        c,_=bank_for(pp,xy,0,range(1,21));np.testing.assert_array_equal(b,c)
    def test59_all_topic_error_isolation(self):
        # Driver runs all topics in separate subprocesses without a success-dependent gate.
        text=(Path(__file__).parents[1]/'run.py').read_text()
        self.assertIn("for topic in cfg['topics']:",text);self.assertNotIn('if p.returncode:break',text)
    def test60_no_training_import_path(self):
        root=Path(__file__).parents[1]
        text=(root/'triage/predictors.py').read_text()
        self.assertNotIn('optimizer.step',text);self.assertNotIn('.backward(',text)

if __name__=='__main__':unittest.main()
