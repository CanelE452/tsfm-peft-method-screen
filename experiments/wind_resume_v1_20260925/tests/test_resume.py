"""Local fixtures only. No real wind data, Chronos inference or GitHub writes."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import resume_core as c

# Tests receive only a source directory already available to the author. This is
# not a production switch: the executable run.py exposes no testing parameters.
ORIGINAL = Path(os.environ.get('WIND_RESUME_TEST_ORIGINAL', '/nonexistent'))

def remanifest(pkg):
    listed = {p.relative_to(pkg).as_posix():c.digest(p) for p in sorted(pkg.rglob('*'))
              if p.is_file() and p.name != 'PACKAGE_MANIFEST.json'
              and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    c.write_json(pkg/'PACKAGE_MANIFEST.json', {'name':c.NAME,'version':'local-fixture-only','files':listed})

class Basic(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_01_exclusive_retry_claim(self):
        p=c.claim_once(self.root,{'attempt':1});old=p.read_bytes()
        with self.assertRaises(c.ContractError):c.claim_once(self.root,{'attempt':2})
        self.assertEqual(old,p.read_bytes())
    def test_02_only_wind_worker(self):
        cmd=c.worker_command(Path('python'),Path('/engine'),Path('/repo'),Path('/cache'),Path('/out'),Path('/data/p.csv'),Path('/data/l.csv'))
        self.assertEqual(cmd[cmd.index('--topic')+1],'wind')
        self.assertNotIn('triage.run',cmd);self.assertNotIn('respiration',cmd)
        self.assertEqual(cmd[cmd.index('--config')+1],'/engine/RUN_CONFIG.json')
    def test_03_existing_original_marker_not_used_as_retry_lock(self):
        old=self.root/'old.started';old.write_text('old')
        c.claim_once(self.root/'new',{})
        self.assertEqual(old.read_text(),'old')
    def test_04_input_size_check(self):
        p=self.root/'data.csv';p.write_bytes(b'a,b\n1,2\n')
        info=c.inspect_input(p,{'name':p.name,'bytes':8})
        self.assertEqual(info['md5'],hashlib.md5(p.read_bytes()).hexdigest())
        with self.assertRaises(c.ContractError):c.inspect_input(p,{'name':p.name,'bytes':9})
    def test_05_input_name_not_disguised(self):
        p=self.root/'a.csv';p.write_bytes(b'x')
        with self.assertRaises(c.ContractError):c.inspect_input(p,{'name':'b.csv','bytes':1})
    def test_06_input_metadata_md5(self):
        p=self.root/'a.csv';p.write_bytes(b'x');info={'csv':c.inspect_input(p,{'name':'a.csv','bytes':1})}
        m=self.root/'meta.json';c.write_json(m,{'files':[{'name':'a.csv','size':1,'computed_md5':hashlib.md5(b'x').hexdigest()}]})
        result=c.check_cached_metadata(m,{'csv':{'name':'a.csv','bytes':1}},info)
        self.assertEqual(result['selected_files'][0]['md5'],info['csv']['md5'])
    def test_07_bad_md5_stops(self):
        m=self.root/'meta.json';c.write_json(m,{'files':[{'name':'a.csv','size':1,'computed_md5':'0'*32}]})
        with self.assertRaises(c.ContractError):
            c.check_cached_metadata(m,{'csv':{'name':'a.csv','bytes':1}},{'csv':{'md5':'1'*32}})
    def test_08_missing_metadata_no_download(self):
        with self.assertRaises(c.ContractError):c.check_cached_metadata(self.root/'absent',{}, {})
    def test_09_snapshot_detects_change(self):
        p=self.root/'a';p.write_text('first');b=c.snapshot_tree(self.root);p.write_text('second')
        self.assertNotEqual(b,c.snapshot_tree(self.root))
    def test_10_snapshot_rejects_symlink(self):
        p=self.root/'a';p.write_text('x');(self.root/'b').symlink_to(p)
        with self.assertRaises(c.ContractError):c.snapshot_tree(self.root)
    def test_11_reject_renamed_retry_destination(self):
        dest=self.root/'experiments'/c.NAME;dest.mkdir(parents=True)
        with self.assertRaises(c.ContractError):c.copy_retry_package(ROOT,self.root,{'files':{}})
    def test_12_manifest_tamper_detected(self):
        pkg=self.root/'pkg';pkg.mkdir();(pkg/'run.py').write_text('x');remanifest(pkg)
        c.verify_package(pkg);(pkg/'run.py').write_text('z')
        with self.assertRaises(c.ContractError):c.verify_package(pkg)
    def test_13_extra_python_file_rejected(self):
        pkg=self.root/'pkg';pkg.mkdir();(pkg/'run.py').write_text('x');remanifest(pkg)
        (pkg/'extra.py').write_text('y')
        with self.assertRaises(c.ContractError):c.verify_package(pkg)
    def test_14_bytecode_not_in_manifest(self):
        pkg=self.root/'pkg';pkg.mkdir();(pkg/'run.py').write_text('x');remanifest(pkg)
        (pkg/'__pycache__').mkdir();(pkg/'__pycache__'/'run.pyc').write_bytes(b'cache')
        self.assertEqual(set(c.verify_package(pkg)['files']),{'run.py'})
    def test_15_no_topic_or_retry_override_flags(self):
        p=subprocess.run([sys.executable,'-B',str(ROOT/'run.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0)
        self.assertNotIn('--topic ',p.stdout);self.assertNotIn('--force',p.stdout)
    def test_16_science_config_pin_includes_all_modules(self):
        pin=c.read_json(ROOT/'ENGINE_PIN.json')
        for p in ('triage/wind.py','triage/statistics.py','triage/predictors.py','triage/engine.py','RUN_CONFIG.json'):
            self.assertIn(p,pin['source_files_sha256'])
        self.assertEqual(pin['topic_timeout_seconds'],900)
        self.assertEqual(pin['parent_commit'],'c085531186fa5d6a7505c2aa690d3d93a4fbcf86')

@unittest.skipUnless((ORIGINAL/'PACKAGE_MANIFEST.json').is_file(), 'original package required for local integration')
class Integration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.top=Path(self.tmp.name);self.repo=self.top/'repo';self.repo.mkdir()
        self.pkg=self.top/'package';shutil.copytree(ROOT,self.pkg,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        self.pin=c.read_json(self.pkg/'ENGINE_PIN.json')
        source=self.repo/self.pin['source_path'];shutil.copytree(ORIGINAL,source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        parent=self.repo/self.pin['parent_result_path'];(parent/'wind').mkdir(parents=True)
        c.write_json(parent/'wind'/'RESULT.json',{'topic':'wind','axis_status':'BLOCKED_DATA_OR_ENVIRONMENT',
                     'reason':self.pin['allowed_parent_reason'],'scientific_no_go':False})
        c.write_json(parent/'ALL_TOPICS.json',{'respiration':'original blocked','coordinates':'original result'})
        self.pin['parent_wind_git_blob_sha1']=c.git_blob_sha(parent/'wind'/'RESULT.json')
        marker=self.repo/self.pin['original_marker_path'];marker.parent.mkdir(parents=True);marker.write_text('old marker\n')
        inputs={};md=[]
        for key,f in self.pin['inputs'].items():
            path=self.repo/'.cache'/'sdwpf_20260925'/f['name'];path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'fixture raw only\n')
            f['bytes']=path.stat().st_size;inputs[key]=path
            md.append({'name':f['name'],'size':f['bytes'],'computed_md5':c.digest(path,'md5')})
        c.write_json(self.repo/self.pin['original_metadata_path'],{'files':md})
        (self.repo/'.gitignore').write_text('.cache/\n__pycache__/\n')
        (self.repo/'.venv'/'bin').mkdir(parents=True);(self.repo/'.venv'/'bin'/'python').symlink_to(sys.executable)
        (self.repo/'.gitignore').write_text('.cache/\n__pycache__/\n.venv/\n')
        c.git(self.repo,'init','-b','main');c.git(self.repo,'config','user.name','Local Test');c.git(self.repo,'config','user.email','local@example.invalid')
        (self.repo/'unrelated.txt').write_text('unchanged\n')
        c.git(self.repo,'add','.');c.git(self.repo,'commit','-m','fixture parent')
        self.pin['parent_commit']=c.git(self.repo,'rev-parse','HEAD').stdout.strip()
        c.write_json(self.pkg/'ENGINE_PIN.json',self.pin);remanifest(self.pkg)
        self.bare=self.top/'bare.git'
        subprocess.run(['git','init','--bare',str(self.bare)],check=True,capture_output=True)
        c.git(self.repo,'remote','add','origin',str(self.bare));c.git(self.repo,'push','origin','main')
        self.args=argparse.Namespace(repo=str(self.repo),publish=True,wind_csv=str(inputs['wind_csv']),wind_locations=str(inputs['wind_locations']))
        sys.path.insert(0,str(source));self.gitops=importlib.import_module('triage.gitops')
        self.parent=parent;self.marker=marker
        self.before_marker=marker.read_bytes();self.before_parent=c.snapshot_tree(parent)
        self.calls=[]
    def tearDown(self):self.tmp.cleanup()
    def validation(self,repo,publish):
        self.assertFalse(c.git(repo,'diff','--cached','--name-only').stdout.strip())
        h=c.git(repo,'rev-parse','HEAD').stdout.strip()
        self.assertEqual(c.git(repo,'ls-remote','--heads','origin','refs/heads/main').stdout.split()[0],h)
        return {'head':h,'branch':'main','dirty_before':c.git(repo,'status','--porcelain').stdout}
    def fake_worker(self,command,source,public,timeout):
        self.calls.append(command)
        self.assertEqual(command[command.index('--topic')+1],'wind');self.assertEqual(timeout,900)
        (public/'CONSOLE.txt').write_text('test double only; NO native run\n')
        c.write_json(public/'RESULT.json',{'topic':'wind','axis_status':'TEST_DOUBLE_ONLY','action':'NOT_A_SCIENTIFIC_RESULT'})
        return {'command':command,'returncode':0,'elapsed_s':0.0,'only_topic':'wind'}
    def go(self,worker=None):
        with mock.patch.object(self.gitops,'validate',side_effect=self.validation),mock.patch.object(c,'run_worker',side_effect=worker or self.fake_worker):
            return c.execute(self.pkg,self.args)
    def test_17_full_local_publish_preserves_originals(self):
        (self.repo/'unrelated.txt').write_text('local work not staged\n')
        self.assertEqual(self.go(),0);self.assertEqual(len(self.calls),1)
        self.assertEqual(self.before_marker,self.marker.read_bytes());self.assertEqual(self.before_parent,c.snapshot_tree(self.parent))
        receipt=c.read_json(self.repo/'.cache'/c.NAME/'EXECUTION_RECEIPT.json')
        self.assertEqual(receipt['status'],'PUSH_VERIFIED')
        tracked=c.git(self.repo,'show','--pretty=','--name-only','HEAD').stdout.splitlines()
        self.assertTrue(all(p.startswith('experiments/'+c.NAME+'/') or p.startswith('results/'+c.NAME+'/') for p in tracked if p))
        self.assertNotIn('unrelated.txt',tracked);self.assertEqual((self.repo/'unrelated.txt').read_text(),'local work not staged\n')
        self.assertTrue(all('.cache/' not in p for p in tracked));self.assertFalse(c.git(self.repo,'diff','--cached','--name-only').stdout.strip())
    def test_18_second_claim_does_not_run_worker(self):
        self.assertEqual(self.go(),0)
        with self.assertRaises(c.ContractError):self.go()
        self.assertEqual(len(self.calls),1)
    def test_19_blocked_worker_is_published_not_go(self):
        def fail(cmd,source,public,timeout):
            (public/'CONSOLE.txt').write_text('missing model; fixture only\n')
            c.write_json(public/'RESULT.json',{'topic':'wind','axis_status':'BLOCKED_MODEL','scientific_no_go':False})
            return {'command':cmd,'returncode':3,'elapsed_s':0.0,'only_topic':'wind'}
        self.assertEqual(self.go(fail),3)
        r=c.read_json(self.repo/'.cache'/c.NAME/'EXECUTION_RECEIPT.json')
        self.assertEqual(r['status'],'PUSH_VERIFIED');self.assertEqual(r['axis_status'],'BLOCKED_MODEL')
    def test_20_source_tamper_blocks_before_worker(self):
        (self.repo/self.pin['source_path']/'triage'/'wind.py').write_text('changed')
        with self.assertRaises(c.ContractError):self.go()
        self.assertEqual(len(self.calls),0)
    def test_21_previous_wind_score_prevents_retry(self):
        (self.parent/'wind'/'SCORE_ROWS.csv').write_text('existing score')
        with self.assertRaises(c.ContractError):self.go()
        self.assertEqual(len(self.calls),0)
    def test_22_runtime_original_mutation_refuses_publish(self):
        old_head=c.git(self.repo,'rev-parse','HEAD').stdout.strip()
        def mutate(cmd,source,public,timeout):
            ret=self.fake_worker(cmd,source,public,timeout)
            self.marker.write_text('tampered');return ret
        self.assertEqual(self.go(mutate),2)
        self.assertEqual(old_head,c.git(self.repo,'rev-parse','HEAD').stdout.strip())
        r=c.read_json(self.repo/'.cache'/c.NAME/'EXECUTION_RECEIPT.json')
        self.assertEqual(r['status'],'PUBLISH_REFUSED_INTEGRITY_FAILURE')
    def test_23_missing_data_does_not_consume_authorization(self):
        Path(self.args.wind_csv).unlink()
        with self.assertRaises(c.ContractError):self.go()
        self.assertFalse((self.repo/'.cache'/c.NAME/(c.AUTHORIZATION+'.started.json')).exists())
    def test_24_existing_staged_changes_are_not_committed(self):
        (self.repo/'unrelated.txt').write_text('staged unrelated');c.git(self.repo,'add','unrelated.txt')
        self.args.publish=False
        # Actual validate checks staged state before branch/source writes.
        remote='https://github.com/CanelE452/tsfm-peft-method-screen'
        c.git(self.repo,'remote','set-url','origin',remote)
        with self.assertRaises(self.gitops.ContractError):c.execute(self.pkg,self.args)
        self.assertEqual(len(self.calls),0)
        self.assertEqual(c.git(self.repo,'diff','--cached','--name-only').stdout.strip(),'unrelated.txt')

if __name__=='__main__':unittest.main(verbosity=2)
