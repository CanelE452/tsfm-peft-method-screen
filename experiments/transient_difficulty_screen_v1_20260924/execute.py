from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--publish',action='store_true');a=ap.parse_args();root=Path(__file__).resolve().parent;mf=json.loads((root/'PACKAGE_MANIFEST.json').read_text())
 for rel,exp in mf['files'].items():
  if sha(root/rel)!=exp:raise SystemExit('Package hash mismatch '+rel)
 repo=Path(a.repo).resolve();py=repo/'.venv/bin/python';py=py if py.exists() else Path(sys.executable);env=os.environ.copy();env['PYTHONPATH']=str(root)+os.pathsep+env.get('PYTHONPATH','');t=subprocess.run([str(py),'-m','unittest','selftest','-v'],cwd=root,env=env)
 if t.returncode:raise SystemExit(t.returncode)
 cmd=[str(py),str(root/'difficulty_screen.py'),'--repo',str(repo),'--package-root',str(root)]+(['--publish'] if a.publish else []);raise SystemExit(subprocess.run(cmd,cwd=root,env=env).returncode)
if __name__=='__main__':main()
