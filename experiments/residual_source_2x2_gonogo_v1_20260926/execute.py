import argparse, hashlib, json, os, subprocess
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--publish',action='store_true');a=ap.parse_args()
root=Path(__file__).resolve().parent; mf=json.loads((root/'PACKAGE_MANIFEST.json').read_text())
for rel,ex in mf['files'].items():
    if sha(root/rel)!=ex: raise SystemExit('PACKAGE_HASH_MISMATCH '+rel)
repo=Path(a.repo).resolve(); py=repo/'.venv'/'bin'/'python'
if not py.exists(): raise SystemExit('repo .venv missing')
env=os.environ.copy();env['PYTHONPATH']=str(root)+os.pathsep+env.get('PYTHONPATH','')
t=subprocess.run([str(py),'-B',str(root/'selftest.py')],cwd=root,env=env)
if t.returncode: raise SystemExit(t.returncode)
cmd=[str(py),'-B',str(root/'run.py'),'--repo',str(repo),'--package-root',str(root)]
if a.publish:cmd.append('--publish')
raise SystemExit(subprocess.run(cmd,cwd=root,env=env).returncode)
