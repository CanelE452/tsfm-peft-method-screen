from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--publish',action='store_true');a=ap.parse_args();root=Path(__file__).resolve().parent;mf=json.loads((root/'PACKAGE_MANIFEST.json').read_text())
    for rel,ex in mf['files'].items():
        if sha(root/rel)!=ex: raise SystemExit('package hash mismatch '+rel)
    repo=Path(a.repo).resolve();py=repo/'.venv'/'bin'/'python';
    if not py.exists(): raise SystemExit('repo .venv missing')
    t=subprocess.run([str(py),'-B',str(root/'run.py'),'--repo',str(repo),'--package-root',str(root),'--selftest'])
    if t.returncode: raise SystemExit(t.returncode)
    cmd=[str(py),'-B',str(root/'run.py'),'--repo',str(repo),'--package-root',str(root)]
    if a.publish:cmd.append('--publish')
    raise SystemExit(subprocess.run(cmd).returncode)
if __name__=='__main__':main()
