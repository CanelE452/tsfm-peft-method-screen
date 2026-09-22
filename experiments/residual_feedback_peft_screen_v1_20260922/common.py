from pathlib import Path
import os,json,hashlib,time
NAME='residual_feedback_peft_screen_v1_20260922'
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'experiments'/NAME;RESULTS=ROOT/'results'/NAME;CACHE=ROOT/'.cache'/NAME
SEEDS=[92401,92402];STEPS=[0,128,512]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v),allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def append(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def event(kind,**kw):
    x=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),kind=kind,**kw);append(RESULTS/'events.jsonl',x);print(json.dumps(x),flush=True)
def source_files():return {p.relative_to(EXP).as_posix():sha(p) for p in EXP.rglob('*') if p.is_file() and p.suffix in ['.py','.txt','.md','.json'] and '__pycache__' not in p.parts and 'analysis' not in p.parts}
def sealcheck():
    for n,h in read(RESULTS/'SOURCE_SEAL.json')['files'].items():assert sha(EXP/n)==h,n
