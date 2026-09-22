from pathlib import Path
import os,json,hashlib,time
NAME='temporal_lora_init_v1_20260922'
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'experiments'/NAME;RESULTS=ROOT/'results'/NAME;CACHE=ROOT/'.cache'/NAME
SOURCES=['Electricity','ETTh1'];ARMS=['RANDOM','PCA','TEMP','SHUFFLE'];SEEDS=[92341,92342];STEPS=[0,32,128,256,512]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v),allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def event(kind,**kw):
    x=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),kind=kind,**kw);RESULTS.mkdir(parents=True,exist_ok=True)
    with (RESULTS/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(x)+'\n')
    print(json.dumps(x),flush=True)
