from pathlib import Path
import hashlib,json,os,time,contextlib
NAME='ab_path_ensemble_screen_v1_20260922'
ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments'/NAME
RESULTS=ROOT/'results'/NAME
CACHE=ROOT/'.cache'/NAME
SEEDS=[92301,92302]
ARMS_A=['FULL9','MEDIAN1','FIXED3','MEDOID3','GLOBAL3','CONTEXT3']
ARMS_B=['TARGET','CONTROL','MOMENTS','SET','MEMBER','SCENARIO3']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def default(v):
    if isinstance(v,Path):return str(v)
    if hasattr(v,'tolist'):return v.tolist()
    if hasattr(v,'item'):return v.item()
    raise TypeError(type(v).__name__)

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=default,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def event(kind,**kw):
    row=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),kind=kind,**kw)
    with (RESULTS/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,default=default)+'\n')
    print(json.dumps(row,default=default),flush=True)

def sources(candidate):
    files=list(EXP.glob('*.py'))+list((EXP/candidate).glob('*.py'))+list((EXP/'contract').glob('*'))
    return {str(p.relative_to(EXP)):sha(p) for p in sorted(files) if p.is_file()}

def verify_sources(receipt):
    for p,digest in receipt.items():assert sha(EXP/p)==digest, f'Sealed source changed: {p}'

@contextlib.contextmanager
def executor_lock():
    path=CACHE/'EXECUTOR.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:yield
    finally:path.unlink()
