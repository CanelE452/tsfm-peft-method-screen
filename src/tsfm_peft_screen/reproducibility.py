import hashlib,json,os,random,resource,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
INITIAL_COMMIT=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('Committed_AS:')))*1024
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False

def source_hashes():return {str(p.relative_to(ROOT)):sha(p) for folder in ['src','configs'] for p in sorted((ROOT/folder).rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
def guard(start=None):
    import psutil
    if start is not None and time.monotonic()-start>7200:raise TimeoutError('GPU job timeout 7200s')
    if psutil.virtual_memory().available<2*2**30:raise MemoryError('RAM available below 2 GiB')
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    mode=Path('/proc/sys/vm/overcommit_memory').read_text().strip()
    if mode=='2' and mem['Committed_AS']>mem['CommitLimit']*.95:raise MemoryError('Strict commit charge exceeds 95%')
    if mode!='2' and mem['Committed_AS']-INITIAL_COMMIT>8*2**30:raise MemoryError('Heuristic-mode commit growth exceeds 8 GiB')
    if psutil.Process().memory_info().rss>6*2**30:raise MemoryError('Process RSS exceeds 6 GiB')
    if torch.cuda.is_available():
        if torch.cuda.memory_allocated()>8*2**30:raise MemoryError('GPU allocation exceeds 8 GiB')
        free,total=torch.cuda.mem_get_info()
        if free<512*2**20:raise MemoryError('GPU physical headroom below 512 MiB')
    return {'rss_bytes':psutil.Process().memory_info().rss,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'gpu_peak_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
