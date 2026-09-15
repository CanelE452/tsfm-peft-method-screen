"""Track-local execution guard and auditable state helpers, no automatic retries."""
import csv,fcntl,gc,hashlib,json,os,random,resource,subprocess,threading,time,copy
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import psutil
import torch
ROOT=Path(__file__).resolve().parents[2]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');q.replace(p)
def csvwrite(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def parameters(m):return {n:p for n,p in m.named_parameters() if p.requires_grad}
def cpu_state(m):return {n:p.detach().cpu().clone() for n,p in parameters(m).items()}
def tensor_hash(state):
    h=hashlib.sha256()
    for n,p in sorted(state.items()):h.update(n.encode());h.update(str((tuple(p.shape),p.dtype)).encode());h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def frozen_hash(m):return tensor_hash({n:p for n,p in m.named_parameters() if not p.requires_grad})
def restore(m,state):
    p=parameters(m);assert p.keys()==state.keys()
    with torch.no_grad():
        for n,v in p.items():v.copy_(state[n])
@contextmanager
def preserve_rng():
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:yield
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
def cleanup():gc.collect();torch.cuda.empty_cache()
class ResourceError(RuntimeError):pass
class Watch:
    def __init__(self,out,phase,wall_cap=28800):
        self.out=Path(out);self.phase=phase;self.wall_cap=wall_cap;self.lock=open(ROOT/'.cache/gpu.lock','a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.start=time.monotonic();self.wait=0.;self.events=[];self.latest=None;self.error=None;self.stop=threading.Event()
        p=self.out/'gpu_budget.json';self.budget=read(p) if p.exists() else dict(started_at=time.time(),wait_seconds=0.)
        save(p,self.budget);self.thread=threading.Thread(target=self.poll,daemon=True);self.thread.start()
    def sample(self):
        vals=[v.strip() for v in subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')]
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
        own={os.getpid()}|{p.pid for p in psutil.Process().children(recursive=True)};apps=[]
        for line in raw.splitlines():
            p,n,m=[v.strip() for v in line.split(',')];apps.append(dict(pid=int(p),name=n,memory_mib=int(m),own=int(p) in own))
        return dict(at=time.time(),gpu=vals[0],total_mib=int(vals[1]),used_mib=int(vals[2]),free_mib=int(vals[3]),utilization=int(vals[4]),apps=apps,busy=any(not a['own'] for a in apps) or int(vals[3])<1024)
    def poll(self):
        while not self.stop.is_set():
            try:
                r=self.sample();self.latest=r;self.events.append(r)
                with open(self.out/f'gpu_{self.phase}.jsonl','a') as f:f.write(json.dumps(r)+'\n')
            except BaseException as e:self.error=repr(e);return
            self.stop.wait(.5)
    def limits(self):
        if self.error:raise ResourceError('GPU_MONITOR_ERROR '+self.error)
        if time.time()-self.budget['started_at']>self.wall_cap:raise ResourceError('TRACK_WALL_CAP')
        if self.wait+self.budget['wait_seconds']>=600:raise ResourceError('BLOCKED_GPU_BUSY')
        if psutil.virtual_memory().available<2*2**30:raise ResourceError('RAM_BELOW_2GIB')
    def boundary(self,startup=False):
        stable=None
        while True:
            self.limits();r=self.latest;good=r and time.time()-r['at']<5 and not r['busy'] and r['free_mib']>=(4096 if startup else 1024)
            if good:
                if not startup:break
                if stable is None:stable=time.monotonic()
                if time.monotonic()-stable>=30:break
            else:stable=None
            tick=time.monotonic();self.stop.wait(.25);self.wait+=time.monotonic()-tick
    def before(self):self.boundary();return self.sample()
    def after(self,start):
        r=self.sample();self.limits()
        # C is fixed-update, so contamination is recorded then paused at next boundary.
        contaminated=any(v['busy'] for v in self.events if v['at']>=start['at']) or r['busy']
        return r,contaminated
    def close(self):
        self.stop.set();self.thread.join(timeout=10);self.budget['wait_seconds']+=self.wait;save(self.out/'gpu_budget.json',self.budget)
        save(self.out/f'{self.phase}_wall.json',dict(seconds=time.monotonic()-self.start,wait_seconds=self.wait,minimum_free_mib=min((r['free_mib'] for r in self.events),default=None),external_compute_samples=sum(any(not a['own'] for a in r['apps']) for r in self.events),peak_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()
