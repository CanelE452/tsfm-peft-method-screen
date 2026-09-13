"""External-process-aware GPU monitor. Never terminates another job."""
import os,subprocess,time
from ..reproducibility import ROOT,write_json
class GPUWatch:
    def __init__(self,path):self.path=path;self.rows=[];self.last=0
    def read(self,phase):
        cmd=[str(ROOT/'scripts/with_cuda.sh'),'nvidia-smi']
        line=subprocess.check_output(cmd+['--query-gpu=memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]
        total,used,free,util=[float(x.strip()) for x in line.split(',')]
        raw=subprocess.check_output(cmd+['--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        pids=[int(v.strip()) for v in raw.splitlines() if v.strip().isdigit()]
        row=dict(time_unix=time.time(),phase=phase,total_mib=total,used_mib=used,free_mib=free,utilization_percent=util,compute_pids=pids,external_pids=[p for p in pids if p!=os.getpid()])
        self.rows.append(row);write_json(self.path,self.rows);self.last=time.monotonic();return row
    def wait_idle(self):
        idle=None
        while True:
            r=self.read('startup_wait')
            good=not r['external_pids'] and r['free_mib']>=4096 and r['utilization_percent']<=20
            if good:
                if idle is None:idle=time.monotonic()
                if time.monotonic()-idle>=30:return
            else:idle=None;print('GPU WAIT',r,flush=True)
            time.sleep(5)
    def check(self,phase,force=False):
        if not force and time.monotonic()-self.last<5:return
        r=self.read(phase)
        while r['external_pids'] or r['free_mib']<1024:
            print('GPU PAUSE at step boundary',r,flush=True)
            time.sleep(5);r=self.read('paused_'+phase)
