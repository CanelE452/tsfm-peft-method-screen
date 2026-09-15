"""Fixed resource screen and conditional 12-fit query pilot; one worker, no retry."""
import argparse,csv,fcntl,gc,json,os,subprocess,sys,threading,time,traceback,resource
from pathlib import Path
from contextlib import nullcontext,contextmanager
import numpy as np
import psutil
import torch
import run_forecast_query_checkpoint_diagnostic as S
from tsfm_peft_screen.forecast_query.budget import BudgetModel,block_indices,micro_origins,batch,choose_option,TrainingClock,seal_valid
from priority12.numerics import check_parity,path_check,difference
from tsfm_peft_screen.forecast_query.equal_time import preserve_rng
from tsfm_peft_screen.backbone import MODEL_ID,REVISION,native_loss
from tsfm_peft_screen.metrics import score
from tsfm_peft_screen.reproducibility import ROOT,sha,digest,write_json,seed_all,guard

RUN='query_budget_numeric_v2_20260915'
OUT=ROOT/'results'/RUN;CACHE=ROOT/'.cache'/RUN
CONFIG=ROOT/'configs/query_budget_numeric_v2_20260915.json'
DOC=ROOT/'docs/PEFT_PRIORITY12_PROTOCOL_20260915.md'
DATASETS=['ettm2','electricity'];ARMS=['standard','side','query'];SEEDS=[39000,39001]
PAIRS=[[17408,18432],[19456,20480],[17408,18432]]
CP={'standard':[0,3,6,9,12],'side':[0,12],'query':[0,12]}
LR={'standard':1e-4,'side':1e-3,'query':1e-4}
REQUIRED=['docs/FORECAST_QUERY_EQUAL_TIME_PLAN.md','docs/FORECAST_QUERY_EQUAL_TIME_EXECUTION.md','configs/forecast_query_equal_time_plan.json','research/reopen_review_20260914/QUERY_RESOURCE_FRONTIER.md','src/tsfm_peft_screen/forecast_query/model.py','src/tsfm_peft_screen/forecast_query/equal_time.py','src/tsfm_peft_screen/forecast_query/checkpoint_diagnostic.py','scripts/run_forecast_query_equal_time.py','scripts/run_reopen_query_resources.py','src/tsfm_peft_screen/backbone.py','src/tsfm_peft_screen/lora.py','src/tsfm_peft_screen/metrics.py','docs/RESULTS_INDEX.md']

def read(p):return json.loads(Path(p).read_text())
def save(name,obj):write_json(OUT/name,obj)
def rel(p):return str(Path(p).relative_to(ROOT))
def csvwrite(path,rows,fields=None):
    keys=fields or list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def options():return [dict(dataset=d,arm=a,cp=k,micro=b,id=f'{d}_{a}_cp{k}_mb{b}') for d in DATASETS for a in ARMS for k in CP[a] for b in [1,2]]
def origins(split):return {'train':list(range(16384,22529,32)),'validation':list(range(23040,23713,96)),'evaluation':list(range(24576,26017,96))}[split]
def sources():
    paths=list((ROOT/'src').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'scripts/finalize_query_numeric_v2.py',ROOT/'scripts/run_forecast_query_checkpoint_diagnostic.py',ROOT/'tests/test_query_budget_pilot.py',ROOT/'tests/test_priority12_numerics.py',ROOT/'scripts/priority12/numerics.py',CONFIG,DOC]
    return {rel(p):sha(p) for p in sorted(paths)}
def check_contract():
    c=read(OUT/'contract.json');assert all(sha(ROOT/p)==h for p,h in c['sources'].items()),'Sources/config changed after contract seal'
    for d in c['data'].values():
        for p,h in d['staged'].items():assert sha(ROOT/p)==h
    return c

def prepare():
    assert not OUT.exists() and not CACHE.exists(),'Existing run, no overwrite or retry'
    OUT.mkdir();CACHE.mkdir();cfg=read(CONFIG)
    r=subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=ROOT,text=True,capture_output=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    save('cpu_tests.json',dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stdout+r.stderr
    historical=read(ROOT/'results/priority12_20260915/historical_hashes.json')
    data={}
    previous=read(ROOT/'results/forecast_query_equal_time/contract.json')
    for name in DATASETS:
        folder=ROOT/'data/processed'/name;meta=read(folder/'manifest.json')
        assert meta['files']==previous['data'][name]['source_hashes']
        for f,h in meta['files'].items():assert sha(folder/f)==h
        for f,h in meta['sources'].items():assert sha(f)==h
        with np.load(folder/'fit.npz') as z:prefix=z['values'][:,:4];channels=z['channels'][:4].tolist()
        with np.load(folder/'evaluation.npz') as z:values=np.concatenate([prefix,z['tail'][:,:4]])
        assert channels==meta['channels'][:4] and values.shape[0]>=26064 and values.shape[1]==4
        scale=np.maximum(np.nanstd(values[16384:22576],axis=0,dtype=np.float64),1e-6)
        assert np.isfinite(values[:26064]).all() and np.array_equal(scale,np.array(previous['data'][name]['scale']))
        for split,end in [('train',22576),('development',23760),('evaluation',26064)]:
            np.savez_compressed(CACHE/f'{name}_{split}.npz',values=values[:end],scale=scale)
        data[name]=dict(channels=channels,raw_sources=meta['sources'],processed_sources={rel(folder/f):h for f,h in meta['files'].items()},
            manifest_sha256=sha(folder/'manifest.json'),scale=scale.tolist(),staged={rel(p):sha(p) for p in CACHE.glob(name+'_*.npz')},
            staging_scope='Mechanically read existing raw-derived arrays including prior E; separate Train/Train+V/E files, no model or E scoring at prepare',E_scope='previously evaluated development targets reused')
        del values,prefix
    from huggingface_hub import hf_hub_download
    model={}
    for name,h in read(ROOT/'results/screening_summary/common_integrity.json')['model_files'].items():
        p=hf_hub_download(MODEL_ID,name,revision=REVISION,local_files_only=True);assert sha(p)==h;model[p]=h
    ops=options();orders={str(b):np.random.default_rng(59000+b).permutation([v['id'] for v in ops]).tolist() for b in range(3)}
    cells=[dict(dataset=d,seed=s,arm=a,lr=LR[a]) for d in DATASETS for s in SEEDS for a in ARMS]
    order=np.random.default_rng(59010).permutation(12);cells=[dict(attempt_order=i,**cells[int(v)]) for i,v in enumerate(order)]
    schedules={f'{d}_{s}':np.random.default_rng(s).choice(origins('train'),size=(8192,2)).tolist() for d in DATASETS for s in SEEDS}
    save('schedules.json',schedules);save('fit_manifest.json',cells);save('profile_order.json',orders)
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('contract.json',dict(base_commit='9051d03cebe23ae35d9689e7af576c63c4d5edd5',prepare_commit=current,
        baseline_delta=subprocess.check_output(['git','diff','6a8bf6224a94fb790d26a96a2b9cb7505f24b70d','--']+REQUIRED,cwd=ROOT,text=True),
        sources=sources(),required_read_hashes={p:sha(ROOT/p) for p in REQUIRED},config=cfg,data=data,model_files=model,
        historical_hashes=historical,profile_order_hash=sha(OUT/'profile_order.json'),fit_manifest_hash=sha(OUT/'fit_manifest.json'),schedule_hash=sha(OUT/'schedules.json'),
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,checkpoint_use_reentrant=False),
        isolation='One sequential worker; each option constructed in a bounded function, all model/optimizer tensors released, gc and empty_cache between options only; no cross-window cache',
        A_update_allocation=dict(state_warmup=12,cold_optimizer_creation=36,kernel_warmup=36,fp32_checks=36,bf16_timed=324,total=444,absolute_cap=480),
        timing='Complete update; CPU state/gradient export and validation/I/O excluded using synchronized timing segments. Same segments for every A option. B complete updates include batch transfer, frozen encoder, checks, clip and step.'))
    (OUT/'PROTOCOL.md').write_text(DOC.read_text())
    save('status.json',dict(status='PREPARED',numeric_update_attempts=0,numeric_updates=0,A_update_attempts=0,A_updates=0,B_fit_attempts=0,B_fits_completed=0,B_updates=0))
    for n in ['resource_measurements.csv','resource_budget_table.csv','fit_attempts.csv','trajectories.csv','metrics.csv','resources.csv']:csvwrite(OUT/n,[])
    print('PREPARED: contracts and CPU tests verified; no GPU model execution',flush=True)


class ResourceError(RuntimeError):pass
class NumericalError(RuntimeError):pass


class Watch:
    def __init__(self,phase):
        self.phase=phase;self.lock=open(ROOT/'.cache/gpu.lock','a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.started=time.monotonic();self.wait=0.;self.stop_event=threading.Event();self.latest=None;self.error=None;self.events=[];self.guard_start=time.monotonic()
        self.budget=read(OUT/'gpu_budget.json') if (OUT/'gpu_budget.json').exists() else dict(started_at=time.time(),wait_seconds=0.)
        save('gpu_budget.json',self.budget);self.mutex=threading.Lock();self.thread=threading.Thread(target=self.poll,daemon=True);self.thread.start()
    def read_gpu(self):
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=4).strip().split(',')
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True,timeout=4)
        own={os.getpid()}|{p.pid for p in psutil.Process().children(recursive=True)};apps=[]
        for line in raw.splitlines():
            p,n,m=[v.strip() for v in line.split(',')];apps.append(dict(pid=int(p),name=n,memory_mib=int(m),own=int(p) in own))
        return dict(at=time.time(),gpu=gpu[0],total_mib=int(gpu[1]),used_mib=int(gpu[2]),free_mib=int(gpu[3]),utilization=int(gpu[4]),apps=apps,
            busy=any(not a['own'] for a in apps) or int(gpu[3])<1024)
    def poll(self):
        while not self.stop_event.is_set():
            try:
                r=self.read_gpu()
                with self.mutex:self.latest=r;self.events.append(r)
                with open(OUT/f'gpu_{self.phase}.jsonl','a') as f:f.write(json.dumps(r)+'\n')
            except BaseException as e:self.error=repr(e);return
            self.stop_event.wait(.5)
    def limits(self):
        if self.error:raise ResourceError('GPU_MONITOR_ERROR '+self.error)
        if time.time()-self.budget['started_at']>14400:raise ResourceError('GPU_WALL_CAP_4H')
        if self.budget['wait_seconds']+self.wait>=600:raise ResourceError('BLOCKED_GPU_BUSY: cumulative 600s wait cap')
        if psutil.virtual_memory().available<2*2**30:raise ResourceError('RAM_BELOW_2_GIB')
    def boundary(self,startup=False):
        stable=None
        while True:
            self.limits();r=self.latest
            good=r and time.time()-r['at']<5 and not r['busy'] and r['free_mib']>=(4096 if startup else 1024)
            if good:
                if not startup:break
                if stable is None:stable=time.monotonic()
                if time.monotonic()-stable>=30:break
            else:stable=None
            tick=time.monotonic();time.sleep(.25);self.wait+=time.monotonic()-tick
        guard()
    def before(self):
        self.boundary();r=self.read_gpu()
        if r['busy']:
            self.latest=r;self.boundary();r=self.read_gpu()
        self.limits();return r
    def after(self,start):
        r=self.read_gpu()
        with self.mutex:bad=[v for v in self.events if v['at']>=start['at'] and v['busy']]
        if bad or r['busy']:
            save('timing_contamination.json',dict(start=start,end=r,observed_during_update=bad))
            raise ResourceError('TIMING_CONTAMINATED: external compute or physical memory pressure')
        self.limits();return r
    def close(self):
        self.stop_event.set();self.thread.join(timeout=10)
        self.budget['wait_seconds']+=self.wait;save('gpu_budget.json',self.budget)
        save(f'{self.phase}_wall.json',dict(seconds=time.monotonic()-self.started,gpu_wait_seconds=self.wait,peak_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            minimum_free_mib=min((r['free_mib'] for r in self.events),default=None),external_compute_samples=sum(any(not a['own'] for a in r['apps']) for r in self.events),sampling_interval_seconds=.5))
        fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()


class Trace:
    def __init__(self):self.rows=[];self.peak=0;self.reserved=0
    def cache(self,c):
        tensors=c.get('k',[])+c.get('v',[])+[v for v in c.values() if isinstance(v,torch.Tensor)]
        self.cache_unique_bytes=sum({v.untyped_storage()._cdata:v.untyped_storage().nbytes() for v in tensors}.values())
    def capture(self):
        self.peak=max(self.peak,torch.cuda.max_memory_allocated());self.reserved=max(self.reserved,torch.cuda.max_memory_reserved())
    @contextmanager
    def phase(self,name):
        torch.cuda.synchronize();self.capture();torch.cuda.reset_peak_memory_stats()
        start=torch.cuda.memory_allocated()
        try:yield
        finally:
            torch.cuda.synchronize();self.capture();self.rows.append(dict(phase=name,start_allocated=start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))


def load_data(name,split):
    with np.load(CACHE/f'{name}_{split}.npz') as z:v=z['values'];s=z['scale']
    assert len(v)=={'train':22576,'development':23760,'evaluation':26064}[split]
    return v,s

def make_model(arm,seed,cp):
    seed_all(seed);m=BudgetModel(arm,seed,cp);opt=torch.optim.AdamW(S.params(m).values(),lr=LR[arm],weight_decay=0,betas=(.9,.999),eps=1e-8)
    ps=S.params(m);assert sum(p.numel() for p in ps.values())==1179648
    assert all(p.dtype==torch.float32 for p in m.parameters()) and not any(p.requires_grad for p in m.base.output_patch_embedding.parameters())
    assert all(('lora_' in n if arm!='side' else n.startswith('adapter.')) for n in ps)
    return m,opt

def frozen_hash(m):return S.tree_hash({n:p for n,p in m.named_parameters() if not p.requires_grad})
def restore_parameters(m,state):
    ps=S.params(m);assert ps.keys()==state.keys()
    with torch.no_grad():
        for n,p in ps.items():p.copy_(state[n])
def cleanup():gc.collect();torch.cuda.empty_cache()


def step(m,opt,values,oo,micro,w,precision='bf16',export=False,trace=False,limit=None,on_applied=None):
    """One effective update. Export time is excluded, all optimizer work is included."""
    ps=S.params(m);before=S.cpu_tree(ps) if export else None;trace_obj=Trace() if trace else None;m.observer=trace_obj
    start_gpu=w.before();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start_alloc=torch.cuda.memory_allocated();start_reserved=torch.cuda.memory_reserved();tick=time.perf_counter();active=0.;export_seconds=0.
    outputs=[];losses=[];residuals=[]
    opt.zero_grad(set_to_none=True)
    for pair,weight in micro_origins(oo,micro):
        with trace_obj.phase('batch_and_transfer') if trace_obj else nullcontext():x,y,g=batch(values,pair)
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=True) if precision=='bf16' else nullcontext():
            z,p,loc,scale=m(x,g);loss=native_loss(z,y,loc,scale)*weight
        if not torch.isfinite(loss) or not torch.isfinite(p).all():raise FloatingPointError('Nonfinite loss or prediction')
        with trace_obj.phase('backward') if trace_obj else nullcontext():loss.backward()
        outputs.append((z.detach(),p.detach()));losses.append(loss.detach())
        if export:residuals.append((((y-loc)/scale).asinh()[:,None,:]-z).detach().cpu())
        del x,y,g,z,p,loc,scale,loss
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps.values())
    if export:
        torch.cuda.synchronize();active+=time.perf_counter()-tick;ex=time.perf_counter()
        rawgrad=S.flatten({n:p.grad.detach().cpu().clone() for n,p in ps.items()})
        export_seconds+=time.perf_counter()-ex;tick=time.perf_counter()
    with trace_obj.phase('clip_and_optimizer') if trace_obj else nullcontext():
        norm=torch.nn.utils.clip_grad_norm_(list(ps.values()),1.,error_if_nonfinite=True);opt.step()
        if on_applied:on_applied()
    assert all(torch.isfinite(p).all() for p in ps.values())
    torch.cuda.synchronize();active+=time.perf_counter()-tick
    peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    if trace_obj:trace_obj.capture();peak=max(peak,trace_obj.peak);reserved=max(reserved,trace_obj.reserved)
    end_gpu=w.after(start_gpu);m.observer=None
    resource=dict(seconds=active,start_allocated=start_alloc,start_reserved=start_reserved,peak_allocated=peak,peak_reserved=reserved,
        loss=float(sum(losses)),gradient_norm=float(norm),nvml_start_used_mib=start_gpu['used_mib'],nvml_end_used_mib=end_gpu['used_mib'],
        physical_free_mib=min(start_gpu['free_mib'],end_gpu['free_mib']),process_nvml_mib=max((a['memory_mib'] for a in end_gpu['apps'] if a['pid']==os.getpid()),default=None),
        phase_peaks=trace_obj.rows if trace_obj else [],precision=precision,cpu_export_seconds=export_seconds)
    artifact=None
    if export:
        ex=time.perf_counter();now=S.cpu_tree(ps)
        artifact=dict(z=torch.cat([a.cpu() for a,b in outputs]),raw=torch.cat([b.cpu() for a,b in outputs]),loss=sum(losses).cpu().reshape(1),raw_gradient=rawgrad,
            clipped_gradient=S.flatten({n:p.grad.detach().cpu().clone() for n,p in ps.items()}),update=S.flatten({n:now[n]-before[n] for n in now}),
            adam=torch.cat([v.detach().cpu().reshape(-1) for _,s in sorted(opt.state_dict()['state'].items()) for _,v in sorted(s.items()) if isinstance(v,torch.Tensor)]),rng=S.tree_hash(S.rng()))
        artifact['tensor_layout']=[(n,p.numel()) for n,p in sorted(ps.items())]
        ids={i:n for i,(n,p) in zip(opt.state_dict()['param_groups'][0]['params'],ps.items())}
        artifact['adam_layout']=[(ids[i]+'.'+k,v.numel()) for i,st in sorted(opt.state_dict()['state'].items()) for k,v in sorted(st.items()) if isinstance(v,torch.Tensor)]
        artifact['residual']=torch.cat(residuals)
        assert float(artifact['update'].abs().sum())>0
        resource['cpu_export_seconds']+=time.perf_counter()-ex
    if limit is not None and peak>limit:raise ResourceError('MEMORY_BUDGET_VIOLATION')
    return resource,artifact


def profile():
    state=read(OUT/'status.json');assert state['status']=='NUMERICS_BOUNDED','Numeric gate required; no A retry'
    c=check_contract();w=Watch('A');records=[];parity=[];initial=[];warmfiles={};artifacts={};memory=[]
    def update(m,opt,v,oo,micro,tag,**kw):
        assert state['A_update_attempts']<512
        state['A_update_attempts']+=1;save('status.json',state)
        def applied():state['A_updates']+=1
        r,a=step(m,opt,v,oo,micro,w,on_applied=applied,**kw)
        row=dict(tag=tag,**r);memory.append(row);save('A_memory_records.json',memory);save('status.json',state)
        return r,a
    def finish_tables():
        csvwrite(OUT/'resource_measurements.csv',[{k:v for k,v in r.items() if k!='phase_peaks'} for r in records]);save('parity.json',parity)
        save('artifact_index.json',artifacts);save('initial_F0_parity.json',initial)
    try:
        state['status']='A_WAITING_FOR_GPU';save('status.json',state);w.boundary(startup=True)
        state['status']='A_RUNNING';save('status.json',state)
        for d in DATASETS:
            values,scale=load_data(d,'train')
            for arm in ARMS:
                m,opt=make_model(arm,39000,0);frozen=frozen_hash(m)
                # Read-only initial-path comparison, both precisions, no optimizer counts.
                x,y,g=batch(values,PAIRS[0])
                for precision in ['fp32','bf16']:
                    w.boundary()
                    with preserve_rng(),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=True) if precision=='bf16' else preserve_rng():
                        if precision=='fp32':
                            with torch.no_grad():a=m(x,g);b=m.f0(x,g)
                        else:a=m(x,g);b=m.f0(x,g)
                    initial.append(dict(dataset=d,arm=arm,precision=precision,z=S.delta(b[0].cpu(),a[0].cpu()),raw=S.delta(b[1].cpu(),a[1].cpu()),trainable_parameters=sum(p.numel() for p in S.params(m).values())))
                    del a,b
                del x,y,g
                for j in range(2):update(m,opt,values,PAIRS[j],2,f'warm_{d}_{arm}_{j}')
                assert frozen_hash(m)==frozen
                path=CACHE/f'warm_{d}_{arm}.pt';torch.save(S.snapshot(m,opt),path);warmfiles[f'{d}_{arm}']=dict(path=rel(path),sha256=sha(path),frozen=frozen)
                del m,opt;cleanup()
        save('warm_states.json',warmfiles)
        # FP32 checks use one reference per option (36 total); all CP/micro pairs compare the same restored state.
        for o in options():
            values,scale=load_data(o['dataset'],'train');shared=torch.load(ROOT/warmfiles[f"{o['dataset']}_{o['arm']}"]['path'],weights_only=False,map_location='cpu')
            torch.cuda.reset_peak_memory_stats();m,opt=make_model(o['arm'],39000,o['cp']);frozen=frozen_hash(m);load_peak=torch.cuda.max_memory_allocated()
            restore_parameters(m,shared['parameters'])
            cold,_=update(m,opt,values,PAIRS[0],o['micro'],o['id']+'_cold',trace=True)
            S.restore(m,opt,shared)
            kernel,_=update(m,opt,values,PAIRS[0],o['micro'],o['id']+'_kernel',trace=True)
            S.restore(m,opt,shared)
            fp,a=update(m,opt,values,PAIRS[0],o['micro'],o['id']+'_fp32',precision='fp32',export=True)
            path=CACHE/(o['id']+'_fp32.pt');torch.save(a,path);artifacts[o['id']+'_fp32']=dict(path=rel(path),sha256=sha(path))
            # Train-only evaluation origin1 for every option; no V/E arrays loaded.
            S.restore(m,opt,shared);torch.cuda.reset_peak_memory_stats();x,y,g=batch(values,[PAIRS[0][0]])
            with preserve_rng(),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=True):z,p,loc,sc=m(x,g)
            eval_peak=torch.cuda.max_memory_allocated();assert torch.isfinite(p).all()
            save(o['id']+'_setup.json',dict(option=o,cold=cold,kernel=kernel,fp32=fp,evaluation_peak=eval_peak,load_peak=load_peak,initial_state_sha256=S.tree_hash(shared)))
            assert frozen_hash(m)==frozen
            del m,opt,a,shared,x,y,g,z,p,loc,sc;cleanup()
            print('A SETUP',o['id'],flush=True)
        for o in options():
            a=torch.load(ROOT/artifacts[o['id']+'_fp32']['path'],weights_only=False)
            refid=f"{o['dataset']}_{o['arm']}_cp0_mb{o['micro']}_fp32"
            b=torch.load(ROOT/artifacts[refid]['path'],weights_only=False)
            p=check_parity(b,a,'fp32',scale=load_data(o['dataset'],'train')[1]);parity.append(dict(option=o['id'],reference=refid,kind='checkpoint',update=0,**p))
            if o['micro']==1 and o['cp']==0:
                refid=f"{o['dataset']}_{o['arm']}_cp0_mb2_fp32";b=torch.load(ROOT/artifacts[refid]['path'],weights_only=False)
                p=check_parity(b,a,'fp32',micro=True,scale=load_data(o['dataset'],'train')[1]);parity.append(dict(option=o['id'],reference=refid,kind='microbatch',update=0,**p))
            del a,b
        finish_tables()
        if not all(p['passed'] for p in parity):raise NumericalError('FP32_CP_OR_MICROBATCH_EQUIVALENCE_FAILED')
        orders=read(OUT/'profile_order.json');byid={o['id']:o for o in options()}
        for block in range(3):
            for oid in orders[str(block)]:
                o=byid[oid];values,scale=load_data(o['dataset'],'train');shared=torch.load(ROOT/warmfiles[f"{o['dataset']}_{o['arm']}"]['path'],weights_only=False,map_location='cpu')
                m,opt=make_model(o['arm'],39000,o['cp']);frozen=frozen_hash(m);restore_tick=time.monotonic();S.restore(m,opt,shared);restore_seconds=time.monotonic()-restore_tick
                assert S.tree_hash(S.snapshot(m,opt))==S.tree_hash(shared)
                for j,oo in enumerate(PAIRS):
                    r,a=update(m,opt,values,oo,o['micro'],f'{oid}_b{block+1}_u{j}',export=True)
                    records.append(dict(**o,block=block+1,update=j,restore_seconds=restore_seconds,**r))
                    if block==0:
                        path=CACHE/f'{oid}_bf16_{j}.pt';torch.save(a,path);artifacts[f'{oid}_bf16_{j}']=dict(path=rel(path),sha256=sha(path))
                    del a
                assert frozen_hash(m)==frozen
                del m,opt,shared;cleanup();finish_tables()
                print('A BLOCK',block+1,oid,flush=True)
            if block==0:
                for o in options():
                    _,scale=load_data(o['dataset'],'train')
                    for j in range(3):
                        a=torch.load(ROOT/artifacts[f"{o['id']}_bf16_{j}"]['path'],weights_only=False)
                        refid=f"{o['dataset']}_{o['arm']}_cp0_mb{o['micro']}_bf16_{j}";b=torch.load(ROOT/artifacts[refid]['path'],weights_only=False)
                        p=check_parity(b,a,'bf16',scale=scale);parity.append(dict(option=o['id'],reference=refid,kind='checkpoint',update=j,**p))
                        if o['cp']==0 and o['micro']==1:
                            refid=f"{o['dataset']}_{o['arm']}_cp0_mb2_bf16_{j}";b=torch.load(ROOT/artifacts[refid]['path'],weights_only=False)
                            p=check_parity(b,a,'bf16',micro=True,scale=scale);parity.append(dict(option=o['id'],reference=refid,kind='microbatch',update=j,**p))
                        del a,b
                finish_tables()
                if not all(p['passed'] for p in parity):raise NumericalError('BF16_CP_OR_MICROBATCH_EQUIVALENCE_FAILED')
                # Fix B* and options using block1 only. Later blocks may invalidate safety, never reselect.
                entries=aggregate_options(records);selection=select_resources(entries);save('resource_selection_block1.json',selection)
        entries=aggregate_options(records);selection=read(OUT/'resource_selection_block1.json');selected=selection.get('choices',[])
        if selection['status']!='CANDIDATE_BUDGET':
            selection['stop_reason']=selection['status'];state['status']='STOP_RESOURCE_SCREEN'
        else:
            budget=selection['budget_bytes']
            for choice in selected:
                latest=next(r for r in entries if r['id']==choice['id'])
                if latest['peak_allocated']>.95*budget:raise ResourceError('LATE_PROFILE_MEMORY_VARIATION: no reselection')
            gains=[]
            for d in DATASETS:
                st=next(r for r in selected if r['dataset']==d and r['arm']=='standard');qu=next(r for r in selected if r['dataset']==d and r['arm']=='query')
                vals=[]
                for b in [2,3]:
                    ts=float(np.median([r['seconds'] for r in records if r['id']==st['id'] and r['block']==b]));tq=float(np.median([r['seconds'] for r in records if r['id']==qu['id'] and r['block']==b]));vals.append(100*(ts-tq)/ts)
                gains.append(dict(dataset=d,block2_gain_percent=vals[0],block3_gain_percent=vals[1]))
            selection['speed_validation']=gains
            status='RESOURCE_SIGNAL' if any(min(g['block2_gain_percent'],g['block3_gain_percent'])>=5 for g in gains) else 'STOP_RESOURCE_SCREEN'
            if max(r['seconds'] for r in records if r['id'] in {v['id'] for v in selected})>1:status='INVALID_TIMING_PREFLIGHT'
            state['status']=status
        selection.update(status=state['status'],all_options=entries,measurements_hash=sha(OUT/'resource_measurements.csv'),parity_hash=sha(OUT/'parity.json'),sealed_at=time.time())
        selection['seal_hash']=digest(selection);save('resource_selection.json',selection)
        csvwrite(OUT/'resource_budget_table.csv',budget_rows(entries));assert state['A_updates']==444
    except BaseException as e:
        state.update(status='INCONCLUSIVE_NUMERICS' if isinstance(e,NumericalError) else 'BLOCKED_GPU_BUSY' if 'BLOCKED_GPU_BUSY' in str(e) else 'INCONCLUSIVE_EXECUTION',error=repr(e),traceback=traceback.format_exc())
        finish_tables();save('resource_selection.json',dict(status=state['status'],reason=repr(e),B_fits=0,no_options_excluded_to_favor_Query=True))
        csvwrite(OUT/'resource_budget_table.csv',[dict(budget_gib=b,status='NOT_IDENTIFIED_DUE_TO_EXECUTION_OR_INTEGRITY',Query_advantage='UNDETERMINED') for b in [1,2,4,8]])
    finally:
        save('status.json',state);w.close();from finalize_query_numeric_v2 import finalize;finalize()


def aggregate_options(records):
    entries=[]
    for o in options():
        setup=read(OUT/(o['id']+'_setup.json'));rr=[r for r in records if r['id']==o['id']]
        peak=max([setup['load_peak'],setup['cold']['peak_allocated'],setup['kernel']['peak_allocated'],setup['evaluation_peak']]+[r['peak_allocated'] for r in rr])
        entries.append(dict(**o,valid=True,peak_allocated=peak,evaluation_peak=setup['evaluation_peak'],block1_seconds=float(np.median([r['seconds'] for r in rr if r['block']==1])),
            block_indices=block_indices(o['cp']) if o['arm']=='standard' else list(range(12)) if o['cp'] else []))
    return entries

def budget_rows(entries):return [dict(budget_gib=b,**r,feasible=r['valid'] and r['peak_allocated']<=.95*b*2**30) for b in [1,2,4,8] for r in entries]

def select_resources(entries):
    for b in [1,2,4,8]:
        choices=[choose_option([r for r in entries if r['dataset']==d and r['arm']==a],b*2**30) for d in DATASETS for a in ARMS]
        if all(choices):break
    else:return dict(status='NO_COMMON_FEASIBLE_BUDGET',choices=[])
    binding=[]
    for d in DATASETS:
        unrestricted=choose_option([r for r in entries if r['dataset']==d and r['arm']=='standard'],float('inf'))
        binding.append(dict(dataset=d,unrestricted=unrestricted['id'],infeasible=unrestricted['peak_allocated']>.95*b*2**30))
    status='CANDIDATE_BUDGET' if any(r['infeasible'] for r in binding) else 'NO_BINDING_MEMORY_CONSTRAINT'
    if any(r['evaluation_peak']>b*2**30 for r in entries):status='EVALUATION_MEMORY_LIMIT'
    return dict(status=status,budget_gib=b,budget_bytes=b*2**30,choices=choices,binding=binding,selection_rule='Block1 only; choices fixed before blocks2/3',sealed_at=time.time())


def evaluate(m,values,scale,oo,tag,w,limit,f0=False):
    pp=[];yy=[];peaks=[];start=time.monotonic()
    with preserve_rng(),torch.no_grad():
        for o in oo:
            w.boundary();torch.cuda.reset_peak_memory_stats();x,y,g=batch(values,[o])
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=True):z,p,loc,sc=m.f0(x,g) if f0 else m(x,g)
            torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();peaks.append(peak)
            if peak>limit:raise ResourceError('EVALUATION_MEMORY_BUDGET_VIOLATION')
            pp.append(p.cpu().numpy());yy.append(y.cpu().numpy())
    pred=np.stack(pp);target=np.stack(yy);path=CACHE/(tag+'.npz');assert not path.exists(),'No prediction overwrite'
    np.savez_compressed(path,prediction=pred,target=target,scale=scale,origins=np.array(oo))
    return dict(metrics=score(pred,target,scale),prediction_file=rel(path),prediction_hash=sha(path),evaluation_seconds=time.monotonic()-start,evaluation_peak=max(peaks))


def train_evaluate():
    state=read(OUT/'status.json');assert state['status']=='RESOURCE_SIGNAL' and state['B_fit_attempts']==0,'B forbidden or already attempted'
    c=check_contract();selection=read(OUT/'resource_selection.json');assert digest({k:v for k,v in selection.items() if k!='seal_hash'})==selection['seal_hash']
    w=Watch('B');fits=[];trajectory=[];evaluation=[];limit=selection['budget_bytes'];schedules=read(OUT/'schedules.json')
    try:
        w.boundary(startup=True)
        for cell in read(OUT/'fit_manifest.json'):
            assert state['B_fit_attempts']<12;state['B_fit_attempts']+=1
            d,a,seed=cell['dataset'],cell['arm'],cell['seed'];fid=f"{cell['attempt_order']:02}_{d}_{seed}_{a}";choice=next(r for r in selection['choices'] if r['dataset']==d and r['arm']==a)
            fit=dict(fit=fid,**cell,status='RUNNING',updates=0,choice=choice);fits.append(fit);save('fits.json',fits)
            state.update(status='B_TRAINING',current_fit=fid);save('status.json',state);started=time.monotonic();records=[];fr=[]
            m=opt=None
            try:
                w.boundary();torch.cuda.reset_peak_memory_stats();m,opt=make_model(a,seed,choice['cp']);loading_peak=torch.cuda.max_memory_allocated();frozen=frozen_hash(m);initial=S.tree_hash(S.cpu_tree(S.params(m)))
                values,scale=load_data(d,'development');clock=TrainingClock((30.,60.,120.),1.,8192);best=None
                def checkpoint_model(nominal):
                    nonlocal best
                    with preserve_rng():
                        r=evaluate(m,values,scale,origins('validation'),f'V_{fid}_{int(nominal)}',w,limit)
                        path=CACHE/f'{fid}_{int(nominal)}s.pt';tick=time.monotonic();torch.save(S.cpu_tree(S.params(m)),path)
                        row=dict(fit=fid,dataset=d,seed=seed,arm=a,lr=cell['lr'],budget_bytes=limit,cp=choice['cp'],micro=choice['micro'],nominal_seconds=nominal,active_seconds=clock.elapsed,updates=clock.updates,
                            checkpoint_file=rel(path),checkpoint_hash=sha(path),checkpoint_seconds=time.monotonic()-tick,**r)
                    trajectory.append(row);fr.append(row);save('trajectories.json',trajectory)
                    if best is None or (r['metrics']['scaled_2pinball'],nominal)<(best['metrics']['scaled_2pinball'],best['nominal_seconds']):best=row
                    print('B',fid,nominal,clock.elapsed,clock.updates,r['metrics']['scaled_2pinball'],flush=True)
                if loading_peak>limit:raise ResourceError('MODEL_LOADING_MEMORY_BUDGET_VIOLATION')
                checkpoint_model(0)
                while not clock.complete:
                    def applied():fit['updates']+=1;state['B_updates']+=1
                    r,_=step(m,opt,values,schedules[f'{d}_{seed}'][clock.updates],choice['micro'],w,limit=limit,on_applied=applied)
                    records.append(r)
                    boundary=clock.add(r['seconds'])
                    if boundary is not None:checkpoint_model(boundary);save('status.json',state);save('fits.json',fits)
                assert frozen_hash(m)==frozen and S.tree_hash(S.cpu_tree(S.params(m)))!=initial
                fit.update(status='COMPLETE',active_seconds=clock.elapsed,wall_seconds=time.monotonic()-started,best=best,
                    loading_peak=loading_peak,peak_allocated=max(r['peak_allocated'] for r in records),peak_reserved=max(r['peak_reserved'] for r in records),
                    examples=8*clock.updates,validation_seconds=sum(r['evaluation_seconds'] for r in fr),checkpoint_seconds=sum(r['checkpoint_seconds'] for r in fr),initial_hash=initial,frozen_unchanged=True)
                state['B_fits_completed']+=1
            except BaseException as e:
                fit.update(status='EXECUTION_ERROR',error=repr(e),wall_seconds=time.monotonic()-started)
                # No replacement attempts. Continuing only safe scheduled cells; global resource failures stop.
                if m is not None and opt is not None:torch.save(S.snapshot(m,opt),CACHE/f'{fid}_interrupted.pt')
                if isinstance(e,ResourceError):raise
            finally:
                save(f'{fid}_steps.json',records);save('fits.json',fits);save('status.json',state);m=opt=None;cleanup()
        if state['B_fits_completed']!=12:
            state['status']='INCONCLUSIVE_EXECUTION';return
        choices=[r['best'] for r in fits];seal=dict(selections=choices,contract_hash=sha(OUT/'contract.json'),resource_selection_hash=sha(OUT/'resource_selection.json'),sealed_at=time.time())
        seal['seal_hash']=digest(seal);save('selection_seal.json',seal)
        expected=[(d,s,a) for d in DATASETS for s in SEEDS for a in ARMS];assert seal_valid(seal,sha(OUT/'contract.json'),expected)
        reloads=[]
        for r in choices:
            w.boundary();m,opt=make_model(r['arm'],r['seed'],r['cp']);restore_parameters(m,torch.load(ROOT/r['checkpoint_file'],weights_only=True));values,scale=load_data(r['dataset'],'development')
            replay=evaluate(m,values,scale,origins('validation'),'RELOAD_'+r['fit'],w,limit)
            with np.load(ROOT/r['prediction_file']) as z,np.load(ROOT/replay['prediction_file']) as b:err=float(abs(z['prediction']-b['prediction']).max())
            assert err==0 and replay['metrics']==r['metrics'];reloads.append(dict(fit=r['fit'],error=err,**replay));del m,opt;cleanup()
        save('checkpoint_reloads.json',reloads);check_contract()
        for p,h in c['historical_hashes'].items():assert sha(ROOT/p)==h
        save('evaluation_access.json',dict(opened_at=time.time(),selection_hash=sha(OUT/'selection_seal.json'),reload_hash=sha(OUT/'checkpoint_reloads.json'),scope='One final E evaluation stage; prior development E reused'))
        for d in DATASETS:
            values,scale=load_data(d,'evaluation')
            for seed in SEEDS:
                for a in ARMS:
                    r=next(r for r in choices if (r['dataset'],r['seed'],r['arm'])==(d,seed,a));m,opt=make_model(a,seed,r['cp'])
                    if seed==39000 and a=='standard':
                        z=evaluate(m,values,scale,origins('evaluation'),f'E_{d}_F0',w,limit,f0=True);evaluation.append(dict(dataset=d,seed=None,arm='F0',role='F0',**z))
                    z=evaluate(m,values,scale,origins('evaluation'),f'E_{d}_{seed}_{a}_initial',w,limit);evaluation.append(dict(dataset=d,seed=seed,arm=a,role='initial',**z))
                    restore_parameters(m,torch.load(ROOT/r['checkpoint_file'],weights_only=True))
                    z=evaluate(m,values,scale,origins('evaluation'),f'E_{d}_{seed}_{a}_selected',w,limit);evaluation.append(dict(dataset=d,seed=seed,arm=a,role='selected',selected_seconds=r['nominal_seconds'],**z));save('evaluation.json',evaluation)
                    del m,opt;cleanup()
        state['status']='B_COMPLETE'
    except BaseException as e:state.update(status='BLOCKED_GPU_BUSY' if 'BLOCKED_GPU_BUSY' in str(e) else 'INCONCLUSIVE_EXECUTION',error=repr(e),traceback=traceback.format_exc())
    finally:
        save('status.json',state);w.close();from finalize_query_numeric_v2 import finalize;finalize()


def configure_backend():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)

def backend_record():
    return dict(torch=torch.__version__,matmul_precision=torch.get_float32_matmul_precision(),
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        threads=torch.get_num_threads(),deterministic=torch.are_deterministic_algorithms_enabled(),api='legacy allow_tf32 only; no mixed fp32_precision API')

def numeric():
    state=read(OUT/'status.json');assert state['status']=='PREPARED','No numeric retry'
    c=check_contract();save('numeric_backend.json',backend_record());checks=[];paths=[];artifacts={};memory=[];warmfiles={};semantic=[]
    old=ROOT/'results/query_budget_pilot_20260915';oldindex=read(old/'artifact_index.json');oldreplay=[]
    from tsfm_peft_screen.forecast_query.budget import check_parity as v1_check
    for r in read(old/'parity.json'):
        a=torch.load(ROOT/oldindex[r['option']+'_fp32']['path'],map_location='cpu',weights_only=False)
        b=torch.load(ROOT/oldindex[r['reference']]['path'],map_location='cpu',weights_only=False)
        z=v1_check(b,a,'fp32',micro=r['kind']=='microbatch');assert z=={k:r[k] for k in z}
        oldreplay.append(dict(option=r['option'],kind=r['kind'],original_passed=z['passed'],unchanged=True))
    del a,b;save('historical_parity_replay.json',oldreplay)
    w=Watch('numeric')
    def update(m,opt,v,pair,micro,tag,**kw):
        assert state['numeric_update_attempts']<192
        state['numeric_update_attempts']+=1;save('status.json',state)
        def applied():state['numeric_updates']+=1
        r,a=step(m,opt,v,pair,micro,w,on_applied=applied,**kw);memory.append(dict(tag=tag,**r))
        save('numeric_resources.json',memory);save('status.json',state)
        return a
    def keep(tag,artifact):
        p=CACHE/(tag+'.pt');assert not p.exists();torch.save(artifact,p);artifacts[tag]=dict(path=rel(p),sha256=sha(p));save('numeric_artifacts.json',artifacts)
    pairs=c['config']['numeric_pairs'];seq=c['config']['path_pairs']
    assert all(o in origins('train') for p in pairs+seq for o in p)
    try:
        state['status']='NUMERIC_WAITING_FOR_GPU';save('status.json',state);w.boundary(startup=True)
        for d in DATASETS:
            v,scale=load_data(d,'train')
            for arm in ARMS:
                m,opt=make_model(arm,39000,0);frozen=frozen_hash(m)
                for j in range(2):update(m,opt,v,PAIRS[j],2,f'warm_{d}_{arm}_{j}')
                shared=S.snapshot(m,opt);tag=f'warm_{d}_{arm}';keep(tag,shared);warmfiles[tag]=artifacts[tag]
                # Actual group mask construction, target-independent context and same-shape replay.
                x,y,g=batch(v,pairs[0]);assert g.tolist()==[0]*4+[1]*4
                masks=[]
                def hook(module,args,kwargs):
                    mask=kwargs['group_time_mask'].detach().cpu();masks.append(mask)
                h=m.base.encoder.block[0].register_forward_pre_hook(hook,with_kwargs=True)
                with preserve_rng(),torch.no_grad():z0=m(x,g)[1].cpu()
                h.remove()
                assert masks and all((mask[..., :4,4:]==torch.finfo(mask.dtype).min).all() and (mask[...,4:,:4]==torch.finfo(mask.dtype).min).all() for mask in masks)
                with preserve_rng(),torch.no_grad():z1=m(x,g)[1].cpu()
                assert torch.equal(z0,z1)
                semantic.append(dict(dataset=d,arm=arm,origin_ids=g.cpu().tolist(),cross_origin_mask_exact=True,same_shape_fp32_replay_max_abs=float((z0-z1).abs().max()),target_not_passed_to_forward=True,train_only=True))
                del x,y,g,z0,z1,masks
                for precision in ['fp32','bf16']:
                    for j,pair in enumerate(pairs):
                        aa={}
                        for micro in [2,1]:
                            S.restore(m,opt,shared);assert S.tree_hash(S.snapshot(m,opt))==S.tree_hash(shared)
                            tag=f'{d}_{arm}_{precision}_pair{j}_mb{micro}'
                            aa[micro]=update(m,opt,v,pair,micro,tag,precision=precision,export=True);keep(tag,aa[micro])
                        result=check_parity(aa[2],aa[1],precision,micro=True,scale=scale)
                        checks.append(dict(dataset=d,arm=arm,pair=pair,reference=f'{d}_{arm}_{precision}_pair{j}_mb2',actual=f'{d}_{arm}_{precision}_pair{j}_mb1',**result));save('numeric_checks.json',checks)
                        del aa
                    ends={}
                    for micro in [2,1]:
                        S.restore(m,opt,shared)
                        for j,pair in enumerate(seq):update(m,opt,v,pair,micro,f'{d}_{arm}_{precision}_path_mb{micro}_u{j}',precision=precision)
                        x,y,g=batch(v,seq[-1])
                        with preserve_rng(),torch.no_grad(),(torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()):
                            raw=m(x,g)[1].cpu()
                        ends[micro]=dict(parameters=S.cpu_tree(S.params(m)),raw=raw);tag=f'{d}_{arm}_{precision}_path_mb{micro}';keep(tag,ends[micro]);del x,y,g,raw
                    result=path_check(shared['parameters'],ends[2]['parameters'],ends[1]['parameters'],ends[2]['raw'],ends[1]['raw'],scale,precision)
                    paths.append(dict(dataset=d,arm=arm,precision=precision,reference=f'{d}_{arm}_{precision}_path_mb2',actual=f'{d}_{arm}_{precision}_path_mb1',warm=f'warm_{d}_{arm}',**result));save('path_checks.json',paths);del ends
                assert frozen_hash(m)==frozen;semantic[-1]['frozen_hash_unchanged']=True
                save('leakage_checks.json',semantic);del m,opt,shared;cleanup()
                print('Q NUMERIC',d,arm,'updates',state['numeric_updates'],flush=True)
        assert state['numeric_updates']==180
        state['status']='NUMERICS_BOUNDED' if all(r['passed'] for r in checks+paths) else 'INCONCLUSIVE_NUMERICS_V2'
    except BaseException as e:
        state.update(status='BLOCKED_GPU_BUSY' if 'BLOCKED_GPU_BUSY' in str(e) else 'INCONCLUSIVE_NUMERICS_V2',error=repr(e),traceback=traceback.format_exc())
    finally:
        save('status.json',state);save('numeric_checks.json',checks);save('path_checks.json',paths);save('leakage_checks.json',semantic);w.close()
        print('Q NUMERIC END',state['status'],state['numeric_updates'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','numeric','profile','train-evaluate','status']);a=p.parse_args()
    if a.stage=='status':print(json.dumps(read(OUT/'status.json') if (OUT/'status.json').exists() else dict(status='NOT_PREPARED'),ensure_ascii=False,indent=2))
    else:
        configure_backend();globals()[a.stage.replace('-','_')]()
if __name__=='__main__':main()
