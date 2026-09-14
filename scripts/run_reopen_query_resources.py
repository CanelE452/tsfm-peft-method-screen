"""Train-only storage comparison; reuse historical quality, no forecasting fits."""
import fcntl,gc,json,time,subprocess,gzip
import pandas as pd
from contextlib import contextmanager,nullcontext
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
import run_temporal_transfer_gradient_v1 as G
import run_forecast_query_checkpoint_diagnostic as S
from tsfm_peft_screen.forecast_query.equal_time import EqualTimeModel
from tsfm_peft_screen.backbone import native_loss
from tsfm_peft_screen.reproducibility import ROOT,sha,write_json,seed_all,guard
OUT=ROOT/'results/reopen_query_resource_20260914'


def blocks(k):return np.linspace(0,11,k,dtype=int).tolist() if k else []
@contextmanager
def partial_cp(m,k):
    saved=[]
    for i in blocks(k):
        b=m.encoder.block[i];fn=b.forward;saved.append((b,fn))
        def wrapped(*args,_fn=fn,**kw):return checkpoint(_fn,*args,use_reentrant=False,**kw)
        b.forward=wrapped
    try:yield
    finally:
        for b,fn in saved:b.forward=fn


def dominates(a,b):
    return all(a[k]<=b[k] for k in ['loss','memory','time']) and any(a[k]<b[k] for k in ['loss','memory','time'])


def main():
    assert not OUT.exists(),'No overwrite/retry'
    assert subprocess.check_output(['git','ls-files','--error-unmatch',__file__],cwd=ROOT)
    OUT.mkdir();G.OUT=OUT;watch=G.Budget();start=time.monotonic();updates=0
    lock=open(ROOT/'.cache/gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    quality_path=ROOT/'results/forecast_query_equal_time/evaluation.json';quality=json.loads(quality_path.read_text())
    configs={a:list(range(0,13,3)) if a=='standard' else [0,12] if a=='query' else [0] for a in ['standard','query','side','head']}
    write_json(OUT/'contract.json',dict(execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256=sha(__file__),quality_json_sha256=sha(quality_path),configs=configs,blocks={str(k):blocks(k) for k in range(0,13,3)},
        seed=30000,origins=[17408,18432],context=4096,channels=[0,1,2,3],lr=3e-5,
        warmup='one shared no-CP update; three-step equivalence from shared state; one config warmup then 3 measured updates',
        precision='BF16; inherited equivalence absolute/relative tolerance1e-4 for each tensor category',
        optimizer_updates_cap=158,V_E_array_access=False,new_quality_scores=0,new_fits=0))
    records=[];parity=[];failures=[]
    def save(status='RUNNING',**kw):write_json(OUT/'receipt.json',dict(status=status,optimizer_updates=updates,new_fits=0,new_quality_scores=0,V_E_array_reads=0,seconds=time.monotonic()-start,**kw))
    try:
        for ds in ['ettm2','electricity']:
            # Read raw rows only through the last fixed training target; no V/E arrays.
            meta=json.loads((ROOT/f'data/processed/{ds}/manifest.json').read_text())
            for path,h in meta['sources'].items():assert sha(path)==h
            if ds=='ettm2':
                values=pd.read_csv(ROOT/'data/raw/ETTm2.csv',nrows=18480,usecols=[1,2,3,4]).to_numpy(dtype=np.float32)
            else:
                with gzip.open(ROOT/'data/raw/electricity.txt.gz','rt') as f:
                    values=np.loadtxt(f,delimiter=',',usecols=[0,1,2,3],max_rows=18480,dtype=np.float32)
            assert values.shape==(18480,4)
            x=torch.tensor(np.concatenate([values[o-4096:o].T for o in [17408,18432]]),device='cuda')
            y=torch.tensor(np.concatenate([values[o:o+48].T for o in [17408,18432]]),device='cuda')
            g=torch.arange(2,device='cuda').repeat_interleave(4)
            for arm in configs:
                watch.resource('setup');seed_all(30000);m=EqualTimeModel(arm,30000);opt=torch.optim.AdamW(S.params(m).values(),lr=3e-5,weight_decay=0.)
                frozen_before=S.tree_hash({n:p for n,p in m.named_parameters() if not p.requires_grad})
                def step(k,export=False):
                    nonlocal updates
                    assert updates<158
                    opt.zero_grad(set_to_none=True)
                    before=S.cpu_tree(S.params(m)) if export else None
                    m.checkpoint_enabled=arm=='query' and bool(k)
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();allocated=torch.cuda.memory_allocated();tick=time.perf_counter()
                    with partial_cp(m.base,k) if arm=='standard' else nullcontext():
                        with torch.autocast('cuda',dtype=torch.bfloat16):z,p,l,s=m(x,g);loss=native_loss(z,y,l,s)
                    assert torch.isfinite(loss);loss.backward()
                    rawgrad=S.flatten({n:v.grad.detach().cpu().clone() for n,v in S.params(m).items()}) if export else None
                    norm=torch.nn.utils.clip_grad_norm_(S.params(m).values(),1.,error_if_nonfinite=True)
                    updates+=1;opt.step();torch.cuda.synchronize()
                    resource=dict(seconds=time.perf_counter()-tick,start_allocated=allocated,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
                    artifact=None
                    if export:
                        now=S.cpu_tree(S.params(m))
                        adam=torch.cat([v.detach().cpu().reshape(-1) for _,st in sorted(opt.state_dict()['state'].items()) for _,v in sorted(st.items()) if isinstance(v,torch.Tensor)])
                        artifact=dict(output=p.detach().cpu(),loss=loss.detach().cpu().reshape(1),gradient=rawgrad,
                            clipped_gradient=S.flatten({n:v.grad.detach().cpu().clone() for n,v in S.params(m).items()}),
                            update=S.flatten({n:now[n]-before[n] for n in now}),adam=adam)
                    return resource,artifact
                step(0);shared=S.snapshot(m,opt)
                refs=[step(0,True)[1] for _ in range(3)]
                for k in configs[arm]:
                    watch.resource('config');S.restore(m,opt,shared)
                    if k==0:metrics=[{key:dict(max_absolute=0.,relative_l2=0.) for key in refs[0]} for _ in range(3)]
                    else:
                        metrics=[]
                        for i in range(3):
                            _,a=step(k,True);metrics.append({key:S.delta(refs[i][key],a[key]) for key in a});del a
                    ok=all(v['max_absolute']<=1e-4 and v['relative_l2']<=1e-4 for row in metrics for v in row.values())
                    parity.append(dict(dataset=ds,arm=arm,checkpoint_blocks=k,passed=ok,steps=metrics))
                    if not ok:
                        failures.append(dict(dataset=ds,arm=arm,k=k,reason='NUMERICAL_INEQUIVALENCE'));continue
                    S.restore(m,opt,shared);step(k);measured=[]
                    for i in range(3):
                        watch.resource('measure');resource,_=step(k);measured.append(resource)
                    records.append(dict(dataset=ds,arm=arm,checkpoint_blocks=k,block_indices=blocks(k) if arm=='standard' else None,
                        measured=measured,peak_allocated=max(r['peak_allocated'] for r in measured),peak_reserved=max(r['peak_reserved'] for r in measured),
                        median_seconds=float(np.median([r['seconds'] for r in measured])),trainable_parameters=sum(p.numel() for p in S.params(m).values())))
                    write_json(OUT/'measurements.json',records);write_json(OUT/'parity.json',parity);save(configs=len(records))
                    print(json.dumps(dict(dataset=ds,arm=arm,k=k,seconds=records[-1]['median_seconds'],memory=records[-1]['peak_allocated'])),flush=True)
                assert frozen_before==S.tree_hash({n:p for n,p in m.named_parameters() if not p.requires_grad})
                del m,opt,shared,refs;gc.collect();torch.cuda.empty_cache()
            del x,y,g;gc.collect();torch.cuda.empty_cache()
        points=[]
        for r in records:
            losses=[v['metrics']['scaled_2pinball'] for v in quality if v['dataset']==r['dataset'] and v['arm']==r['arm']]
            assert len(losses)==2
            points.append(dict(dataset=r['dataset'],arm=r['arm'],cp=r['checkpoint_blocks'],loss=float(np.mean(losses)),seed_losses=losses,memory=r['peak_allocated'],time=r['median_seconds']))
        decisions=[]
        for p in points:
            if p['arm']!='query':continue
            dominators=[r for r in points if r['dataset']==p['dataset'] and r['arm'] in ['standard','side'] and dominates(r,p)]
            decisions.append(dict(query=p,dominators=dominators,verdict='QUERY_DOMINATED' if dominators else 'QUERY_RESOURCE_FRONTIER'))
        write_json(OUT/'frontier.json',dict(points=points,query_decisions=decisions,failures=failures,scope='Historical selected quality mean of 2 seeds; resource seed30000 fixed train batch. Not a common training-time or fresh-quality superiority claim.'))
        save('COMPLETED',configurations=len(records),failures=failures,resources=guard(start),exit_code=0)
    except BaseException as e:save('EXECUTION_ERROR',error=repr(e),exit_code=1);raise


if __name__=='__main__':main()
