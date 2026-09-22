import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,gc,time
import numpy as np
import torch
from common import *
from model import ModelA,ModelB
from engine import tensors,distribution
from evaluation import apply_cal,quantile
from A.data import Solar
from bdata_adapter import Wind
from references import make,load_gb
from fair_runtime_support import cpu_once as cpu_predict,direct_cpu_input

def run():
    from runner import selected_state,check_selection
    check_selection()
    if (RESULTS/'FAIR_RUNTIME_COMPLETE.json').exists():return
    for c in read(RESULTS/'EXECUTION_PLAN.json')['completed_candidates']:
        d=Solar() if c=='A' else Wind();cal=read(RESULTS/c/'CALIBRATION.json')
        arms=ARMS_A+['F0_NATIVE','F0_MEDIAN','CHRONOS2'] if c=='A' else ARMS_B+['F0','NAIVE','GBQR']
        rows=[];counts=[]
        for roundno in range(2):
            ordering=arms if roundno==0 else arms[::-1]
            for arm in ordering:
                seed=SEEDS[0] if arm in (ARMS_A if c=='A' else ARMS_B) else 0
                cpu=arm in ['NAIVE','GBQR']
                if seed:
                    m=ModelA(arm,seed) if c=='A' else ModelB(arm,d.dim,seed);m.load_learned(selected_state(c,seed,arm))
                else:m=None if cpu else make(c,arm,d)
                gb=load_gb() if arm=='GBQR' else None;coefficients=cal[str(seed)][arm]
                if roundno==0:
                    measured=m if hasattr(m,'parameters') else (m.pipeline.model if m is not None else None)
                    counts.append(dict(arm=arm,seed=seed,trainable_parameters=sum(p.numel() for p in measured.parameters() if p.requires_grad) if measured is not None and seed else 0,
                        resident_parameter_bytes=sum(p.numel()*p.element_size() for p in measured.parameters()) if measured is not None else None))
                    del measured
                for batch in [1,8]:
                    packet=d.batch(d.pairs['TRAIN'][:batch]) if c=='A' else d.batch(d.indices['TRAIN'][:batch])
                    if c=='B':
                        indices=packet['pairs'][:,1];packet['weather']=d.weather_raw[indices];packet['control']=d.control_raw[indices]
                        if arm in ['TARGET','F0','NAIVE']:packet.pop('weather');packet.pop('control')
                        elif arm=='CONTROL':packet.pop('weather')
                        elif arm!='GBQR':packet.pop('control')
                    for rep in range(20):
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                        incoming={k:np.array(v,copy=True) for k,v in packet.items()}
                        if c=='B':
                            for field in ['weather','control']:
                                if field in incoming:incoming[field]=(incoming[field]-d.weather_mean)/d.weather_std
                        if cpu:pred=cpu_predict(arm,incoming,gb)
                        elif arm=='CHRONOS2':
                            with torch.no_grad():pred=direct_cpu_input(m,incoming)
                        else:
                            with torch.no_grad():z,p,_=distribution(m,tensors(incoming),c)
                            pred=dict(z=z.cpu().numpy(),p=p.cpu().numpy(),sigma=incoming['sigma'])
                        adjusted=apply_cal(pred,coefficients)
                        readout=np.stack([quantile(adjusted,pred['p'],q) for q in np.arange(1,10)/10],-1)
                        assert np.isfinite(readout).all();torch.cuda.synchronize();seconds=time.perf_counter()-start
                        if rep>=5:rows.append(dict(candidate=c,arm=arm,seed=seed,batch=batch,round=roundno,repeat=rep-5,seconds=seconds,
                           peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
                    if not cpu and arm!='CHRONOS2':del z,p
                del m,gb;gc.collect();torch.cuda.empty_cache()
        with (RESULTS/c/'LATENCY_REPETITIONS_FAIR.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summary=[]
        for arm in arms:
            for batch in [1,8]:
                chosen=[r for r in rows if r['arm']==arm and r['batch']==batch];times=[r['seconds'] for r in chosen]
                summary.append(dict(arm=arm,batch=batch,repetitions=len(times),median_seconds=float(np.median(times)),q25=float(np.quantile(times,.25)),q75=float(np.quantile(times,.75)),
                    peak_allocated=max(r['peak_allocated'] for r in chosen),peak_reserved=max(r['peak_reserved'] for r in chosen)))
        with (RESULTS/c/'RESOURCES_FAIR.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
        save(RESULTS/c/'RUNTIME_AUDIT_FAIR.json',dict(status='PASS',warmup=10,timed=30,batches=[1,8],round_orders=[arms,arms[::-1]],model_one_at_a_time=True,
            a_first_f0_computed_every_call=True,b_target_forward_once=True,includes='CPU copy, weather normalization, H2D, model, routing/member preparation, D2H, CAL, nine discrete CDF quantiles',
            excludes='disk I/O and model loading',unused_weather_not_transferred=True,parameter_counts=counts,benchmark_seed=SEEDS[0]))
    save(RESULTS/'FAIR_RUNTIME_COMPLETE.json',dict(status='COMPLETE',additional_optimizer_updates=0))

if __name__=='__main__':
    from model import setup
    setup()
    assert read(RESULTS/'RUN_COMPLETE.json')['status']=='COMPLETE'
    with executor_lock():run()
