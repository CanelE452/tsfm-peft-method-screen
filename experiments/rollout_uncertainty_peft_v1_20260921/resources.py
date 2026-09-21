import gc
import time
import numpy as np
import pandas as pd
import torch
from common import *
from model import rollout,native,mc16,direct
from evaluation import apply_affine

def measure_all():
    from runner import SOURCES,model_specs,open_model,load,key_for
    selections=read_json(RESULTS/'MODEL_SELECTION.json')
    calibration=read_json(RESULTS/'CALIBRATION_PARAMETERS.json')
    rows=[]
    for source in SOURCES:
        d=load(source); origin=d['origins']['CALIBRATION'][0]
        for name,seed,kind,selection in model_specs(selections,source):
            started=time.perf_counter(); model=open_model(name,kind,selection); torch.cuda.synchronize()
            loading=time.perf_counter()-started
            for bs in [1,8]:
                indexes=np.arange(bs)%len(d['sigma'])
                xcpu=torch.tensor(d['values'][origin-512:origin,indexes].T.copy(),dtype=torch.float32)
                scpu=torch.tensor(d['sigma'][indexes],dtype=torch.float32)
                params=calibration[f'{source}/{key_for(name,seed)}']
                def infer(variant):
                    with torch.no_grad():
                        x=xcpu.cuda(); sigma=scpu.cuda()
                        if kind=='single': q=rollout(model,x,sigma)[0]
                        elif kind=='native': q=native(model,x).sort(dim=-1).values
                        elif kind=='mc16':
                            u=torch.from_numpy(np.random.default_rng(1701 if seed==92121 else 1702).random((bs,4,16,64)).astype('float32')).cuda()
                            q=mc16(model,x,u)[0].sort(dim=-1).values
                        else: q=direct(model,x).sort(dim=-1).values
                        arr=q.cpu().numpy()
                        if variant=='affine': arr=apply_affine(arr[None],scpu.numpy(),params)[0]
                        assert np.isfinite(arr).all()
                        return arr
                for variant in ['ordered','affine']:
                    infer(variant)
                    times=[]; peaks=[]
                    for repeat in range(3):
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        start=time.perf_counter(); infer(variant); torch.cuda.synchronize()
                        times.append(time.perf_counter()-start); peaks.append(torch.cuda.max_memory_allocated())
                    rows.append({'source':source,'method':name,'seed':seed,'variant':variant,'batch_size':bs,'median_wall_seconds':float(np.median(times)),'min_wall_seconds':min(times),'max_wall_seconds':max(times),'peak_allocated_bytes':max(peaks),'model_loading_seconds':loading,'conditional_contexts_per_example':{'single':4,'native':28,'mc16':49,'direct':1}[kind],'scope':'CPU context to GPU, model, metadata, sorting, optional affine, CPU transfer; model loading excluded','repeats':3})
            del model; gc.collect(); torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(RESULTS/'RESOURCES.csv',index=False)
    event('resources_complete',rows=len(rows))
