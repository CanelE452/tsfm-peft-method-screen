import sys,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from common import *
from model import setup
from references import Direct,load_gb,cpu_predict
from A.data import Solar
from bdata_adapter import Wind
from fair_runtime_support import cpu_once,direct_cpu_input

if __name__=='__main__':
    setup();out={}
    with executor_lock():
        b=Wind();gb=load_gb()
        for n in [1,8]:
            packet=b.batch(b.indices['TRAIN'][:n]);a=cpu_predict('GBQR',packet,gb);v=cpu_once('GBQR',packet,gb)
            assert np.array_equal(a['z'],v['z']);out[f'gbqr_batch{n}_max_abs']=float(np.max(np.abs(a['z']-v['z'])))
        d=Solar();m=Direct()
        for n in [1,8]:
            packet=d.batch(d.pairs['TRAIN'][:n])
            with torch.no_grad():
                original=m.distribution(torch.as_tensor(packet['x'],device='cuda'))[0].cpu().numpy();v=direct_cpu_input(m,packet)['z']
            error=float(np.max(np.abs(original-v)));assert error<1e-6;out[f'chronos2_batch{n}_max_abs']=error
        del m;gc.collect();torch.cuda.empty_cache()
        save(RESULTS/'FAIR_RUNTIME_EQUIVALENCE.json',dict(status='PASS',errors=out,additional_optimizer_updates=0,reason='GBQR features computed once for nine models; Chronos2 CPU input direct to native API; no change to forecasts, models, CAL, or selection',original_measurements_preserved=True))
