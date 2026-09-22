import gc
import time
import numpy as np
import torch
from common import *
from model import ForecastModel,CoefficientModel,BiasModel,extract_basis,setup
from engine import fit_model,torch_batch
from ledger import PhaseTimer
import data


def compare(a,b,sigma):
    torch.testing.assert_close(a/sigma[:,None,None],b/sigma[:,None,None],atol=1e-5,rtol=1e-5)
    return float(((a-b)/sigma[:,None,None]).abs().max())


def main():
    assert read(RESULTS/'P0_COMPLETE.json')['native_parity']
    if (RESULTS/'SMOKE_STARTED.json').exists():
        raise RuntimeError('Smoke already attempted; inspect ledger instead of replay')
    save(RESULTS/'SMOKE_STARTED.json',dict(cap=18,source_hashes=source_hashes()))
    setup();d=data.load_data();f0=ForecastModel();reports={}
    for group in ['h1','h2']:
        if not read(RESULTS/f'{group.upper()}_PROBLEM_GATE.json')['passed']:
            continue
        timer=PhaseTimer(group,'smoke')
        seed=92250 if group=='h1' else 92260
        pairs=data.donor_schedule(seed,steps=2,d=d)
        x,_,sigma=torch_batch(d,pairs[0],'donor',seed=seed,step=1,missing=group=='h2')
        with torch.no_grad():reference=f0(x)
        if group=='h1':
            m=ForecastModel(lora=True,seed=seed)
            with torch.no_grad():zero=compare(m(x),reference,sigma)
            cps,r=fit_model(m,d,pairs,'smoke_h1_shared','h1','donor',1e-4,seed,phase='smoke',timer=timer)
            basis=extract_basis(m)
            with torch.no_grad():shared=m(x).detach()
            details=[r]
            for arm in ['coefficient','ridge']:
                cm=CoefficientModel(basis)
                with torch.no_grad():initial=compare(cm(x),shared,sigma)
                _,r=fit_model(cm,d,pairs,f'smoke_h1_{arm}','h1','donor',.01,seed,ridge=arm=='ridge',phase='smoke',timer=timer)
                r['initial_shared_parity_scaled']=initial;details.append(r)
                del cm
            m.load_learned(cps[2])
            _,r=fit_model(m,d,pairs,'smoke_h1_local','h1','donor',1e-4,seed,phase='smoke',timer=timer)
            details.append(r)
            reports['h1']=dict(status='PASS',initial_f0_parity_scaled=zero,code_paths=details,updates=8,
                               basis_modules=list(basis),coefficient_parameters=36*8)
            del m,basis,cps
        else:
            details=[]
            for arm in CONFIG['h2']['arms']:
                m=ForecastModel(lora=True,seed=seed) if arm=='QV_LORA' else BiasModel(arm)
                with torch.no_grad():initial=compare(m(x),reference,sigma)
                _,r=fit_model(m,d,pairs,f'smoke_h2_{arm}','h2','donor',1e-4 if arm=='QV_LORA' else .01,seed,missing=True,phase='smoke',timer=timer)
                r['initial_f0_parity_scaled']=initial
                if arm!='QV_LORA':
                    clean,_,s,_=d.batch(pairs[0],group='donor')
                    clean=torch.as_tensor(clean,device='cuda')
                    with torch.no_grad():
                        r['trained_clean_parity_scaled']=compare(m(clean),f0(clean),torch.as_tensor(s,device='cuda'))
                        if arm=='SET_BIAS':
                            aligned=clean.clone();aligned[:,-48:]=torch.nan
                            r['trained_aligned_parity_scaled']=compare(m(aligned),f0(aligned),torch.as_tensor(s,device='cuda'))
                details.append(r)
                del m;gc.collect();torch.cuda.empty_cache()
            reports['h2']=dict(status='PASS',code_paths=details,updates=10)
        timer.close();save(RESULTS/f'SMOKE_{group.upper()}.json',reports[group])
        event('smoke_group_complete',group=group,updates=reports[group]['updates'])
    save(RESULTS/'SMOKE_COMPLETE.json',dict(status='PASS',groups=list(reports),source_hashes=source_hashes()))


if __name__=='__main__':
    main()
