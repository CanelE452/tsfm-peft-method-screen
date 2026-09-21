"""Read-only validation/checkpoint audit; does not load models or score TEST."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'/'rollout_uncertainty_peft_v1_20260921'))
import numpy as np
from common import *
from data import load

def audit(source):
    d=load(source)
    y=np.stack([d['values'][o:o+256].T for o in d['origins']['VALIDATION']]).astype('float64')
    fits=[read_json(f) for f in sorted((RESULTS/'fits').glob(f'{source}_*.json'))]
    assert len(fits)==12
    checks=[]; initial=[]
    for fit in fits:
        assert set(fit['validation'])=={'0','128','256','512'}
        for step,record in fit['validation'].items():
            path=CACHE/'predictions'/source/'VALIDATION'/f"{fit['fit_id']}_step{step}.npy"
            receipt=read_json(path.with_suffix('.json'))
            assert sha(path)==receipt['sha256']
            assert sha(ROOT/record['checkpoint'])==record['checkpoint_sha256']
            q=np.sort(np.load(path),axis=-1).astype('float64')
            # Quantile-wise accumulation independently of the production metric.
            score=0.
            for j in range(9):
                e=y-q[...,j]; tau=(j+1)/10
                score+=np.where(e>=0,tau*e,(tau-1)*e).__truediv__(d['sigma'][None,:,None]).mean()*2/9
            error=abs(score-record['scaled_pinball'])
            assert error<1e-10
            checks.append({'fit_id':fit['fit_id'],'step':int(step),'recomputed_validation':float(score),'absolute_error':float(error)})
            if step=='0': initial.append(receipt['sha256'])
        winner=min(fit['validation'],key=lambda s:(fit['validation'][s]['scaled_pinball'],int(s)))
        assert int(winner)==fit['selected_step']
    assert len(set(initial))==1
    for seed in [92120,92121,92122]:
        same=[f for f in fits if f['seed']==seed]
        assert len({f['initial_lora_hash'] for f in same})==1
        assert len({f['initial_adapter_hash'] for f in same if f['arm']!='R_ROLLOUT_LORA'})==1
    rows=[json.loads(l) for l in (RESULTS/'UPDATE_LEDGER.jsonl').read_text().splitlines()]
    ids={f['fit_id'] for f in fits}
    updates=[r for r in rows if r['kind']=='main' and r['fit_id'] in ids]
    assert len(updates)==6144 and len({(r['fit_id'],r['step']) for r in updates})==6144
    assert all(r['conditional_calls']==4 for r in updates)
    out={'status':'PASS','source':source,'fits':12,'main_updates':6144,'conditional_batch_forwards_backwards':24576,'validation_checkpoints_recomputed':48,'initial_validation_prediction_hash_shared_across_all_arms_seeds_lrs':initial[0],'max_absolute_error':max(r['absolute_error'] for r in checks),'checks':checks,'source_script_sha256':sha(Path(__file__))}
    save_json(RESULTS/f'VALIDATION_AUDIT_{source}.json',out)
    print(source,'PASS: 12 fits, 6144 updates, 48 recomputed VAL scores; max error',out['max_absolute_error'])

if __name__=='__main__':
    for source in sys.argv[1:] or ['Electricity','ETTh1']: audit(source)
