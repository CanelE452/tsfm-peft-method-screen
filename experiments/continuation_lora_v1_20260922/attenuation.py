import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import gc
import time
import numpy as np
import pandas as pd
import torch
from common import *
from model import BranchModel, setup
from metrics import point_loss

OLD='branch_mixture_lora_v1_20260922'
LAMBDAS=(0.,.25,.5,1.)

@torch.no_grad()
def infer(model,x):
    return np.concatenate([model.forecast(torch.as_tensor(x[j:j+8],device='cuda'))[0].cpu().numpy()
                           for j in range(0,len(x),8)])

def main():
    setup()
    assert not (RESULTS/'ATTENUATION_DIAGNOSTIC.json').exists(), 'Diagnostic already completed'
    oldresults=ROOT/'results'/OLD
    selections=read(oldresults/'SELECTIONS.json')
    rows=[r for r in selections if r['arm']=='MIXTURE']
    assert [r['seed'] for r in rows]==[92231,92232]
    path=ROOT/'.cache'/OLD/'data/ETTh2.npz'
    audit=read(oldresults/'DATA_AUDIT.json')
    # The archived audit must contain the exact prepared file hash.
    assert sha(path) in __import__('json').dumps(audit)
    with np.load(path,allow_pickle=True) as z:
        values=z['values']; sigma=z['sigma']; origins=z['origins_VALIDATION']
    x=np.stack([values[o-512:o].T for o in origins]).reshape(-1,512)
    y=np.stack([values[o:o+128].T for o in origins])
    base=BranchModel(lora=False)
    f0=infer(base,x)
    del base
    gc.collect(); torch.cuda.empty_cache()
    scores=[]; checks=[]; receipts=[]
    start=time.perf_counter()
    for row in rows:
        rec=row['checkpoint']
        assert sha(ROOT/rec['path'])==rec['sha256']
        state=torch.load(ROOT/rec['path'],map_location='cpu',weights_only=False)['learned']
        assert any('lora_B' in k for k in state)
        m=BranchModel(row['seed'])
        archive=ROOT/'.cache'/OLD/'predictions/VALIDATION'/f"{row['key']}_step{row['selected_step']}.npz"
        receipt=read(archive.with_suffix('.json'))
        assert sha(archive)==receipt['sha256']
        with np.load(archive) as z: reference=z['q'].reshape(-1,128,9)
        for strength in LAMBDAS:
            scaled={k:(v*strength if 'lora_B' in k else v.clone()) for k,v in state.items()}
            m.load_learned(scaled)
            q=infer(m,x)
            if strength in (0.,1.):
                expected=f0 if strength==0 else reference
                np.testing.assert_allclose(q,expected,atol=1e-5,rtol=1e-5)
                checks.append(dict(seed=row['seed'],strength=strength,max_abs=float(np.abs(q-expected).max())))
            shaped=q.reshape(len(origins),len(sigma),128,9)
            score=float(point_loss(y[:,:,64:],np.sort(shaped[:,:,64:],axis=-1),sigma).mean())
            scores.append(dict(seed=row['seed'],**{'lambda':strength},score=score,selected_step=row['selected_step']))
            event('attenuation_validation',**scores[-1])
        receipts.append(dict(seed=row['seed'],checkpoint=rec,validation_prediction_sha=receipt['sha256']))
        del m
        gc.collect(); torch.cuda.empty_cache()
    frame=pd.DataFrame(scores)
    frame.to_csv(RESULTS/'attenuation_scores.csv',index=False)
    means=frame.groupby('lambda').score.mean()
    selected=min(LAMBDAS,key=lambda a:(means.loc[a],a))
    save(RESULTS/'ATTENUATION_DIAGNOSTIC.json',dict(complete=True,role='VALIDATION_ONLY',old_test_read_or_scored=False,
        optimizer_updates=0,checkpoint_receipts=receipts,prepared_data_sha=sha(path),checks=checks,
        selected_lambda=selected,mean_scores={str(a):float(means.loc[a]) for a in LAMBDAS},
        selection_rule='lowest equal-seed mean VALIDATION tail score; smaller lambda tie',
        changes_to_new_training_protocol=False,seconds=time.perf_counter()-start,score_csv_sha=sha(RESULTS/'attenuation_scores.csv')))

if __name__=='__main__': main()
