"""Post-run CPU verification using direct quantile arithmetic and saved states."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from data import Data
from model import hash_tensors
import torch
import numpy as np
import pandas as pd

torch.set_num_threads(4)
assert read(RESULTS/'VERIFICATION.json')['status']=='PASS'
scores=pd.read_csv(RESULTS/'SCORES.csv')
effects=pd.read_csv(RESULTS/'EFFECTS.csv')
checks=[];overlaps=[]
for source in SOURCES:
    d=Data(source)
    assert json.loads(json.dumps(d.audit,default=str))==read(RESULTS/source/'DATA_AUDIT.json')
    bases=torch.load(CACHE/f'{source}_bases.pt',weights_only=False)
    assert sha(CACHE/f'{source}_bases.pt')==read(RESULTS/source/'BASIS_AUDIT.json')['basis_file_sha256']
    for control in ['PCA','SHUFFLE']:
        for layer in range(6):
            a,b=bases['TEMP'][layer],bases[control][layer]
            overlap=float(np.square(a@b.T).sum()/8)
            assert 0<=overlap<=1.00001
            overlaps.append(dict(source=source,control=control,layer=layer,subspace_overlap=overlap))
    for seed in SEEDS:
        with np.load(CACHE/f'{source}_schedule_{seed}.npz') as z:assert np.array_equal(z['pairs'],d.schedule(seed))
        for arm in ARMS:
            f=read(RESULTS/'fits'/f'{source}_{arm}_{seed}.json')
            folder=CACHE/'fits'/f['key']
            states=torch.load(folder/'states.pt',weights_only=True)
            assert hash_tensors(states[0].items())==f['initial_sha']
            assert hash_tensors(states[f['selected']].items())==f['selected_sha']
            for step in [32,128,256,512]:
                cp=torch.load(folder/f'checkpoint_{step}.pt',weights_only=True)
                assert cp['step']==step and hash_tensors(cp['model'].items())==hash_tensors(states[step].items())
                for v in cp['optimizer']['state'].values():assert int(v['step'])==step
            for variant,step in [('raw128',128),('selected_raw',f['selected']),('selected_cal',f['selected'])]:
                with np.load(CACHE/source/'predictions'/f'{arm}_{seed}_TEST_{step}.npz') as z:
                    assert np.array_equal(z['pairs'],d.pairs['TEST'])
                    assert np.array_equal(z['y'],d.batch(d.pairs['TEST'])['y'])
                    q=z['q'].copy();y=z['y'].astype(float);sigma=z['sigma'].copy()
                    if variant=='selected_cal':
                        center=q[:,:,4,None].copy()
                        q=center+f['cal']['beta']*sigma[:,None,None]+f['cal']['alpha']*(q-center)
                    # Separate loop and positive/negative residual formula.
                    total=np.zeros_like(y)
                    for j in range(9):
                        r=y-q[:,:,j].astype(float);level=(j+1)/10
                        total+=2*(level*np.maximum(r,0)+(1-level)*np.maximum(-r,0))/9
                    result=float((total/sigma[:,None]).mean())
                    stored=float(scores.query('source==@source and arm==@arm and seed==@seed and variant==@variant').pinball.iloc[0])
                    assert np.isclose(result,stored,atol=1e-14,rtol=0),(source,arm,variant,result,stored)
            checks.append(dict(source=source,arm=arm,seed=seed,states_verified=5,optimizer_states_verified=4,scores_verified=3))
for r in effects.itertuples():
    a=scores[(scores.source==r.source)&(scores.arm==r.candidate)&(scores.variant==r.variant)].pinball.mean()
    b=scores[(scores.source==r.source)&(scores.arm==r.control)&(scores.variant==r.variant)].pinball.mean()
    assert abs(100*(1-a/b)-r.improvement_pct)<1e-10
pd.DataFrame(overlaps).to_csv(RESULTS/'BASIS_OVERLAP.csv',index=False)
save(RESULTS/'POSTRUN_AUDIT.json',dict(status='PASS',fit_checks=checks,raw_data_rechecked=True,packet_schedule_equal=True,independent_score_formula=True,ratio_of_mean_scores_verified=True,main_extra_updates=0,model_forward_calls=0,scope='Saved arrays and CPU checkpoint tensors; not an independent data replication'))
print('PASS: 16 fits, 80 states, 64 optimizer checkpoints, 48 direct score checks and all effect ratios')
