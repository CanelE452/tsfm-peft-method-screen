"""Read-only CPU checkpoint, selection and independently recomputed score audit."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from data import Data,scale
import torch,numpy as np,pandas as pd

torch.set_num_threads(4)
def loadstate(arm,seed,step):return torch.load(CACHE/'checkpoints'/f'{arm}_{seed}_{step}.pt',weights_only=True,map_location='cpu')
def equals(a,b):return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
def independent_metric(q,y,s):
    total=np.zeros_like(y,dtype=np.float64)
    for k in range(9):
        r=y.astype(float)-q[:,:,k].astype(float);p=(k+1)/10;total+=2*(p*np.maximum(r,0)+(1-p)*np.maximum(-r,0))/9
    return (total/s[:,None]).mean(1)
def score(d,track,role,arm,seed,tag):
    pairs=d.episodes(track,role);folder=CACHE/track/'predictions'/role/arm/f'{seed}_{tag}';q=np.stack([np.load(folder/f'{s}_{t}.npz')['q'] for s,t in pairs]);y=d.scorer_targets(pairs,role);sc=np.array([scale(d.asof(s,t)[-256:]) for s,t in pairs]);return float(independent_metric(q,y,sc).mean())
def main():
    assert read(RESULTS/'ALL_MAIN_COMPLETE.json')['status']=='COMPLETE';assert read(RESULTS/'VERIFICATION.json')['status']=='PASS';d=Data();fits=[];checkpoints=0
    for seed in SEEDS:
        g0=loadstate('G0',seed,512)['state']['bank'];encoders={}
        for arm in ['G0','STATIC','A_NOERROR','A_SET','A_TIME','B_FULLGEN','B_SET','B_TIME']:
            receipt=read(RESULTS/'fits'/f'{arm}_{seed}.json');assert receipt['frozen_unchanged'] and receipt['updates']==512 and len(receipt['trace'])==512
            for step in STEPS:
                p=CACHE/'checkpoints'/f'{arm}_{seed}_{step}.pt';assert sha(p)==receipt['checkpoints'][str(step)];cp=loadstate(arm,seed,step);assert cp['step']==step;bank=cp['state']['bank'];assert sum(v.numel() for v in bank.values())==294912
                assert all(torch.isfinite(v).all() for v in bank.values())
                if step:
                    for v in cp['optimizer']['state'].values():assert int(v['step'])==step
                elif arm=='G0':assert all(torch.count_nonzero(v)==0 for n,v in bank.items() if 'lora_B' in n)
                else:
                    assert equals(bank,g0),'New methods must start from exactly same new G0 bank'
                    gs=cp['state']['generator']
                    if gs is not None:
                        assert torch.count_nonzero(gs['head.weight'])==0 and torch.count_nonzero(gs['head.bias'])==0
                        encoders[arm]={n:v for n,v in gs.items() if n.startswith('encoder.')}
                checkpoints+=1
            fits.append(dict(arm=arm,seed=seed,status='PASS'))
        enc=list(encoders.values());assert all(equals(enc[0],v) for v in enc)
    streams=0;field_parameters={}
    for p in (CACHE/'B'/'stream_states').glob('*.pt'):
        cp=torch.load(p,weights_only=True,map_location='cpu')
        for v in cp['optimizer']['state'].values():assert int(v['step'])==96
        if cp['cell'] is not None:field_parameters['B_COSA_CELL']=sum(v.numel() for v in cp['cell'].values())
        elif cp['coefficient'] is not None:field_parameters['B_COEFF']=cp['coefficient'].numel()
        else:field_parameters['B_WAIT_AND_MASKED']=sum(v.numel() for v in cp['bank'].values())
        streams+=1
    assert streams==128
    selection_checks=[];scorechecks=0
    for track in ['A','B']:
        out=RESULTS/track;selection=read(out/'MODEL_SELECTION.json');bs=read(out/'BASELINE_SELECTION.json');raw=pd.read_csv(out/'SEED_SCORES.csv')
        for entry in selection['validation']:
            scores={step:score(d,track,'DEV',entry['arm'],entry['seed'],step) for step in STEPS}
            for step,value in scores.items():assert abs(value-entry['scores'][str(step)])<1e-12
            assert min(scores,key=lambda s:(scores[s],s))==entry['selected'];selection_checks.append(dict(track=track,arm=entry['arm'],seed=entry['seed']))
        if track=='A':
            for arm in ['A_LOCAL','A_COEFF']:
                ks={k:np.mean([score(d,track,'DEV',arm,s,k) for s in SEEDS]) for k in [0,1,4,8]};assert min(ks,key=lambda k:(ks[k],k))==bs['kselected'][arm]
        means={arm:float(np.mean([score(d,track,'DEV',arm,s,tag[str(s)]) for s in SEEDS])) for arm,tag in bs['tags'].items()}
        assert min([a for a in means if a!=track+'_TIME'],key=lambda a:(means[a],a))==bs['baseline']
        for arm,tags in bs['tags'].items():
            for seed in SEEDS:
                value=score(d,track,'TEST',arm,seed,tags[str(seed)]);r=raw[(raw.arm==arm)&(raw.seed==seed)].pinball.iloc[0];assert abs(value-r)<1e-12;scorechecks+=1
        for r in pd.read_csv(out/'EFFECTS.csv').itertuples():
            a=raw[raw.arm==r.candidate].pinball.mean();b=raw[raw.arm==r.control].pinball.mean();assert abs(100*(1-a/b)-r.improvement_pct)<1e-10
    save(RESULTS/'CHECKPOINT_AND_METRIC_AUDIT.json',dict(status='PASS',offline_fits=fits,checkpoint_files=checkpoints,stream_final_optimizer_states=streams,field_trainable_parameters=field_parameters,initial_new_G0_bank_equal=True,initial_generator_encoder_equal=True,initial_heads_zero=True,dev_selection_checks=selection_checks,independent_test_score_checks=scorechecks,ratio_of_seed_mean_scores=True,extra_optimizer_updates=0,model_forward_calls=0))
    print('PASS',checkpoints,'offline checkpoint files;',streams,'online optimizer states;',scorechecks,'independent TEST scores')

if __name__=='__main__':main()
