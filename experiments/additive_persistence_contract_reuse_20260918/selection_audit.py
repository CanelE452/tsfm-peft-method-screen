"""Recompute only the contracted V choices and published effect ratios, CPU only."""
import pandas as pd
from .audit import ROOT,OUT,PARENT,read,save,sha,old
from experiments.additive_persistence_validation_v1_20260917.train import choice
from experiments.additive_persistence_validation_v1_20260917.evaluate import VSTATES
from experiments.additive_persistence_validation_v1_20260917.prepare import STATES
np=old.np

def run():
    checked=[]
    for source,d in read(PARENT/'LR_SELECTION.json').items():
        for arm,r in d.items():assert choice(source,arm,85550 if source=='ettm2' else 81550)==r;checked.append([source,arm])
    cal=read(PARENT/'CALIBRATION_SELECTION.json');calhash={}
    for key,r in cal.items():
        source,seed=key.rsplit('_',1);paths=[old.CACHE/'V_predictions'/f'{source}_{a}_{seed}.npy' for a in ['C0','C2']]+[old.CACHE/'conditions'/source/'V_SELECT_y.npy']
        for p in paths:calhash[str(p.relative_to(ROOT))]=sha(p)
        p0,p2,y=[np.load(p) for p in paths];sig=np.tile(read(PARENT/'DATA_MANIFEST.json')[source]['sigma_train_population'],len(y)//4);n=len(y)//20
        ids=np.concatenate([np.arange(STATES.index(s)*2*n,(STATES.index(s)+1)*2*n) for s in VSTATES]);obj=[]
        for alpha in [0.,.25,.5,.75,1.]:
            pred=p0 if alpha==0 else p2 if alpha==1 else p0+alpha*(p2-p0);obj.append((float(np.mean(abs(pred[ids,4]-y[ids])/sig[ids,None])),alpha))
        assert min(obj)[1]==r['alpha'];np.testing.assert_allclose(np.median(((y[ids]-p0[ids,4])/sig[ids,None]).reshape(-1)),r['beta'],rtol=1e-10,atol=1e-12)
    effects=pd.read_csv(PARENT/'UNCERTAINTY.csv');raw=pd.read_csv(PARENT/'RAW_SCORES.csv');count=0
    for r in effects[effects.condition!='HISTORY_SUBSET'].to_dict('records'):
        f=raw[(raw.panel==r['panel'])&(raw.kind==r['kind'])&(raw.stage=='selected')]
        if r['seed_scope']=='original2':f=f[f.seed.isin([81551,81552])]
        f=f[f.condition.str.startswith(('POINT','BURST'))] if r['condition']=='FAULT' else f[f.condition==r['condition']]
        a=float(f[f.arm==r['new']].nmae.mean());b=float(f[f.arm==r['baseline']].nmae.mean());np.testing.assert_allclose([a,b,100*(1-a/b)],[r['new_nmae'],r['baseline_nmae'],r['gain_pct']],rtol=1e-10,atol=1e-10);count+=1
    assert effects.groupby('panel').common_block_draw_sha256.nunique().eq(1).all()
    seal=read(PARENT/'MASTER_SEAL.json');original=read(PARENT/'pre_E_statistics_repair/MASTER_SEAL.json');changed={bucket:[p for p in seal[bucket] if original[bucket][p]!=seal[bucket][p]] for bucket in ['source_hashes','data_hashes']}
    assert changed['source_hashes']==['experiments/additive_persistence_validation_v1_20260917/statistics.py'];assert changed['data_hashes']==['results/additive_persistence_validation_v1_20260917/MASTER_PROTOCOL.md'];amend=read(PARENT/'SEAL_AMENDMENT_01.json');assert not amend['new_E_scored'] and amend['extra_optimizer_updates']==0
    assert amend['at']<read(PARENT/'GLOBAL_EVALUATION_SEAL.json')['at']<read(PARENT/'ALL_PREDICTIONS_SAVED.json')['at']
    save(OUT/'SELECTION_EFFECT_REPLAY.json',dict(status='VERIFIED',lr_choices=checked,calibration_choices=len(cal),v_cache_hashes=calhash,raw_effect_rows_replayed=count,history_subset_rows_not_in_this_ratio_replay=int((effects.condition=='HISTORY_SUBSET').sum()),common_time_draws_per_panel=1,pre_E_contract_alignment_changed_only=changed,training_and_model_selection_unchanged=True,optimizer_updates=0));print('SELECTION_EFFECT_REPLAY',len(checked),len(cal),count,flush=True)
if __name__=='__main__':run()
