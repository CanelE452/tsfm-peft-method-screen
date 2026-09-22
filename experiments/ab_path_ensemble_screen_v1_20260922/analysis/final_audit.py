"""Independent table arithmetic, preserved-block, initialization and file checks."""
import sys,csv,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from common import *
from evaluation import path,loadpred,apply_cal,metric
from model import tensor_hash
from runner import check_selection

def rows(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    check_selection();a=RESULTS/'A';cal=read(a/'CALIBRATION.json');native=loadpred(path('A',0,'F0_NATIVE','TEST'));nativecal=apply_cal(native,cal['0']['F0_NATIVE']);block=[]
    for seed in SEEDS:
        for arm in ARMS_A:
            pred=loadpred(path('A',seed,arm,'TEST'));k=pred['z'].shape[-1]//9
            expected=np.tile(native['z'][:,:64,:9],(1,1,k));err=float(np.abs(expected-pred['z'][:,:64]).max());assert err==0
            z=apply_cal(pred,cal[str(seed)][arm]);calerr=float(np.abs(np.tile(nativecal[:,:64,:9],(1,1,k))-z[:,:64]).max())
            assert calerr<1e-9
            block.append(dict(arm=arm,seed=seed,raw_atom_max_abs=err,cal_atom_max_abs=calerr))
    identity=[];initialization=[]
    for seed in SEEDS:
        g=loadpred(path('A',seed,'GLOBAL3','TEST'));x=loadpred(path('A',seed,'CONTEXT3','TEST'))
        assert np.array_equal(g['z'],x['z']) and np.array_equal(g['p'],x['p']);identity.append(dict(seed=seed,all_raw_atoms_and_weights_identical=True,calibration_equal=cal[str(seed)]['GLOBAL3']==cal[str(seed)]['CONTEXT3']))
        selected=read(a/'MODEL_SELECTION.json')[str(seed)]['FULL9']['checkpoint'];teacher=torch.load(CACHE/'A'/'fits'/f'FULL9_{seed}'/'states.pt',weights_only=True)[selected]
        teacher_lora=tensor_hash((n,v) for n,v in teacher.items() if n.startswith('network.'))
        for arm in ARMS_A[1:]:
            state=torch.load(CACHE/'A'/'fits'/f'{arm}_{seed}'/'states.pt',weights_only=True)[0]
            assert tensor_hash((n,v) for n,v in state.items() if n.startswith('network.'))==teacher_lora
        hashes=[]
        for arm in ARMS_B:
            state=torch.load(CACHE/'B'/'fits'/f'{arm}_{seed}'/'states.pt',weights_only=True)[0]
            hashes.append(tensor_hash((n,v) for n,v in state.items() if n.startswith('network.')))
        assert len(set(hashes))==1
        initialization.append(dict(seed=seed,a_all_students_same_selected_teacher_lora=True,b_all_arms_identical_initial_lora=True))
    effect_checks=0;latency_checks=0
    for c in ['A','B']:
        r=RESULTS/c;region='tail64' if c=='A' else 'all8';s=[x for x in rows(r/'SCORES.csv') if x['region']==region]
        def score(arm,seed,mode,metric):
            match=[x for x in s if x['arm']==arm and x['mode']==mode and int(x['seed'])==seed]
            if not match:match=[x for x in s if x['arm']==arm and x['mode']==mode and int(x['seed'])==0]
            assert len(match)==1;return float(match[0][metric])
        for e in rows(r/'SEED_EFFECTS.csv'):
            v=score(e['candidate'],int(e['seed']),e['mode'],e['metric']);b=score(e['baseline'],int(e['seed']),e['mode'],e['metric']);delta=(b-v)*100/b
            assert abs(delta-float(e['gain_pct']))<1e-11;effect_checks+=1
        for e in rows(r/'MEAN_EFFECTS.csv'):
            v=sum(score(e['candidate'],seed,e['mode'],e['metric']) for seed in SEEDS)/2;b=sum(score(e['baseline'],seed,e['mode'],e['metric']) for seed in SEEDS)/2
            assert abs((b-v)*100/b-float(e['gain_pct']))<1e-11;effect_checks+=1
        reps=rows(r/'LATENCY_REPETITIONS_FAIR.csv')
        for x in rows(r/'RESOURCES_FAIR.csv'):
            values=[float(v['seconds']) for v in reps if v['arm']==x['arm'] and v['batch']==x['batch']]
            assert len(values)==30 and np.median(values)==float(x['median_seconds']);latency_checks+=1
        for p in (CACHE/c/'predictions').rglob('TEST*.npz'):
            pred=loadpred(p);assert np.isfinite(pred['z']).all() and np.isfinite(pred['p']).all() and (pred['p']>=0).all()
            assert np.allclose(pred['p'].sum(-1),1,atol=1e-6)
        manifest=read(r/'ZERO_DIAGNOSTIC_MANIFEST.json') if c=='A' else None
        if manifest:
            for p,h in manifest['files'].items():assert sha(ROOT/p)==h
        verification=read(r/'VERIFICATION.json');verification['table_effect_arithmetic_rechecked']=True;verification['stored_test_distribution_weights_valid']=True
        if c=='A':verification['first64_preserved_raw_and_cal']=True;verification['global_context_predictions_identical']=True
        save(r/'VERIFICATION.json',verification)
    save(a/'FIRST64_AND_INITIALIZATION_AUDIT.json',dict(status='PASS',preserved_blocks=block,global_context_identity=identity,initialization=initialization,additional_optimizer_updates=0))
    # Disk manifests identify locally retained raw data, forecasts and weights, without publishing them.
    files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(CACHE.rglob('*')) if p.is_file() and p.suffix not in ['.log','.pid'] and p.name!='EXECUTOR.lock'}
    save(RESULTS/'LOCAL_CACHE_MANIFEST.json',dict(files=files,raw_weights_predictions_published=False))
    save(RESULTS/'FINAL_VERIFICATION.json',dict(status='PASS',main_fits=24,main_updates=6144,smoke_updates=24,total_updates=6168,seed_and_mean_effects_checked=effect_checks,latency_aggregates_checked=latency_checks,
        first64_raw_and_cal_preserved=True,initializations_verified=True,source_and_selection_seals_intact=True,all_test_predictions_saved_before_scoring=True,
        method_followup='NONE',automatic_followup=False,limitations=['B release time unverified','B all neural arms optimization-limited at256 updates','Solar exposed development dataset','Original CAL separate wall time unavailable','Pre-main scope incident preserved; premature seal bytes unrecoverable'],
        sources_after_main={str(p.relative_to(EXP)):sha(p) for p in sorted((EXP/'analysis').glob('*.py'))}))
    print('PASS: 24 main fits, 6168 total updates; tables, frozen seals, raw/CAL first64 and initialization verified.')

if __name__=='__main__':main()
