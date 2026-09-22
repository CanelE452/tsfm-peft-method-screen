import gc
import sys
import time
import traceback
import numpy as np
import torch
from common import *
from model import ForecastModel,CoefficientModel,BiasModel,extract_basis,setup,tensor_hash
from engine import fit_model
from evaluation import packet,predict,predpath,calibration,select,readpred
from ledger import PhaseTimer
import data


H1_METHODS=['F0','SHARED','LOCAL_LORA','RIDGE','UNSHRUNK','FIXED_0','FIXED_0.25','FIXED_0.5','FIXED_1','U_SHRINK']
H1_FINAL=['F0','SHARED','LOCAL_LORA','RIDGE','UNSHRUNK','FIXED_SHRINK','U_SHRINK','PERMUTED_G']
H2_METHODS=['F0_NATIVE','F0_INTERP']+CONFIG['h2']['arms']


def cached_fit(model,d,pairs,key,**kwargs):
    folder=CACHE/'fits'/key
    if (folder/'complete.json').exists():
        report=read(folder/'complete.json')
        assert read(RESULTS/'OPTIMIZER_LEDGER.json')['fits'][key]['status']=='COMPLETE'
        assert report['steps']==len(pairs)
        assert tensor_hash(model.learned().items())==report['initial_sha256']
        assert model.frozen_hash()==report['frozen_sha256']
        with np.load(folder/'schedule.npz') as z:np.testing.assert_array_equal(z['pairs'],np.asarray(pairs).astype(str))
        cps=torch.load(folder/'checkpoints.pt',weights_only=True)
        assert tensor_hash(cps[max(cps)].items())==report['final_sha256']
        model.load_learned(cps[max(cps)])
        return cps
    cps,_=fit_model(model,d,pairs,key,**kwargs)
    return cps


def clients(m,cm,d,seed,group,shared,timer):
    states={};zero={n:torch.zeros_like(v) for n,v in cm.learned().items()}
    for sid in d.group_ids(group):
        a=data.block_schedule(sid,group,'adapt_a',seed,d=d)
        b=data.block_schedule(sid,group,'adapt_b',seed,d=d)
        combined=np.stack([a,b],axis=1).reshape(64,4,2)
        current={}
        for arm,schedule in [('BLOCK_A',a.reshape(32,4,2)),('BLOCK_B',b.reshape(32,4,2)),('RIDGE',combined),('LOCAL_LORA',combined)]:
            target=m if arm=='LOCAL_LORA' else cm
            target.load_learned(shared[256] if arm=='LOCAL_LORA' else zero)
            current[arm]=cached_fit(target,d,schedule,f'h1_{group}_{sid}_{arm}_s{seed}',candidate='h1',datagroup=group,
                 lr=1e-4 if arm=='LOCAL_LORA' else .01,seed=seed,ridge=arm=='RIDGE',timer=timer)
        states[sid]=current
    return states


def coefficients(client_states,stage,tau2=None):
    stats={}
    for sid,arms in client_states.items():
        a=arms['BLOCK_A'][[0,16,32][stage]];b=arms['BLOCK_B'][[0,16,32][stage]]
        mean={n:(a[n]+b[n])/2 for n in a}
        variance=float(torch.cat([(a[n].double()-b[n].double()).flatten() for n in a]).square().mean()/4)
        energy=float(torch.cat([v.double().flatten() for v in mean.values()]).square().mean())
        stats[sid]=dict(mean=mean,variance=variance,energy=energy)
    if tau2 is None:tau2=max(float(np.mean([s['energy']-s['variance'] for s in stats.values()])),0.)
    for value in stats.values():
        denominator=tau2+value['variance'];value['g']=tau2/denominator if denominator else 0.
    return stats,tau2


def h1_states(client_states,stage,arm,tau2):
    if arm in ['RIDGE','LOCAL_LORA']:
        return {sid:a[arm][[0,32,64][stage]] for sid,a in client_states.items()}
    stats,_=coefficients(client_states,stage,tau2);ids=list(stats)
    result={}
    for j,sid in enumerate(ids):
        g=1.
        if arm=='U_SHRINK':g=stats[sid]['g']
        elif arm=='PERMUTED_G':g=stats[ids[(j+1)%len(ids)]]['g']
        elif arm.startswith('FIXED_'):g=float(arm.split('_')[1])
        result[sid]={n:v*g for n,v in stats[sid]['mean'].items()}
    return result


def h1_predict(m,cm,f0,p,path,shared,client_states,stage,arm,tau2):
    if arm=='F0':return predict(f0,p,path)
    if arm=='SHARED':
        m.load_learned(shared[[0,128,256][stage]])
        return predict(m,p,path)
    target=m if arm=='LOCAL_LORA' else cm
    return predict(target,p,path,h1_states(client_states,stage,arm,tau2))


def prepare_h1(d):
    if (RESULTS/'PREPARED_H1.json').exists():return
    timer=PhaseTimer('h1','train_select_and_eval_cal');allstates={};taureceipt={}
    try:
        for seed in CONFIG['h1']['seeds']:
            m=ForecastModel(lora=True,seed=seed);f0=ForecastModel()
            shared=cached_fit(m,d,data.donor_schedule(seed,d=d),f'h1_SHARED_s{seed}',candidate='h1',datagroup='donor',lr=1e-4,seed=seed,timer=timer)
            basis=extract_basis(m);cm=CoefficientModel(basis)
            ds=clients(m,cm,d,seed,'dev',shared,timer)
            taus=[coefficients(ds,j)[1] for j in range(3)]
            taureceipt[str(seed)]={str(j):dict(tau2=taus[j],clients={sid:{k:v for k,v in s.items() if k!='mean'} for sid,s in coefficients(ds,j)[0].items()}) for j in range(3)}
            for stage in range(3):
                for arm in H1_METHODS:
                    for part in ['cal','validation']:
                        timer.check();p=packet(d,'h1','dev',part)
                        h1_predict(m,cm,f0,p,predpath('h1','dev',seed,arm,stage,part),shared,ds,stage,arm,taus[stage])
            allstates[seed]=dict(shared=shared,basis=basis,dev=ds,tau2=taus)
            del m,cm,f0;gc.collect();torch.cuda.empty_cache()
        selection=select('h1',H1_METHODS,CONFIG['h1']['seeds'])
        save(RESULTS/'H1_DEV_VARIANCE.json',taureceipt)
        cals={};gates={}
        for seed in CONFIG['h1']['seeds']:
            state=allstates[seed];m=ForecastModel(lora=True,seed=seed);cm=CoefficientModel(state['basis']);f0=ForecastModel()
            es=clients(m,cm,d,seed,'eval',state['shared'],timer);state['eval']=es
            stats,_=coefficients(es,2,state['tau2'][2]);gates[str(seed)]={sid:{k:v for k,v in s.items() if k!='mean'} for sid,s in stats.items()}
            cals[str(seed)]={}
            for arm in H1_FINAL:
                source=selection['details']['FIXED_SHRINK']['source_method'] if arm=='FIXED_SHRINK' else arm
                p=packet(d,'h1','eval','cal')
                pred=h1_predict(m,cm,f0,p,predpath('h1','eval',seed,arm,2,'cal'),state['shared'],es,2,source,state['tau2'][2])
                cals[str(seed)][arm]=calibration(pred)
            del m,cm,f0;gc.collect();torch.cuda.empty_cache()
        torch.save(allstates,CACHE/'H1_states.pt')
        save(RESULTS/'CALIBRATION_H1.json',cals);save(RESULTS/'H1_EVAL_VARIANCE.json',gates)
        save(RESULTS/'PREPARED_H1.json',dict(selection_sha=sha(RESULTS/'SELECTION_H1.json'),calibration_sha=sha(RESULTS/'CALIBRATION_H1.json'),states_sha=sha(CACHE/'H1_states.pt'),test_scores_accessed=False))
    finally:timer.close()


def prepare_h2(d):
    if (RESULTS/'PREPARED_H2.json').exists():return
    timer=PhaseTimer('h2','train_select_and_eval_cal');cals={}
    try:
        for seed in CONFIG['h2']['seeds']:
            cals[str(seed)]={}
            for arm in H2_METHODS:
                m=ForecastModel(lora=True,seed=seed) if arm=='QV_LORA' else (ForecastModel() if arm.startswith('F0') else BiasModel(arm))
                cps=None
                if not arm.startswith('F0'):
                    cps=cached_fit(m,d,data.donor_schedule(seed,d=d),f'h2_{arm}_s{seed}',candidate='h2',datagroup='donor',
                                   lr=1e-4 if arm=='QV_LORA' else .01,seed=seed,missing=True,timer=timer)
                for stage in range(3):
                    if cps is not None:m.load_learned(cps[[0,128,256][stage]])
                    for part in ['cal','validation']:
                        timer.check();p=packet(d,'h2','dev',part,interpolate=arm=='F0_INTERP')
                        predict(m,p,predpath('h2','dev',seed,arm,stage,part))
                del m;gc.collect();torch.cuda.empty_cache()
        selection=select('h2',H2_METHODS,CONFIG['h2']['seeds'])
        for seed in CONFIG['h2']['seeds']:
            for arm in H2_METHODS:
                m=ForecastModel(lora=True,seed=seed) if arm=='QV_LORA' else (ForecastModel() if arm.startswith('F0') else BiasModel(arm))
                if not arm.startswith('F0'):
                    cps=torch.load(CACHE/'fits'/f'h2_{arm}_s{seed}'/'checkpoints.pt',weights_only=True);m.load_learned(cps[256])
                p=packet(d,'h2','eval','cal',interpolate=arm=='F0_INTERP')
                pred=predict(m,p,predpath('h2','eval',seed,arm,2,'cal'));cals[str(seed)][arm]=calibration(pred)
                del m;gc.collect();torch.cuda.empty_cache()
        save(RESULTS/'CALIBRATION_H2.json',cals)
        save(RESULTS/'PREPARED_H2.json',dict(selection_sha=sha(RESULTS/'SELECTION_H2.json'),calibration_sha=sha(RESULTS/'CALIBRATION_H2.json'),test_scores_accessed=False))
    finally:timer.close()


def seal():
    if (RESULTS/'ALL_SELECTIONS_SEALED.json').exists():
        check_seal()
        return
    files=['SELECTION_H1.json','CALIBRATION_H1.json','PREPARED_H1.json','SELECTION_H2.json','CALIBRATION_H2.json','PREPARED_H2.json','REFERENCE_SELECTION.json']
    states=[CACHE/'H1_states.pt']+sorted((CACHE/'fits').glob('h2_*/checkpoints.pt'))
    save(RESULTS/'ALL_SELECTIONS_SEALED.json',dict(files={n:sha(RESULTS/n) for n in files},
         states={str(p.relative_to(ROOT)):sha(p) for p in states},source_hashes=source_hashes(),test_scores_accessed=False))
    event('all_selections_sealed',test_scores_accessed=False)


def check_seal():
    receipt=read(RESULTS/'ALL_SELECTIONS_SEALED.json')
    verify_sources(receipt['source_hashes'])
    for name,expected in receipt['files'].items():assert sha(RESULTS/name)==expected
    for name,expected in receipt['states'].items():assert sha(ROOT/name)==expected


def verify_sources(expected):
    for name,digest in expected.items():assert sha(EXP/name)==digest, f'Changed sealed source: {name}'


def test_h1(d):
    timer=PhaseTimer('h1','sealed_test_forecasts')
    try:
        check_seal();states=torch.load(CACHE/'H1_states.pt',weights_only=True)
        selection=read(RESULTS/'SELECTION_H1.json')
        for seed,state in states.items():
            m=ForecastModel(lora=True,seed=seed);cm=CoefficientModel(state['basis']);f0=ForecastModel()
            for arm in H1_FINAL:
                timer.check();source=selection['details']['FIXED_SHRINK']['source_method'] if arm=='FIXED_SHRINK' else arm
                h1_predict(m,cm,f0,packet(d,'h1','eval','test'),predpath('h1','eval',seed,arm,2,'test'),
                           state['shared'],state['eval'],2,source,state['tau2'][2])
            del m,cm,f0;gc.collect();torch.cuda.empty_cache()
    finally:timer.close()


def test_h2(d):
    timer=PhaseTimer('h2','sealed_test_forecasts')
    try:
        check_seal()
        for seed in CONFIG['h2']['seeds']:
            for arm in H2_METHODS:
                m=ForecastModel(lora=True,seed=seed) if arm=='QV_LORA' else (ForecastModel() if arm.startswith('F0') else BiasModel(arm))
                if not arm.startswith('F0'):
                    cps=torch.load(CACHE/'fits'/f'h2_{arm}_s{seed}'/'checkpoints.pt',weights_only=True);m.load_learned(cps[256])
                for condition in ['BLOCK48','IID48','ALIGNED48_LAST','CLEAN']:
                    timer.check();p=packet(d,'h2','eval','test',condition,arm=='F0_INTERP')
                    predict(m,p,predpath('h2','eval',seed,arm,2,'test_'+condition))
                del m;gc.collect();torch.cuda.empty_cache()
    finally:timer.close()


def main():
    assert read(RESULTS/'SMOKE_COMPLETE.json')['status']=='PASS'
    if (RESULTS/'RUN_COMPLETE.json').exists():raise RuntimeError('Already complete: no automatic new work')
    p0=read(RESULTS/'P0_COMPLETE.json')
    assert p0['h1']==p0['h2']=='PASS_PROBLEM_GATE' and p0['native_parity']
    if (RESULTS/'MAIN_START.json').exists():
        prior=read(RESULTS/'MAIN_START.json')
        verify_sources(prior['source_hashes'])
        assert prior['design_sha']==sha(DESIGN_DIR/'DESIGN.json')
    if (RESULTS/'ALL_SELECTIONS_SEALED.json').exists():check_seal()
    setup();d=data.load_data()
    if not (RESULTS/'MAIN_START.json').exists():save(RESULTS/'MAIN_START.json',dict(source_hashes=source_hashes(),design_sha=sha(DESIGN_DIR/'DESIGN.json')))
    prepare_h1(d);prepare_h2(d)
    from reference import prepare_reference,test_reference
    prepare_reference(d);seal()
    test_h1(d);test_h2(d);test_reference(d)
    save(RESULTS/'RUN_COMPLETE.json',dict(status='COMPLETE',ledger=read(RESULTS/'OPTIMIZER_LEDGER.json')['counts'],source_hashes=source_hashes(),test_predictions_complete=True))
    event('run_complete',counts=read(RESULTS/'OPTIMIZER_LEDGER.json')['counts'])


if __name__=='__main__':
    try:main()
    except Exception as e:
        event('runner_error',error=repr(e),traceback=traceback.format_exc())
        save(RESULTS/'RUN_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
