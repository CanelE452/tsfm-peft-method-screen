import gc,sys,time,traceback
import numpy as np
import torch
from common import *
from model import ModelA,ModelB,setup,tensor_hash
from engine import fit,tensors
from evaluation import predict,path,loadpred,score,calibrate
from A.data import Solar
from bdata_adapter import Wind

def seal_sources(c):
    p=RESULTS/c/'SOURCE_SEAL.json'
    if p.exists():verify_sources(read(p)['sources']);return
    assert read(RESULTS/c/'MODEL_SMOKE_AUDIT.json')['status']=='PASS'
    receipts=['DATA_AUDIT.json','MODEL_SMOKE_AUDIT.json','TRAINING_BUDGET.json','INFORMATION_CONTRACT.json']
    save(p,dict(sources=sources(c),receipts={n:sha(RESULTS/c/n) for n in receipts},master_sha=sha(EXP/'contract/MASTER_CLI.txt'),main_updates_before=0))
    event('source_sealed',candidate=c)

def check_source(c):
    r=read(RESULTS/c/'SOURCE_SEAL.json');verify_sources(r['sources'])
    for n,h in r['receipts'].items():assert sha(RESULTS/c/n)==h

def fitted(model,c,key,packets,validation):
    p=CACHE/c/'fits'/key/'states.pt';receipt=RESULTS/c/'fits'/f'{key}.json'
    if p.exists() and receipt.exists():
        report=read(receipt);assert read(RESULTS/c/'BUDGET_STATE.json')['fits'][key]['status']=='COMPLETE'
        assert report['initial_sha']==tensor_hash(model.learned().items())
        return torch.load(p,weights_only=True),report
    return fit(model,c,key,packets,validation=validation)

def validation_callback(packet,c,seed,arm):
    steps=iter([0,128,256])
    def call(m):
        step=next(steps);pred=predict(m,packet,c,path(c,seed,arm,'VAL',step));return score(pred,c)
    return call

def choose(report):return min((int(k) for k in report['curves']),key=lambda k:(report['curves'][str(k)] if str(k) in report['curves'] else report['curves'][k],k))

def selected_state(c,seed,arm):
    selection=read(RESULTS/c/'MODEL_SELECTION.json');step=selection[str(seed)][arm]['checkpoint']
    states=torch.load(CACHE/c/'fits'/f'{arm}_{seed}'/'states.pt',weights_only=True)
    assert tensor_hash(states[step].items())==selection[str(seed)][arm]['selected_state_sha']
    return states[step]

def prepare_a():
    c='A';check_source(c);d=Solar();r=RESULTS/c
    if (r/'CALIBRATION.json').exists():return
    selections={};cal={};cache_manifest=[]
    for seed in SEEDS:
        selections[str(seed)]={};cal[str(seed)]={};schedule=d.schedule(seed)
        unique,inverse=np.unique(schedule.reshape(-1,2),axis=0,return_inverse=True);train=d.batch(unique)
        m=ModelA('FULL9',seed);start=time.perf_counter();first=[]
        with torch.no_grad():
            for i in range(0,len(unique),8):first.append(m.first(torch.as_tensor(train['x'][i:i+8],device='cuda')).cpu().numpy())
        first=np.concatenate(first);firstpath=CACHE/c/f'first_TRAIN_{seed}.npz'
        np.savez_compressed(firstpath,first=first,pairs=unique)
        cache_manifest.append(dict(kind='first_F0',role='TRAIN',seed=seed,path=firstpath,sha256=sha(firstpath),examples=len(unique),seconds=time.perf_counter()-start,bytes=firstpath.stat().st_size))
        packets=[]
        for step in range(256):
            b=d.batch(schedule[step]);b['first']=first[inverse.reshape(256,4)[step]];packets.append(b)
        val=d.batch(d.pairs['VAL']);cals=d.batch(d.pairs['CAL'])
        cps,receipt=fitted(m,c,f'FULL9_{seed}',packets,validation_callback(val,c,seed,'FULL9'));chosen=choose(receipt);m.load_learned(cps[chosen]);teacher_state=cps[chosen]
        selections[str(seed)]['FULL9']=dict(checkpoint=chosen,raw_validation=receipt['curves'],selected_state_sha=tensor_hash(teacher_state.items()))
        for role,packet in [('CAL',cals),('VAL',val)]:predict(m,packet,c,path(c,seed,'FULL9',role))
        cal[str(seed)]['FULL9']=calibrate(loadpred(path(c,seed,'FULL9','CAL')),c)
        start=time.perf_counter();teacher=predict(m,{**train,'first':first},c)
        teacherfile=CACHE/c/f'teacher_TRAIN_{seed}.npz';np.savez_compressed(teacherfile,z=teacher['z'][:,64:],p=teacher['p'][:,64:],pairs=unique)
        cache_manifest.append(dict(kind='selected_same_seed_teacher',role='TRAIN',seed=seed,checkpoint=chosen,path=teacherfile,sha256=sha(teacherfile),examples=len(unique),seconds=time.perf_counter()-start,bytes=teacherfile.stat().st_size))
        for step,b in enumerate(packets):
            ix=inverse.reshape(256,4)[step];b['teacher_z']=teacher['z'][ix,64:];b['teacher_p']=teacher['p'][ix,64:]
        del m;gc.collect();torch.cuda.empty_cache()
        for arm in ARMS_A[1:]:
            m=ModelA(arm,seed);m.load_lora(teacher_state)
            cps,receipt=fitted(m,c,f'{arm}_{seed}',packets,validation_callback(val,c,seed,arm));chosen=choose(receipt);m.load_learned(cps[chosen])
            selections[str(seed)][arm]=dict(checkpoint=chosen,raw_validation=receipt['curves'],selected_state_sha=tensor_hash(cps[chosen].items()),teacher_checkpoint=selections[str(seed)]['FULL9']['checkpoint'])
            for role,packet in [('CAL',cals),('VAL',val)]:predict(m,packet,c,path(c,seed,arm,role))
            cal[str(seed)][arm]=calibrate(loadpred(path(c,seed,arm,'CAL')),c)
            del m;gc.collect();torch.cuda.empty_cache()
    save(r/'TEACHER_CACHE_MANIFEST.json',dict(caches=cache_manifest,teacher_test_used_in_training=False,all_reduced_arms_same_teacher_and_packets=True))
    save(r/'MODEL_SELECTION.json',selections);save(r/'CALIBRATION.json',cal)

def prepare_b():
    c='B';check_source(c);d=Wind();r=RESULTS/c
    if (r/'CALIBRATION.json').exists():return
    selections={};cal={};val=d.packet('VAL');cals=d.packet('CAL')
    for seed in SEEDS:
        selections[str(seed)]={};cal[str(seed)]={};packets=[d.batch(v) for v in d.schedule(seed)]
        for arm in ARMS_B:
            m=ModelB(arm,d.dim,seed);cps,receipt=fitted(m,c,f'{arm}_{seed}',packets,validation_callback(val,c,seed,arm))
            chosen=choose(receipt);m.load_learned(cps[chosen]);selections[str(seed)][arm]=dict(checkpoint=chosen,raw_validation=receipt['curves'],selected_state_sha=tensor_hash(cps[chosen].items()))
            for role,packet in [('CAL',cals),('VAL',val)]:predict(m,packet,c,path(c,seed,arm,role))
            cal[str(seed)][arm]=calibrate(loadpred(path(c,seed,arm,'CAL')),c)
            del m;gc.collect();torch.cuda.empty_cache()
    save(r/'MODEL_SELECTION.json',selections);save(r/'CALIBRATION.json',cal)

def baseline(c):
    from references import prepare
    prepare(c)
    cal=read(RESULTS/c/'CALIBRATION.json');methods=ARMS_A[1:-1] if c=='A' else ['CONTROL','MOMENTS','SET','MEMBER','GBQR']
    scores={}
    for arm in methods:
        values=[]
        for seed in SEEDS:
            s=0 if arm=='GBQR' else seed
            values.append(score(loadpred(path(c,s,arm,'VAL')),c,cal[str(s)][arm]))
        scores[arm]=float(np.mean(values))
    winner=min(methods,key=lambda a:scores[a]);save(RESULTS/c/'BASELINE_SELECTION.json',dict(baseline=winner,validation_scores=scores,selection_only_seeds=[],repeat_seeds=SEEDS))

def selection_seal():
    p=RESULTS/'ALL_SELECTIONS_SEALED.json'
    if p.exists():check_selection();return
    candidates=read(RESULTS/'EXECUTION_PLAN.json')['completed_candidates']
    files=[RESULTS/c/n for c in candidates for n in ['MODEL_SELECTION.json','CALIBRATION.json','BASELINE_SELECTION.json','SOURCE_SEAL.json']]+[RESULTS/'EXECUTION_PLAN.json']
    save(p,dict(files={str(f.relative_to(RESULTS)):sha(f) for f in files},test_prediction_started=False))
    event('all_selections_sealed')

def check_selection():
    for c in read(RESULTS/'EXECUTION_PLAN.json')['completed_candidates']:check_source(c)
    for name,h in read(RESULTS/'ALL_SELECTIONS_SEALED.json')['files'].items():assert sha(RESULTS/name)==h

def test_predictions(c):
    check_selection();d=Solar() if c=='A' else Wind();packet=d.batch(d.pairs['TEST']) if c=='A' else d.packet('TEST')
    for seed in SEEDS:
        for arm in (ARMS_A if c=='A' else ARMS_B):
            m=ModelA(arm,seed) if c=='A' else ModelB(arm,d.dim,seed);m.load_learned(selected_state(c,seed,arm))
            predict(m,packet,c,path(c,seed,arm,'TEST'));del m;gc.collect();torch.cuda.empty_cache()
    from references import test
    test(c,d,packet)
    save(RESULTS/c/'PREDICTIONS_MANIFEST.json',dict(files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted((CACHE/c/'predictions').rglob('*.npz'))},all_test_predictions_saved=True,test_scoring_started=False))

def main():
    setup()
    assert read(RESULTS/'PRE_MAIN_AUDIT.json')['status']=='PASS'
    completed=[];blocked={}
    for c,prepare in [('A',prepare_a),('B',prepare_b)]:
        try:
            assert read(RESULTS/c/'DATA_AUDIT.json')['status']=='PASS','BLOCKED_DATA'
            seal_sources(c);prepare();baseline(c);completed.append(c)
        except Exception as e:
            category='RESOURCE_BLOCK' if 'out of memory' in str(e).lower() else ('BLOCKED_DATA' if 'BLOCKED' in str(e) else 'IMPLEMENTATION_FAILURE')
            blocked[c]=dict(classification=category,error=repr(e),traceback=traceback.format_exc())
            save(RESULTS/c/'BLOCKED.json',blocked[c]);event('candidate_blocked',candidate=c,category=category)
            gc.collect();torch.cuda.empty_cache()
    save(RESULTS/'EXECUTION_PLAN.json',dict(completed_candidates=completed,blocked=blocked,no_budget_transfer=True))
    selection_seal()
    for c in completed:test_predictions(c)
    save(RESULTS/'ALL_TEST_PREDICTIONS_COMPLETE.json',dict(status='COMPLETE',test_scoring_started=False))
    from benchmark import run
    run()
    save(RESULTS/'RUN_COMPLETE.json',dict(status='COMPLETE',automatic_followup=False))

if __name__=='__main__':
    try:
        with executor_lock():main()
    except Exception as e:
        save(RESULTS/'RUN_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),classification='RESOURCE_BLOCK' if 'out of memory' in str(e).lower() else 'IMPLEMENTATION_FAILURE'))
        raise
