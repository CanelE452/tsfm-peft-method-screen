import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gc
import time
import traceback
import warnings
import numpy as np
import pandas as pd
import torch
from common import *
from model import *
from ledger import Ledger,restore
from smoke import batch
from data import load,schedule
from evaluation import scaled_pinball,fit_affine,apply_affine,score_tables,bootstrap_effect

SOURCES=['Electricity','ETTh1']
SEEDS=[92121,92122]
LRS=[1e-4,3e-4]

def fit_id(source,arm,seed,lr):
    return f'{source}_{arm}_s{seed}_lr{lr:g}'

def source_hashes():
    return {str(f.relative_to(ROOT)):sha(f) for f in sorted(EXP.rglob('*.py'))}

def panel(d,role):
    origins=d['origins'][role]
    x=np.stack([d['values'][o-512:o].T for o in origins])
    return x.reshape(-1,512)

def targets(d,role):
    return np.stack([d['values'][o:o+256].T for o in d['origins'][role]])

@torch.no_grad()
def predictions(model,d,source,role,key,kind='single',sampling_seed=None):
    if role=='TEST':
        seal=read_json(RESULTS/'ALL_SELECTIONS_SEALED.json')
        assert seal['model_selection_sha256']==sha(RESULTS/'MODEL_SELECTION.json')
        assert seal['calibration_sha256']==sha(RESULTS/'CALIBRATION_PARAMETERS.json')
    folder=CACHE/'predictions'/source/role; folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{key}.npy'; receipt=folder/f'{key}.json'
    if receipt.exists():
        assert sha(path)==read_json(receipt)['sha256']
        return np.load(path,mmap_mode='r')
    xs=panel(d,role); nseries=len(d['sigma'])
    sigmas=np.tile(d['sigma'],len(d['origins'][role]))
    bs=64 if kind=='single' else (32 if kind=='direct' else 8)
    out=np.lib.format.open_memmap(path.with_suffix('.partial.npy'),mode='w+',dtype='float32',shape=(len(xs),256,9))
    rng=np.random.default_rng(sampling_seed) if kind=='mc16' else None
    elapsed=time.perf_counter(); crosses=[]
    for j in range(0,len(xs),bs):
        x=torch.as_tensor(xs[j:j+bs],device='cuda',dtype=torch.float32)
        sigma=torch.as_tensor(sigmas[j:j+bs],device='cuda',dtype=torch.float32)
        if kind=='single':
            _,raw=rollout(model,x,sigma)
        elif kind=='native':
            trace=[]; raw=native(model,x,trace)
            crosses.extend([{k:v for k,v in t.items() if k!='quantiles'} for t in trace])
        elif kind=='mc16':
            u=torch.from_numpy(rng.random((len(x),4,16,64)).astype('float32')).cuda()
            raw,cross=mc16(model,x,u)
            crosses.append({'batch':len(x),'conditional_crossing_rates':cross})
        else:
            raw=direct(model,x)
        assert torch.isfinite(raw).all()
        out[j:j+len(x)]=raw.cpu().numpy()
        if j%(bs*20)==0:
            save_json(RESULTS/'PROGRESS.json',{'stage':'prediction','source':source,'role':role,'model':key,'completed':j+len(x),'total':len(xs),'elapsed_seconds':time.perf_counter()-elapsed})
    out.flush(); del out
    # Restore panel dimensions in the saved raw prediction file.
    partial=path.with_suffix('.partial.npy')
    flat=np.load(partial,mmap_mode='r')
    np.save(path,flat.reshape(len(d['origins'][role]),nseries,256,9))
    del flat; partial.unlink()
    info={'source':source,'role':role,'model':key,'kind':kind,'path':str(path.relative_to(ROOT)),'sha256':sha(path),'shape':[len(d['origins'][role]),nseries,256,9],'batch':bs,'seconds':time.perf_counter()-elapsed,'sampling_seed':sampling_seed,'conditional_crossing':crosses,'raw_before_final_sort':True}
    save_json(receipt,info)
    event('predictions_saved',source=source,role=role,model=key,seconds=info['seconds'])
    return np.load(path,mmap_mode='r')

def validation(model,d,source,key):
    raw=predictions(model,d,source,'VALIDATION',key)
    return scaled_pinball(targets(d,'VALIDATION'),np.sort(raw,axis=-1),d['sigma'])

def fit(source,arm,seed,lr):
    ident=fit_id(source,arm,seed,lr)
    complete=RESULTS/'fits'/f'{ident}.json'
    if complete.exists(): return read_json(complete)
    d=load(source); packet=schedule(source,seed)
    model=RolloutModel(arm,seed)
    frozen=model.frozen_hash(); initial_hash=tensor_hash(model.learned().items())
    lora_hash=tensor_hash([(n,v) for n,v in model.learned().items() if not n.startswith('adapter.')])
    adapter_hash=tensor_hash([(n,v) for n,v in model.learned().items() if n.startswith('adapter.')])
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    ledger=Ledger(ident,'main',model,opt,{'source':source,'arm':arm,'seed':seed,'lr':lr,'source_hashes':source_hashes()})
    recovered_updates=ledger.step
    vals_path=ledger.folder/'validation.json'
    vals=read_json(vals_path) if vals_path.exists() else {}
    started=time.perf_counter()
    # Recover validation from saved checkpoints without repeating spent updates.
    if ledger.step>0:
        for step in [0,128,256,512]:
            checkpoint=ledger.folder/f'step_{step:04d}.pt'
            if step<=ledger.step and str(step) not in vals and checkpoint.exists():
                restore(checkpoint,model)
                score=validation(model,d,source,f'{ident}_step{step}')
                vals[str(step)]={'scaled_pinball':score,'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha(checkpoint)}
                save_json(vals_path,vals)
                event('recovered_checkpoint_validation',fit_id=ident,step=step,scaled_pinball=score)
        restore(ledger.latest,model,opt)
    while True:
        if ledger.step in [0,128,256,512] and str(ledger.step) not in vals:
            checkpoint=ledger.checkpoint()
            score=validation(model,d,source,f'{ident}_step{ledger.step}')
            vals[str(ledger.step)]={'scaled_pinball':score,'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha(checkpoint)}
            save_json(vals_path,vals); event('validation',fit_id=ident,step=ledger.step,scaled_pinball=score)
        if ledger.step==512: break
        bx,by,bs=batch(d,packet[ledger.step])
        metrics=ledger.update(lambda:train_update(model,opt,bx,by,bs))
        if ledger.step%32==0:
            event('fit_progress',fit_id=ident,step=ledger.step,loss=metrics['loss'],elapsed_seconds=time.perf_counter()-started)
            save_json(RESULTS/'PROGRESS.json',{'stage':'training','fit_id':ident,'step':ledger.step,'total':512,'elapsed_seconds':time.perf_counter()-started})
    assert model.frozen_hash()==frozen
    best=min(vals,key=lambda s:(vals[s]['scaled_pinball'],int(s)))
    record={'fit_id':ident,'source':source,'arm':arm,'seed':seed,'lr':lr,'main_updates':512,'conditional_batch_forwards_backwards':2048,'validation':vals,'selected_step':int(best),'selected_validation':vals[best]['scaled_pinball'],'checkpoint':vals[best]['checkpoint'],'checkpoint_sha256':vals[best]['checkpoint_sha256'],'initial_learned_hash':initial_hash,'initial_lora_hash':lora_hash,'initial_adapter_hash':adapter_hash,'frozen_weights_buffers_unchanged':True,'elapsed_this_process_seconds':time.perf_counter()-started}
    record['recovered_updates_without_replay']=recovered_updates
    record['frozen_digest_scope']='recovery model only; see PRESEAL_EXECUTION_RECOVERY.json' if recovered_updates==512 else 'before and after fit in this process'
    save_json(complete,record)
    del model,opt,ledger; gc.collect(); torch.cuda.empty_cache()
    return record

def train_all():
    assert read_json(RESULTS/'MODEL_AND_SMOKE_AUDIT.json')['status']=='PASS'
    assert sha(EXP/'contract/MASTER_CLI.txt')=='8b4a50ed2ca3215b4a2b9ea45cada6e3e7d6febef17a72b55f440448e826bfac'
    seal_path=RESULTS/'TRAINING_CONTRACT_SEALED.json'
    current={'master_sha256':sha(EXP/'contract/MASTER_CLI.txt'),'source_hashes':source_hashes(),'sources':SOURCES,'arms':ARMS,'selection_seed':92120,'repeat_seeds':SEEDS,'lrs':LRS,'steps':512,'checkpoints':[0,128,256,512],'main_cap':12288,'smoke_cap':12,'fits':24,'test_scored':False,'affine_tie':'squared distance from (1,0), alpha, abs(beta), signed beta','bootstrap':'noncircular moving block 7, 2000 draws, keep all series and horizons paired'}
    if seal_path.exists():
        assert read_json(seal_path)['source_hashes']==current['source_hashes'],'Source changed after seal: preserve original/diff before resuming'
    else: save_json(seal_path,current)
    all_fits=[]; lr_selection={}; selections={}
    for source in SOURCES:
        for arm in ARMS:
            trials=[fit(source,arm,92120,lr) for lr in LRS]; all_fits+=trials
            best=min(trials,key=lambda r:(r['selected_validation'],r['lr'],r['selected_step']))
            lr_selection[f'{source}/{arm}']={'lr':best['lr'],'selection_fit':best['fit_id'],'excluded_from_repeat_mean':True,'trials':[t['fit_id'] for t in trials]}
            save_json(RESULTS/'LR_SELECTION.json',lr_selection)
            for seed in SEEDS:
                result=fit(source,arm,seed,best['lr']); all_fits.append(result)
                selections[f'{source}/{arm}/{seed}']=result
            save_json(RESULTS/'MODEL_SELECTION.json',selections)
            pd.DataFrame([{k:v for k,v in r.items() if k!='validation'} for r in all_fits]).to_csv(RESULTS/'FIT_LEDGER.csv',index=False)
    assert len(all_fits)==24 and len(selections)==12
    for source in SOURCES:
        for seed in [92120,*SEEDS]:
            rows=[r for r in all_fits if r['source']==source and r['seed']==seed]
            assert len({r['initial_lora_hash'] for r in rows})==1
            assert len({r['initial_adapter_hash'] for r in rows if r['arm'] in ARMS[1:]})==1
    event('all_training_selection_complete',fits=24,main_updates=12288)

def selected_model(selection):
    m=RolloutModel(selection['arm'],selection['seed'])
    assert sha(ROOT/selection['checkpoint'])==selection['checkpoint_sha256']
    restore(ROOT/selection['checkpoint'],m)
    return m

def model_specs(selections,source):
    specs=[('F0_MEDIAN',0,'single',None),('F0_NATIVE',0,'native',None),('CHRONOS2_DIRECT',0,'direct',None)]
    for seed in SEEDS:
        for arm in ARMS: specs.append((arm,seed,'single',selections[f'{source}/{arm}/{seed}']))
        for name,kind in [('R_NATIVE','native'),('R_MC16','mc16')]: specs.append((name,seed,kind,selections[f'{source}/{ARMS[0]}/{seed}']))
    return specs

def open_model(name,kind,selection):
    if kind=='direct': return load_direct()
    if selection is not None: return selected_model(selection)
    return RolloutModel(name,92120,lora=False)

def key_for(name,seed): return f'{name}_seed{seed}'

def calibration_all():
    selections=read_json(RESULTS/'MODEL_SELECTION.json'); assert len(selections)==12
    params=read_json(RESULTS/'CALIBRATION_PARAMETERS.json') if (RESULTS/'CALIBRATION_PARAMETERS.json').exists() else {}
    for source in SOURCES:
        d=load(source)
        for name,seed,kind,selection in model_specs(selections,source):
            key=key_for(name,seed); pkey=f'{source}/{key}'
            if pkey in params: continue
            m=open_model(name,kind,selection)
            raw=predictions(m,d,source,'CALIBRATION',key,kind,1701 if seed==92121 else 1702)
            params[pkey]=fit_affine(targets(d,'CALIBRATION'),np.sort(raw,axis=-1),d['sigma'])
            save_json(RESULTS/'CALIBRATION_PARAMETERS.json',params)
            del m,raw; gc.collect(); torch.cuda.empty_cache()
    assert len(params)==26
    save_json(RESULTS/'ALL_SELECTIONS_SEALED.json',{'model_selection_sha256':sha(RESULTS/'MODEL_SELECTION.json'),'lr_selection_sha256':sha(RESULTS/'LR_SELECTION.json'),'calibration_sha256':sha(RESULTS/'CALIBRATION_PARAMETERS.json'),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'test_scores_exist':False})

def test_predictions_all():
    selections=read_json(RESULTS/'MODEL_SELECTION.json')
    for source in SOURCES:
        d=load(source)
        for name,seed,kind,selection in model_specs(selections,source):
            key=key_for(name,seed)
            if (CACHE/'predictions'/source/'TEST'/f'{key}.json').exists(): continue
            m=open_model(name,kind,selection)
            predictions(m,d,source,'TEST',key,kind,1701 if seed==92121 else 1702)
            del m; gc.collect(); torch.cuda.empty_cache()
    files=[read_json(f) for f in sorted((CACHE/'predictions').rglob('*.json'))]
    assert sum(r['role']=='TEST' for r in files)==26
    save_json(RESULTS/'PREDICTIONS_MANIFEST.json',{'selection_seal_sha256':sha(RESULTS/'ALL_SELECTIONS_SEALED.json'),'all_test_predictions_saved_before_scoring':True,'files':files})

def main(stage):
    setup(); warnings.filterwarnings('ignore',message='We recommend keeping prediction length')
    if stage in ['train','all']: train_all()
    if stage in ['calibrate','all']: calibration_all()
    if stage in ['predict','all']: test_predictions_all()
    if stage in ['resources','all']:
        from resources import measure_all
        measure_all()
    if stage in ['evaluate','all']:
        from finalize import evaluate_and_report
        evaluate_and_report()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['train','calibrate','predict','resources','evaluate','all'],default='all')
    args=p.parse_args()
    try: main(args.stage)
    except Exception as exc:
        event('execution_failure',stage=args.stage,exception=repr(exc),traceback=traceback.format_exc())
        raise
