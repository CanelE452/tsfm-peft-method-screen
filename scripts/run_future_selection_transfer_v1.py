"""Approved one-shot RECENT4/SPREAD4 comparison: frozen checkpoint reuse, no fits."""
import gc
import gzip
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_temporal_transfer_gradient_v1 as G
import run_temporal_transfer_diagnostic_v1 as A
from tsfm_peft_screen.backbone import MODEL_ID, REVISION
from tsfm_peft_screen.reassessment import DiagnosticModel
from tsfm_peft_screen.metrics import score, independent
from tsfm_peft_screen.reproducibility import ROOT, sha, digest, write_json, seed_all, guard

OUT=ROOT/'results/future_selection_transfer_v1'
CACHE=ROOT/'.cache/future_selection_transfer_v1'
OLD=ROOT/'results/anchor_window_study_20260914'
RANGES={'beijing':{'S':[18432,19872,96],'D':[20992,23968,96]},
        'electricity_new':{'S':[18432,19872,96],'D':[20992,23968,96]},
        'ettm2_later':{'S':[45056,46496,96],'D':[47616,50592,96]}}
RULES={'RECENT4':[12,13,14,15],'SPREAD4':[0,5,10,15],'ALL16':list(range(16))}


def read(p):return json.loads(Path(p).read_text())
def oo(r):return list(range(r[0],r[1]+1,r[2]))


def selection(candidates, losses):
    return min(candidates,key=lambda c:(losses[c['id']],c['step'],c['lr'],c['id']))


def assert_seal():
    seal=read(OUT/'selection_seal.json')
    assert digest({k:v for k,v in seal.items() if k!='sha256'})==seal['sha256']
    assert len(seal['selections'])==12 and seal['D_forward_before_seal']==0
    assert sha(OUT/'plan.json')==seal['plan_sha256']
    assert sha(OUT/'selection_scores.csv')==seal['selection_scores_sha256']
    for r in seal['selections']:
        for c in r['selected'].values():
            assert sha(ROOT/c['checkpoint_file'])==c['checkpoint_sha256']
    return seal


def load_values(source, split, contract):
    if split=='D':assert_seal()
    else:assert split=='S'
    spec=contract['config']['data'][source];path=ROOT/spec['raw_path']
    assert sha(path)==spec['raw_sha256']
    n=RANGES[source][split][1]+48
    if source=='beijing':
        with zipfile.ZipFile(path) as z:
            df=pd.read_csv(z.open('PRSA_data_2010.1.1-2014.12.31.csv'),nrows=n)
        dates=pd.to_datetime(df[['year','month','day','hour']]);delta=pd.Timedelta(hours=1)
        values=df[spec['channels']].to_numpy(dtype=np.float32)
    elif source=='ettm2_later':
        df=pd.read_csv(path,nrows=n);dates=pd.to_datetime(df['date']);delta=pd.Timedelta(minutes=15)
        values=df[spec['channels']].to_numpy(dtype=np.float32)
    else:
        with gzip.open(path,'rt') as f:
            values=np.loadtxt(f,delimiter=',',usecols=spec['channels'],max_rows=n,dtype=np.float32)
    if source!='electricity_new':
        assert dates.is_unique and dates.is_monotonic_increasing and (dates.diff().dropna()==delta).all()
    assert values.shape==(n,4) and not np.isinf(values).any()
    tr=spec['train'];sc=np.nanstd(values[tr[0]-1024:tr[1]+48].astype(np.float64),axis=0)
    assert np.array_equal(sc,contract['inputs'][source]['train_scale'])
    assert RANGES[source]['S'][1]+48<=RANGES[source]['D'][0]-1024
    targets=np.stack([values[o:o+48].T for o in oo(RANGES[source][split])])
    assert np.isfinite(targets).sum((0,2)).min()>0
    return values,sc


def audit_exposure(contract):
    """Inspect origin metadata only; never read historical forecast/target arrays."""
    records=[];unknown=[];overlap=[]
    for folder in ['.cache','data/processed']:
        for p in sorted((ROOT/folder).rglob('*.npz')):
            name=str(p.relative_to(ROOT))
            if not any(s in name.lower() for s in ['beijing','ettm2','electricity']):continue
            source='beijing' if 'beijing' in name.lower() else ('ettm2_later' if 'ettm2' in name.lower() else 'electricity_new')
            different_channels=source=='electricity_new' and 'electricity_new' not in name and 'anchor_window_study_20260914' not in name
            with np.load(p,allow_pickle=False) as f:
                keys=[k for k in f.files if 'origin' in k]
                for key in keys:
                    origins=np.asarray(f[key])
                    if origins.dtype.kind not in 'iu' or origins.ndim!=1:
                        unknown.append(dict(path=name,key=key));continue
                    records.append(dict(path=name,key=key,n=len(origins),minimum=int(origins.min()) if len(origins) else None,
                        maximum=int(origins.max()) if len(origins) else None,source=source,different_electricity_channels=different_channels))
                    for split in ['S','D']:
                        new=oo(RANGES[source][split])
                        hits=[int(o) for o in origins if any(int(o)<v+48 and int(o)+48>v for v in new)]
                        if hits and not different_channels:overlap.append(dict(path=name,key=key,split=split,hits=hits))
    # Earlier same-channel fitted ranges and forecast maxima are bounded by the original contract.
    for source,spec in contract['config']['data'].items():
        assert spec['evaluation'][1]+48<=RANGES[source]['S'][0]-1024
    assert read(ROOT/'data/processed/electricity/manifest.json')['channels']==[f'column_{i:03}' for i in range(4)]
    return dict(status='BLOCKED' if overlap or unknown else 'PASS',origin_metadata=records,overlaps=overlap,unknown=unknown,
        scope='This checkout prediction-origin metadata, source/config history and original window-study ranges; pretraining and external experiments unknown. Raw schema/missingness was mechanically inspected earlier; unused means not scored/trained on these target series/time windows here, not never read as raw bytes.',
        electricity_caveat='Historical Electricity columns 0..3 were trained/evaluated at overlapping dates. Current columns 4..7 are different series, not a new independent source. Earlier loaders inspected: data.py, run_block_shape_pilot.py and original staged manifest.')


class Budget(G.Budget):
    def __init__(self):
        super().__init__()
        self.count=dict(forward=0,S_forward=0,D_forward=0,backward=0,new_fits=0,optimizer_updates=0,
                        cache_hits=0,old_E_array_reads=0,D_reads_before_seal=0)
    def call(self,split):
        self.resource(split)
        assert self.count['forward']<1536 and time.monotonic()-self.start<14400
        self.count['forward']+=1;self.count[split+'_forward']+=1;self.save(phase=split)


def main():
    assert not OUT.exists() and not CACHE.exists(),'No overwrite/retry'
    contract=read(OLD/'contract.json')
    audit=audit_exposure(contract)
    OUT.mkdir();CACHE.mkdir();G.OUT=OUT
    write_json(OUT/'exposure_audit.json',audit)
    assert audit['status']=='PASS','Do not score exposed or unresolved windows'
    from huggingface_hub import hf_hub_download
    for name,h in contract['model_files'].items():assert sha(hf_hub_download(MODEL_ID,name,revision=REVISION,local_files_only=True))==h
    cells=[]
    for source in RANGES:
        for seed in [34000,34001]:
            for arm in ['native','native_anchor']:
                sid=f'{source}_233_{seed}_{arm}';candidates=[]
                for recipe in [0,1]:
                    for row in read(OLD/f'anchor/{sid}_{recipe}_trajectory.json'):
                        assert sha(ROOT/row['checkpoint_file'])==row['checkpoint_sha256']
                        if row['step']==0 and recipe==1:continue
                        candidates.append({**{k:row[k] for k in ['checkpoint_file','checkpoint_sha256','step','lr']},
                            'id':row['fit_id']+'_step'+str(row['step'])})
                assert len(candidates)==7
                cells.append(dict(id=sid,source=source,seed=seed,arm=arm,candidates=candidates))
    plan=dict(authorization='User: 그렇게 해줘 (after explicit proposal of this future-window comparison)',
        parent_proposal_sha256=sha(ROOT/'results/temporal_transfer_diagnostic_v1/next_experiment_proposal.md'),
        ranges=RANGES,rules=RULES,cells=cells,model_revision=REVISION,context=1024,horizon=48,
        candidates_per_cell=7,total_candidate_states=84,train_scale='Original train-only scale, byte-exact recheck',
        precision='Original BF16 autocast with FP32 parameters; no training',max_forward=1536,
        max_backward=0,max_fits=0,max_optimizer_updates=0,wall_cap_seconds=14400,
        resource_guard='Existing stricter 2h guard retained; exact RustDesk <=512MiB exception, startup free>=4GiB/util<=35%, ongoing free>=1GiB',
        tie_rule='minimum S score, earlier step, lower LR, candidate ID',
        primary='Paired D loss RECENT4 minus SPREAD4 per source, and source-balanced difference in percent gain versus F0. Positive favors SPREAD4. Seeds and arms are repeats within source.',
        stop='No universal PASS cutoff, no post-D selector tuning, no new PEFT structure; D only union of selected/F0/fixed reference states, never full candidate oracle.')
    write_json(OUT/'plan.json',plan)
    history=G.snapshot_history();write_json(OUT/'source_and_history_hashes.json',history)
    budget=Budget();results={};selections=[];cache_index=[];selection_scores=[]
    try:
        for k in range(4):
            budget.resource(f'startup_{k}',startup=True)
            if k<3:time.sleep(10)
        def forecast(model,cell,c,split,values,sc):
            canonical=cell['source']+'_F0' if c['step']==0 else c['id']
            key=(canonical,split)
            if key in results:
                budget.count['cache_hits']+=1;return results[key]
            if split=='D':assert_seal()
            params={n:p for n,p in model.named_parameters() if p.requires_grad}
            weights=torch.load(ROOT/c['checkpoint_file'],map_location='cpu',weights_only=True)
            if c['step']==0:
                assert all(torch.count_nonzero(v)==0 for n,v in weights.items() if n.endswith('lora_B'))
            G.restore(params,weights)
            before=G.tensor_hash(params)
            origins=oo(RANGES[cell['source']][split]);pred=[]
            with torch.no_grad():
                for start in range(0,len(origins),2):
                    subset=origins[start:start+2]
                    x=np.concatenate([values[o-1024:o].T for o in subset]).astype(np.float32)
                    x=torch.from_numpy(x).cuda();groups=torch.arange(len(subset),device='cuda').repeat_interleave(4)
                    budget.call(split)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        p=model(x,groups,frozen=c['step']==0)[1]
                    assert torch.isfinite(p).all()
                    pred.append(p.detach().cpu().numpy().reshape(len(subset),4,21,48))
                    del p,x,groups
            assert before==G.tensor_hash(params) and sha(ROOT/c['checkpoint_file'])==c['checkpoint_sha256']
            pred=np.concatenate(pred);target=np.stack([values[o:o+48].T for o in origins])
            loss=score(pred,target,sc)['scaled_2pinball'];scalar=independent(pred,target,sc);assert abs(loss-scalar)<=1e-10
            path=CACHE/f'{canonical}_{split}.npz'
            np.savez_compressed(path,prediction=pred,target=target,scale=sc,origins=np.array(origins))
            num,count=A.components(pred,target,sc)
            r=dict(canonical=canonical,split=split,loss=loss,num=num,count=count)
            results[key]=r
            cache_index.append(dict(canonical=canonical,source=cell['source'],split=split,prediction_file=str(path.relative_to(ROOT)),
                prediction_sha256=sha(path),loss=loss,metric_replay_error=abs(loss-scalar),checkpoint_file=c['checkpoint_file'],checkpoint_sha256=c['checkpoint_sha256']))
            write_json(OUT/'prediction_index.json',cache_index)
            return r
        # Complete and seal ALL source/seed/arm selections before reading any D targets.
        for cell in cells:
            seed_all(cell['seed']);model=DiagnosticModel('anchor',cell['arm'],cell['seed'])
            frozen=lambda:G.tensor_hash({n:p for n,p in model.named_parameters() if not p.requires_grad})
            frozen_before=frozen()
            values,sc=load_values(cell['source'],'S',contract)
            candidates=cell['candidates'];data={c['id']:forecast(model,cell,c,'S',values,sc) for c in candidates}
            chosen={'F0':next(c for c in candidates if c['step']==0),'FIXED':next(c for c in candidates if c['step']==150 and c['lr']==3e-5)}
            for rule,idx in RULES.items():
                losses={c['id']:A.pooled(data[c['id']]['num'],data[c['id']]['count'],idx) for c in candidates}
                chosen[rule]=selection(candidates,losses)
                selection_scores.extend(dict(cell=cell['id'],rule=rule,candidate=c['id'],loss=losses[c['id']]) for c in candidates)
            selections.append(dict(cell=cell['id'],source=cell['source'],seed=cell['seed'],arm=cell['arm'],selected=chosen))
            assert frozen()==frozen_before
            del model,values,data;gc.collect();torch.cuda.empty_cache()
            budget.save(phase='S',cells_selected=len(selections))
            print(json.dumps(dict(phase='S',cells=len(selections),forward=budget.count['forward'])),flush=True)
        G.csv_write(OUT/'selection_scores.csv',selection_scores)
        seal=dict(plan_sha256=sha(OUT/'plan.json'),selections=selections,selection_scores_sha256=sha(OUT/'selection_scores.csv'),D_forward_before_seal=budget.count['D_forward'],S_forward=budget.count['S_forward'])
        seal['sha256']=digest(seal);write_json(OUT/'selection_seal.json',seal)
        evaluation=[]
        for chosen in assert_seal()['selections']:
            cell=next(c for c in cells if c['id']==chosen['cell'])
            seed_all(cell['seed']);model=DiagnosticModel('anchor',cell['arm'],cell['seed'])
            frozen=lambda:G.tensor_hash({n:p for n,p in model.named_parameters() if not p.requires_grad})
            frozen_before=frozen();values,sc=load_values(cell['source'],'D',contract)
            for rule,c in chosen['selected'].items():
                r=forecast(model,cell,c,'D',values,sc)
                evaluation.append(dict(cell=cell['id'],source=cell['source'],seed=cell['seed'],arm=cell['arm'],rule=rule,candidate=c['id'],canonical=r['canonical'],loss=r['loss']))
            assert frozen()==frozen_before
            del model,values;gc.collect();torch.cuda.empty_cache()
            G.csv_write(OUT/'evaluation.csv',evaluation)
            budget.save(phase='D',cells_evaluated=len(evaluation)//5)
            print(json.dumps(dict(phase='D',cells=len(evaluation)//5,forward=budget.count['forward'])),flush=True)
        assert all(sha(ROOT/p)==h for p,h in history.items())
        budget.save('COMPLETED',historical_files_unchanged=len(history),resources=guard(budget.start),
            selection_seal_sha256=sha(OUT/'selection_seal.json'),prediction_caches=len(cache_index),automatic_followup=False,git_push=False)
    except BaseException as e:
        budget.save('INTERRUPTED',error=repr(e));raise


if __name__=='__main__':main()
