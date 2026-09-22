import time
import numpy as np
import torch
from common import *
from engine import context_packet
from metrics import affine,point_loss
import data


def packet(d,candidate,group,part,condition=None,interpolate=False):
    if part=='test':assert (RESULTS/'ALL_SELECTIONS_SEALED.json').exists()
    choices=[('CLEAN',0,1.)] if candidate=='h1' else [('CLEAN',0,.5),('IID48',0,.25),('IID48',1,.25)]
    if condition is not None:
        choices=[(condition,r,1./2) for r in range(2)] if condition in ['IID48','BLOCK48'] else [(condition,0,1.)]
    pieces=[]
    for kind,repeat,weight in choices:
        p=context_packet(d,group,part,kind,repeat)
        if interpolate:
            p['x']=np.stack([data.interpolate_context(x,mask) for x,mask in zip(p['x'],p['masks'])])
        p['weights']=np.full(len(p['x']),weight)
        p['conditions']=np.array([kind]*len(p['x']))
        p['repeats']=np.full(len(p['x']),repeat)
        pieces.append(p)
    return {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}


def predpath(candidate,group,seed,arm,stage,part):
    return CACHE/'predictions'/candidate/group/str(seed)/arm/f'{stage}_{part}.npz'


def readpred(path):
    assert path.exists() and sha(path)==read(path.with_suffix('.json'))['sha256']
    with np.load(path) as z:return {k:z[k] for k in z.files}


def predict(model,p,path,states=None):
    if path.exists():return readpred(path)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    if states is None:q=model.predict(p['x'])
    else:
        q=np.empty((len(p['x']),24,9),dtype=np.float32)
        for sid,state in states.items():
            ix=p['pairs'][:,0]==str(sid)
            model.load_learned(state);q[ix]=model.predict(p['x'][ix])
    torch.cuda.synchronize()
    out={k:v for k,v in p.items() if k!='x'};out['q']=q
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**out)
    save(path.with_suffix('.json'),dict(sha256=sha(path),seconds=time.perf_counter()-start,examples=len(q),
         peak_allocated_bytes=torch.cuda.max_memory_allocated(),finite=bool(np.isfinite(q).all())))
    return out


def calibration(cal):
    result={}
    for sid in dict.fromkeys(cal['pairs'][:,0]):
        ix=cal['pairs'][:,0]==sid
        result[str(sid)]=affine(cal['q'][ix],cal['y'][ix],cal['weights'][ix])
    return result


def adjusted(pred,cal,mode):
    q=pred['q'].astype(np.float64,copy=True)
    if mode=='CAL':
        for sid,(a,b) in cal.items():
            ix=pred['pairs'][:,0]==str(sid);q[ix]=a*q[ix]+b
    return q


def score(pred,cal,mode):
    errors=point_loss(adjusted(pred,cal,mode),pred['y'],pred['sigma'])
    scores=[]
    for sid in dict.fromkeys(pred['pairs'][:,0]):
        ix=pred['pairs'][:,0]==sid
        scores.append(float(np.average(errors[ix],weights=pred['weights'][ix])))
    return float(np.mean(scores))


def select(candidate,methods,seeds):
    details={};curves={}
    for arm in methods:
        modes={}
        for mode in ['RAW','CAL']:
            vals=[]
            for seed in seeds:
                cal=readpred(predpath(candidate,'dev',seed,arm,2,'cal'))
                val=readpred(predpath(candidate,'dev',seed,arm,2,'validation'))
                vals.append(score(val,calibration(cal),mode))
            modes[mode]=float(np.mean(vals))
        mode=min(modes,key=lambda m:(modes[m],m!='RAW'))
        details[arm]=dict(mode=mode,score=modes[mode],options=modes,source_method=arm)
        curves[arm]={}
        for stage in range(3):
            vals=[]
            for seed in seeds:
                cal=readpred(predpath(candidate,'dev',seed,arm,stage,'cal'))
                val=readpred(predpath(candidate,'dev',seed,arm,stage,'validation'))
                vals.append(score(val,calibration(cal),mode))
            curves[arm][str(stage)]=float(np.mean(vals))
    if candidate=='h1':
        fixed=min([f'FIXED_{g}' for g in [0,.25,.5,1]],key=lambda name:(details[name]['score'],float(name.split('_')[1])))
        details['FIXED_SHRINK']=details[fixed].copy();curves['FIXED_SHRINK']=curves[fixed]
        candidates=['F0','SHARED','LOCAL_LORA','RIDGE','UNSHRUNK','FIXED_SHRINK']
        proposal='U_SHRINK';learned=['SHARED','LOCAL_LORA','RIDGE','UNSHRUNK','FIXED_SHRINK']
    else:
        candidates=['F0_NATIVE','F0_INTERP','QV_LORA','CENTROID_BIAS','KEY_BIAS','GENERIC_BIAS']
        proposal='SET_BIAS';learned=['QV_LORA','CENTROID_BIAS','KEY_BIAS','GENERIC_BIAS']
    winner=min(candidates,key=lambda name:details[name]['score'])
    learned_winner=min(learned,key=lambda name:details[name]['score'])
    def decreasing(name):
        a,b,c=[curves[name][str(j)] for j in range(3)]
        return a>b>c and (b-c)/b>=.01
    result=dict(details=details,curves=curves,baseline=winner,learned_baseline=learned_winner,proposal=proposal,
                undertrained=bool(decreasing(proposal) and decreasing(learned_winner)),selection_only_seeds=[],repeat_seeds=seeds)
    save(RESULTS/f'SELECTION_{candidate.upper()}.json',result)
    return result
