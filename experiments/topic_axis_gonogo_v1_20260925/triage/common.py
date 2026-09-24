"""Shared contracts; all generated text is LF, UTF-8 and strict JSON."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv, hashlib, json, math, time
import numpy as np

class ContractError(RuntimeError): pass
class Blocked(RuntimeError): pass

def require(condition, message):
    if not condition: raise ContractError(message)

def clean_json(x):
    if isinstance(x,dict): return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean_json(v) for v in x]
    if isinstance(x,np.ndarray): return clean_json(x.tolist())
    if isinstance(x,np.generic): return clean_json(x.item())
    if isinstance(x,float) and not math.isfinite(x): return None
    if isinstance(x,Path): return str(x)
    return x

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(clean_json(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(text,encoding='utf-8',newline='\n');tmp.replace(path)

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()
def hash_obj(x):return hashlib.sha256(json.dumps(clean_json(x),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text('',encoding='utf-8');return
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader()
        for r in rows:w.writerow({k:json.dumps(clean_json(v),ensure_ascii=False) if isinstance(v,(list,dict,np.ndarray)) else v for k,v in r.items()})

@dataclass
class Case:
    uid:str
    group:str
    x:np.ndarray          # context, variables. Targets first; all others past-only covariates.
    y:np.ndarray          # horizon, n_target; never passed into predictors or proxy builders.
    score:float
    alternate:float      # fixed estimator perturbation / TRAIN-only re-estimation
    nuisance:np.ndarray
    simple:np.ndarray    # forecast made with context alone
    scale:float
    extra:np.ndarray     # structure features computed only from context / TRAIN bank
    mask:np.ndarray|None=None
    base_uid:str|None=None
    angle:float=0.0
    def validate(self):
        require(self.x.ndim==self.y.ndim==2,'Expected time x channel arrays')
        require(self.x.shape[0]>1 and self.y.shape[0]>0 and self.y.shape[1]<=self.x.shape[1],'Bad lengths')
        require(np.isfinite(self.x).all(),'Non-finite past input')
        require(np.isfinite(self.simple).all() and self.simple.shape==self.y.shape,'Bad simple forecast')
        require(math.isfinite(self.scale) and self.scale>0,'Invalid context-only error scale')
        require(np.isfinite(self.nuisance).all() and np.isfinite(self.extra).all(),'Nonfinite features')
        return self

def entropy(x):
    x=np.asarray(x,float);x=x-x.mean()
    p=np.abs(np.fft.rfft(x))**2
    p=p[1:]
    if len(p)<2 or p.sum()<1e-20:return 0.0
    p=p/p.sum();return float(-np.sum(p*np.log(p+1e-15))/np.log(len(p)))

def nuisance(x,extra=0.):
    v=np.asarray(x,float).reshape(-1);std=max(float(v.std()),1e-9)
    return np.array([entropy(v),np.log(std),float(np.mean(np.abs(np.diff(v)))/std),float(abs(v[-1]-v[0])/std),float(extra)])

def scale_of(x): return max(float(np.std(x)),1e-6)
def normalized_error(case,pred):
    require(pred.shape==case.y.shape and np.isfinite(pred).all(),'Prediction shape/nonfinite')
    mask=np.isfinite(case.y) if case.mask is None else case.mask & np.isfinite(case.y)
    if not mask.any():return float('nan')
    if case.y.shape[1]==2: # vector trajectory: Euclidean ADE, not coordinate-wise MAE
        ok=mask.all(1)
        return float(np.linalg.norm((pred-case.y)[ok],axis=1).mean()/case.scale) if ok.any() else float('nan')
    return float(np.abs(pred-case.y)[mask].mean()/case.scale)

def normalized_features(case):
    # Same raw information rights for generic vs structure-augmented ridge.
    mu=case.x.mean(0);sd=case.x.std(0);sd=np.maximum(sd,1e-6)
    raw=((case.x-mu)/sd).T.reshape(-1)
    return np.r_[raw,mu,sd],mu[:case.y.shape[1]],sd[:case.y.shape[1]]
