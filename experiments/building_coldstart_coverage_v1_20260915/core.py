"""Fixed calendar/metric/rule contracts. No model or future data access."""
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
RUN='building_coldstart_coverage_v1_20260915'
OUT=ROOT/'results'/RUN
CACHE=ROOT/'.cache'/RUN
EXP=ROOT/'experiments'/RUN
CONFIG=ROOT/'configs/building_coldstart_coverage_v1_20260915.json'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(p)
def csvwrite(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def csvread(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def daytype(t):return int(pd.Timestamp(t).dayofweek>=5)
def windows(history,origin):
    h=np.asarray(history,dtype=np.float64);assert len(h) in (72,336) and np.isfinite(h).all()
    origin=pd.Timestamp(origin);assert origin.hour==origin.minute==origin.second==0
    start=origin-pd.Timedelta(hours=len(h));offsets=list(range(24,len(h)-23,24))
    x=np.stack([h[i-24:i] for i in offsets]);y=np.stack([h[i:i+24] for i in offsets]);dates=[start+pd.Timedelta(hours=i) for i in offsets]
    assert len(offsets)==len(h)//24-1 and all(d+pd.Timedelta(hours=24)<=origin for d in dates)
    coverage=sum(daytype(d)==daytype(origin) for d in dates)
    return x,y,dates,coverage
QUANTILES=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99],dtype=np.float64)
def metrics(q,y,history):
    q=np.asarray(q,dtype=np.float64);y=np.asarray(y,dtype=np.float64);h=np.asarray(history,dtype=np.float64)
    assert q.shape==(21,24) and y.shape==(24,) and all(np.isfinite(v).all() for v in [q,y,h])
    scale=max(float(h.std(ddof=0)),1e-6);e=q[10]-y;rmse=float(np.sqrt(np.mean(e*e)));d=y[None,:]-q
    pin=float((2*np.maximum(QUANTILES[:,None]*d,(QUANTILES[:,None]-1)*d)).mean())
    return dict(scaled_RMSE=rmse/scale,raw_RMSE=rmse,raw_MAE=float(np.mean(np.abs(e))),scaled_2pinball=pin/scale,history_std=scale,official_style_NRMSE_percent=100*rmse/float(y.mean()) if y.mean()>0 else None)
def interpolate(f,l,alpha):
    assert 0<=alpha<=1
    if alpha==0:return f.copy()
    if alpha==1:return l.copy()
    q=(1-alpha)*f+alpha*l
    assert np.all(np.diff(q,axis=-2)>=0)
    return q

def affine(q,y):
    """Positive constrained OLS; a>=1e-6, centered closed-form float64."""
    x=np.asarray(q,dtype=np.float64)[:,10,:].ravel();v=np.asarray(y,dtype=np.float64).ravel()
    variance=float(np.mean((x-x.mean())**2))
    if variance<=np.finfo(float).eps*max(1.,float(np.mean(x*x))):raise ValueError('AFFINE_ILL_CONDITIONED')
    a=max(1e-6,float(np.mean((x-x.mean())*(v-v.mean()))/variance));b=float(v.mean()-a*x.mean())
    assert np.isfinite([a,b]).all();return dict(a=a,b=b,parameters=2,objective='squared median error; a>=1e-6; centered closed-form OLS',predictor_variance=variance)
def gain(base,candidate):
    if base<=0:raise ValueError('GAIN_UNDEFINED_ZERO_BASELINE')
    return 100*(base-candidate)/base

def source_hashes():
    files=list(EXP.glob('*.py'))+[ROOT/'scripts/run_building_coldstart_coverage_v1.py',ROOT/'scripts/finalize_building_coldstart_coverage_v1.py',CONFIG,OUT/'PROTOCOL.md',OUT/'literature_boundary.md',ROOT/'src/tsfm_peft_screen/backbone.py',ROOT/'src/tsfm_peft_screen/lora.py',ROOT/'scripts/priority12/common.py',ROOT/'scripts/with_cuda.sh']
    return {str(p.relative_to(ROOT)):sha(p) for p in files}
def check_seal():
    seal=read(OUT/'prepare_seal.json')
    for p,h in seal['sources'].items():assert sha(ROOT/p)==h,('SOURCE_CHANGED',p)
    for p,h in seal['staged'].items():assert sha(ROOT/p)==h,('STAGED_CHANGED',p)
    return seal
