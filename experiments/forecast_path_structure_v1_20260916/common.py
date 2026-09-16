"""Run-local CPU definitions; no historical runner globals or label access in payload."""
import csv, hashlib, json, time, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RUN='forecast_path_structure_v1_20260916'; OUT=ROOT/'results'/RUN; CACHE=ROOT/'.cache'/RUN
FEATURES=['temperature_2m','wind_speed_10m','shortwave_radiation']
ARMS=['LATEST','DROP','PATH','POINT']; SEEDS=[61730,61731]; LRS=[1e-4,3e-5]
TAUS=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99])
REV='dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');q.replace(p)
def read(p):return json.loads(Path(p).read_text())
def csvwrite(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def stable_seed(*parts):return int.from_bytes(hashlib.sha256('\x1f'.join(map(str,parts)).encode()).digest()[:8],'big')
def point_offsets(t,o,seed,block):return np.random.default_rng(stable_seed('POINT_V1',t,o,seed,block)).permutation(np.tile(np.arange(4,dtype=np.int64),6))
def select_path(paths,epoch,arm,offsets=None):
 a=np.asarray(paths);assert a.shape==(4,3,24) and np.isfinite(a).all()
 if arm in ('LATEST','DROP'):return a[0].copy()
 if arm=='PATH':return a[epoch%4].copy()
 assert arm=='POINT';d=np.asarray(offsets);assert d.shape==(24,) and np.array_equal(np.bincount(d,minlength=4),[6]*4)
 return np.stack([a[(epoch%4+d[h])%4,:,h] for h in range(24)],axis=1)
def metric(q,y,sigma):
 q=np.asarray(q,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
 assert q.shape==(21,24) and y.shape==(24,) and sigma>0 and np.isfinite(q).all() and np.isfinite(y).all()
 e=y[None]-q;p=2*np.maximum(TAUS[:,None]*e,(TAUS[:,None]-1)*e);med=q[np.flatnonzero(TAUS==.5)[0]];d=med-y
 return dict(primary=float(p.mean()/sigma),raw_2pinball=float(p.mean()),RMSE=float(np.sqrt(np.mean(d*d))),MAE=float(np.mean(abs(d))),coverage80=float(np.mean((y>=q[np.flatnonzero(TAUS==.1)[0]])&(y<=q[np.flatnonzero(TAUS==.9)[0]]))),width80=float(np.mean(q[np.flatnonzero(TAUS==.9)[0]]-q[np.flatnonzero(TAUS==.1)[0]])))
def calibrate(q,y):
 assert q.shape==(len(y),4,21,24) and y.shape==(len(q),24)
 e=y[:,None,None,:]-q
 return np.array([np.quantile(e[:,:,i,:].reshape(-1),t,method='linear') for i,t in enumerate(TAUS)])
def apply_cal(q,delta):return np.sort(np.asarray(q,dtype=float)+np.asarray(delta)[:,None],axis=-2)
def payload(x,scaling,k=0,arm=None,epoch=0,offsets=None,mask=None,history=False):
 # x contains context and forecasts only; this API cannot access a label file.
 if history:return dict(target=x['context'].copy())
 mu=np.asarray(scaling['weather_mean'])[:,None];sd=np.asarray(scaling['weather_std'])[:,None]
 past=((x['past'].astype(float)-mu)/sd).astype('float32')
 fut=x['paths'][k].copy() if arm is None else select_path(x['paths'],epoch,arm,offsets)
 fut=((fut.astype(float)-mu)/sd).astype('float32')
 if mask is not None:past=past*mask[:,None];fut=fut*mask[:,None]
 return dict(target=x['context'].copy(),past_covariates={f:past[i] for i,f in enumerate(FEATURES)},future_covariates={f:fut[i] for i,f in enumerate(FEATURES)})
