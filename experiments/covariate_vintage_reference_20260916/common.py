import hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RUN='covariate_vintage_reference_20260916';OUT=ROOT/'results'/RUN;CACHE=ROOT/'.cache'/RUN
SOURCE=ROOT/'.cache/covariate_availability_audit_20260916'
FEATURES=['temperature_2m','wind_speed_10m','shortwave_radiation']
TAUS=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99])
BASE=['F0','PAST','LATEST','MEAN_INPUT','MIXTURE']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def read(p):return json.loads(Path(p).read_text())
def mixture(q,taus=TAUS):
 """q[K,Q,H]; monotone inverse-CDF interpolation with clamped tails."""
 q=np.asarray(q,dtype=np.float64);k,n,h=q.shape;assert n==len(taus) and np.all(np.diff(q,axis=1)>=0)
 result=np.empty((len(taus),h))
 for j in range(h):
  low=np.full(len(taus),q[:,0,j].min());high=np.full(len(taus),q[:,-1,j].max())
  for _ in range(64):
   mid=(low+high)/2
   cdf=sum(np.interp(mid,np.r_[a[0],a,a[-1]],np.r_[0,taus,1],left=0,right=1) for a in q[:,:,j])/k
   upper=cdf>=taus;high=np.where(upper,mid,high);low=np.where(upper,low,mid)
  result[:,j]=high
 return result

def metrics(q,y,sd):
 q=np.asarray(q,dtype=np.float64);y=np.asarray(y,dtype=np.float64);e=y[None]-q;pin=float((2*np.maximum(TAUS[:,None]*e,(TAUS[:,None]-1)*e)).mean());d=q[10]-y
 return dict(primary=pin/sd,raw_2pinball=pin,raw_RMSE=float(np.sqrt(np.mean(d*d))),raw_MAE=float(np.mean(abs(d))),coverage80=float(np.mean((y>=q[2])&(y<=q[18]))),width80=float(np.mean(q[18]-q[2])))
def scalar_primary(q,y,sd):
 total=0
 for i,t in enumerate(TAUS):
  for j,v in enumerate(y):
   d=float(v)-float(q[i,j]);total+=2*(t*d if d>=0 else (t-1)*d)
 return total/(len(TAUS)*len(y)*sd)
def calibrate(qs,ys,sds):
 qs=np.asarray(qs);ys=np.asarray(ys);sds=np.asarray(sds);b=float(np.median((ys-qs[:,10,:])/sds[:,None]));rows=[]
 for s in [.5,.75,1.,1.25,1.5,2.]:
  pred=qs[:,10:11]+b*sds[:,None,None]+s*(qs-qs[:,10:11]);loss=np.mean([metrics(q,y,d)['primary'] for q,y,d in zip(pred,ys,sds)])
  rows.append(dict(s=s,primary=float(loss)))
 best=min(rows,key=lambda x:(x['primary'],x['s']));return dict(b=b,s=best['s'],grid=rows)
def apply(q,sd,c):return q[10:11]+c['b']*sd+c['s']*(q-q[10:11])
def checks():
 u=np.array([0.,1.]);toy=np.array([[[0.],[1.]],[[10.],[11.]]]);out=mixture(toy,u)
 # Quantiles at 0 and 1 use the finite support endpoints.
 assert np.allclose(out[:,0],[0,11],atol=1e-12)
 dense=np.linspace(0,1,101);toy=np.stack([dense,dense+10])[:,:,None];out=mixture(toy,dense)
 assert abs(out[10,0]-.2)<1e-12 and abs(out[90,0]-10.8)<1e-12
 single=np.linspace(-3,4,21)[:,None];same=mixture(np.repeat(single[None],4,axis=0));assert np.max(abs(same-single))<1e-12
 atom=np.array([np.zeros((21,1)),np.ones((21,1))*10]);aq=mixture(atom);assert np.allclose(aq[TAUS<=.5],0,atol=1e-12) and np.allclose(aq[TAUS>.5],10,atol=1e-12)
 y=np.arange(24);q=np.tile(np.linspace(-2,25,21)[:,None],(1,24));assert abs(metrics(q,y,3)['primary']-scalar_primary(q,y,3))<1e-12
 return dict(uniform_mixture=True,identical_components=True,atomic_distributions=True,scalar_pinball=True)
