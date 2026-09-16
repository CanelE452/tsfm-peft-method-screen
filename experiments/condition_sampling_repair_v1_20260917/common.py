"""Immutable nine-condition study definitions; no historical runner invocation."""
import csv,hashlib,json,os,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
NAME='condition_sampling_repair_v1_20260917'
OUT=ROOT/'results'/NAME
CACHE=ROOT/'.cache'/NAME
EXP=ROOT/'experiments'/NAME
ORDER=['N01','N02','N03','R04','N07','R08']
ARMS={'N01':['A0','A1','A2','A3'],'N02':['B0','B1','B2','B3'],'N03':['C0','C1','C2','C3'],'R04':['D0','D1','D2','D3'],'R05':['E0','E1','E2','E3','E4'],'N06':['F0','F1','F2','F3'],'N07':['G0','G1','G2','G3'],'R08':['H0','H1','H2','H3'],'R09':['I0','I1','I2','I3','I4']}
NAMES={'N01':'ASYNC','N02':'ARCHIVE','N03':'CLOCK','R04':'REVISION','R05':'VINTAGE','N06':'JOINT','N07':'SPECTRAL','R08':'LEAD','R09':'MIXED'}
SOURCES={'N01':'electricity','N02':'traffic','N03':'ettm1','R04':'traffic','N07':'electricity','R08':'traffic','R09':'electricity'}
QUESTIONS={'N01':'부분 채널 지연에서 age 보정의 추가 가치','N02':'관측 완료 과거 사례의 continuation 보정 가치','N03':'새 관측 밀도에서 학습 kernel의 추가 가치','R04':'정확도 제약 안에서 innovation 가중 수정 억제','R05':'최신 정확도를 보호하는 vintage 정책','N06':'동일 주변분포의 하루 전체 위험 결합','N07':'작지만 예측 가능한 주파수 성분 적응','R08':'관측 가능한 선행 구간에 따른 보정','R09':'상세·집계 혼합 학습의 패턴 보존'}
CONTRASTS={'N01':['A2','A1','A0'],'N02':['B2','B1','BLEND'],'N03':['C2','C0','C1'],'R04':['D2','D1','BLEND'],'R05':['E2','E3','E0'],'N06':['F0','F1'],'N07':['G1','G3','G0'],'R08':['H2','H1','BLEND'],'R09':['I3','I2','I1']}
PROPOSED={'N01':'A3','N02':'B3','N03':'C3','R04':'D3','R05':'E4','N06':'F2','N07':'G2','R08':'H3','R09':'I4'}
ROLES=['TRAIN','V_SELECT','V_CAL','E_DISCOVERY'];LRS=[1e-4,3e-5];SEEDS=[73100,73101,73102]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def digest(*x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n');q.replace(p)
def csvwrite(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)) or ['status']);w.writeheader();w.writerows(rows)
def npz(p,**x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**x)
 tmp.replace(p)
def rng(*key):return np.random.default_rng(int(digest(*key)[:16],16))
def ridge(x,y,alpha):
 x=np.asarray(x,float);y=np.asarray(y,float);xm=x.mean(0);xs=x.std(0);xs=np.where(xs>1e-10,xs,1.);xx=(x-xm)/xs;ym=y.mean(0);b=np.linalg.solve(xx.T@xx+alpha*np.eye(x.shape[1]),xx.T@(y-ym))/xs.reshape((-1,)+(1,)*(y.ndim-1));return np.concatenate([np.asarray(ym-xm@b).reshape((1,)+y.shape[1:]),b],axis=0)
def ridge_cv(x,y):
 n=len(x);bounds=np.linspace(0,n,5,dtype=int);scores={};folds=[]
 for a in [.1,1.,10.]:
  err=[]
  for k in [1,2,3]:
   tr=bounds[k];lo,hi=bounds[k:k+2];b=ridge(x[:tr],y[:tr],a);err.extend(((b[0]+x[lo:hi]@b[1:]-y[lo:hi])**2).reshape(-1).tolist())
  scores[a]=float(np.mean(err))
 a=min(scores,key=lambda v:(scores[v],v));return ridge(x,y,a),dict(alpha=a,fold_mse=scores,fit_pairs=n,fold_train_ends=bounds[1:4].tolist())
def nrms(p,y,sigma):
 e=(np.asarray(p,float)-np.asarray(y,float))/np.asarray(sigma)[...,None]
 return float(np.sqrt(np.mean(e*e,axis=(0,-1))).mean())
def block_counts(origins,block):
 oo=np.asarray(origins);blocks=(oo//block);unique=np.arange(blocks.min(),blocks.max()+1);g=np.random.default_rng(73300);b=g.multinomial(len(unique),np.full(len(unique),1/len(unique)),2000);return b[:,blocks-unique[0]]

OLD=ROOT/'results/condition_studies_v1_20260916'
OLDCACHE=ROOT/'.cache/condition_studies_v1_20260916'
SOURCES={t:SOURCES[t] for t in ORDER}
CLOSEST={'N01':'A2','N02':'B2','N03':'C2','R04':'D2','N07':'G3','R08':'H2'}
