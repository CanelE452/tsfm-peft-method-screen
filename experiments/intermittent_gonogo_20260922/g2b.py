"""G2-B: 목표 데이터 적응 여지. REF-LGB(공변량 없음) vs ZS-QMEAN, 둘 다 α* 보정 후 RMSSE."""
import sys, json, time
from pathlib import Path
import numpy as np
M5=Path("/home/minjae/Documents/github/m5dataset"); sys.path.insert(0,str(M5))
OUT=Path(__file__).resolve().parent
import pandas as pd, lightgbm as lgb
print("lightgbm",lgb.__version__,flush=True)

# ---- 데이터 (pandas 직접 로드: chronos venv 밖) ----
sales=pd.read_csv(M5/"data/sales_train_evaluation.csv")
dcols=[f"d_{i}" for i in range(1,1942)]
X=sales[dcols].to_numpy(dtype=np.float32)          # (30490,1941)
train=X[:,:1913]; test=X[:,1913:1941]
SPLIT_END=1885; H=28
elig=(train[:,:SPLIT_END]>0).sum(1)>=20
print("적격",elig.sum(),flush=True)

LAGS=[1,7,14,28]; ROLL=[7,28,56,365]
def feats(o):
    """원점 o (1-indexed d_o 까지 관측). 반환 (N, F) 계열별 특성."""
    h=X[:,:o]
    f=[h[:,-L] for L in LAGS]
    f+=[h[:,-R:].mean(1) for R in ROLL]
    f+=[(h[:,-R:]==0).mean(1) for R in [56,365]]
    f+=[h[:,-365:].std(1)]
    return np.stack(f,1).astype(np.float32)
FN=[f"lag{L}" for L in LAGS]+[f"rm{R}" for R in ROLL]+["zr56","zr365","sd365"]+["h"]

def build(origins):
    Fs,Ys=[],[]
    for o in origins:
        base=feats(o)
        for h in range(1,H+1):
            Fs.append(np.column_stack([base,np.full(len(base),h,np.float32)]))
            Ys.append(X[:,o+h-1])
    return np.vstack(Fs),np.concatenate(Ys)

TR_ORI=[1857,1829,1801,1773]         # 타깃 끝 <= d_1885 (validation 이전)
for o in TR_ORI: assert o+H<=SPLIT_END, o
t0=time.time(); Xtr,ytr=build(TR_ORI); print("train",Xtr.shape,"%.0fs"%(time.time()-t0),flush=True)
Xva,yva=build([SPLIT_END]); print("valid",Xva.shape,flush=True)   # 타깃 d_1886..d_1913
Xte,_  =build([1913])                                             # 타깃 d_1914..d_1941

m=lgb.train(dict(objective="tweedie",tweedie_variance_power=1.1,learning_rate=0.05,
                 feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,
                 num_leaves=63,min_data_in_leaf=100,seed=20260922,verbose=-1),
            lgb.Dataset(Xtr,ytr,feature_name=FN), num_boost_round=2000,
            valid_sets=[lgb.Dataset(Xva,yva,feature_name=FN)],
            callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(200)])
print("best_iter",m.best_iteration,flush=True)
def reshape(p): return np.clip(p,0,None).reshape(H,-1).T      # (N,H)
lgb_val=reshape(m.predict(Xva,num_iteration=m.best_iteration))
lgb_test=reshape(m.predict(Xte,num_iteration=m.best_iteration))

zs_val=np.load(OUT/"val_zs_qmean.npy"); val_true=np.load(OUT/"val_true.npy")
d=np.load(OUT/"curves.npz")  # 미사용
zs_test=None
import numpy as _np
Qz=_np.load(OUT/"Q_zs.npy")
QSPEC=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
QTR=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
def interp(Q,tgt,src):
    out=np.empty(Q.shape[:-1]+(len(tgt),),np.float32)
    for i,t in enumerate(tgt): out[...,i]=np.array([np.interp(t,src,Q[a,b]) for a in range(Q.shape[0]) for b in range(Q.shape[1])]).reshape(Q.shape[:2])
    return out
lv=np.asarray(QSPEC,float); lve=np.concatenate([[0.],lv,[1.]])
idx=[QTR.index(q) for q in QSPEC]
Qs=Qz[:,:,idx]
ve=np.concatenate([Qs[...,:1],Qs,Qs[...,-1:]],-1)
zs_test=np.clip(np.trapezoid(ve,x=lve,axis=-1),0,None)

# ---- α* (validation 에서만) ----
a_zs = val_true.sum()/zs_val.sum()
a_lgb= val_true.sum()/lgb_val.sum()
print(f"alpha*: ZS={a_zs:.4f}  LGB={a_lgb:.4f}",flush=True)

# ---- RMSSE (지시문 §5.4) ----
def rmsse(pred,truth,hist):
    num=((truth-pred)**2).mean(1)
    den=np.empty(len(hist))
    for i in range(len(hist)):
        r=hist[i]; nz=np.nonzero(r)[0]
        if len(nz)==0: den[i]=np.nan; continue
        s=r[nz[0]:]; den[i]=np.mean(np.diff(s)**2) if len(s)>1 else np.nan
    return np.sqrt(num/den), den
hist=train[:,:1913]
r_zs,den=rmsse(zs_test*a_zs,test,hist)
r_lgb,_ =rmsse(lgb_test*a_lgb,test,hist)
ok=elig&~np.isnan(r_zs)&~np.isnan(r_lgb)&np.isfinite(r_zs)&np.isfinite(r_lgb)
print(f"RMSSE 계산 가능 {ok.sum()} (분모 0/NaN 제외 {int(elig.sum()-ok.sum())})",flush=True)
m_zs,m_lgb=r_zs[ok].mean(),r_lgb[ok].mean()
dr=100*(m_zs-m_lgb)/m_zs
rng=np.random.default_rng(20260922); N=ok.sum(); ix=np.where(ok)[0]
bs=np.array([100*(r_zs[s].mean()-r_lgb[s].mean())/r_zs[s].mean() for s in (ix[rng.integers(0,N,N)] for _ in range(2000))])
lo,hi=np.percentile(bs,[2.5,97.5])
p=(dr>=1.0) and (lo>0)
print(f"\n=== G2-B ===")
print(f"  ZS-QMEAN x a*  RMSSE={m_zs:.4f}")
print(f"  REF-LGB  x a*  RMSSE={m_lgb:.4f}")
print(f"  ΔRMSSE% = {dr:+.2f}%   CI=[{lo:+.2f},{hi:+.2f}]   (기준 ≥1.0 & CI하한>0)")
print(f"  >>> {'PASS' if p else 'FAIL'}")
json.dump({"alpha_zs":float(a_zs),"alpha_lgb":float(a_lgb),"rmsse_zs":float(m_zs),
           "rmsse_lgb":float(m_lgb),"delta_pct":float(dr),"ci":[float(lo),float(hi)],
           "pass":bool(p),"n":int(ok.sum()),"best_iter":int(m.best_iteration),
           "train_origins":TR_ORI,"features":FN},open(OUT/"g2b.json","w"),indent=2)
np.save(OUT/"lgb_test.npy",lgb_test.astype(np.float32))
print("[DONE]",OUT/"g2b.json")
