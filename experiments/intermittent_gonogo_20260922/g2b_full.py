"""G2-B 원안: REF-LGB 에 달력·SNAP·가격 공변량 포함 (지시문 §6 G2 D-M5)."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
M5=Path("/home/minjae/Documents/github/m5dataset"); OUT=Path(__file__).resolve().parent
H=28; SPLIT_END=1885
sales=pd.read_csv(M5/"data/sales_train_evaluation.csv")
cal=pd.read_csv(M5/"data/calendar.csv")
X=sales[[f"d_{i}" for i in range(1,1942)]].to_numpy(np.float32)
train=X[:,:1913]; test=X[:,1913:1941]
elig=(train[:,:SPLIT_END]>0).sum(1)>=20
state=sales["state_id"].to_numpy()
print("적격",elig.sum(),flush=True)

cal=cal.set_index("d")
dow=cal["wday"].to_dict(); mon=cal["month"].to_dict()
ev=(cal["event_name_1"].notna()|cal["event_name_2"].notna()).astype(int).to_dict()
snap={s:cal[f"snap_{s}"].to_dict() for s in ["CA","TX","WI"]}
snap_arr={s:np.array([snap[s][f"d_{i}"] for i in range(1,1942)],np.float32) for s in snap}
dow_a=np.array([dow[f"d_{i}"] for i in range(1,1942)],np.float32)
mon_a=np.array([mon[f"d_{i}"] for i in range(1,1942)],np.float32)
ev_a =np.array([ev[f"d_{i}"]  for i in range(1,1942)],np.float32)
st_idx=np.select([state=="CA",state=="TX",state=="WI"],[0,1,2],0)
snap_mat=np.stack([snap_arr["CA"],snap_arr["TX"],snap_arr["WI"]])   # (3,1941)

# 가격: item x store x wm_yr_wk -> d 단위
print("가격 조인",flush=True); t0=time.time()
sp=pd.read_csv(M5/"data/sell_prices.csv")
wk=cal["wm_yr_wk"].to_dict(); wk_a=np.array([wk[f"d_{i}"] for i in range(1,1942)])
key=sales["item_id"].astype(str)+"_"+sales["store_id"].astype(str)
sp["key"]=sp["item_id"].astype(str)+"_"+sp["store_id"].astype(str)
pv=sp.pivot_table(index="key",columns="wm_yr_wk",values="sell_price",aggfunc="first")
pv=pv.reindex(key.to_numpy())
cols=pv.columns.to_numpy(); ci={w:j for j,w in enumerate(cols)}
pvv=pv.to_numpy(np.float32)
price=np.full((len(sales),1941),np.nan,np.float32)
for i,w in enumerate(wk_a):
    if w in ci: price[:,i]=pvv[:,ci[w]]
print("  %.0fs  결측 %.2f%%"%(time.time()-t0,100*np.isnan(price).mean()),flush=True)

LAGS=[1,7,14,28]; ROLL=[7,28,56,365]
def feats(o,h):
    hist=X[:,:o]; ti=o+h-1        # 타깃 날짜 인덱스(0-based)
    f=[hist[:,-L] for L in LAGS]
    f+=[hist[:,-R:].mean(1) for R in ROLL]
    f+=[(hist[:,-R:]==0).mean(1) for R in [56,365]]
    f+=[hist[:,-365:].std(1)]
    f+=[np.full(len(hist),h,np.float32), np.full(len(hist),dow_a[ti]), np.full(len(hist),mon_a[ti]),
        np.full(len(hist),ev_a[ti]), snap_mat[st_idx,ti].astype(np.float32),
        price[:,ti], price[:,ti]/np.nanmean(price[:,max(0,ti-28):ti+1],axis=1)]
    return np.stack(f,1).astype(np.float32)
FN=[f"lag{L}" for L in LAGS]+[f"rm{R}" for R in ROLL]+["zr56","zr365","sd365","h","dow","month","event","snap","price","price_rel"]
def build(origins):
    Fs,Ys=[],[]
    for o in origins:
        for h in range(1,H+1):
            Fs.append(feats(o,h)); Ys.append(X[:,o+h-1])
    return np.vstack(Fs),np.concatenate(Ys)
TR=[1857,1829,1801,1773]
t0=time.time(); Xtr,ytr=build(TR); print("train",Xtr.shape,"%.0fs"%(time.time()-t0),flush=True)
Xva,yva=build([SPLIT_END]); Xte,_=build([1913])
m=lgb.train(dict(objective="tweedie",tweedie_variance_power=1.1,learning_rate=0.05,
                 feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,num_leaves=63,
                 min_data_in_leaf=100,seed=20260922,verbose=-1),
            lgb.Dataset(Xtr,ytr,feature_name=FN),num_boost_round=2000,
            valid_sets=[lgb.Dataset(Xva,yva,feature_name=FN)],
            callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(300)])
print("best_iter",m.best_iteration,flush=True)
print("중요도 top8:",sorted(zip(FN,m.feature_importance("gain")),key=lambda z:-z[1])[:8],flush=True)
rs=lambda p: np.clip(p,0,None).reshape(H,-1).T
lgb_val=rs(m.predict(Xva,num_iteration=m.best_iteration)); lgb_test=rs(m.predict(Xte,num_iteration=m.best_iteration))
zs_val=np.load(OUT/"val_zs_qmean.npy"); val_true=np.load(OUT/"val_true.npy")
QTR=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
QS=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
Qz=np.load(OUT/"Q_zs.npy")[:,:,[QTR.index(q) for q in QS]]
lve=np.concatenate([[0.],np.array(QS),[1.]]); ve=np.concatenate([Qz[...,:1],Qz,Qz[...,-1:]],-1)
zs_test=np.clip(np.trapezoid(ve,x=lve,axis=-1),0,None)
a_zs=val_true.sum()/zs_val.sum(); a_lgb=val_true.sum()/lgb_val.sum()
print(f"alpha*: ZS={a_zs:.4f} LGB={a_lgb:.4f}",flush=True)
def rmsse(pred,truth,hist):
    num=((truth-pred)**2).mean(1); den=np.empty(len(hist))
    for i in range(len(hist)):
        nz=np.nonzero(hist[i])[0]
        if len(nz)==0: den[i]=np.nan; continue
        s=hist[i][nz[0]:]; den[i]=np.mean(np.diff(s)**2) if len(s)>1 else np.nan
    return np.sqrt(num/den)
r_zs=rmsse(zs_test*a_zs,test,train); r_lgb=rmsse(lgb_test*a_lgb,test,train)
ok=elig&np.isfinite(r_zs)&np.isfinite(r_lgb)
m_zs,m_lgb=r_zs[ok].mean(),r_lgb[ok].mean(); dr=100*(m_zs-m_lgb)/m_zs
rng=np.random.default_rng(20260922); ix=np.where(ok)[0]; N=len(ix)
bs=np.array([100*(r_zs[s].mean()-r_lgb[s].mean())/r_zs[s].mean() for s in (ix[rng.integers(0,N,N)] for _ in range(2000))])
lo,hi=np.percentile(bs,[2.5,97.5]); p=(dr>=1.0)and(lo>0)
print(f"\n=== G2-B (공변량 포함, 지시문 원안) ===")
print(f"  ZS-QMEAN x a*  RMSSE={m_zs:.4f}")
print(f"  REF-LGB  x a*  RMSSE={m_lgb:.4f}")
print(f"  ΔRMSSE% = {dr:+.2f}%  CI=[{lo:+.2f},{hi:+.2f}]  (기준 ≥1.0 & CI하한>0)")
print(f"  >>> {'PASS' if p else 'FAIL'}")
json.dump({"rmsse_zs":float(m_zs),"rmsse_lgb":float(m_lgb),"delta_pct":float(dr),
           "ci":[float(lo),float(hi)],"pass":bool(p),"best_iter":int(m.best_iteration),
           "alpha_zs":float(a_zs),"alpha_lgb":float(a_lgb),"features":FN,"n":int(ok.sum())},
          open(OUT/"g2b_full.json","w"),indent=2)
print("[DONE]")
