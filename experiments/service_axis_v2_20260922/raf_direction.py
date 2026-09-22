"""RAF 방향 확인 (§5.3, §8.6). 판정 게이트 아님 — 부호만 본다."""
import sys, json, csv, time
from pathlib import Path
import numpy as np, torch, pandas as pd
R=Path("/home/minjae/Documents/github/tsfm-peft-method-screen")
sys.path.insert(0,str(R/"experiments/service_axis_v2_20260922")); import core
CACHE=R/".cache/service_axis_v2_20260922"; OUT=R/"results/service_axis_v2_20260922"
XLS=CACHE/"raf/RAF data - 7 years demand  - 5000 items.xls"
Hh=6; L=1; RP=1; P=2; TRAIN_END=36
CAL_A=[36,42]; CAL_B=[48,54]; EVAL=[60,66,72,78]; WIN=CAL_A+CAL_B+EVAL
TAUS=[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.975,0.99]
def log(m): print(m,flush=True)
df=pd.read_excel(XLS)
meta=["Item Ref no","DESCRIPTION","Lead Time (months)","PRICE (£)"]
D=df[[c for c in df.columns if c not in meta]].values.astype(np.float32)
log(f"RAF {D.shape}")
elig=(D[:,:TRAIN_END]>0).sum(1)>=2
log(f"적격(1~36월 양수>=2): {elig.sum()}/{len(D)}")

def nz_patch():
    import chronos.chronos_bolt as cb
    def f(self,x,loc_scale=None):
        od=x.dtype; xf=x.to(torch.float32)
        if loc_scale is None:
            nan=torch.isnan(xf); xc=torch.nan_to_num(xf,nan=0.0); nzm=(xc>0)&(~nan)
            cnt=nzm.float().sum(-1,keepdim=True).clamp(min=1.0)
            loc=(xc*nzm.float()).sum(-1,keepdim=True)/cnt
            sc=(((xc-loc)*nzm.float()).square().sum(-1,keepdim=True)/cnt.clamp(min=1.0)).sqrt()
            allz=(nzm.float().sum(-1)==0)
            if allz.any():
                ol=torch.nan_to_num(torch.nanmean(xf,dim=-1,keepdim=True),nan=0.0)
                os_=torch.nan_to_num((xf-ol).square().nanmean(-1,keepdim=True).sqrt(),nan=1.0)
                loc=torch.where(allz.unsqueeze(-1),ol,loc); sc=torch.where(allz.unsqueeze(-1),os_,sc)
            sc=torch.where(sc==0,torch.tensor(self.eps,device=sc.device,dtype=sc.dtype),sc)
            loc_scale=(loc,sc)
        l,s=loc_scale; z=(xf-l)/s
        if self.use_arcsinh: z=torch.arcsinh(z)
        return z.to(od),(l,s)
    cb.InstanceNorm.forward=f

MODE=sys.argv[1]
P_DIR=CACHE/"raf_preds"; P_DIR.mkdir(exist_ok=True)
if MODE in ("zs","nz"):
    if MODE=="nz": nz_patch()
    from chronos import Chronos2Pipeline
    pipe=Chronos2Pipeline.from_pretrained("amazon/chronos-2",device_map="cuda",dtype=torch.bfloat16)
    qlev=list(pipe.quantiles); tag=MODE.upper()
    for o in WIN:
        f_=P_DIR/f"{tag}_QMEAN21_o{o}.npy"
        if f_.exists(): continue
        inp=[torch.tensor(D[i,:o],dtype=torch.float32) for i in range(len(D))]
        qp,_=pipe.predict_quantiles(inp,prediction_length=Hh,quantile_levels=qlev,batch_size=256)
        Q=np.stack([x[0].float().cpu().numpy() for x in qp])
        np.save(f_,core.qmean21(Q).astype(np.float32)); log(f"  {tag} o{o} done")
elif MODE=="classical":
    from statsforecast.models import SeasonalNaive, SimpleExponentialSmoothingOptimized, CrostonSBA, TSB
    mk={"SNAIVE":lambda:SeasonalNaive(season_length=12),"SES":lambda:SimpleExponentialSmoothingOptimized(),
        "SBA":lambda:CrostonSBA(),"TSB":lambda:TSB(alpha_d=0.1,alpha_p=0.1)}
    for o in WIN:
        need=[m for m in mk if not (P_DIR/f"{m}_o{o}.npy").exists()]
        if not need: continue
        res={m:np.zeros((len(D),Hh),np.float32) for m in need}
        for i in range(len(D)):
            y=D[i,:o].astype(np.float64)
            for m in need: res[m][i]=mk[m]().forecast(y=y,h=Hh)["mean"]
        for m in need: np.save(P_DIR/f"{m}_o{o}.npy",np.clip(res[m],0,None))
        log(f"  classical o{o} done")
elif MODE=="lgb":
    import lightgbm as lgb
    LAGS=[1,2,3,6]; ROLL=[3,6,12,24]
    def feats(o,h):
        hist=D[:,:o]
        f=[hist[:,-l] for l in LAGS]+[hist[:,-r:].mean(1) for r in ROLL]
        f+=[(hist[:,-r:]==0).mean(1) for r in [6,24]]+[hist[:,-24:].std(1)]
        f+=[np.full(len(hist),h,np.float32)]
        return np.stack(f,1).astype(np.float32)
    FN=[f"lag{l}" for l in LAGS]+[f"rm{r}" for r in ROLL]+["zr6","zr24","sd24","h"]
    def build(os_):
        F,Y=[],[]
        for o in os_:
            for h in range(1,Hh+1): F.append(feats(o,h)); Y.append(D[:,o+h-1])
        return np.vstack(F),np.concatenate(Y)
    TR=[30-6*k for k in range(4)]
    assert max(TR)+Hh<=TRAIN_END
    Xtr,ytr=build(TR); Xva,yva=build(CAL_A)
    m=lgb.train(dict(objective="tweedie",tweedie_variance_power=1.1,learning_rate=0.05,feature_fraction=0.8,
                     bagging_fraction=0.8,bagging_freq=1,num_leaves=31,min_data_in_leaf=50,seed=20260922,verbose=-1),
                lgb.Dataset(Xtr,ytr,feature_name=FN),num_boost_round=2000,valid_sets=[lgb.Dataset(Xva,yva,feature_name=FN)],
                callbacks=[lgb.early_stopping(100,verbose=False)])
    log(f"  RAF LGB best_iter {m.best_iteration}")
    for o in WIN:
        Xte,_=build([o]); p=np.clip(m.predict(Xte,num_iteration=m.best_iteration),0,None).reshape(Hh,-1).T
        np.save(P_DIR/f"REF-LGB-U_o{o}.npy",p.astype(np.float32))
elif MODE=="score":
    F={"ZS-Q":"ZS_QMEAN21_o{o}.npy","NZ-Q":"NZ_QMEAN21_o{o}.npy","SNAIVE":"SNAIVE_o{o}.npy",
       "SES":"SES_o{o}.npy","SBA":"SBA_o{o}.npy","TSB":"TSB_o{o}.npy","REF-LGB-U":"REF-LGB-U_o{o}.npy"}
    ARMS=list(F)
    valid=elig.copy()
    for a in ARMS:
        for o in WIN:
            p=np.load(P_DIR/F[a].format(o=o)); valid &= np.isfinite(p).all(1)&(p>=-1e-9).all(1)
    IDX=np.where(valid)[0]; log(f"공통 계열 {len(IDX)}")
    def mu(a,o): return np.load(P_DIR/F[a].format(o=o))[IDX].mean(1)
    def res(a): return core.erp_residuals([mu(a,o) for o in CAL_B],[D[IDX,o:o+Hh] for o in CAL_B],P)
    RES={a:res(a) for a in ARMS}
    def i90(a,o):
        m_=mu(a,o); Y=D[IDX,o:o+Hh]; U=[];I=[]
        for t in TAUS:
            S=core.erp_S(m_,RES[a],P,t); sv,dm,cl=core.simulate(S,L,Y)
            u,i=core.unit_metrics(sv,dm,cl); U.append(u); I.append(i)
        return core.interp_at(np.array(U),np.array(I))[0]
    calb={a:float(np.nanmean([i90(a,o) for o in CAL_B])) for a in ARMS}
    S_best=min(["SNAIVE","SES","SBA","TSB"],key=lambda k:calb[k])
    ev={a:[i90(a,o) for o in EVAL] for a in ARMS}
    for a in ARMS: log(f"  {a:10s} EVAL I@90 {[round(x,2) if x==x else None for x in ev[a]]}  평균 {np.nanmean(ev[a]):.3f}")
    FREE=["ZS-Q","NZ-Q","SNAIVE","SES","SBA","TSB"]
    BF=min(FREE,key=lambda k:np.nanmean(ev[k]))
    v1g=-100*(np.array(ev[S_best])-np.array(ev["ZS-Q"]))/np.array(ev[S_best])
    v3d=100*(np.array(ev[BF])-np.array(ev["REF-LGB-U"]))/np.array(ev[BF])
    log(f"\n  [RAF V1] 기준 {S_best}: 창별 {[round(x,2) for x in v1g]}  평균 {v1g.mean():+.2f}%")
    log(f"  [RAF V3] BEST-FREE {BF}: 창별 {[round(x,2) for x in v3d]}  평균 {v3d.mean():+.2f}%")
    with open(OUT/"RAF_DIRECTION.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["comparison","baseline","arm"]+[f"o{o}" for o in EVAL]+["mean"])
        w.writerow(["V1_gap",S_best,"ZS-Q"]+[f"{x:.4f}" for x in v1g]+[f"{v1g.mean():.4f}"])
        w.writerow(["V3_delta",BF,"REF-LGB-U"]+[f"{x:.4f}" for x in v3d]+[f"{v3d.mean():.4f}"])
    json.dump({"n_common":int(len(IDX)),"S_best":S_best,"BEST_FREE":BF,
               "eval_i90":{a:[float(x) if x==x else None for x in ev[a]] for a in ARMS},
               "V1_gap":v1g.tolist(),"V1_mean":float(v1g.mean()),
               "V3_delta":v3d.tolist(),"V3_mean":float(v3d.mean())},
              open(OUT/"RAF_DIRECTION.json","w"),indent=2,ensure_ascii=False)
    log("저장 완료")
