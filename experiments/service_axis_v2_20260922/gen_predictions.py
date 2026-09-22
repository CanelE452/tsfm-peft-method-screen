"""STEP 1-3: 전 창 x 전 arm 점예측 생성 (채점 없음). 결과는 .cache 에 저장."""
import sys, time, json
from pathlib import Path
import numpy as np, torch
R = Path("/home/minjae/Documents/github/tsfm-peft-method-screen")
sys.path.insert(0, str(R/"experiments/service_axis_v2_20260922"))
sys.path.insert(0, "/home/minjae/Documents/github/m5dataset")
import core
from common.data_split import get_split
CACHE = R/".cache/service_axis_v2_20260922"; CACHE.mkdir(parents=True, exist_ok=True)
(CACHE/"preds").mkdir(exist_ok=True)
H=28; CONTEXT=512; BS=256; TRAIN_END=1549
CAL_A=[1549,1577]; CAL_B=[1605,1633,1661,1689]; EVAL=[1717,1745,1773,1801,1829,1857,1885,1913]
WINDOWS = CAL_A + CAL_B + EVAL
def log(m): print(m, flush=True)

ids, train, test = get_split()
X = np.concatenate([train, test], 1)            # d_1..d_1941
elig = (X[:, :TRAIN_END] > 0).sum(1) >= 20
log(f"적격 {elig.sum()}/{len(elig)}  창 {len(WINDOWS)}개")
np.save(CACHE/"elig.npy", elig)

# ---------- STEP 1: TSFM ----------
def nz_patch():
    import chronos.chronos_bolt as cb
    def f(self, x, loc_scale=None):
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
    cb.InstanceNorm.forward = f

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
if MODE in ("tsfm_nz",): nz_patch()

if MODE in ("all", "tsfm_zs", "tsfm_nz"):
    from chronos import Chronos2Pipeline
    tag = "NZ" if MODE == "tsfm_nz" else "ZS"
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda", dtype=torch.bfloat16)
    qlev = list(pipe.quantiles)
    for o in WINDOWS:
        f_qm = CACHE/f"preds/{tag}_QMEAN21_o{o}.npy"
        if f_qm.exists(): log(f"  {tag} o{o} 이미 있음"); continue
        ctx = X[:, max(0,o-CONTEXT):o]
        inp = [torch.tensor(ctx[i], dtype=torch.float32) for i in range(ctx.shape[0])]
        t0=time.time()
        qp,_ = pipe.predict_quantiles(inp, prediction_length=H, quantile_levels=qlev, batch_size=BS)
        Q = np.stack([x[0].float().cpu().numpy() for x in qp])
        np.save(f_qm, core.qmean21(Q).astype(np.float32))
        np.save(CACHE/f"preds/{tag}_MED_o{o}.npy", np.clip(Q[:,:,qlev.index(0.5)],0,None).astype(np.float32))
        if o == 1913: np.save(CACHE/f"preds/{tag}_Q_full_o1913.npy", Q.astype(np.float32))
        log(f"  {tag} o{o}: {time.time()-t0:.1f}s  ctx {ctx.shape[1]}")
    log(f"STEP1 {tag} 완료")

# ---------- STEP 2: 고전 ----------
if MODE in ("all", "classical"):
    from statsforecast.models import SeasonalNaive, SimpleExponentialSmoothingOptimized, CrostonSBA, TSB
    mk = {"SNAIVE": lambda: SeasonalNaive(season_length=7),
          "SES":    lambda: SimpleExponentialSmoothingOptimized(),
          "SBA":    lambda: CrostonSBA(),
          "TSB":    lambda: TSB(alpha_d=0.1, alpha_p=0.1)}
    for o in WINDOWS:
        need = [m for m in mk if not (CACHE/f"preds/{m}_o{o}.npy").exists()]
        if not need: log(f"  classical o{o} 이미 있음"); continue
        t0=time.time(); res={m: np.zeros((len(X), H), np.float32) for m in need}
        for i in range(len(X)):
            y = X[i, :o].astype(np.float64)
            for m in need:
                res[m][i] = mk[m]().forecast(y=y, h=H)["mean"]
        for m in need: np.save(CACHE/f"preds/{m}_o{o}.npy", np.clip(res[m],0,None))
        log(f"  classical o{o}: {time.time()-t0:.1f}s  {need}")
    log("STEP2 완료")

# ---------- STEP 3: REF-LGB-U ----------
if MODE in ("all", "lgb"):
    import lightgbm as lgb
    LAGS=[1,7,14,28]; ROLL=[7,28,56,365]
    import pandas as pd
    cal = pd.read_csv("/home/minjae/Documents/github/m5dataset/data/calendar.csv").set_index("d")
    dow = np.array([cal["wday"][f"d_{i}"] for i in range(1,1942)], np.float32)
    FN=[f"lag{L}" for L in LAGS]+[f"rm{Rr}" for Rr in ROLL]+["zr56","zr365","sd365","dow","h"]
    def feats(o,h):
        hist=X[:,:o]; ti=o+h-1
        f=[hist[:,-L] for L in LAGS]+[hist[:,-Rr:].mean(1) for Rr in ROLL]
        f+=[(hist[:,-Rr:]==0).mean(1) for Rr in [56,365]]+[hist[:,-365:].std(1)]
        f+=[np.full(len(hist),dow[ti],np.float32), np.full(len(hist),h,np.float32)]
        return np.stack(f,1).astype(np.float32)
    def build(origins):
        Fs,Ys=[],[]
        for o in origins:
            for h in range(1,H+1): Fs.append(feats(o,h)); Ys.append(X[:,o+h-1])
        return np.vstack(Fs), np.concatenate(Ys)
    TR=[1521-28*k for k in range(12)]
    assert max(TR)+H <= TRAIN_END, TR
    t0=time.time(); Xtr,ytr=build(TR); log(f"  LGB train {Xtr.shape} {time.time()-t0:.0f}s")
    Xva,yva=build(CAL_A)
    m=lgb.train(dict(objective="tweedie",tweedie_variance_power=1.1,learning_rate=0.05,
                     feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,num_leaves=63,
                     min_data_in_leaf=100,seed=20260922,verbose=-1),
                lgb.Dataset(Xtr,ytr,feature_name=FN), num_boost_round=2000,
                valid_sets=[lgb.Dataset(Xva,yva,feature_name=FN)],
                callbacks=[lgb.early_stopping(100,verbose=False)])
    log(f"  best_iter {m.best_iteration}")
    for o in WINDOWS:
        Xte,_=build([o])
        p=np.clip(m.predict(Xte,num_iteration=m.best_iteration),0,None).reshape(H,-1).T
        np.save(CACHE/f"preds/REF-LGB-U_o{o}.npy", p.astype(np.float32))
    json.dump({"best_iter":int(m.best_iteration),"train_origins":TR,"features":FN,
               "importance":{k:float(v) for k,v in zip(FN,m.feature_importance("gain"))}},
              open(R/"results/service_axis_v2_20260922/REF_LGB_U.json","w"), indent=2)
    log("STEP3 완료")
log("ALL DONE")
