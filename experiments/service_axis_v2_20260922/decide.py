"""STEP 4-6: CAL-B 채점 -> 봉인 -> EVAL 채점 (V1, V3). EVAL 은 봉인 후에만 연다."""
import sys, json, hashlib
from pathlib import Path
import numpy as np
R = Path("/home/minjae/Documents/github/tsfm-peft-method-screen")
sys.path.insert(0, str(R/"experiments/service_axis_v2_20260922"))
sys.path.insert(0, "/home/minjae/Documents/github/m5dataset")
import core
from common.data_split import get_split
CACHE = R/".cache/service_axis_v2_20260922"; OUT = R/"results/service_axis_v2_20260922"
H=28; L=7; RP=1; P=L+RP; TRAIN_END=1549
CAL_A=[1549,1577]; CAL_B=[1605,1633,1661,1689]; EVAL=[1717,1745,1773,1801,1829,1857,1885,1913]
TAUS=[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.975,0.99]
ARMS=["ZS-Q","NZ-Q","SNAIVE","SES","SBA","TSB","REF-LGB-U"]
FILE={"ZS-Q":"ZS_QMEAN21_o{o}.npy","NZ-Q":"NZ_QMEAN21_o{o}.npy","ZS-MED":"ZS_MED_o{o}.npy",
      "SNAIVE":"SNAIVE_o{o}.npy","SES":"SES_o{o}.npy","SBA":"SBA_o{o}.npy","TSB":"TSB_o{o}.npy",
      "REF-LGB-U":"REF-LGB-U_o{o}.npy"}
def log(m): print(m, flush=True)
ids, train, test = get_split()
X = np.concatenate([train, test], 1)
elig = np.load(CACHE/"elig.npy")

def load(arm,o): return np.load(CACHE/"preds"/FILE[arm].format(o=o))
def truth(o):    return X[:, o:o+H]

# ---------- 공통 계열 집합 (§5.4) ----------
valid = elig.copy(); excl={}
for arm in ARMS+["ZS-MED"]:
    bad=np.zeros(len(X),bool)
    for o in CAL_B+EVAL+CAL_A:
        p=load(arm,o); bad |= ~np.isfinite(p).all(1) | (p<-1e-9).any(1)
    excl[arm]=int((valid&bad).sum()); valid &= ~bad
scale=np.array([np.mean(np.diff(X[i,:TRAIN_END][np.nonzero(X[i,:TRAIN_END])[0][0]:])**2) if (X[i,:TRAIN_END]>0).any() else np.nan for i in range(len(X))])
valid &= np.isfinite(scale) & (scale>0)
log(f"공통 계열 {valid.sum()} (적격 {elig.sum()}), arm별 제외 {excl}")
IDX=np.where(valid)[0]

# ---------- 창 채점 ----------
def mu_of(arm,o): return load(arm,o)[IDX].mean(1)
def resid_of(arm):
    return core.erp_residuals([mu_of(arm,o) for o in CAL_B], [truth(o)[IDX] for o in CAL_B], P)
def curve(arm,o,resid):
    mu=mu_of(arm,o); Y=truth(o)[IDX]; U=[];I=[]
    for t in TAUS:
        S=core.erp_S(mu,resid,P,t); sv,dm,cl=core.simulate(S,L,Y)
        u,i=core.unit_metrics(sv,dm,cl); U.append(u); I.append(i)
    return np.array(U),np.array(I)

log("=== STEP 4: CAL-B 채점 (ERP 잔차, 임계값, S-best) ===")
RESID={arm:resid_of(arm) for arm in ARMS}
calb={}
for arm in ARMS:
    rows=[curve(arm,o,RESID[arm]) for o in CAL_B]
    i90=[core.interp_at(U,I)[0] for U,I in rows]
    calb[arm]={"i90_by_window":[None if x!=x else float(x) for x in i90],
               "mean_i90":float(np.nanmean(i90)),
               "nonmono":[core.nonmonotone(U) for U,_ in rows]}
    log(f"  {arm:10s} CAL-B I@90 평균 {calb[arm]['mean_i90']:.3f}  창별 {[None if x!=x else round(x,1) for x in i90]}")
sba,tsb=calb["SBA"]["mean_i90"],calb["TSB"]["mean_i90"]
d_ref=abs(sba-tsb)/((sba+tsb)/2)*100
T_stock=max(3.0,d_ref)
S_best=min(["SNAIVE","SES","SBA","TSB"],key=lambda k:calb[k]["mean_i90"])
log(f"  Δ_ref(SBA vs TSB) = {d_ref:.3f}%  ->  T_stock = {T_stock:.3f}%")
log(f"  S-best = {S_best}")
TH={"delta_ref_pct":float(d_ref),"T_stock":float(T_stock),"S_best":S_best,
    "calb_i90":{k:v for k,v in calb.items()},"taus":TAUS,"P":P,"L":L,"R":RP}
json.dump(TH,open(OUT/"THRESHOLDS.json","w"),indent=2,ensure_ascii=False)

log("=== STEP 5: 봉인 ===")
def h_(o):
    return hashlib.sha256(json.dumps(o,sort_keys=True,default=str).encode()).hexdigest()[:32]
SEAL={"T_stock":float(T_stock),"S_best":S_best,"arms":ARMS,"taus":TAUS,
      "windows":{"CAL_A":CAL_A,"CAL_B":CAL_B,"EVAL":EVAL},"n_common":int(valid.sum()),
      "resid_hash":{a:hashlib.sha256(RESID[a].tobytes()).hexdigest()[:16] for a in ARMS},
      "thresholds_hash":h_(TH),
      "code_hash":hashlib.sha256((R/"experiments/service_axis_v2_20260922/decide.py").read_bytes()).hexdigest()[:16]}
json.dump(SEAL,open(OUT/"SEAL.json","w"),indent=2,ensure_ascii=False)
log(f"  SEAL 저장 (n_common {valid.sum()}, T_stock {T_stock:.3f})")

log("=== STEP 6: EVAL 채점 ===")
EV={}
for arm in ARMS:
    rows=[curve(arm,o,RESID[arm]) for o in EVAL]
    EV[arm]={"i90":[core.interp_at(U,I)[0] for U,I in rows],
             "flags":[core.interp_at(U,I)[1] for U,I in rows],
             "U":[U.tolist() for U,_ in rows],"I":[I.tolist() for _,I in rows]}
    m=np.nanmean(EV[arm]["i90"])
    log(f"  {arm:10s} EVAL I@90 평균 {m:.3f}  미도달 {sum(1 for f in EV[arm]['flags'] if f=='not_reached')}/8")

def delta(base,arm):
    b=np.array(EV[base]["i90"],float); a=np.array(EV[arm]["i90"],float)
    return 100*(b-a)/b

# V1
v1_gap=-delta(S_best,"ZS-Q")
V1=core.verdict(v1_gap,T_stock)
log(f"\n[V1] 기준 {S_best} vs ZS-Q  격차 평균 {V1['mean'] if V1['mean'] is not None else 'NA'}")
log(f"     {V1}")
# V3
FREE=["ZS-Q","NZ-Q","SNAIVE","SES","SBA","TSB"]
BEST_FREE=min(FREE,key=lambda k:np.nanmean(EV[k]["i90"]))
v3_d=delta(BEST_FREE,"REF-LGB-U")
V3=core.verdict(v3_d,T_stock)
log(f"\n[V3] BEST-FREE = {BEST_FREE} (EVAL 평균 최소)  arm REF-LGB-U")
log(f"     Δ_w = {[round(x,2) if x==x else None for x in v3_d]}")
log(f"     {V3}")

import csv
for name,base,arm,dd in [("DECISION_V1.csv",S_best,"ZS-Q",v1_gap),("DECISION_V3.csv",BEST_FREE,"REF-LGB-U",v3_d)]:
    with open(OUT/name,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["window",f"I@90_{base}",f"I@90_{arm}","delta_pct"])
        for k,o in enumerate(EVAL):
            w.writerow([f"d_{o+1}-d_{o+H}",f"{EV[base]['i90'][k]:.4f}",f"{EV[arm]['i90'][k]:.4f}",f"{dd[k]:.4f}"])
        V=V1 if "V1" in name else V3
        w.writerow(["MEAN","","",f"{V['mean']:.4f}" if V['mean'] is not None else "NA"])
        w.writerow(["t_low_95","","",f"{V['t_low']:.4f}" if V['t_low'] is not None else "NA"])
        w.writerow(["n_pos","","",V['n_pos']]); w.writerow(["T_stock","","",f"{T_stock:.4f}"])
        w.writerow(["PASS","","",V['pass']])
json.dump({"V1":{"baseline":S_best,"gap":v1_gap.tolist(),**V1},
           "V3":{"baseline":BEST_FREE,"delta":v3_d.tolist(),**V3},
           "eval_i90":{k:EV[k]["i90"] for k in ARMS},
           "eval_flags":{k:EV[k]["flags"] for k in ARMS}},
          open(OUT/"DECISION.json","w"),indent=2,ensure_ascii=False)
np.savez_compressed(CACHE/"eval_curves.npz",**{f"{a}_{t}":np.array(EV[a][t]) for a in ARMS for t in ["U","I"]})
log(f"\n>>> V1 {'PASS' if V1['pass'] else 'FAIL'} / V3 {'PASS' if V3['pass'] else 'FAIL'}")
