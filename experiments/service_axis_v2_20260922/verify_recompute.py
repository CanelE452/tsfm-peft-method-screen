"""§9 VERIFY_RECOMPUTE: 저장된 곡선에서 DECISION/LADDER 표를 numpy 만으로 독립 재계산."""
import json
from pathlib import Path
import numpy as np
R=Path("/home/minjae/Documents/github/tsfm-peft-method-screen")
OUT=R/"results/service_axis_v2_20260922"; CACHE=R/".cache/service_axis_v2_20260922"
TAUS=[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.975,0.99]
c=np.load(CACHE/"eval_curves.npz"); dec=json.load(open(OUT/"DECISION.json"))
def i90_indep(U,I):
    """독립 구현: 첫 교차 선형보간 (core.interp_at 를 쓰지 않음)"""
    for k in range(len(U)):
        if U[k]>=0.90:
            if k==0: return float(I[0])
            return float(I[k-1]+(0.90-U[k-1])/(U[k]-U[k-1])*(I[k]-I[k-1]))
    return float("nan")
diffs={}; worst=0.0
for arm in ["ZS-Q","NZ-Q","SNAIVE","SES","SBA","TSB","REF-LGB-U"]:
    U=c[f"{arm}_U"]; I=c[f"{arm}_I"]
    mine=[i90_indep(U[w],I[w]) for w in range(U.shape[0])]
    theirs=dec["eval_i90"][arm]
    d=max(abs(a-b) for a,b in zip(mine,theirs)); diffs[arm]=d; worst=max(worst,d)
# Δ 재계산
def delta(b,a):
    B=np.array(dec["eval_i90"][b]); A=np.array(dec["eval_i90"][a]); return 100*(B-A)/B
v1=-delta(dec["V1"]["baseline"],"ZS-Q"); v3=delta(dec["V3"]["baseline"],"REF-LGB-U")
d1=float(np.abs(v1-np.array(dec["V1"]["gap"])).max()); d3=float(np.abs(v3-np.array(dec["V3"]["delta"])).max())
# t-구간 재계산
def tlow(x):
    x=np.asarray(x); return float(x.mean()-2.364624*x.std(ddof=1)/np.sqrt(len(x)))
dt1=abs(tlow(v1)-dec["V1"]["t_low"]); dt3=abs(tlow(v3)-dec["V3"]["t_low"])
worst=max(worst,d1,d3,dt1,dt3)
res={"max_abs_diff":float(worst),"i90_diff_by_arm":{k:float(v) for k,v in diffs.items()},
     "V1_delta_max_diff":d1,"V3_delta_max_diff":d3,"V1_tlow_diff":float(dt1),"V3_tlow_diff":float(dt3),
     "criterion":"<= 1e-9","pass":bool(worst<=1e-9)}
json.dump(res,open(OUT/"VERIFY_RECOMPUTE.json","w"),indent=2)
print(f"독립 재계산 최대 차이 {worst:.3e}  (기준 <=1e-9)  {'PASS' if worst<=1e-9 else 'FAIL'}")
for k,v in diffs.items(): print(f"  {k:10s} {v:.3e}")
