"""§8.3 함께 보고: 12창(CAL-B+EVAL) x arm 의 PRB. '체계적 과소예측' 사전 규칙 적용."""
import sys, json, csv
from pathlib import Path
import numpy as np
R=Path("/home/minjae/Documents/github/tsfm-peft-method-screen")
sys.path.insert(0,str(R/"experiments/service_axis_v2_20260922")); sys.path.insert(0,"/home/minjae/Documents/github/m5dataset")
import core
from common.data_split import get_split
CACHE=R/".cache/service_axis_v2_20260922"; OUT=R/"results/service_axis_v2_20260922"
H=28; CAL_B=[1605,1633,1661,1689]; EVAL=[1717,1745,1773,1801,1829,1857,1885,1913]; WIN=CAL_B+EVAL
F={"ZS-MED":"ZS_MED_o{o}.npy","ZS-Q":"ZS_QMEAN21_o{o}.npy","NZ-Q":"NZ_QMEAN21_o{o}.npy",
   "SNAIVE":"SNAIVE_o{o}.npy","SES":"SES_o{o}.npy","SBA":"SBA_o{o}.npy","TSB":"TSB_o{o}.npy",
   "REF-LGB-U":"REF-LGB-U_o{o}.npy"}
ids,train,test=get_split(); X=np.concatenate([train,test],1)
elig=np.load(CACHE/"elig.npy"); IDX=np.where(elig)[0]
rows={}
for a,f in F.items():
    prbs=[]
    for o in WIN:
        p=np.load(CACHE/"preds"/f.format(o=o))[IDX]; y=X[IDX,o:o+H]
        prbs.append(float((p-y).sum()/y.sum()))
    rows[a]=prbs
# 사전 규칙: 12창 중 10창 이상 음수이고 t-구간 상한 < 0
def systematic(v):
    v=np.array(v); n_neg=int((v<0).sum()); m=v.mean(); sd=v.std(ddof=1)
    hi=m+core.T_CRIT_DF7*sd/np.sqrt(len(v))  # 보수적으로 df=7 임계 사용
    return {"n_neg":n_neg,"mean":float(m),"t_high":float(hi),
            "systematic":bool(n_neg>=10 and hi<0)}
summ={a:systematic(v) for a,v in rows.items()}
with open(OUT/"PRB_BY_WINDOW.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["arm"]+[f"o{o}" for o in WIN]+["mean","n_neg","t_high","systematic"])
    for a in F:
        s=summ[a]; w.writerow([a]+[f"{x:.6f}" for x in rows[a]]+
                              [f"{s['mean']:.6f}",s["n_neg"],f"{s['t_high']:.6f}",s["systematic"]])
for a in F:
    s=summ[a]; print(f"  {a:10s} PRB 평균 {s['mean']:+.4f}  음수창 {s['n_neg']}/12  t상한 {s['t_high']:+.4f}  체계적 {s['systematic']}",flush=True)
json.dump({"windows":WIN,"prb":rows,"summary":summ},open(OUT/"PRB_BY_WINDOW.json","w"),indent=2,ensure_ascii=False)
print("저장 완료")
