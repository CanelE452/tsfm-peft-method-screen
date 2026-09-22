"""G1 조건 (i)(iii)의 계열 부트스트랩 CI. 지시문 §5.6: 2000회, seed 20260922, percentile 95%."""
import numpy as np, sys, json
from pathlib import Path
M5=Path("/home/minjae/Documents/github/m5dataset"); sys.path.insert(0,str(M5))
from common.data_split import get_split
OUT=Path(__file__).resolve().parent
_,train,test=get_split()
d=np.load(OUT/"point_forecasts.npz")
J=json.load(open(OUT/"precheck_a.json"))

SPLIT_END=1885; SBC_ADI, SBC_CV2 = 1.32, 0.49
hist=train[:,:SPLIT_END]
adi=np.full(len(hist),np.nan); cv2=np.full(len(hist),np.nan)
for i,r in enumerate(hist):
    nz=np.nonzero(r)[0]
    if len(nz)==0: continue
    adi[i]=(nz[-1]-nz[0]+1)/len(nz)          # 논문 정의
    pos=r[r>0]; cv2[i]=(pos.std(ddof=0)/pos.mean())**2 if pos.mean()>0 else 0.0
lab=np.full(len(adi),"none",dtype=object); ok=~np.isnan(adi)
lab[ok&(adi< SBC_ADI)&(cv2< SBC_CV2)]="smooth";       lab[ok&(adi< SBC_ADI)&(cv2>=SBC_CV2)]="erratic"
lab[ok&(adi>=SBC_ADI)&(cv2< SBC_CV2)]="intermittent"; lab[ok&(adi>=SBC_ADI)&(cv2>=SBC_CV2)]="lumpy"
elig=(hist>0).sum(1)>=20
idx_all=np.where(elig)[0]
idx_il =np.where(elig&((lab=="intermittent")|(lab=="lumpy")))[0]
idx_sm =np.where(elig&(lab=="smooth"))[0]

rng=np.random.default_rng(20260922); B=2000
def prb_idx(p,idx): 
    y=test[idx]; return (p[idx]-y).sum()/y.sum()

print(f"n: all={len(idx_all)} int+lumpy={len(idx_il)} smooth={len(idx_sm)}\n")
print(f"{'arm':16s} {'PRB':>9s} {'CI_low':>9s} {'CI_high':>9s}  {'상한<0?':>7s}")
res={}
for k in ["MED","A_spec_const","B_spec_linear","C_train_const","SNAIVE_week"]:
    p=d[k]; pt=prb_idx(p,idx_all)
    bs=np.array([prb_idx(p,idx_all[rng.integers(0,len(idx_all),len(idx_all))]) for _ in range(B)])
    lo,hi=np.percentile(bs,[2.5,97.5]); res[k]=(pt,lo,hi)
    print(f"{k:16s} {pt:+9.4f} {lo:+9.4f} {hi:+9.4f}  {'YES' if hi<0 else 'NO'}")

print(f"\n=== G1 조건 (iii): (int+lumpy) PRB - smooth PRB ===")
for k in ["MED","A_spec_const"]:
    p=d[k]
    pt=prb_idx(p,idx_il)-prb_idx(p,idx_sm)
    bs=np.array([prb_idx(p,idx_il[rng.integers(0,len(idx_il),len(idx_il))])
                -prb_idx(p,idx_sm[rng.integers(0,len(idx_sm),len(idx_sm))]) for _ in range(B)])
    lo,hi=np.percentile(bs,[2.5,97.5])
    print(f"{k:16s} diff={pt:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  상한<0? {'YES' if hi<0 else 'NO'}")
    print(f"{'':16s} int+lumpy={prb_idx(p,idx_il):+.4f}  smooth={prb_idx(p,idx_sm):+.4f}")

print(f"\n=== 지시문 §6 G1 추출교란 판정 ===")
med_hi=res["MED"][2]; qm=res["A_spec_const"]
print(f"ZS-MED   PRB CI 상한 {med_hi:+.4f}  ->  <0 ? {'YES' if med_hi<0 else 'NO'}")
print(f"ZS-QMEAN PRB CI      [{qm[1]:+.4f},{qm[2]:+.4f}]  ->  0 포함 or 양수 ? {'YES' if qm[2]>=0 else 'NO'}")
print("판정:", "EXTRACTION_ONLY (M5는 통과로 치지 않음)" if (med_hi<0 and qm[2]>=0) else "추출 교란 아님 — 조건(i) 충족")
json.dump({k:{"prb":float(v[0]),"ci":[float(v[1]),float(v[2])]} for k,v in res.items()}, open(OUT/"ci.json","w"), indent=2)
