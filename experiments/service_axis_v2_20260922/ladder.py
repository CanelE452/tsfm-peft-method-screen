"""V2 사다리 L0-L6: v1 +43.78% 가 어디서 왔는지 한 번에 한 요인만 바꿔 측정 (보고 전용)."""
import sys, json, csv
from pathlib import Path
import numpy as np
from scipy import stats as sps
R=Path("/home/minjae/Documents/github/tsfm-peft-method-screen")
sys.path.insert(0,str(R/"experiments/service_axis_v2_20260922")); sys.path.insert(0,"/home/minjae/Documents/github/m5dataset")
import core
from common.data_split import get_split
CACHE=R/".cache/service_axis_v2_20260922"; V1C=R/".cache/intermittent_gonogo_20260922"; OUT=R/"results/service_axis_v2_20260922"
QTR=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
QSPEC=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
H=28; ALPHA=np.round(np.arange(0.50,4.0001,0.05),2)
CAL_B=[1605,1633,1661,1689]; EVAL=[1717,1745,1773,1801,1829,1857,1885,1913]
TAUS=[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.975,0.99]
def log(m): print(m,flush=True)
ids,train,test=get_split(); X=np.concatenate([train,test],1)
Qzs=np.load(V1C/"Q_zs.npy"); Qnz=np.load(V1C/"Q_nz.npy")
elig_v1=(train[:,:1885]>0).sum(1)>=20; Y1913=test[elig_v1]; N=len(Y1913)

def qmean13(Q):
    idx=[QTR.index(q) for q in QSPEC]; Qs=Q[:,:,idx]
    lv=np.asarray(QSPEC,float); lve=np.concatenate([[0.],lv,[1.]])
    ve=np.concatenate([Qs[...,:1],Qs,Qs[...,-1:]],-1)
    return np.clip(np.trapezoid(ve,x=lve,axis=-1),0,None)
def sig_iqr(Q,clip=False):
    Qc=np.clip(Q,0,None) if clip else Q; nq=Qc.shape[-1]
    return np.maximum(np.maximum(0,Qc[...,int(nq*0.75)]-Qc[...,int(nq*0.25)])/1.35,0.01)
def pmq_sim(fm,fs,Y,L=7,RP=7,tau=0.90):
    prot=L+RP; z=sps.norm.ppf(tau); n=len(Y)
    S=np.maximum(0,fm*prot+z*fs*np.sqrt(prot)); s=np.maximum(0,S-fm*RP)
    oh=S.copy(); pipe=np.zeros((L,n)); ps=np.zeros(n); pwd=np.zeros(n); ohs=np.zeros(n); sv=np.zeros(n)
    for t in range(H):
        oh=oh+pipe[0]; pipe=np.roll(pipe,-1,0); pipe[-1]=0
        d=Y[:,t]; pos=d>0; pwd+=pos; ps+=pos&(oh>=d)
        sv+=np.minimum(oh,d); oh=np.maximum(0,oh-d)
        ip=oh+pipe.sum(0); pipe[-1]=np.where(ip<=s,np.maximum(0,S-ip),0.0); ohs+=oh
    return np.where(pwd>0,ps/np.maximum(pwd,1),1.0), ohs/H, sv, Y.sum(1)

def stock_eqw(fm,fs,Y):   # 동일가중 충족 + 절대재고 (v1)
    F=np.zeros((len(ALPHA),len(Y))); I=np.zeros((len(ALPHA),len(Y)))
    for k,a in enumerate(ALPHA): F[k],I[k],_,_=pmq_sim(fm*a,fs*a,Y)
    return core.interp_at(F.mean(1),I.mean(1))[0]
def stock_unit(fm,fs,Y):  # 단위가중 (U, I)
    U=np.zeros(len(ALPHA)); I=np.zeros(len(ALPHA))
    for k,a in enumerate(ALPHA):
        _,cl,sv,dm=pmq_sim(fm*a,fs*a,Y); U[k]=sv.sum()/dm.sum(); I[k]=cl.sum()
    return core.interp_at(U,I)[0]

rows=[]
def add(step,zs_st,nz_st,note):
    g=100*(zs_st-nz_st)/zs_st
    rows.append({"step":step,"ZS_stock":float(zs_st),"NZ_stock":float(nz_st),"gap_pct":float(g),"note":note})
    log(f"  {step}: ZS {zs_st:.3f}  NZ {nz_st:.3f}  격차 {g:+.2f}%   {note}")
    return g

log("=== V2 사다리 (원점 1913) ===")
zs13,nz13=qmean13(Qzs),qmean13(Qnz); szs,snz=sig_iqr(Qzs),sig_iqr(Qnz)
zs80=np.clip(np.stack([np.interp(0.80,QTR,Qzs[a,b]) for a in range(Qzs.shape[0]) for b in range(28)]).reshape(Qzs.shape[:2]),0,None)
zs90=np.clip(np.stack([np.interp(0.90,QTR,Qzs[a,b]) for a in range(Qzs.shape[0]) for b in range(28)]).reshape(Qzs.shape[:2]),0,None)
# L0: v1 그대로 (ZS-best = {QMEAN13,Q80,Q90} 중 최소)
cands={"ZS-QMEAN":stock_eqw(zs13[elig_v1].mean(1),szs[elig_v1].mean(1),Y1913),
       "ZS-Q80":stock_eqw(zs80[elig_v1].mean(1),szs[elig_v1].mean(1),Y1913),
       "ZS-Q90":stock_eqw(zs90[elig_v1].mean(1),szs[elig_v1].mean(1),Y1913)}
zb=min([k for k,v in cands.items() if v==v],key=lambda k:cands[k])
nz_l0=stock_eqw(nz13[elig_v1].mean(1),snz[elig_v1].mean(1),Y1913)
g0=add("L0",cands[zb],nz_l0,f"v1 그대로 (ZS-best={zb})")
# L1: 단위가중
g1=add("L1",stock_unit(zs13[elig_v1].mean(1),szs[elig_v1].mean(1),Y1913),
            stock_unit(nz13[elig_v1].mean(1),snz[elig_v1].mean(1),Y1913),"+ 단위가중 (U,I)")
# L2: sigma clip
g2=add("L2",stock_unit(zs13[elig_v1].mean(1),sig_iqr(Qzs,True)[elig_v1].mean(1),Y1913),
            stock_unit(nz13[elig_v1].mean(1),sig_iqr(Qnz,True)[elig_v1].mean(1),Y1913),"+ sigma_clip (음수분위수 제거)")
# L3: QMEAN21
zs21,nz21=core.qmean21(Qzs),core.qmean21(Qnz)
g3=add("L3",stock_unit(zs21[elig_v1].mean(1),sig_iqr(Qzs,True)[elig_v1].mean(1),Y1913),
            stock_unit(nz21[elig_v1].mean(1),sig_iqr(Qnz,True)[elig_v1].mean(1),Y1913),"+ QMEAN21")
# L4: ZS -> ZS x alpha_class (SBC 유형별, CAL-B 4창)
hist=X[:,:1549]
adi=np.full(len(hist),np.nan); cv2=np.full(len(hist),np.nan)
for i,r in enumerate(hist):
    nz=np.nonzero(r)[0]
    if len(nz)==0: continue
    adi[i]=(nz[-1]-nz[0]+1)/len(nz); pos=r[r>0]; cv2[i]=(pos.std(ddof=0)/pos.mean())**2 if pos.mean()>0 else 0.0
lab=np.full(len(adi),"none",dtype=object); ok=~np.isnan(adi)
lab[ok&(adi<1.32)&(cv2<0.49)]="smooth"; lab[ok&(adi<1.32)&(cv2>=0.49)]="erratic"
lab[ok&(adi>=1.32)&(cv2<0.49)]="intermittent"; lab[ok&(adi>=1.32)&(cv2>=0.49)]="lumpy"
ac={}
for g in ["smooth","erratic","intermittent","lumpy"]:
    m=(lab==g)&elig_v1
    num=sum(X[m,o:o+H].sum() for o in CAL_B); den=sum(np.load(CACHE/f"preds/ZS_QMEAN21_o{o}.npy")[m].sum() for o in CAL_B)
    ac[g]=float(num/den) if den>0 else 1.0
log(f"  alpha_class: { {k:round(v,4) for k,v in ac.items()} }")
aw=np.array([ac.get(l,1.0) for l in lab])[elig_v1]
g4=add("L4",stock_unit(zs21[elig_v1].mean(1)*aw,sig_iqr(Qzs,True)[elig_v1].mean(1)*aw,Y1913),
            stock_unit(nz21[elig_v1].mean(1),sig_iqr(Qnz,True)[elig_v1].mean(1),Y1913),"+ ZS->ZS_alpha_class")
# L5: ERP (L=7,R=1) 원점 1913
elig_v2=np.load(CACHE/"elig.npy"); IDX=np.where(elig_v2)[0]
def erp_stock(arm,o,resid):
    mu=np.load(CACHE/f"preds/{arm}_o{o}.npy")[IDX].mean(1); Yw=X[IDX,o:o+H]
    U=[];I=[]
    for t in TAUS:
        S=core.erp_S(mu,resid,8,t); sv,dm,cl=core.simulate(S,7,Yw)
        u,i=core.unit_metrics(sv,dm,cl); U.append(u); I.append(i)
    return core.interp_at(np.array(U),np.array(I))[0]
def resid_for(arm):
    return core.erp_residuals([np.load(CACHE/f"preds/{arm}_o{o}.npy")[IDX].mean(1) for o in CAL_B],
                              [X[IDX,o:o+H] for o in CAL_B],8)
rz,rn=resid_for("ZS_QMEAN21"),resid_for("NZ_QMEAN21")
g5=add("L5",erp_stock("ZS_QMEAN21",1913,rz),erp_stock("NZ_QMEAN21",1913,rn),"ERP 로 교체 (L=7,R=1, alpha 없음)")
# L6: EVAL 8창
zs_e=[erp_stock("ZS_QMEAN21",o,rz) for o in EVAL]; nz_e=[erp_stock("NZ_QMEAN21",o,rn) for o in EVAL]
g6=add("L6",float(np.nanmean(zs_e)),float(np.nanmean(nz_e)),"EVAL 8창 평균")
dw=100*(np.array(zs_e)-np.array(nz_e))/np.array(zs_e)
V=core.verdict(dw,json.load(open(OUT/"THRESHOLDS.json"))["T_stock"])
log(f"  L6 창별 Δ={[round(x,2) for x in dw]}  -> {'PASS (E1: NZ 가 정의 이상의 정보)' if V['pass'] else 'FAIL'}")
# 주원인
main=None
for i in range(1,len(rows)):
    if abs(rows[i]["gap_pct"]) <= 0.5*abs(rows[0]["gap_pct"]): main=rows[i]["step"]; break
log(f"  주원인(격차가 L0의 50% 이하로 처음 떨어진 단계): {main}")
with open(OUT/"LADDER_V2.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["step","ZS_stock","NZ_stock","gap_pct","delta_vs_prev","note"])
    for i,r in enumerate(rows):
        w.writerow([r["step"],f"{r['ZS_stock']:.4f}",f"{r['NZ_stock']:.4f}",f"{r['gap_pct']:.4f}",
                    "" if i==0 else f"{r['gap_pct']-rows[i-1]['gap_pct']:+.4f}", r["note"]])
json.dump({"rows":rows,"main_cause":main,"L6_window_deltas":dw.tolist(),"L6_verdict":V,
           "alpha_class":ac},open(OUT/"LADDER_V2.json","w"),indent=2,ensure_ascii=False)
log("저장 완료")
