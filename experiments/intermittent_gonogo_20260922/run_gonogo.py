"""G1-(ii) + G2-A 판정. 논문 평가기 정의를 그대로 이식(점예측/σ̂/order-up-to)."""
import sys, time, json
from pathlib import Path
import numpy as np, torch
from scipy import stats as sps

M5=Path("/home/minjae/Documents/github/m5dataset"); sys.path.insert(0,str(M5))
from common.data_split import get_split
OUT=Path(__file__).resolve().parent
H=28; CONTEXT=512; BS=256; SPLIT_END=1885
LEAD_TIME=7; REVIEW_PERIOD=7; TAU=0.90          # 지시문 §5.5 [제안값]
ALPHA=np.round(np.arange(0.50,4.0001,0.05),2)   # §5.2 71점
QSPEC=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
def log(m): print(m,flush=True)

# ---------- STEP 1 데이터 ----------
log("[STEP 1] 데이터")
ids,train,test=get_split()
elig=(train[:,:SPLIT_END]>0).sum(1)>=20
log(f"  적격 {elig.sum()}/{len(elig)}")
ctx=train[:,-CONTEXT:]

# ---------- STEP 2 예측 (ZS / NZ) ----------
from chronos import Chronos2Pipeline
from chronos.utils import interpolate_quantiles

def patch_nz(pipeline):
    """논문 run_raf_instancenorm.py:154 patch_instancenorm 이식."""
    inz=pipeline.inner_model.instance_norm
    inz._orig=inz.forward
    def _f(x, loc_scale=None):
        od=x.dtype; xf=x.to(torch.float32)
        if loc_scale is None:
            nan=torch.isnan(xf); xc=torch.nan_to_num(xf,nan=0.0)
            nzm=(xc>0)&(~nan)
            cnt=nzm.float().sum(-1,keepdim=True).clamp(min=1.0)
            loc=(xc*nzm.float()).sum(-1,keepdim=True)/cnt
            sc=(((xc-loc)*nzm.float()).square().sum(-1,keepdim=True)/cnt.clamp(min=1.0)).sqrt()
            allz=(nzm.float().sum(-1)==0)
            if allz.any():
                ol=torch.nan_to_num(torch.nanmean(xf,dim=-1,keepdim=True),nan=0.0)
                os_=torch.nan_to_num((xf-ol).square().nanmean(-1,keepdim=True).sqrt(),nan=1.0)
                loc=torch.where(allz.unsqueeze(-1),ol,loc); sc=torch.where(allz.unsqueeze(-1),os_,sc)
            sc=torch.where(sc==0,torch.tensor(inz.eps,device=sc.device,dtype=sc.dtype),sc)
            loc_scale=(loc,sc)
        l,s=loc_scale; z=(xf-l)/s
        if inz.use_arcsinh: z=torch.arcsinh(z)
        return z.to(od),(l,s)
    inz.forward=_f; return pipeline

def infer(nz):
    p=Chronos2Pipeline.from_pretrained("amazon/chronos-2",device_map="cuda",dtype=torch.bfloat16)
    if nz: p=patch_nz(p)
    q=list(p.quantiles)
    inp=[torch.tensor(ctx[i],dtype=torch.float32) for i in range(ctx.shape[0])]
    t0=time.time()
    qp,_=p.predict_quantiles(inp,prediction_length=H,quantile_levels=q,batch_size=BS)
    torch.cuda.synchronize()
    Q=np.stack([x[0].float().cpu().numpy() for x in qp])
    log(f"  {'NZ' if nz else 'ZS'} 추론 {time.time()-t0:.1f}s  Q{Q.shape}")
    del p; torch.cuda.empty_cache()
    return Q,q

f_zs,f_nz=OUT/"Q_zs.npy",OUT/"Q_nz.npy"
if f_zs.exists() and f_nz.exists():
    Qzs,Qnz=np.load(f_zs),np.load(f_nz); qlev=None
    log("  저장된 예측 재사용")
else:
    log("[STEP 2] 추론")
    Qzs,qlev=infer(False); np.save(f_zs,Qzs.astype(np.float32))
    Qnz,_   =infer(True);  np.save(f_nz,Qnz.astype(np.float32))
if qlev is None:
    qlev=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]

# ---------- STEP 3 arm별 점예측 + σ̂ ----------
log("[STEP 3] arm 구성 (점예측 = 지시문 §5.1, σ̂ = 논문 IQR/1.35)")
def qmean(Q,lv):
    lv=np.asarray(lv,float)
    lv_e=np.concatenate([[0.],lv,[1.]])
    v_e=np.concatenate([Q[...,:1],Q,Q[...,-1:]],-1)
    return np.clip(np.trapezoid(v_e,x=lv_e,axis=-1),0,None)
def sig_iqr(Q,lv):
    nq=Q.shape[-1]; hi=Q[...,min(nq-1,int(nq*0.75))]; lo=Q[...,max(0,int(nq*0.25))]
    return np.maximum(np.maximum(0,hi-lo)/1.35,0.01)
def qat(Q,lv,t): return np.clip(interpolate_quantiles([t],lv,torch.from_numpy(Q)).numpy()[...,0],0,None)

sig_zs,sig_nz=sig_iqr(Qzs,qlev),sig_iqr(Qnz,qlev)
Qzs_s=interpolate_quantiles(QSPEC,qlev,torch.from_numpy(Qzs)).numpy()
Qnz_s=interpolate_quantiles(QSPEC,qlev,torch.from_numpy(Qnz)).numpy()
arms={
 "ZS-QMEAN":(qmean(Qzs_s,QSPEC),sig_zs),
 "ZS-Q80"  :(qat(Qzs,qlev,0.80),sig_zs),
 "ZS-Q90"  :(qat(Qzs,qlev,0.90),sig_zs),
 "NZ-QMEAN":(qmean(Qnz_s,QSPEC),sig_nz),
}
# SBA / TSB (원점 d_1913 이하 관측으로만 적합)
log("  SBA/TSB 적합")
hist=train  # d_1..d_1913
def croston_sba(x,a=0.1):
    nz=np.nonzero(x)[0]
    if len(nz)==0: return 0.0,0.01
    z=x[nz[0]]; p=1.0; last=nz[0]
    for t in nz[1:]:
        z+=a*(x[t]-z); p+=a*((t-last)-p); last=t
    return (1-a/2)*z/max(p,1e-9), max(np.std(x[nz])/1.35,0.01)
def tsb(x,ad=0.1,ap=0.1):
    z=0.0;pr=0.0;first=True
    for v in x:
        if v>0:
            if first: z=v;first=False
            else: z+=ad*(v-z)
            pr+=ap*(1-pr)
        else: pr+=ap*(0-pr)
    nz=x[x>0]
    return z*pr,(max(np.std(nz)/1.35,0.01) if len(nz) else 0.01)
t0=time.time(); sba=np.zeros(len(hist)); sba_s=np.zeros(len(hist))
tsbm=np.zeros(len(hist)); tsb_s=np.zeros(len(hist))
for i in range(len(hist)):
    sba[i],sba_s[i]=croston_sba(hist[i]); tsbm[i],tsb_s[i]=tsb(hist[i])
log(f"  SBA/TSB {time.time()-t0:.1f}s")
arms["SBA"]=(np.repeat(sba[:,None],H,1),np.repeat(sba_s[:,None],H,1))
arms["TSB"]=(np.repeat(tsbm[:,None],H,1),np.repeat(tsb_s[:,None],H,1))

# ---------- STEP 4 order-up-to 시뮬 (벡터화) ----------
log(f"[STEP 4] order-up-to (L={LEAD_TIME}, R={REVIEW_PERIOD}, tau={TAU}), α 격자 {len(ALPHA)}점")
Y=test[elig]; N=len(Y); z=sps.norm.ppf(TAU); prot=LEAD_TIME+REVIEW_PERIOD
def simulate(fmean,fstd):
    """fmean,fstd: (N,) 기간당. 반환 (fill, avg_onhand)"""
    S=np.maximum(0,fmean*prot+z*fstd*np.sqrt(prot)); s=np.maximum(0,S-fmean*REVIEW_PERIOD)
    oh=S.copy(); pipe=np.zeros((LEAD_TIME,N)); ps=np.zeros(N); pwd=np.zeros(N); ohs=np.zeros(N)
    for t in range(H):
        if LEAD_TIME>0:
            oh=oh+pipe[0]; pipe=np.roll(pipe,-1,0); pipe[-1]=0
        d=Y[:,t]; pos=d>0
        pwd+=pos; ps+=pos&(oh>=d); oh=np.maximum(0,oh-d)
        ip=oh+pipe.sum(0)
        pipe[-1]=np.where(ip<=s,np.maximum(0,S-ip),0.0)
        ohs+=oh
    fill=np.where(pwd>0,ps/np.maximum(pwd,1),1.0)
    return fill,ohs/H
curves={}
for name,(pm,ps_) in arms.items():
    fm=pm[elig].mean(1); fs=ps_[elig].mean(1)
    F=np.zeros((len(ALPHA),N)); I=np.zeros((len(ALPHA),N))
    for k,a in enumerate(ALPHA):
        F[k],I[k]=simulate(fm*a,fs*a)
    curves[name]=(F,I)
    log(f"  {name:10s} fill(α=1)={F[list(ALPHA).index(1.0)].mean():.4f}  max fill={F.mean(1).max():.4f}")

# ---------- STEP 5 Stock@90 + 판정 ----------
log("[STEP 5] 판정")
def stock_at(F,I,target=0.90):
    mf=F.mean(1); mi=I.mean(1)
    k=np.argmax(mf>=target) if (mf>=target).any() else -1
    if k<=0: return (mi[0],"reached_at_min") if k==0 else (np.nan,"not_reached")
    f0,f1=mf[k-1],mf[k]; w=(target-f0)/(f1-f0) if f1>f0 else 0.0
    return mi[k-1]+w*(mi[k]-mi[k-1]),"interp"
res={}
for n,(F,I) in curves.items():
    st,flag=stock_at(F,I); res[n]={"stock90":float(st) if st==st else None,"flag":flag,
        "fill_a1":float(F[list(ALPHA).index(1.0)].mean()),"max_fill":float(F.mean(1).max())}
    log(f"  {n:10s} Stock@90={st:.3f} ({flag})   fill(α=1)={res[n]['fill_a1']:.4f}")

# G1-(ii): ZS-QMEAN fill(α=1) - max(SBA,TSB fill) 의 CI 상한 < 0 ?
rng=np.random.default_rng(20260922); B=2000
i1=list(ALPHA).index(1.0)
zq=curves["ZS-QMEAN"][0][i1]; sb=curves["SBA"][0][i1]; tb=curves["TSB"][0][i1]
best=np.where(sb.mean()>=tb.mean(),sb,tb); bname="SBA" if sb.mean()>=tb.mean() else "TSB"
d_pt=zq.mean()-best.mean()
bs=np.array([ (zq[ix].mean()-best[ix].mean()) for ix in (rng.integers(0,N,N) for _ in range(B)) ])
lo,hi=np.percentile(bs,[2.5,97.5])
g1ii = hi<0
log(f"\n  G1-(ii) ZS-QMEAN fill {zq.mean():.4f} - {bname} fill {best.mean():.4f} = {d_pt:+.4f}")
log(f"          CI=[{lo:+.4f},{hi:+.4f}]  상한<0? {'YES -> 통과' if g1ii else 'NO -> 미통과'}")

# G2-A: ZS-best(Stock@90 최소) 대비 NZ-QMEAN
zs_c={k:res[k]["stock90"] for k in ["ZS-QMEAN","ZS-Q80","ZS-Q90"] if res[k]["stock90"] is not None}
g2a=None
if zs_c and res["NZ-QMEAN"]["stock90"] is not None:
    zb=min(zs_c,key=zs_c.get); base=zs_c[zb]; armv=res["NZ-QMEAN"]["stock90"]
    dstock=100*(base-armv)/base
    def boot_stock(F,I,ix): 
        return stock_at(F[:,ix],I[:,ix])[0]
    bsd=[]
    for _ in range(500):
        ix=rng.integers(0,N,N)
        b=boot_stock(*curves[zb],ix); a=boot_stock(*curves["NZ-QMEAN"],ix)
        if b==b and a==a and b>0: bsd.append(100*(b-a)/b)
    bsd=np.array(bsd); clo,chi=np.percentile(bsd,[2.5,97.5])
    g2a = (dstock>=3.0) and (clo>0)
    log(f"\n  G2-A  ZS-best={zb} Stock@90={base:.3f}  NZ-QMEAN={armv:.3f}")
    log(f"         ΔStock%={dstock:+.2f}%  CI=[{clo:+.2f},{chi:+.2f}]  (기준 ≥3.0 & CI하한>0)")
    log(f"         {'통과' if g2a else '미통과'}")
else:
    log("\n  G2-A 계산 불가 (Stock@90 미도달)")

tok = "G1ii=%s G2A=%s" % ("PASS" if g1ii else "FAIL","PASS" if g2a else ("FAIL" if g2a is not None else "NA"))
log(f"\n  >>> {tok}")
json.dump({"res":res,"g1_ii":{"delta":float(d_pt),"ci":[float(lo),float(hi)],"pass":bool(g1ii),"baseline":bname},
           "g2_a":({"zs_best":zb,"base":float(base),"arm":float(armv),"delta_pct":float(dstock),
                    "ci":[float(clo),float(chi)],"pass":bool(g2a)} if g2a is not None else None),
           "config":{"L":LEAD_TIME,"R":REVIEW_PERIOD,"tau":TAU,"n":int(N),"context":CONTEXT}},
          open(OUT/"gonogo.json","w"),indent=2)
np.savez_compressed(OUT/"curves.npz",alpha=ALPHA,**{f"{k}_fill":v[0].mean(1) for k,v in curves.items()},
                    **{f"{k}_inv":v[1].mean(1) for k,v in curves.items()})
log(f"[DONE] {OUT/'gonogo.json'}")
