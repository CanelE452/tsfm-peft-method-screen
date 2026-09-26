from __future__ import annotations
import argparse,gzip,hashlib,json,math,shutil,subprocess,time
from pathlib import Path
import numpy as np
from scipy.stats import norm

class E(RuntimeError): pass
def req(x,m):
    if not x: raise E(m)
def sh(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def blobsha(p):
    b=Path(p).read_bytes()
    return hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest()
def rj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def wj(p,o):
    Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False,
        default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def git(a,cwd):
    p=subprocess.run(['git',*a],cwd=cwd,text=True,capture_output=True)
    return p.returncode,p.stdout,p.stderr
def origins(a,b,s,H):
    x=np.arange(a,b,s,dtype=int);return x[x+H<=b]

def rawpit(q,levels,y):
    lev=np.asarray(levels,float);z=np.empty_like(y,dtype=float)
    for ix in np.ndindex(y.shape):
        o,c,h=ix;qq=q[o,c,h];yy=y[ix]
        uq,j=np.unique(qq,return_index=True);ul=lev[j]
        z[ix]=.5 if len(uq)<2 else np.interp(yy,uq,ul,left=ul[0]*.5,right=1-(1-ul[-1])*.5)
    return z

class Emp:
    def __init__(self,X): self.cols=[np.sort(X[:,h]) for h in range(X.shape[1])]
    def u(self,X):
        z=np.empty_like(X,dtype=float)
        for h,a in enumerate(self.cols):
            z[:,h]=(np.searchsorted(a,X[:,h],side='right')+.5)/(len(a)+1)
        return z
    def z(self,X,clip):
        return norm.ppf(np.clip(self.u(X),clip,1-clip))

class PitCal:
    def __init__(self,P):self.cols=[np.sort(P[:,:,h].ravel()) for h in range(P.shape[2])]
    def u(self,P):
        Z=np.empty_like(P,dtype=float)
        for h,a in enumerate(self.cols):
            for o in range(P.shape[0]):
                for c in range(P.shape[1]):
                    Z[o,c,h]=(np.searchsorted(a,P[o,c,h],side='right')+.5)/(len(a)+1)
        return Z

def corr(Z,shrink):
    R=np.corrcoef(Z,rowvar=False);R=np.nan_to_num(R,nan=0.);np.fill_diagonal(R,1)
    R=(1-shrink)*R+shrink*np.eye(R.shape[0])
    w,V=np.linalg.eigh((R+R.T)/2);w=np.maximum(w,1e-6);R=(V*w)@V.T
    d=np.sqrt(np.diag(R));R=R/d[:,None]/d[None,:];np.fill_diagonal(R,1)
    return R
def fro(A,B):
    return float(np.linalg.norm(A-B,'fro')/max(np.linalg.norm(B,'fro'),1e-12))
def lagprof(R,lags):
    return {str(k):float(np.mean(np.diag(R,k))) for k in lags}
def lagmae(A,B,lags):
    return float(np.mean([abs(np.mean(np.diag(A,k))-np.mean(np.diag(B,k))) for k in lags]))
def agg(R):
    one=np.ones(R.shape[0]);return float(one@R@one/R.shape[0])
def effrank(R):
    w=np.maximum(np.linalg.eigvalsh(R),1e-12);p=w/w.sum()
    return float(np.exp(-(p*np.log(p)).sum()))
def ks_uniform(u):
    x=np.sort(np.asarray(u).ravel());n=len(x)
    return float(max(np.max(np.arange(1,n+1)/n-x),np.max(x-np.arange(n)/n)))
def pinball(q,levels,y):
    lev=np.asarray(levels,float);e=y[...,None]-q
    return float(np.mean(np.maximum(lev*e,(lev-1)*e)))

def read_cases(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def block_ci(rows,c,base,new,reps,L,seed,level):
    rr=[r for r in rows if r['channel']==c]
    A=np.array([r[base]['crps'] for r in rr]);B=np.array([r[new]['crps'] for r in rr]);n=len(A)
    def gain(a,b):return 100*(a-b)/a
    rng=np.random.default_rng(seed+c*97);vals=[]
    for _ in range(reps):
        ix=[]
        while len(ix)<n:
            s=int(rng.integers(0,n));ix.extend([(s+j)%n for j in range(L)])
        ix=np.asarray(ix[:n]);vals.append(gain(A[ix].mean(),B[ix].mean()))
    a=(1-level)/2
    return {'gain_pct':gain(A.mean(),B.mean()),'ci_low':float(np.quantile(vals,a)),
            'ci_high':float(np.quantile(vals,1-a))}
def case_summary(rows,c,m):
    rr=[r for r in rows if r['channel']==c]
    return {k:float(np.mean([r[m][k] for r in rr])) for k in rr[0][m]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--package-root',required=True);ap.add_argument('--publish',action='store_true')
    a=ap.parse_args();repo=Path(a.repo).resolve();pkg=Path(a.package_root).resolve();cfg=rj(pkg/'RUN_CONFIG.json')
    src=repo/cfg['source_2x2_run']/'FINAL_DECISION.json'
    req(src.is_file() and blobsha(src)==cfg['source_2x2_final_git_blob_sha'],'2x2 source FINAL mismatch')
    cases=read_cases(repo/cfg['source_2x2_run']/'PILOT_CASES.jsonl')
    calcases=read_cases(repo/cfg['source_2x2_run']/'CAL_CASES.jsonl')
    # locate exact forecast cache
    hits=[]
    for p in (repo/'.cache').rglob('*.npz'):
        try:
            if sh(p)==cfg['forecast_cache_sha256']:hits.append(p)
        except:pass
    req(len(hits)==1,f'cache match count={len(hits)}');cache=hits[0];D=np.load(cache)
    keys=['q_train','y_train','q_cal','y_cal','q_pilot','y_pilot']
    req(all(k in D.files for k in keys),'cache keys missing')
    qtr,ytr,qc,yc,qp,yp=[D[k].astype(float) for k in keys]
    # raw prefix only
    rp=repo/cfg['raw_data_path'];req(rp.is_file() and sh(rp)==cfg['raw_data_sha256'],'raw data mismatch')
    with gzip.open(rp,'rt',encoding='utf-8') as f:
        raw=np.loadtxt(f,delimiter=',',usecols=[0,1,2,3],max_rows=7200,dtype=float)
    req(raw.shape==(7200,4),'raw prefix shape')
    s=cfg['splits'];H=s['horizon']
    ot=origins(s['train_start'],s['train_stop'],s['stride'],H);oc=origins(s['cal_start'],s['cal_stop'],s['stride'],H);op=origins(s['pilot_start'],s['pilot_stop'],s['stride'],H)
    req((len(ot),len(oc),len(op))==(len(ytr),len(yc),len(yp)),'origin mismatch')
    def Y(oo):return np.stack([[raw[o:o+H,c] for c in range(4)] for o in oo])
    rtr,rc,rpilo=Y(ot),Y(oc),Y(op);req(np.allclose(rtr,ytr) and np.allclose(rc,yc) and np.allclose(rpilo,yp),'truth mismatch')
    levels=rj(repo/cfg['source_forecast_manifest'])['config']['model']['quantiles']
    clip=cfg['analysis']['pit_clip'];shrink=cfg['analysis']['corr_shrinkage'];lags=cfg['analysis']['lags']
    # symmetric source transforms exactly as 2x2
    pittr=rawpit(qtr,levels,ytr);pitc=rawpit(qc,levels,yc);pitp=rawpit(qp,levels,yp)
    rawmaps=[Emp(rtr[:,c]) for c in range(4)];pitmaps=[Emp(pittr[:,c]) for c in range(4)]
    def rawz(R):return np.stack([rawmaps[c].z(R[:,c],clip) for c in range(4)],axis=1)
    def resz(P):return np.stack([pitmaps[c].z(P[:,c],clip) for c in range(4)],axis=1)
    gtr,gc,gp=rawz(rtr),rawz(rc),rawz(rpilo);ztr,zc,zp=resz(pittr),resz(pitc),resz(pitp)
    # separate common marginal calibration diagnostic
    cal=PitCal(pittr); u_pilot=cal.u(pitp)
    out=[];align_sign=[];disp_sign=[]
    for c in range(4):
        Rraw=corr(gtr[:,c],shrink);Rres=corr(ztr[:,c],shrink)
        Rtarget_cal=corr(zc[:,c],shrink);Rtarget_p=corr(zp[:,c],shrink)
        d_raw=fro(Rraw,Rtarget_p);d_res=fro(Rres,Rtarget_p)
        lm_raw=lagmae(Rraw,Rtarget_p,lags);lm_res=lagmae(Rres,Rtarget_p,lags)
        ar,ae,at=agg(Rraw),agg(Rres),agg(Rtarget_p)
        fr_raw,fr_res=abs(ar-at),abs(ae-at)
        gainci=block_ci(cases,c,'RAW_FIXED','RESID_FIXED',
             cfg['analysis']['bootstrap_reps'],cfg['analysis']['block_length_days'],
             cfg['seed'],cfg['analysis']['ci_level'])
        crpswin=gainci['gain_pct']>0
        structural=(d_res<d_raw and lm_res<lm_raw)
        dispersion=(fr_res<fr_raw)
        align_sign.append(crpswin==structural);disp_sign.append(crpswin==dispersion)
        # train stability: first/second half; train->CAL target-source within same representation
        half=len(gtr)//2
        raw_half=fro(corr(gtr[:half,c],shrink),corr(gtr[half:,c],shrink))
        res_half=fro(corr(ztr[:half,c],shrink),corr(ztr[half:,c],shrink))
        raw_train_cal=fro(Rraw,corr(gc[:,c],shrink));res_train_cal=fro(Rres,Rtarget_cal)
        rawcase=case_summary(cases,c,'RAW_FIXED');rescase=case_summary(cases,c,'RESID_FIXED')
        rec={
          'channel':c,'crps_gain_raw_to_resid_fixed_pct':gainci['gain_pct'],
          'crps_gain_ci90':[gainci['ci_low'],gainci['ci_high']],
          'resid_crps_win':crpswin,
          'pilot_source_target':{
             'raw_fro':d_raw,'resid_fro':d_res,'resid_fro_closer':d_res<d_raw,
             'raw_lag_mae':lm_raw,'resid_lag_mae':lm_res,'resid_lag_closer':lm_res<lm_raw},
          'aggregate_factor':{
             'raw_source':ar,'resid_source':ae,'pilot_resid_target':at,
             'raw_abs_error':fr_raw,'resid_abs_error':fr_res,'resid_closer':dispersion},
          'source_strength':{
             'raw_mean_abs_offdiag':float(np.mean(np.abs(Rraw-np.eye(H)))),
             'resid_mean_abs_offdiag':float(np.mean(np.abs(Rres-np.eye(H)))),
             'raw_effective_rank':effrank(Rraw),'resid_effective_rank':effrank(Rres)},
          'stability':{
             'raw_train_half_fro':raw_half,'resid_train_half_fro':res_half,
             'raw_train_to_cal_fro':raw_train_cal,'resid_train_to_cal_fro':res_train_cal},
          'marginal':{
             'pilot_calibrated_pit_mean':float(np.mean(u_pilot[:,c])),
             'pilot_calibrated_pit_var':float(np.var(u_pilot[:,c])),
             'pilot_calibrated_pit_ks':ks_uniform(u_pilot[:,c]),
             'pilot_pinball':pinball(qp[:,c],levels,yp[:,c])},
          'pilot_distribution':{
             'raw_fixed':rawcase,'resid_fixed':rescase,
             'width90_ratio_resid_over_raw':rescase['w90']/rawcase['w90'] if rawcase['w90'] else None},
          'lag_profiles':{'raw_train':lagprof(Rraw,lags),'resid_train':lagprof(Rres,lags),'pilot_resid_target':lagprof(Rtarget_p,lags)}
        }
        out.append(rec)
    # diagnosis token: exploratory, never a paper-level claim
    structural_match=sum(align_sign);disp_match=sum(disp_sign)
    winners=[x for x in out if x['resid_crps_win']];losers=[x for x in out if not x['resid_crps_win']]
    robust_winner=any(x['crps_gain_ci90'][0]>0 for x in winners)
    if structural_match>=3 and robust_winner:
        token='STRUCTURAL_ALIGNMENT_PLAUSIBLE'
    elif disp_match>=3:
        token='DISPERSION_ALIGNMENT_PLAUSIBLE'
    else:
        # marginal limitation only if every loser has worse KS than every winner
        if winners and losers and min(x['marginal']['pilot_calibrated_pit_ks'] for x in losers) > max(x['marginal']['pilot_calibrated_pit_ks'] for x in winners):
            token='MARGINAL_LIMITATION_PLAUSIBLE'
        else: token='HETEROGENEITY_UNEXPLAINED'
    result={'status':'COMPLETED_DIAGNOSIS','diagnosis_token':token,
      'channels':out,'diagnostic_agreement':{'structural_sign_match_count':structural_match,'dispersion_sign_match_count':disp_match},
      'limitations':['Only 4 channels. Exploratory mechanism diagnosis, not causal proof.',
                     'PILOT residual target correlation is used only post hoc for diagnosis, never for fitting.',
                     '24x24 correlation from 25 PILOT origins is noisy; lag profile and block bootstrap are reported alongside.'],
      'contracts':{'chronos_inference_count':0,'training_count':0,'protected_values_at_or_after_7200_access':False,
                   'forecast_cache_sha256':sh(cache),'raw_sha256':sh(rp)}}
    rid=time.strftime('run_%Y%m%dT%H%M%SZ',time.gmtime());od=repo/'results'/cfg['name']/rid;od.mkdir(parents=True)
    wj(od/'DIAGNOSIS.json',result)
    # concise CSV
    with open(od/'CHANNEL_SUMMARY.csv','w',newline='',encoding='utf-8') as f:
        cols=['channel','crps_gain_pct','ci_low','ci_high','raw_fro','resid_fro','raw_lag_mae','resid_lag_mae','raw_agg_err','resid_agg_err','pit_ks','pinball']
        w=__import__('csv').DictWriter(f,fieldnames=cols);w.writeheader()
        for x in out:w.writerow({'channel':x['channel'],'crps_gain_pct':x['crps_gain_raw_to_resid_fixed_pct'],
          'ci_low':x['crps_gain_ci90'][0],'ci_high':x['crps_gain_ci90'][1],
          'raw_fro':x['pilot_source_target']['raw_fro'],'resid_fro':x['pilot_source_target']['resid_fro'],
          'raw_lag_mae':x['pilot_source_target']['raw_lag_mae'],'resid_lag_mae':x['pilot_source_target']['resid_lag_mae'],
          'raw_agg_err':x['aggregate_factor']['raw_abs_error'],'resid_agg_err':x['aggregate_factor']['resid_abs_error'],
          'pit_ks':x['marginal']['pilot_calibrated_pit_ks'],'pinball':x['marginal']['pilot_pinball']})
    wj(od/'RUN_MANIFEST.json',{'head':git(['rev-parse','HEAD'],repo)[1].strip(),'config':cfg,'source_cache':str(cache)})
    if a.publish:
        exp=repo/'experiments'/cfg['name']
        if not exp.exists():shutil.copytree(pkg,exp,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        rc,so,se=git(['add','--',str(exp.relative_to(repo)),str(od.relative_to(repo))],repo);req(rc==0,'git add failed')
        rc,so,se=git(['diff','--cached','--check'],repo);req(rc==0,'git diff check failed')
        rc,so,se=git(['commit','-m',f'Diagnose residual-source channel heterogeneity ({rid})'],repo);req(rc==0,'commit failed '+se)
        sha=git(['rev-parse','HEAD'],repo)[1].strip()
        rc,so,se=git(['push','origin','HEAD:main'],repo);req(rc==0,'push failed '+se)
        result['publish_commit']=sha
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
