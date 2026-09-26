from __future__ import annotations
import argparse, gzip, hashlib, json, math, shutil, subprocess, time, traceback
from pathlib import Path
import numpy as np
from scipy.stats import norm, rankdata

class ContractError(RuntimeError): pass
def require(x,m):
    if not x: raise ContractError(m)
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def readj(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def writej(p,o):
    Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False,
        default=lambda x: x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def git(args,cwd):
    p=subprocess.run(['git',*args],cwd=cwd,text=True,capture_output=True)
    return p.returncode,p.stdout,p.stderr
def gain(a,b): return 100*(a-b)/a if a else 0.0

def locate_cache(repo,cfg):
    expected=cfg['forecast_cache_sha256']
    direct=repo/'.cache'/'residual_dependence_gonogo_v1_20260925'/'run_20260925T150030Z_f0_forecasts.npz'
    if direct.is_file() and sha256(direct)==expected: return direct
    hits=[]
    for p in (repo/'.cache').rglob('*.npz'):
        try:
            if sha256(p)==expected: hits.append(p)
        except: pass
    require(len(hits)==1,f'forecast cache match count={len(hits)}')
    return hits[0]

def origins(a,b,stride,H):
    x=np.arange(a,b,stride,dtype=int)
    return x[x+H<=b]

def raw_matrix(repo,cfg):
    p=repo/cfg['raw_data_path']; require(p.is_file(),'raw electricity missing')
    require(sha256(p)==cfg['raw_data_sha256'],'raw electricity hash mismatch')
    with gzip.open(p,'rt',encoding='utf-8') as f:
        x=np.loadtxt(f,delimiter=',',usecols=[0,1,2,3],max_rows=7200,dtype=np.float64)
    require(x.shape==(7200,4),'unexpected raw prefix')
    return x

class EmpiricalMap:
    def __init__(self, X):
        # X N,H; TRAIN-only map per horizon
        self.cols=[np.sort(X[:,h]) for h in range(X.shape[1])]
    def to_z(self,X,clip=.005):
        Z=np.empty_like(X,dtype=float)
        for h,a in enumerate(self.cols):
            r=np.searchsorted(a,X[:,h],side='right')
            u=(r+.5)/(len(a)+1.0)
            Z[:,h]=norm.ppf(np.clip(u,clip,1-clip))
        return Z

def raw_pit(q,levels,y):
    lev=np.asarray(levels,float); out=np.empty_like(y,dtype=float)
    for idx in np.ndindex(y.shape):
        o,c,h=idx; qq=q[o,c,h]; yy=y[idx]
        uq,ii=np.unique(qq,return_index=True); ul=lev[ii]
        out[idx]=.5 if len(uq)<2 else np.interp(yy,uq,ul,left=ul[0]*.5,right=1-(1-ul[-1])*.5)
    return out

class PitCal:
    def __init__(self,pit):
        self.a=[np.sort(pit[:,:,h].ravel()) for h in range(pit.shape[2])]
    def fwd(self,u,h):
        a=self.a[h]; return (np.searchsorted(a,u,side='right')+.5)/(len(a)+1.0)
    def inv(self,v,h):
        a=self.a[h]; p=(np.arange(len(a))+.5)/len(a)
        return np.interp(v,p,a,left=a[0],right=a[-1])

def residual_z(q,y,levels,cal,clip):
    p=raw_pit(q,levels,y); z=np.empty_like(p)
    for h in range(p.shape[2]):
        for o in range(p.shape[0]):
            for c in range(p.shape[1]):
                z[o,c,h]=norm.ppf(np.clip(cal.fwd(float(p[o,c,h]),h),clip,1-clip))
    return z

def inv_margin(q,levels,v,cal):
    # q H,Q ; v M,H calibrated uniforms
    M,H=v.shape; lev=np.asarray(levels,float); out=np.empty((M,H))
    for h in range(H):
        u=np.asarray([cal.inv(float(x),h) for x in v[:,h]])
        qq=q[h]
        lo=qq[0]-(qq[1]-qq[0]); hi=qq[-1]+(qq[-1]-qq[-2])
        out[:,h]=np.interp(u,np.r_[0.,lev,1.],np.r_[lo,qq,hi])
    return np.maximum(out,0)

def corr(Z,shrink):
    R=np.corrcoef(Z,rowvar=False); R=np.nan_to_num(R,nan=0.0)
    np.fill_diagonal(R,1); R=(1-shrink)*R+shrink*np.eye(R.shape[0])
    w,V=np.linalg.eigh((R+R.T)/2); w=np.maximum(w,1e-6); R=(V*w)@V.T
    d=np.sqrt(np.diag(R)); R=R/d[:,None]/d[None,:]; np.fill_diagonal(R,1)
    return R

def wcorr(Z,w,prior,mix,shrink):
    w=np.asarray(w,float); w=w/w.sum(); mu=(Z*w[:,None]).sum(0); X=Z-mu
    C=(X*w[:,None]).T@X; d=np.sqrt(np.maximum(np.diag(C),1e-9)); R=C/d[:,None]/d[None,:]
    return corr_from_R((1-mix)*R+mix*prior,shrink)

def corr_from_R(R,shrink):
    R=np.nan_to_num(np.asarray(R,float),nan=0.0); np.fill_diagonal(R,1)
    R=(1-shrink)*R+shrink*np.eye(R.shape[0])
    w,V=np.linalg.eigh((R+R.T)/2); w=np.maximum(w,1e-6); R=(V*w)@V.T
    d=np.sqrt(np.diag(R)); R=R/d[:,None]/d[None,:]; np.fill_diagonal(R,1)
    return R

def paths(q,levels,cal,R,baseZ):
    L=np.linalg.cholesky(corr_from_R(R,0))
    return inv_margin(q,levels,norm.cdf(baseZ@L.T),cal)

def crps(s,y):
    s=np.sort(np.asarray(s)); M=len(s)
    return float(np.mean(np.abs(s-y))-np.sum((2*np.arange(1,M+1)-M-1)*s)/(M*M))
def met(P,y):
    t=P.sum(1); yt=float(y.sum()); q05,q10,q90,q95=np.quantile(t,[.05,.10,.90,.95])
    return {'crps':crps(t,yt),'c80':float(q10<=yt<=q90),'w80':float(q90-q10),
            'c90':float(q05<=yt<=q95),'w90':float(q95-q05)}

def moving_block_ci(dayrows,a,b,cfg,seed):
    A=np.array([r[a]['crps'] for r in dayrows]); B=np.array([r[b]['crps'] for r in dayrows])
    n=len(A); L=cfg['bootstrap']['block_length_days']; reps=cfg['bootstrap']['reps']; rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(reps):
        ix=[]
        while len(ix)<n:
            s=int(rng.integers(0,n))
            ix.extend([(s+j)%n for j in range(L)])
        ix=np.asarray(ix[:n]); vals.append(gain(A[ix].mean(),B[ix].mean()))
    alpha=(1-cfg['bootstrap']['ci_level'])/2
    return {'gain_pct':gain(A.mean(),B.mean()),'ci_low':float(np.quantile(vals,alpha)),
            'ci_high':float(np.quantile(vals,1-alpha))}

def average_origin(rows):
    out=[]
    for o in sorted(set(r['origin'] for r in rows)):
        rr=[r for r in rows if r['origin']==o]; z={'origin':o}
        for m in ['RAW_FIXED','RAW_RECENT','RESID_FIXED','RESID_RECENT','IND']:
            z[m]={k:float(np.mean([x[m][k] for x in rr])) for k in rr[0][m]}
        out.append(z)
    return out

def channel_gain(rows,a,b):
    z=[]
    for c in range(4):
        rr=[r for r in rows if r['channel']==c]
        A=np.mean([r[a]['crps'] for r in rr]); B=np.mean([r[b]['crps'] for r in rr])
        z.append({'channel':c,'gain_pct':gain(A,B)})
    return z

def evaluate(split,orig,q,y,rawZ,resZ,train_raw_fixed,train_res_fixed,
             past_orig,past_rawZ,past_resZ,levels,cal,cfg):
    sh=cfg['copula']['corr_shrinkage']; half=cfg['copula']['recent_half_life_days']
    mix=cfg['copula']['recent_prior_mix']; M=cfg['copula']['n_scenarios']; rows=[]
    Rraw=[corr(train_raw_fixed[:,c],sh) for c in range(4)]
    Rres=[corr(train_res_fixed[:,c],sh) for c in range(4)]
    ho=list(map(int,past_orig)); hraw=[past_rawZ[i] for i in range(len(past_orig))]
    hres=[past_resZ[i] for i in range(len(past_orig))]
    for i,o in enumerate(orig):
        for c in range(4):
            seed=cfg['seed']+int(o)*101+c*10007
            rng=np.random.default_rng(seed)
            baseZ=rng.standard_normal((M,24))   # SAME base draws for all Gaussian cells
            uni=rng.random((M,24))
            # fixed
            pf=paths(q[i,c],levels,cal,Rraw[c],baseZ)
            rf=paths(q[i,c],levels,cal,Rres[c],baseZ)
            # recent using identical strictly-past history policy
            Zr=[]; Zs=[]; W=[]
            for j,old in enumerate(ho):
                age=(int(o)-old)/24
                if age>0:
                    Zr.append(hraw[j][c]); Zs.append(hres[j][c]); W.append(.5**(age/half))
            Rrr=wcorr(np.asarray(Zr),np.asarray(W),Rraw[c],mix,sh) if len(Zr)>=30 else Rraw[c]
            Rsr=wcorr(np.asarray(Zs),np.asarray(W),Rres[c],mix,sh) if len(Zs)>=30 else Rres[c]
            pr=paths(q[i,c],levels,cal,Rrr,baseZ)
            rr=paths(q[i,c],levels,cal,Rsr,baseZ)
            ind=inv_margin(q[i,c],levels,uni,cal)
            truth=y[i,c]
            rows.append({'origin':int(o),'channel':c,'RAW_FIXED':met(pf,truth),'RAW_RECENT':met(pr,truth),
                         'RESID_FIXED':met(rf,truth),'RESID_RECENT':met(rr,truth),'IND':met(ind,truth)})
        # reveal current source vectors only AFTER scoring current origin
        ho.append(int(o)); hraw.append(rawZ[i]); hres.append(resZ[i])
    return rows

def summarize(rows):
    out={}
    for m in ['RAW_FIXED','RAW_RECENT','RESID_FIXED','RESID_RECENT','IND']:
        out[m]={k:float(np.mean([r[m][k] for r in rows])) for k in rows[0][m]}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); ap.add_argument('--package-root',required=True); ap.add_argument('--publish',action='store_true')
    a=ap.parse_args(); repo=Path(a.repo).resolve(); pkg=Path(a.package_root).resolve(); cfg=readj(pkg/'RUN_CONFIG.json')
    # source contracts
    src=repo/cfg['source_run']/'FINAL_DECISION.json'; require(src.is_file(),'source FINAL missing')
    blob=src.read_bytes(); git_blob_sha=hashlib.sha1(f"blob {len(blob)}\0".encode()+blob).hexdigest(); require(git_blob_sha==cfg['source_final_sha'],'source FINAL git-blob SHA mismatch')
    cache=locate_cache(repo,cfg); raw=raw_matrix(repo,cfg)
    D=np.load(cache); keys=set(D.files); expected={'q_train','y_train','q_cal','y_cal','q_pilot','y_pilot'}
    require(expected.issubset(keys),f'cache keys missing {expected-keys}')
    qtr,ytr,qc,yc,qp,yp=[D[k].astype(float) for k in ['q_train','y_train','q_cal','y_cal','q_pilot','y_pilot']]
    require(qtr.shape[:3]==ytr.shape and qc.shape[:3]==yc.shape and qp.shape[:3]==yp.shape,'cache shape mismatch')
    levels=readj(repo/cfg['source_run']/'RUN_MANIFEST.json')['config']['model']['quantiles']
    # exact origin contracts
    s=cfg['splits']; otr=origins(s['train_start'],s['train_stop'],s['stride'],s['horizon'])
    oc=origins(s['cal_start'],s['cal_stop'],s['stride'],s['horizon'])
    op=origins(s['pilot_start'],s['pilot_stop'],s['stride'],s['horizon'])
    require((len(otr),len(oc),len(op))==(qtr.shape[0],qc.shape[0],qp.shape[0]),'origin/cache count mismatch')
    # raw source paths corresponding to exact horizons
    def Yraw(oo): return np.stack([[raw[o:o+24,c] for c in range(4)] for o in oo])
    rtr,rc,rp=Yraw(otr),Yraw(oc),Yraw(op)
    require(np.allclose(rtr,ytr) and np.allclose(rc,yc) and np.allclose(rp,yp),'raw/cache truth mismatch')
    # Common forecast marginal calibration used by ALL scenario generators.
    pit=raw_pit(qtr,levels,ytr); cal=PitCal(pit)

    # Dependence-source transforms are deliberately symmetric:
    # RAW and residual PIT each get a TRAIN-only channel×horizon empirical-CDF -> normal-score map.
    raw_maps=[EmpiricalMap(rtr[:,c]) for c in range(4)]
    pit_maps=[EmpiricalMap(pit[:,c]) for c in range(4)]
    def rawz(R):
        return np.stack([raw_maps[c].to_z(R[:,c],cfg['copula']['pit_clip']) for c in range(4)],axis=1)
    def resz(q,y):
        P=raw_pit(q,levels,y)
        return np.stack([pit_maps[c].to_z(P[:,c],cfg['copula']['pit_clip']) for c in range(4)],axis=1)
    gtr,gc,gp=rawz(rtr),rawz(rc),rawz(rp)
    ztr,zc,zp=resz(qtr,ytr),resz(qc,yc),resz(qp,yp)
    # CAL: strictly TRAIN history. PILOT: TRAIN+CAL history
    calrows=evaluate('cal',oc,qc,yc,gc,zc,gtr,ztr,otr,gtr,ztr,levels,cal,cfg)
    pilotrows=evaluate('pilot',op,qp,yp,gp,zp,gtr,ztr,np.r_[otr,oc],np.concatenate([gtr,gc]),np.concatenate([ztr,zc]),levels,cal,cfg)
    cs,ps=summarize(calrows),summarize(pilotrows); cd,pd=average_origin(calrows),average_origin(pilotrows)
    contrasts={}
    for A,B in [('RAW_FIXED','RESID_FIXED'),('RAW_RECENT','RESID_RECENT'),('RESID_FIXED','RESID_RECENT'),
                ('RAW_FIXED','RAW_RECENT'),('IND','RAW_FIXED'),('IND','RESID_FIXED')]:
        contrasts[f'{A}_TO_{B}']={'cal_gain_pct':gain(cs[A]['crps'],cs[B]['crps']),
            'pilot':moving_block_ci(pd,A,B,cfg,cfg['seed']+17),
            'pilot_channels':channel_gain(pilotrows,A,B)}
    d=cfg['decision']; fixed=contrasts['RAW_FIXED_TO_RESID_FIXED']; recent=contrasts['RAW_RECENT_TO_RESID_RECENT']
    source=(fixed['cal_gain_pct']>=d['source_gain_min_pct'] and fixed['pilot']['gain_pct']>=d['source_gain_min_pct']
            and fixed['pilot']['ci_low']>0 and sum(x['gain_pct']>0 for x in fixed['pilot_channels'])>=d['min_channel_wins']
            and recent['pilot']['gain_pct']>=0)
    ad=contrasts['RESID_FIXED_TO_RESID_RECENT']
    adapt=(ad['cal_gain_pct']>=d['adapt_gain_min_pct'] and ad['pilot']['gain_pct']>=d['adapt_gain_min_pct']
           and ad['pilot']['ci_low']>0 and sum(x['gain_pct']>0 for x in ad['pilot_channels'])>=d['min_channel_wins'])
    dep=contrasts['IND_TO_RAW_FIXED']
    if source and adapt: token='GO_SMALL_ADAPTIVE_RESIDUAL_MODULE'
    elif source: token='STATIC_RESIDUAL_POSTPROCESSING_ONLY'
    elif dep['pilot']['gain_pct']>=2 and dep['pilot']['ci_low']>0: token='RAW_DEPENDENCE_BASELINE_ONLY'
    else:
        # if fixed source gain is positive but CI crosses zero, preserve uncertainty
        token='INCONCLUSIVE_SOURCE_EFFECT' if fixed['pilot']['gain_pct']>0 else 'STOP_DEPENDENCE_TOPIC'
    runid=time.strftime('run_%Y%m%dT%H%M%SZ',time.gmtime())
    out=repo/'results'/cfg['name']/runid; out.mkdir(parents=True,exist_ok=True)
    result={'status':'COMPLETED_2X2','token':token,
      'gates':{'RESIDUAL_SOURCE_SUPPORTED':bool(source),'RESIDUAL_ADAPTATION_SUPPORTED':bool(adapt)},
      'cal_summary':cs,'pilot_summary':ps,'contrasts':contrasts,
      'contracts':{'chronos_inference_count':0,'training_count':0,'forecast_cache':str(cache),
                   'forecast_cache_sha256':sha256(cache),'raw_sha256':sha256(repo/cfg['raw_data_path']),
                   'protected_values_at_or_after_7200_access':False,'common_gaussian_draws':True,
                   'same_gaussian_copula_family_for_raw_and_residual':True,
                   'fixed_train_only_for_cal_and_pilot':True}}
    writej(out/'FINAL_DECISION.json',result)
    for nm,rows in [('CAL_CASES.jsonl',calrows),('PILOT_CASES.jsonl',pilotrows)]:
        with open(out/nm,'w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    writej(out/'RUN_MANIFEST.json',{'head':git(['rev-parse','HEAD'],repo)[1].strip(),'config':cfg,'source_cache':str(cache)})
    # publish fixed package + results
    if a.publish:
        exp=repo/'experiments'/cfg['name']
        if not exp.exists(): shutil.copytree(pkg,exp,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        rc,so,se=git(['add','--',str(exp.relative_to(repo)),str(out.relative_to(repo))],repo); require(rc==0,'git add failed')
        rc,so,se=git(['diff','--cached','--check'],repo); require(rc==0,'git diff check failed')
        rc,so,se=git(['commit','-m',f'Run RAW-residual 2x2 Go-NoGo ({runid})'],repo); require(rc==0,'git commit failed '+se)
        sha=git(['rev-parse','HEAD'],repo)[1].strip()
        rc,so,se=git(['push','origin','HEAD:main'],repo); require(rc==0,'git push failed '+se)
        result['publish_commit']=sha
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as e:
        print(json.dumps({'status':'TECHNICAL_FAILURE','error':repr(e)},ensure_ascii=False))
        raise
