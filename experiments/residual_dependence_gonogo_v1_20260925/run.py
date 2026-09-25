from __future__ import annotations
import argparse, csv, gzip, hashlib, importlib.metadata as md, json, math, os, random, shutil, subprocess, time, traceback, urllib.request
from pathlib import Path
import numpy as np

class ContractError(RuntimeError): pass
def require(x,msg):
    if not x: raise ContractError(msg)
def write_json(p,o):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,indent=2,ensure_ascii=False,default=lambda x:float(x) if hasattr(x,'item') else str(x)),encoding='utf-8')
def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def git(args,cwd,timeout=300):
    p=subprocess.run(['git',*args],cwd=cwd,text=True,capture_output=True,timeout=timeout)
    return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}

def ensure_data(repo,cfg):
    ds=cfg['dataset']; cand=[Path(repo)/ds['local_path']]+[Path(repo)/p for p in ds['fallback_paths']]
    for p in cand:
        if p.is_file() and sha256(p)==ds['sha256']: return p
    cache=Path(repo)/'.cache'/cfg['name']/'data'; cache.mkdir(parents=True,exist_ok=True); dst=cache/'electricity.txt.gz'
    if not dst.exists() or sha256(dst)!=ds['sha256']:
        tmp=cache/'electricity.part'
        if tmp.exists(): tmp.unlink()
        urllib.request.urlretrieve(ds['download_url'],tmp)
        require(sha256(tmp)==ds['sha256'],'Downloaded electricity hash mismatch')
        tmp.replace(dst)
    return dst

def load_data(path,cfg):
    ds=cfg['dataset']; limit=ds['protected_evaluation_start']
    with gzip.open(path,'rt',encoding='utf-8') as f:
        x=np.loadtxt(f,delimiter=',',usecols=ds['usecols'],max_rows=limit,dtype=np.float32)
    require(x.shape==(limit,len(ds['usecols'])),'Unexpected protected-prefix shape')
    require(np.isfinite(x).all(),'Nonfinite values in used prefix')
    L,H=ds['context'],ds['horizon']; splits={}
    for name,a,b in [('train',ds['train_start'],ds['train_stop']),('cal',ds['cal_start'],ds['cal_stop']),('pilot',ds['pilot_start'],ds['pilot_stop'])]:
        o=np.arange(a,b,ds['origin_stride'],dtype=int); o=o[(o>=L)&(o+H<=b)]
        require(len(o)>=20,f'{name} too few origins'); splits[name]=o
    require(int(splits['pilot'].max()+H)<=limit,'Pilot crosses protected boundary')
    return x,splits

def contexts_truth(x,origins,L,H):
    C=x.shape[1]
    ctx=np.stack([[x[o-L:o,c] for c in range(C)] for o in origins]).astype(np.float32)
    y=np.stack([[x[o:o+H,c] for c in range(C)] for o in origins]).astype(np.float32)
    return ctx,y

def load_pipe(cfg):
    import torch
    require(md.version('chronos-forecasting')==cfg['model']['expected_chronos_version'],'Unexpected Chronos version')
    require(torch.cuda.is_available(),'CUDA unavailable')
    from chronos import Chronos2Pipeline
    try: return Chronos2Pipeline.from_pretrained(cfg['model']['id'],device_map='cuda',dtype=torch.float32,local_files_only=True)
    except Exception: return Chronos2Pipeline.from_pretrained(cfg['model']['id'],device_map='cuda',dtype=torch.float32,local_files_only=False)

def predict_split(pipe,ctx,cfg):
    import torch
    O,C,L=ctx.shape; H=cfg['dataset']['horizon']; lev=cfg['model']['quantiles']; groups=[]; out=[]
    for o in range(O):
        for c in range(C): groups.append(torch.tensor(ctx[o,c][None],dtype=torch.float32))
    bs=cfg['model']['batch_size']
    for s in range(0,len(groups),bs):
        qp,_=pipe.predict_quantiles(groups[s:s+bs],prediction_length=H,quantile_levels=lev,batch_size=bs,context_length=L,cross_learning=False)
        out.extend([q[0].float().cpu().numpy() for q in qp])
    a=np.asarray(out,np.float32).reshape(O,C,H,len(lev)); crossing=float(np.mean(np.diff(a,axis=-1)<0)); a=np.maximum.accumulate(a,axis=-1)
    return a,crossing

def raw_pit(q,levels,y):
    lev=np.asarray(levels,float); fq=q.reshape(-1,q.shape[-1]); fy=y.reshape(-1); out=np.empty(len(fy))
    for i,(qq,yy) in enumerate(zip(fq,fy)):
        uq,idx=np.unique(qq,return_index=True); ul=lev[idx]
        out[i]=0.5 if len(uq)<2 else np.interp(yy,uq,ul,left=ul[0]*0.5,right=1-(1-ul[-1])*0.5)
    return out.reshape(y.shape)

class PitCal:
    def __init__(self,pits): self.by_h=[np.sort(pits[:,:,h].reshape(-1)) for h in range(pits.shape[2])]
    def forward(self,u,h):
        a=self.by_h[h]; return (np.searchsorted(a,u,side='right')+0.5)/(len(a)+1.0)
    def inverse(self,v,h):
        a=self.by_h[h]; p=(np.arange(len(a))+0.5)/len(a); return np.interp(v,p,a,left=a[0],right=a[-1])

def calibrated_z(q,levels,y,cal,clip):
    from scipy.stats import norm
    rp=raw_pit(q,levels,y); z=np.empty_like(rp,float)
    for h in range(rp.shape[2]):
        v=np.asarray([[cal.forward(float(rp[o,c,h]),h) for c in range(rp.shape[1])] for o in range(rp.shape[0])])
        z[:,:,h]=norm.ppf(np.clip(v,clip,1-clip))
    return z,rp

def inv_margin(q,levels,v,cal):
    M,H=v.shape; out=np.empty((M,H)); lev=np.asarray(levels,float)
    for h in range(H):
        u=np.asarray([cal.inverse(float(x),h) for x in v[:,h]]); qq=q[h]
        lo=qq[0]-(qq[1]-qq[0]); hi=qq[-1]+(qq[-1]-qq[-2]); out[:,h]=np.interp(u,np.r_[0.,lev,1.],np.r_[lo,qq,hi])
    return np.maximum(out,0.0)

def nearest_corr(R,shrink=.1):
    R=np.nan_to_num(np.asarray(R,float),nan=0.0,posinf=0.0,neginf=0.0); np.fill_diagonal(R,1.0); R=(1-shrink)*R+shrink*np.eye(R.shape[0])
    w,V=np.linalg.eigh((R+R.T)/2); w=np.maximum(w,1e-5); R=(V*w)@V.T; d=np.sqrt(np.diag(R)); R=R/d[:,None]/d[None,:]; np.fill_diagonal(R,1); return R

def corr_rows(Z,shrink): return nearest_corr(np.corrcoef(Z,rowvar=False),shrink)
def weighted_corr(Z,w,prior,mix,shrink):
    w=np.asarray(w,float); w=w/w.sum(); mu=(Z*w[:,None]).sum(0); X=Z-mu; C=(X*w[:,None]).T@X; d=np.sqrt(np.maximum(np.diag(C),1e-8)); R=C/d[:,None]/d[None,:]
    return nearest_corr((1-mix)*R+mix*prior,shrink)

def independent_paths(q,levels,cal,M,rng): return inv_margin(q,levels,rng.random((M,q.shape[0])),cal)
def gaussian_paths(q,levels,cal,R,M,rng):
    from scipy.stats import norm
    L=np.linalg.cholesky(nearest_corr(R,0)); v=norm.cdf(rng.standard_normal((M,q.shape[0]))@L.T); return inv_margin(q,levels,v,cal)
def schaake_paths(q,levels,cal,templates,M,rng):
    from scipy.stats import rankdata
    idx=rng.choice(len(templates),size=M,replace=len(templates)<M); T=templates[idx]; base=inv_margin(q,levels,rng.random((M,q.shape[0])),cal); out=np.empty_like(base)
    for h in range(q.shape[0]): out[:,h]=np.sort(base[:,h])[rankdata(T[:,h],method='ordinal').astype(int)-1]
    return out

def crps_ensemble(samples,y):
    s=np.sort(np.asarray(samples,float)); M=len(s); term1=np.mean(np.abs(s-y)); coef=2*np.arange(1,M+1)-M-1; term2=np.sum(coef*s)/(M*M); return float(term1-term2)
def metrics(paths,y):
    totals=paths.sum(1); yt=float(y.sum()); q05,q10,q90,q95=np.quantile(totals,[.05,.10,.90,.95])
    return {'crps_total':crps_ensemble(totals,yt),'cover80':float(q10<=yt<=q90),'width80':float(q90-q10),'cover90':float(q05<=yt<=q95),'width90':float(q95-q05),'median_abs_total':float(abs(np.median(totals)-yt))}
METHODS=['IND','RAW_SCHAAKE','RESID_FIXED','RESID_RECENT']

def evaluate(name,origins,q,y,hist_orig,hist_y,hist_z,cal,cfg):
    M=cfg['paths']['n_scenarios']; lev=cfg['model']['quantiles']; sh=cfg['paths']['corr_shrinkage']; mix=cfg['paths']['recent_fixed_mix']; half=cfg['paths']['recent_half_life_days']
    fixed=[corr_rows(hist_z[:,c,:],sh) for c in range(hist_z.shape[1])]; raw=[hist_y[:,c,:] for c in range(hist_y.shape[1])]; rows=[]
    zcur,_=calibrated_z(q,lev,y,cal,cfg['paths']['pit_clip']); ho=list(map(int,hist_orig)); hz=[hist_z[i] for i in range(len(hist_orig))]
    for oi,o in enumerate(origins):
        for c in range(y.shape[1]):
            seed=cfg['seed']+(11 if name=='cal' else 29)+int(o)*101+c*10007; qi=q[oi,c]; truth=y[oi,c]
            # independent deterministic streams to avoid method-order dependence
            ind=independent_paths(qi,lev,cal,M,np.random.default_rng(seed+1))
            rs=schaake_paths(qi,lev,cal,raw[c],M,np.random.default_rng(seed+2))
            fx=gaussian_paths(qi,lev,cal,fixed[c],M,np.random.default_rng(seed+3))
            Z=[];W=[]
            for j,old in enumerate(ho):
                age=(int(o)-old)/24.0
                if age>0: Z.append(hz[j][c]); W.append(0.5**(age/half))
            Rr=weighted_corr(np.asarray(Z),np.asarray(W),fixed[c],mix,sh) if len(Z)>=30 else fixed[c]
            rc=gaussian_paths(qi,lev,cal,Rr,M,np.random.default_rng(seed+4))
            rows.append({'origin':int(o),'channel':int(c),'IND':metrics(ind,truth),'RAW_SCHAAKE':metrics(rs,truth),'RESID_FIXED':metrics(fx,truth),'RESID_RECENT':metrics(rc,truth)})
        ho.append(int(o)); hz.append(zcur[oi]); raw=[np.concatenate([raw[c],y[oi:oi+1,c,:]],axis=0) for c in range(y.shape[1])]
    return rows

def summarize(rows):
    out={}
    for m in METHODS: out[m]={k:float(np.mean([r[m][k] for r in rows])) for k in rows[0][m]}
    return out

def origin_average(rows):
    out=[]
    for o in sorted(set(r['origin'] for r in rows)):
        rr=[r for r in rows if r['origin']==o]; z={'origin':o}
        for m in METHODS:z[m]={k:float(np.mean([r[m][k] for r in rr])) for k in rr[0][m]}
        out.append(z)
    return out

def channel_gains(rows,base,new):
    out=[]
    for c in sorted(set(r['channel'] for r in rows)):
        rr=[r for r in rows if r['channel']==c]; a=np.mean([r[base]['crps_total'] for r in rr]); b=np.mean([r[new]['crps_total'] for r in rr]); out.append({'channel':c,'gain_pct':100*(a-b)/a})
    return out

def bootstrap(rows,base,new,reps,seed,level):
    a=np.asarray([r[base]['crps_total'] for r in rows]); b=np.asarray([r[new]['crps_total'] for r in rows]); rng=np.random.default_rng(seed); g=[]; n=len(a)
    for _ in range(reps):
        ix=rng.integers(0,n,n); aa=a[ix].mean(); bb=b[ix].mean(); g.append(100*(aa-bb)/aa)
    al=(1-level)/2
    return {'mean_gain_pct':float(100*(a.mean()-b.mean())/a.mean()),'ci_low':float(np.quantile(g,al)),'ci_high':float(np.quantile(g,1-al))}

def run_experiment(x,splits,cfg,out,cache):
    pipe=load_pipe(cfg); qall={}; yall={}; crossings={}
    for sp,orig in splits.items():
        ctx,y=contexts_truth(x,orig,cfg['dataset']['context'],cfg['dataset']['horizon']); q,c=predict_split(pipe,ctx,cfg); qall[sp]=q; yall[sp]=y; crossings[sp]=c
    np.savez_compressed(cache,**{f'q_{k}':v for k,v in qall.items()},**{f'y_{k}':v for k,v in yall.items()})
    lev=cfg['model']['quantiles']; pit=raw_pit(qall['train'],lev,yall['train']); cal=PitCal(pit); ztrain,_=calibrated_z(qall['train'],lev,yall['train'],cal,cfg['paths']['pit_clip'])
    calrows=evaluate('cal',splits['cal'],qall['cal'],yall['cal'],splits['train'],yall['train'],ztrain,cal,cfg)
    zcal,_=calibrated_z(qall['cal'],lev,yall['cal'],cal,cfg['paths']['pit_clip'])
    train2=np.r_[splits['train'],splits['cal']]; y2=np.concatenate([yall['train'],yall['cal']],0); z2=np.concatenate([ztrain,zcal],0)
    pilotrows=evaluate('pilot',splits['pilot'],qall['pilot'],yall['pilot'],train2,y2,z2,cal,cfg)
    cs,ps=summarize(calrows),summarize(pilotrows); cd,pd=origin_average(calrows),origin_average(pilotrows); reps=cfg['paths']['bootstrap_reps']; ci=cfg['decision']['bootstrap_ci_level']; comp={}
    pairs=[('IND','RAW_SCHAAKE'),('IND','RESID_FIXED'),('IND','RESID_RECENT'),('RAW_SCHAAKE','RESID_FIXED'),('RAW_SCHAAKE','RESID_RECENT'),('RESID_FIXED','RESID_RECENT')]
    for base,new in pairs: comp[f'{base}_TO_{new}']={'cal':bootstrap(cd,base,new,reps,cfg['seed']+3,ci),'pilot':bootstrap(pd,base,new,reps,cfg['seed']+4,ci),'pilot_channels':channel_gains(pilotrows,base,new)}
    d=cfg['decision']; bestdep=min(('RAW_SCHAAKE','RESID_FIXED','RESID_RECENT'),key=lambda m:ps[m]['crps_total']); c1=comp[f'IND_TO_{bestdep}']
    dep=(c1['cal']['mean_gain_pct']>=d['dependence_gain_vs_ind_pct'] and c1['pilot']['mean_gain_pct']>=d['dependence_gain_vs_ind_pct'] and c1['pilot']['ci_low']>0 and sum(x['gain_pct']>0 for x in c1['pilot_channels'])>=d['min_channel_wins'])
    bestres=min(('RESID_FIXED','RESID_RECENT'),key=lambda m:ps[m]['crps_total']); c2=comp[f'RAW_SCHAAKE_TO_{bestres}']
    resid=(c2['cal']['mean_gain_pct']>=d['residual_gain_vs_raw_pct'] and c2['pilot']['mean_gain_pct']>=d['residual_gain_vs_raw_pct'] and c2['pilot']['ci_low']>0 and sum(x['gain_pct']>0 for x in c2['pilot_channels'])>=d['min_channel_wins'])
    c3=comp['RESID_FIXED_TO_RESID_RECENT']; covdeg=max(abs(ps['RESID_RECENT']['cover80']-.8)-abs(ps['RESID_FIXED']['cover80']-.8),abs(ps['RESID_RECENT']['cover90']-.9)-abs(ps['RESID_FIXED']['cover90']-.9))*100
    adapt=(c3['cal']['mean_gain_pct']>=d['recent_gain_vs_fixed_pct'] and c3['pilot']['mean_gain_pct']>=d['recent_gain_vs_fixed_pct'] and c3['pilot']['ci_low']>0 and sum(x['gain_pct']>0 for x in c3['pilot_channels'])>=d['min_channel_wins'] and covdeg<=d['max_coverage_degradation_pp'])
    token='GO_ONLINE_RESIDUAL_ADAPTER' if dep and resid and adapt else ('STATIC_RESIDUAL_SUFFICIENT' if dep and resid else ('SIMPLE_RAW_DEPENDENCE_FIRST' if dep else 'NO_GO_DEPENDENCE_AXIS'))
    # diagnostics: residual stability and raw-vs-residual dependence difference
    half=len(ztrain)//2; stab=[]; rawdiff=[]
    from scipy.stats import rankdata,norm
    for c in range(ztrain.shape[1]):
        r1=corr_rows(ztrain[:half,c],cfg['paths']['corr_shrinkage']); r2=corr_rows(ztrain[half:,c],cfg['paths']['corr_shrinkage']); stab.append(float(np.linalg.norm(r1-r2,'fro')/np.linalg.norm(r1,'fro')))
        Y=yall['train'][:,c,:]; G=np.empty_like(Y,float)
        for h in range(Y.shape[1]):
            rr=rankdata(Y[:,h],method='average'); G[:,h]=norm.ppf((rr-.5)/len(rr))
        rawR=corr_rows(G,cfg['paths']['corr_shrinkage']); resR=corr_rows(ztrain[:,c,:],cfg['paths']['corr_shrinkage']); rawdiff.append(float(np.linalg.norm(rawR-resR,'fro')/np.linalg.norm(rawR,'fro')))
    report={'status':'COMPLETED_GONOGO','token':token,'gates':{'DEPENDENCE_RELEVANT':dep,'RESIDUAL_STRUCTURE_VALUE':resid,'ADAPTATION_NEEDED':adapt},'best_dependence_method':bestdep,'best_residual_method':bestres,'cal_summary':cs,'pilot_summary':ps,'comparisons':comp,'coverage_degradation_recent_vs_fixed_pp':covdeg,'residual_corr_split_half_relative_frobenius':stab,'raw_vs_residual_corr_relative_frobenius':rawdiff,'quantile_crossing_rate_before_monotone_fix':crossings,'protected_evaluation_values_access':False}
    write_json(out/'FINAL_DECISION.json',report)
    for name,rows in [('CAL_CASES.jsonl',calrows),('PILOT_CASES.jsonl',pilotrows)]:
        with open(out/name,'w',encoding='utf-8',newline='\n') as f:
            for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    return report

def publish(repo,pkg,out):
    exp=repo/'experiments'/'residual_dependence_gonogo_v1_20260925'
    if not exp.exists(): shutil.copytree(pkg,exp,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    rel=[str(exp.relative_to(repo)),str(out.relative_to(repo))]; a=git(['add','--',*rel],repo); require(a['returncode']==0,'git add failed'); ck=git(['diff','--cached','--check'],repo); require(ck['returncode']==0,'git diff --check failed '+ck['stdout']+ck['stderr']); c=git(['commit','-m',f'Run residual-dependence Go-NoGo ({out.name})'],repo); require(c['returncode']==0,'commit failed '+c['stderr']); sha=git(['rev-parse','HEAD'],repo)['stdout'].strip(); p=git(['push','origin','HEAD:main'],repo); return {'commit':sha,'push_returncode':p['returncode'],'stdout':p['stdout'],'stderr':p['stderr']}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); ap.add_argument('--package-root',required=True); ap.add_argument('--publish',action='store_true'); ap.add_argument('--selftest',action='store_true'); a=ap.parse_args()
    if a.selftest:
        # pure numerical tests
        assert abs(crps_ensemble(np.ones(10),1.0))<1e-12
        R=nearest_corr(np.array([[1,1.2],[1.2,1]]),.1); assert np.linalg.eigvalsh(R).min()>0
        pits=np.linspace(.1,.9,80).reshape(2,4,10); pc=PitCal(pits); assert 0<=pc.inverse(.5,0)<=1
        print('SELFTEST_PASS'); return 0
    repo=Path(a.repo).resolve(); pkg=Path(a.package_root).resolve(); cfg=read_json(pkg/'RUN_CONFIG.json'); rid=time.strftime('run_%Y%m%dT%H%M%SZ',time.gmtime()); out=repo/'results'/cfg['name']/rid; out.mkdir(parents=True)
    write_json(out/'RUN_MANIFEST.json',{'run_id':rid,'head':git(['rev-parse','HEAD'],repo)['stdout'].strip(),'config':cfg,'backbone_training':0,'lora_training':0,'protected_evaluation_values_access':False})
    try:
        path=ensure_data(repo,cfg); x,splits=load_data(path,cfg); write_json(out/'DATA.json',{'path':str(path),'sha256':sha256(path),'parsed_shape':list(x.shape),'splits':{k:v.tolist() for k,v in splits.items()},'protected_evaluation_start':cfg['dataset']['protected_evaluation_start'],'protected_evaluation_values_access':False})
        cache=repo/'.cache'/cfg['name']/f'{rid}_f0_forecasts.npz'; cache.parent.mkdir(parents=True,exist_ok=True); final=run_experiment(x,splits,cfg,out,cache); final['forecast_cache_sha256']=sha256(cache); write_json(out/'FINAL_DECISION.json',final)
    except Exception as e:
        final={'status':'TECHNICAL_FAILURE','token':'TECHNICAL_FAILURE','error':repr(e),'traceback':traceback.format_exc(),'protected_evaluation_values_access':False}; write_json(out/'FAILURE.json',final)
    if a.publish: print(json.dumps(publish(repo,pkg,out),indent=2))
    print(json.dumps(final,indent=2,ensure_ascii=False)); return 0 if final['status']!='TECHNICAL_FAILURE' else 2
if __name__=='__main__': raise SystemExit(main())
