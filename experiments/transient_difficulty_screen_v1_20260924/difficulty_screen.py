from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,os,random,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
import numpy as np

class ContractError(RuntimeError): pass
def require(x,msg):
    if not x: raise ContractError(msg)
def write_json(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,ensure_ascii=False,default=lambda x:float(x) if hasattr(x,'item') else str(x)),encoding='utf-8')
def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def cmdrun(cmd,cwd=None,timeout=600):
    p=subprocess.run(cmd,cwd=cwd,text=True,capture_output=True,timeout=timeout)
    return {'cmd':cmd,'cwd':str(cwd) if cwd else None,'returncode':p.returncode,'stdout':p.stdout[-8000:],'stderr':p.stderr[-8000:]}
def rankdata(a):
    a=np.asarray(a,float);order=np.argsort(a,kind='mergesort');r=np.empty(len(a),float);i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]:j+=1
        r[order[i:j]]=(i+j-1)/2+1;i=j
    return r
def spearman(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return float('nan')
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])

MEAS=['reaTZon_y','reaTSup_y','reaTRet_y','reaPHeaPum_y','reaQHeaPumCon_y','reaQFloHea_y','reaTSetHea_y','weaSta_reaWeaTDryBul_y']
CTRL=['oveHeaPumY_u','ovePum_u','oveFan_u']
def req(base,method,path,payload=None,timeout=300):
    if payload is None and method in ('POST','PUT'):payload={}
    data=None if payload is None else json.dumps(payload).encode()
    q=urllib.request.Request(f'{base}/{path}',data=data,method=method,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(q,timeout=timeout) as f:body=f.read().decode()
    try:return json.loads(body)
    except json.JSONDecodeError:return {'payload':body.strip(),'raw':True}
def service_ready(base):
    try:return True,req(base,'GET','testcases',timeout=10)
    except Exception as e:return False,repr(e)
def maybe_start(repo,base):
    ok,obs=service_ready(base)
    if ok:return {'status':'ALREADY_UP'}
    bop=Path(repo).parent/'project1-boptest'
    if not bop.is_dir():return {'status':'NO_LOCAL_BOPTEST_REPO','initial_error':obs}
    r=cmdrun(['docker','compose','up','-d','--no-build','web','worker','provision'],cwd=bop,timeout=300)
    if r['returncode']!=0:return {'status':'COMPOSE_UP_FAILED','receipt':r}
    for _ in range(60):
        time.sleep(5);ok,_=service_ready(base)
        if ok:return {'status':'STARTED_EXISTING_LOCAL_SERVICE','receipt':r}
    return {'status':'SERVICE_NOT_READY_AFTER_START','receipt':r}
def make_schedule(nsteps,seed):
    rng=random.Random(seed);levels=[0.0,0.35,0.7,1.0];dwell=[6,12,18,24,36];out=[];cur=levels[seed%4]
    while len(out)<nsteps:
        cur=rng.choice([x for x in levels if x!=cur]);out.extend([cur]*rng.choice(dwell))
    out=out[:nsteps];require(sum(out[i]!=out[i-1] for i in range(1,len(out)))>=8,'Too few transitions');return out
def collect_episode(cfg,i,outdir):
    base=cfg['boptest_url'];sel=req(base,'POST',f"testcases/{cfg['testcase']}/select",timeout=120);tid=sel.get('testid') or sel.get('payload',{}).get('testid');require(tid,'No testid')
    try:
        step=float(cfg['control_step_s']);start=(cfg['episode_start_days'][i]-1)*86400.0
        req(base,'PUT',f'step/{tid}',{'step':step});req(base,'PUT',f'initialize/{tid}',{'start_time':start,'warmup_period':cfg['warmup_s']},timeout=600)
        nsteps=int(cfg['episode_hours']*3600/step);sched=make_schedule(nsteps,cfg['schedule_seed_base']+i)
        for u in sched:req(base,'POST',f'advance/{tid}',{'oveHeaPumY_activate':1,'oveHeaPumY_u':u,'ovePum_activate':1,'ovePum_u':1.0 if u>0 else 0.0,'oveFan_activate':1,'oveFan_u':1.0 if u>0 else 0.0},timeout=300)
        res=req(base,'PUT',f'results/{tid}',{'point_names':MEAS+CTRL,'start_time':start,'final_time':start+nsteps*step+step},timeout=600)['payload']
        cols=['time']+[c for c in MEAS+CTRL if c in res];n=len(res['time']);require(n>100 and 'reaTZon_y' in cols and 'reaTSup_y' in cols and 'oveHeaPumY_u' in cols,'Bad result')
        path=Path(outdir)/f'episode_{i:02d}.csv.gz'
        with gzip.open(path,'wt',newline='') as f:
            w=csv.writer(f);w.writerow(cols)
            for k in range(n):w.writerow([res[c][k] for c in cols])
        meta={'episode':i,'start_day':cfg['episode_start_days'][i],'n_samples':n,'stored_dt_s':float(res['time'][1]-res['time'][0]),'n_transitions':sum(sched[j]!=sched[j-1] for j in range(1,len(sched))),'schedule_sha256':hashlib.sha256(json.dumps(sched).encode()).hexdigest(),'file_sha256':sha256(path)}
        write_json(Path(outdir)/f'episode_{i:02d}.json',meta);return meta
    finally:
        try:req(base,'PUT',f'stop/{tid}',timeout=120)
        except Exception:pass

def load_gz(p):
    with gzip.open(p,'rt') as f:rows=list(csv.reader(f))
    h=rows[0];a=np.asarray(rows[1:],float);return {k:a[:,j] for j,k in enumerate(h)}
def regularize(d,sample_s):
    dt=float(np.median(np.diff(d['time'])));stride=max(1,int(round(sample_s/dt)));require(abs(stride*dt-sample_s)<1e-6,'Bad resample');idx=np.arange(0,len(d['time']),stride);return {k:v[idx] for k,v in d.items()}
def transitions(u):
    z=np.zeros(len(u),bool);z[1:]=np.abs(np.diff(u))>1e-8;return z
def win(i,sec,dt):n=max(1,int(round(sec/dt)));return max(0,i-n+1),i+1
def slope(y,dt):
    if len(y)<2:
        return 0.0
    x=np.arange(len(y))*dt
    x=x-x.mean(); yy=y-y.mean(); den=(x*x).sum()
    return float((x*yy).sum()/den) if den else 0.0
def target_feat(y,i,dt):
    f=[float(y[i])]
    for sec in (300,900,1800,3600,7200,14400,28800):
        j=i-int(round(sec/dt));f.append(float(y[j]) if j>=0 else float(y[0]))
    for sec in (600,1800,3600,7200):
        a,b=win(i,sec,dt);w=y[a:b];f.extend([float(w.mean()),float(w.std()),slope(w,dt)])
    return f
def command_feat(u,i,dt):
    cur=float(u[i]);o=[]
    for sec in (1800,7200,21600):
        a,b=win(i,sec,dt);w=u[a:b];o.extend([float(w.mean()),float(w.std()),abs(cur-float(w.mean())),float(transitions(w).sum()),float(np.abs(np.diff(w)).sum()) if len(w)>1 else 0.])
    z=np.where(transitions(u[:i+1]))[0];o.append(float((i-z[-1])*dt if len(z) else (i+1)*dt));return o
def origins(d,cfg,ep):
    q=regularize(d,cfg['sample_s']);y=q[cfg['target']];u=q['oveHeaPumY_u'];t=q['time'];dt=float(cfg['sample_s']);ctx=int(round(cfg['context_s']/dt));H=int(round(cfg['horizon_s']/dt));stride=max(1,int(round(cfg['origin_stride_s']/dt)));out=[]
    for i in range(ctx-1,len(y)-H,stride):
        base=target_feat(y,i,dt)+[float(u[i])];hist=base+command_feat(u,i,dt)
        out.append({'episode':ep,'origin_time':float(t[i]),'X_current':base,'X_history':hist,'Y':y[i+1:i+1+H].copy(),'target_context':y[i-ctx+1:i+1].copy(),'command_context':u[i-ctx+1:i+1].copy(),'command_future':u[i+1:i+1+H].copy()})
    return out
class Ridge:
    def __init__(self,a=1.):self.a=a
    def fit(self,X,Y):
        X=np.asarray(X,float);Y=np.asarray(Y,float);self.mx=X.mean(0);self.sx=X.std(0);self.sx[self.sx<1e-8]=1.;self.my=Y.mean(0);Z=(X-self.mx)/self.sx;self.B=np.linalg.solve(Z.T@Z+self.a*np.eye(Z.shape[1]),Z.T@(Y-self.my));return self
    def predict(self,X):X=np.asarray(X,float);return ((X-self.mx)/self.sx)@self.B+self.my
def gather(rows,k):return np.asarray([r[k] for r in rows],float)
def mae(a,b):return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))
def pairfit(rows,a):return Ridge(a).fit(gather(rows,'X_current'),gather(rows,'Y')),Ridge(a).fit(gather(rows,'X_history'),gather(rows,'Y'))
def escore(c,h,rows):
    y=gather(rows,'Y');ec=mae(y,c.predict(gather(rows,'X_current')));eh=mae(y,h.predict(gather(rows,'X_history')));return {'mae_current':ec,'mae_history':eh,'history_gain_pct':100*(ec-eh)/ec if ec else 0.}
def screen(cfg,out):
    train=cfg['splits']['train'];dev=cfg['splits']['dev'];data={i:load_gz(out/'episodes'/f'episode_{i:02d}.csv.gz') for i in train+dev};res={};cache={}
    for cc in cfg['candidate_configs']:
        rr={i:origins(data[i],cc,i) for i in train+dev};require(all(len(rr[i])>=8 for i in train+dev),'Too few origins');fold=[]
        for hold in train:
            tr=[r for e in train if e!=hold for r in rr[e]];c,h=pairfit(tr,cfg['ridge_alpha']);s=escore(c,h,rr[hold]);s['episode']=hold;fold.append(s)
        gains=[x['history_gain_pct'] for x in fold];elig=float(np.mean(gains))>=cfg['train_gate']['min_mean_history_gain_pct'] and sum(g>0 for g in gains)>=cfg['train_gate']['min_positive_train_folds']
        res[cc['name']]={'config':cc,'train_folds':fold,'mean_history_gain_pct':float(np.mean(gains)),'positive_folds':int(sum(g>0 for g in gains)),'eligible':elig,'origin_counts':{str(i):len(rr[i]) for i in train+dev}};cache[cc['name']]=rr
    elig=[k for k,v in res.items() if v['eligible']];selected=None if not elig else sorted(elig,key=lambda k:(-res[k]['mean_history_gain_pct'],res[k]['config']['horizon_s']))[0]
    write_json(out/'TRAIN_SCREEN.json',{'candidates':res,'selected':selected})
    if selected is None:return {'status':'NO_HISTORY_SIGNAL','selected':None},None,None,None
    rr=cache[selected];trrows=[r for e in train for r in rr[e]];c,h=pairfit(trrows,cfg['ridge_alpha']);ds=[];all_dev=[]
    for e in dev:s=escore(c,h,rr[e]);s['episode']=e;ds.append(s);all_dev+=rr[e]
    replicated=all(x['history_gain_pct']>0 for x in ds);train_std=max(float(np.std(gather(trrows,'Y'))),1e-8);pc=c.predict(gather(all_dev,'X_current'));ph=h.predict(gather(all_dev,'X_history'));tmd=np.mean(np.abs(ph-pc),axis=1)/train_std;y=gather(all_dev,'Y');benef=np.mean(np.abs(y-pc),axis=1)-np.mean(np.abs(y-ph),axis=1)
    write_json(out/'DEV_HISTORY.json',{'selected':selected,'dev_scores':ds,'replicated':replicated,'tmd':{'n':len(tmd),'min':float(tmd.min()),'median':float(np.median(tmd)),'max':float(tmd.max())},'corr_tmd_history_benefit':spearman(tmd,benef)})
    return {'status':'HISTORY_SIGNAL_REPLICATED' if replicated else 'HISTORY_SIGNAL_NOT_REPLICATED','selected':selected},(c,h),(tmd,all_dev,train_std),res[selected]['config']
def choose(rows,eps,maxn):
    each=max(1,maxn//len(eps));o=[]
    for e in eps:
        rr=[r for r in rows if r['episode']==e];ids=np.arange(len(rr)) if len(rr)<=each else np.linspace(0,len(rr)-1,each).round().astype(int);o += [rr[int(i)] for i in ids]
    return o[:maxn]
def eval_f0(cfg,cc,models,devpack,out):
    import torch
    from chronos import Chronos2Pipeline
    c,h=models;_,rows,train_std=devpack;rows=choose(rows,cfg['splits']['dev'],cfg['f0_max_origins_total']);Xc=gather(rows,'X_current');Xh=gather(rows,'X_history');Y=gather(rows,'Y');pc=c.predict(Xc);ph=h.predict(Xh);tmd=np.mean(np.abs(ph-pc),axis=1)/train_std
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';pipe=Chronos2Pipeline.from_pretrained('amazon/chronos-2',device_map='cpu',dtype=torch.float32,local_files_only=True)
    inp=[{'target':np.asarray(r['target_context'],np.float32),'past_covariates':{'command':np.asarray(r['command_context'],np.float32)},'future_covariates':{'command':np.asarray(r['command_future'],np.float32)}} for r in rows]
    qs,_=pipe.predict_quantiles(inp,prediction_length=Y.shape[1],quantile_levels=list(pipe.quantiles),batch_size=8);mi=min(range(len(pipe.quantiles)),key=lambda i:abs(float(pipe.quantiles[i])-.5));pred=np.stack([q[0,:,mi].float().cpu().numpy() for q in qs]);ef=np.mean(np.abs(Y-pred),1);ec=np.mean(np.abs(Y-pc),1);eh=np.mean(np.abs(Y-ph),1);q1,q2=np.quantile(tmd,[1/3,2/3]);bins=np.where(tmd<=q1,0,np.where(tmd<=q2,1,2));br=[]
    for b,n in enumerate(('low','mid','high')):
        m=bins==b;br.append({'bin':n,'n':int(m.sum()),'tmd_mean':float(tmd[m].mean()),'f0_mae':float(ef[m].mean()),'current_mae':float(ec[m].mean()),'history_mae':float(eh[m].mean()),'f0_minus_history_norm':float((ef[m]-eh[m]).mean()/train_std),'history_gain_pct':float(100*(ec[m].mean()-eh[m].mean())/ec[m].mean())})
    z={'status':'F0_EVALUATED_ON_DEV_ONLY','selected_config':cc['name'],'n':len(rows),'f0_mae':float(ef.mean()),'current_mae':float(ec.mean()),'history_mae':float(eh.mean()),'corr_tmd_f0_error':spearman(tmd,ef),'corr_tmd_f0_minus_history':spearman(tmd,ef-eh),'bins':br,'reserve_scored':False};write_json(out/'F0_DEV.json',z);return z
def publish(repo,pkg,out):
    dst=repo/'experiments'/'transient_difficulty_screen_v1_20260924';require(not dst.exists(),'Experiment destination exists');shutil.copytree(pkg,dst)
    staged=cmdrun(['git','diff','--cached','--name-only'],cwd=repo);require(not staged['stdout'].strip(),'Pre-existing staged changes')
    rels=[str(dst.relative_to(repo)),str(out.relative_to(repo))];a=cmdrun(['git','add','--',*rels],cwd=repo);require(a['returncode']==0,'git add failed');d=cmdrun(['git','diff','--cached','--check'],cwd=repo);require(d['returncode']==0,'diff check failed');c=cmdrun(['git','commit','-m',f'Screen transition-memory difficulty ({out.name}); preserve fixed protocol'],cwd=repo,timeout=300);require(c['returncode']==0,'commit failed '+c['stderr']);sha=cmdrun(['git','rev-parse','HEAD'],cwd=repo)['stdout'].strip();p=cmdrun(['git','push','origin','HEAD:main'],cwd=repo,timeout=300);return {'commit':sha,'push_returncode':p['returncode'],'push_stdout':p['stdout'],'push_stderr':p['stderr']}
def main(repo,pkg,pub):
    repo=Path(repo).resolve();pkg=Path(pkg).resolve();require((repo/'.git').exists() and repo.name=='tsfm-peft-method-screen','Wrong repo');cfg=read_json(pkg/'RUN_CONFIG.json');runid=time.strftime('run_%Y%m%dT%H%M%SZ',time.gmtime());out=repo/'results'/'transient_difficulty_screen_v1_20260924'/runid;require(not out.exists(),'Run exists');(out/'episodes').mkdir(parents=True)
    man={'run_id':runid,'head_at_start':cmdrun(['git','rev-parse','HEAD'],cwd=repo)['stdout'].strip(),'config':cfg,'research_fit':0,'lora_fit':0,'peft_fit':0,'reserve_scored':False};write_json(out/'RUN_MANIFEST.json',man);svc=maybe_start(repo,cfg['boptest_url']);write_json(out/'BOPTEST_SERVICE.json',svc);require(svc['status'] in ('ALREADY_UP','STARTED_EXISTING_LOCAL_SERVICE'),'BOPTEST unavailable')
    meta=[]
    for i in range(8):meta.append(collect_episode(cfg,i,out/'episodes'))
    write_json(out/'EPISODES.json',{'episodes':meta,'splits':cfg['splits'],'reserve_note':'episode 7 collected and hashed only; analysis code never opens it'})
    sc,models,dp,cc=screen(cfg,out);dec={'screen_status':sc['status'],'selected':sc['selected'],'f0':None,'reserve_scored':False,'lora_fit':0,'peft_fit':0}
    if sc['status']=='HISTORY_SIGNAL_REPLICATED':
        f0=eval_f0(cfg,cc,models,dp,out);dec['f0']=f0;hi=f0['bins'][-1]['f0_minus_history_norm'];lo=f0['bins'][0]['f0_minus_history_norm'];corr=f0['corr_tmd_f0_minus_history'];strat=bool(np.isfinite(corr) and corr>=.2 and hi-lo>=.02);dec['f0_stratification_observed']=strat;dec['token']='DIFFICULTY_CANDIDATE_F0_STRATIFIED' if strat else 'DIFFICULTY_CANDIDATE_REPLICATED'
    elif sc['status']=='HISTORY_SIGNAL_NOT_REPLICATED':dec['token']='HISTORY_SIGNAL_NOT_REPLICATED'
    else:dec['token']='NO_HISTORY_SIGNAL'
    write_json(out/'FINAL_DECISION.json',dec);write_json(out/'REPORT_KO.json',{'purpose':'command-history dependence difficulty screening','decision':dec,'not_claimed':['LoRA residual verified','new PEFT works','novelty established','reserve confirmation'],'next':'ChatGPT에서 분석 후 다음 단계 결정'})
    if pub:
        r=publish(repo,pkg,out);print(json.dumps(r,indent=2))
    print(json.dumps(dec,indent=2,ensure_ascii=False));return 0
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--package-root',required=True);ap.add_argument('--publish',action='store_true');a=ap.parse_args();raise SystemExit(main(a.repo,a.package_root,a.publish))
