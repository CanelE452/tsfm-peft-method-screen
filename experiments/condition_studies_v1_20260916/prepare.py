"""Prepare all protocols, legal packets and TRAIN-only auxiliaries before execution."""
import re,subprocess,sys,math
import pandas as pd
from .common import *
FILES={'electricity':('data/raw/electricity.txt.gz','3c4c069588198c1fcc95cace7bb69c99922129edfd673b7286661dad20badefa'),'traffic':('data/raw/overnight_20260913/traffic.txt.gz','c7be5a00519d344a5ec0eabdbfec5ea0c7dd1eed5f9b1a3843a93bb88086a56d'),'ettm1':('data/raw/overnight_20260913/ETTm1.csv','6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e')}
def pick(candidates,count,phases,period):
 pools={p:np.asarray([o for o in candidates if o%period==p]) for p in phases};quota={p:0 for p in phases}
 for _ in range(count):
  avail=[p for p in phases if quota[p]<len(pools[p])]
  if not avail:raise ValueError('BLOCKED_DATA insufficient origins')
  p=min(avail,key=lambda p:(quota[p],phases.index(p)));quota[p]+=1
 return sorted(int(o) for p in phases if quota[p] for o in pools[p][np.floor(np.linspace(0,len(pools[p])-1,quota[p])).astype(int)])
def observation_map(a,train_origins):
 nb=len(a)//24;detail=np.array([int(digest('R09_BLOCK',b)[:16],16)%4==0 for b in range(nb)])
 chosen=sorted(train_origins,key=lambda o:digest('R09_DETAILED',o))[:16]
 for o in train_origins:detail[o//24]=o in chosen
 observed=a.copy()
 for b in range(nb):
  if not detail[b]:observed[b*24:(b+1)*24]=a[b*24:(b+1)*24].mean(0)
 return observed,detail

def spectral(z,origins):
 oo=np.array(origins);future=np.stack([z[o:o+48].T for o in oo]);p1=np.stack([z[o-48:o].T for o in oo]);p2=np.stack([z[o-96:o-48].T for o in oo]);F=np.fft.fft(future,norm='ortho');P=np.fft.fft(p1,norm='ortho');Q=np.fft.fft(p2,norm='ortho');energy=np.mean(abs(F)**2,0);skill=np.zeros((4,48));coef=np.zeros((4,48,5,2));logs=[]
 for c in range(4):
  for k in range(25):
   x=np.stack([P[:,c,k].real,P[:,c,k].imag,Q[:,c,k].real,Q[:,c,k].imag],1);y=np.stack([F[:,c,k].real,F[:,c,k].imag],1);errs=[];bases=[]
   for fold in [1,2,3]:
    vi=np.arange(16*fold,16*(fold+1));ti=np.flatnonzero(oo+48<=oo[vi[0]]-96-48)
    if len(ti)<5:continue
    fm=z[:int(oo[ti[-1]])+48,c].mean();fs=max(z[:int(oo[ti[-1]])+48,c].std(),1e-6);xf=x.copy()/fs;yf=y.copy()/fs
    if k==0:xf[:,[0,2]]-=math.sqrt(48)*fm/fs;yf[:,0]-=math.sqrt(48)*fm/fs
    b=ridge(xf[ti],yf[ti],1.);errs.extend(((b[0]+xf[vi]@b[1:]-yf[vi])**2).sum(1));bases.extend(((yf[vi]-yf[ti].mean(0))**2).sum(1))
   skill[c,k]=np.clip(1-np.mean(errs)/max(np.mean(bases),1e-12),0,1) if len(errs) else 0
   coef[c,k]=ridge(x,y,1.)
   if k not in [0,24]:skill[c,48-k]=skill[c,k];coef[c,48-k]=coef[c,k]*np.array([1,-1])[None,:]*np.array([1,1,-1,1,-1])[:,None]
   logs.append(dict(channel=c,bin=k,validation_points=len(errs),purge='training label end <= validation context start -48',alpha=1.))
 weights={};pos=slice(1,24)
 for arm in ARMS['N07']:
  w=np.ones((4,48))
  for c in range(4):
   ep=energy[c,pos];med=np.median(ep);eps=1e-8*max(energy[c].mean(),1)
   if arm=='G1':v=1/np.sqrt(np.maximum(ep,eps));v=np.clip(v/v.mean(),.25,4)
   elif arm in ['G2','G3']:
    v=1+np.where(ep<=med,skill[c,pos]*np.minimum(4,med/(ep+eps)),0)
    if arm=='G3':v=rng('N07_SHUFFLE',73100,c).permutation(v)
   else:v=np.ones(23)
   w[c,1:24]=v;w[c,25:]=v[::-1];w[c]/=w[c].mean()
  weights[arm]=w
 return dict(energy=energy.tolist(),skill=skill.tolist(),weights={a:w.tolist() for a,w in weights.items()},ridge_coefficients=coef.tolist(),folds=logs,aliases={a:'G0' for a in ['G2','G3'] if np.array_equal(weights[a],weights['G0'])},cpu_regression_scalar_coefficients=4*48*5*2,convention='DC/Nyquist=1 before global mean normalization; spectral ridge alpha1 fixed before any score')

def prepare_track(t,a,columns,receipt):
 out=OUT/t;cache=CACHE/t;out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
 N=len(a);bounds=[0,int(.6*N),int(.7*N),int(.8*N),N];H=24 if t=='R09' else 48;extra=24 if t=='R04' else 0;C=1344 if t=='N02' else 336;period=96 if t=='N03' else 24
 phases=[0] if t=='R09' else (np.floor(np.linspace(0,95,64)).astype(int).tolist() if t=='N03' else list(range(24)))
 # R09 first select by metadata only; observed-value eligibility follows absolute permissions.
 raw_candidates={r:list(range(max(lo,C,2048 if t=='N02' else C),hi-H-extra+1)) for r,lo,hi in zip(ROLES,bounds[:-1],bounds[1:])}
 origins={r:pick(v,64 if r in ['TRAIN','E_DISCOVERY'] else 32,phases,period) for r,v in raw_candidates.items()}
 if t=='R09':observed,detail=observation_map(a,origins['TRAIN']);eligible_values=observed[:bounds[1]]
 else:eligible_values=a[:bounds[1]]
 if t=='R09':eligible_values=observed[:(bounds[1]//24)*24]
 valid=[i for i in range(a.shape[1]) if np.isfinite(eligible_values[:,i]).all() and eligible_values[:,i].std()>1e-6][:4];assert len(valid)==4,'BLOCKED_DATA channels'
 a=a[:,valid];columns=[columns[i] for i in valid]
 if t=='R09':observed=observed[:,valid];base=observed
 else:base=a
 # Only complete natural windows, never quality/performance based.
 cand={r:[o for o in v if np.isfinite(base[o-C:o]).all() and np.isfinite(a[o:o+H+extra]).all()] for r,v in raw_candidates.items()}
 if t=='N02':
  bank=np.arange(336,bounds[1]-48+1,24)
  if len(bank)>2048:bank=bank[np.floor(np.linspace(0,len(bank)-1,2048)).astype(int)]
  cand={r:[o for o in v if np.sum(bank+48<=min(bounds[1],o-336-24))>=32] for r,v in cand.items()}
 common=[p for p in phases if all(any(o%period==p for o in v) for v in cand.values())];assert common
 final={r:pick(v,64 if r in ['TRAIN','E_DISCOVERY'] else 32,common,period) for r,v in cand.items()}
 if t=='R09':assert final==origins,'R09 permissions require preselected complete origins'
 origins=final
 statistics_end=(bounds[1]//24)*24 if t=='R09' else bounds[1]
 mu=base[:statistics_end].mean(0,dtype=np.float64);sigma=base[:statistics_end].std(0,dtype=np.float64);assert np.all(sigma>1e-6)
 stats=dict(mu=mu.tolist(),sigma=sigma.tolist(),statistics_role='TRAIN',train_end=statistics_end,split_train_end=bounds[1],basis='permitted completed TRAIN blocks: detailed plus block means' if t=='R09' else 'raw TRAIN only',columns=columns)
 aux={}
 z=(a-mu)/sigma
 if t=='N01':
  bs=[];rs=[]
  for c in range(4):
   others=[j for j in range(4) if j!=c];b,r=ridge_cv(z[:bounds[1],others],z[:bounds[1],c]);bs.append(b.tolist());rs.append(r)
  aux=dict(ridge=bs,regressions=rs,regression_pairs=4*bounds[1],regression_coefficients=16)
 if t=='R08':
  donors=[];lags=[];bs=[];rs=[]
  for c in range(4):
   options=[]
   for j in range(4):
    if j==c:continue
    cs=[float(np.corrcoef(z[l:bounds[1],c],z[:bounds[1]-l,j])[0,1]) for l in range(1,25)];lag=min(range(1,25),key=lambda l:(-abs(cs[l-1]),l));options.append((abs(cs[lag-1]),j,lag,cs[lag-1]))
   picked=sorted(options,key=lambda v:(-v[0],v[1],v[2]))[:2];ds=[v[1] for v in picked];ls=[v[2] for v in picked];ii=np.arange(24,bounds[1]);x=np.stack([z[ii-l,j] for j,l in zip(ds,ls)],1);b,r=ridge_cv(x,z[ii,c]);donors.append(ds);lags.append(ls);bs.append(b.tolist());rs.append(dict(**r,correlations=[v[3] for v in picked]))
  aux=dict(donors=donors,lags=lags,ridge=bs,regressions=rs,regression_pairs=4*(bounds[1]-24),regression_coefficients=12)
 if t=='N07':aux=spectral(z,origins['TRAIN'])
 if t=='R09':aux=dict(detail_blocks=detail.tolist(),train_detailed=[o for o in origins['TRAIN'] if detail[o//24]],map_rule='sha256(R09_BLOCK,b) mod4; train targets override first16 R09_DETAILED hashes',training_fine_unique=16,training_aggregate_unique=48)
 retrieval=[];packets=[]
 for role,oo in origins.items():
  xs=[];ys=[];rights=[];lookup_p=[];lookup_f=[];lookup_idx=[]
  for o in oo:
   xs.append(base[o-C:o].T.astype(np.float32));y=a[o:o+H+extra].T.astype(np.float64)
   if t=='R09':
    rights.append(np.repeat(detail[o//24-14:o//24],24).astype(np.float32));d=bool(detail[o//24]);
    if role=='TRAIN' and not d:y=np.repeat(y.mean(-1,keepdims=True),H,axis=-1)
   ys.append(y)
   if t=='N02':
    legal=bank[bank+48<=min(bounds[1],o-336-24)];pp=[];ff=[];idx=[]
    for c in range(4):
     query=a[o-336:o,c].astype(float);qm=query.mean();qs=max(query.std(),1e-6);bp=np.stack([a[r-336:r,c] for r in legal]).astype(float);bm=bp.mean(1);sd=np.maximum(bp.std(1),1e-6);dist=np.mean(((bp-bm[:,None])/sd[:,None]-(query-qm)/qs)**2,1);order=np.lexsort((legal,dist))[:2]
     for j in order:
      r=int(legal[j]);pp.append((bp[j]-bm[j])/sd[j]*qs+qm);ff.append((a[r:r+48,c]-bm[j])/sd[j]*qs+qm);idx.append(r);retrieval.append(dict(role=role,origin=o,channel=c,bank_origin=r,continuation_end=r+48,max_legal_end=min(bounds[1],o-360),distance=float(dist[j]),query_sigma=qs))
    lookup_p.append(pp);lookup_f.append(ff);lookup_idx.append(idx)
  inp=dict(context=np.array(xs),origins=np.array(oo))
  if t=='R04':inp['late_context']=np.stack([a[o+24-336:o+24].T for o in oo]).astype(np.float32);inp['innovation_observed']=np.stack([a[o:o+24].T for o in oo]).astype(np.float32)
  if t=='R09':inp['resolution']=np.array(rights);inp['detailed']=np.array([detail[o//24] for o in oo]) if role=='TRAIN' else np.ones(len(oo),bool)
  if t=='N02':inp.update(retrieval_past=np.array(lookup_p,dtype=np.float32),retrieval_future=np.array(lookup_f,dtype=np.float32),retrieval_origins=np.array(lookup_idx))
  npz(cache/f'{role}_inputs.npz',**inp);npz(cache/f'{role}_labels.npz',y=np.array(ys));packets.extend([cache/f'{role}_inputs.npz',cache/f'{role}_labels.npz'])
 save(out/'origins.json',origins);csvwrite(out/'origins.csv',[dict(role=r,origin=o,phase=o%period,horizon_end=o+H+extra) for r,oo in origins.items() for o in oo]);save(out/'train_statistics.json',stats);save(out/'feature_or_transform_manifest.json',aux)
 if retrieval:csvwrite(out/'retrieval_receipt.csv',retrieval)
 permissions=dict(context='strictly earlier than issuance; target labels separate',selection='V_SELECT only',calibration='V_CAL only; no output calibration in neural tracks',evaluation='E_DISCOVERY labels scorer only after seal',source_train_end=bounds[1],R09_absolute_block_permission=t=='R09',R04_early_late_separate_groups=t=='R04',normalization=stats['basis'],input_shapes={r:list(np.load(cache/f'{r}_inputs.npz')['context'].shape) for r in ROLES},target_columns=columns,rows='first4 real targets; appended masks/ages/retrieval are auxiliaries without supervision')
 save(out/'permissions.json',permissions);save(out/'data_receipt.json',dict(**receipt,shape=list(a.shape),selected_columns=columns,bounds=bounds,period=period,origins_count={r:len(v) for r,v in origins.items()},phase_counts={r:{str(p):sum(o%period==p for o in v) for p in common} for r,v in origins.items()},cache_hashes={str(p.relative_to(ROOT)):sha(p) for p in packets},source_arrays_read_for_preparation=True,raw_R09_hidden_values_not_in_training_packets=t=='R09'))
 return dict(source=SOURCES[t],C=C,H=H,frequency_minutes=15 if t=='N03' else 60,bounds=bounds,origins_hash=sha(out/'origins.json'),packet_hashes={str(p.relative_to(ROOT)):sha(p) for p in packets},auxiliary_hash=sha(out/'feature_or_transform_manifest.json'))

def prepare_all(contract):
 assert not (OUT/'MASTER_SEAL.json').exists(),'Already sealed; audit instead of overwrite'
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 text=Path(contract).read_text();(OUT/'MASTER_PROTOCOL.md').write_text(text)
 hist={str(p.relative_to(ROOT)):sha(p) for folder in ['results','research'] for p in (ROOT/folder).rglob('*') if p.is_file() and OUT not in p.parents};save(OUT/'historical_hashes.json',hist)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save(OUT/'repository_audit.json',dict(head=head,base='485b15b3990236d0372fc074f1c80c7f1df057e2',diff=subprocess.check_output(['git','diff','485b15b..HEAD','--stat'],cwd=ROOT,text=True),dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),historical_files=len(hist),duplicate='No exact point-MSE rank8 condition comparison at baseline. Prior48rank1 native-quantile fits reused only as R05/N06 dependencies.',contract_original=contract,contract_sha256=sha(contract)))
 arrays={};receipts={}
 for s,(path,h) in FILES.items():
  p=ROOT/path;assert p.exists() and sha(p)==h,('BLOCKED_DATA',path)
  if s=='ettm1':df=pd.read_csv(p);a=df.iloc[:,1:].to_numpy(float);columns=list(df.columns[1:]);assert pd.to_datetime(df.iloc[:,0]).diff().dropna().eq(pd.Timedelta(minutes=15)).all()
  else:a=np.loadtxt(p,delimiter=',',dtype=np.float64);columns=[f'column_{i:03}' for i in range(a.shape[1])]
  arrays[s]=(a,columns);receipts[s]=dict(path=path,sha256=h,raw_shape=list(a.shape),bytes=p.stat().st_size,download_bytes=0)
 specs=[]
 for t in ORDER:
  out=OUT/t;out.mkdir(exist_ok=True);C=336;H=48
  if t in SOURCES:di=prepare_track(t,*arrays[SOURCES[t]],receipts[SOURCES[t]]);C=di['C'];H=di['H']
  else:
   old=ROOT/'results/forecast_path_structure_v1_20260916';files=['data_seal.json','train_scaling.json','selection_seal.json','prediction_manifest.json','calibration_parameters.json','evaluation_seal.json','completion_audit.json'];di=dict(source='saved_forecast_path_structure',dependencies={str((old/p).relative_to(ROOT)):sha(old/p) for p in files},exposure='REANALYSIS_REUSED_E');save(out/'data_receipt.json',di);save(out/'permissions.json',dict(fits=0,policy='R05 V_SELECT7월 정책; N06 V_CAL8월 dependence; E10~12월 재사용',no_new_neural_training=True));save(out/'train_statistics.json',read(old/'train_scaling.json'));save(out/'feature_or_transform_manifest.json',dict(policy_grid=[0,.25,.5,.75,1] if t=='R05' else None,sample_count=256 if t=='N06' else None));H=24
  section={'N01':5,'N02':6,'N03':7,'R04':8,'R05':9,'N06':10,'N07':11,'R08':12,'R09':13}[t];match=re.search(r'\n'+str(section)+r'\. .*?(?=\n={20,}\n'+str(section+1)+r'\.)',text,re.S);assert match
  excerpt=match.group(0);(out/'TOPIC_ONEPAGE.md').write_text('# '+t+' '+NAMES[t]+'\n\n'+excerpt+'\n\n[설계] 전체 계약의 공통 조항도 적용. 실행/효과/단순 대안/신규성을 분리한다.\n')
  spec=dict(id=t,name=NAMES[t],question=QUESTIONS[t],arms=ARMS[t],primary='mean_channel sqrt(mean_origin,h normalized_squared_error)' if t in SOURCES and t!='R04' else {'R04':'accuracy-constrained raw revision RMS','R05':'S0<=1.01 E0; delayed mean and worst pinball','N06':'sum ensemble CRPS / (24 sigma)'}.get(t),proposed=PROPOSED[t],direct_controls=CONTRASTS[t],C=C,H=H,data=di,main_fit_cap=4*len(ARMS[t]) if t in SOURCES else 0,main_update_cap=512*4*len(ARMS[t]) if t in SOURCES else 0,smoke_updates=2*len(ARMS[t]) if t in SOURCES else 0,LRs=LRS,seeds=SEEDS,selection='seed73100 twoLR select by V; repeats73101/2 only fixedLR; step0/256/512',loss='FP32 pointMSE normalized TRAIN sigma; contract-specific terms',bootstrap=dict(replicates=2000,seed=73300,block_slots=672 if t=='N03' else 168),contract_section=excerpt,implementation_choices=dict(N01_target='selected TRAIN origin ordinal i mod4; delay [0,6,24][epoch mod3]',N03_mask='sha256 CLOCK,origin,channel,epoch; V/E epoch key=-1; irregular alternating1/3 with hash start0..3/order',N07_ridge_alpha=1.,N07_shuffle_seed=73100,R04_reference='repeat-specific selected D0; seed73100 selected across2LR',N01_age_rows='elapsed slots/24 clipped2',N03_age_rows='elapsed slots/4; first unseen sentinel336/4',N02_zero_std_floor=1e-6,R09_resolution_rows='one repeated block resolution row shared all4 channels; same allarms',CPU_blend_ties='smaller alpha; R04 accuracy then revision constraint same as neural'))
  save(out/'PROTOCOL.json',spec);save(out/'STATUS.json',dict(EXECUTION='PREPARED',EVIDENCE='NOT_MEASURED',NOVELTY='KNOWN_CONTROL' if t in ['R05','N06'] else 'UNVERIFIED_VARIANT'));specs.append(spec)
 save(OUT/'MASTER_MANIFEST.json',dict(queue=ORDER,specs=specs,budget=dict(main_fits=116,main_updates=59392,discarded_updates=96,total_updates=59488,forwards=200000,controller_seconds=86400,wait_seconds=1800,download_bytes=2*2**30,cache_bytes=100*2**30,disk_free_bytes=10*2**30),prepared_at=time.time(),contract_sha256=sha(OUT/'MASTER_PROTOCOL.md')))
 (OUT/'SOURCE_AND_EXPOSURE_LEDGER.md').write_text('# 자료·노출 장부\n\nElectricity/Traffic/ETT는 기존 연구에 사용된 공개 원천이며 기간 비중복과 Chronos 사전학습 비중복을 주장하지 않는다. E_DISCOVERY는 재사용 개발 평가다. 원자료 판독·통계와 모델선택 정답 사용은 구분한다. R05/N06의 T0/T1/T2 10~12월 점수는 485b15b에서 이미 공개됐으며 REANALYSIS_REUSED_E로만 해석한다.\n\n기존48경로의 가중치와 예측은 hash 검증 후 참조한다. 이번116경로는 rank8 pointMSE, 새분할·정보권한·seed로 정확히 동일한 완료 비교가 아니다. 기존 결과는 historical_hashes.json으로 보존한다. 다운로드0바이트; 로컬 공개자료만 재사용. 단독문서의 실제 파일명은 repository_audit.json 참조. 다운로드 ZIP들에서 MASTER_CLI/reference_checks는 발견되지 않아 동봉 synthetic 검산은 실행했다고 주장하지 않는다. 실제 runner의 별도 검사를 수행한다.\n')
 save(OUT/'QUEUE_STATUS.json',dict(queue=ORDER,current=None,status='PREPARED',completed=[]))
 print('PREPARED_ALL',[(s['id'],s['main_fit_cap']) for s in specs],flush=True)
