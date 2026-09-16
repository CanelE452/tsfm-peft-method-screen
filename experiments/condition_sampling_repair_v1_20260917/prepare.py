"""Prepare all protocols, legal packets and TRAIN-only auxiliaries before execution."""
import re,subprocess,sys,math
import pandas as pd
from .common import *
FILES={'electricity':('data/raw/electricity.txt.gz','3c4c069588198c1fcc95cace7bb69c99922129edfd673b7286661dad20badefa'),'traffic':('data/raw/overnight_20260913/traffic.txt.gz','c7be5a00519d344a5ec0eabdbfec5ea0c7dd1eed5f9b1a3843a93bb88086a56d'),'ettm1':('data/raw/overnight_20260913/ETTm1.csv','6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e')}
from .sampling import pick
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
