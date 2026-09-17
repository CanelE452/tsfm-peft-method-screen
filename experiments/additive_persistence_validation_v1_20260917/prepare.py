"""Read-only historical audit and performance-free data/panel construction."""
import subprocess,hashlib,shutil
import pandas as pd
from .common import *
from experiments.outlier_signal_peft_v1_20260917.reference_core import rng,scale,transform,STATES,select_days_reference
SHAPES=['STEP6_D17','STEP6_D63','STEP12_D17','STEP12_D63','STEP8_D31','STEP8_D33','RAMP8_D32','PULSE8_D32']

def links():
    for name in ['data','conditions']:
        (CACHE/name).mkdir(parents=True,exist_ok=True)
        for source in ['electricity','ettm1']:
            p=CACHE/name/source
            if not p.exists():p.symlink_to((OLDC/name/source).resolve(),target_is_directory=True)

def historical():
    if (OUT/'SOURCE_MANIFEST.json').exists():
        for p,h in read(OUT/'SOURCE_MANIFEST.json')['immutable_hashes'].items():assert sha(ROOT/p)==h,p
        return
    manifest=read(OLD/'SOURCE_MANIFEST.json')['immutable_hashes']; hashes=dict(manifest)
    for directory in [OLD,ROOT/'experiments/additive_b0_adapter_v1_20260917']:
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:hashes[str(p.relative_to(ROOT))]=sha(p)
    b=read(OLD/'BASELINE_MANIFEST.json');save(OUT/'BASELINE_MANIFEST.json',{**b,'new_baselines_file':'NEW_BASELINES.json','new_B0_fits_count_in_58':7})
    models=[dict(r,reused=True) for r in read(OLD/'MODEL_SELECTION.json')]
    for source in ['electricity','ettm1']:
        for seed in [81551,81552]:models.append(dict(b['models'][f'{source}_{seed}'],arm='C0',reused=True))
    for r in models:
        assert sha(ROOT/r['checkpoint'])==r['sha256'];hashes[r['checkpoint']]=r['sha256']
        # Preserve final epoch weights for the predeclared supplemental comparison.
        if r['arm']!='C0':
            receipt=read(OLD/'fits'/r['fit']/'receipt.json');last=next(c for c in receipt['checkpoints'] if c['step']==1024)
            assert sha(ROOT/last['checkpoint'])==last['sha256'];hashes[last['checkpoint']]=last['sha256']
    for name,h in hashes.items():assert sha(ROOT/name)==h,name
    save(OUT/'REUSED_MODELS.json',models)
    save(OUT/'SOURCE_MANIFEST.json',dict(initial_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),immutable_hashes=hashes,completed_exact_contract_reuse=False))
    (OUT/'REPO_AUDIT.md').write_text('# 저장소 및 가중치 감사\n\n기준 196820e와 origin/main 및 로컬 HEAD 일치. 사용자 변경 없음. 기존 additive 실험 24경로가 COMPLETE/VERIFIED이며 본 계약의 완료 결과는 없었다. 기존 B0 및 C1/C2/C3 selected 가중치와 최종1024 checkpoint를 hash로 확인했다. 과거 경로는 읽기 전용 참조로 보존한다. GPU는 RustDesk만 compute 예외로 승인되어 있다.\n')

def exposure(frame):
    p=OUT/'SERIES_PANEL_MANIFEST.json'
    if p.exists():return read(p)
    evidence=[]
    repos=[ROOT,ROOT.parent/'mlmltime',ROOT.parent/'timeseries',ROOT.parent/'forecast-revision-peft',ROOT.parent/'intermittent-tsfm-calibration',ROOT.parent/'occurrence-magnitude-coupling']
    for repo in repos:
        for directory in ['results','research','manifests']:
            folder=repo/directory
            if not folder.exists():continue
            for f in folder.rglob('*'):
                if not f.is_file() or f.suffix not in ['.md','.json','.yaml','.yml'] or OUT in f.parents or f.stat().st_size>4_000_000:continue
                text=f.read_text(errors='replace')
                if 'electricity' not in text.lower() and 'ettm2' not in text.lower():continue
                # Evidence inventory distinguishes references from demonstrated runs below.
                evidence.append(dict(path=str(f),sha256=sha(f),mentions_electricity='electricity' in text.lower(),mentions_ettm2='ettm2' in text.lower()))
    save(OUT/'EXPOSURE_EVIDENCE_FILES.json',evidence)
    old=read(OLD/'DATA_MANIFEST.json')['electricity'];origins=np.load(CACHE/'data/electricity/E_DISCOVERY_inputs.npz')['origins'];b=old['bounds'][1]
    known=set(range(64)) # actual 64-channel R2 training/evaluation in priority12, not merely download.
    rows=[];eligible=[]
    for c in frame.columns:
        v=frame[c].to_numpy(float);s=float(np.std(v[:b]));ok=np.isfinite(v[:b]).all() and s>1e-8 and all(np.isfinite(v[o-512:o]).all() for o in origins)
        cid=str(c);order=hashlib.sha256(('additive-persistence-v1\0'+cid).encode()).hexdigest()
        status='CURRENT_B0_TRAINED' if int(c)<4 else 'PRIOR_PERFORMANCE_EXPOSED' if int(c) in known else 'UNKNOWN_PRIOR_EXPOSURE'
        rows.append(dict(source='electricity',canonical_id=cid,exposure=status,current_model_trained=int(c)<4,eligible=bool(ok),hash_order=order,sigma_train=s,pretraining='UNKNOWN_PRETRAINING_OVERLAP',evidence='results/priority12_20260915/channel_data_manifest.json; results/priority12_resume_20260915/REPORT.md' if int(c) in known else 'No exhaustive historical nonuse proof; inventory absence does not establish nonuse'))
        if ok and int(c)>=4:eligible.append((order,cid))
    chosen=[v[1] for v in sorted(eligible)[:16]];assert len(chosen)==16
    rows += [dict(source='ettm2',canonical_id=c,exposure='PRIOR_PERFORMANCE_EXPOSED',current_model_trained=False,eligible=True,hash_order='',sigma_train='',pretraining='UNKNOWN_PRETRAINING_OVERLAP',evidence='results/screening_summary/ettm2_manifest.json; research/peft_rethink_2026_09_13/REPORT.md') for c in ['HUFL','HULL','MUFL','MULL']]
    csvwrite(OUT/'DATA_EXPOSURE_MANIFEST.csv',rows)
    d=dict(canonical_id_definition='zero-based raw electricity column string, same as original selected_columns',selected_columns=chosen,selection='sha256(additive-persistence-v1\\0 + canonical_id), TRAIN eligibility only',verified_unused_count=0,independent_series_status='UNRESOLVED',panel_role='exploratory untrained-in-current-model series transfer; reused source/dates, not independent data',meter_groups='UNAVAILABLE; channel separation only; shared weather correlations remain',prior_exposure_by_channel={r['canonical_id']:r['exposure'] for r in rows if r['source']=='electricity' and r['canonical_id'] in chosen},evidence_files=len(evidence))
    save(p,d);return d

def make_packets(source,frame,columns,period,bounds,origins_by_role,times=None):
    a=frame[columns].to_numpy(float);sigma=a[:bounds[1]].std(0,ddof=0);assert (sigma>1e-8).all()
    f=CACHE/'data'/source;f.mkdir(parents=True,exist_ok=True);packets={};audit=[]
    for role,origins in origins_by_role.items():
        role_id={'TRAIN':0,'V_SELECT':1,'E_DISCOVERY':2}[role];lo,hi=bounds[role_id:role_id+2]
        assert (origins>=max(512,lo)).all() and (origins+64<=hi).all()
        x=np.stack([a[o-512:o].T for o in origins]).astype(np.float32);assert np.isfinite(x).all()
        # Future packet is separate, not inspected for magnitudes/selection or transform generation.
        y=np.stack([a[o:o+64].T for o in origins])
        np.savez_compressed(f/f'{role}_inputs.npz',x=x,sigma=sigma,origins=origins)
        np.savez_compressed(f/f'{role}_labels.npz',y=y)
        packets[role]={str(p.relative_to(ROOT)):sha(p) for p in [f/f'{role}_inputs.npz',f/f'{role}_labels.npz']}
        target=origins[:,None]+np.arange(64);unique,multiplicity=np.unique(target,return_counts=True)
        audit.append(dict(source=source,role=role,days=len(np.unique(origins//period)),count=len(origins),full_days=max(0,(hi-64-(period-1))//period-(max(512,lo)+period-1)//period+1),phase_histogram=json.dumps(np.bincount(origins%period,minlength=period).tolist()),date_deciles=json.dumps(np.histogram(origins//period,bins=10)[0].tolist()),week_blocks=len(np.unique(origins//(7*period))),first_origin=int(origins.min()),last_origin=int(origins.max()),first_timestamp=str(times.iloc[origins.min()]) if times is not None else '',last_timestamp=str(times.iloc[origins.max()]) if times is not None else '',unique_target_timestamps=len(unique),total_target_timestamps=target.size,max_overlap=int(multiplicity.max()),missing_input_values=0,missing_target_values=int((~np.isfinite(y)).sum())))
    return dict(rows=len(frame),selected_columns=list(map(str,columns)),period=period,bounds=bounds,sigma_train_population=sigma.tolist(),packet_hashes=packets),audit

def eval_conditions(source,role):
    f=CACHE/'conditions'/source;f.mkdir(parents=True,exist_ok=True)
    d=np.load(CACHE/'data'/source/f'{role}_inputs.npz');nc=len(d['sigma']);base=d['x'].reshape(-1,512);s=np.tile(d['sigma'],len(d['origins']));n=len(base)
    xs=[];offset=[]
    for state in STATES:
        for g in range(2):
            for i in range(n):
                x,delta,meta=transform(base[i],s[i],state,rng(81900 if role=='V_SELECT' else 82000,source,role,state,g,i));xs.append(x);offset.append(delta)
    np.save(f/f'{role}_x.npy',np.array(xs,np.float32));np.save(f/f'{role}_sigma.npy',np.tile(s,20).astype(np.float32));np.save(f/f'{role}_offset.npy',np.array(offset,float))
    if role=='V_SELECT':
        y=np.load(CACHE/'data'/source/f'{role}_labels.npz')['y'].reshape(n,64);assert np.isfinite(y).all()
        np.save(f/'V_SELECT_y.npy',np.tile(y,(20,1))+np.array(offset)[:,None])

def shape_packet(source):
    f=CACHE/'shapes'/source;f.mkdir(parents=True,exist_ok=True)
    d=np.load(CACHE/'data'/source/'E_DISCOVERY_inputs.npz');idx=np.linspace(0,127,64,dtype=int);base=d['x'][idx].reshape(-1,512);s=np.tile(d['sigma'],64)
    obs=[];offset=[];names=SHAPES+['PAIRED_SHIFT8_D32']
    for name in names:
        for sign in [-1,1]:
            for x0,sigma in zip(base,s):
                x=x0.astype(float).copy();r=scale(x,sigma)
                if name.startswith('STEP'):a,dur=map(int,name.replace('STEP','').split('_D'))
                else:a,dur=8,32
                delta=sign*a*r
                if name.startswith('RAMP'):x[-32:]+=np.linspace(0,delta,32)
                else:x[-dur:]+=delta
                obs.append(x);offset.append(0. if name.startswith('PULSE') else delta)
    obs=np.array(obs,np.float32);n=len(base)
    assert np.array_equal(obs[7*2*n:8*2*n],obs[8*2*n:9*2*n])
    np.save(f/'x.npy',obs);np.save(f/'sigma.npy',np.tile(s,len(names)*2).astype(np.float32));np.save(f/'offset.npy',np.array(offset,float));np.save(f/'origin_indices.npy',idx)
    save(f/'manifest.json',dict(states=names,origin_indices=idx.tolist(),origins=d['origins'][idx].tolist(),signs=[-1,1],labels_read=False,paired_reference_not_new_form='PAIRED_SHIFT8_D32 is the existing SHIFT8 form with balanced signs, used solely to pair PULSE; additionally score legacy PULSE using exact historical SHIFT8 inputs/predictions'))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True);links();historical()
    if (OUT/'DATA_READY.json').exists():
        for p,h in read(OUT/'DATA_READY.json')['hashes'].items():assert sha(ROOT/p)==h,p
        return
    data=read(OLD/'DATA_MANIFEST.json');df=pd.read_csv(ROOT/data['electricity']['path'],header=None);panel=exposure(df)
    for source in ['ettm1']:
        t=pd.to_datetime(pd.read_csv(ROOT/data[source]['path'],usecols=['date']).date);assert t.is_monotonic_increasing and not t.duplicated().any() and t.diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    audit=[]
    origins=np.load(CACHE/'data/electricity/E_DISCOVERY_inputs.npz')['origins']
    d,a=make_packets('electricity_transfer',df,[int(x) for x in panel['selected_columns']],24,data['electricity']['bounds'],{'E_DISCOVERY':origins});d.update(path=data['electricity']['path'],sha256=data['electricity']['sha256'],evaluation_role=panel['panel_role']);data['electricity_transfer']=d;audit+=a
    receipts=read(OUT/'official_source_receipts.json')['zhouhaoyi/ETDataset'];p=ROOT/'data/raw/ETTm2.csv';assert sha(p)==receipts['ETTm2_official_sha256']
    df=pd.read_csv(p);times=pd.to_datetime(df.pop('date'));assert times.is_monotonic_increasing and not times.duplicated().any() and times.diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    n=len(df);bounds=[0,int(.6*n),int(.8*n),n];columns=['HUFL','HULL','MUFL','MULL'];assert np.isfinite(df[columns].to_numpy()).all()
    origins={}
    for k,(role,count) in enumerate([('TRAIN',256),('V_SELECT',64),('E_DISCOVERY',128)]):
        lo,hi=bounds[k:k+2];legal=np.arange(max(512,lo),hi-64+1);origins[role]=select_days_reference(legal,96,count,85700+k)
        assert np.ptp(np.bincount(origins[role]%96,minlength=96))<=1
    d,a=make_packets('ettm2',df,columns,96,bounds,origins,times);d.update(path=str(p.relative_to(ROOT)),sha256=sha(p),evaluation_role='previously performance-exposed benchmark family, different site procedure replication; not independent test');data['ettm2']=d;audit+=a
    save(OUT/'ETTM2_DATA_RECEIPT.json',dict(path=str(p.relative_to(ROOT)),sha256=sha(p),official_commit=receipts['commit'],official_bytes_equal_local=True,downloaded_new_copy=False,rows=n,columns=columns,step_minutes=15,first=str(times.iloc[0]),last=str(times.iloc[-1]),prior_performance_exposed=True,source_key='ettm2',phase_seeds=[85700,85701,85702]))
    save(OUT/'DATA_MANIFEST.json',data)
    for source in ['electricity','ettm1']:
        old=read(OLD/'origin_audit.json')[source]
        for role,v in old.items():audit.append(dict(source=source,role=role,days=v['distinct_days'],count=v['required_days'],full_days=v['full_eligible_days'],phase_histogram=json.dumps(v['phase_histogram']),date_deciles=json.dumps(v['date_decile_counts']),week_blocks=v['distinct_weeks'],first_origin=v['first_origin'],last_origin=v['last_origin'],first_timestamp='',last_timestamp='',unique_target_timestamps=v['unique_target_slots'],total_target_timestamps=v['total_target_slots'],max_overlap=max(map(int,v['target_overlap_histogram'])),missing_input_values=0,missing_target_values=0))
    csvwrite(OUT/'ORIGIN_AUDIT.csv',audit)
    # Reuse the exact matched generator implementation for ETTm2 only via module-local paths.
    from experiments.outlier_signal_followup_v2_20260917 import prepare as gen
    # Its audit is replaced only in memory, not in historical source; metadata contract is not reused as current contract.
    temp=CACHE/'generator_receipt';temp.mkdir(exist_ok=True)
    gen.OUT=temp;gen.CACHE=CACHE;gen.SOURCES=['ettm2'];gen.audit=lambda:None
    oldcond=CACHE/'empty_old_conditions';(oldcond/'conditions/ettm2').mkdir(parents=True,exist_ok=True);gen.OLDC=oldcond
    gen.prepare()
    for role in ['V_SELECT','E_DISCOVERY']:eval_conditions('ettm2',role)
    eval_conditions('electricity_transfer','E_DISCOVERY')
    for source in data:shape_packet(source)
    hashes={}
    for base in [CACHE/'data',CACHE/'conditions',CACHE/'shapes']:
        for p in base.rglob('*'):
            if p.is_file():hashes[str(p.relative_to(ROOT))]=sha(p)
    # rglob does not traverse symlink directories; historical condition hashes separately preserved in SOURCE_MANIFEST.
    save(OUT/'DATA_READY.json',dict(hashes=hashes,original_sources_exact_reuse=True,ettm2_generator='v2 exact source code with only source key/path changed',new_future_labels_separate=True,exposure='exploratory',shape_reference='8 new forms + matching already-defined SHIFT8 reference, no new training form'))
    print('DATA_READY',panel['selected_columns'],flush=True)

if __name__=='__main__':setup();prepare()
