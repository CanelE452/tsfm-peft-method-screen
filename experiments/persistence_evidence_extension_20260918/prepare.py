import subprocess
import pandas as pd
from .common import *
from experiments.additive_persistence_validation_v1_20260917 import prepare as previous_prepare

def prepare():
    OUT.mkdir(exist_ok=True,parents=True);CACHE.mkdir(exist_ok=True,parents=True);old.check_seal()
    if (OUT/'DATA_READY.json').exists():
        for p,h in read(OUT/'DATA_READY.json')['hashes'].items():assert sha(ROOT/p)==h
        return
    raw=CACHE/'raw';f=pd.read_csv(raw/'neso_2025.csv');assert len(f)==17520
    assert not f[['SETTLEMENT_DATE','SETTLEMENT_PERIOD']].duplicated().any()
    dates=pd.to_datetime(f.SETTLEMENT_DATE);days=pd.date_range('2025-01-01','2025-12-31')
    assert set(dates.unique())==set(days.to_numpy());counts=f.groupby('SETTLEMENT_DATE').size()
    utc=dates.dt.tz_localize('Europe/London').dt.tz_convert('UTC')+pd.to_timedelta((f.SETTLEMENT_PERIOD-1)*30,unit='min')
    for day,group in f.groupby('SETTLEMENT_DATE'):
        local=pd.Timestamp(day,tz='Europe/London');n=int(((local+pd.DateOffset(days=1)).tz_convert('UTC')-local.tz_convert('UTC')).total_seconds()/1800)
        assert sorted(group.SETTLEMENT_PERIOD.tolist())==list(range(1,n+1)),day
    series=pd.Series(f.ND.to_numpy(float),index=pd.DatetimeIndex(utc)).sort_index()
    expected=pd.date_range('2025-01-01','2026-01-01',freq='30min',inclusive='left',tz='UTC')
    assert series.index.equals(expected) and np.isfinite(series).all()
    # ND is power in MW: hourly mean, not sum; all half-hours in every bin required.
    assert (series.resample('h').count()==2).all();hourly=series.resample('h').mean();assert len(hourly)==8760
    a=hourly.to_numpy();split=int(hourly.index.searchsorted(pd.Timestamp('2025-07-01',tz='UTC')));assert split==4344
    origins=select_days_reference(np.arange(split,len(a)-64+1),24,128,88301)
    assert len(set(origins//24))==128 and np.ptp(np.bincount(origins%24,minlength=24))<=1
    # Reuse exact original transformation functions, changing output paths only in memory.
    paths=(previous_prepare.OUT,previous_prepare.CACHE)
    try:
        previous_prepare.OUT=OUT;previous_prepare.CACHE=CACHE
        d,audit=previous_prepare.make_packets('neso_2025',pd.DataFrame({'ND':a}),['ND'],24,[0,split,split,len(a)],{'E_DISCOVERY':origins},pd.Series(hourly.index))
        previous_prepare.eval_conditions('neso_2025','E_DISCOVERY');previous_prepare.shape_packet('neso_2025')
    finally:previous_prepare.OUT,previous_prepare.CACHE=paths
    d.update(path=str((raw/'neso_2025.csv').relative_to(ROOT)),sha256=sha(raw/'neso_2025.csv'),time_aggregation='UTC hourly arithmetic mean of both 30-minute MW values',normalization='Jan1-Jun30 2025 only; no target fitting or V selection',role='external-source frozen Electricity model transfer; not NESO local adaptation',revision_caveat='provider-revised historical outturn, not archived real-time vintage; provider notes quality issues')
    manifest=read(PRIOR/'DATA_MANIFEST.json');manifest['neso_2025']=d;save(OUT/'DATA_MANIFEST.json',manifest);csvwrite(OUT/'ORIGIN_AUDIT.csv',audit)
    tree=read(raw/'bolt_tree_2024.json');h=next(x['lfs']['oid'] for x in tree if x['path']=='model.safetensors')
    snapshots=read(PRIOR/'SOURCE_MANIFEST.json')['immutable_hashes'];weights=[p for p in snapshots if 'chronos-bolt-small' in p and p.endswith('model.safetensors')];assert len(weights)==1 and sha(ROOT/weights[0])==h
    configs=[p for p in snapshots if 'chronos-bolt-small' in p and p.endswith('config.json')];assert sha(ROOT/configs[0])==sha(raw/'bolt_config_2024.json')
    commits=read(raw/'bolt_commits.json');c=next(x for x in commits if x['id']=='2680039ae7e6bede10e530821b4bb4c7f1e4f517');assert c['date'].startswith('2024-')
    # Search only pre-existing tracked project text. No claim about unlogged human/external use.
    names=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines();scanned=[];hits=[]
    import re
    pat=re.compile(r'neso|demanddata_2025|historic.demand.data',re.I)
    for name in names:
        p=ROOT/name
        if p.suffix in ['.py','.md','.json','.txt','.yaml','.yml','.csv'] and p.stat().st_size<5*2**20:
            scanned.append(name)
            if pat.search(p.read_text(errors='replace')):hits.append(name)
    save(OUT/'EXTERNAL_SOURCE_AUDIT.json',dict(download_receipts='DOWNLOAD_RECEIPTS.json',days=365,half_hour_rows=17520,hour_rows=8760,day_lengths={str(k):int(v) for k,v in counts.value_counts().items()},no_missing_ND=True,no_duplicate_times=True,old_model_weight_sha256=h,weight_bytes_equal_2024=True,config_bytes_equal_2024=True,weight_public_commit=c['id'],weight_public_date=c['date'],pretraining_2025_observation_overlap='EXCLUDED_BY_IDENTICAL_2024_WEIGHT_BYTES',earlier_same_provider_data_overlap='UNKNOWN; not claimed absent',project_text_files_scanned=len(scanned),project_prior_keyword_hits=hits,global_or_unlogged_exposure='NOT_AUDITED',new_source_not_newly_collected_data=True,independent_industrial_events=False,one_aggregate_series_not_16_independent_meters=True))
    save(OUT/'PRIOR_EXPOSURE_SEARCH.json',dict(pattern=pat.pattern,files=scanned,hits=hits))
    hashes={str(p.relative_to(ROOT)):sha(p) for base in [CACHE/'raw',CACHE/'data',CACHE/'conditions',CACHE/'shapes'] for p in base.rglob('*') if p.is_file()}
    save(OUT/'DATA_READY.json',dict(at=time.time(),hashes=hashes,labels_separate=True,optimizer_updates=0))

def plan():
    rows=read(PRIOR/'MODEL_SELECTION.json');jobs=[]
    def add(panel,kind,arm,trained,gate):
        src='electricity' if panel in ['electricity','electricity_transfer','neso_2025'] else panel
        for seed in [81551,81552,81553]:
            row=next(r for r in rows if (r['source'],r['arm'],r['seed'])==(src,trained,seed))
            jobs.append(dict(id=f'{panel}__{kind}__{arm}__{seed}',panel=panel,kind=kind,arm=arm,trained_arm=trained,gate_arm=gate,seed=seed,row=row))
    for panel in ['electricity','electricity_transfer','ettm1']:
        for arm in old.CONTROLS:add(panel,'shape',arm,arm,arm)
        for kind in ['standard','shape']:
            for arm,(trained,gate) in SWAPS.items():add(panel,kind,arm,trained,gate)
    for arm in ARMS:
        for kind in ['standard','shape']:add('neso_2025',kind,arm,arm,arm)
    assert len(jobs)==105 and len({j['id'] for j in jobs})==105
    return jobs

def seal():
    if (OUT/'SEAL.json').exists():check_seal();return
    jobs=plan();save(OUT/'INFERENCE_PLAN.json',jobs)
    hashes=dict(read(OUT/'DATA_READY.json')['hashes'])
    for base in [PRIOR,ROOT/'experiments/additive_persistence_validation_v1_20260917']:
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:hashes[str(p.relative_to(ROOT))]=sha(p)
    # All older model/import dependencies and all selected model snapshots are immutable.
    hashes.update(read(PRIOR/'MASTER_SEAL.json')['source_hashes']);hashes.update(read(PRIOR/'SOURCE_MANIFEST.json')['immutable_hashes'])
    for row in read(PRIOR/'MODEL_SELECTION.json'):hashes[row['checkpoint']]=row['sha256']
    for p in EXP.glob('*.py'):
        if p.name!='report.py':hashes[str(p.relative_to(ROOT))]=sha(p)
    for name in ['PROTOCOL.md','DATA_MANIFEST.json','EXTERNAL_SOURCE_AUDIT.json','ORIGIN_AUDIT.csv','INFERENCE_PLAN.json','LITERATURE_UPDATE.md','DOWNLOAD_RECEIPTS.json']:
        hashes[str((OUT/name).relative_to(ROOT))]=sha(OUT/name)
    for p,h in hashes.items():assert sha(ROOT/p)==h,p
    save(OUT/'SEAL.json',dict(at=time.time(),initial_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),hashes=hashes,main_fit_cap=0,optimizer_update_cap=0,new_prediction_view_cap=111,primary='NESO SHIFT8 C3/C0,C3/C2,C3/M_RECENCY jointly; prior-panel shapes and swaps explanatory',no_new_E_scoring=True,automatic_successor=False))
