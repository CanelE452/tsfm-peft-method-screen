import subprocess,re,urllib.request
import pandas as pd
from .common import *
from experiments.additive_persistence_validation_v1_20260917 import prepare as gen
URL='https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource/8a4a771c-3929-4e56-93ad-cdf13219dea5/download/demanddataupdate_2026.csv'

def hourly(path,year,end):
    f=pd.read_csv(path);d=pd.to_datetime(f.SETTLEMENT_DATE);f=f.loc[d<pd.Timestamp(end)].copy();d=pd.to_datetime(f.SETTLEMENT_DATE)
    assert not f[['SETTLEMENT_DATE','SETTLEMENT_PERIOD']].duplicated().any()
    for day,g in f.groupby('SETTLEMENT_DATE'):
        local=pd.Timestamp(day,tz='Europe/London');n=int(((local+pd.DateOffset(days=1)).tz_convert('UTC')-local.tz_convert('UTC')).total_seconds()/1800)
        assert sorted(g.SETTLEMENT_PERIOD.tolist())==list(range(1,n+1)),day
    utc=d.dt.tz_localize('Europe/London').dt.tz_convert('UTC')+pd.to_timedelta((f.SETTLEMENT_PERIOD-1)*30,unit='min')
    series=pd.Series(f.ND.to_numpy(float),index=pd.DatetimeIndex(utc)).sort_index()
    # end local midnight may be UTC-1h; require all requested UTC hours separately.
    assert not series.index.duplicated().any() and np.isfinite(series).all()
    return series

def prepare():
    OUT.mkdir(exist_ok=True,parents=True);CACHE.mkdir(exist_ok=True,parents=True);ready.check_seal()
    if (OUT/'SEAL.json').exists():check_seal();return
    if not (OUT/'EXPOSURE_AUDIT.json').exists():
        pat=re.compile(r'neso_2026|demanddataupdate_2026|NESO.{0,30}2026.{0,20}(H1|상반기)',re.I);files=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines();hits=[];scanned=0
        for name in files:
            p=ROOT/name
            if NAME in name or p.suffix not in ['.py','.md','.json','.csv','.txt','.yml','.yaml'] or p.stat().st_size>5*2**20:continue
            scanned+=1
            if pat.search(p.read_text(errors='replace')):hits.append(name)
        save(OUT/'EXPOSURE_AUDIT.json',dict(initial_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files_scanned=scanned,hits=hits,pattern=pat.pattern,scope='tracked local project text; unlogged/global use unknown',same_provider_as_2025=True,independent_source=False,actual_event_labels=False))
        assert not hits,('PRIOR_EXPOSURE_REQUIRES_REVIEW',hits)
    raw=CACHE/'raw';raw.mkdir(exist_ok=True);p=raw/'neso_2026.csv'
    if not p.exists():
        with urllib.request.urlopen(URL,timeout=60) as r:b=r.read();headers=dict(r.headers)
        p.write_bytes(b);save(OUT/'DOWNLOAD_RECEIPT.json',dict(url=URL,path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=len(b),headers=headers,at=time.time(),official_page='https://www.neso.energy/data-portal/historic-demand-data',provider_caveat='21 days arrears; retrospective corrections; reported quality issues'))
    assert sha(p)==read(OUT/'DOWNLOAD_RECEIPT.json')['sha256']
    a=hourly(ext.CACHE/'raw/neso_2025.csv',2025,'2026-01-01');b=hourly(p,2026,'2026-07-02');series=pd.concat([a,b]);series=series.loc[(series.index>=pd.Timestamp('2025-01-01',tz='UTC'))&(series.index<pd.Timestamp('2026-07-01',tz='UTC'))]
    expected=pd.date_range('2025-01-01','2026-07-01',freq='30min',inclusive='left',tz='UTC');assert series.index.equals(expected)
    assert series.resample('h').count().eq(2).all();h=series.resample('h').mean();assert len(h)==13104
    origins=select_days_reference(np.arange(8760,len(h)-64+1),24,128,90301);assert len(set(origins//24))==128 and np.ptp(np.bincount(origins%24,minlength=24))<=1
    prev=gen.OUT,gen.CACHE
    try:
        gen.OUT,gen.CACHE=OUT,CACHE
        d,audit=gen.make_packets(NEW,pd.DataFrame({'ND':h.to_numpy()}),['ND'],24,[0,4344,8760,len(h)],{'E_DISCOVERY':origins},pd.Series(h.index))
        gen.eval_conditions(NEW,'E_DISCOVERY');gen.shape_packet(NEW)
    finally:gen.OUT,gen.CACHE=prev
    np.testing.assert_array_equal(d['sigma_train_population'],read(ext.OUT/'DATA_MANIFEST.json')['neso_2025']['sigma_train_population'])
    assert audit[0]['missing_target_values']==0
    d.update(evaluation_interval='2026-01-01 inclusive to 2026-07-01 exclusive UTC',sigma_period='2025-01-01 to 2025-07-01 UTC; unchanged',prior_context='2025 late ND allowed as observed past only',independent_source=False,provider='NESO retrospective ND outturn')
    save(OUT/'DATA_MANIFEST.json',{NEW:d});csvwrite(OUT/'ORIGIN_AUDIT.csv',audit)
    rows=read(PRIOR/'MODEL_SELECTION.json');plan=[]
    for kind in ['standard','shape']:
        for arm,trained,gate in [(a,a,a) for a in ARMS]+[(a,t,g) for a,(t,g) in SWAPS.items()]:
            for seed in [81551,81552,81553]:
                r=next(r for r in rows if (r['source'],r['arm'],r['seed'])==('electricity',trained,seed));assert sha(ROOT/r['checkpoint'])==r['sha256']
                plan.append(dict(panel=NEW,kind=kind,arm=arm,trained_arm=trained,gate_arm=gate,seed=seed,row=r))
        for arm in ['F0','PERSISTENCE','SEASONAL']:plan.append(dict(panel=NEW,kind=kind,arm=arm,trained_arm=arm,gate_arm=None,seed=0,row=None))
    assert len(plan)==60
    for j in plan:j['id']='__'.join(str(j[k]) for k in ['panel','kind','arm','seed'])
    save(OUT/'PLAN.json',plan)
    hashes=dict(read(ready.OUT/'SEAL.json')['hashes'])
    for base in [PRIOR,ext.OUT,ready.OUT,CACHE,EXP]:
        for f in base.rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts:hashes[str(f.relative_to(ROOT))]=sha(f)
    for name in ['PROTOCOL.md','EXPOSURE_AUDIT.json','DOWNLOAD_RECEIPT.json','DATA_MANIFEST.json','ORIGIN_AUDIT.csv','PLAN.json','CPU_CHECKS.json']:hashes[str((OUT/name).relative_to(ROOT))]=sha(OUT/name)
    for j in plan:
        if j['row']:hashes[j['row']['checkpoint']]=j['row']['sha256']
    save(OUT/'SEAL.json',dict(at=time.time(),hashes=hashes,logical_views_cap=60,main_fits_cap=0,optimizer_updates_cap=0,selection_on_new_labels=False))
    check_seal();print('SEALED 60 views / 0 updates',audit,flush=True)
if __name__=='__main__':prepare()
