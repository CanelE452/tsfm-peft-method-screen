"""Pinned single-target data feasibility audit; no training or forecast scoring."""
import hashlib
import json
import platform
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = 'covariate_availability_audit_20260916'
OUT = ROOT / 'research' / RUN
CACHE = ROOT / '.cache' / RUN
HF = 'OpenSTEF/liander2024-energy-forecasting-benchmark'
REV = 'dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044'
GH = 'fdc58170714e7a79ad60a286c82ca069d6f6c96a'
TARGET = 'mv_feeder/OS Gorredijk.parquet'

def sha(b):
    return hashlib.sha256(b).hexdigest()

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def fetch(url, path):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            b = r.read(32*1024**2+1)
        assert len(b) <= 32*1024**2
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b)
    return path.read_bytes()

def select(df, origin):
    return df[df.available_at <= origin].sort_values(['timestamp','available_at']).drop_duplicates('timestamp', keep='last').set_index('timestamp').sort_index()

def main():
    OUT.mkdir(exist_ok=True, parents=True)
    CACHE.mkdir(exist_ok=True, parents=True)
    tracked = subprocess.check_output(['git','ls-files','results','research'], cwd=ROOT,text=True).splitlines()
    before = {p:sha((ROOT/p).read_bytes()) for p in tracked if not p.startswith(f'research/{RUN}/')}
    scope_sha = sha((OUT/'SCOPE.md').read_bytes())
    files = ['README.md','liander2024_targets.yaml']+[f'{c}/{TARGET}' for c in ['load_measurements','weather_measurements','weather_forecasts_versioned']]
    manifest=[]
    for p in files:
        url = f'https://huggingface.co/datasets/{HF}/resolve/{REV}/'+urllib.parse.quote(p)
        b = fetch(url,CACHE/p)
        manifest.append(dict(path=p,url=url,bytes=len(b),sha256=sha(b)))
        if p.endswith('.parquet'):
            directory = p.rsplit('/',1)[0]
            metaurl=f'https://huggingface.co/api/datasets/{HF}/tree/{REV}/'+urllib.parse.quote(directory)
            meta=json.loads(fetch(metaurl,CACHE/(directory.replace('/','_')+'_tree.json')))
            entry=next(x for x in meta if x['path']==p)
            assert entry['size']==len(b)
            assert entry['lfs']['oid']==sha(b)
            manifest[-1]['lfs_sha256_verified']=True
    assert sum(x['bytes'] for x in manifest) <= 64*1024**2
    sourcepaths=['packages/openstef-beam/src/openstef_beam/benchmarking/benchmarks/liander2024.py','packages/openstef-core/src/openstef_core/datasets/versioned_timeseries_dataset.py','examples/tutorials/datasets.py']
    for p in sourcepaths:
        url=f'https://raw.githubusercontent.com/OpenSTEF/openstef/{GH}/{p}'
        b=fetch(url,CACHE/'sources'/Path(p).name)
        manifest.append(dict(path=p,url=url,bytes=len(b),sha256=sha(b)))
    dump('download_manifest.json',manifest)
    summaries={}; tables={}
    for component in ['load_measurements','weather_measurements','weather_forecasts_versioned']:
        frame=pd.read_parquet(CACHE/component/TARGET)
        if 'timestamp' not in frame.columns:
            frame=frame.reset_index()
        assert 'timestamp' in frame.columns, frame.columns
        for col in [c for c in ['timestamp','available_at'] if c in frame.columns]:
            frame[col]=pd.to_datetime(frame[col],utc=True)
            assert frame[col].notna().all()
        if 'available_at' not in frame.columns:
            assert component == 'weather_measurements'
            summaries[component]=dict(rows=len(frame),columns=list(frame.columns),timestamp_min=str(frame.timestamp.min()),timestamp_max=str(frame.timestamp.max()),available_at_present=False,causal_error_calibration='NOT_VERIFIED_NO_RELEASE_TIMESTAMPS')
            tables[component]=frame
            continue
        dup=int(frame.duplicated(['timestamp','available_at']).sum())
        assert dup==0, f'duplicate keys: {component}'
        lead=(frame.timestamp-frame.available_at).dt.total_seconds()/3600
        summaries[component]=dict(rows=len(frame),columns=list(frame.columns),dtypes={k:str(v) for k,v in frame.dtypes.items()},timestamp_min=str(frame.timestamp.min()),timestamp_max=str(frame.timestamp.max()),available_min=str(frame.available_at.min()),available_max=str(frame.available_at.max()),duplicate_keys=dup,lead_hours_min=float(lead.min()),lead_hours_max=float(lead.max()),lead_hours_distinct=sorted(map(float,lead.unique()))[:200],distinct_valid_times=int(frame.timestamp.nunique()),distinct_available_times=int(frame.available_at.nunique()))
        tables[component]=frame
    f=tables['weather_forecasts_versioned']
    features=[x for x in f if x not in ['timestamp','available_at']]
    origins=pd.date_range('2024-02-01 08:00', '2024-12-29 08:00',freq='D',tz='UTC')
    rows=[]
    for origin in origins:
        idx=pd.date_range(origin, periods=192,freq='15min')
        window=f[f.timestamp.isin(idx)]
        selected=select(window,origin).reindex(idx)
        present=selected.available_at.notna()
        finite=np.isfinite(selected[features].to_numpy(dtype=float)).all(axis=1)
        eligible=window[window.available_at<=origin].groupby('timestamp').size().reindex(idx,fill_value=0)
        rows.append(dict(origin=str(origin),requested=192,present=int(present.sum()),all_features_finite=int(finite.sum()),eligible_vintages_min=int(eligible.min()),eligible_vintages_max=int(eligible.max()),future_availability_violations=int((selected.available_at>origin).sum())))
    pd.DataFrame(rows).to_csv(OUT/'availability_by_origin.csv',index=False)
    checks=[]
    for j in np.linspace(0,len(origins)-1,12,dtype=int):
        origin=origins[j]; idx=pd.date_range(origin,periods=192,freq='15min')
        window=f[f.timestamp.isin(idx)].copy(); chosen=select(window,origin).reindex(idx)
        # Independent exact-timestamp row search, with argmax availability.
        manual=[]
        for valid in idx:
            r=window.loc[(window.timestamp==valid)&(window.available_at<=origin)]
            if len(r): manual.append(r.iloc[int(r.available_at.to_numpy().argmax())][features].to_numpy(dtype=float))
            else: manual.append(np.full(len(features),np.nan))
        np.testing.assert_array_equal(chosen[features].to_numpy(dtype=float),np.asarray(manual))
        future=window.available_at>origin
        window.loc[future,features]=123456.0
        poisoned=select(window,origin).reindex(idx)
        pd.testing.assert_frame_equal(chosen,poisoned)
        checks.append(dict(origin=str(origin),manual_match=True,poison_invariant=True,poisoned_future_rows=int(future.sum())))
    tutorial_origin=pd.Timestamp('2024-12-27 08:00',tz='UTC')
    tutorial_valid=pd.Timestamp('2024-12-28 12:00',tz='UTC')
    sample=f[f.timestamp==tutorial_valid].sort_values('available_at')[['timestamp','available_at','temperature_2m']].copy()
    sample['eligible_at_2024_12_27_08_utc']=sample.available_at<=tutorial_origin
    sample.to_csv(OUT/'tutorial_availability_example.csv',index=False)
    for p,h in before.items(): assert sha((ROOT/p).read_bytes())==h,p
    assert sha((OUT/'SCOPE.md').read_bytes())==scope_sha
    dump('historical_hashes.json',before)
    dump('schema_summary.json',summaries)
    dump('verification.json',dict(status='COMPLETE',fits=0,optimizer_updates=0,model_forwards=0,gpu_used=False,score_computations=0,python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),scope_sha256=scope_sha,script_sha256=sha(Path(__file__).read_bytes()),hf_revision=REV,github_source_revision=GH,origins=len(rows),full_coverage_origins=sum(r['present']==192 for r in rows),full_finite_origins=sum(r['all_features_finite']==192 for r in rows),total_requested=len(rows)*192,total_present=sum(r['present'] for r in rows),total_finite=sum(r['all_features_finite'] for r in rows),future_violations=sum(r['future_availability_violations'] for r in rows),independent_checks=checks,historical_files_preserved=len(before),predictive_evidence='NOT_EVALUATED',novelty='NOT_EVALUATED',availability_provenance='SIMULATED_TIMESTAMPS_NOT_VERIFIED_REAL_RELEASES',errors=[{'stage':'environment_probe','detail':'pyarrow absent in .venv-channel; used existing .venv with pyarrow25.0.1; no installation and no fit attempt'}, {'stage':'schema_probe', 'detail':'First audit attempt asserted available_at for all three tables; historical weather has no available_at. Corrected inspector to report missing provenance without fabricating timestamps; no training or score attempt.'}, {'stage':'independent_checker', 'detail':'Second attempt found original dataframe index labels are non-unique, so label-based idxmax selected multiple rows. Independent checker now uses positional argmax. Production as-of selection unchanged; two audit reruns total.'}]))
    print((OUT/'verification.json').read_text())

if __name__=='__main__': main()
