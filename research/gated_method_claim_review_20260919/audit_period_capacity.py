"""Read timestamp columns only; do not choose origins or inspect ND values."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    receipt_path=ROOT/'results/c3_identifiability_temporal_20260918/DOWNLOAD_RECEIPT.json'
    manifest_path=ROOT/'results/c3_identifiability_temporal_20260918/DATA_MANIFEST.json'
    receipt=json.loads(receipt_path.read_text());raw=ROOT/receipt['path'];assert sha(raw)==receipt['sha256']
    manifest=json.loads(manifest_path.read_text())['neso_2026_h1']
    assert manifest['evaluation_interval']=='2026-01-01 inclusive to 2026-07-01 exclusive UTC'
    f=pd.read_csv(raw,usecols=['SETTLEMENT_DATE','SETTLEMENT_PERIOD'])
    assert not f.duplicated().any()
    lengths={}
    for day,g in f.groupby('SETTLEMENT_DATE'):
        t=pd.Timestamp(day,tz='Europe/London')
        n=int(((t+pd.DateOffset(days=1)).tz_convert('UTC')-t.tz_convert('UTC')).total_seconds()/1800)
        assert sorted(g.SETTLEMENT_PERIOD.tolist())==list(range(1,n+1))
        lengths[str(n)]=lengths.get(str(n),0)+1
    date=pd.to_datetime(f.SETTLEMENT_DATE)
    time=date.dt.tz_localize('Europe/London').dt.tz_convert('UTC')+pd.to_timedelta((f.SETTLEMENT_PERIOD-1)*30,unit='min')
    assert not time.duplicated().any()
    count=pd.Series(1,index=pd.DatetimeIndex(time)).resample('h').count();hours=count[count==2].index
    start=pd.Timestamp('2026-07-01',tz='UTC');valid=[]
    for t in hours[hours>=start]:
        if len(pd.date_range(t-pd.Timedelta(hours=512),t+pd.Timedelta(hours=63),freq='h').difference(hours))==0:valid.append(t)
    strict=[t for t in valid if t-pd.Timedelta(hours=512)>=start]
    def summary(times):
        return dict(candidate_hours=len(times),distinct_UTC_origin_days=len({t.date() for t in times}),first_origin=str(min(times)),last_origin=str(max(times)),last_inclusive_target=str(max(times)+pd.Timedelta(hours=63)))
    # Separate existing origin metadata confirms its complete targets end before July.
    old=ROOT/'.cache/c3_identifiability_temporal_20260918/data/neso_2026_h1/E_DISCOVERY_inputs.npz'
    assert sha(old)==manifest['packet_hashes']['E_DISCOVERY'][str(old.relative_to(ROOT))]
    with np.load(old) as data:origins=data['origins']
    zero=pd.Timestamp('2025-01-01',tz='UTC')
    previous_last_target=zero+pd.Timedelta(hours=int(origins.max())+63)
    assert previous_last_target<start
    # Restricted records search; not a proof of all historical/global non-exposure.
    search=subprocess.run(['rg','-l','-i','neso.?2026|neso.{0,40}2026','results','experiments','--glob','*MANIFEST*.json','--glob','PROTOCOL.md','--glob','prepare.py','--glob','*PLAN*.json','--glob','*EXPOSURE*.json','--glob','*SEAL*.json'],cwd=ROOT,text=True,capture_output=True,check=True)
    result=dict(status='METADATA_CAPACITY_VERIFIED',raw_sha256=sha(raw),read_columns=list(f.columns),ND_values_inspected=False,first_settlement_day=str(date.min().date()),last_settlement_day=str(date.max().date()),settlement_day_lengths=lengths,complete_hour_rows=len(hours),previous_evaluation_last_inclusive_target=str(previous_last_target),candidate_start=str(start),target_only_new=summary(valid),context_and_target_new=summary(strict),can_supply_64_distinct_days=False,can_supply_128_distinct_days=False,scored_origins=0,new_model_inferences=0,optimizer_updates=0,origin_selection_performed=False,scope='Metadata upper bound only: target validity/ND missingness and global or unlogged prior exposure not assessed; no independent-source claim.',prior_records_search=search.stdout.splitlines(),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),receipt_path,manifest_path,old]})
    assert result['target_only_new']['distinct_UTC_origin_days']==56
    assert result['context_and_target_new']['distinct_UTC_origin_days']==35
    (OUT/'PERIOD_CAPACITY.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ['status','last_settlement_day','target_only_new','context_and_target_new']},indent=2))
if __name__=='__main__':main()
