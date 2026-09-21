"""Audit official raw system load and past-only strata, without forecast scoring."""
from pathlib import Path
import csv, hashlib, json
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
NAME='mag_real_levelshift_confirmation_v1_20260921'
OUT=ROOT/'results'/NAME/'isone_acquisition'
CACHE=ROOT/'.cache'/NAME

def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def shift_score(context,sigma):
    assert len(context)==512
    reference=context[-160:-32];recent=context[-32:]
    center=np.median(reference)
    scale=max(1.4826*np.median(np.abs(reference-center)),.1*sigma)
    return float(abs(np.median(recent)-center)/scale)

def main():
    receipts=json.loads((OUT/'DOWNLOAD_RECEIPTS.json').read_text())
    expected=[(date(2026,1,1)+timedelta(days=i)).strftime('%Y%m%d') for i in range(243)]
    assert sorted(r['day'] for r in receipts)==expected
    rows=[];counts={};schemas=set();dst_mismatch=0
    for receipt in receipts:
        raw=(ROOT/receipt['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==receipt['sha256']
        packet=json.loads(raw)['HourlySystemLoads']['HourlySystemLoad']
        counts[receipt['day']]=len(packet)
        for row in packet:
            schemas.add(tuple(sorted(row)))
            assert row['Location']=={'@LocId':'32','$':'NEPOOL AREA'}
            parsed=datetime.fromisoformat(row['BeginDate']);local=parsed.astimezone(ZoneInfo('America/New_York'))
            dst_mismatch+=int(local.replace(tzinfo=None)!=parsed.replace(tzinfo=None) or local.utcoffset()!=parsed.utcoffset())
            assert parsed.strftime('%Y%m%d')==receipt['day']
            rows.append({'timestamp_utc':parsed.astimezone(timezone.utc),'load':row['Load'],'native_load':row['NativeLoad'],'ard_demand':row['ArdDemand']})
    assert dst_mismatch==0
    d=pd.DataFrame(rows).sort_values('timestamp_utc')
    duplicates=int(d.timestamp_utc.duplicated().sum());assert duplicates==0
    assert all(t.minute==0 and t.second==0 for t in d.timestamp_utc)
    d=d.set_index('timestamp_utc')
    full=pd.date_range('2026-01-01','2026-09-01',freq='h',inclusive='left',tz='UTC')
    series=d['load'].reindex(full).astype(float)
    values=series.to_numpy();finite=np.isfinite(values)
    trainmask=(full<pd.Timestamp('2026-05-01',tz='UTC'))&finite
    sigma=float(np.std(values[trainmask],ddof=0));assert sigma>0
    windows=[('TRAIN','2026-01-01','2026-05-01'),('V_SELECT','2026-05-01','2026-07-01'),('E_CONFIRM','2026-07-01','2026-09-01')]
    origins=[];strata=[]
    for split,start,end in windows:
        lo=pd.Timestamp(start,tz='UTC');hi=pd.Timestamp(end,tz='UTC')
        for i,origin in enumerate(full):
            if not lo<=origin<hi:continue
            reason='ELIGIBLE'
            if i<512:reason='INSUFFICIENT_PAST'
            elif origin+pd.Timedelta(hours=64)>hi:reason='TARGET_OUTSIDE_SPLIT'
            elif not finite[i-512:i+64].all():reason='NONFINITE_CONTEXT_OR_TARGET'
            origins.append({'split':split,'origin_utc':origin.isoformat(),'index':i,'eligible':reason=='ELIGIBLE','reason':reason})
            if split=='E_CONFIRM' and reason=='ELIGIBLE':
                score=shift_score(values[i-512:i],sigma)
                binname='S<1' if score<1 else '1<=S<2' if score<2 else '2<=S<3' if score<3 else 'S>=3'
                strata.append({'origin_utc':origin.isoformat(),'shift_score':score,'bin':binname,'utc_date':origin.date().isoformat()})
    pd.DataFrame(origins).to_csv(OUT/'ORIGIN_AUDIT.csv',index=False)
    st=pd.DataFrame(strata);st.to_csv(OUT/'SHIFT_STRATA.csv',index=False)
    bins={b:{'origins':int((st.bin==b).sum()),'distinct_utc_days':int(st.loc[st.bin==b,'utc_date'].nunique())} for b in ['S<1','1<=S<2','2<=S<3','S>=3']}
    primary=bins['S>=3'];coverage=primary['origins']>=50 and primary['distinct_utc_days']>=14
    # Independently recompute population std with Python scalars.
    numbers=[float(v) for v in values[trainmask]];mean=sum(numbers)/len(numbers)
    scalar_sigma=(sum((v-mean)**2 for v in numbers)/len(numbers))**.5
    assert abs(sigma-scalar_sigma)<1e-9
    output=CACHE/'isone_hourly_utc.csv';pd.DataFrame({'timestamp_utc':full,'load':values}).to_csv(output,index=False)
    audit={'status':'ACQUIRED_AND_DATA_AUDITED_NOT_TRAINED','provider':'ISO-NE','report':'Hourly system actual load','location_id':32,'location':'NEPOOL AREA','target_field':'Load','unit':'MW hourly system load','raw_rows':len(d),'raw_daily_files':len(receipts),'schema':[list(s) for s in schemas],'hourly_utc_grid_rows':len(full),'finite_hours':int(finite.sum()),'missing_hours':int((~finite).sum()),'missing_timestamps_utc':[t.isoformat() for t in full[~finite]],'duplicate_utc_timestamps':duplicates,'timezone_roundtrip_mismatches':dst_mismatch,'non_24_hour_days':{day:n for day,n in counts.items() if n!=24},'native_plus_ard_max_abs_difference':float(np.max(np.abs(d.load-d.native_load-d.ard_demand))),'sigma_train_population':sigma,'sigma_train_finite_hours':len(numbers),'eligible_origins':{s:sum(r['split']==s and r['eligible'] for r in origins) for s,_,_ in windows},'strata':bins,'coverage_status':'SUFFICIENT_REAL_SHIFT_COVERAGE' if coverage else 'INSUFFICIENT_REAL_SHIFT_COVERAGE','scored_predictions':0,'fits':0,'synthetic_E_transforms':0,'interpolation':False,'normalized_series':{'path':str(output.relative_to(ROOT)),'sha256':hashlib.sha256(output.read_bytes()).hexdigest()},'source_equivalence_basis':'Official business documentation maps hourlysysload to entire control-area hourly system load; location32 is system total. Website CSV byte/value parity not established because CSV access remains blocked.'}
    write('DATA_AUDIT.json',audit)
    write('VERIFICATION.json',{'status':'DATA_CHECKS_PASSED_NOT_MODEL_VERIFIED','all_daily_hashes_verified':True,'expected_243_days_complete':True,'location_identity_verified':True,'timezone_offsets_verified':True,'independent_population_sigma_verified':True,'data_score_access':'quality/coverage only; no model performance','auth_material_published':False,'main_updates':0,'smoke_updates':0})
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
