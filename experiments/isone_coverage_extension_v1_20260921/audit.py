"""Raw quality and fixed E-condition prevalence only; no forecast target scoring."""
from .common import *
from datetime import date,datetime,timedelta,timezone
from zoneinfo import ZoneInfo
import math,statistics,collections
import numpy as np
import pandas as pd
from experiments.mag_real_levelshift_confirmation_v1_20260921.audit_isone import shift_score
BINS=['S<1','1<=S<2','2<=S<3','S>=3']

def classify(s):return BINS[0] if s<1 else BINS[1] if s<2 else BINS[2] if s<3 else BINS[3]

def main():
    seal=check_seal();preserved=check_parent()
    receipts=read(OUT/'DOWNLOAD_RECEIPTS.json');first=date.fromisoformat(seal['first_day']);last=date.fromisoformat(seal['last_day_inclusive'])
    expected_days=[first+timedelta(days=i) for i in range((last-first).days+1)]
    assert sorted(r['day'] for r in receipts)==[d.strftime('%Y%m%d') for d in expected_days]
    records=[];daily=[];schemas=collections.Counter();tzbad=[]
    for receipt in sorted(receipts,key=lambda r:r['day']):
        raw=(ROOT/receipt['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==receipt['sha256']
        packet=json.loads(raw)['HourlySystemLoads'].get('HourlySystemLoad',[])
        if isinstance(packet,dict):packet=[packet]
        localday=datetime.strptime(receipt['day'],'%Y%m%d').date()
        lo=datetime.combine(localday,datetime.min.time(),ZoneInfo('America/New_York')).astimezone(timezone.utc)
        hi=datetime.combine(localday+timedelta(days=1),datetime.min.time(),ZoneInfo('America/New_York')).astimezone(timezone.utc)
        expected=set(lo+timedelta(hours=i) for i in range(int((hi-lo).total_seconds()/3600)))
        actual=[];badcount=0
        for row in packet:
            schemas[tuple(sorted(row))]+=1
            assert row['Location']=={'@LocId':'32','$':'NEPOOL AREA'},'SYSTEM_ID_MISMATCH'
            parsed=datetime.fromisoformat(row['BeginDate']);assert parsed.tzinfo is not None
            back=parsed.astimezone(ZoneInfo('America/New_York'))
            if parsed.replace(tzinfo=None)!=back.replace(tzinfo=None) or parsed.utcoffset()!=back.utcoffset():tzbad.append(row['BeginDate'])
            assert parsed.date()==localday,'DAY_FILE_MISMATCH'
            utc=parsed.astimezone(timezone.utc);assert utc.minute==0 and utc.second==0 and utc.microsecond==0
            actual.append(utc)
            value=float(row['Load']) if row['Load'] is not None else float('nan');badcount+=not math.isfinite(value)
            records.append((utc,value,receipt['day'],row['BeginDate'],hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()))
        daily.append({'day':localday.isoformat(),'expected_hours':len(expected),'raw_rows':len(packet),'unique_utc_hours':len(set(actual)),'missing_hours':len(expected-set(actual)),'unexpected_hours':len(set(actual)-expected),'nonfinite_load':badcount,'duplicate_utc':len(actual)-len(set(actual))})
    pd.DataFrame(daily).to_csv(OUT/'DAILY_QUALITY.csv',index=False)
    df=pd.DataFrame(records,columns=['timestamp_utc','load','source_day','raw_timestamp','full_row_sha256']).sort_values('timestamp_utc')
    duplicated=df[df.duplicated('timestamp_utc',keep=False)]
    duplicated.to_csv(OUT/'DUPLICATE_ROWS.csv',index=False)
    quality={'daily_files':len(receipts),'raw_rows':len(df),'schema_counts':[{'columns':list(k),'rows':v} for k,v in schemas.items()],'dst_mismatches':tzbad,'duplicate_rows':len(duplicated),'daily_missing_hours':sum(d['missing_hours'] for d in daily),'daily_nonfinite_load':sum(d['nonfinite_load'] for d in daily),'unexpected_hours':sum(d['unexpected_hours'] for d in daily),'non_24_hour_dates':[d for d in daily if d['expected_hours']!=24],'first_actual_utc':str(df.timestamp_utc.min()),'last_actual_utc':str(df.timestamp_utc.max()),'interpolation':False,'duplicate_resolution':'collapse only byte-equivalent canonical full rows; preserve raw; conflicting full rows block'}
    save(OUT/'DATA_QUALITY.json',quality)
    assert not tzbad and quality['unexpected_hours']==0,'BLOCKED_TIMESTAMP_QUALITY'
    conflicts=[str(t) for t,g in duplicated.groupby('timestamp_utc') if g.full_row_sha256.nunique()!=1]
    quality['conflicting_duplicate_timestamps']=conflicts
    assert not conflicts,'BLOCKED_CONFLICTING_DUPLICATES'
    unique=df.drop_duplicates('timestamp_utc',keep='first')
    quality['exact_duplicate_copies_collapsed']=len(df)-len(unique)
    quality['unique_hourly_rows']=len(unique)
    save(OUT/'DATA_QUALITY.json',quality)
    end=pd.Timestamp('2026-09-01',tz='UTC')
    grid=pd.date_range(pd.Timestamp(seal['first_day'],tz='UTC'),end,freq='h',inclusive='left')
    values=unique.set_index('timestamp_utc').load.reindex(grid).to_numpy(dtype=float)
    finite=np.isfinite(values)
    pd.DataFrame({'timestamp_utc':grid[~finite],'reason':'MISSING_OR_NONFINITE_NO_INTERPOLATION'}).to_csv(OUT/'MISSING_UTC_HOURS.csv',index=False)
    cache_series=CACHE/'hourly_utc.csv';pd.DataFrame({'timestamp_utc':grid,'load':values}).to_csv(cache_series,index=False)
    mask=(grid>=pd.Timestamp(seal['splits']['TRAIN'][0],tz='UTC'))&(grid<pd.Timestamp(seal['splits']['TRAIN'][1],tz='UTC'))&finite
    sigma=float(np.std(values[mask],ddof=0));assert math.isfinite(sigma) and sigma>0
    invalid=np.r_[0,np.cumsum(~finite)]
    origins=[];strata=[];target_counts=np.zeros(len(grid)+1,dtype=int)
    for split,(start,stop) in seal['splits'].items():
        left=pd.Timestamp(start,tz='UTC');right=pd.Timestamp(stop,tz='UTC')
        for i in np.flatnonzero((grid>=left)&(grid<right)):
            i=int(i);origin=grid[i];reason='ELIGIBLE'
            if i<512:reason='INSUFFICIENT_PAST'
            elif origin+pd.Timedelta(hours=64)>right:reason='TARGET_OUTSIDE_SPLIT'
            elif invalid[i+64]-invalid[i-512]>0:reason='NONFINITE_CONTEXT_OR_TARGET'
            origins.append({'split':split,'origin_utc':origin.isoformat(),'index':i,'eligible':reason=='ELIGIBLE','reason':reason})
            if split=='E_CONFIRM' and reason=='ELIGIBLE':
                score=shift_score(values[i-512:i],sigma)
                strata.append({'index':i,'origin_utc':origin.isoformat(),'utc_date':origin.date().isoformat(),'year':origin.year,'S':score,'bin':classify(score)})
                target_counts[i]+=1;target_counts[i+64]-=1
    origins=pd.DataFrame(origins);st=pd.DataFrame(strata)
    assert len(st),'BLOCKED_NO_LEGAL_E_ORIGINS'
    origins.to_csv(OUT/'ORIGIN_AUDIT.csv',index=False);st.to_csv(OUT/'SHIFT_STRATA.csv',index=False)
    bins=[];annual=[]
    for label in BINS:
        sub=st[st.bin==label];bins.append({'bin':label,'origins':len(sub),'unique_utc_dates':sub.utc_date.nunique(),'origin_fraction':len(sub)/len(st)})
    for year in sorted(st.year.unique()):
        for label in BINS:
            sub=st[(st.year==year)&(st.bin==label)];annual.append({'year':int(year),'bin':label,'origins':len(sub),'unique_utc_dates':sub.utc_date.nunique()})
    pd.DataFrame(bins).to_csv(OUT/'BIN_COUNTS.csv',index=False);pd.DataFrame(annual).to_csv(OUT/'ANNUAL_BIN_COUNTS.csv',index=False)
    # Independent scalar calculation: all scores, population std, legal windows.
    train_values=[float(v) for v in values[mask]];avg=math.fsum(train_values)/len(train_values)
    scalar_sigma=math.sqrt(math.fsum((v-avg)**2 for v in train_values)/len(train_values))
    assert abs(scalar_sigma-sigma)<1e-9
    maxdiff=0;independent_bins=collections.Counter();independent_dates=collections.defaultdict(set)
    for row in st.itertuples(index=False):
        i=row.index;ref=values[i-160:i-32].tolist();median=statistics.median(ref)
        denom=max(1.4826*statistics.median([abs(v-median) for v in ref]),.1*scalar_sigma)
        s=abs(statistics.median(values[i-32:i].tolist())-median)/denom
        maxdiff=max(maxdiff,abs(s-row.S));assert abs(s-row.S)<1e-10
        label=classify(s);independent_bins[label]+=1;independent_dates[label].add(row.utc_date)
    for row in bins:assert independent_bins[row['bin']]==row['origins'] and len(independent_dates[row['bin']])==row['unique_utc_dates']
    # Direct scalar finite-window check for every grid origin (no prefix-count reuse).
    checked=0
    for row in origins.itertuples(index=False):
        i=row.index;split_end=pd.Timestamp(seal['splits'][row.split][1],tz='UTC')
        legal=i>=512 and grid[i]+pd.Timedelta(hours=64)<=split_end and len(values[i:i+64])==64
        if legal:legal=all(math.isfinite(v) for v in values[i-512:i+64])
        assert bool(legal)==row.eligible;checked+=1
    primary=bins[-1];ready=primary['origins']>=50 and primary['unique_utc_dates']>=14
    counts=np.cumsum(target_counts)[:-1];used=counts[counts>0]
    status='READY_FOR_REAL_CONFIRMATION_TRAINING' if ready else 'REAL_SHIFT_CONDITION_TOO_RARE_FOR_PRIMARY_STUDY'
    summary={'status':status,'protocol_sha256':seal['protocol_sha256'],'splits':seal['splits'],'requested_local_days':[seal['first_day'],seal['last_day_inclusive']],'utc_grid_hours':len(grid),'raw_rows':len(df),'finite_utc_hours':int(finite.sum()),'missing_utc_hours':int((~finite).sum()),'sigma_train_population':sigma,'sigma_train_finite_hours':int(mask.sum()),'legal_origins':{s:int(((origins.split==s)&origins.eligible).sum()) for s in seal['splits']},'E_unique_dates':int(st.utc_date.nunique()),'bins':bins,'primary_requirement':{'min_origins':50,'min_unique_utc_dates':14},'target_overlap':{'E_origin_target_pairs':len(st)*64,'unique_target_hours':len(used),'max_origin_coverage_per_target_hour':int(used.max()),'mean_origin_coverage_per_used_target_hour':float(used.mean()),'not_independent_samples':True},'past_only_score_distribution':{'min':float(st.S.min()),'median':float(st.S.median()),'max':float(st.S.max())},'series_receipt':{'path':str(cache_series.relative_to(ROOT)),'sha256':sha(cache_series)},'old_2026_result':'INCONCLUSIVE_DATA_COVERAGE (9 origins / 2 days)','optimizer_updates':0,'model_predictions':0,'target_error_scores':0,'automatic_training':False,'source_threshold_gate_changes':False}
    save(OUT/'COVERAGE_SUMMARY.json',summary)
    check_seal();check_parent()
    save(OUT/'VERIFICATION.json',{'status':'VERIFIED_COVERAGE_ONLY','daily_file_hashes_verified':len(receipts),'preserved_parent_hashes_verified':preserved,'sealed_protocol_unchanged':True,'score_code_unchanged':True,'independent_scalar_E_scores':len(st),'scalar_max_abs_difference':maxdiff,'scalar_sigma_difference':abs(sigma-scalar_sigma),'direct_legal_origin_checks':checked,'independent_bin_counts_agree':True,'all_utc_offsets_valid':True,'canonical_utc_timestamps_unique':True,'raw_exact_duplicate_copies':quality['exact_duplicate_copies_collapsed'],'raw_conflicting_duplicates':len(conflicts),'optimizer_updates':0,'model_predictions':0,'target_error_scores':0,'gpu_model_initializations':0})
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
