"""Official evaluation values; raw/cleaned observation provenance; no imputation."""
import unicodedata, time
from core import *

def prepare_data():
    if (OUT/'data_manifest.json').exists():
        print('Existing prepared data retained');return read(OUT/'data_manifest.json')
    paths=sorted((CACHE/'data/BuildingsBench/BDG-2').glob('*.csv'));assert len(paths)==8
    tables=[]
    for p in paths:
        d=pd.read_csv(p,index_col='timestamp',parse_dates=True);assert d.index.is_unique and d.index.is_monotonic_increasing
        assert np.all(np.diff(d.index.asi8)==3600*10**9)
        if len(d.columns):tables.append(d)
    frames=[]
    for site in ['Bear','Fox','Panther','Rat']:
        frames.append(pd.concat([d for d in tables if d.columns[0].startswith(site+'_')]).sort_index())
    official=pd.concat(frames,axis=1);assert official.index.is_unique and official.columns.is_unique
    columns=list(official.columns)
    upstream={}
    for name in ['electricity.csv','electricity_cleaned.csv']:
        d=pd.read_csv(CACHE/'official_source'/name,usecols=['timestamp']+columns,index_col='timestamp',parse_dates=True)
        assert d.index.is_unique;upstream[name]=d.reindex(official.index)
    meta=pd.read_csv(CACHE/'official_source/metadata.csv');assert meta['building_id'].is_unique
    meta=meta.set_index('building_id');eligible=[];details=[];groups={};replacements=[]
    for bid in columns:
        assert bid in meta.index and str(meta.loc[bid,'electricity'])=='Yes',bid
        kaggle=meta.loc[bid,'building_id_kaggle']
        group='kaggle:'+str(int(kaggle)) if pd.notna(kaggle) else 'bdg:'+bid
        groups.setdefault(group,[]).append(bid)
    for group,members in groups.items():
        bid=sorted(members)[0];series=official[bid].to_numpy(dtype=np.float64);raw=upstream['electricity.csv'][bid].to_numpy(dtype=np.float64);clean=upstream['electricity_cleaned.csv'][bid].to_numpy(dtype=np.float64)
        observed=np.isfinite(series)&np.isfinite(raw)&np.isfinite(clean)&np.isclose(series,raw,rtol=0,atol=1e-7)&np.isclose(series,clean,rtol=0,atol=1e-7)
        idx=official.index;edges=np.diff(np.r_[False,observed,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
        block=None
        for s,e in zip(starts,ends):
            s=int(s);e=int(e)
            while s<e and idx[s].hour!=0:s+=1
            if e-s>=120*24:block=(s,s+120*24);break
        row=dict(building_id=bid,physical_group=group,meter_ids=';'.join(sorted(members)),hash=__import__('hashlib').sha256(unicodedata.normalize('NFC',bid).encode('utf-8')).hexdigest(),eligible=False,reason='NO_120D_OBSERVED_BLOCK',observed_hours=int(observed.sum()),archive_finite_hours=int(np.isfinite(series).sum()),timezone=str(meta.loc[bid,'timezone']))
        if block is not None:
            s,e=block;row.update(block_start=str(idx[s]),block_end_exclusive=str(idx[e-1]+pd.Timedelta(hours=1)))
            # Earliest Wednesday/Saturday pair in same calendar week and month.
            wed=next((j for j in range(s+14*24,e-3*24-23,24) if idx[j].dayofweek==2 and idx[j].month==idx[j+3*24].month),None)
            if wed is not None:
                cells=[]
                for j in [wed,wed+3*24]:
                    for h in [3,14]:
                        hist=series[j-h*24:j];x,y,dates,c=windows(hist,idx[j]);assert np.isfinite(series[j:j+24]).all()
                        cells.append(dict(building_id=bid,physical_group=group,origin=str(idx[j]),forecast_type='Wednesday' if idx[j].dayofweek==2 else 'Saturday',history_days=h,coverage_count=c,adaptation_windows=len(x),history_start=str(idx[j-h*24]),history_end_exclusive=str(idx[j]),target_end_exclusive=str(idx[j]+pd.Timedelta(hours=24)),history_std=float(hist.std()),j=j))
                valid=all(c['history_std']>1e-6 and (c['coverage_count']>0 if c['history_days']==14 or c['forecast_type']=='Wednesday' else c['coverage_count']==0) for c in cells)
                if valid:row.update(eligible=True,reason='ELIGIBLE');eligible.append((row,cells,series))
                else:row['reason']='EPISODE_HISTORY_STD_OR_COVERAGE';replacements.append(dict(building_id=bid,reason=row['reason']))
            else:row['reason']='NO_FIXED_PAIR'
        details.append(row)
    details.sort(key=lambda r:r['hash']);csvwrite(OUT/'eligibility.csv',details)
    save(OUT/'building_groups.json',dict(tag='확인',groups=groups,rule='Metadata building_id_kaggle groups if present, otherwise unique BDG building_id; electricity only; multiple years grouped. First lexicographic meter per physical group.',replacements=replacements))
    eligible.sort(key=lambda x:x[0]['hash'])
    if len(eligible)<14:
        d=dict(tag='확인',status='INSUFFICIENT_ELIGIBLE_BUILDINGS',eligible=len(eligible),required=14,stageA='NOT_RUN',fits=0);save(OUT/'data_manifest.json',d);return d
    split={};episodes=[];staged={}
    for n,(row,cells,series) in enumerate(eligible[:14]):
        role='recipe' if n<4 else 'dev' if n<8 else 'heldout';split.setdefault(role,[]).append(row['building_id'])
        for cell in cells:
            j=cell.pop('j');eid=f"{row['building_id']}_{cell['forecast_type']}_H{cell['history_days']}";folder=CACHE/'episodes'/role;folder.mkdir(parents=True,exist_ok=True)
            hp=folder/(eid+'_history.npy');yp=folder/(eid+'_target.npy');np.save(hp,series[j-24*cell['history_days']:j]);np.save(yp,series[j:j+24]);staged.update({str(p.relative_to(ROOT)):sha(p) for p in [hp,yp]})
            cell.update(id=eid,role=role,history_file=str(hp.relative_to(ROOT)),target_file=str(yp.relative_to(ROOT)),history_sha256=sha(hp),target_sha256=sha(yp),calendar_timezone=row['timezone']);episodes.append(cell)
    assert len({r['physical_group'] for r in episodes})==14
    save(OUT/'building_split.json',dict(tag='확인',**split,rule='NFC UTF-8 building_id sha256 ascending, 4/4/6 physical groups'))
    save(OUT/'episodes.json',episodes);csvwrite(OUT/'episode_manifest.csv',episodes)
    d=dict(tag='확인',status='DATA_READY',eligible=len(eligible),selected=14,episodes=len(episodes),dataset='BuildingsBench v1.0.0 BDG-2 actual hourly electricity',timestamp_policy='Provided naive timestamp calendar; BDG metadata timezone recorded, no DST reconstruction',observation_policy=read(CONFIG)['observations'],eligibility_statistics='Only finite/equality masks across 120d; std computed only in exposed episode histories; no future performance selection',evaluation_archive=read(OUT/'download_manifest.json'),upstream=read(OUT/'observation_sources.json'),official_csv={str(p.relative_to(ROOT)):sha(p) for p in paths},official_code={str(p.relative_to(ROOT)):sha(p) for p in (CACHE/'official_source').rglob('*.py')},staged=staged,source_transfer=False,prepared_at=time.time(),heldout_access='Only mechanical eligibility, episode slicing and hashes; no heldout predictions or performance')
    save(OUT/'data_manifest.json',d);print('DATA_READY eligible',len(eligible),'split',split,flush=True);return d

def episodes(role):return [e for e in read(OUT/'episodes.json') if e['role']==role]
def history(e):
    p=ROOT/e['history_file'];assert sha(p)==e['history_sha256'];h=np.load(p);assert len(h)==24*e['history_days'];return h

def target(e):
    if e['role']=='heldout':assert (OUT/'heldout_seal.json').exists(),'Held-out unopened'
    assert e['role']!='recipe','Recipe episode future is never evaluated'
    p=ROOT/e['target_file'];assert sha(p)==e['target_sha256'];return np.load(p)
