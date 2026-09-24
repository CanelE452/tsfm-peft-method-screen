"""SDWPF/KDD diagnostic subset; no ERA5 future reanalysis input.
Proxy: change of TRAIN-estimated direction-sector neighbour correlations.
This is an empirical relation score, NOT an identified causal wake graph.
Ndir is a nacelle-heading proxy. Geometry is used for distances only: no
unverified north/east orientation is assigned to anonymized x/y coordinates.
"""
from __future__ import annotations
import contextlib,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .common import Case,Blocked,require,write_json,nuisance,scale_of,hash_file

ARTICLE='https://api.figshare.com/v2/articles/24798654/versions/2'
COLS=['TurbID','Day','Tmstamp','Wspd','Wdir','Ndir','Pab1','Pab2','Pab3','Patv']

def source(fetch,options,public):
    if options.get('wind_csv'):
        csv=Path(options['wind_csv']).resolve();loc=Path(options.get('wind_locations','')).resolve()
        if not csv.is_file() or not loc.is_file():raise Blocked('WIND_LOCAL_CSV_AND_LOCATIONS_REQUIRED')
        return ('csv',csv,loc)
    if options.get('wind_archive'):
        p=Path(options['wind_archive']).resolve()
        if not p.is_file():raise Blocked('WIND_LOCAL_ARCHIVE_NOT_FOUND')
    else:
        meta=fetch.json(ARTICLE,'figshare_article_v2.json')
        files=meta.get('files',[])
        choices=[f for f in files if 'sdwpf_kddcup' in f.get('name','').lower() and f.get('name','').lower().endswith('.zip')]
        write_json(public/'SDWPF_FILES.json',{'article':ARTICLE,'available':[{'name':f.get('name'),'bytes':f.get('size')} for f in files]})
        if len(choices)!=1:raise Blocked('WIND_OFFICIAL_KDD_ARCHIVE_NOT_UNAMBIGUOUS; provide existing --wind-csv and --wind-locations')
        f=choices[0]
        if int(f.get('size',10**12))>960*1024**2:raise Blocked('WIND_OFFICIAL_ARCHIVE_EXCEEDS_QUICK_BUDGET')
        md5=f.get('computed_md5') or f.get('supplied_md5')
        p=fetch.get(f['download_url'],'sdwpf_kddcup.zip',cap=960*1024**2,md5=md5)
    with zipfile.ZipFile(p) as z:
        names=z.namelist();targets=[n for n in names if Path(n).name=='sdwpf_245days_v1.csv']
        coords=[n for n in names if 'location' in Path(n).name.lower() and n.lower().endswith('.csv') and z.getinfo(n).file_size<1024**2]
        if len(targets)!=1 or len(coords)!=1:raise Blocked('WIND_ZIP_SCHEMA_UNSUPPORTED; no guessed or substitute dataset')
        return ('zip',p,(targets[0],coords[0]))

def load_panel(src):
    typ,p,other=src
    @contextlib.contextmanager
    def opened():
        if typ=='zip':
            with zipfile.ZipFile(p) as z, z.open(other[0]) as f:
                yield f,pd.read_csv(z.open(other[1]))
        else:
            with p.open('rb') as f:yield f,pd.read_csv(other)
    pieces=[];total=0
    with opened() as (handle,loc):
        loc.columns=loc.columns.str.strip()
        # Strict documented coordinate schema, with only capitalization normalized.
        rename={c:c.lower() for c in loc.columns};loc=loc.rename(columns=rename)
        if not {'turbid','x','y'}<=set(loc):raise Blocked('WIND_LOCATION_SCHEMA_UNSUPPORTED')
        for chunk in pd.read_csv(handle,usecols=COLS,chunksize=250000):
            total+=len(chunk)
            if total>8_000_000:raise Blocked('WIND_CSV_ROW_BUDGET; use documented KDD file')
            sub=chunk[(chunk.TurbID>=1)&(chunk.TurbID<=12)&(chunk.Day>=1)&(chunk.Day<=42)]
            if len(sub):pieces.append(sub)
    if not pieces:raise Blocked('WIND_REQUIRED_DAYS_OR_TURBINES_MISSING')
    d=pd.concat(pieces,ignore_index=True)
    tm=d.Tmstamp.astype(str).str.split(':',expand=True)
    minute=tm[0].astype(int)*60+tm[1].astype(int)
    require((minute%10==0).all(),'Wind timestamp not on 10-minute grid')
    d['index']=(d.Day.astype(int)-1)*144+minute//10
    require(not d.duplicated(['index','TurbID']).any(),'Duplicate wind timestamps')
    panel={}
    for col in ('Patv','Wspd','Wdir','Ndir','Pab1','Pab2','Pab3'):
        a=np.full((42*144,12),np.nan)
        a[d['index'].to_numpy(int),d.TurbID.to_numpy(int)-1]=d[col].to_numpy(float)
        panel[col]=a
    loc=loc.set_index('turbid')
    if not set(range(1,13))<=set(loc.index):raise Blocked('WIND_LOCATION_IDS_MISSING')
    xy=loc.loc[list(range(1,13)),['x','y']].to_numpy(float)
    require(np.isfinite(xy).all(),'Nonfinite positions')
    # Public dataset's invalid/unknown-target rules, applied to labels only at scoring time.
    valid=np.isfinite(panel['Patv']) & (panel['Patv']>=0)
    valid &= ~((panel['Patv']<=0)&(panel['Wspd']>2.5))
    valid &= np.abs(panel['Wdir'])<=180
    valid &= np.abs(panel['Ndir'])<=720
    for c in ('Pab1','Pab2','Pab3'):valid &= np.isfinite(panel[c]) & (panel[c]<=89)
    panel['valid']=valid
    return panel,xy,{'rows_scanned':total,'selected_rows':len(d),'geometry':'distance only; coordinate orientation not assumed'}

def nearest(xy,i):
    ds=np.linalg.norm(xy-xy[i],axis=1);ds[i]=np.inf
    return np.argsort(ds,kind='stable')[:3]

def bank_for(panel,xy,i,days):
    neigh=nearest(xy,i);power=panel['Patv'];valid=panel['valid']
    ix=np.concatenate([np.arange((d-1)*144,d*144-1) for d in days])
    ix=ix[ix>0]  # never read index -1 (which would be outside TRAIN)
    y=power[ix+1,i]-power[ix,i]
    xx=power[ix][:,neigh]-power[ix-1][:,neigh] # ix=0 excluded below
    yaw=panel['Ndir'][ix,i];sector=(np.floor(np.mod(np.nan_to_num(yaw),360)/45)).astype(int)%8
    ok=(ix>0)&valid[ix,i]&valid[ix+1,i]&np.isfinite(yaw)&np.isfinite(xx).all(1)&np.isfinite(y)
    ok &= valid[ix][:,neigh].all(1)&valid[ix-1][:,neigh].all(1)
    bank=np.zeros((8,3));supports=[]
    for k in range(8):
        m=ok&(sector==k);supports.append(int(m.sum()))
        if m.sum()<24:
            bank[k]=1/3;continue
        for j in range(3):
            a,b=xx[m,j],y[m]
            corr=0. if a.std()<1e-8 or b.std()<1e-8 else abs(float(np.corrcoef(a,b)[0,1]))
            bank[k,j]=corr+.05
        bank[k]/=bank[k].sum()
    return bank,supports

def soft_weights(yaw,bank):
    s=np.mod(yaw,360)/45;k=np.floor(s).astype(int)%8;f=s-np.floor(s)
    return (1-f[:,None])*bank[k]+f[:,None]*bank[(k+1)%8]

def turnover(yaw,bank):
    w=soft_weights(yaw,bank)
    return float(np.abs(np.diff(w,axis=0)).sum(1).mean()/2)

def interp_context(a,valid):
    b=np.array(a,float).copy()
    if np.mean(valid)<.9 or not valid.any():return None
    b[~valid]=np.interp(np.where(~valid)[0],np.where(valid)[0],b[valid])
    return b

def make_cases(panel,xy):
    banks={};alts={};support={}
    for i in (0,3,6,9):
        banks[i],support[i]=bank_for(panel,xy,i,list(range(1,21)))
        alts[i],_=bank_for(panel,xy,i,list(range(1,21,2)))
    if sum(sum(s>=24 for s in support[i])>=4 for i in support)<3:
        raise Blocked('WIND_DIRECTION_SECTOR_TRAIN_SUPPORT_INSUFFICIENT')
    parts={'train':range(1,21),'cal':range(21,28),'test':range(28,42)};out={k:[] for k in parts};rejected=0
    C,H=36,6
    for split,days in parts.items():
        for day in days:
            for offset in (48,84,120):
                end=(day-1)*144+offset
                for i in (0,3,6,9):
                    neigh=nearest(xy,i);r=slice(end-C,end)
                    own=interp_context(panel['Patv'][r,i],panel['valid'][r,i])
                    nb=[interp_context(panel['Patv'][r,j],panel['valid'][r,j]) for j in neigh]
                    yaw=panel['Ndir'][r,i];rel=panel['Wdir'][r,i];sp=panel['Wspd'][r,i]
                    if own is None or any(x is None for x in nb) or not np.isfinite(np.r_[yaw,rel,sp]).all():
                        rejected+=1;continue
                    if np.any(np.abs(yaw)>720) or np.any(np.abs(rel)>180):rejected+=1;continue
                    rad=np.deg2rad(yaw)
                    x=np.column_stack([own,sp,np.sin(rad),np.cos(rad),rel/180,*nb])
                    w=soft_weights(yaw,banks[i]);weighted=(w*np.stack(nb,axis=1)).sum(1)
                    y=panel['Patv'][end:end+H,i,None].copy();mask=panel['valid'][end:end+H,i,None].copy()
                    c=Case(f'day{day}:t{i+1}:o{offset}',f'day{day}',x,y,
                       turnover(yaw,banks[i]),turnover(yaw,alts[i]),nuisance(own,float(sp.mean())),
                       np.full((H,1),own[-1]),max(scale_of(own),1.),weighted,mask)
                    out[split].append(c.validate())
    return out,{'sector_support_train':support,'banks_train':banks,'alternate':'odd TRAIN days only',
         'past_qc_rejected':rejected,'primary':'Ndir-sector empirical neighbour-relation turnover',
         'unresolved':'Ndir is nacelle heading, not a verified geographic wind vector; no causal wake claim',
         'splits':{k:list(v) for k,v in parts.items()},'reserve_day42':'not used in predictors or scores; archive contains it',
         'target':'turbines 1,4,7,10; nearest 3 in fixed turbine subset 1..12',
         'forecast':'6 h context -> 1 h power at 10 min; no future weather or wind',
         'time_group':'whole day shared across all turbines; turbines are not independent farms'}

def prepare(fetch,public,config):
    src=source(fetch,config.get('paths',{}),public)
    panel,xy,info=load_panel(src);out,meta=make_cases(panel,xy)
    info['input_sha256']=hash_file(src[1])
    info['local_path_supplied']=bool(config.get('paths'))
    if src[0]=='csv':info['location_sha256']=hash_file(src[2])
    write_json(public/'DATA_QC.json',{**info,**meta})
    return out
