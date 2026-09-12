"""Chronological windows. E targets stay in a separate file until seal."""
import json
from pathlib import Path
import numpy as np
from .reproducibility import ROOT,sha,write_json,digest
class Panel:
    def __init__(self,name):
        self.name=name;self.root=ROOT/'data/processed'/name
        self.meta=json.loads((self.root/'manifest.json').read_text())
        z=np.load(self.root/'fit.npz');self.values=z['values'];self.scale=z['scale'];self.rms_scale=z['rms_scale'];self.caps=z['caps'];self.zero_fraction=z['zero_fraction'];self.channels=z['channels']
        self.origins={k:np.array(v,dtype=int) for k,v in self.meta['origins'].items()};self.e_open=False
    def open_e(self,selection_path,contract):
        from .selection import require_seal
        require_seal(selection_path,contract)
        if self.e_open:raise RuntimeError('E already opened')
        z=np.load(self.root/'evaluation.npz');self.values=np.concatenate([self.values,z['tail']],axis=0);self.e_open=True
    def window(self,origin,channels=None,target=True):
        c=np.arange(len(self.channels)) if channels is None else np.asarray(channels)
        assert origin>=336 and origin<=len(self.values)
        indices=np.arange(origin-336,origin);assert indices.max()<origin
        x=self.values[indices][:,c].T.copy()
        y=self.values[origin:origin+48,c].T.copy() if target else None
        if target:assert y.shape==(len(c),48)
        return x,y

def prepare():
    import pandas as pd
    raw=ROOT/'data/raw';dest=ROOT/'data/processed';dest.mkdir(exist_ok=True)
    def save(name,a,channels,bounds,origins,freq,sources,extra=None):
        a=np.asarray(a,dtype=np.float32);g,t,v,e=bounds
        train=a[t:v];scale=np.nanstd(train,axis=0,dtype=np.float64);scale=np.maximum(scale,1e-6)
        rms=np.nanmean(np.diff(train.astype(float),axis=0)**2,axis=0);rms=np.maximum(rms,1e-6)
        caps=np.array([max(1.,np.quantile(col[col>0],.6)) if (col>0).any() else 1. for col in train.T])
        d=dest/name;d.mkdir(exist_ok=True)
        np.savez_compressed(d/'fit.npz',values=a[:e],scale=scale,rms_scale=rms,caps=caps,zero_fraction=(train==0).mean(0),channels=np.array(channels,dtype=str))
        np.savez_compressed(d/'evaluation.npz',tail=a[e:])
        for key,oo in origins.items():
            lo,hi={'gate':(g,t),'train':(t,v),'validation':(v,e),'evaluation':(e,len(a))}[key]
            assert not oo or (min(oo)>=lo and max(oo)+48<=hi),(name,key,lo,hi)
        meta=dict(name=name,frequency=freq,bounds=dict(gate_start=g,train_start=t,validation_start=v,evaluation_start=e,end=len(a)),origins=origins,channels=list(channels),scale=scale.tolist(),caps=caps.tolist(),sources={str(p):sha(p) for p in sources},files={p.name:sha(p) for p in d.glob('*.npz')},extra=extra or {})
        write_json(d/'manifest.json',meta);write_json(ROOT/'results/screening_summary'/f'{name}_manifest.json',meta)
        print(name,a.shape,{k:len(o) for k,o in origins.items()},flush=True)
    # Jena: Aug 1 precontext, Aug 15--21 gate, Aug 21--Sep 20 train,
    # Sep 20--Oct 4 V, Oct 4--Nov 5 E; untouched by earlier pilot E.
    p=raw/'jena/mpi_roof_2021b.csv';df=pd.read_csv(p,encoding='latin1');date=pd.to_datetime(df.iloc[:,0],format='%d.%m.%Y %H:%M:%S')
    df=df.iloc[:,1:5].apply(pd.to_numeric,errors='coerce');df.index=date;df=df.mask(df<=-9999);df=df.groupby(level=0).first();df=df.loc[df.index.minute==0]
    df=df.reindex(pd.date_range('2021-08-01','2021-11-05',freq='h',inclusive='left'))
    ix=lambda s:int((pd.Timestamp(s)-df.index[0]).total_seconds()/3600)
    g,t,v,e=map(ix,['2021-08-15','2021-08-21','2021-09-20','2021-10-04'])
    save('jena',df.values,df.columns,(g,t,v,e),dict(gate=list(range(g,t-47,6)),train=list(range(t,v-71,24)),validation=list(range(v,e-71,24)),evaluation=list(range(e,e+30*24,24))),'1h',[p],{'dates':[str(df.index[0]),str(df.index[-1])],'stream_origins':30,'duplicate_policy':'first; raw duplicate timestamp audit below','duplicate_timestamps':int(date.duplicated().sum())})
    # M5: item eligibility and scale use days 337..1400 only. No E-informed filtering.
    p=raw/'m5/sales_train_evaluation.csv';df=pd.read_csv(p);cols=[f'd_{i}' for i in range(1,1942)];a=df[cols].to_numpy(dtype=np.float32)
    tr=a[:,336:1400];eligible=np.flatnonzero(((tr==0).mean(1)>=.5)&((tr>0).sum(1)>=40)&(tr.std(1)>0))
    selected=sorted(eligible,key=lambda i:digest(str(df.iloc[i]['id'])))[:256];assert len(selected)==256
    ids=df.iloc[selected]['id'].tolist()
    save('m5',a[selected].T,ids,(0,336,1400,1650),dict(gate=[],train=list(range(336,1400-47,24)),validation=[1400,1450,1500,1550,1600],evaluation=[1650,1698,1746,1794,1842,1890]),'1d',[p],{'selection':'train zero_fraction>=0.5, positives>=40; SHA256 id ordering; first256'})
    p=raw/'ETTm2.csv';df=pd.read_csv(p);a=df.iloc[:,1:].to_numpy(dtype=np.float32)
    save('ettm2',a,df.columns[1:],(336,1024,6000,7200),dict(gate=list(range(336,336+128*4,4)),train=list(range(1024,6000-47,24)),validation=list(range(6000,7200-47,96)),evaluation=list(range(7200,7200+24*96,96))),'15min',[p])
    p=raw/'electricity.txt.gz';a=np.loadtxt(p,delimiter=',',dtype=np.float32)
    valid=[i for i in range(a.shape[1]) if np.isfinite(a[336:6000,i]).all() and np.std(a[336:6000,i])>0][:4]
    save('electricity',a[:,valid],[f'column_{i:03}' for i in valid],(0,336,6000,7200),dict(gate=[],train=list(range(336,6000-47,24)),validation=list(range(6000,7200-47,96)),evaluation=list(range(7200,7200+30*24,24))),'1h',[p],{'metadata_order':'source column index, train-only finite nonconstant','timestamps':'integer hourly offsets; source provides no timestamp column'})
