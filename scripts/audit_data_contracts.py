import json
import numpy as np
import pandas as pd
from tsfm_peft_screen.data import Panel
from tsfm_peft_screen.reproducibility import ROOT,write_json,sha
records={}
for name in ['jena','m5','ettm2','electricity']:
    p=Panel(name);meta=p.meta;assert len(p.channels)==len(set(p.channels.tolist()))
    for split in ['train','validation']:
        for o in p.origins[split]:
            x,y=p.window(int(o));assert np.isfinite(y).all(),(name,split,int(o))
    records[name]={'train_validation_targets_finite':True,'unique_channels':len(p.channels),'input_manifest_sha256':sha(p.root/'manifest.json'),'no_future_index_assertions':True}
    if name=='jena':assert meta['extra']['duplicate_timestamps']==0
    if name=='m5':assert len(p.channels)==256 and np.all(p.zero_fraction>=.5)
dates=pd.to_datetime(pd.read_csv(ROOT/'data/raw/ETTm2.csv',usecols=['date'])['date'])
assert dates.is_monotonic_increasing and dates.is_unique
steps=dates.diff().dropna();assert (steps==pd.Timedelta(minutes=15)).all()
records['ettm2']['timestamps_strict_15min']=True
write_json(ROOT/'results/screening_summary/data_integrity.json',{'status':'PASS','datasets':records,'scope':'train/V target integrity; full raw ETT timestamp metadata only, no E target scores'})
print('Data integrity PASS')
