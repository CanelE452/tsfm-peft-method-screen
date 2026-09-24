"""BIDMC impedance RESP: waveform forecasting, NOT respiratory-rate diagnosis.
Peak detection uses only the current context. No manual future breath annotations.
"""
from __future__ import annotations
import re,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from .common import Case,Blocked,require,write_json,nuisance,scale_of

BASE='https://physionet.org/files/bidmc/1.0.0/bidmc_csv/'

def period_stats(x,fs=5.,prom=.3):
    # Past-only 3-sample moving average with an explicit known 1-sample lag.
    smooth=np.convolve(x,np.ones(3)/3,mode='valid')
    peaks,_=find_peaks(smooth,distance=int(fs),prominence=prom*max(float(np.std(smooth)),1e-8))
    peaks=peaks+2  # endpoints of causal 3-sample averages
    intervals=np.diff(peaks)/fs
    if len(intervals)<4:return float('nan'),None,peaks
    med=float(np.median(intervals))
    cv=float(np.median(np.abs(intervals-med))*1.4826/max(med,1/fs))
    return cv,med,peaks

def cycle_forecast(x,h,fs=5.):
    _,period,peaks=period_stats(x,fs)
    if period is None or len(peaks)<2:return np.repeat(x[-1],h)[:,None]
    a,b=peaks[-2],peaks[-1]
    template=x[a:b+1];P=b-a
    # Continue the last fully observed cycle; no future time-warp/peaks are used.
    future=(np.arange(1,h+1)+len(x)-1-b)%P
    return np.interp(future,np.arange(len(template)),template)[:,None]

def read_signal(path):
    import pandas as pd
    d=pd.read_csv(path);d.columns=d.columns.str.strip()
    require('RESP' in d and 'Time [s]' in d,'BIDMC CSV schema changed')
    t=d['Time [s]'].to_numpy(float);x=d['RESP'].to_numpy(float)
    require(len(x)>50000 and np.allclose(np.diff(t),.008,atol=1e-5),'BIDMC sampling contract mismatch')
    # Nonoverlapping right-closed block averages: no future sample enters a context block.
    n=(len(x)//25)*25
    return np.mean(x[:n].reshape(-1,25),1)

def cases_for_signal(x,record,group):
    cases=[];excluded=0;C,H=320,40
    for end in range(C,len(x)-H+1,160):
        past=x[end-C:end].copy();future=x[end:end+H,None].copy()
        if not np.isfinite(past).all() or np.std(past)<1e-6:excluded+=1;continue
        score,period,pk=period_stats(past,prom=.3);alt,_,_=period_stats(past,prom=.4)
        if not np.isfinite(score) or not np.isfinite(alt):excluded+=1;continue
        pred=cycle_forecast(past,H)
        extras=np.r_[pred[:,0],period,len(pk)]
        c=Case(f'bidmc{record:02d}:{end}',group,past[:,None],future,score,alt,
               nuisance(past,period),pred,scale_of(past),extras)
        cases.append(c.validate())
    return cases,excluded

def prepare(fetch,public,config):
    records=[]
    for i in range(1,25):
        p=fetch.get(BASE+f'bidmc_{i:02d}_Fix.txt',f'fix_{i:02d}.txt',cap=4096)
        m=re.search(r'MIMIC II matched wdb ID:\s*(\S+)',p.read_text())
        if not m:raise Blocked('BIDMC_PATIENT_ID_UNRESOLVED')
        records.append((i,m.group(1)))
    # At most one recording per actual source patient; no train/test patient overlap.
    unique=[];seen=set()
    for item in records:
        if item[1] not in seen:unique.append(item);seen.add(item[1])
    if len(unique)<20:raise Blocked('BIDMC_TOO_FEW_UNIQUE_PATIENTS_IN_FIXED_24_RECORDS')
    parts={'train':unique[:12],'cal':unique[12:16],'test':unique[16:]}
    write_json(public/'DATA_SPLIT.json',{'record_ids':{k:[i for i,_ in v] for k,v in parts.items()},
          'duplicate_recordings_excluded':[i for i,g in records if (i,g) not in unique],
          'kind':'disjoint original MIMIC patient identifiers; medical metadata is not published',
          'task':'64 s past -> 8 s future RESP block-average waveform at 5 Hz; not the original RR-estimation task'})
    out={};counts={}
    for split,items in parts.items():
        out[split]=[]
        for i,g in items:
            p=fetch.get(BASE+f'bidmc_{i:02d}_Signals.csv',f'signals_{i:02d}.csv',cap=6*1024**2)
            arr=read_signal(p);group='patient_'+hashlib.sha256(g.encode()).hexdigest()[:10]
            cs,n=cases_for_signal(arr,i,group);out[split].extend(cs);counts[str(i)]={'used':len(cs),'context_excluded':n}
    write_json(public/'DATA_QC.json',{'counts':counts,'future_breath_annotation_used':False,
          'peak_agreement':'fixed prominence .3 vs .4 sensitivity, not certified measurement reliability',
          'health_claim':'NONE; no diagnosis or clinical decision'})
    return out
