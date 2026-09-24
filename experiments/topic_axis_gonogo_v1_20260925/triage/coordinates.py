"""Coordinate dependence on ETH/UCY trajectories distributed by EqMotion authors.
Natural heading novelty and controlled yaw stress are separate analyses.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from .common import Case,require,write_json,nuisance

REF='5aec2e0b61c511fa93a24138dd90da59a089084b'
BASE=f'https://raw.githubusercontent.com/MediaBrain-SJTU/EqMotion/{REF}/eth_ucy/data/'
FILES={
 'eth':['eth/biwi_eth.txt'],
 'hotel':['hotel/biwi_hotel.txt'],
 'zara1':['eth/crowds_zara01_train.txt','eth/crowds_zara01_val.txt'],
 'zara2':['eth/crowds_zara02_train.txt','eth/crowds_zara02_val.txt'],
 'univ':['eth/students001_train.txt','eth/students001_val.txt']}

def rotate(x,angle):
    t=np.deg2rad(angle);R=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    return np.asarray(x)@R.T

def heading(x,short=False):
    v=x[-1]-x[-4 if short else 0]
    return float(np.arctan2(v[1],v[0]))

def canonical(x):return rotate(x,-np.rad2deg(heading(x)))

def cv_predict(x,h):
    v=np.mean(np.diff(x[-4:],axis=0),axis=0)
    return x[-1]+np.arange(1,h+1)[:,None]*v

def read_track(path):
    # Author preprocessing explicitly sets xind=13 and zind=15 (not generic 4-column format).
    rows=[]
    for line in path.read_text().splitlines():
        a=line.split()
        if not a:continue
        require(len(a)==17 and a[2]=='Pedestrian','EqMotion trajectory schema changed')
        rows.append([int(float(a[0])),int(float(a[1])),float(a[13]),float(a[15])])
    a=np.array(rows,float);require(len(a)>100,'Too few trajectory points')
    return a

def heading_bank(cases):
    ang=np.array([heading(c.x) for c in cases]);bins=np.floor((ang+np.pi)/(2*np.pi)*16).astype(int)%16
    cnt=np.bincount(bins,minlength=16).astype(float)+1
    return cnt/cnt.sum()

def novelty(x,bank,short=False):
    a=heading(x,short);k=int(np.floor((a+np.pi)/(2*np.pi)*16))%16
    # Circular neighbourhood smoothing; no model output or target is used.
    p=(bank[(k-1)%16]+2*bank[k]+bank[(k+1)%16])/4
    return float(-np.log(p))

def windows(a,scene):
    fmin,fmax=int(a[:,0].min()),int(a[:,0].max());duration=fmax-fmin+1
    bounds={'train':(fmin,fmin+int(.5*duration)),
            'cal':(fmin+int(.5*duration),fmin+int(.7*duration)),
            'test':(fmin+int(.7*duration),fmin+int(.9*duration))}
    # Last 10% is not scored. Entire 8+12-frame window must fit a single partition.
    out={k:[] for k in bounds};C,H=8,12
    for pid in sorted(set(a[:,1])):
        z=a[a[:,1]==pid];z=z[np.argsort(z[:,0])]
        require(len(set(z[:,0]))==len(z),'Duplicate trajectory frame/id')
        for i in range(0,len(z)-C-H+1,C+H):
            w=z[i:i+C+H]
            if not np.all(np.diff(w[:,0])==1):continue
            p=w[:C,2:4];q=w[C:,2:4];offset=p[-1].copy();p=p-offset;q=q-offset
            if not np.isfinite(p).all() or not np.isfinite(q).all() or np.linalg.norm(p[-1]-p[0])<.1:continue
            split=next((k for k,(lo,hi) in bounds.items() if w[0,0]>=lo and w[-1,0]<hi),None)
            if split is None:continue
            pred=cv_predict(p,H);speed=np.linalg.norm(np.diff(p,axis=0),axis=1)
            scale=max(float(speed.mean()*H),1e-3)
            c=Case(f'{scene}:{int(pid)}:{int(w[0,0])}',scene,p,q,0.,0.,
                   np.array([np.log(scale),float(speed.std()/max(speed.mean(),1e-9)),
                     float(np.linalg.norm(p[-1]-p[0])/max(speed.sum(),1e-9)),
                     float(np.linalg.norm(np.diff(p,n=2,axis=0),axis=1).mean()/max(speed.mean(),1e-9)),0.]),
                   pred,scale,canonical(p).reshape(-1))
            out[split].append(c.validate())
    limits={'train':64,'cal':16,'test':32}
    for k,cs in out.items():
        cs.sort(key=lambda c:c.uid)
        if len(cs)>limits[k]:out[k]=[cs[i] for i in np.linspace(0,len(cs)-1,limits[k]).round().astype(int)]
    return out,bounds

def prepare(fetch,public,config):
    out={'train':[],'cal':[],'test':[]};meta={}
    for scene,files in FILES.items():
        arrays=[]
        for rel in files:
            p=fetch.get(BASE+rel,rel.replace('/','_'),cap=5*1024**2);arrays.append(read_track(p))
        a=np.concatenate(arrays);parts,bounds=windows(a,scene)
        for k in out:out[k].extend(parts[k])
        meta[scene]={'bounds':bounds,'n':{k:len(v) for k,v in parts.items()}}
    bank=heading_bank(out['train'])
    for split in out:
        for c in out[split]:c.score=novelty(c.x,bank);c.alternate=novelty(c.x,bank,True)
    write_json(public/'DATA_SPLIT.json',{'scenes':meta,'heading_histogram_train':bank,
       'source_commit':REF,'coordinates':'author x/z planar positions; diagnostic trajectories, not full EqMotion benchmark',
       'sampling':'8 observed and 12 future samples; dataset convention 2.5 Hz (3.2 s / 4.8 s)',
       'uncertainty_unit':'physical scene (5), not each rotated copy or each pedestrian',
       'held_out':'chronological windows within scenes; NOT unseen-scene or guaranteed unseen-person generalization',
       'last_tenth':'not scored; raw download contains these rows'})
    return out

def rotated_cases(cases,angle):
    return [replace(c,uid=c.uid+f'@yaw{angle}',base_uid=c.uid,angle=angle,
         x=rotate(c.x,angle),y=rotate(c.y,angle),simple=rotate(c.simple,angle),
         extra=canonical(rotate(c.x,angle)).reshape(-1)) for c in cases]
