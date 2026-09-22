import gc,pickle,time
import numpy as np
import torch
from chronos import Chronos2Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from common import *
from model import ModelA,ModelB,equal_atoms
from evaluation import predict,path,loadpred,calibrate
from A.data import Solar
from bdata_adapter import Wind

class Direct:
    arm='CHRONOS2'
    def __init__(self):
        info=read(RESULTS/'A/SOURCE_AND_MODEL_MANIFEST.json')['models']['amazon/chronos-2']
        self.pipeline=Chronos2Pipeline.from_pretrained(info['path'],device_map='cuda',torch_dtype=torch.float32,local_files_only=True)
    def distribution(self,x,first=None):
        q,_=self.pipeline.predict_quantiles([v[None].cpu() for v in x],prediction_length=128,quantile_levels=[i/10 for i in range(1,10)],batch_size=len(x),context_length=512,cross_learning=False)
        q=torch.cat(q,0).to(x.device);return q,equal_atoms(q),q[:,:64].transpose(1,2)

def features(packet):
    history=np.repeat(packet['x'][:,-8:,None],8,axis=2).transpose(0,2,1)
    weather=packet['weather'];summary=np.concatenate([weather.mean(1),weather.std(1),np.quantile(weather,[.1,.5,.9],axis=1).transpose(1,2,0,3).reshape(len(weather),8,-1)],-1)
    control=np.repeat(packet['control'].reshape(len(weather),1,-1),8,axis=1)
    lead=np.broadcast_to(np.arange(1,9)[None,:,None],(len(weather),8,1))
    return np.concatenate([history,summary,control,lead],-1).reshape(len(weather)*8,-1)

def cpu_predict(arm,packet,models=None):
    if arm=='NAIVE':
        # issue+24h minus24h is unavailable issue itself: repeat issue-24h recursively.
        z=packet['x'][:,[-7,-6,-5,-4,-3,-2,-1,-8]][...,None]
    else:z=np.stack([m.predict(features(packet)) for m in models],-1).reshape(len(packet['x']),8,9)
    return dict(z=z.astype(np.float32),p=np.full_like(z,1/z.shape[-1],dtype=np.float32),**{k:packet[k] for k in ['y','sigma','pairs']})

def save_cpu(pred,p,start):
    p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**pred)
    save(p.with_suffix('.json'),dict(sha256=sha(p),seconds=time.perf_counter()-start,examples=len(pred['z']),bytes=p.stat().st_size))

def load_gb():
    with (CACHE/'B/gbqr.pkl').open('rb') as f:return pickle.load(f)

def make(c,arm,d):
    if c=='A':return Direct() if arm=='CHRONOS2' else ModelA(arm,lora=False)
    return ModelB('F0',d.dim,lora=False)

def prepare(c):
    r=RESULTS/c;cal=read(r/'CALIBRATION.json')
    if '0' in cal:return
    cal['0']={};d=Solar() if c=='A' else Wind();cpu_models=None
    arms=['F0_NATIVE','F0_MEDIAN','CHRONOS2'] if c=='A' else ['F0','NAIVE','GBQR']
    if c=='B':
        train=d.packet('TRAIN');xx=features(train);yy=train['y'].ravel();start=time.perf_counter();cpu_models=[]
        for q in np.arange(1,10)/10:
            m=GradientBoostingRegressor(loss='quantile',alpha=q,n_estimators=100,max_depth=2,learning_rate=.05,random_state=92300)
            m.fit(xx,yy);cpu_models.append(m)
        with (CACHE/'B/gbqr.pkl').open('wb') as f:pickle.dump(cpu_models,f)
        save(r/'CPU_STATISTICAL_FITS.json',dict(fits=9,seconds=time.perf_counter()-start,parameters=cpu_models[0].get_params(),feature_count=xx.shape[-1],training_rows=len(xx),sha256=sha(CACHE/'B/gbqr.pkl')))
    for arm in arms:
        m=None if arm in ['NAIVE','GBQR'] else make(c,arm,d)
        for role in ['CAL','VAL']:
            packet=d.batch(d.pairs[role]) if c=='A' else d.packet(role);p=path(c,0,arm,role)
            if m is not None:predict(m,packet,c,p)
            elif not p.exists():
                start=time.perf_counter();save_cpu(cpu_predict(arm,packet,cpu_models),p,start)
        cal['0'][arm]=calibrate(loadpred(path(c,0,arm,'CAL')),c)
        del m;gc.collect();torch.cuda.empty_cache()
    save(r/'CALIBRATION.json',cal)

def test(c,d,packet):
    arms=['F0_NATIVE','F0_MEDIAN','CHRONOS2'] if c=='A' else ['F0','NAIVE','GBQR'];cpu_models=load_gb() if c=='B' else None
    for arm in arms:
        p=path(c,0,arm,'TEST')
        if arm in ['NAIVE','GBQR']:
            if not p.exists():
                start=time.perf_counter();save_cpu(cpu_predict(arm,packet,cpu_models),p,start)
        else:
            m=make(c,arm,d);predict(m,packet,c,p);del m;gc.collect();torch.cuda.empty_cache()
