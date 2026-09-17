"""Build separate observed-input packets and generator-only metadata."""
import shutil
import numpy as np
from .common import *
from .reference_core import transform,rng,STATES,scale

def packet(source,role):
    return np.load(CACHE/'data'/source/f'{role}_inputs.npz')

def prepare():
    for info in read(OUT/'DATA_MANIFEST.json').values():
        assert sha(ROOT/info['path'])==info['sha256']
        for files in info['packet_hashes'].values():
            for path,h in files.items():assert sha(ROOT/path)==h,path
    marker=OUT/'CONDITION_MANIFEST.json'
    if marker.exists():
        for p,h in read(marker)['hashes'].items():assert sha(ROOT/p)==h,p
        return
    archive=OUT/'preflight_e11fa67';archive.mkdir(exist_ok=True)
    for name in ['REPORT.md','FINAL_DECISION.md','status.json','verification.json','implementation_checks.json','MACHINE_CONTRACT.json','SOURCE_MANIFEST.json','FIT_LEDGER.csv']:
        if (OUT/name).exists() and not (archive/name).exists():shutil.copy2(OUT/name,archive/name)
    hashes={};checks={}
    for source in SOURCES:
        folder=CACHE/'conditions'/source;folder.mkdir(parents=True,exist_ok=True)
        d=packet(source,'TRAIN');base=d['x'].reshape(-1,512);sigma=np.tile(d['sigma'],256)
        y0=np.load(CACHE/'data'/source/'TRAIN_labels.npz')['y'].reshape(-1,64)
        xs=np.empty((32,1024,512),np.float32);a0=np.empty_like(xs);ys=np.empty((32,1024,64),np.float32)
        audit=[]
        for epoch in range(32):
            for i in range(1024):
                state=['REFERENCE','POINT','BURST','SHIFT'][(epoch+i)%4]
                x,delta,meta=transform(base[i],sigma[i],state,rng(81800,source,'TRAIN',epoch,i),True)
                xs[epoch,i]=x;a0[epoch,i]=base[i] if state in ['POINT','BURST'] else x
                ys[epoch,i]=y0[i]+delta
                audit.append(dict(epoch=epoch,example=i,**meta))
        for name,value in [('train_x',xs),('train_a0',a0),('train_y',ys),('train_sigma',sigma.astype(np.float32))]:np.save(folder/f'{name}.npy',value)
        save(folder/'train_generator_audit.json',audit)
        checks[source]=dict(train_labels_shared_all_arms_sha256=sha(folder/'train_y.npy'),
                             augmentation_shared_A1_to_A5_sha256=sha(folder/'train_x.npy'),
                             state_counts_per_example=[8,8,8,8],negative_train_observed=int((xs<0).sum()),
                             negative_train_targets=int((ys<0).sum()))
        for role in ['V_SELECT','E_DISCOVERY']:
            d=packet(source,role);base=d['x'].reshape(-1,512);n=len(base);s=np.tile(d['sigma'],len(d['origins']))
            obs=[];offset=[];metadata=[]
            for state in STATES:
                for g in range(2):
                    for i in range(n):
                        x,delta,meta=transform(base[i],s[i],state,rng(81900 if role=='V_SELECT' else 82000,source,role,state,g,i))
                        obs.append(x);offset.append(delta);metadata.append(dict(state=state,generator=g,example=i,**{k:v for k,v in meta.items() if k!='state'}))
            # Scenario names, r0, fault locations, clean past and offsets are NOT model input fields.
            np.save(folder/f'{role}_x.npy',np.array(obs,np.float32))
            np.save(folder/f'{role}_sigma.npy',np.tile(s,20).astype(np.float32))
            np.save(folder/f'{role}_offset.npy',np.array(offset,np.float64))
            save(folder/f'{role}_generator_audit.json',metadata)
            if role=='V_SELECT':
                y=np.load(CACHE/'data'/source/f'{role}_labels.npz')['y'].reshape(n,64)
                np.save(folder/'V_SELECT_y.npy',np.tile(y,(20,1))+np.array(offset)[:,None])
            checks[source][role]=dict(examples=len(obs),negative_observed=int((np.array(obs)<0).sum()),
                                     E_future_not_read_by_generator=role=='E_DISCOVERY')
        train=packet(source,'TRAIN');tr=train['x'].reshape(-1,512);ss=np.tile(train['sigma'],256)
        score=np.array([abs(np.median(x[-32:])-np.median(x[-64:-32]))/scale(x,s) for x,s in zip(tr,ss)])
        threshold=float(np.quantile(score,.9))
        ed=packet(source,'E_DISCOVERY');ex=ed['x'].reshape(-1,512);es=np.tile(ed['sigma'],128)
        tag=np.array([abs(np.median(x[-32:])-np.median(x[-64:-32]))/scale(x,s)>threshold for x,s in zip(ex,es)])
        np.save(folder/'E_history_subset.npy',tag)
        checks[source]['history_threshold_train90']=threshold;checks[source]['history_subset_examples']=int(tag.sum())
        for p in folder.iterdir():hashes[str(p.relative_to(ROOT))]=sha(p)
    save(marker,dict(authorship='locally implemented at explicit user request; not supplied attachment code',
                     provenance_override='너가 직접 만들어서 해줘',epoch_example_index='origin_ordinal*4+channel',
                     train_generator_key=81800,validation_key=81900,evaluation_key=82000,
                     evaluation_states=STATES,hampel_edges='truncated centered window within observed past',
                     even_median='average two middle values',hashes=hashes,checks=checks,
                     E_labels='stored separately in prior data packets; only scorer reads after global prediction save'))
    print('PREPARED_TRANSFORMS',flush=True)
