import json,sys
import numpy as np
import pandas as pd
import torch
from common import *
from model import ModelA,ModelB,setup,tensor_hash
from engine import tensors,distribution
from bdata_adapter import Wind
from A.data import Solar

def main():
    setup();d=Wind();a=Solar();checks=[]
    audit=read(RESULTS/'B/DATA_AUDIT.json')
    for r in audit['source_manifest'].values():assert sha(r['path'])==r['sha256']
    rawdir=CACHE/'B/data/raw'
    power=pd.read_csv(rawdir/'pre_processing__Data__Sweden_Zone3_Power.csv',parse_dates=['time'])
    series=power.set_index('time').wind_power;assert series.index.is_unique
    for i,issue in enumerate(d.raw['issue']):
        t=pd.Timestamp(issue);assert t>=pd.Timestamp('2015-01-05') and t<pd.Timestamp('2019-09-01')
        np.testing.assert_array_equal(d.raw['x'][i],series.loc[pd.date_range(t-pd.Timedelta(hours=384),periods=128,freq='3h')].to_numpy())
        np.testing.assert_array_equal(d.raw['y'][i],series.loc[pd.date_range(t+pd.Timedelta(hours=3),periods=8,freq='3h')].to_numpy())
    ens=pd.read_csv(rawdir/'pre_processing__Data__Sweden_Zone3_Ensembles.csv',parse_dates=['time'])
    ens['issue']=ens.time-pd.to_timedelta(ens.horizon,unit='h')
    selected=ens[ens.issue.isin(pd.to_datetime(d.raw['issue']))]
    assert not selected.duplicated(['issue','number','horizon']).any()
    assert selected.groupby(['issue','number']).size().eq(8).all()
    assert selected.groupby('issue').number.nunique().eq(50).all()
    assert all(np.isfinite(d.raw[k]).all() for k in ['x','y','weather','control'])
    assert len(d.indices['TEST'])>=60
    for c in ['A','B']:
        state=read(RESULTS/c/'BUDGET_STATE.json');assert state['main_updates']==0 and state['smoke_updates']==12
        assert all(f['status']=='COMPLETE' and f['intent']==f['commit']==2 for f in state['fits'].values())
        assert read(RESULTS/c/'MODEL_SMOKE_AUDIT.json')['status']=='PASS'
    # No new optimization: restore actual saved smoke state and retest final input wiring.
    m=ModelA('CONTEXT3');states=torch.load(CACHE/'A/fits/smoke_CONTEXT3/states.pt',weights_only=True);m.load_learned(states[2])
    b=tensors(a.batch(a.schedule(SEEDS[0])[0]))
    with torch.no_grad():
        z=distribution(m,b,'A')[0];b['y'].fill_(-999999);torch.testing.assert_close(distribution(m,b,'A')[0],z,rtol=0,atol=0)
    del m;torch.cuda.empty_cache()
    m=ModelB('SCENARIO3',d.dim);states=torch.load(CACHE/'B/fits/smoke_SCENARIO3/states.pt',weights_only=True);m.load_learned(states[2])
    b=tensors(d.batch(d.schedule(SEEDS[0])[0]));z,p,_=distribution(m,b,'B')
    loss=(z*p).sum();grads=torch.autograd.grad(loss,[m.pool.queries.weight,m.pool.mix.weight])
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads)
    save(RESULTS/'PRE_MAIN_AUDIT.json',dict(status='PASS',b_actual_time_rows=len(d.raw['x']),b_weather_rows=len(selected),same_issue_member_keys_verified=True,
        model_restore_actual_smoke_checkpoints=True,a_future_target_replacement=True,b_router_gradient_final_data=True,
        additional_optimizer_updates=0,smoke_updates_preserved=24,main_updates=0,
        role_counts={r:len(v) for r,v in d.indices.items()},scope_failure_record='IMPLEMENTATION_INCIDENT.json'))
    print('PRE_MAIN_AUDIT PASS')

if __name__=='__main__':
    with executor_lock():main()
