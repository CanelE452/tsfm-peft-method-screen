"""A preselected 55-full-day temporal panel; no new training data."""
import pandas as pd
from .common import *
from experiments.additive_persistence_validation_v1_20260917 import prepare as gen
from experiments.outlier_signal_peft_v1_20260917.reference_core import scale,select_days_reference
from experiments.c3_identifiability_temporal_20260918.prepare import hourly

def shape_packet(source):
    f=CACHE/'shapes'/source;f.mkdir(parents=True,exist_ok=True)
    d=np.load(CACHE/'data'/source/'E_DISCOVERY_inputs.npz');idx=np.arange(len(d['origins']));base=d['x'][idx].reshape(-1,512);s=np.tile(d['sigma'],len(idx))
    obs=[];offset=[];names=SHAPES+['PAIRED_SHIFT8_D32']
    for name in names:
        for sign in [-1,1]:
            for x0,sigma in zip(base,s):
                x=x0.astype(float).copy();r=scale(x,sigma)
                if name.startswith('STEP'):a,dur=map(int,name.replace('STEP','').split('_D'))
                else:a,dur=8,32
                delta=sign*a*r
                if name.startswith('RAMP'):x[-32:]+=np.linspace(0,delta,32)
                else:x[-dur:]+=delta
                obs.append(x);offset.append(0. if name.startswith('PULSE') else delta)
    obs=np.array(obs,np.float32);n=len(base)
    assert np.array_equal(obs[7*2*n:8*2*n],obs[8*2*n:9*2*n])
    np.save(f/'x.npy',obs);np.save(f/'sigma.npy',np.tile(s,len(names)*2).astype(np.float32));np.save(f/'offset.npy',np.array(offset,float));np.save(f/'origin_indices.npy',idx)
    save(f/'manifest.json',dict(states=names,origin_indices=idx.tolist(),origins=d['origins'][idx].tolist(),signs=[-1,1],labels_read=False,paired_reference_not_new_form='PAIRED_SHIFT8_D32 is the existing SHIFT8 form with balanced signs, used solely to pair PULSE; additionally score legacy PULSE using exact historical SHIFT8 inputs/predictions'))

def prepare_data():
    path=ROOT/'.cache/c3_identifiability_temporal_20260918/raw/neso_2026.csv'
    assert sha(path)=='3b36bdfca0c23b3b949e19a7606c3a414bc0d87f313be5e1f36fc327fc6fecd1'
    a=hourly(ext.CACHE/'raw/neso_2025.csv',2025,'2026-01-01')
    b=hourly(path,2026,'2026-08-28')
    series=pd.concat([a,b]);series=series[series.index>=pd.Timestamp('2025-01-01',tz='UTC')]
    expected=pd.date_range(pd.Timestamp('2025-01-01',tz='UTC'),series.index[-1],freq='30min',tz='UTC')
    assert series.index.equals(expected)
    assert series.resample('h').count().eq(2).all()
    h=series.resample('h').mean();start=int((pd.Timestamp('2026-07-01',tz='UTC')-h.index[0]).total_seconds()/3600)
    origins=select_days_reference(np.arange(start,len(h)-64+1),24,55,90301)
    assert len(set(origins//24))==55 and np.ptp(np.bincount(origins%24,minlength=24))<=1
    previous=np.load(ROOT/'.cache/c3_identifiability_temporal_20260918/data/neso_2026_h1/E_DISCOVERY_inputs.npz')['origins']
    assert previous.max()+63<origins.min()
    prev=gen.OUT,gen.CACHE
    try:
        gen.OUT,gen.CACHE=OUT,CACHE
        manifest,audit=gen.make_packets(NEW,pd.DataFrame({'ND':h.to_numpy()}),['ND'],24,[0,4344,start,len(h)],{'E_DISCOVERY':origins},pd.Series(h.index))
        gen.eval_conditions(NEW,'E_DISCOVERY')
    finally:gen.OUT,gen.CACHE=prev
    shape_packet(NEW)
    np.testing.assert_array_equal(manifest['sigma_train_population'],read(ext.OUT/'DATA_MANIFEST.json')['neso_2025']['sigma_train_population'])
    assert audit[0]['missing_target_values']==0
    manifest.update(raw_path=str(path.relative_to(ROOT)),raw_sha256=sha(path),independent_source=False,previously_downloaded=True,previous_target_overlap=False,context_may_overlap_previous_period=True,origin_days=55,partial_final_origin_day_excluded_before_scoring=True,selection_reason='Existing full-day selector; all 55 legal complete origin days; last partial day excluded without reading performance',evaluation_role='same-source previously unscored temporal extension; exposure audit cannot establish global nonuse',sigma_period='2025 H1 fixed',synthetic_event_labels=True,real_event_labels=False)
    save(OUT/'DATA_MANIFEST.json',{NEW:manifest});csvwrite(OUT/'ORIGIN_AUDIT.csv',audit)
