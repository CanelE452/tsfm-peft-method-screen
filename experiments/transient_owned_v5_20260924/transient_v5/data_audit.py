"""Read-only recomputation of the ALREADY EXPOSED BOPTEST diagnostic CSVs.

No simulation, new dataset, fitted PEFT, model selection, or new target/horizon.
Recorded taus are held fixed. Actual residuals are y - fitted curve; endpoint
changes are never renamed residuals or identified slow modes. Downsampling uses
the SAME trajectory and reports phase dependence. Linear interpolation is an
OFFLINE reconstruction diagnostic that uses later samples, NOT a forecast.
"""
from __future__ import annotations
from pathlib import Path
import csv
import json
import math
import numpy as np
from .util import require,write_json,read_json,file_hash

SOURCE='results/boptest_data_contract_20260924'
SIGNALS={'reaTSup_y':'K','reaTRet_y':'K','reaTZon_y':'K','reaPHeaPum_y':'W','reaQHeaPumCon_y':'W'}
COMMAND='oveHeaPumY_u'


def read_table(path: Path):
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f); names=reader.fieldnames; rows=list(reader)
    require(names is not None and 'time' in names and COMMAND in names,'Unexpected CSV schema')
    values={n:np.array([float(r[n]) for r in rows],dtype=np.float64) for n in names}
    require(len(rows)>2 and all(np.isfinite(v).all() for v in values.values()),'Invalid/nonfinite CSV values')
    diff=np.diff(values['time']); require((diff>0).all(),'Timestamp order/duplicates')
    require(np.allclose(diff,diff[0],atol=1e-7,rtol=0),'Nonuniform sampling; no implicit interpolation permitted')
    return values,float(diff[0])


def on_segment(d):
    changes=np.where(np.diff(d[COMMAND])>.5)[0]
    require(len(changes)>0,'No recorded OFF-to-ON crossing (>0.5; source analysis rule)')
    first=int(changes[0]+1)
    falls=np.where(np.diff(d[COMMAND][first:])<-.5)[0]
    stop=int(first+falls[0]+1) if len(falls) else len(d['time'])
    require(stop-first>=3,'Too few ON samples')
    return first,stop


def write_csv(path, rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def audit(repo: Path,out: Path):
    out.mkdir(parents=True,exist_ok=True)
    paths={n:repo/SOURCE/n for n in ('fast_probe.csv','transient_response.csv','TAU_FAST.json')}
    require(all(p.is_file() for p in paths.values()),'Missing recorded BOPTEST artifacts; do not generate substitutes')
    before={n:file_hash(p) for n,p in paths.items()}
    fast,dtfast=read_table(paths['fast_probe.csv']); slow,dtslow=read_table(paths['transient_response.csv'])
    original=read_json(paths['TAU_FAST.json'])
    a,b=on_segment(fast)
    t=fast['time'][a:b]-fast['time'][a]
    fitted=[];residual_rows=[]
    for name,unit in SIGNALS.items():
        require(name in fast and name in original['per_signal'],'Signal not in verified source')
        tau=float(original['per_signal'][name]['tau_fit_s'])
        require(tau>0 and math.isfinite(tau),'Invalid source tau')
        y=fast[name][a:b]
        x=np.column_stack([np.ones(len(t)),np.exp(-t/tau)])
        beta=np.linalg.lstsq(x,y,rcond=None)[0];pred=x@beta;resid=y-pred
        endpoint=float(y[-1]-y[0])
        source_span=float(original['per_signal'][name]['slow_part_span_20min'])
        sse=float(resid@resid);sst=float(((y-y.mean())**2).sum())
        fitted.append({'signal':name,'unit':unit,'tau_from_prior_file_s':tau,
                       'tau_reestimated':False,'n_fit_samples':len(y),'fit_start':float(fast['time'][a]),
                       'fit_end':float(fast['time'][b-1]),'endpoint_change':endpoint,
                       'source_slow_part_span':source_span,'span_matches_endpoint_change':abs(source_span-endpoint)<1e-7,
                       'sample_interval_change_at_on':float(fast[name][a]-fast[name][a-1]),
                       'sample_interval_s':dtfast,'coefficient_intercept':float(beta[0]),'coefficient_exponential':float(beta[1]),
                       'residual_rmse':float(np.sqrt(np.mean(resid**2))),'residual_mae':float(np.mean(abs(resid))),
                       'residual_mean':float(resid.mean()),'residual_at_first':float(resid[0]),
                       'residual_at_last':float(resid[-1]),'r2':None if sst==0 else 1-sse/sst,
                       'physical_modes_identified':False})
        residual_rows.extend({'signal':name,'time':float(fast['time'][a+i]),'observed':float(y[i]),
                              'fixed_tau_fit':float(pred[i]),'residual_observed_minus_fit':float(resid[i])} for i in range(len(y)))
    # Same 30-second trajectory, all offset phases. No separate-run values enter these errors.
    ia,ib=on_segment(slow);time=slow['time'];y=slow['reaTSup_y'];origin=time[ia-1]
    window=(time>=origin-120)&(time<=origin+600)
    sweeps=[]
    for stride in (2,10,20,120):
        for phase in range(stride):
            ids=np.arange(phase,len(time),stride)
            tt=time[ids];yy=y[ids]
            ev=np.flatnonzero(window & (time>=tt[0]) & (time<=tt[-1]))
            if len(ev)==0:continue
            query=time[ev]
            linear=np.interp(query,tt,yy)
            prior=np.searchsorted(tt,query,side='right')-1
            causal=yy[prior]
            row={'source':'transient_response.csv','stride':stride,'phase_samples':phase,
                 'spacing_s':stride*dtslow,'n_compared':len(ev),
                 'offline_linear_mae_to_same_dense_trace':float(np.mean(abs(linear-y[ev]))),
                 'previous_sample_hold_mae_to_same_dense_trace':float(np.mean(abs(causal-y[ev]))),
                 'offline_linear_uses_later_samples':True,'is_model_forecast_error':False}
            sweeps.append(row)
    ks=np.array([ia+120*k for k in (-1,0,1,2,3)])
    f3={}
    if (ks>=0).all() and (ks<len(time)).all():
        ix=ia-1
        reconstruction=float(np.interp(time[ix],time[ks],y[ks]))
        f3={'source_rule':'first_ON_stored_index + 120*k, k=-1,0,1,2,3; matches prior F3',
            'at_pre_ON_record_time':float(time[ix]),'reference_dense_value':float(y[ix]),
            'offline_linear_value':reconstruction,'difference_not_prediction_error':reconstruction-float(y[ix]),
            'uses_future_samples':True}
    write_csv(out/'fixed_tau_residuals.csv',residual_rows)
    write_csv(out/'same_trace_sampling_phases.csv',sweeps)
    report={'status':'COMPUTED_DESCRIPTIVE_AUDIT','inputs_sha256':before,
            'fast_n':len(fast['time']),'fast_dt':dtfast,'slow_n':len(time),'slow_dt':dtslow,
            'fixed_tau_diagnostics':fitted,'f3_reconstruction':f3,
            'cross_run_comparison':{'same_trajectory':False,
                'fast_first_ON_record_offset_s':float(fast['time'][a]-fast['time'][0]),
                'slow_first_ON_record_offset_s':float(time[ia]-time[0]),
                'fast_pre_ON_zone_K':float(fast['reaTZon_y'][a-1]),
                'slow_pre_ON_zone_K':float(slow['reaTZon_y'][ia-1]),
                'causal_explanation_identified':False},
            'limits':['Previously exposed diagnostic simulation only; no independent evaluation.',
                      'Endpoint changes are not residuals or isolated slow-mode amplitudes.',
                      'Fixed-tau curve fits do not identify physical time constants.',
                      'Subsampling/interpolation depends on phase; it is not a model forecast.',
                      'Forecast target/horizon are NOT selected by this program.'],
            'source_unchanged':all(file_hash(p)==before[n] for n,p in paths.items())}
    require(report['source_unchanged'],'Input data unexpectedly changed')
    write_json(out/'BOPTEST_AUDIT.json',report)
    return report
