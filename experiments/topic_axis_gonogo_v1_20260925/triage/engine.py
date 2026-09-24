"""One bounded pipeline per topic. Failure in one topic never turns others into no-go.
No new PEFT is designed here. Every topic evaluates F0, regardless of ridge success.
"""
from __future__ import annotations
import importlib,os,time,traceback
from pathlib import Path
import numpy as np
from .common import *
from .network import Fetcher
from .statistics import tune_ridge,predict_ridge,group_mean,analyze,Ridge,bootstrap_mean
from .predictors import Native
from .coordinates import rotated_cases,rotate


def validate_splits(data):
    for name in ('train','cal','test'):
        if name not in data or len(data[name])<20:raise Blocked('INSUFFICIENT_DATA_SUPPORT_'+name)
        ids=[c.uid for c in data[name]];require(len(ids)==len(set(ids)),'Duplicate case ID')
        for c in data[name]:c.validate()
    require(not set(c.uid for c in data['train'])&set(c.uid for c in data['test']),'Train/test case overlap')

def choose_simple(cal,preds):
    vals={k:float(np.mean(list(group_mean([normalized_error(c,p) for c,p in zip(cal,v)],[c.group for c in cal]).values()))) for k,v in preds.items() if k!='F0'}
    require(vals and all(np.isfinite(v) for v in vals.values()),'No valid CAL baseline')
    best=min(vals,key=lambda k:(vals[k],k))
    return best,vals

def seal_value(data,models,history,best,cal_values,config,adj_hash):
    tr=np.array([c.score for c in data['train']]);tr=tr[np.isfinite(tr)]
    return {'created':utc(),'scope':'frozen before TEST forecast scoring; raw data acquisition may include held-out values',
       'config_sha256':hash_obj(config),'strata_cuts_train':np.quantile(tr,[1/3,2/3]),
       'ridge_hashes':{k:m.identity() for k,m in models.items()},'alpha_selection_cal':history,
       'selected_simple_cal':best,'baseline_selection_scores_cal':cal_values,'nuisance_model_cal_hash':adj_hash,
       'test_identifiers_predeclared':[c.uid for c in data['test']],
       'test_labels_used_for_selection':False,'new_peft_or_lora_fit':0}

def coordinate_stress(native,cases,base_preds,canonical_preds,config):
    # The rotations are matched representation interventions, not new independent data.
    rows=[];per_scene={};equiv=[]
    for angle in (45.,90.,135.):
        rot=rotated_cases(cases,angle);preds=native.predict(rot)
        for c,rc,base,cp,p in zip(cases,rot,base_preds,canonical_preds,preds):
            defect=float(np.linalg.norm(p-rotate(base,angle),axis=1).mean()/c.scale)
            penalty=normalized_error(rc,p)-normalized_error(c,base)
            key=c.group;per_scene.setdefault(key,[]).append(penalty);equiv.append(defect)
            rows.append({'uid':c.uid,'group':c.group,'yaw':angle,'equivariance_defect':defect,'paired_error_penalty':penalty,
                         'canonical_baseline_error':normalized_error(c,cp),'raw_base_error':normalized_error(c,base)})
    ci=bootstrap_mean([np.mean(v) for v in per_scene.values()],config['alpha_family'])
    status='INCONCLUSIVE_ROTATION_STRESS'
    if ci['n_groups']>=config['min_contrast_groups']:
        if ci['lower']>0 and np.mean(equiv)>1e-3:status='GO_REPRESENTATION_STRESS_PILOT'
        elif ci['upper']<=0 or max(equiv,default=0)<1e-5:status='NO_GO_TESTED_ROTATION_STRESS'
    return {'status':status,'mean_equivariance_defect':float(np.mean(equiv)),
         'paired_penalty_by_scene':ci,'rotations_are_independent_samples':False,
         'intrinsic_motion_complexity_claimed':False,
         'scope':'mean fixed-yaw penalty, not worst-angle selected after inspecting labels; only five scenes'},rows

def main_topic(topic,repo,cache,public,config):
    public=Path(public);public.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    write_json(public/'STATUS.json',{'status':'STARTED','topic':topic,'utc':utc()})
    fetch=Fetcher(Path(cache)/topic,public,max_bytes=config['download_bytes_per_topic'],seconds=config['download_seconds'])
    module=importlib.import_module('triage.'+topic)
    data=module.prepare(fetch,public,config);validate_splits(data)
    write_json(public/'WINDOW_COUNTS.json',{k:{'windows':len(v),'groups':len(set(c.group for c in v))} for k,v in data.items()})
    models={};hist={};cal_preds={'SIMPLE':[c.simple for c in data['cal']]}
    for name,struct in (('RIDGE_RAW',False),('RIDGE_STRUCTURE',True)):
        models[name],hist[name]=tune_ridge(data['train'],data['cal'],config['alphas'],struct)
        cal_preds[name]=predict_ridge(models[name],data['cal'],struct)
    native=None;backend_failure=None
    try:
        native=Native(config['device']);cal_preds['F0']=native.predict(data['cal'])
        if topic=='coordinates':cal_preds['CANONICAL_F0']=native.predict(data['cal'],True)
    except Blocked as e:backend_failure=str(e)
    best,cal_values=choose_simple(data['cal'],cal_preds)
    adjhash=None
    if 'F0' in cal_preds:
        ce=np.array([normalized_error(c,p) for c,p in zip(data['cal'],cal_preds['F0'])]);ok=np.isfinite(ce)
        if ok.sum()<12:raise Blocked('INSUFFICIENT_CAL_VALID_TARGETS')
        adj=Ridge().fit(np.array([c.nuisance for c in data['cal']])[ok],np.log1p(ce[ok,None]),10.);adjhash=adj.identity()
    seal=seal_value(data,models,hist,best,cal_values,config,adjhash)
    write_json(public/'PRE_TEST_SEAL.json',seal);sealhash=hash_file(public/'PRE_TEST_SEAL.json')
    pred={'SIMPLE':[c.simple for c in data['test']]}
    for name,struct in (('RIDGE_RAW',False),('RIDGE_STRUCTURE',True)):
        pred[name]=predict_ridge(models[name],data['test'],struct)
    if native is not None:
        pred['F0']=native.predict(data['test'])
        if topic=='coordinates':pred['CANONICAL_F0']=native.predict(data['test'],True)
    report,errors=analyze(data['train'],data['cal'],data['test'],pred,cal_preds,best,config)
    if adjhash:
        require(report['adjusted_F0_contrast']['adjustment_model_hash']==adjhash,'CAL nuisance model changed after seal')
    score_rows=[]
    for i,c in enumerate(data['test']):
        row={'uid':c.uid,'group':c.group,'score':c.score,'alternate':c.alternate,'error_scale_from_context':c.scale}
        row.update({f'nuisance_{j}':float(v) for j,v in enumerate(c.nuisance)})
        row.update({k:float(e[i]) for k,e in errors.items()});score_rows.append(row)
    write_csv(public/'SCORE_ROWS.csv',score_rows)
    if topic=='coordinates' and native is not None:
        stress,rows=coordinate_stress(native,data['test'],pred['F0'],pred['CANONICAL_F0'],config)
        write_csv(public/'PAIRED_ROTATION_STRESS.csv',rows);report['coordinate_stress']=stress
    require(hash_file(public/'PRE_TEST_SEAL.json')==sealhash,'Seal changed')
    report.update(topic=topic,source_kind='real_public_dataset',backend_error=backend_failure,
        native_backend=None if native is None else getattr(native,'info',{'scope':'test_double_only'}),
        model_operations=None if native is None else {'actual_forward_calls':native.forwards,'forecast_windows':native.windows,'optimizer_updates':0},
        ridge_coefficient_fits=2*len(config['alphas'])+(2 if adjhash else 0),
        wall_seconds=time.monotonic()-start,seal_sha256=sealhash,
        selection_limit='GO means a diagnostic axis deserves further LoRA tests, not method novelty or practical value',
        estimator_agreement_threshold_status='predeclared heuristic screening value, NOT literature-established cutoff')
    write_json(public/'RESULT.json',report)
    # Only aggregated scores are public. Original waveforms, locations and trajectories remain in cache.
    private=Path(cache)/'private_predictions'/topic;private.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(private/'predictions.npz',**{k:np.array(v) for k,v in pred.items()})
    if native:native.close()
    write_json(public/'STATUS.json',{'status':'COMPLETED','topic':topic,'axis_status':report['axis_status'],'utc':utc()})
    return report
