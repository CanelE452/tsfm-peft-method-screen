"""Nested expanding-prefix Ridge comparisons. DEV never chooses alpha or a task.

Bounded: 5 tasks x 3 arms x (4 inner folds x 7 alphas + <=8 refits) = <=540
ridge coefficient fits. Each path shares one SVD. All failures count as attempts.
"""
from __future__ import annotations
import os,platform,sys,time,traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
from .common import require,write_json,read_json,write_csv,file_hash,canonical_hash,utc,Ledger
from .data import PinnedSource,prepare,ARMS
from .ridge import ridge_path,choose_alpha,RidgeModel
from .evaluate import export_evaluation,summarize


def concatenate(parts,ids,arm):
    require(ids==sorted(ids) and len(set(ids))==len(ids),'Non-chronological training episode list')
    return np.concatenate([parts[e].X[arm] for e in ids]),np.concatenate([parts[e].Y for e in ids])


def assert_temporal(parts,train_ids,hold):
    require(hold not in train_ids and max(train_ids)<hold,'Future/held-out training episode detected')
    end=max(parts[e].audit['source_end'] for e in train_ids)
    start=parts[hold].audit['source_start']
    require(end<start,'Training/evaluation episode time ranges overlap')


def train_task(cc,parts,config,public,private,ledger):
    grid=config['alpha_grid'];outer=config['outer_train_holdouts'];tr=config['train_episodes']
    predictions={h:{'PERSISTENCE':parts[h].persistence} for h in outer}
    alphas={h:{} for h in outer}
    final_models={}; final_meta={}; cv_rows=[]; selection_rows=[]
    for arm in ARMS:
        cv={}
        # Each validation episode k is predicted only from 0..k-1.
        for hold in tr[1:]:
            train_ids=[e for e in tr if e<hold]
            assert_temporal(parts,train_ids,hold)
            X,Y=concatenate(parts,train_ids,arm)
            models=ridge_path(X,Y,grid,ledger,{'stage':'INNER_TRAIN_ONLY','config':cc['name'],'arm':arm,
                                            'train_episodes':train_ids,'validation_episode':hold})
            fold={}
            for alpha,m in models.items():
                p=m.predict(parts[hold].X[arm]);score=float(np.mean(np.abs(p-parts[hold].Y)))
                fold[alpha]=score
                cv_rows.append({'config':cc['name'],'arm':arm,'train_episodes':','.join(map(str,train_ids)),
                    'validation_episode':hold,'alpha':alpha,'MAE_K':score,'train_rows':len(X),
                    'validation_origins':len(p),'n_features':X.shape[1]})
            cv[hold]=fold
        for hold in outer:
            used=[k for k in sorted(cv) if k<hold]
            require(used and max(used)<hold,'Alpha selection saw outer evaluation labels')
            alpha,curve=choose_alpha([cv[k] for k in used],grid)
            train_ids=[e for e in tr if e<hold];X,Y=concatenate(parts,train_ids,arm)
            models=ridge_path(X,Y,sorted(set([alpha,float(config['fixed_alpha_control'])])),ledger,
                 {'stage':'OUTER_TRAIN_REFIT','config':cc['name'],'arm':arm,'train_episodes':train_ids,'holdout_episode':hold})
            for suffix,a in [('TUNED',alpha),('A1',float(config['fixed_alpha_control']))]:
                key=arm+'_'+suffix;predictions[hold][key]=models[a].predict(parts[hold].X[arm]);alphas[hold][key]=a
            selection_rows.append({'config':cc['name'],'arm':arm,'purpose':'OUTER_TRAIN',
                'evaluation_episode':hold,'tuning_validation_episodes':used,'selected_alpha':alpha,
                'training_episodes':train_ids,'mean_validation_MAE_by_alpha':{str(a):v for a,v in curve.items()},
                'alpha_on_grid_boundary':alpha in (min(grid),max(grid))})
        # Fit once for DEV, choosing every hyperparameter using TRAIN folds only.
        alpha,curve=choose_alpha([cv[k] for k in tr[1:]],grid)
        X,Y=concatenate(parts,tr,arm)
        models=ridge_path(X,Y,sorted(set([alpha,float(config['fixed_alpha_control'])])),ledger,
              {'stage':'FINAL_TRAIN_REFIT','config':cc['name'],'arm':arm,'train_episodes':tr})
        for suffix,a in [('TUNED',alpha),('A1',float(config['fixed_alpha_control']))]:
            name=arm+'_'+suffix
            path=Path(private)/'models'/f"{cc['name']}__{name}.npz";path.parent.mkdir(parents=True,exist_ok=True)
            models[a].save(path)
            restored=RidgeModel.load(path)
            require(np.array_equal(restored.predict(X[:2]),models[a].predict(X[:2])), 'Ridge saved state mismatch')
            final_models[name]=restored
            final_meta[name]={'alpha':a,'file':str(path.relative_to(private)),'sha256':file_hash(path),
                              'n_train':len(X),'n_features':X.shape[1],'n_target_steps':Y.shape[1]}
        selection_rows.append({'config':cc['name'],'arm':arm,'purpose':'FINAL_TRAIN_FOR_DEV',
                'evaluation_episode':None,'tuning_validation_episodes':tr[1:],'selected_alpha':alpha,
                'training_episodes':tr,'mean_validation_MAE_by_alpha':{str(a):v for a,v in curve.items()},
                'alpha_on_grid_boundary':alpha in (min(grid),max(grid))})
    path=Path(public)/'selection';path.mkdir(parents=True,exist_ok=True)
    write_csv(path/f"{cc['name']}_CV_GRID.csv",cv_rows,list(cv_rows[0]))
    write_json(path/f"{cc['name']}_SELECTION.json",selection_rows)
    scores=[];effects=[]
    for hold in outer:
        s,e=export_evaluation(public,'TRAIN_FORWARD',cc,parts[hold],predictions[hold],alphas[hold])
        scores+=s;effects+=e
    return final_models,final_meta,scores,effects,selection_rows


def run_engine(repo,package,public,private,config=None,source=None):
    started=time.monotonic();public=Path(public);private=Path(private)
    public.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
    config=config or read_json(Path(package)/'RUN_CONFIG.json')
    ledger=Ledger(public/'FIT_EVENTS.jsonl',config['budgets'])
    source=source or PinnedSource(repo,config,public)
    manifest={'started':utc(),'kind':config['purpose'],'config_sha256':canonical_hash(config),
              'source_commit':config['source_commit'],'source_result':config['source_result'],
              'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
              'results_are_retrospective':True,'reserve_open_or_scoring_allowed':False,
              'future_plan_provenance':'deterministic plan reconstruction, not operational issue-time log',
              'alphas':config['alpha_grid'],'scope':'Ridge only; no simulator, Chronos, torch, LoRA or TMD',
              'prior_negative_result_preserved':True,'status':'STARTED'}
    write_json(public/'RUN_MANIFEST.json',manifest)
    final={};final_meta={};scores=[];effects=[];schemas={};train_hashes={}
    try:
        require(config['train_episodes']==[0,1,2,3,4] and config['dev_episodes']==[5,6], 'Fixed split changed')
        require(config['never_open_episodes']==[7], 'Reserve contract changed')
        # No loading DEV until ALL model families and candidate tasks are fixed below.
        training={e:source.load(e) for e in config['train_episodes']}
        for cc in config['candidates']:
            print('TRAIN',cc['name'],flush=True)
            parts={e:prepare(v,cc,config) for e,v in training.items()}
            schemas[cc['name']]=parts[0].schema
            write_json(public/'schemas'/f"{cc['name']}.json",{'features':parts[0].schema,'audit':[p.audit for p in parts.values()],
                      'panels':'operational command-change strata, NOT identified physical settling'})
            fm,meta,s,e,selection=train_task(cc,parts,config,public,private,ledger)
            final[cc['name']]=fm;final_meta[cc['name']]=meta;scores+=s;effects+=e
            require(sum(p.stat().st_size for p in private.rglob('*') if p.is_file()) <= config['budgets']['max_private_mib']*(1<<20), 'Private model-size budget exceeded')
        seal={'utc':utc(),'config_sha256':canonical_hash(config),'train_episodes':config['train_episodes'],
              'target_selection':'NONE_ALL_FIVE_RETAINED','model_states':final_meta,
              'source_train_hashes':{str(k):v for k,v in source.touched.items()},
              'alpha_selection':'TRAIN-only expanding-prefix validation, equal-episode ALL-origin MAE',
              'selection_files':{p.name:file_hash(p) for p in sorted((public/'selection').iterdir())},
              'dev_must_not_redefine_panels_models_or_thresholds':True}
        seal['seal_sha256']=canonical_hash(seal)
        write_json(public/'PRE_DEV_SEAL.json',seal)
        seal_file_hash=file_hash(public/'PRE_DEV_SEAL.json')
        source.enable_dev(public/'PRE_DEV_SEAL.json')
        for ep in config['dev_episodes']:
            episode=source.load(ep)
            for cc in config['candidates']:
                print('DEV',ep,cc['name'],flush=True)
                parts=prepare(episode,cc,config)
                require(parts.schema==schemas[cc['name']],'DEV feature schema differs from TRAIN')
                pred={'PERSISTENCE':parts.persistence};aa={}
                for name,m in final[cc['name']].items():
                    arm=name.rsplit('_',1)[0]
                    pred[name]=m.predict(parts.X[arm]);aa[name]=m.alpha
                s,e=export_evaluation(public,'DEV',cc,parts,pred,aa)
                scores+=s;effects+=e
        require(file_hash(public/'PRE_DEV_SEAL.json')==seal_file_hash,'Pre-DEV seal changed after evaluation')
        for cc,states in final_meta.items():
            for name,meta in states.items():
                require(file_hash(private/meta['file'])==meta['sha256'],'Model mutated after pre-DEV seal')
        summary=summarize(public,scores,effects)
        source.verify_unchanged()
        require(not (set(source.opened_episodes)&set(config['never_open_episodes'])),'Reserve was accessed')
        result={'status':'COMPLETED_BOUNDED_RECHECK','research_verdict':'NOT_AUTOMATICALLY_DECIDED',
                'primary_question':'Does bounded past-command information add out-of-episode predictive value when known future plans are shared?',
                'data_scope':'same building simulator; TRAIN was already examined in v1; DEV is development, NOT an untouched final test',
                'old_NO_HISTORY_SIGNAL_replaced_by':'No broad information-existence claim is emitted.',
                'next_action':'Return pushed commit to ChatGPT for review; no auto-run of PEFT or other candidates.',
                'reserve_accessed':False,'neural_evaluation':False,'TMD_computed':False,
                'counts':ledger.counts(),
                'dev_tuned_primary_panels':[x for x in summary if x['stage']=='DEV' and x['panel'] in ('ALL','RECENT_OR_PLANNED_CHANGE') and 'TUNED__vs__' in x['comparison']]}
        write_json(public/'FINAL_DECISION.json',result)
        lines=['# Bounded information-contract recheck v2','',
            'COMPLETED_BOUNDED_RECHECK. Descriptive retrospective comparison, not a PEFT success/failure claim.',
            '', 'Future plans are identical across arms. CURRENT intentionally removes past-command information; this is an information ablation, not an equal-information PEFT comparison.',
            'History summaries and raw lags use the same bounded raw history with different representations.',
            'All five tasks and fixed-alpha controls are retained. No DEV-based task/alpha selection.',
            'Command-change panels are operational time windows, not measured physical tau or validated complexity.',
            'DEV episodes are 5/6. Episode 7 is not opened or hashed. Two DEV episodes do not justify population uncertainty estimates.',
            '', '| task | comparison | panel | DEV episodes supported | positive episodes | mean MAE reduction K |',
            '|---|---|---|---:|---:|---:|']
        for row in result['dev_tuned_primary_panels']:
            val=row['mean_episode_MAE_reduction_K'];s='NA' if val is None else f'{val:.9g}'
            lines.append(f"| {row['config']} | {row['comparison']} | {row['panel']} | {row['episodes_with_support']} | {row['positive_episodes']} | {s} |")
        lines+=['','A positive number favors the history model in the specified comparison only.',
                'See predictions/, origins/, SCORES.csv, PAIRED_EFFECTS.csv, EFFECT_SUMMARY.json and PRE_DEV_SEAL.json for exact support.',
                'No absolute physical-error sufficiency threshold, significance claim, new complexity index or automatic continuation decision is defined.']
        (public/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        manifest['status']='COMPLETED'
        return result
    except Exception as exc:
        manifest['status']='FAILED'
        write_json(public/'FAILURE.json',{'utc':utc(),'error_type':type(exc).__name__,'error':str(exc),
                   'traceback':traceback.format_exc(),'counts_including_failed_attempts':ledger.counts(),
                   'opened_episodes':source.opened_episodes,'scientific_failure_claimed':False})
        raise
    finally:
        manifest.update({'finished':utc(),'elapsed_seconds':time.monotonic()-started,'actual_counts':ledger.counts(),
                         'opened_episodes':source.opened_episodes,'unit_tests_excluded_from_fit_counts':True})
        write_json(public/'RUN_MANIFEST.json',manifest)
