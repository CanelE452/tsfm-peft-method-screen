"""Origin/episode-balanced descriptive evaluation with a raw-prediction replay.

No p-values or bootstrap CI from two DEV episodes. Overlapping forecast windows
are never counted as independent replications. Reporting panels may overlap.
"""
from __future__ import annotations
from collections import defaultdict
import csv,gzip
from pathlib import Path
import numpy as np
from .common import require,write_gzip_csv,write_csv,write_json

MODEL_NAMES = ['CURRENT_PLAN_TUNED','CURRENT_PLAN_A1',
               'HISTORY_SUMMARY_PLAN_TUNED','HISTORY_SUMMARY_PLAN_A1',
               'HISTORY_LAGS_PLAN_TUNED','HISTORY_LAGS_PLAN_A1','PERSISTENCE']


def metrics(y,p):
    a=np.abs(p-y)
    return {'mae_K':float(a.mean()),'rmse_K':float(np.sqrt(((p-y)**2).mean())),
            'p90_abs_K':float(np.quantile(a,.9))}


def export_evaluation(public,stage,cc,prepared,predictions,alphas):
    require(set(predictions)==set(MODEL_NAMES),'Missing prediction arm')
    for name,p in predictions.items():
        require(p.shape==prepared.Y.shape and np.isfinite(p).all(),'Prediction shape/value error: '+name)
    prefix=f"{stage}_{cc['name']}_ep{prepared.episode:02d}"
    fields=['origin_time','lead_s','target_time','target_K']+MODEL_NAMES
    def raw_rows():
        for i,t in enumerate(prepared.origins):
            for j in range(prepared.Y.shape[1]):
                row={'origin_time':float(t),'lead_s':(j+1)*cc['sample_s'],
                     'target_time':float(prepared.target_times[i,j]),'target_K':float(prepared.Y[i,j])}
                row.update({name:float(p[i,j]) for name,p in predictions.items()})
                yield row
    path=Path(public)/'predictions'/f'{prefix}.csv.gz'
    write_gzip_csv(path,raw_rows(),fields)
    origin_errors={name:np.mean(np.abs(p-prepared.Y),axis=1) for name,p in predictions.items()}
    origin_rows=[]
    for i,t in enumerate(prepared.origins):
        row={'origin_time':float(t), **{k:bool(v[i]) for k,v in prepared.masks.items()},
             **prepared.metadata[i], **{name:float(v[i]) for name,v in origin_errors.items()}}
        origin_rows.append(row)
    origin_fields=list(origin_rows[0])
    write_csv(Path(public)/'origins'/f'{prefix}.csv',origin_rows,origin_fields)
    # A separately read CSV re-computes the per-origin/overall MAE; never trust only the summary.
    replay=defaultdict(lambda:defaultdict(list))
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            t=float(row['origin_time']);y=float(row['target_K'])
            for name in MODEL_NAMES: replay[t][name].append(abs(float(row[name])-y))
    require(len(replay)==len(prepared.origins),'Replay origin count mismatch')
    max_replay_diff=0.
    for i,t in enumerate(prepared.origins):
        for name in MODEL_NAMES:
            require(len(replay[float(t)][name])==prepared.Y.shape[1],'Replay horizon count mismatch')
            diff=abs(float(np.mean(replay[float(t)][name]))-origin_errors[name][i])
            require(diff<=1e-10,'Saved prediction scores do not reproduce in-memory scores')
            max_replay_diff=max(max_replay_diff,diff)
    score_rows=[]
    for panel,m in prepared.masks.items():
        n=int(m.sum())
        for name,p in predictions.items():
            record={'stage':stage,'config':cc['name'],'episode':prepared.episode,'panel':panel,'model':name,
                    'n_origins':n,'n_forecast_points':n*prepared.Y.shape[1],
                    'alpha':alphas.get(name),'status':'OBSERVED' if n else 'NO_PANEL_SUPPORT'}
            if n:record.update(metrics(prepared.Y[m],p[m]))
            else:record.update({'mae_K':None,'rmse_K':None,'p90_abs_K':None})
            score_rows.append(record)
    # Per-origin paired effects; not a test of statistical significance or causality.
    effects=[]
    for mode in ('TUNED','A1'):
        base=f'CURRENT_PLAN_{mode}'
        for history in ('HISTORY_SUMMARY_PLAN','HISTORY_LAGS_PLAN'):
            hist=f'{history}_{mode}'
            for panel,m in prepared.masks.items():
                n=int(m.sum());d=origin_errors[base][m]-origin_errors[hist][m]
                cur=float(origin_errors[base][m].mean()) if n else None
                delta=float(d.mean()) if n else None
                effects.append({'stage':stage,'config':cc['name'],'episode':prepared.episode,
                    'panel':panel,'comparison':hist+'__vs__'+base,'n_origins':n,
                    'mean_MAE_reduction_K':delta,
                    'relative_MAE_reduction_pct':100*delta/cur if n and cur>0 else None,
                    'origin_positive_fraction':float((d>0).mean()) if n else None,
                    'status':'OBSERVED' if n else 'NO_PANEL_SUPPORT'})
    write_json(Path(public)/'replays'/f'{prefix}.json',{'status':'PASS','max_origin_MAE_diff':max_replay_diff,
              'n_origins':len(prepared.origins),'n_horizon_points':prepared.Y.shape[1],
              'window_overlap_is_not_independent_replication':True})
    return score_rows,effects


def summarize(public,scores,effects):
    write_csv(Path(public)/'SCORES.csv',scores,list(scores[0]))
    write_csv(Path(public)/'PAIRED_EFFECTS.csv',effects,list(effects[0]))
    groups=defaultdict(list)
    for row in effects:
        groups[(row['stage'],row['config'],row['panel'],row['comparison'])].append(row)
    summary=[]
    for (stage,cc,panel,comparison),rows in sorted(groups.items()):
        supported=[x for x in rows if x['n_origins']>0]
        expected=2 if stage=='DEV' else 3
        summary.append({'stage':stage,'config':cc,'panel':panel,'comparison':comparison,
            'episodes_with_support':len(supported),'expected_episodes':expected,
            'positive_episodes':sum(x['mean_MAE_reduction_K']>0 for x in supported),
            'mean_episode_MAE_reduction_K':float(np.mean([x['mean_MAE_reduction_K'] for x in supported])) if supported else None,
            'mean_episode_relative_reduction_pct':float(np.mean([x['relative_MAE_reduction_pct'] for x in supported if x['relative_MAE_reduction_pct'] is not None])) if supported and all(x['relative_MAE_reduction_pct'] is not None for x in supported) else None,
            'min_origins_per_supported_episode':min(x['n_origins'] for x in supported) if supported else 0,
            'inferential_status':'DESCRIPTIVE_ONLY_NO_SIGNIFICANCE_OR_EQUIVALENCE_CLAIM'})
    write_json(Path(public)/'EFFECT_SUMMARY.json',summary)
    return summary
