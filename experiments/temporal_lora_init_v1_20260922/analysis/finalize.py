"""CPU-only analysis of the sealed, fully saved experiment; no new training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
from evaluation import load, predpath, metrics, adjust, calibrate, score
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def csv(name, rows):
    pd.DataFrame(rows).to_csv(RESULTS / name, index=False)


def improvement(a, b):
    return float(100 * (1 - np.mean(a) / np.mean(b)))


def bootstrap(a, b):
    # Inputs: repeat seed x origin, after equal series/lead weighting.
    assert a.shape == b.shape and a.ndim == 2
    rng = np.random.default_rng(92343)
    n = a.shape[1]
    effects = []
    for _ in range(2000):
        starts = rng.integers(0, n - 7 + 1, size=(n + 6) // 7)
        ix = np.concatenate([np.arange(s, s + 7) for s in starts])[:n]
        effects.append(improvement(a[:, ix], b[:, ix]))
    return np.quantile(effects, [.025, .975]).tolist()


def main():
    assert read(RESULTS/'ALL_TEST_SAVED.json')['status'] == 'COMPLETE'
    checked = 0
    for seal, root, key in [('SOURCE_SEAL.json', EXP, 'files'),
                            ('SOURCE_SEAL.json', RESULTS, 'artifacts'),
                            ('SELECTION_SEAL.json', RESULTS, 'files'),
                            ('PREDICTIONS_MANIFEST.json', ROOT, 'files')]:
        for name, h in read(RESULTS/seal)[key].items():
            assert sha(root/name) == h, name
            checked += 1
    assert not list(RESULTS.glob('ERROR_*.json'))
    event('test_analysis_started', verified_hashes=checked)
    fits = [read(p) for p in sorted((RESULTS/'fits').glob('*.json'))]
    assert len(fits) == 16
    selection = read(RESULTS/'SELECTION.json')
    ledger = [json.loads(x) for x in (RESULTS/'UPDATE_LEDGER.jsonl').read_text().splitlines()]
    intents = [(x['key'], x['step'], x['phase']) for x in ledger if x['kind']=='intent']
    commits = [(x['key'], x['step'], x['phase']) for x in ledger if x['kind']=='commit']
    assert intents == commits and len(set(intents)) == 8200
    assert sum(x[2]=='main' for x in intents)==8192
    assert sum(x[2]=='smoke' for x in intents)==8
    rows, leadrows, seriesrows, originrows, curves, resources, targets = [], [], [], [], [], [], []
    origin_scores = {}
    audits = 0
    for source in SOURCES:
        base = selection[source]['baseline']
        calcbase = {arm: np.mean([f['curves']['128'] for f in fits if f['source']==source and f['arm']==arm]) for arm in ['RANDOM','PCA']}
        assert min(calcbase, key=calcbase.get)==base
        f0 = load(predpath(source, 0, 'F0', 'TEST', 0))
        f0cal = calibrate(load(predpath(source, 0, 'F0', 'CAL', 0)))
        assert all(f0cal[k]==selection[source]['f0_cal'][k] for k in ['alpha','beta','grid_evaluations'])
        assert abs(f0cal['cal_pinball']-selection[source]['f0_cal']['cal_pinball'])<1e-14
        selected_fits = [f for f in fits if f['source']==source]
        entries = [(f, f['arm'], f['seed']) for f in selected_fits] + [(None,'F0',0)]
        for f, arm, seed in entries:
            if f:
                assert f['frozen_unchanged'] and f['steps']==512 and len(f['trace'])==512
                assert sum(k==f['key'] and p=='main' for k,s,p in intents)==512
                assert min(STEPS, key=lambda s:(f['curves'][str(s)],s))==f['selected']
                c = calibrate(load(predpath(source,seed,arm,'CAL',f['selected'])))
                assert all(c[k]==f['cal'][k] for k in ['alpha','beta','grid_evaluations'])
                assert abs(c['cal_pinball']-f['cal']['cal_pinball'])<1e-14
                threshold = next(x for x in selected_fits if x['arm']=='RANDOM' and x['seed']==seed)['curves']['512']*1.01
                reached = [s for s in STEPS if f['curves'][str(s)]<=threshold]
                first = min(reached) if reached else None
                status = 'NO_ADAPTATION_HEADROOM' if first==0 else 'REACHED' if first else 'CENSORED'
                targets.append(dict(source=source,arm=arm,seed=seed,threshold=threshold,first_step=first,time_seconds=f['times'][str(first)] if first is not None else None,status=status))
                for s in STEPS:
                    val=load(predpath(source,seed,arm,'VAL',s))
                    assert np.isclose(score(val),f['curves'][str(s)],rtol=0,atol=1e-14)
                    if s==0:
                        assert np.array_equal(val['q'],load(predpath(source,0,'F0','VAL',0))['q'])
                    curves.append(dict(source=source,arm=arm,seed=seed,step=s,val_pinball=f['curves'][str(s)],charged_seconds=f['times'][str(s)]))
                resources.append({k:f[k] for k in ['source','arm','seed','steps','selected','prep_seconds','model_load_seconds','fit_wall_seconds','peak_allocated','peak_reserved','trainable_parameters']} | dict(adaptation_seconds512=f['times']['512']))
                variants=[('raw128',128,False),('selected_raw',f['selected'],False),('selected_cal',f['selected'],True)]
                audits+=1
            else:
                c=f0cal
                variants=[('raw128',0,False),('selected_raw',0,False),('selected_cal',0,True)]
            for variant, step, corrected in variants:
                p=load(predpath(source,seed,arm,'TEST',step))
                for k in ['pairs','y','sigma']:assert np.array_equal(p[k],f0[k])
                q=adjust(p,c) if corrected else p['q']
                m=metrics(q,p['y'],p['sigma'])
                row=dict(source=source,arm=arm,seed=seed,variant=variant,step=step)
                rows.append(row | {k:float(v.mean()) for k,v in m.items()})
                for h in range(64):leadrows.append(row | dict(lead=h+1) | {k:float(v[:,h].mean()) for k,v in m.items()})
                for s in np.unique(p['pairs'][:,0]):
                    mask=p['pairs'][:,0]==s
                    seriesrows.append(row | dict(series_index=int(s)) | {k:float(v[mask].mean()) for k,v in m.items()})
                perorigin=[]
                for o in np.unique(p['pairs'][:,1]):
                    mask=p['pairs'][:,1]==o
                    value=float(m['pinball'][mask].mean());perorigin.append(value)
                    originrows.append(row | dict(origin=int(o),pinball=value))
                origin_scores[source,arm,seed,variant]=np.array(perorigin)
    csv('SCORES.csv',rows);csv('LEAD_SCORES.csv',leadrows);csv('SERIES_SCORES.csv',seriesrows);csv('ORIGIN_SCORES.csv',originrows)
    csv('CURVES.csv',curves);csv('RESOURCES.csv',resources);csv('TIME_TO_TARGET.csv',targets)
    csv('FIT_LEDGER.csv',[dict(key=f['key'],main_updates=f['steps'],frozen_unchanged=f['frozen_unchanged'],selected=f['selected'],initial_sha=f['initial_sha'],selected_sha=f['selected_sha']) for f in fits])
    effects=[];seed_effects=[]
    for source in SOURCES:
        for variant in ['raw128','selected_raw','selected_cal']:
            for candidate in ARMS:
                a=np.stack([origin_scores[source,candidate,s,variant] for s in SEEDS])
                for control in ['F0']+ARMS:
                    if candidate==control:continue
                    b=np.stack([origin_scores[source,control,0 if control=='F0' else s,variant] for s in SEEDS])
                    low,high=bootstrap(a,b)
                    effects.append(dict(source=source,variant=variant,candidate=candidate,control=control,candidate_score=float(a.mean()),control_score=float(b.mean()),improvement_pct=improvement(a,b),ci_low=low,ci_high=high))
                    for i,s in enumerate(SEEDS):seed_effects.append(dict(source=source,variant=variant,candidate=candidate,control=control,seed=s,candidate_score=float(a[i].mean()),control_score=float(b[i].mean()),improvement_pct=improvement(a[i],b[i])))
    csv('EFFECTS.csv',effects);csv('SEED_EFFECTS.csv',seed_effects)
    decisions={}
    for source in SOURCES:
        base=selection[source]['baseline']
        e=[x for x in effects if x['source']==source and x['candidate']=='TEMP' and x['variant']=='raw128' and x['control'] in [base,'SHUFFLE']]
        se=[x for x in seed_effects if x['source']==source and x['candidate']=='TEMP' and x['variant']=='raw128' and x['control'] in [base,'SHUFFLE']]
        mechanism=all(x['improvement_pct']>=.5 for x in e) and all(x['improvement_pct']>0 for x in se)
        tt=[x for x in targets if x['source']==source and x['arm'] in ['TEMP','RANDOM']]
        valid=all(x['status']=='REACHED' for x in tt)
        saving=improvement([x['time_seconds'] for x in tt if x['arm']=='TEMP'],[x['time_seconds'] for x in tt if x['arm']=='RANDOM']) if valid else None
        selected=next(x for x in effects if x['source']==source and x['candidate']=='TEMP' and x['control']==base and x['variant']=='selected_cal')
        status=('GO_EFFICIENCY_SCREEN' if valid and saving>=20 and selected['improvement_pct']>=-1 else 'MECHANISM_ONLY_COST_NOT_MET') if mechanism else ('NO_GO_CURRENT_RECIPE' if any(x['improvement_pct']<=0 for x in e) else 'HOLD')
        decisions[source]=dict(status=status,baseline=base,mechanism=mechanism,primary=e,seed_effects=se,time_saving_pct=saving,time_comparison_valid=valid,selected_cal_effect=selected)
    overall='GO_EFFICIENCY_SCREEN' if all(x['status']=='GO_EFFICIENCY_SCREEN' for x in decisions.values()) else 'MECHANISM_ONLY_COST_NOT_MET' if all(x['mechanism'] for x in decisions.values()) else 'NO_GO_CURRENT_RECIPE' if any(x['status']=='NO_GO_CURRENT_RECIPE' for x in decisions.values()) else 'HOLD'
    save(RESULTS/'DECISION.json',dict(overall=overall,datasets=decisions,classification='SCIENTIFIC_SCREEN_RESULT',automatic_followup=False))
    figures=RESULTS/'figures';figures.mkdir(exist_ok=True)
    colors=dict(RANDOM='#555555',PCA='#2377ad',TEMP='#cb453e',SHUFFLE='#b38b24')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,source in enumerate(SOURCES):
        for arm in ARMS:
            ff=[f for f in fits if f['source']==source and f['arm']==arm]
            vals=np.array([[f['curves'][str(s)] for s in STEPS] for f in ff])
            times=np.array([[f['times'][str(s)] for s in STEPS] for f in ff])
            for j,x in enumerate([np.array(STEPS),times.mean(0)]):
                axes[i,j].plot(x,vals.mean(0),'-o',label=arm,color=colors[arm],markersize=4)
                axes[i,j].fill_between(x,vals.min(0),vals.max(0),color=colors[arm],alpha=.1)
        for j in range(2):
            axes[i,j].set(title=source,xlabel='Optimizer updates' if j==0 else 'Adaptation seconds (including initialization)',ylabel='Validation normalized pinball (lower is better)')
            axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=8)
    fig.suptitle('Fixed checkpoints: mean and range of two seeds');fig.savefig(figures/'learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,source in zip(axes,SOURCES):
        es=[x for x in effects if x['source']==source and x['candidate']=='TEMP' and x['variant']=='raw128']
        for i,e in enumerate(es):
            ax.plot([e['ci_low'],e['ci_high']],[i,i],color='#cb453e',linewidth=3)
            ax.scatter(e['improvement_pct'],i,color='#cb453e',s=60)
            ss=[s['improvement_pct'] for s in seed_effects if s['source']==source and s['candidate']=='TEMP' and s['variant']=='raw128' and s['control']==e['control']]
            ax.scatter(ss,[i-.12,i+.12],color='#444444',s=18)
        ax.axvline(0,color='black',linewidth=.8);ax.set(yticks=range(len(es)),yticklabels=[e['control'] for e in es],xlabel='TEMP improvement at 128 updates (%)',title=source);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Red: paired time-block 95% interval; black: individual seeds');fig.savefig(figures/'test_effects.png',dpi=180);plt.close(fig)
    save(RESULTS/'VERIFICATION.json',dict(status='PASS',verified_input_hashes=checked,main_fits=16,main_updates=8192,smoke_updates=8,unmatched_intents=0,unique_updates=8200,all_fits512=True,checkpoint_and_cal_recomputed=audits,initial_val_exact_f0=True,all_test_pairs_targets_scales_equal=True,source_seal_unchanged=True,selection_seal_unchanged=True,all_test_saved_before_scoring=True,bootstrap=dict(repeats=2000,block_origins=7,seed=92343,joint_series_seeds=True),classification='SCIENTIFIC_SCREEN_RESULT'))
    print(json.dumps(dict(overall=overall,datasets=decisions),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
