"""Reopen evidence audit. Historical verdicts remain unchanged."""
import argparse,csv,json,subprocess
from pathlib import Path
import numpy as np
from tsfm_peft_screen.metrics import score,independent
from tsfm_peft_screen.reproducibility import ROOT,sha,write_json,digest
R=ROOT/'research/reopen_review_20260914'


def read(p):return json.loads(Path(p).read_text())
def writecsv(p,rows):
    with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def gain(a,b):return 100*(b-a)/b


def anchor():
    p=ROOT/'results/anchor_window_study_20260914';old=read(p/'summary.json');ev=read(p/'evaluation.json');seal=read(p/'evaluation_seal.json')
    assert read(p/'verification.json')['completed_fits']==48
    errors=[];index={};rows=[]
    for r in ev:
        f=ROOT/r['prediction_file'];assert sha(f)==r['prediction_sha256']
        with np.load(f,allow_pickle=False) as z:
            v=score(z['prediction'],z['target'],z['scale'])['scaled_2pinball'];i=independent(z['prediction'],z['target'],z['scale'])
        errors.append(max(abs(v-i),abs(v-r['metrics']['scaled_2pinball'])));index[r['dataset'],r['seed'],r['arm']]=r
    assert max(errors)<=1e-10
    for c in old['cells']:
        for seed in [34000,34001]:
            f0=index[c['dataset'],None,'F0']['metrics']['scaled_2pinball']
            plain=index[c['dataset'],seed,'native']['metrics']['scaled_2pinball']
            for arm in ['native','native_anchor']:
                r=index[c['dataset'],seed,arm];loss=r['metrics']['scaled_2pinball'];s=r['selection']
                rows.append(dict(source=c['source'],budget=c['training_windows'],seed=seed,arm=arm,F0_loss=f0,loss=loss,
                    relative_gain_vs_native=gain(loss,plain),gain_vs_F0=gain(loss,f0),selected_lr=s['lr'],selected_step=s['step']))
    writecsv(R/'ANCHOR_EFFECTS.csv',rows)
    budget={b:float(np.mean([c['anchor_gain_percent'] for c in old['cells'] if c['training_windows']==b])) for b in [32,233]}
    macro=float(np.mean(list(budget.values())));interaction=budget[32]-budget[233]
    assert abs(macro-read(ROOT/'results/temporal_transfer_diagnostic_v1/aggregate_replay.json')['source_balanced_macro'])<1e-10
    lines=['# Anchoring retrospective classification','', 'ANCHOR_CONDITIONAL. Original WINDOW_BUDGET_HYPOTHESIS_NOT_SUPPORTED remains unchanged. No repeated fit. Positive macro but Beijing positive, ETTm2 zero, Electricity negative; sparse32 does not show the proposed advantage.','',
        f'Source-balanced macro {macro:+.6f}%; sparse32 {budget[32]:+.6f}%; dense233 {budget[233]:+.6f}%; sparse-minus-dense {interaction:+.6f} percentage points.','',
        '| source/budget | native loss | anchor loss | anchor/native gain | native/F0 gain | anchor/F0 gain |','|---|---:|---:|---:|---:|---:|']
    for c in old['cells']:lines.append(f"| {c['dataset']} | {c['plain_loss']:.9f} | {c['anchor_loss']:.9f} | {c['anchor_gain_percent']:+.6f}% | {c['plain_vs_f0_percent']:+.6f}% | {c['anchor_vs_f0_percent']:+.6f}% |")
    lines+=['','All seed-level losses, F0 references, corrected gains and selections: ANCHOR_EFFECTS.csv. Historical seed_gains_percent was a ratio×100, not a gain; the prior ERRATUM is preserved. 30 original E caches independently replayed here; no new E forecasts.']
    (R/'ANCHOR_REVIEW.md').write_text('\n'.join(lines)+'\n')
    write_json(R/'anchor_evidence.json',dict(verdict='ANCHOR_CONDITIONAL',macro=macro,budget=budget,interaction=interaction,
        cache_replays=30,max_error=max(errors),original_verdict=old['decision'],new_fits=0,original_summary_sha256=sha(p/'summary.json')))


def patch():
    p=ROOT/'results/patchphase_v2_support_complete';r=read(p/'receipt.json');assert r['status']=='COMPLETED' and r['counts']['fits_completed']==12 and r['counts']['updates']==8640
    contract=read(p/'contract.json');seal=read(p/'selection_seal.json');traj=read(p/'trajectory.json');ev=read(p/'evaluation.json');idx=read(p/'prediction_index.json')
    assert digest({k:v for k,v in seal.items() if k!='sha256'})==seal['sha256'] and seal['evaluation_opens']==0
    arrays={};errors=[]
    for row in idx:
        path=ROOT/row['path'];assert sha(path)==row['sha256']
        with np.load(path,allow_pickle=False) as f:
            loss=score(f['prediction'],f['target'],f['scale'])['scaled_2pinball'];scalar=independent(f['prediction'],f['target'],f['scale'])
            if row['split']=='evaluation':arrays[row['tag'],row['phase']]=(f['prediction'].copy(),f['scale'].copy())
        errors.append(max(abs(loss-row['loss']),abs(loss-scalar)))
    assert max(errors)<=1e-10
    assert len(traj)==84 and len(idx)==366
    for row in seal['selections']:
        choices=[t for t in traj if t['arm']==row['arm'] and t['seed']==row['seed']]
        assert row==min(choices,key=lambda t:(t['primary'],t['step'],t['lr']))
    for t in traj:assert sha(ROOT/t['checkpoint'])==t['checkpoint_sha256']
    for e in ev:
        s=e['selection'];tag=f"{s['arm']}_{s['seed']}_E"
        pp=[arrays[tag,phase][0].astype(np.float64)/arrays[tag,phase][1][None,:,None,None] for phase in [3,7,11,15]]
        assert abs(float(np.var(np.stack(pp),axis=0).mean())-e['unseen']['variance'])<=1e-12
    pairs=[]
    for seed in [30000,30001]:
        d={e['selection']['arm']:e for e in ev if e['selection']['seed']==seed};a=d['augmented'];c=d['conditioned'];s=d['standard']
        pairs.append(dict(seed=seed,conditioned_primary=c['unseen']['primary'],augmented_primary=a['unseen']['primary'],standard_primary=s['unseen']['primary'],
            conditioned_gain_vs_augmented=gain(c['unseen']['primary'],a['unseen']['primary']),augmented_gain_vs_standard=gain(a['unseen']['primary'],s['unseen']['primary']),
            conditioned_variance=c['unseen']['variance'],augmented_variance=a['unseen']['variance'],variance_reduction=gain(c['unseen']['variance'],a['unseen']['variance']),
            conditioned_canonical=c['canonical']['primary'],augmented_canonical=a['canonical']['primary'],canonical_degradation=100*(c['canonical']['primary']-a['canonical']['primary'])/a['canonical']['primary']))
    both=all(x['conditioned_gain_vs_augmented']>0 and x['variance_reduction']>0 for x in pairs)
    noharm=all(x['conditioned_canonical']<=x['augmented_canonical'] for x in pairs)
    if both and noharm:verdict='PATCH_V2_SUPPORTED'
    elif not any(x['conditioned_gain_vs_augmented']>0 or x['variance_reduction']>0 for x in pairs):verdict='PATCH_V2_NOT_SUPPORTED'
    else:verdict='PATCH_V2_MIXED'
    lines=['# PatchPhase v2 support-complete','',f'{verdict}. 12/12 fits, 8640 updates; smoke6 updates separate. Historical Candidate03 verdict unchanged. ETTm2 reused development follow-up, not fresh test.','',
        '| seed | conditioned loss | augmented loss | conditioned gain | variance reduction | canonical loss C/A |','|---|---:|---:|---:|---:|---:|']
    for x in pairs:lines.append(f"| {x['seed']} | {x['conditioned_primary']:.9f} | {x['augmented_primary']:.9f} | {x['conditioned_gain_vs_augmented']:+.6f}% | {x['variance_reduction']:+.6f}% | {x['conditioned_canonical']:.9f}/{x['augmented_canonical']:.9f} |")
    lines+=['','Canonical loss is separate from unseen primary. Both seeds must favor primary and phase variance; a positive result with ambiguous clean retention is conservatively MIXED. No arbitrary 30% robustness gate. Full per-phase/V/checkpoint details reside in results/patchphase_v2_support_complete/evaluation.json and trajectory.json.',
        'The final-hidden branch is closed if NOT_SUPPORTED; MIXED is insufficient for automatic expansion. No patch-level v3 constructed.']
    (R/'PATCHPHASE_V2_RESULT.md').write_text('\n'.join(lines)+'\n');write_json(R/'patch_evidence.json',dict(verdict=verdict,pairs=pairs,cache_replays=len(idx),max_error=max(errors)))


def query():
    p=ROOT/'results/reopen_query_resource_20260914';r=read(p/'receipt.json');assert r['status']=='COMPLETED'
    f=read(p/'frontier.json');par=read(p/'parity.json');assert all(v['passed'] for v in par)
    verdict='QUERY_DOMINATED' if all(x['dominators'] for x in f['query_decisions']) else 'QUERY_RESOURCE_FRONTIER'
    lines=['# Query resource frontier','',f'{verdict}. New forecasting fits0, V/E array reads0, old equal-time quality reused. Shared-state equivalence includes output/loss/raw and clipped gradient/update/Adam.','',
        '| dataset | arm | CP blocks | old mean loss | peak MiB | median ms |','|---|---|---:|---:|---:|---:|']
    for x in f['points']:lines.append(f"| {x['dataset']} | {x['arm']} | {x['cp']} | {x['loss']:.9f} | {x['memory']/2**20:.2f} | {1000*x['time']:.3f} |")
    for x in f['query_decisions']:
        q=x['query'];dom=', '.join(f"{d['arm']} CP{d['cp']}" for d in x['dominators']) or 'none'
        lines.append(f"\n{q['dataset']} Query CP{q['cp']}: {x['verdict']}; dominators: {dom}.")
    lines+=['','Old quality scores come from selected two-seed models at the historical nominal equal-time budget. These new measurements are three fixed-train-batch steps at seed30000, not new quality experiments or proof that storage changes quality. Near timing ties are descriptive, not statistically established superiority. Peak reserved and start allocated plus each measured step are in measurements.json. Side may be faster/smaller yet worse quality; such a tradeoff is not domination.']
    (R/'QUERY_RESOURCE_FRONTIER.md').write_text('\n'.join(lines)+'\n');write_json(R/'query_evidence.json',dict(verdict=verdict,frontier=f,receipt=r))


def diagnostics():
    f=read(ROOT/'results/reopen_fr_diagnostic_20260914/summary.json');c=read(ROOT/'results/reopen_censor_diagnostic_20260914/summary.json')
    fr=['# FR entry diagnostic','','New fits0; historical Jena all-step0 result is preserved. This diagnostic uses historical V-selected positive native states on ETTh1/Traffic and fixed overlapping origin pairs. Actual future targets within those already reused development windows are scored; no independent test claim.','']
    for r in f['states']:fr.append(f"- {r['dataset']} seed{r['seed']}: {r['verdict']}; native/F0 losses {r['mean_future_loss']:.9f}/{r['mean_F0_future_loss']:.9f}, correction-revision range {r['FR_range']:.9g}.")
    fr+=['','Raw revision, correction revision, correction magnitude and actual loss for every fixed pair are in results/reopen_fr_diagnostic_20260914/pair_diagnostics.csv. Reopening means an entry condition exists, not that FR improves training or that revision is a causal failure mechanism.']
    (R/'FR_ENTRY_DIAGNOSTIC.md').write_text('\n'.join(fr)+'\n')
    cc=['# Censor tail-gradient diagnostic','','New fits0, model backward0. Existing CDF only, raw-output autograd on selected checkpoints. All360 original train sampling batches replayed for each of two states. Censoring is the original synthetic sale cap, not real stockout annotations. This is a snapshot diagnostic, not historical gradient telemetry.','',
        '| state | censored occurrences | saturated | zero gradient | saturated zero | censor-loss share |','|---|---:|---:|---:|---:|---:|']
    for r in c['states']:cc.append(f"| {r['arm']} | {r['total_censored_positions']} | {r['saturated_fraction']:.6%} | {r['zero_gradient_fraction']:.6%} | {r['saturated_zero_gradient_fraction']:.6%} | {r['saturated_zero_loss_fraction']:.6%} |")
    cc+=['','Materiality is interpreted from the continuous fractions and loss share, not a fabricated performance cutoff. Support-gap quantiles are preserved in summary.json; distances use raw sales units. A large zero-gradient share motivates a corrected tail-loss design but no corrected loss or Censor v2 is trained here.']
    (R/'CENSOR_TAIL_DIAGNOSTIC.md').write_text('\n'.join(cc)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['anchor','patch','query','diagnostics']);a=p.parse_args();globals()[a.phase]()
