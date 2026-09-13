"""Independent artifact replay and the prospectively fixed continuation decision."""
import hashlib
import json
import subprocess
import sys
import numpy as np
import torch
from tsfm_peft_screen.reproducibility import ROOT, sha, digest, write_json
from tsfm_peft_screen.metrics import score, replay
OUT = ROOT/'results/block_shape_pilot'
CACHE = ROOT/'.cache/block_shape_pilot'
read = lambda n: json.loads((OUT/n).read_text())
contract, seal, fits, trajectories, evaluation, status = [read(n+'.json') for n in ['contract', 'selection_seal', 'fits', 'trajectories', 'evaluation', 'status']]
assert status['status']=='COMPLETE' and len(fits)==48 and len(trajectories)==192 and len(evaluation)==32
assert status['proposed_training_iterations']==5760
cfg = contract['config']
assert digest({k:v for k,v in seal.items() if k!='seal_sha256'})==seal['seal_sha256']
assert sha(OUT/'contract.json')==seal['contract_sha256']
assert seal['source_hashes']==contract['source_hashes']
commit = contract['execution_commit']
for name, value in contract['source_hashes'].items():
    historical = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
    assert hashlib.sha256(historical).hexdigest()==value
for name, value in contract['historical_results_sha256'].items():
    assert sha(ROOT/name)==value, ('Historical result changed', name)
for f in fits:
    assert f['steps']==120 and len(f['step_resources'])==120
    assert sha(CACHE/(f['fit']+'_best.pt'))==f['checkpoint_sha256']
    if f['best_blend']:
        assert sha(CACHE/(f['fit']+'_blend.pt'))==f['blend_checkpoint_sha256']
    assert f['peak_allocated_bytes']==max(r['peak_allocated_bytes'] for r in f['step_resources'])
    assert f['trainable_parameters']==(1179648 if f['arm']=='lora' else 6272 if f['arm']=='center' else 14976)
    if f['arm'] in ['pooled', 'block']:
        for r in f['step_resources']:
            g=r['gate'];b=np.array(g['block_changes'])
            bound=float(b.mean()+(0 if f['arm']=='pooled' else 1)*b.std(ddof=1)/np.sqrt(8))
            assert abs(bound-g['criterion'])<1e-12 and (bound<=0)==g['accept']
errors=[]
for row in trajectories+evaluation:
    p=CACHE/(row['prediction_tag']+'.npz')
    assert sha(p)==row['prediction_sha256']
    errors.append(replay(p))
    with np.load(p) as z:
        m=score(z['prediction'], z['target'], z['scale'])
    for key in m:
        assert m[key]==row['metrics'][key]
for r in seal['selections']:
    pool=[f for f in fits if f['dataset']==r['dataset'] and f['seed']==r['seed'] and f['arm']==r['arm']]
    best=min(pool, key=lambda f:(f['best']['metrics']['scaled_2pinball'],f['best']['step'],f['recipe']))
    assert r['fit']==best['fit'] and r['step']==best['best']['step'] and r['checkpoint_sha256']==best['checkpoint_sha256']
for r in seal['blends']:
    pool=[f for f in fits if f['dataset']==r['dataset'] and f['seed']==r['seed'] and f['best_blend']]
    best=min(pool,key=lambda f:(f['best_blend']['metrics']['scaled_2pinball'],f['best_blend']['step'],f['best_blend']['alpha'],f['arm'],f['recipe']))
    assert r['fit']==best['fit'] and r['alpha']==best['best_blend']['alpha'] and r['step']==best['best_blend']['step']
for receipt in read('feature_cache.json'):
    assert sha(CACHE/(receipt['tag']+'_features.pt'))==receipt['sha256']
# Replay every V blend without relying on reported blend metrics.
for name in cfg['datasets']:
    vf=torch.load(CACHE/(name+'_V_features.pt'), weights_only=True)
    f0=(vf['f0']*vf['scale'][:,None]+vf['loc'][:,None]).numpy().reshape(-1,4,21,48)
    for r in read('blend_trajectories.json'):
        if r['dataset']!=name:continue
        with np.load(CACHE/f"V_{r['fit']}_{r['step']}.npz") as z:
            m=score(r['alpha']*np.sort(z['prediction'],axis=2)+(1-r['alpha'])*f0,z['target'],z['scale'])
        assert m==r['metrics']
    with np.load(CACHE/(name+'_development.npz')) as z:
        scale=z['scale']
    y=vf['raw_target'].numpy().reshape(-1,4,48)
    c=next(r for r in seal['calibrations'] if r['dataset']==name)
    choices=[]
    for shift in cfg['calibration_shift']:
        for width in cfg['calibration_width']:
            med=vf['f0'][:,10:11]
            p=((med+shift+width*(vf['f0']-med))*vf['scale'][:,None]+vf['loc'][:,None]).numpy().reshape(-1,4,21,48)
            choices.append(dict(dataset=name,shift=shift,width=width,metrics=score(p,y,scale)))
    assert c==min(choices,key=lambda r:(r['metrics']['scaled_2pinball'],abs(r['shift']),abs(r['width']-1)))
baseline_order=['F0','lora','split','center','anchor','pooled','blend','calibration']
decisions=[]
for name in cfg['datasets']:
    per_seed=[]
    for seed in cfg['seeds']:
        ev={r['arm']:r for r in evaluation if r['dataset']==name and r['seed'] in [None,seed]}
        candidate=ev['block'];best=min([ev[a] for a in baseline_order],key=lambda r:r['metrics']['scaled_2pinball'])
        per_seed.append(dict(seed=seed,block_loss=candidate['metrics']['scaled_2pinball'],
            strongest_baseline=best['arm'],baseline_loss=best['metrics']['scaled_2pinball'],
            f0_ratio=candidate['metrics']['scaled_2pinball']/ev['F0']['metrics']['scaled_2pinball'],
            block_coverage_error=abs(candidate['metrics']['interval80_coverage']-.8),
            baseline_coverage_error=abs(best['metrics']['interval80_coverage']-.8),
            selected_step=candidate['step'],shape_displacement=candidate['shape_log_multiplier_abs_mean']))
    ratio=float(np.mean([r['block_loss'] for r in per_seed])/np.mean([r['baseline_loss'] for r in per_seed]))
    coverage_difference=float(np.mean([r['block_coverage_error']-r['baseline_coverage_error'] for r in per_seed]))
    checks=dict(primary=ratio<=.995,f0_safety=all(r['f0_ratio']<=1.01 for r in per_seed),
                coverage=coverage_difference<=.01,active_shape=all(r['selected_step']>0 and r['shape_displacement']>0 for r in per_seed))
    decisions.append(dict(dataset=name,per_seed=per_seed,block_to_strongest_baseline_ratio=ratio,
                          coverage_error_difference=coverage_difference,checks=checks,pass_gate=all(checks.values())))
verdict='PASS_CONTINUATION' if all(d['pass_gate'] for d in decisions) else 'STOP'
verification=dict(status='PASS',prediction_cache_replays=len(errors),max_independent_metric_error=max(errors),
                  historical_result_files_unchanged=len(contract['historical_results_sha256']),
                  fit_count=48,proposed_training_iterations=5760,verdict=verdict,decisions=decisions)
if '--verify-only' in sys.argv:
    assert verification==read('verification.json')
    print('BLOCK SHAPE VERIFIED',verdict,len(errors),'prediction caches; historical files unchanged')
    raise SystemExit(0)
write_json(OUT/'verification.json',verification)
write_json(OUT/'decision.json',dict(verdict=verdict,decisions=decisions))
lines=['# Block-conditioned shape adaptation: first pilot', '',
       f'**{verdict}** on the prospectively fixed research continuation gate. 48/48 fits, two datasets, two seeds, 5,760 proposed iterations. This is not a publication or novelty verdict.', '',
       f'Execution commit: {commit}. All choices were sealed before heldout feature extraction and scoring. New chronological periods on existing source series; not external dataset replication.', '',
       '| Dataset | Seed | Arm | Selected step | Scaled 2-pinball ↓ | 80% coverage | Width |',
       '|---|---:|---|---:|---:|---:|---:|']
for r in evaluation:
    m=r['metrics']
    lines.append(f"| {r['dataset']} | {r['seed'] if r['seed'] is not None else 'shared'} | {r['arm']} | {r.get('step','—')} | {m['scaled_2pinball']:.8f} | {m['interval80_coverage']:.4f} | {m['interval80_width']:.4f} |")
lines+=['', '## What this pilot established', '',
'The update mechanism was active: every selected candidate had a nonzero shape change. It nevertheless failed the primary criterion on both datasets. This is a learning-quality failure under the fixed protocol, not an implementation block or GPU capacity failure.',
'Conservative LoRA has lower seed-mean primary loss than the block candidate on both datasets. Relative to pooled acceptance, the block variability penalty changes heldout loss very little and provides no gain here. Preventing some training shape steps was therefore insufficient to improve generalization.',
'The ETTm2 scalar calibration baseline illustrates a tradeoff: 80% coverage approaches its nominal target, but its primary loss worsens. Coverage alone would give a misleading success signal. On Electricity, F0 already has coverage near 80%.',
'These findings reject this particular first rule as a current lead; they do not establish that every form of uncertainty-aware PEFT must fail.',
'', '## Fixed decision','']
for d in decisions:
    lines.append(f"- {d['dataset']}: block / strongest baseline seed-mean primary ratio **{d['block_to_strongest_baseline_ratio']:.6f}** (required <=0.995), coverage-error difference {d['coverage_error_difference']:+.6f}. Checks: {d['checks']}.")
    for r in d['per_seed']:
        lines.append(f"  Seed {r['seed']}: strongest comparator {r['strongest_baseline']}; selected block step {r['selected_step']}, mean absolute log-gap displacement {r['shape_displacement']:.6g}.")
lines+=['','The strongest comparator is the predeclared conservative E-oracle across separately V-selected baselines. No candidate checkpoint or recipe was selected by E.', '',
        '## Mechanism and resources','',
        '| Dataset | Seed | Rule | Recipe | Accepted shape proposals /120 | Best step | Peak allocated MiB | Median step ms |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
for f in fits:
    if f['arm'] in ['pooled','block']:
        lines.append(f"| {f['dataset']} | {f['seed']} | {f['arm']} | {f['recipe']} | {f['accepted_shape_steps']} | {f['best']['step']} | {f['peak_allocated_bytes']/2**20:.1f} | {f['median_step_seconds']*1000:.2f} |")
gpu=read('gpu_monitor.json')
active=[r for r in gpu if r['phase']!='startup_wait']
lines+=['',f"GPU monitor: {len(gpu)} samples; minimum observed active free VRAM {min(r['free_mib'] for r in active):.0f} MiB; samples with external compute {sum(bool(r['external_pids']) for r in active)}. Model work pauses at step boundaries when other compute appears. Worker wall time: {status['wall_seconds']:.1f}s.",
        '',f"Frozen feature extraction total: {sum(r['seconds'] for r in read('feature_cache.json')):.1f}s. Cached adapter steps exclude repeated backbone computation. Full training-step resources remain in fits.json; this is not an equal-end-to-end-cost comparison or memory-efficiency claim.",
        '', '## Interpretation limits','',
        'The architecture overlaps established monotonic quantile calibration and output-adapter methods. The tested incremental hypothesis is shape-only finite-step acceptance penalized by temporal-block variability. This is not a confidence interval or a distribution-free calibration guarantee.',
        'Only two existing source datasets and one later split are evaluated. Two seeds address optimizer variation, not independent sampling. Four original channels, fixed learning-rate grids and one anchor coefficient limit the comparison. No official AdaPTS or delta-Adapter reproduction is claimed. A positive continuation decision would still require stronger baselines, additional sources and a full novelty assessment.',
        '', '## Verification','',
        f"{len(errors)} saved prediction caches replay the primary loss with maximum independent error {max(errors):.3g}; every reported cached metric replays exactly. Selection seals, checkpoint hashes, blend/calibration choices, and {verification['historical_result_files_unchanged']} historical files verified. Gate decisions replay from recorded block changes. No additional tuning or experiment expansion.",
        '', 'Protocol and primary literature links: [fixed protocol](../../docs/BLOCK_SHAPE_PROTOCOL.md).']
(OUT/'RESULT.md').write_text('\n'.join(lines)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(12,4))
for ax,name in zip(axs,cfg['datasets']):
    arms=['F0','lora','split','center','anchor','pooled','block','blend','calibration']
    losses=[np.mean([r['metrics']['scaled_2pinball'] for r in evaluation if r['dataset']==name and r['arm']==a]) for a in arms]
    reference=losses[0]
    gains=[100*(reference-loss)/reference for loss in losses]
    ax.bar(arms,gains,color=['#167d9a' if a=='block' else '#a4adb5' for a in arms])
    ax.set_title(name+' — later chronological evaluation')
    ax.axhline(0,color='black',linewidth=.7)
    ax.set_ylabel('Primary loss reduction vs F0 (%) ↑')
    ax.tick_params(axis='x',rotation=55)
fig.tight_layout()
(OUT/'figures').mkdir(exist_ok=True)
fig.savefig(OUT/'figures/primary.png',dpi=160)
plt.close(fig)
print('BLOCK SHAPE FINALIZED',verdict,verification)
