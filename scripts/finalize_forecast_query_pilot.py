"""Verify real fits, validation sealing, prediction replay and training resources."""
import hashlib,json,subprocess,sys
import numpy as np
from tsfm_peft_screen.reproducibility import ROOT,sha,digest,write_json
from tsfm_peft_screen.metrics import score,independent
OUT=ROOT/'results/forecast_query_pilot';CACHE=ROOT/'.cache/forecast_query_pilot'
def read(name):return json.loads((OUT/name).read_text())
c=read('contract.json');cfg=c['config'];s=read('status.json');fits=read('fits.json');tr=read('trajectories.json');ev=read('evaluation.json');seal=read('selection_seal.json');smoke=read('smoke.json');gpu=read('gpu_monitor.json')
assert s['status']=='COMPLETE' and len(fits)==16 and s['fit_count']==16 and s['training_optimizer_updates']==3840 and s['smoke_optimizer_updates']==8
assert len(tr)==96 and len(ev)==10 and len(smoke)==4
for name,value in c['source_hashes'].items():
    assert hashlib.sha256(subprocess.check_output(['git','show',f"{c['execution_commit']}:{name}"],cwd=ROOT)).hexdigest()==value,name
assert sha(ROOT/'configs/forecast_query_pilot.json')==c['config_sha256']
for name in cfg['datasets']:
    assert sha(ROOT/'data/processed'/name/'fit.npz')==c['fit_data_hashes'][name]
    assert sha(ROOT/'data/processed'/name/'manifest.json')==c['data_manifest_hashes'][name]
sealvalue=seal['seal_sha256'];unsigned={k:v for k,v in seal.items() if k!='seal_sha256'};assert digest(unsigned)==sealvalue
assert seal['source_hashes']==c['source_hashes'] and seal['contract_sha256']==sha(OUT/'contract.json')
maxerror=0;count=0
for row in tr+ev:
    tag=f"V_{row['fit']}_{row['step']}" if 'recipe' in row else f"E_{row['dataset']}_{row['arm']}"
    with np.load(CACHE/f'{tag}.npz') as f:
        actual=score(f['prediction'],f['target'],f['scale']);scalar=independent(f['prediction'],f['target'],f['scale'])
    assert abs(actual['scaled_2pinball']-scalar)<1e-10
    for k,v in actual.items():maxerror=max(maxerror,abs(v-row['metrics'][k]))
    count+=1
assert maxerror<1e-10 and count==106
for f in fits:
    assert f['steps']==240 and len(f['step_resources'])==240 and f['trainable_parameters']==1179648
    records=[v for v in tr if v['fit']==f['fit']];best=min(records,key=lambda r:(r['metrics']['scaled_2pinball'],r['step']))
    assert best['step']==f['best_step'] and best['metrics']==f['best_validation']
    assert sha(CACHE/f"{f['fit']}_best.pt")==f['checkpoint_sha256']
    assert f['peak_allocated_bytes']==max(r['peak_allocated_bytes'] for r in f['step_resources'])
    assert f['peak_reserved_bytes']==max(r['peak_reserved_bytes'] for r in f['step_resources'])
    assert f['median_step_seconds']==float(np.median([r['seconds'] for r in f['step_resources']]))
    assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in f['step_resources'])
for choice in seal['selections']:
    candidate=min([f for f in fits if f['dataset']==choice['dataset'] and f['arm']==choice['arm']],key=lambda f:(f['best_validation']['scaled_2pinball'],f['best_step'],f['recipe']))
    assert choice['fit']==candidate['fit'] and choice['checkpoint_sha256']==candidate['checkpoint_sha256']
for a in smoke:
    assert a['parameters']==1179648 and a['fp32_initial_relative']<=1e-5 and a['fp32_initial_max_abs']<=1e-4 and a['bf16_initial_relative']<=.02
    assert a['changed_parameter_tensors']>0 and a['forecast_change_max_abs']>0 and a['group_isolation_max_abs']<=1e-4
    if a['arm']=='query':assert a['query_backward_mlp_token_counts']==[4] and a['missing_key_parity_relative']<=1e-5
computed=[]
for name in cfg['datasets']:
    e={r['arm']:r for r in ev if r['dataset']==name};sel={r['arm']:r for r in seal['selections'] if r['dataset']==name};fr={arm:next(f for f in fits if f['fit']==choice['fit']) for arm,choice in sel.items()}
    q=e['query']['metrics']['scaled_2pinball'];standard=e['standard']['metrics']['scaled_2pinball'];simple=min(e[a]['metrics']['scaled_2pinball'] for a in ['head','side']);f0=e['F0']['metrics']['scaled_2pinball'];reduction=1-fr['query']['peak_allocated_bytes']/fr['standard']['peak_allocated_bytes']
    passed=sel['query']['best_step']>0 and q<f0 and q<=1.01*standard and reduction>=.20 and q<=.995*simple
    computed.append(dict(dataset=name,query_vs_standard_loss_ratio=q/standard,query_vs_best_simple_loss_ratio=q/simple,query_vs_f0_loss_ratio=q/f0,query_peak_reduction=reduction,query_step=sel['query']['best_step'],pass_gate=passed))
assert computed==s['decisions'];assert s['verdict']==('PASS' if all(d['pass_gate'] for d in computed) else 'FAIL')
old=['results/local_backward_feasibility','results/memory_feasibility','results/candidate_01_v2','results/candidate_05_repaired','results/screening_summary']+[f'results/candidate_{i:02}' for i in range(1,8)]
assert not subprocess.check_output(['git','diff','39ab580b512f1f8054c49aae592816703c539c27','--',*old],cwd=ROOT)
active=[r for r in gpu if r['phase']!='startup_wait']
receipt=dict(status='PASS',verdict=s['verdict'],fits=16,updates=3840,smoke_updates=8,prediction_cache_replays=count,max_raw_metric_replay_error=maxerror,selection_seal_valid=True,all_historical_results_unchanged=True,trainable_parameters_per_arm=1179648,gpu_samples=len(gpu),min_observed_active_free_mib=min(r['free_mib'] for r in active),max_observed_active_used_mib=max(r['used_mib'] for r in active),external_compute_samples=sum(bool(r['external_pids']) for r in active),paused_samples=sum(r['phase'].startswith('paused_') for r in active),metrics_sha256=sha(OUT/'evaluation.json'),fits_sha256=sha(OUT/'fits.json'),status_sha256=sha(OUT/'status.json'))
if '--verify-only' in sys.argv:
    assert read('verification.json')==receipt
    print('FORECAST QUERY VERIFICATION PASS:',s['verdict'],'16 fits,3840 updates,106 caches');raise SystemExit(0)
write_json(OUT/'verification.json',receipt)
lines=['# Frozen past / forecast-token adaptation pilot: '+s['verdict'],'',
'Completed **16/16 actual learning fits**,3,840 optimizer updates, plus8 discarded diagnostic updates. Two datasets × four arms × two fixed arm-appropriate learning rates × first seed30000. This is chronological development forecasting evidence, not a fixed-state gradient screen and not an independent holdout.','',
'## Selected development results','',
'Primary is raw-scale equal-channel scaled2-pinball (lower is better), matching the historical scoring convention. Native asinh-space loss is also reported. Choices use11 validation origins, six checkpoints and two recipes per arm; all8 choices were sealed before evaluation arrays were loaded. Evaluation uses16 origins per dataset.','',
'| Dataset | Arm | Recipe / step | Primary loss | Native loss | Train-step peak MiB | Median step ms | Whole fit seconds |','|---|---|---|---:|---:|---:|---:|---:|']
for name in cfg['datasets']:
    e={r['arm']:r for r in ev if r['dataset']==name}
    for arm in ['F0']+cfg['arms']:
        if arm=='F0':
            v=e[arm]['metrics'];lines.append(f"| {name} | F0 | no fit | {v['scaled_2pinball']:.8f} | {v['native_loss']:.8f} | — | — | — |")
        else:
            choice=next(r for r in seal['selections'] if r['dataset']==name and r['arm']==arm);f=next(v for v in fits if v['fit']==choice['fit']);v=e[arm]['metrics']
            lines.append(f"| {name} | {arm} | R{f['recipe']} / {f['best_step']} | {v['scaled_2pinball']:.8f} | {v['native_loss']:.8f} | {f['peak_allocated_bytes']/2**20:.1f} | {1000*f['median_step_seconds']:.2f} | {f['wall_seconds']:.2f} |")
lines+=['','## Fixed decision','',
'| Dataset | Query loss / standard | Query loss / best head-or-side | Query loss / F0 | Query peak reduction | Selected step | Gate |','|---|---:|---:|---:|---:|---:|---|']
for d in s['decisions']:lines.append(f"| {d['dataset']} | {d['query_vs_standard_loss_ratio']:.5f} | {d['query_vs_best_simple_loss_ratio']:.5f} | {d['query_vs_f0_loss_ratio']:.5f} | {100*d['query_peak_reduction']:.2f}% | {d['query_step']} | {'PASS' if d['pass_gate'] else 'FAIL'} |")
lines+=['','Gate fixed before runs: on both datasets query must select a nonzero step and beat F0, remain within1% of Standard LoRA loss, reduce actual training-step peak by20%, and beat the best head/side loss by0.5%. An initial checkpoint alone cannot establish adaptation value. No recipe/threshold changed after validation or evaluation. The first-seed16-fit budget is complete; no new seed, architecture expansion, or fixed-memory batch experiment was launched in this run.','',
'## Implementation and fairness','',
'All arms train exactly1,179,648 parameters and use the same frozen native quantile decoder, BF16 autocast, FP32 parameters/AdamW moments, batch8 (two windows × four channels), context4096, horizon48. Standard uses all-token rank8 LoRA and exact non-reentrant checkpointing. Query uses the same96 attention LoRA projections only on REG+3 forecast tokens. A fresh full frozen bidirectional no_grad F0 pass supplies history K/V at every layer, preserving original RoPE positions, key-validity and group masks. Only4 tokens enter the trainable native attention/group/MLP branch. This is a restricted adaptation model after updates, not equivalent to ordinary all-token LoRA or a causal prefix cache. The candidate branch itself was not checkpointed in this fixed implementation. Frozen history K/V still occupy memory, and concatenated attention K/V plus backward temporaries are counted. No separate operator-level inventory was run here, so the peak increase is measured but its per-operator attribution is not established.','',
'Head is a768→768→768 residual SiLU head before the native decoder. Side is an explicitly adapted LST-style12-layer,64-wide time/group side network reading full frozen per-layer representations; its learned lateral projections retain those features and their memory is counted. Zero output matrices start the controls at F0. Both side and query are prior-based candidates, not established novel contributions. The full frozen pass, cache construction, train branch, clipping and actual optimizer.step are included in step peaks/times. Validation and checkpoint copying are included in whole-fit wall time. No cache persists across input windows.','',
'FP32 initialization parity, BF16 tolerance, nonzero update/effect, group isolation, missing-key handling and query-only MLP graph were tested on Chronos itself. BF16 query initialization need not be bit-identical to F0 because GEMM/attention shapes differ; the measured discrepancy is in `smoke.json`. No claim of exact BF16 gradients or identical trajectories is made. Training loss uses native probabilistic quantiles; sorted quantiles are used consistently for raw-scale scoring. Native loss is recorded from the model;106 saved raw prediction arrays independently replay the raw-scale metrics.','',
'## GPU occupancy and resource limits','',
f"Monitoring recorded{receipt['gpu_samples']} samples. During active work, minimum observed free VRAM was{receipt['min_observed_active_free_mib']:.0f}MiB; maximum observed device memory used was{receipt['max_observed_active_used_mib']:.0f}MiB. External-compute samples:{receipt['external_compute_samples']}; paused samples:{receipt['paused_samples']}. Sampled device occupancy is distinct from per-step CUDA allocator peaks; brief transients can occur between monitor samples.",'',
'Startup initially waited on a<=20% utilization threshold. Before any model run, pmon identified graphics-only use from Code/Chrome/Xorg/GNOME with around9.1GiB free and no compute PID. The waiting worker alone was interrupted; the startup rule was revised to30seconds without external compute, free>=4GiB and utilization<90%. The original wait is preserved in `results/forecast_query_startup_wait/`. No external job was stopped. During the pilot, monitor at step boundaries at least every5seconds and pause on another compute PID or free<1GiB. Desktop graphics remained active, so timing is a monitored shared-display measurement, not an isolated-GPU benchmark.','',
f"Worker wall time excluding initial startup wait:{s['wall_seconds']:.2f}s. Total fit wall time:{sum(f['wall_seconds'] for f in fits):.2f}s. All240 step resources per fit, validation time and GPU timeline are committed. Curves show fixed-update and observed fixed-wall-clock progress at the same batch; no throughput claim from larger batches is made.",'',
'## Provenance and limits','',
f"Execution commit:`{c['execution_commit']}`. Source/config/data/checkpoint hashes and selection seal are verified. All prior seven-candidate, recovery, Freshness v2, global memory and local backward results remain unchanged. Historical totals plus this pilot:62 completed fits and9 stream attempts (8 complete,1 historical abort). The8 new smoke updates are separate from the3,840 pilot updates. One seed, four channels, a short development training period and previously exposed evaluation periods limit generalization.",'',
'Prior boundaries: [LST](https://papers.neurips.cc/paper_files/paper/2022/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html), [Activated LoRA](https://papers.neurips.cc/paper_files/paper/2025/hash/4d0b6303d4a4811445f69f357bf6def5-Abstract-Conference.html), [EfficientFSL](https://arxiv.org/abs/2601.08499), [TS-Memory](https://arxiv.org/html/2602.11550v1).','',
'Replay with `scripts/with_cuda.sh .venv/bin/python scripts/finalize_forecast_query_pilot.py --verify-only`. Local ignored checkpoints/prediction arrays are required; model weights/raw data are not committed.','',
'![Validation learning curves](figures/learning.png)','',
'![GPU monitoring](figures/gpu.png)','']
(OUT/'RESULT.md').write_text('\n'.join(lines))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors=dict(standard='#4a6c93',head='#c58a26',side='#3b957a',query='#c84e64')
fig,axs=plt.subplots(2,2,figsize=(12,8))
for i,name in enumerate(cfg['datasets']):
    for arm in cfg['arms']:
        selected=next(r for r in seal['selections'] if r['dataset']==name and r['arm']==arm);points=[r for r in tr if r['fit']==selected['fit']]
        for j,key in enumerate(['step','elapsed_seconds']):axs[i,j].plot([r[key] for r in points],[r['metrics']['scaled_2pinball'] for r in points],marker='o',label=arm,color=colors[arm])
    for j in range(2):axs[i,j].set(title=name+' selected recipe',xlabel='Optimizer update' if j==0 else 'Fit elapsed seconds (includes V)',ylabel='Validation scaled2-pinball');axs[i,j].grid(alpha=.2);axs[i,j].legend(fontsize=8)
fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/learning.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True);t0=gpu[0]['time_unix'];times=[(r['time_unix']-t0)/60 for r in gpu]
axs[0].plot(times,[r['used_mib'] for r in gpu],label='Device memory used');axs[0].plot(times,[r['free_mib'] for r in gpu],label='Device memory free');axs[0].set(ylabel='MiB',title='Sampled GPU occupancy (display activity included)');axs[0].legend()
axs[1].plot(times,[r['utilization_percent'] for r in gpu],color='#4a6c93');axs[1].set(xlabel='Minutes since startup monitor',ylabel='GPU utilization (%)',ylim=(0,105))
for ax in axs:ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'figures/gpu.png',dpi=160);plt.close(fig)
print('FORECAST QUERY REPORT COMPLETE:',s['verdict'])
