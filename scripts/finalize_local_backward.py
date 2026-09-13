"""Replay local codec gradient diagnostics; preserve all historical outputs."""
import hashlib,json,subprocess,sys
import numpy as np
import torch
from tsfm_peft_screen.reproducibility import ROOT,sha,write_json
OUT=ROOT/'results/local_backward_feasibility';CACHE=ROOT/'.cache/local_backward_feasibility'
def read(name):return json.loads((OUT/name).read_text())
c=read('contract.json');s=read('status.json');rows=read('metrics.json');audit=read('integration_parity.json');cfg=c['config']
assert s['status']=='COMPLETE' and s['optimizer_updates']==0 and not s['e_access'] and s['parameter_state_unchanged']
assert len(rows)==32 and len(audit)==10 and s['completed_backward_passes']==216
for name,value in c['source_hashes'].items():
    assert hashlib.sha256(subprocess.check_output(['git','show',f"{c['execution_commit']}:{name}"],cwd=ROOT)).hexdigest()==value,name
assert sha(ROOT/'configs/local_backward_feasibility.json')==c['config_sha256']
for name in cfg['datasets']:
    assert sha(ROOT/'data/processed'/name/'fit.npz')==c['data_hashes'][name]
    assert sha(ROOT/'.cache/memory_feasibility'/f'{name}_warm.pt')==c['warm_hashes'][name]
for r in audit:
    assert r['input_gradient_relative_l2']<=1e-5 and r['B_relative_l2']<=1e-5 and r['forward_max_abs_error']==0
    if r['method']=='all_details':assert r['A_relative_l2']<=1e-5
max_replay_error=0.;cached_draws=0;refs={}
for row in rows:
    name,length,method=row['dataset'],row['context'],row['method'];runs=row['repeats']
    assert len(runs)==(16 if method in ['prac','residual'] else 3)
    assert row['peak_allocated_bytes']==int(np.median([r['peak_allocated_bytes'] for r in runs[:3]]))
    assert row['median_seconds']==float(np.median([r['seconds'] for r in runs[:3]]))
    assert row['A_error_rms']==float(np.sqrt(np.mean([r['gradient']['A']['relative_l2']**2 for r in runs])))
    for r in runs:
        assert np.isfinite(r['loss']) and all(np.isfinite(e['relative_l2']) for e in r['gradient'].values())
        if not method.startswith('amp'):assert r['forward_max_abs_error']==0 and r['gradient']['B']['relative_l2']<=1e-5
        if method in ['standard','checkpoint']:assert r['gradient']['A']['relative_l2']<=1e-5
    key=(name,length)
    if key not in refs:refs[key]=torch.load(CACHE/f'{name}_{length}_reference.pt',weights_only=True)
    ref=refs[key];names=[n for n in ref if n.endswith('lora_A')];rvec=torch.cat([ref[n].flatten().double() for n in names]);assert rvec.norm()>0
    if method in ['prac','residual']:
        mean=torch.zeros_like(rvec);norms=[]
        for r in runs:
            saved=torch.load(CACHE/f"{name}_{length}_{method}_{r['seed']}.pt",weights_only=True)
            v=torch.cat([saved[n].flatten().double() for n in names]);err=float((v-rvec).norm()/rvec.norm())
            max_replay_error=max(max_replay_error,abs(err-r['gradient']['A']['relative_l2']));mean+=v/len(runs);norms.append(float(v.square().sum()));cached_draws+=1
        meanerr=float((mean-rvec).norm()/rvec.norm());variance=max(0.,(np.mean(norms)-float(mean.square().sum()))/float(rvec.square().sum()))
        max_replay_error=max(max_replay_error,abs(meanerr-row['mean_A_relative_error']),abs(variance-row['A_centered_variance_relative']))
    else:
        saved=torch.load(CACHE/f'{name}_{length}_{method}.pt',weights_only=True)
        for label in ['all','A','B']:
            selected=[n for n in ref if label=='all' or n.endswith('lora_'+label)]
            a=torch.cat([saved[n].flatten().double() for n in selected]);r=torch.cat([ref[n].flatten().double() for n in selected]);err=float((a-r).norm()/r.norm())
            max_replay_error=max(max_replay_error,abs(err-runs[0]['gradient'][label]['relative_l2']))
assert max_replay_error<1e-12 and cached_draws==128
computed=[]
for name in cfg['datasets']:
    r={v['method']:v for v in rows if v['dataset']==name and v['context']==4096};v=r['residual'];standard=r['standard']
    reduction=1-v['peak_allocated_bytes']/standard['peak_allocated_bytes'];ratio=v['median_seconds']/standard['median_seconds'];errorratio=v['A_error_rms']/r['local_fp16']['A_error_rms']
    dominated=any(r[k]['peak_allocated_bytes']<=v['peak_allocated_bytes'] and r[k]['median_seconds']<=v['median_seconds'] for k in ['checkpoint','amp_checkpoint'])
    passed=reduction>=.05 and ratio<=1.15 and errorratio<=1 and not dominated
    computed.append(dict(dataset=name,peak_reduction=reduction,time_ratio=ratio,A_rms_error_ratio_vs_local_fp16=errorratio,dominated_by_checkpoint=dominated,pass_gate=passed))
assert computed==s['decisions']
assert s['verdict']==('PILOT_ELIGIBLE' if all(d['pass_gate'] for d in computed) else 'STOP')
historical=['results/memory_feasibility','results/candidate_01_v2','results/candidate_05_repaired','results/screening_summary']+[f'results/candidate_{i:02}' for i in range(1,8)]
assert not subprocess.check_output(['git','diff','a1bf744988fdc456ba4f117da56bc3496159ac7f','--',*historical],cwd=ROOT)
receipt=dict(status='PASS',verdict=s['verdict'],cases=32,measured_comparison_backwards=200,total_backward_passes=216,optimizer_updates=0,pilot_fits=0 if s['verdict']=='STOP' else None,A_draw_cache_replays=cached_draws,max_gradient_replay_error=max_replay_error,all_historical_results_unchanged=True,max_input_gradient_relative_error=max(r['input_gradient_relative_l2'] for r in audit),max_B_gradient_relative_error=max(r['B_relative_l2'] for r in audit),max_all_details_A_relative_error=max(r['A_relative_l2'] for r in audit if r['method']=='all_details'),metrics_sha256=sha(OUT/'metrics.json'),status_sha256=sha(OUT/'status.json'))
if '--verify-only' in sys.argv:
    assert read('verification.json')==receipt
    print('LOCAL BACKWARD VERIFICATION PASS:',s['verdict'],'128 stochastic gradient-cache replays, max error',max_replay_error);raise SystemExit(0)
write_json(OUT/'verification.json',receipt)
lines=['# Local residual-corrected LoRA backward: '+s['verdict'],'',
'**구현 정확성은 통과했지만, 잔차 보정 후보는 학습 진입 기준을 통과하지 못했습니다.** 두 데이터에서 local FP16보다 메모리를 더 사용하고 A-gradient 추정 오차가 훨씬 큽니다. 학습 허용은 반영했으나 사전 고정한 조건에 따라 새 학습은0 fits입니다.','',
'## What completed','',
'32 method/dataset/context cases:8 arms ×2 datasets ×2 contexts. Three rotated-order timing repeats per arm; PRAC/residual16 stochastic draws total per case.200 measured backwards,4 exact references and12 integration audit backwards =216. Zero optimizer updates; warmed adapter tensors verified unchanged. No V/E arrays accessed by this diagnosis. The conditional16-fit pilot was authorized and preregistered but not launched after gate failure.','',
'Pinned Chronos-2, standard rank8 LoRA, frozen backbone/head, multiplier2, float32 parameters; contexts1024/4096, horizon48, batch8, first4 source-order channels, fit origins4352/4608. The existing four-update warm state for each dataset ensures nonzero A-gradient. FP32 AdamW first/second moments are resident in every arm, including BF16 autocast. Peaks include live tensors and temporary GPU allocations for forward/backward, not an executed optimizer step or the entire driver/process footprint.','',
'## Correctness','',
f"Forward outputs were bit-identical for all FP32 codecs. Integration audits give maximum input-gradient relative error `{receipt['max_input_gradient_relative_error']}` and B-gradient relative error `{receipt['max_B_gradient_relative_error']}`. Deterministically storing all details reproduces A-gradient within `{receipt['max_all_details_A_relative_error']:.3g}` relative error. CPU exhaustive enumeration verifies the small estimator expectation;128 actual stochastic A-gradient caches replay with maximum metric discrepancy `{max_replay_error}`.",'',
'Only local A-gradient is approximated. Z and native B backward remain exact; incoming gradient propagates through A exactly. No approximate X is supplied to normalization, attention, MLP or the frozen base. Residual, CARE and PRAC calculate A-gradient directly without reconstructing full X. Generic local FP16 reconstructs only its local X. Group-attention tensors are put in B,T,D order for pairing; channels/batches are not mixed. Last4 REG/future tokens remain exact in residual coding.','',
'## Context4096 measurements','',
'| Dataset | Method | Peak MiB | Reduction vs standard | Median ms | Time / standard | A-gradient RMS relative error |','|---|---|---:|---:|---:|---:|---:|']
for name in cfg['datasets']:
    case={v['method']:v for v in rows if v['dataset']==name and v['context']==4096};base=case['standard']
    for method in cfg['methods']:
        v=case[method];lines.append(f"| {name} | {method} | {v['peak_allocated_bytes']/2**20:.1f} | {100*(1-v['peak_allocated_bytes']/base['peak_allocated_bytes']):.2f}% | {1000*v['median_seconds']:.2f} | {v['median_seconds']/base['median_seconds']:.3f}× | {100*v['A_error_rms']:.5f}% |")
lines+=['','These are **gradient diagnostics, not forecasting-loss degradation**. AMP is compared against the FP32 reference, so its numerical gradient difference is expected and is not an exact-parity failure. CARE/PRAC A-error does not establish their downstream training quality. Timing repeats use one fixed batch, not independent training seeds. Full1024 results and every draw are in `metrics.json`.','',
'## Stochastic estimator','',
'| Dataset | Codec | Single-draw A-error RMS | Error of16-draw mean | Centered variance / exact A norm² |','|---|---|---:|---:|---:|']
for r in rows:
    if r['context']==4096 and r['method'] in ['prac','residual']:lines.append(f"| {r['dataset']} | {r['method']} | {r['A_error_rms']*100:.4f}% | {r['mean_A_relative_error']*100:.4f}% | {r['A_centered_variance_relative']:.6f} |")
lines+=['','The16-draw mean is only a diagnostic; a training step would use one draw. Finite-sample mean error is not evidence of estimator bias or a proof of unbiasedness on the full network. Inverse-probability correction has an exact-arithmetic conditional expectation identity; floating-point arithmetic adds roundoff. Clipping and Adam are nonlinear, so that identity would not guarantee unbiased updates or forecasting gains.','',
'## Memory accounting and prior scope','',
'The residual codec stores FP32 pair means, one-quarter as many sampled details as pairs, their indices/inverse probabilities and exact special tokens. It shares48 payloads across96 LoRA modules in this model. Local FP16 also shares these inputs. At4096, cumulative encoded payload bytes are193,609,728 for residual versus153,354,240 for local FP16; these sums are not peak memory. CARE stores A-specific Z/decoders and does not share them; its Z also serves native B backward. Weak references preserve storage identity without retaining original X. All basis, decoder and reconstruction temporaries enter peak/timing measurements.','',
'CARE is a ridge-decoder primitive based on [CARE-LoRA](https://arxiv.org/abs/2607.11940). PRAC is a principal-random primitive based on [PRAC](https://arxiv.org/abs/2602.23111), with a rank8 randomized principal subspace and rank8 orthogonal random tail rebuilt every forward. It does not reproduce the official SVD/lazy-update algorithm. Those much smaller payloads are not byte-matched to residual; their metrics describe these fixed primitives only. Local FP16 is the closest direct control and already rules out a practical advantage for the proposed setting. No new contribution is claimed for principal+random correction or A-only approximation.','',
'The earlier global saved-tensor FP16 study compressed more than LoRA-local inputs and omitted optimizer moments in Stage B. Its peak numbers are not interchangeable with this local-only, optimizer-resident study. The remaining frozen MLP and attention activations are unchanged here. ReLU mask optimization, side networks, and other temporal codecs were not pursued.','',
'## Fixed decision','',
'Before GPU measurement, the conditional learning gate required both datasets at4096: at least5% peak reduction, at most15% step-time overhead versus standard, A-error RMS no worse than local FP16, and no peak/time domination by exact or AMP checkpointing. The candidate meets its modest reduction/time bounds but fails the error comparison and is dominated by AMP checkpointing. Local FP16 itself has lower peak and lower A-error. The gradient gate is a conservative budget decision, not a publication-success definition. No threshold, sampling budget, precision or learning recipe was changed after results.','',
'**STOP this fixed primitive;0 new fits.** User authorization for learning is recorded in the preregistered config, but the specified prerequisite did not hold. The cap of16 fits was never a completed-fit claim. Historical totals remain46 completed fits and9 stream attempts (8 complete,1 historical abort). All original screening, recovery, Freshness v2 and global memory-feasibility artifacts remain byte-for-byte unchanged.','',
f"Execution commit: `{c['execution_commit']}`. See [contract](contract.json), [configuration](../../configs/local_backward_feasibility.json), [protocol](../../docs/LOCAL_BACKWARD_PROTOCOL.md) and [verification](verification.json). Reproduce verification with `scripts/with_cuda.sh .venv/bin/python scripts/finalize_local_backward.py --verify-only`; local ignored warm/gradient caches are required. Raw data, model weights and gradient tensors are not committed.",'',
'![Local backward tradeoffs](figures/tradeoffs.png)','']
(OUT/'RESULT.md').write_text('\n'.join(lines))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors=dict(standard='#555555',checkpoint='#208b77',local_fp16='#356aa0',care='#b17400',prac='#9261aa',residual='#d24752',amp='#4a9cb9',amp_checkpoint='#7b9e2b')
fig,axs=plt.subplots(1,2,figsize=(12,5))
for r in rows:
    if r['context']!=4096:continue
    label=r['method'] if r['dataset']=='ettm2' else None;marker='o' if r['dataset']=='ettm2' else '^';color=colors[r['method']]
    axs[0].scatter(r['peak_allocated_bytes']/2**20,r['median_seconds']*1000,c=color,marker=marker,label=label,s=65)
    axs[1].scatter(r['peak_allocated_bytes']/2**20,max(r['A_error_rms']*100,1e-5),c=color,marker=marker,s=65)
axs[0].set(xlabel='Peak CUDA allocated (MiB)',ylabel='Forward/backward time (ms)',title='Actual memory / time: lower is better');axs[0].legend(fontsize=8,ncol=2)
axs[1].set(xlabel='Peak CUDA allocated (MiB)',ylabel='A-gradient RMS relative error (%)',yscale='log',title='Residual loses to local FP16')
for ax in axs:ax.grid(alpha=.2)
fig.text(.5,.01,'Context4096; circles ETTm2, triangles Electricity. Exact errors plotted at0.00001%. No optimizer updates or forecasting evaluation.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,1));(OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/tradeoffs.png',dpi=160);plt.close(fig)
print('LOCAL BACKWARD REPORT COMPLETE:',s['verdict'])
