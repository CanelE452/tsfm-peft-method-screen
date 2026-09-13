"""Verify immutable train-only diagnostic and render its own report."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from tsfm_peft_screen.memory.common import OUT,CACHE,ROOT,DATASETS
from tsfm_peft_screen.reproducibility import sha,write_json

def read(name):return json.loads((OUT/name).read_text())
a=read('stage_a_metrics.json');b=read('stage_b_metrics.json');sa=read('stage_a_status.json');sb=read('stage_b_status.json')
assert sa['status']==sb['status']=='COMPLETE' and len(a)==8 and len(b)==20
assert sa['optimizer_updates']==8 and sb['optimizer_updates']==0
assert not sa['e_access'] and not sb['e_access']
for stage in ['a','b']:
    c=read(f'stage_{stage}_contract.json')
    for file,value in c['source_hashes'].items():
        actual=hashlib.sha256(subprocess.check_output(['git','show',f"{c['execution_commit']}:{file}"],cwd=ROOT)).hexdigest()
        assert actual==value,(stage,file)
ca=read('stage_a_contract.json');cb=read('stage_b_contract.json')
for name in DATASETS:
    folder=ROOT/'data/processed'/name;meta=json.loads((folder/'manifest.json').read_text())
    assert sha(folder/'manifest.json')==ca['data_manifests'][name]
    assert sha(folder/'fit.npz')==meta['files']['fit.npz']
    assert sha(CACHE/f'{name}_warm.pt')==cb['warm_checkpoint_hashes'][name]
    assert max(ca['origins'])+48<=meta['bounds']['validation_start']
for row in b:
    assert len(row['repeats'])==3
    for repeat in row['repeats']:
        assert repeat['forward_max_abs_error']==0
        assert all(np.isfinite(v['relative_l2']) for v in repeat['gradient'].values())
        if row['method'] in ['exact','checkpoint']:assert repeat['gradient']['all']['relative_l2']<1e-6
    assert row['peak_allocated_bytes']==int(np.median([v['peak_allocated_bytes'] for v in row['repeats']]))
    assert row['median_seconds']==float(np.median([v['seconds'] for v in row['repeats']]))
    assert all(v['gradient']==row['gradient'] for v in row['repeats'])
decisions=[]
for name in DATASETS:
    r={v['method']:v for v in b if v['dataset']==name and v['context']==4096};t=r['temporal'];q=r['int8']
    reduction=1-t['peak_allocated_bytes']/r['exact']['peak_allocated_bytes']
    matched=all(x['compression']['payload_bytes']==y['compression']['payload_bytes'] for x,y in zip(t['repeats'],q['repeats']))
    passed=matched and reduction>=.30 and all(t['gradient'][k]['relative_l2']<=.01 and t['gradient'][k]['relative_l2']<=.9*q['gradient'][k]['relative_l2'] for k in ['all','A'])
    decisions.append(dict(dataset=name,peak_reduction=reduction,matched_payload_bytes=matched,pass_gate=passed))
assert decisions==sb['decisions']
assert sb['verdict']==('PILOT_ELIGIBLE' if all(d['pass_gate'] for d in decisions) else 'STOP')
# Historical results must stay byte-for-byte identical to their completed snapshot.
paths=['results/candidate_01_v2','results/candidate_05_repaired','results/screening_summary']+[f'results/candidate_{i:02}' for i in range(1,8)]
assert not subprocess.check_output(['git','diff','619e03367f52ebf11ad32ba6a5ba57563c18af9f','--',*paths],cwd=ROOT)
receipt=dict(status='PASS',stage_a_cases=8,stage_b_cases=20,stage_b_measured_passes=60,optimizer_updates=8,forward_exact=True,checkpoint_gradient_parity=True,temporal_int8_payload_bytes_matched=all(d['matched_payload_bytes'] for d in decisions),historical_results_unchanged=True,data_scope='fit.npz SHA256 verified; no E or V arrays read',verdict=sb['verdict'],metric_sha256={f:sha(OUT/f) for f in ['stage_a_metrics.json','stage_b_metrics.json','stage_a_status.json','stage_b_status.json']})
if '--verify-only' in sys.argv:
    assert read('verification.json')==receipt
    print('MEMORY FEASIBILITY VERIFICATION PASS:',sb['verdict']);raise SystemExit(0)
write_json(OUT/'verification.json',receipt)
lines=['# Memory-efficient TSFM PEFT feasibility: '+sb['verdict'],'',
'**The activation bottleneck is real; this adjacent-pair temporal compression primitive fails the fixed gradient gate. No learning pilot was launched.** This does not rule out every possible temporal method. It provides no forecasting-accuracy or publication-success evidence.','',
'## Scope and controls','',
'Pinned Chronos-2, rank8 attention LoRA, float32 computation, batch8, horizon48, first4 channels of ETTm2/Electricity, fit-only origins4352/4608. Four warmup updates per dataset ensure nonzero A gradients. Stage B fixes the warm state and makes zero optimizer updates. Three timing repeats use the same batch, not independent seeds. AdamW optimizer states are resident in Stage A and omitted for all Stage-B arms; compare peak reductions within Stage B. CUDA peak allocated includes model, gradients and temporary compression/reconstruction buffers, but not all driver allocations or optimizer-step peak.','',
'Compression changes saved backward activations only. All measured forward outputs are bit-identical. Hidden temporal pair means in FP16 and per-channel INT8 have exactly equal retained bytes, including exact last4 REG/future tokens and explicit temporal scale-sized padding. Both use FP16 elsewhere, preserve ReLU zero/sign status, deduplicate backing storage and restore original views. Checkpointing recomputes whole encoder blocks exactly. These are scoped primitives, not official LoRAct/CARE/VeLoRA reproductions.','',
'## Actual activation bottleneck','',
'| Dataset | Context | CUDA peak allocated MiB | Unique saved activation MiB |','|---|---:|---:|---:|']
for r in a:lines.append(f"| {r['dataset']} | {r['context']} | {r['peak_allocated_bytes']/2**20:.1f} | {r['unique_saved_activation_bytes']/2**20:.1f} |")
lines+=['','Saved activation storage is deduplicated by storage; its sum is not itself a measured peak.','',
'## Fixed-state comparison at context4096','',
'| Dataset | Method | Peak MiB | Peak reduction | Full gradient relative L2 | A gradient relative L2 | Step time ms | Time / exact |','|---|---|---:|---:|---:|---:|---:|---:|']
for name in DATASETS:
    r={v['method']:v for v in b if v['dataset']==name and v['context']==4096}
    for method in cb['methods']:
        v=r[method];lines.append(f"| {name} | {method} | {v['peak_allocated_bytes']/2**20:.1f} | {100*(1-v['peak_allocated_bytes']/r['exact']['peak_allocated_bytes']):.2f}% | {100*v['gradient']['all']['relative_l2']:.4f}% | {100*v['gradient']['A']['relative_l2']:.4f}% | {1000*v['median_seconds']:.2f} | {v['median_seconds']/r['exact']['median_seconds']:.2f}× |")
lines+=['','Relative gradient error is not forecast-loss degradation. Timing includes Python hooks for compressed arms and recomputation for checkpointing. Small batches, two origins and one warmed initialization limit generalization. Full context1024 results and all repetitions are in `stage_b_metrics.json`.','',
'## Decision and accounting','',
'Gate fixed before measurement: both datasets at4096 must show at least30% peak reduction, at most1% full and A-gradient relative error, and both errors at least10% lower than byte-matched INT8. The temporal primitive fails. Checkpointing and generic compression are essential controls for any future memory proposal. There is no evidence to fund a learning pilot for this primitive; CARE or more elaborate temporal variants were not implemented after this gate failure.','',
'New work:8 diagnostic warmup optimizer updates; Stage A8 inventory and24 measured backward passes; completed Stage B4 reference and60 measured backward passes. One earlier implementation attempt completed1 reference and2 exact-control backwards before an observer-hook error; its contract, traceback and status are preserved in `implementation_attempt_01/`. It added no optimizer updates. The fix did not change the gate, data or compression primitive. Historical totals remain46 completed fits and9 stream attempts (8 complete,1 historical abort). No new fitted accuracy result, V/E evaluation or Round2 was produced.','',
f"Stage A execution: `{ca['execution_commit']}`. Stage B completed execution: `{cb['execution_commit']}`. Source hashes, data hashes, fixed warm checkpoint hashes, exact-control parity, repeated metrics and historical artifact invariance are checked by `scripts/finalize_memory_feasibility.py --verify-only`. Raw data/model/adapters remain ignored local caches.",'',
'Prior boundaries: [LoRAct](https://arxiv.org/html/2509.23472v1), [CARE-LoRA](https://arxiv.org/html/2607.11940v1), [VeLoRA](https://arxiv.org/abs/2405.17991). These already study generic or LoRA-local activation compression; merely applying temporal pooling to a TSFM is not established novelty.','',
'![Memory and gradient diagnostic](figures/diagnostic.png)','']
(OUT/'RESULT.md').write_text('\n'.join(lines))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10})
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
for name in DATASETS:
    r=[v for v in a if v['dataset']==name]
    axs[0].plot([v['context'] for v in r],[v['peak_allocated_bytes']/2**20 for v in r],marker='o',label=name)
axs[0].set(xlabel='Context length',ylabel='CUDA peak allocated (MiB)',title='Stage A: activation memory grows with context');axs[0].legend();axs[0].grid(alpha=.2)
colors=dict(exact='#444444',checkpoint='#2a9d8f',fp16='#457b9d',int8='#e9a23b',temporal='#d1495b')
for r in b:
    if r['context']!=4096:continue
    axs[1].scatter(r['peak_allocated_bytes']/2**20,max(r['gradient']['all']['relative_l2']*100,1e-5),marker='o' if r['dataset']=='ettm2' else '^',color=colors[r['method']],s=65,label=r['method'] if r['dataset']=='ettm2' else None)
axs[1].set(xlabel='CUDA peak allocated (MiB)',ylabel='Full gradient relative L2 error (%)',title='Stage B: temporal pooling loses to INT8',yscale='log')
axs[1].axhline(1,color='#777777',linestyle='--',linewidth=1,label='1% error gate')
axs[1].legend(fontsize=8);axs[1].grid(alpha=.2)
fig.text(.5,.01,'Context4096; circles ETTm2, triangles Electricity. Exact errors plotted at 0.00001%. No forecast accuracy test.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,1));(OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/diagnostic.png',dpi=160);plt.close(fig)
print('MEMORY FEASIBILITY REPORT COMPLETE:',sb['verdict'])
