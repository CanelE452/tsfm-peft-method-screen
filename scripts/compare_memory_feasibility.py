"""Fixed Stage-B diagnostic; zero optimizer updates, train-only probes."""
import fcntl,gc,json,subprocess,time
from contextlib import nullcontext
import numpy as np
import torch
from tsfm_peft_screen.memory.common import *
from tsfm_peft_screen.memory.compression import CompressedSaved,checkpoint_blocks
from tsfm_peft_screen.reproducibility import sha,write_json,source_hashes

METHODS=['exact','checkpoint','fp16','int8','temporal']
def errors(actual,reference):
    out={}
    assert actual.keys()==reference.keys()
    for label in ['all','A','B']:
        names=[n for n in reference if label=='all' or n.endswith('lora_'+label)]
        a=torch.cat([actual[n].flatten().double() for n in names]);r=torch.cat([reference[n].flatten().double() for n in names])
        assert r.norm()>0 and torch.isfinite(a).all()
        out[label]=dict(relative_l2=float((a-r).norm()/r.norm()),cosine=float(torch.nn.functional.cosine_similarity(a,r,dim=0)),reference_norm=float(r.norm()))
    return out

lock=open(ROOT/'.cache/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'stage_b_contract.json').exists(),'Immutable stage'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'Commit before GPU work'
assert json.loads((OUT/'stage_a_status.json').read_text())['status']=='COMPLETE'
idle=None
while True:
    p=subprocess.run([str(ROOT/'scripts/with_cuda.sh'),'nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if p.stdout.strip():idle=None;print('WAIT external GPU job',flush=True)
    else:
        if idle is None:idle=time.monotonic()
        if time.monotonic()-idle>=30:break
    time.sleep(10)
start=time.monotonic();rows=[]
contract=dict(stage='B_gradient_memory',execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes=source_hashes(),methods=METHODS,contexts=[1024,4096],repeats=3,optimizer_updates=0,origins=ORIGINS,data_scope='fit.npz only; no E or V evaluation',warm_checkpoint_hashes={n:sha(CACHE/f'{n}_warm.pt') for n in DATASETS},gate=dict(peak_reduction_min=.30,full_and_A_relative_gradient_error_max=.01,temporal_vs_byte_matched_int8_error_ratio_max=.9,required='both datasets at context4096; all conditions; forward exact; checkpoint gradient parity; no learning if any fails'),measurement='Optimizer states omitted for ALL Stage-B arms. Same fixed warmed adapters. Timings include hooks/compression/reconstruction and exact checkpoint recomputation. Three rotated-order repeats. Stage-A timing/peaks not direct denominators.',compression='Recognizable d768 hidden storage only: temporal adjacent-pair means FP16 vs INT8 per-channel scales, same retained bytes including last4 exact tokens and explicit temporal padding. Other large float32 nonparameter storage uses FP16 in both; ReLU sign preserved. Generic FP16 compresses all large nonparameter storage. No official method reproduction.')
write_json(OUT/'stage_b_contract.json',contract)
try:
    for name in DATASETS:
        m=build();restore(torch.load(CACHE/f'{name}_warm.pt',weights_only=True),m)
        for length in [1024,4096]:
            x,y,g=batch(name,length);zero(m);backward(m,x,y,g);reference=gradient(m)
            cached=torch.load(CACHE/f'{name}_{length}_gradient.pt',weights_only=True)
            assert errors(reference,cached)['all']['relative_l2']==0
            with torch.no_grad():expected=forward(m,x,g)[0].cpu()
            results={method:[] for method in METHODS}
            for repeat in range(3):
                order=METHODS[repeat:]+METHODS[:repeat]
                for method in order:
                    zero(m);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
                    ctx=checkpoint_blocks(m) if method=='checkpoint' else CompressedSaved(m,method,len(x),length//16+4) if method not in ('exact','checkpoint') else nullcontext()
                    with ctx:
                        z,p,l,s=forward(m,x,g);loss=native_loss(z,y,l,s);loss.backward()
                    torch.cuda.synchronize();elapsed=time.monotonic()-tick;peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
                    forward_error=float((z.detach().cpu()-expected).abs().max());assert forward_error==0
                    e=errors(gradient(m),reference);assert all(torch.isfinite(v.grad).all() for v in params(m))
                    if method in ('exact','checkpoint'):assert e['all']['relative_l2']<1e-6
                    stats=ctx.stats.copy() if isinstance(ctx,CompressedSaved) else None
                    results[method].append(dict(seconds=elapsed,peak_allocated_bytes=peak,peak_reserved_bytes=reserved,loss=float(loss.detach()),forward_max_abs_error=forward_error,gradient=e,compression=stats))
                    del z,p,l,s,loss,ctx;guard(start)
                    print('COMPARE',name,length,repeat,method,'peakMiB',round(peak/2**20,1),'graderr',round(e['all']['relative_l2'],6),flush=True)
            for method in METHODS:
                runs=results[method];rows.append(dict(dataset=name,context=length,method=method,peak_allocated_bytes=int(np.median([r['peak_allocated_bytes'] for r in runs])),median_seconds=float(np.median([r['seconds'] for r in runs])),gradient=runs[0]['gradient'],repeats=runs))
            write_json(OUT/'stage_b_metrics.json',rows)
        del m,x,y,g;gc.collect();torch.cuda.empty_cache()
    assert source_hashes()==contract['source_hashes']
    decisions=[]
    for name in DATASETS:
        r={v['method']:v for v in rows if v['dataset']==name and v['context']==4096};t=r['temporal'];q=r['int8']
        reduction=1-t['peak_allocated_bytes']/r['exact']['peak_allocated_bytes']
        matched=all(a['compression']['payload_bytes']==b['compression']['payload_bytes'] for a,b in zip(t['repeats'],q['repeats']))
        passed=matched and reduction>=.30 and all(t['gradient'][k]['relative_l2']<=.01 and t['gradient'][k]['relative_l2']<=.9*q['gradient'][k]['relative_l2'] for k in ['all','A'])
        decisions.append(dict(dataset=name,peak_reduction=reduction,matched_payload_bytes=matched,pass_gate=passed))
    write_json(OUT/'stage_b_status.json',dict(status='COMPLETE',verdict='PILOT_ELIGIBLE' if all(d['pass_gate'] for d in decisions) else 'STOP',decisions=decisions,cases=len(rows),measured_backward_passes=60,optimizer_updates=0,e_access=False,wall_seconds=time.monotonic()-start))
except Exception as e:
    write_json(OUT/'stage_b_status.json',dict(status='IMPLEMENTATION_BLOCKED',error=str(e),completed_cases=len(rows)));raise
