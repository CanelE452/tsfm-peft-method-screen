"""Bounded train-only bottleneck diagnosis: eight warmup updates total."""
import fcntl,gc,json,subprocess,time
import numpy as np
import torch
from tsfm_peft_screen.memory.common import *
from tsfm_peft_screen.reproducibility import sha,write_json,source_hashes
from tsfm_peft_screen.lora import snapshot
lock=open(ROOT/'.cache/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not OUT.exists(),'Do not overwrite a diagnostic stage'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'Commit before GPU work'
idle=None
while True:
    p=subprocess.run([str(ROOT/'scripts/with_cuda.sh'),'nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if p.stdout.strip():idle=None;print('WAIT external GPU job',flush=True)
    else:
        if idle is None:idle=time.monotonic()
        if time.monotonic()-idle>=30:break
    time.sleep(10)
OUT.mkdir();CACHE.mkdir();start=time.monotonic()
contract=dict(stage='A_memory_inventory',execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes=source_hashes(),datasets=DATASETS,context_lengths=LENGTHS,origins=ORIGINS,first_channels=4,batch_series=8,horizon=48,warmup_updates_per_dataset=4,warmup_context=1024,lr=1e-4,rank=8,precision='float32',measured_passes_per_case=3,profile_passes_per_case=1,data_scope='fit.npz only; both origins and labels strictly before V; no E access',candidate_gate='Use actual peak memory and gradient errors; do not equate summed saved tensors with peak. Target whole-GPU reduction30% with forecast degradation<=0.5% remains unproven.',data_manifests={n:sha(ROOT/'data/processed'/n/'manifest.json') for n in DATASETS})
write_json(OUT/'stage_a_contract.json',contract);rows=[]
try:
    for name in DATASETS:
        m=build();x,y,g=batch(name,1024);opt=torch.optim.AdamW(params(m),lr=1e-4,weight_decay=0)
        for i in range(4):
            m.zero_grad(set_to_none=True);backward(m,x,y,g);torch.nn.utils.clip_grad_norm_(params(m),1);opt.step();guard(start)
        torch.save(snapshot(m),CACHE/f'{name}_warm.pt')
        static=sum(p.numel()*p.element_size() for p in m.parameters())
        optim=sum(t.numel()*t.element_size() for state in opt.state.values() for t in state.values() if isinstance(t,torch.Tensor) and t.is_cuda)
        for length in LENGTHS:
            x,y,g=batch(name,length);zero(m);torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            with SavedInventory(m) as inv:loss=backward(m,x,y,g)
            torch.cuda.synchronize();inventory=inv.result();write_json(OUT/f'inventory_{name}_{length}.json',inventory)
            torch.save(gradient(m),CACHE/f'{name}_{length}_gradient.pt')
            peaks=[];seconds=[]
            for repeat in range(3):
                zero(m);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
                actual=backward(m,x,y,g);torch.cuda.synchronize();seconds.append(time.monotonic()-t);peaks.append(torch.cuda.max_memory_allocated());assert actual==loss
                guard(start)
            row=dict(dataset=name,context=length,loss=loss,model_parameter_bytes=static,optimizer_state_bytes=optim,before_forward_allocated_bytes=before,peak_allocated_bytes=int(np.median(peaks)),peak_reserved_bytes=torch.cuda.max_memory_reserved(),median_forward_backward_seconds=float(np.median(seconds)),unique_saved_activation_bytes=inventory['unique_nonparameter_storage_bytes'],timings_seconds=seconds,peaks_bytes=peaks,warm_checkpoint_sha256=sha(CACHE/f'{name}_warm.pt'))
            rows.append(row);write_json(OUT/'stage_a_metrics.json',rows);print('PROFILE',name,length,'peak MiB',round(row['peak_allocated_bytes']/2**20,1),'saved MiB',round(row['unique_saved_activation_bytes']/2**20,1),flush=True)
        del m,opt,x,y,g;gc.collect();torch.cuda.empty_cache()
    assert source_hashes()==contract['source_hashes']
    write_json(OUT/'stage_a_status.json',dict(status='COMPLETE',cases=len(rows),optimizer_updates=8,wall_seconds=time.monotonic()-start,e_access=False))
except Exception as e:
    write_json(OUT/'stage_a_status.json',dict(status='IMPLEMENTATION_BLOCKED',error=str(e),completed_cases=len(rows)));raise
