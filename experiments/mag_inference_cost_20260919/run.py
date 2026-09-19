"""Same-condition inference measurement of existing models; no optimizer or labels."""
import os,json,time,traceback,hashlib,subprocess,collections
from pathlib import Path
import numpy as np
import torch
from experiments.additive_persistence_validation_v1_20260917 import common as base
from experiments.temporal_response_peft_20260919 import common as old
from experiments.learned_gate_comparison_20260919 import common as gate
from experiments.petsa_cell_comparison_20260919 import common as petsa
ROOT=base.ROOT;NAME='mag_inference_cost_20260919';EXP=ROOT/'experiments'/NAME;OUT=ROOT/'results'/NAME
ARMS=['B0','PLAIN','MAG_ONLY','TOKEN_GATE','TOKEN_GATE_ENTROPY','PETSA_XY_OFFLINE']
SOURCES=['electricity','ettm1'];SEEDS=[81551,81552];IDX=np.arange(32)*33
sha=base.sha;save=base.save;read=base.read
class Watch(base.Watch):
 def __init__(self):
  self.last_disk=0;base.OriginalWatch.__init__(self,OUT,'inference',wall_cap=3600)
 def limits(self):base.OriginalWatch.limits(self)
def state(m):return base.tensor_hash(m.state_dict())
def models():
 groups={a:read((petsa.OUT if a in petsa.ARMS else gate.OUT if a in gate.ARMS else old.OUT)/'MODEL_SELECTION.json') for a in ARMS};chosen=[]
 for source in SOURCES:
  for seed in SEEDS:
   for arm in ARMS:
    z=[r for r in groups[arm] if r['source']==source and r['seed']==seed and r['arm']==arm and r['stage']=='selected'];assert len(z)==1,(source,seed,arm,len(z))
    r=dict(z[0]);r['B0_sha256']=base.baseline_row(source,seed)['sha256'];chosen.append(r)
 return chosen
def load(r):
 c=petsa if r['arm'] in petsa.ARMS else gate if r['arm'] in gate.ARMS else old
 m=c.build(r['arm'],r['seed'],r['source'],device='cpu')
 if r['arm']!='B0':base.restore(m,torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True))
 return m.eval()
def used_code():
 files=[ROOT/'scripts/priority12/common.py']+list(EXP.glob('*.py'))+[EXP/'PROTOCOL.md']
 for folder in ['additive_persistence_validation_v1_20260917','temporal_response_peft_20260919','c3_training_factorial_20260918','c3_weakness_controls_20260918','outlier_signal_followup_v2_20260917','outlier_signal_peft_v1_20260917','learned_gate_comparison_20260919','petsa_cell_comparison_20260919']:
  files.extend((ROOT/'experiments'/folder).glob('*.py'))
 return sorted(set(files))
def prepare():
 if (OUT/'SEAL.json').exists():return read(OUT/'SEAL.json')
 rows=models();hashes={str(p.relative_to(ROOT)):sha(p) for p in used_code()};audit={}
 for r in rows:
  p=ROOT/r['checkpoint'];assert sha(p)==r['sha256'];hashes[str(p.relative_to(ROOT))]=r['sha256']
  b=base.baseline_row(r['source'],r['seed']);assert sha(ROOT/b['checkpoint'])==b['sha256'];hashes[b['checkpoint']]=b['sha256']
 for source in SOURCES:
  folder=base.CACHE/'conditions'/source
  for name in ['train_x.npy','train_sigma.npy','train_generator_audit.json']:
   p=folder/name;hashes[str(p.relative_to(ROOT))]=sha(p)
  records=[r for r in read(folder/'train_generator_audit.json') if r['epoch']==0 and r['example'] in IDX]
  assert len(records)==32
  audit[source]={'indices':IDX.tolist(),'states':dict(collections.Counter(r['state'] for r in records)),'amplitudes':dict(collections.Counter(str(r['amplitude']) for r in records))}
 seal=dict(at=time.time(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),models=rows,hashes=hashes,samples=audit,optimizer_updates=0,timed_cap=9216,warmup_cap=864,parity_cap=48,total_forward_cap=10128,scope='existing_model_inference_only',user_authorization='계속해 나한테 대기 물어보지마 goal 해결이 목표니까 계속하라고')
 save(OUT/'SEAL.json',seal);return seal

def main():
 seal=prepare()
 if (OUT/'AUDIT.json').exists():print('ALREADY_COMPLETE_NO_REPEAT');return
 assert not (OUT/'TIMINGS.jsonl').exists(),'Existing measurements preserved; no unplanned repeats'
 for p,h in seal['hashes'].items():assert sha(ROOT/p)==h,p
 base.setup();os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 forward=0;timed=0;warmup=0;parity=0;checks=[];watch=Watch()
 def status(**kw):save(OUT/'status.json',dict(at=time.time(),execution='MEASURING',timed_calls=timed,warmup_calls=warmup,parity_calls=parity,forwards=forward,optimizer_updates=0,**kw))
 try:
  watch.boundary(startup=True)
  save(OUT/'ENVIRONMENT.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),threads=torch.get_num_threads(),tf32=False,inference_mode=True,fp32=True,nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True)))
  for source in SOURCES:
   f=base.CACHE/'conditions'/source
   xx=np.array(np.load(f/'train_x.npy',mmap_mode='r')[0,IDX],copy=True);ss=np.array(np.load(f/'train_sigma.npy',mmap_mode='r')[IDX],copy=True)
   assert xx.shape==(32,512) and ss.shape==(32,)
   for seed in SEEDS:
    rows=[r for r in seal['models'] if r['source']==source and r['seed']==seed]
    ms={r['arm']:load(r) for r in rows};hashes={a:state(m) for a,m in ms.items()};reference={}
    counts={a:sum(p.numel() for p in base.parameters(m).values()) for a,m in ms.items()}
    x=torch.from_numpy(xx).cuda();s=torch.from_numpy(ss).cuda();input_hash=base.tensor_hash({'x':x,'sigma':s})
    with torch.inference_mode():
     for round_id in range(6):
      order=ARMS[round_id:]+ARMS[:round_id]
      for position,arm in enumerate(order):
       watch.boundary();m=ms[arm].cuda().eval()
       if round_id==0:
        reference[arm]=m(x,s).cpu();forward+=1;parity+=1;assert torch.isfinite(reference[arm]).all()
       for batch in ([1,32] if round_id%2==0 else [32,1]):
        watch.boundary();start=watch.sample();assert not start['busy']
        for k in range(3):
         p=m(x[k:k+1] if batch==1 else x,s[k:k+1] if batch==1 else s);forward+=1;warmup+=1;del p
        torch.cuda.synchronize();records=[]
        for k in range(32):
         xi=x[k:k+1] if batch==1 else x;si=s[k:k+1] if batch==1 else s
         ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True)
         torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();allocated=torch.cuda.memory_allocated()
         wall=time.perf_counter();ev0.record();p=m(xi,si);ev1.record();ev1.synchronize();wall_ms=(time.perf_counter()-wall)*1000
         event_ms=ev0.elapsed_time(ev1);peak=torch.cuda.max_memory_allocated();forward+=1;timed+=1
         assert forward<=seal['total_forward_cap'] and timed<=seal['timed_cap'];assert p.shape==(batch,9,64) and torch.isfinite(p).all()
         records.append(dict(source=source,seed=seed,arm=arm,round=round_id,position=position,batch=batch,repeat=k,wall_ms=wall_ms,cuda_stream_ms=event_ms,allocated_baseline=allocated,peak_allocated=peak,transient_allocated=peak-allocated,adaptation_parameters=counts[arm],input_hash=input_hash))
         del p
        end,contaminated=watch.after(start);assert not contaminated,'EXTERNAL_GPU_COMPUTE_MEASUREMENT_INVALID'
        for record in records:base.append(OUT/'TIMINGS.jsonl',record)
        base.append(OUT/'BLOCKS.jsonl',dict(source=source,seed=seed,arm=arm,round=round_id,batch=batch,started_at=start['at'],ended_at=end['at'],contaminated=False))
        status(source=source,seed=seed,arm=arm,round=round_id,batch=batch)
       if round_id==5:
        p=m(x,s).cpu();forward+=1;parity+=1;assert torch.equal(p,reference[arm]);del p
       if getattr(m,'adapter',None) is not None and hasattr(m.adapter,'last_gate'):m.adapter.last_gate=None
       m.cpu();base.cleanup()
      print('ROUND_COMPLETE',source,seed,round_id,'timed',timed,flush=True)
    assert input_hash==base.tensor_hash({'x':x,'sigma':s})
    for arm,m in ms.items():
     assert hashes[arm]==state(m);checks.append(dict(source=source,seed=seed,arm=arm,state_unchanged=True,output_exact=True,state_sha256=hashes[arm],input_unchanged=True,adaptation_parameters=counts[arm]))
    save(OUT/'MODEL_CHECKS.json',checks);del ms,m,x,s,reference;base.cleanup()
  for p,h in seal['hashes'].items():assert sha(ROOT/p)==h,p
  assert (timed,warmup,parity,forward)==(9216,864,48,10128)
  save(OUT/'COUNTS.json',dict(timed=timed,warmup=warmup,parity=parity,total=forward,optimizer_updates=0,backward_calls=0,E_forwards=0))
  save(OUT/'status.json',dict(execution='COMPLETE_MEASUREMENT',at=time.time(),optimizer_updates=0,total_forwards=forward))
 except BaseException as e:
  save(OUT/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),timed=timed,warmup=warmup,parity=parity,forward=forward));save(OUT/'status.json',dict(execution='BLOCKED_MEASUREMENT',error=str(e)));raise
 finally:watch.close()
if __name__=='__main__':main()
