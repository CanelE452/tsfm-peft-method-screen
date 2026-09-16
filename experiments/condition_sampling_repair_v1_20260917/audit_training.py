"""Audit completed neural tracks: exact streams, initializations and full resume snapshots."""
import sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from experiments.condition_sampling_repair_v1_20260917.common import *
from experiments.peft_rank12_20260915.common import tensor_hash

def run():
 results=[];initial={}
 for t in SOURCES:
  if read(OUT/t/'STATUS.json')['EXECUTION']!='COMPLETE':continue
  fits=read(OUT/t/'fits.json');logs=[json.loads(s) for s in (OUT/t/'optimizer_log.jsonl').read_text().splitlines()];packet=np.load(CACHE/t/'TRAIN_inputs.npz');origins=packet['origins'];verified=0;source_counts=[]
  for f in fits:
   if f['status']!='COMPLETE':continue
   initial.setdefault(f['seed'],set()).add(f['initial_lora_hash']);rows=[r for r in logs if r['id']==f['id']];assert len(rows)==512;used=[]
   for epoch in range(8):
    order=rng('ORDER',f['seed'],epoch).permutation(64)
    if f['arm']=='I0':order=np.tile(rng('ORDER',f['seed'],epoch).permutation(np.flatnonzero(packet['detailed'])),4)
    for k,i in enumerate(order):
     r=rows[epoch*64+k];assert r['step']==epoch*64+k+1 and r['epoch']==epoch;assert r['origin_ordinal']==int(i) and r['origin']==int(origins[i]);used.append(int(i))
     if t=='N01':assert r['target']==int(i)%4 and r['delay']==[0,6,24][epoch%3]
     if t=='R09':assert r['detailed']==bool(packet['detailed'][i])
   ids,counts=np.unique(used,return_counts=True);assert len(ids)==(16 if f['arm']=='I0' else 64);assert (counts==(32 if f['arm']=='I0' else 8)).all()
   cp=next(c for c in f['checkpoints'] if c['step']==512);assert sha(ROOT/cp['path'])==cp['sha256'];state=torch.load(CACHE/t/'resume'/f"{f['id']}.pt",map_location='cpu',weights_only=False);assert state['step']==512 and state['seal_sha256']==sha(OUT/'MASTER_SEAL.json');final=torch.load(ROOT/cp['path'],map_location='cpu',weights_only=True);assert final.keys()==state['parameters'].keys();assert all(torch.equal(v,state['parameters'][k]) for k,v in final.items());assert tensor_hash(final)==cp['parameter_hash'];assert len(state['optimizer']['state'])==len(final);assert set(state['rng'])=={'python','numpy','torch','cuda'};verified+=1
   source_counts.append(dict(id=f['id'],unique_origins=len(ids),each_occurrences=int(counts[0]),total_updates=len(rows),contaminated_updates=sum(r['contaminated'] for r in rows),clip_fraction=float(np.mean([r['clip_active'] for r in rows]))));del state,final
  save(OUT/t/'training_record_verification.json',dict(passed=True,complete_resume_states_verified=verified,exact_optimizer_stream=True,exact_final_resume_parameters=True,optimizer_and_rng_present=True,by_fit=source_counts));csvwrite(OUT/t/'training_exposures.csv',source_counts);results.append(dict(track=t,verified_fits=verified))
 assert all(len(v)==1 for v in initial.values()),initial
 save(OUT/'training_record_verification.json',dict(passed=True,at=time.time(),tracks=results,common_seed_initial_LoRA_hash={str(k):next(iter(v)) for k,v in initial.items()},no_extra_optimization=True,no_model_forward=True));print('TRAINING_RECORD_AUDIT',results)
if __name__=='__main__':run()
