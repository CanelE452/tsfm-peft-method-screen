"""CPU equivalence of the portable fixed MAG implementation; no fitting/scoring."""
import ast,copy,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from tsfm_peft_screen.mag.model import MAGForecast,load_bundle,save_bundle,state_hash,file_hash
from experiments.temporal_response_peft_20260919 import common as old
from experiments.additive_persistence_validation_v1_20260917 import common as parent
OUT=ROOT/'results/mag_portable_20260919';CACHE=ROOT/'.cache/mag_portable_20260919'
def save(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 if (OUT/'AUDIT.json').exists():print('ALREADY_VERIFIED_NO_REPEAT');return
 assert not torch.cuda.is_initialized()
 torch.set_num_threads(4)
 # An accidental optimization path must fail before it can run.
 def forbidden(*args,**kwargs):raise RuntimeError('USER_NO_TRAINING: optimizer/backward forbidden')
 torch.optim.Optimizer.__init__=forbidden;torch.Tensor.backward=forbidden
 CACHE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
 receipt=json.loads((ROOT/'results/outlier_signal_followup_v2_20260917/download_receipts.json').read_text())['amazon/chronos-bolt-small'];snapshot=ROOT/receipt['snapshot']
 rows=[r for r in json.loads((ROOT/'results/mag_inference_cost_20260919/SEAL.json').read_text())['models'] if r['arm']=='MAG_ONLY'];assert len(rows)==4
 provenance={str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/'src/tsfm_peft_screen/mag').glob('*.py')}
 for path,value in receipt['files'].items():assert file_hash(ROOT/path)==value;provenance[path]=value
 for p in [Path(__file__),ROOT/'tests/test_mag_portable.py',ROOT/'results/mag_inference_cost_20260919/SEAL.json']:
  provenance[str(p.relative_to(ROOT))]=file_hash(p)
 records=[];calls=0
 for r in rows:
  assert file_hash(ROOT/r['checkpoint'])==r['sha256'];provenance[r['checkpoint']]=r['sha256']
  b=parent.baseline_row(r['source'],r['seed']);assert file_hash(ROOT/b['checkpoint'])==b['sha256'];provenance[b['checkpoint']]=b['sha256']
  folder=parent.CACHE/'conditions'/r['source'];indices=np.arange(8)*33
  for name in ['train_x.npy','train_sigma.npy']:provenance[str((folder/name).relative_to(ROOT))]=file_hash(folder/name)
  x=torch.from_numpy(np.array(np.load(folder/'train_x.npy',mmap_mode='r')[0,indices],copy=True));sigma=torch.from_numpy(np.array(np.load(folder/'train_sigma.npy',mmap_mode='r')[indices],copy=True))
  reference=old.build('MAG_ONLY',r['seed'],r['source'],device='cpu')
  parent.restore(reference,torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True));before=state_hash(reference.state_dict());input_hash=state_hash({'x':x,'sigma':sigma})
  portable=MAGForecast(copy.deepcopy(reference.base),r['seed']);portable.adapter.load_state_dict(reference.adapter.state_dict(),strict=True)
  assert not any(p.requires_grad for p in portable.base.parameters())
  assert sum(p.numel() for p in portable.parameters() if p.requires_grad)==8712
  with torch.inference_mode():
   expected=reference(x,sigma);actual=portable(x,sigma);calls+=2;assert torch.equal(expected,actual)
   off=reference(x,sigma,residual_mode='off');off2=portable(x,sigma,adapter_off=True);calls+=2;assert torch.equal(off,off2)
  bundle=CACHE/f"{r['source']}_{r['seed']}.pt"
  assert not bundle.exists(),'Preserve previous exported bundle; no silent overwrite'
  save_bundle(portable,bundle,snapshot,dict(source=r['source'],seed=r['seed'],selected_step=r['step'],adapter_sha256=r['sha256'],B0_sha256=b['sha256']))
  fresh=load_bundle(snapshot,bundle)
  with torch.inference_mode():replayed=fresh(x,sigma);calls+=1;assert torch.equal(expected,replayed)
  assert state_hash(reference.state_dict())==before
  assert state_hash(portable.state_dict())==state_hash(fresh.state_dict())
  assert input_hash==state_hash({'x':x,'sigma':sigma})
  records.append(dict(source=r['source'],seed=r['seed'],step=r['step'],input_examples=8,legacy_exact=True,adapter_off_exact=True,bundle_restore_exact=True,frozen_and_inputs_unchanged=True,base_hash=state_hash(fresh.base.state_dict()),model_hash=state_hash(fresh.state_dict()),bundle=str(bundle.relative_to(ROOT)),bundle_sha256=file_hash(bundle),bundle_bytes=bundle.stat().st_size,trainable_parameters=8712))
  if len(records)==1:
   np.savez(CACHE/'cli_inputs.npz',observed=x.numpy(),sigma=sigma.numpy())
   env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
   command=[str(ROOT/'.venv/bin/python'),'-m','tsfm_peft_screen.mag','--snapshot',str(snapshot),'--bundle',str(bundle),'--input',str(CACHE/'cli_inputs.npz'),'--output',str(CACHE/'cli_predictions.npy')]
   result=subprocess.run(command,cwd='/tmp',env=env,capture_output=True,text=True,check=True)
   assert np.array_equal(np.load(CACHE/'cli_predictions.npy'),expected.numpy());calls+=1
   save(OUT/'CLI_CHECK.json',dict(cwd='/tmp',stdout=result.stdout,exact=True,CPU_only=True,command=command))
   # CPU-only rejection test; no model forward or optimizer in this path.
   payload=torch.load(bundle,map_location='cpu',weights_only=True);payload['config']['threshold']=4.;bad=CACHE/'invalid_config.pt';torch.save(payload,bad)
   try:load_bundle(snapshot,bad)
   except ValueError as e:assert 'contract' in str(e)
   else:raise AssertionError('Changed method contract accepted')
  print('CPU_PORTABLE_EXACT',r['source'],r['seed'],r['step'],flush=True)
  del reference,portable,fresh
 save(OUT/'MODELS.json',records)
 assert calls==21 and not torch.cuda.is_initialized()
 for name,value in provenance.items():assert file_hash(ROOT/name)==value,name
 tree=ast.parse((ROOT/'src/tsfm_peft_screen/mag/model.py').read_text())
 imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
 assert not any(n.startswith(('experiments','scripts','research')) for n in imports)
 save(OUT/'AUDIT.json',dict(status='VERIFIED_PORTABLE_FIXED_MAG_CPU',models=4,legacy_exact=True,bundle_restore_exact=True,adapter_off_exact=True,cli_from_outside_repo_exact=True,invalid_config_rejected=True,cpu_forward_calls=calls,optimizer_updates=0,backward_calls=0,GPU_initialized=False,E_forward_calls=0,labels_read=False,new_candidates=0,method_changed=False,checkpoint_selection_changed=False,source_hashes=provenance,dependency_scope='core only torch; loader additionally installed chronos/transformers and a hash-matched local snapshot',methodology_goal_complete=False))
 print('VERIFIED',calls,'CPU forwards; updates=0; GPU initialized=False')
if __name__=='__main__':main()
