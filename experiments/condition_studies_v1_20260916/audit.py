"""Seal before execution; independent ledger, replay and preserved-history audit."""
import time,sys,subprocess
import pandas as pd
from .common import *

def seal():
 assert not (OUT/'MASTER_SEAL.json').exists(),'Existing seal is immutable'
 from .report import literature,report_track,report_all
 from .checks import check_all
 literature();checks=check_all()
 files={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name in ['MASTER_PROTOCOL.md','MASTER_MANIFEST.json','PROTOCOL.json','origins.json','origins.csv','data_receipt.json','permissions.json','train_statistics.json','feature_or_transform_manifest.json','TOPIC_ONEPAGE.md','LITERATURE_BOUNDARY.md','input_verification.json','historical_hashes.json']}
 for spec in read(OUT/'MASTER_MANIFEST.json')['specs']:
  files.update(spec['data'].get('packet_hashes',{}));files.update(spec['data'].get('dependencies',{}))
 import chronos.chronos2.model as model,chronos.chronos2.preprocess as prep
 from tsfm_peft_screen.backbone import REVISION
 code=list(EXP.glob('*.py'))+[ROOT/'scripts/run_condition_studies.py',ROOT/'src/tsfm_peft_screen/backbone.py',ROOT/'src/tsfm_peft_screen/lora.py',ROOT/'scripts/priority12/common.py',ROOT/'experiments/peft_rank12_20260915/common.py']
 implementation={str(p.relative_to(ROOT)):sha(p) for p in code};modelpath=Path.home()/'.cache/huggingface/hub/models--amazon--chronos-2/snapshots'/REVISION
 assert modelpath.exists(),'BLOCKED_MODEL pinned snapshot unavailable'
 save(OUT/'MASTER_SEAL.json',dict(at=time.time(),files=files,implementation=implementation,installed_code={str(p):sha(p) for p in [Path(model.__file__),Path(prep.__file__)]},model_files={str(p):sha(p) for p in modelpath.iterdir() if p.is_file()},all_nine_specs_fixed=True,training_updates_before_seal=0,E_scored_before_seal=False,input_checks_passed=True,reference_checks_bundle_found=False,version=sys.version))
 for t in ORDER:report_track(t)
 report_all();print('MASTER_SEALED',sha(OUT/'MASTER_SEAL.json'),flush=True)

def verify_all():
 seal=read(OUT/'MASTER_SEAL.json');n=0
 for path,h in seal['files'].items():assert sha(ROOT/path)==h,('SEALED_DATA_CHANGED',path);n+=1
 for path,h in read(OUT/'historical_hashes.json').items():assert sha(ROOT/path)==h,('HISTORY_CHANGED',path)
 state=read(OUT/'controller_state.json');main=0;smoke=0;fcount=0;details=[]
 for t in ORDER:
  st=read(OUT/t/'STATUS.json')
  if t in SOURCES:
   fits=read(OUT/t/'fits.json') if (OUT/t/'fits.json').exists() else [];logs=[json.loads(l) for l in (OUT/t/'optimizer_log.jsonl').read_text().splitlines()] if (OUT/t/'optimizer_log.jsonl').exists() else [];main+=sum(not r['smoke'] for r in logs);smoke+=sum(r['smoke'] for r in logs);fcount+=len(fits)
   for f in fits:
    ll=[r for r in logs if r['id']==f['id']];assert len(ll)==f['updates'];assert [r['step'] for r in ll]==list(range(1,len(ll)+1));
    if f['status']=='COMPLETE':assert f['updates']==512 and f['frozen_unchanged'] and f['buffers_unchanged']
    for c in f['checkpoints']:assert sha(ROOT/c['path'])==c['sha256']
   csvwrite(OUT/t/'optimizer_log.csv',[{k:(json.dumps(v,sort_keys=True) if isinstance(v,dict) else v) for k,v in row.items()} for row in logs])
   if st['EXECUTION']=='COMPLETE':
    sels=read(OUT/t/'selections.json')['selections'];lr=read(OUT/t/'LR_selection.json');assert {s['seed'] for s in sels}<={73101,73102}
    for s in sels:
     assert s['lr']==lr['learning_rates'][s['arm']];f=next(f for f in fits if f['id']==s['id']);c=s['selected'];assert c in f['checkpoints']
     if t!='R04' or s['arm']=='D0':assert c==min(f['checkpoints'],key=lambda cc:(cc['accuracy'],cc['step']))
    for p in read(OUT/t/'predictions_manifest.json')['predictions']:assert sha(ROOT/p['path'])==p['sha256']
    assert read(OUT/t/'verification.json')['passed']
   details.append(dict(track=t,attempted=len(fits),main_updates=sum(not r['smoke'] for r in logs),smoke_updates=sum(r['smoke'] for r in logs),journal_complete=True))
  elif st['EXECUTION']=='COMPLETE':
   assert st['neural_fits']==0
   for path,h in read(OUT/t/'used_prior_hashes.json').items():assert sha(ROOT/path)==h
   for r in read(OUT/t/'predictions_manifest.json'):assert sha(ROOT/r['path'])==r['sha256']
   details.append(dict(track=t,neural_fits=0,prior_predictions_verified=True))
 assert main==state['main_updates'] and smoke==state['smoke_updates'];assert main<=59392 and smoke<=96 and main+smoke<=59488 and fcount<=116 and sum(state['forwards'].values())<=200000
 save(OUT/'verification.json',dict(passed=True,at=time.time(),sealed_files=n,historical_files=len(read(OUT/'historical_hashes.json')),main_updates=main,smoke_updates=smoke,attempted_fits=fcount,forwards=state['forwards'],tracks=details,cap_checks=True));print('VERIFIED_ALL',main,smoke,fcount,flush=True)
