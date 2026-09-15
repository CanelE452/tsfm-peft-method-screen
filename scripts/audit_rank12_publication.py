"""Audit completed fixed R1/R2 batch and publication. No model forward or updates."""
import csv,hashlib,json,os,re,subprocess,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/peft_rank12_20260915';Q=ROOT/'results/query_budget_repair_v2_20260915';C=ROOT/'results/channel_sharing_screen_v1_20260915'
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def th(state):
    h=hashlib.sha256()
    for n,p in sorted(state.items()):h.update(n.encode());h.update(str((tuple(p.shape),p.dtype)).encode());h.update(p.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def main():
    q=read(Q/'status.json');c=read(C/'status.json');assert q['numeric_updates']==84 and q['A_updates']==q['B_fit_attempts']==0;assert c['fit_attempts']==c['fits_completed']==24 and c['training_updates']==9602 and c['smoke_updates']==24
    fits=read(C/'fits.json');assert sum(f['updates'] for f in fits)==9602 and all(f['status']=='COMPLETE' and f['reload_max_abs']==0 for f in fits)
    v=read(C/'independent_verification.json');assert v['prediction_replays']==257 and v['max_metric_abs_error']<=1e-10
    assert all(r['exit_code']==0 for r in read(OUT/'sequence.json'))
    hist=read(OUT/'historical_hashes.json')
    for p,h in hist.items():assert sha(ROOT/p)==h,p
    for folder in [Q,C]:
        contract=read(folder/'contract.json')
        for p,h in contract['source_hashes'].items():assert sha(ROOT/p)==h,p
        for p,h in contract['model_files'].items():assert sha(Path(p))==h,p
        for source in contract['data'].values():
            for p,h in source['staged'].items():assert sha(ROOT/p)==h,p
    common={};frequency={};initials=[]
    for r in read(C/'trajectories.json'):
        if r['epoch']!=0:continue
        state=torch.load(ROOT/r['checkpoint_path'],weights_only=True,map_location='cpu');h=th({n:p for n,p in state.items() if n.startswith('head.') or 'lora_' in n});key=(r['dataset'],r['seed']);assert common.setdefault(key,h)==h
        fr={n:p for n,p in state.items() if n.startswith('frequency_adapter.')}
        if fr:
            f=th(fr);assert frequency.setdefault(key,f)==f
        initials.append(dict(fit=r['fit'],head_lora_hash=h,frequency_hash=th(fr) if fr else None))
    curves=[]
    for f in fits:
        steps=read(C/(f['fit']+'_steps.json'));assert len(steps)==f['updates'];assert sum(r['origins'] for r in steps)==f['epochs']*({'electricity':512,'traffic':504}[f['dataset']])
        for epoch in range(1,f['epochs']+1):
            rr=[r for r in steps if r['epoch']==epoch];curves.append(dict(fit=f['fit'],dataset=f['dataset'],seed=f['seed'],arm=f['arm'],epoch=epoch,updates=len(rr),origins=sum(r['origins'] for r in rr),train_standardized_mse=sum(r['loss']*r['origins'] for r in rr)/sum(r['origins'] for r in rr),lr=rr[0]['lr'],active_seconds=sum(r['seconds'] for r in rr)))
    with (C/'training_curves.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(curves[0]),lineterminator='\n');w.writeheader();w.writerows(curves)
    gpu={}
    for folder in [Q,C]:
        for p in folder.glob('gpu_*.jsonl'):
            rr=[json.loads(l) for l in p.open() if l.strip()]
            gpu[str(p.relative_to(ROOT))]=dict(samples=len(rr),minimum_free_mib=min(r['free_mib'] for r in rr),unapproved_external_samples=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in r['apps']) for r in rr),allowed_rustdesk_samples=sum(any(a.get('allowed_desktop',False) for a in r['apps']) for r in rr),busy_samples=sum(r['busy'] for r in rr))
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip();own_active=[]
    for line in apps.splitlines():
        pid=line.split(',')[0].strip();p=Path('/proc')/pid/'cmdline'
        if p.exists() and 'peft_rank12_20260915' in p.read_bytes().decode(errors='replace'):own_active.append(line)
    assert not own_active
    files=[ROOT/'README.md',ROOT/'docs/RESULTS_INDEX.md',ROOT/'scripts/plot_rank12_results.py',ROOT/'scripts/finalize_rank12_publication.py',ROOT/'scripts/audit_rank12_publication.py']
    for folder in [Q,C,OUT,ROOT/'experiments/peft_rank12_20260915']:files += [p for p in folder.rglob('*') if p.is_file() and p.suffix!='.pyc' and p.name!='publication_audit.json']
    links=0
    for p in files:
        if p.suffix=='.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
                if '://' not in target and not target.startswith('#'):assert (p.parent/target.split('#')[0]).resolve().exists(),(p,target);links+=1
    result=dict(audited_at=time.time(),Q_updates=84,Q_fits=0,R2_smoke_updates=24,R2_fits=24,R2_training_updates=9602,R2_prediction_replays=257,metric_max_abs_error=v['max_metric_abs_error'],all_fit_selections_and_reloads_verified=True,initial_head_lora_frequency_exact=initials,historical_files_unchanged=len(hist),sealed_sources_and_model_and_data_hashes_unchanged=True,relative_links_verified=links,gpu_logs=gpu,current_compute_apps=apps,own_gpu_training_remaining=own_active,automatic_followup=False,artifact_files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))},scope='Raw data, model weights and tensor caches are ignored/local. Code, contracts, human reports, metrics and hash evidence are published. No model forward or optimizer update during this audit.')
    (OUT/'publication_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['artifact_files','initial_head_lora_frequency_exact']},indent=2))
if __name__=='__main__':main()
