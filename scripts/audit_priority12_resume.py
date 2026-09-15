"""Audit completed Q/C records, source/history hashes, links and GPU shutdown; no model calls."""
import csv,hashlib,json,re,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results/priority12_resume_20260915';c=root/'results/channel_basis_pilot_resume_20260915';q=root/'results/query_budget_numeric_v2_resume_20260915'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
assert all(r['exit_code']==0 for r in read(out/'runner_exit_codes.json'))
s=read(c/'status.json');assert s['status']=='COMPLETE' and s['fit_attempts']==s['fits_completed']==24 and s['training_updates']==24576
fits=read(c/'fits.json');assert len(fits)==24 and sum(f['updates'] for f in fits)==24576
assert all(f['status']=='COMPLETE' and f['updates']==1024 and f['reload_max_abs']==0 and f['frozen_unchanged'] for f in fits)
assert len(read(c/'evaluation.json'))==28 and len(read(c/'trajectories.json'))==120
v=read(c/'independent_verification.json');assert v['prediction_metric_replays']==172 and v['metric_max_abs_error']<=1e-10
metrics=[r for r in rows(c/'metrics.csv') if r['role']=='selected'];decisions=read(c/'decisions.json')
for d in decisions['source_decisions']:
    for a,m in d['means'].items():
        rr=[float(r['mse']) for r in metrics if r['dataset']==d['dataset'] and r['arm']==a];assert len(rr)==2 and abs(sum(rr)/2-m)<1e-14
for path,h in read(out/'historical_hashes.json').items():assert sha(root/path)==h
for folder,key in [(q,'sources'),(c,'source_hashes')]:
    for path,h in read(folder/'contract.json')[key].items():assert sha(root/path)==h
logs={}
for folder in [q,c]:
    for path in folder.glob('gpu_*.jsonl'):
        rr=[json.loads(s) for s in path.open() if s.strip()]
        if rr:logs[str(path.relative_to(root))]=dict(samples=len(rr),minimum_free_mib=min(r['free_mib'] for r in rr),busy_samples=sum(bool(r.get('busy')) for r in rr),allowed_rustdesk_samples=sum(any(a.get('allowed_desktop') for a in r.get('apps',[])) for r in rr),other_external_compute_samples=sum(any(not a.get('own',False) and not a.get('allowed_desktop',False) for a in r.get('apps',[])) for r in rr))
apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.free,memory.used,utilization.gpu','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()
assert not apps,'Unexpected remaining compute: '+apps
files=[root/'README.md',root/'docs/RESULTS_INDEX.md',root/'scripts/report_priority12_resume.py',root/'scripts/plot_priority12_resume.py',root/'scripts/audit_priority12_resume.py']
for folder in [q,c,out]:files.extend(p for p in folder.rglob('*') if p.is_file() and p.name!='publication_audit.json')
links=[]
for p in files:
    if p.suffix=='.md':
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if '://' not in link and not link.startswith('#'):
                target=(p.parent/link.split('#')[0]).resolve();assert target.exists(),(p,link);links.append(dict(source=str(p.relative_to(root)),target=link))
for p in [root/'scripts/report_priority12_resume.py',root/'scripts/plot_priority12_resume.py',root/'scripts/audit_priority12_resume.py']:compile(p.read_text(),str(p),'exec')
audit=dict(audited_at=time.time(),Q_numeric_updates=read(q/'status.json')['numeric_updates'],Q_forecasting_fits=0,C_completed_fits=24,C_training_updates=24576,C_evaluations_selected=24,C_evaluations_coefficient_swap=4,C_independent_prediction_records=172,C_metric_max_abs_error=v['metric_max_abs_error'],selected_checkpoint_replay_exact_fits=24,frozen_weights_unchanged_fits=24,mean_scores_recalculated=True,historical_files_unchanged=1360,sealed_source_hashes_unchanged=True,valid_relative_markdown_links=len(links),GPU_compute_at_completion=apps,GPU_status_at_completion=gpu,gpu_logs=logs,files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))},scope='Code/reports/manifests/metrics are published. Ignored raw data, weights and prediction/checkpoint caches remain local. No new training or GPU inference in this audit.',automatic_followup=False)
(out/'publication_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k!='files'},indent=2));print('publication_files',len(audit['files']))
