import sys,re,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *

assert read(RESULTS/'VERIFICATION.json')['status']=='PASS'
assert read(RESULTS/'POSTRUN_AUDIT.json')['status']=='PASS'
assert not (CACHE/'EXECUTOR.lock').exists()
for name,h in read(RESULTS/'SOURCE_SEAL.json')['files'].items():assert sha(EXP/name)==h
for name,h in read(RESULTS/'SELECTION_SEAL.json')['files'].items():assert sha(RESULTS/name)==h
links=[]
for p in [RESULTS/'REPORT_KO.md',RESULTS/'FINAL_DECISION.md']:
    for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
        if link.startswith('https://'):continue
        assert (p.parent/link).is_file(),link
        links.append(link)
note=dict(classification='ANALYSIS_VERIFICATION_FIX',main_training_affected=False,extra_optimizer_updates=0,
    issue='Exact Python dict equality rejected CAL loss recomputation differences <= 6e-17; alpha/beta and all selections identical.',
    correction='Require exact chosen alpha/beta/grid count; numerical score tolerance 1e-14, matching independent metric audit.',
    failures_in_data_or_main_training=0,timing_limit='Same update counts reached target; one ETTh1 RANDOM run slower, cause not identified. No algorithmic speedup claim.')
save(RESULTS/'ANALYSIS_NOTES.json',note)
files=[p for directory in [EXP,RESULTS] for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['ARTIFACT_MANIFEST.json','PUBLICATION_CHECK.json']]
assert all(p.suffix not in ['.pt','.npz','.safetensors','.bin','.gz'] for p in files)
assert all(p.stat().st_size<20_000_000 for p in files)
events=[json.loads(s) for s in (RESULTS/'events.jsonl').read_text().splitlines()]
saved=next(i for i,x in enumerate(events) if x['kind']=='all_test_saved')
assert all(i>saved for i,x in enumerate(events) if x['kind']=='test_analysis_started')
save(RESULTS/'PUBLICATION_CHECK.json',dict(status='PASS',local_links_verified=len(links),no_model_or_raw_data_files=True,training_source_seal_unchanged=True,all_test_saved_before_every_analysis_attempt=True,executor_stopped=True,automatic_followup=False,scope='New experiment/results and additive results index entry only'))
files.append(RESULTS/'PUBLICATION_CHECK.json')
save(RESULTS/'ARTIFACT_MANIFEST.json',dict(files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)},self_excluded=True,local_cache_required_for_prediction_replay=True))
print(f'PASS: {len(files)} publication artifacts; {len(links)} local report links')
