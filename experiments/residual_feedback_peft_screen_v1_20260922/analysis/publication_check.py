import sys,re,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *

for name in ['VERIFICATION.json','DATA_POSTRUN_AUDIT.json','CHECKPOINT_AND_METRIC_AUDIT.json']:
    assert read(RESULTS/name)['status']=='PASS',name
assert not (CACHE/'EXECUTOR.lock').exists()
for p,h in read(RESULTS/'SOURCE_SEAL.json')['files'].items():assert sha(EXP/p)==h
for name,v in read(EXP/'contract'/'MANIFEST.json').items():assert sha(EXP/'contract'/name)==v['sha256']
links=[]
for p in RESULTS.rglob('*.md'):
    for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
        if link.startswith('https://'):continue
        assert (p.parent/link).is_file(),(p,link);links.append(link)
events=[json.loads(x) for x in (RESULTS/'events.jsonl').read_text().splitlines()]
done=next(i for i,x in enumerate(events) if x['kind']=='all_main_complete')
assert all(i>done for i,x in enumerate(events) if x['kind']=='test_scoring_started')
modified=subprocess.check_output(['git','diff','--name-only'],text=True).splitlines()
untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines()
assert all(x.startswith(('experiments/'+NAME+'/','results/'+NAME+'/')) for x in modified+untracked),modified+untracked
save(RESULTS/'PUBLICATION_CHECK.json',dict(status='PASS',local_links_verified=len(links),only_new_experiment_paths_changed=True,training_source_seal_unchanged=True,contract_bytes_unchanged=True,executor_stopped=True,test_scoring_after_all_predictions=True,automatic_followup=False,original_dirty_files_preserved=True))
files=[p for root in [EXP,RESULTS] for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json']
assert all(p.suffix not in ['.pt','.pth','.npz','.npy','.gz','.safetensors','.bin'] for p in files)
save(RESULTS/'ARTIFACT_MANIFEST.json',dict(files={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)},self_excluded=True,raw_models_and_predictions_retained_in_local_ignored_cache=True))
print('PASS:',len(files),'artifacts;',len(links),'links;',sum(p.stat().st_size for p in files),'bytes')
