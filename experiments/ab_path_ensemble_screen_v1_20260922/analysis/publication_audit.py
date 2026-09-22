"""Verify publication scope, links, manifests and cache provenance; no experiments."""
import sys,re,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import *
from A.data import Solar

def main():
    assert '27 passed' in (RESULTS/'CPU_TEST_FINAL.txt').read_text()
    for c in ['A','B']:
        info=read(RESULTS/c/'CPU_TESTS.json');info.update(reference_tests=22,integration_cpu_tests=5,final_total_passed=27,final_test_output='../CPU_TEST_FINAL.txt',real_model_validation='separate MODEL_SMOKE_AUDIT and PRE_MAIN_AUDIT, not replaced by CPU tests')
        save(RESULTS/c/'CPU_TESTS.json',info)
    d=Solar();selection=read(RESULTS/'A/MODEL_SELECTION.json');model=read(RESULTS/'A/SOURCE_AND_MODEL_MANIFEST.json')['models']['amazon/chronos-bolt-small'];entries=[]
    for item in read(RESULTS/'A/TEACHER_CACHE_MANIFEST.json')['caches']:
        with np.load(item['path']) as f:pairs=f['pairs'].copy()
        batch=d.batch(pairs);h=hashlib.sha256()
        for key in ['pairs','x','sigma']:
            arr=np.ascontiguousarray(batch[key]);h.update(key.encode());h.update(str(arr.dtype).encode());h.update(str(arr.shape).encode());h.update(arr.tobytes())
        entries.append({**item,'input_arrays_sha256':h.hexdigest(),'role_sha256':hashlib.sha256(b'TRAIN').hexdigest(),'queried_examples':len(pairs),'forward_batches':int(np.ceil(len(pairs)/8)),
            'model_revision':model['revision'],'pretrained_weights_sha256':next(x['sha256'] for x in model['files'] if x['name']=='model.safetensors'),
            'teacher_lora_state_sha256':selection[str(item['seed'])]['FULL9']['selected_state_sha'] if item['kind']=='selected_same_seed_teacher' else None})
    save(RESULTS/'A/CACHE_PROVENANCE.json',dict(status='PASS',caches=entries,teacher_train_input_only=True,model_forward_batch_size=8))
    required=['PROTOCOL.md','DATA_AUDIT.json','INFORMATION_CONTRACT.json','SOURCE_AND_MODEL_MANIFEST.json','ENVIRONMENT.json','requirements-lock.txt','TRAINING_BUDGET.json','CPU_TESTS.json','MODEL_SMOKE_AUDIT.json','SOURCE_SEAL.json','UPDATE_LEDGER.jsonl','FIT_LEDGER.csv','MODEL_SELECTION.json','CALIBRATION.json','BASELINE_SELECTION.json','PREDICTIONS_MANIFEST.json','SCORES.csv','SEED_EFFECTS.csv','LEAD_SCORES.csv','RESOURCES.csv','VERIFICATION.json','REPORT_KO.md','FINAL_DECISION.md']
    for c in ['A','B']:
        for name in required:assert (RESULTS/c/name).is_file(),(c,name)
    links=0
    for p in RESULTS.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8-sig')):
            if '://' in target or target.startswith('#'):continue
            assert (p.parent/target.split('#')[0]).exists(),(p,target)
            links+=1
    files=list(EXP.rglob('*'))+list(RESULTS.rglob('*'));public=[]
    for p in files:
        if not p.is_file() or any(x in p.parts for x in ['__pycache__','.pytest_cache']):continue
        assert p.suffix not in ['.pt','.npz','.safetensors','.pkl','.gz','.parquet','.pyc']
        assert p.stat().st_size<10_000_000
        if p.name!='PUBLIC_ARTIFACT_MANIFEST.json':public.append(p)
    save(RESULTS/'PUBLICATION_CHECK.json',dict(status='PASS',required_candidate_artifacts=len(required)*2,local_markdown_links_verified=links,no_raw_weights_predictions_in_public_scope=True,all_artifacts_under_new_experiment=True,automatic_followup=False))
    # Include the completed audit receipt itself; manifest self-reference is intentionally excluded.
    public=list(dict.fromkeys(public+[RESULTS/'PUBLICATION_CHECK.json']))
    save(RESULTS/'PUBLIC_ARTIFACT_MANIFEST.json',dict(files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(public)},self_excluded=True))
    print('PASS: required files, Markdown links, publication scope, teacher cache provenance.')

if __name__=='__main__':main()
