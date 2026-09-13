import hashlib,json,subprocess,sys
from pathlib import Path
from tsfm_peft_screen.reproducibility import ROOT,digest,sha
from tsfm_peft_screen.metrics import replay
from tsfm_peft_screen.selection import require_seal
subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,check=True)
if '--preflight' in sys.argv:
    print('CPU preflight PASS');raise SystemExit(0)
execution=json.loads((ROOT/'results/screening_summary/execution_source.json').read_text());commit=execution['execution_commit'];historical={}
counts={'fit':0,'stream':0};errors=[];blocked=[]
for i in range(1,8):
    out=ROOT/'results'/f'candidate_{i:02}'
    for f in ['status.json','contract.json','integrity.json','metrics.csv','selections.csv','trajectories.csv','resource_usage.json','RESULT.md','figures/primary.png','figures/diagnostics.png']:assert (out/f).is_file(),(i,f)
    status=json.loads((out/'status.json').read_text());assert status['verdict'] in ['PASS','WEAK','FAIL','NO_PROBLEM','INVALID_CONSTRUCT','NOVELTY_COLLISION','IMPLEMENTATION_BLOCKED']
    counts['fit']+=status.get('fit_count') or 0;counts['stream']+=status.get('stream_count') or 0
    if status['verdict']=='IMPLEMENTATION_BLOCKED':blocked.append(i)
    if (out/'selection.json').exists():
        contract=json.loads((out/'contract.json').read_text());require_seal(out/'selection.json',contract)
        predictions=list((ROOT/'.cache'/f'candidate_{i:02}').glob('E_*.npz'))
        assert predictions,('Local saved predictions are required for replay',i)
        for p in predictions:errors.append(replay(p))
        if 'source_hashes' in contract:
            for name,value in contract['source_hashes'].items():
                if name not in historical:historical[name]=hashlib.sha256(subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=ROOT)).hexdigest()
                assert historical[name]==value,('Historical execution source mismatch',i,name)
    if i==6:assert status['verdict']=='NOVELTY_COLLISION' and status['fit_count']==0
assert counts['fit']<=38 and counts['stream']<=5
print('ARTIFACT VERIFICATION PASS',counts,'max metric replay error',max(errors,default=0),'blocked candidates',blocked)
print('Candidate-level integrity failures remain explicitly recorded; artifact verification does not turn them into PASS.')

if (ROOT/'results/candidate_05_repaired/status.json').exists():
    print('Above counts and blocked list describe the historical screen; checking the separately authorized recovery.',flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/finalize_candidate_05_recovery.py'),'--verify-only'],cwd=ROOT,check=True)

if (ROOT/'results/candidate_01_v2/status.json').exists():
    subprocess.run([sys.executable,str(ROOT/'scripts/finalize_candidate_01_v2.py'),'--verify-only'],cwd=ROOT,check=True)

if (ROOT/'results/memory_feasibility/verification.json').exists():
    subprocess.run([sys.executable,str(ROOT/'scripts/finalize_memory_feasibility.py'),'--verify-only'],cwd=ROOT,check=True)
