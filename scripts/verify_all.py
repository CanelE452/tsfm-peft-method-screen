import json,subprocess,sys
from pathlib import Path
from tsfm_peft_screen.reproducibility import ROOT,digest,sha
from tsfm_peft_screen.metrics import replay
from tsfm_peft_screen.selection import require_seal
subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,check=True)
if '--preflight' in sys.argv:
    print('CPU preflight PASS');raise SystemExit(0)
counts={'fit':0,'stream':0};errors=[]
for i in range(1,8):
    out=ROOT/'results'/f'candidate_{i:02}'
    for f in ['status.json','contract.json','integrity.json','metrics.csv','selections.csv','trajectories.csv','resource_usage.json','RESULT.md','figures/primary.png','figures/diagnostics.png']:assert (out/f).is_file(),(i,f)
    status=json.loads((out/'status.json').read_text());assert status['verdict'] in ['PASS','WEAK','FAIL','NO_PROBLEM','INVALID_CONSTRUCT','NOVELTY_COLLISION','IMPLEMENTATION_BLOCKED']
    counts['fit']+=status.get('fit_count') or 0;counts['stream']+=status.get('stream_count') or 0
    if (out/'selection.json').exists():
        contract=json.loads((out/'contract.json').read_text());require_seal(out/'selection.json',contract)
        for p in (ROOT/'.cache'/f'candidate_{i:02}').glob('E_*.npz'):errors.append(replay(p))
        if 'source_hashes' in contract:
            for name,value in contract['source_hashes'].items():assert sha(ROOT/name)==value,('Changed experimental source after seal',i,name)
    if i==6:assert status['verdict']=='NOVELTY_COLLISION' and status['fit_count']==0
assert counts['fit']<=38 and counts['stream']<=5
print('FINAL INTEGRITY PASS',counts,'max metric replay error',max(errors,default=0))
