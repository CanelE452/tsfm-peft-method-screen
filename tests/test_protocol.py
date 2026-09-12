import json
from pathlib import Path

def test_budget_and_seal_contract():
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'configs/common_screen.yaml').read_text())
    candidates=[json.loads(p.read_text()) for p in sorted((root/'configs').glob('candidate_*.yaml'))]
    assert len(candidates)==7
    assert sum(x['fit_cap'] for x in candidates)==c['fit_cap']==38
    assert sum(x['stream_cap'] for x in candidates)==c['stream_cap']==5
    assert c['round2'] is False
    assert c['checkpoints'][0]==0 and c['checkpoints'][-1]==c['max_steps']==360
