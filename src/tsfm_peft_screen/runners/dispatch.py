import json,sys,time,traceback
from ..reproducibility import ROOT,write_json,guard
from .common_fit import train

def run(candidate):
    out=ROOT/'results'/f'candidate_{candidate:02}'
    if (out/'status.json').exists():raise FileExistsError('Existing candidate verdict: no automatic rerun')
    try:
        if candidate==6:
            write_json(out/'status.json',dict(verdict='NOVELTY_COLLISION',problem_gate='STOP_NOVELTY',novelty_collision=True,fit_count=0,stream_count=0,reason='Baron et al. arXiv2510.02224 section3 and AppendixB: same core method; hidden features and low-rank covariance are parameterization extensions'))
            write_json(out/'contract.json',dict(candidate=6,execution='prohibited by novelty gate',paper='https://arxiv.org/html/2510.02224v1'))
            write_json(out/'integrity.json',dict(status='NOT_APPLICABLE',reason='No GPU or E access'))
            write_json(out/'resource_usage.json',dict(fit_count=0,stream_count=0,wall_seconds=0))
        elif candidate==5:
            from .streaming_eval import run as stream
            stream()
        else:train(candidate)
    except Exception as e:
        attempts=json.loads((out/'attempts.json').read_text()) if (out/'attempts.json').exists() else []
        tb=traceback.format_exc();(out/'failure.txt').write_text(tb)
        write_json(out/'status.json',dict(verdict='IMPLEMENTATION_BLOCKED',problem_gate='ERROR',novelty_collision=False,fit_count=len(attempts) if candidate!=5 else 0,stream_count=len(attempts) if candidate==5 else 0,error=str(e)))
        write_json(out/'failure.json',dict(error_type=type(e).__name__,error=str(e),traceback=tb,nonzero_exit=True))
        raise
