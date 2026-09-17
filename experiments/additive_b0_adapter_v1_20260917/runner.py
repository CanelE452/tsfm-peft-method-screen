"""One authorized finite additive comparison; no automatic research continuation."""
import argparse,signal,traceback
from .common import *
from .prepare import prepare
from .checks import cpu_check,precheck,smoke
from .train import train_select,train_repeat,request_stop
from .evaluate import evaluate,verify

def seal():
    if (OUT/'EXECUTION_SEAL.json').exists():check_seal();return
    assert sum(r['updates'] for r in read(OUT/'smoke_checks.json').values())==12
    c=read(OUT/'MACHINE_CONTRACT.json');c['status']='SEALED_FOR_MAIN_TRAINING';c['microbatch']=read(OUT/'microbatch.json');save(OUT/'MACHINE_CONTRACT.json',c)
    paths=[p for p in EXP.glob('*.py') if p.name not in ['report.py','analysis.py']]+[ROOT/'scripts/run_additive_b0_adapter.py',ROOT/'scripts/priority12/common.py']
    for name in ['outlier_signal_followup_v2_20260917','outlier_signal_peft_v1_20260917']:paths+=list((ROOT/'experiments'/name).glob('*.py'))
    data=[OUT/n for n in ['PROTOCOL.md','MACHINE_CONTRACT.json','SOURCE_MANIFEST.json','BASELINE_MANIFEST.json','DATA_MANIFEST.json','origins.csv','microbatch.json','download_receipts.json']]
    save(OUT/'EXECUTION_SEAL.json',dict(sealed_at=time.time(),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},data_hashes={str(p.relative_to(ROOT)):sha(p) for p in data},prior_E_exposed=True,new_candidate_E_scoring_before_selection=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','train-select','train-repeat','evaluate','verify','run-all']);args=p.parse_args();setup();prepare()
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    if args.action=='prepare':return
    w=Watch(args.action)
    try:
        w.boundary(startup=True)
        if args.action in ['check','run-all']:cpu_check();precheck(w);smoke(w)
        if args.action in ['run-all','train-select','train-repeat']:seal()
        if args.action in ['run-all','train-select']:train_select(w)
        if args.action in ['run-all','train-repeat']:train_repeat(w)
        if args.action in ['run-all','evaluate']:evaluate(w)
        if args.action in ['run-all','verify']:verify(w)
        status(execution='COMPLETE' if (OUT/'verification.json').exists() else 'PARTIAL',automatic_followup=False)
    except BaseException as e:
        save(OUT/'last_execution_error.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),at=time.time()))
        status(execution='PARTIAL_RESOURCE' if isinstance(e,ResourceError) else 'INVALID_IMPLEMENTATION',reason=str(e));raise
    finally:w.close()

if __name__=='__main__':main()
