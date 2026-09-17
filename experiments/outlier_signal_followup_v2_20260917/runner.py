"""Single contract, one GPU worker, finite transitions and exact resume only."""
import argparse,signal,traceback
from .common import *
from .prepare import prepare,audit
from .diagnose import cpu_diagnosis,adapter_diagnosis,finish_diagnosis
from .precheck import precheck
from .checks import cpu_check,smoke
from .train import train_select,train_repeat,request_stop
from .evaluate import evaluate,verify

def seal():
    if (OUT/'EXECUTION_SEAL.json').exists():check_seal();return
    assert len(read(OUT/'smoke_checks.json'))==12
    assert read(OUT/'failure_hypothesis.json')['optimizer_updates']==0
    c=read(OUT/'MACHINE_CONTRACT.json');c.update(status='SEALED_FOR_MAIN_TRAINING',microbatch=read(OUT/'microbatch.json'))
    save(OUT/'MACHINE_CONTRACT.json',c)
    paths=sorted(p for p in EXP.glob('*.py') if p.name not in ['report.py','render_analysis.py'])+[ROOT/'scripts/run_outlier_signal_followup.py',ROOT/'scripts/priority12/common.py']
    # Imported immutable v1 math/model/common helpers are also frozen dependencies.
    paths+=list((ROOT/'experiments/outlier_signal_peft_v1_20260917').glob('*.py'))
    files=[OUT/n for n in ['PROTOCOL.md','DATA_MANIFEST.json','origins.csv','origin_audit.json','AUGMENTATION_MANIFEST.json','MACHINE_CONTRACT.json','microbatch.json','download_receipts.json','SOURCE_MANIFEST.json']]
    save(OUT/'EXECUTION_SEAL.json',dict(sealed_at=time.time(),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},data_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},new_E_scores_seen=False,performance_observations_used=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['diagnose','prepare','check','train-select','train-repeat','evaluate','verify','status','run-all']);args=p.parse_args();setup()
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    if args.action=='prepare':prepare();return
    if args.action=='status':print(json.dumps(status(),indent=2));return
    audit()
    if args.action in ['diagnose','run-all']:cpu_diagnosis()
    if args.action!='diagnose':prepare()
    w=Watch(args.action)
    try:
        w.boundary(startup=True)
        if args.action in ['diagnose','run-all']:adapter_diagnosis(w);finish_diagnosis()
        if args.action in ['check','run-all']:cpu_check();precheck(w);smoke(w)
        if args.action in ['run-all','train-select','train-repeat']:seal()
        if args.action in ['run-all','train-select']:train_select(w)
        if args.action in ['run-all','train-repeat']:train_repeat(w)
        if args.action in ['run-all','evaluate']:evaluate(w)
        if args.action in ['run-all','verify']:verify(w)
        status(execution='COMPLETE' if (OUT/'verification.json').exists() else 'PARTIAL',automatic_followup=False)
    except BaseException as e:
        save(OUT/'last_execution_error.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),time=time.time()))
        status(execution='PARTIAL_RESOURCE' if isinstance(e,ResourceError) else 'INVALID_IMPLEMENTATION',reason=str(e),automatic_background_worker=False)
        raise
    finally:w.close()

if __name__=='__main__':main()
