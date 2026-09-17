"""Executable contract actions with bounded transitions and no follow-up studies."""
import argparse,signal,traceback
from .common import *
from .prepare import prepare
from .checks import cpu_check,smoke
from .reproduce import reproduce
from .train import train_select,train_repeat,request_stop
from .evaluate import evaluate,verify
from .report import report

def seal():
    if (OUT/'EXECUTION_SEAL.json').exists():check_seal();return
    assert len(read(OUT/'smoke_checks.json'))==12
    c=read(OUT/'MACHINE_CONTRACT.json');c.update(status='SEALED_FOR_MAIN_TRAINING',unsealed=[],microbatch=read(OUT/'microbatch.json'))
    save(OUT/'MACHINE_CONTRACT.json',c)
    paths=sorted(EXP.glob('*.py'))+[ROOT/'scripts/run_outlier_signal_peft.py',ROOT/'scripts/priority12/common.py']
    data=[OUT/name for name in ['PROTOCOL.md','DATA_MANIFEST.json','origins.csv','CONDITION_MANIFEST.json','MACHINE_CONTRACT.json','microbatch.json','download_receipts.json']]
    save(OUT/'EXECUTION_SEAL.json',dict(sealed_at=time.time(),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},data_hashes={str(p.relative_to(ROOT)):sha(p) for p in data},performance_observations_used=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','reproduce','train-select','train-repeat','evaluate','verify','report','status','run-all']);args=p.parse_args();setup()
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    if args.action=='prepare':prepare();return
    if args.action=='status':print(json.dumps(read(OUT/'status.json'),indent=2));return
    if args.action=='report':report();return
    if args.action=='run-all':prepare();cpu_check()
    watch=Watch(args.action)
    try:
        watch.boundary(startup=True)
        if args.action in ['run-all','reproduce']:reproduce(watch)
        if args.action in ['run-all','check']:cpu_check();smoke(watch)
        if args.action in ['run-all','train-select','train-repeat']:seal()
        if args.action in ['run-all','train-select']:train_select(watch)
        if args.action in ['run-all','train-repeat']:train_repeat(watch)
        if args.action in ['run-all','evaluate']:evaluate(watch)
        if args.action in ['run-all','verify']:verify(watch)
        if args.action=='run-all':report()
    except BaseException as e:
        save(OUT/'last_execution_error.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),time=time.time()))
        report();status(execution='PARTIAL',reason=str(e),automatic_background_worker=False)
        raise
    finally:watch.close()

if __name__=='__main__':main()
