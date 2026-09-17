"""Finite 58-fit followup with no automatic successor research."""
import argparse,signal,traceback
from .common import *
from .prepare import prepare
from .checks import cpu_check,precheck,smoke
from .train import train_all,request_stop
from .evaluate import predict_all,score
from .verify import verify

def seal():
    if (OUT/'MASTER_SEAL.json').exists():check_seal();return
    paths=[p for p in EXP.glob('*.py') if p.name not in ['report.py','analysis.py']]+[ROOT/'scripts/run_additive_persistence_validation.py',ROOT/'scripts/priority12/common.py']
    for name in ['outlier_signal_peft_v1_20260917','outlier_signal_followup_v2_20260917','additive_b0_adapter_v1_20260917']:
        paths+=list((ROOT/'experiments'/name).glob('*.py'))
    names=['MASTER_CLI.txt','MASTER_PROTOCOL.md','TOPIC_ONEPAGE.md','LITERATURE_BOUNDARY.md','DATA_MANIFEST.json','DATA_READY.json','SOURCE_MANIFEST.json','BASELINE_MANIFEST.json','REUSED_MODELS.json','SERIES_PANEL_MANIFEST.json','DATA_EXPOSURE_MANIFEST.csv','ETTM2_DATA_RECEIPT.json','ORIGIN_AUDIT.csv','official_source_receipts.json']
    save(OUT/'MASTER_SEAL.json',dict(at=time.time(),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},data_hashes={str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in names},main_fit_cap=58,main_update_cap=59392,smoke_cap=36,total_update_cap=59428,primary='P1 SHIFT8 C3/C0 and C3/C2, original two seeds separate from third/all-three',new_E_scored=False,automatic_followup=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','run-all','train','evaluate','verify']);a=p.parse_args();setup();prepare()
    if a.action=='prepare':return
    seal();signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop);w=Watch(a.action)
    try:
        w.boundary(startup=True)
        if a.action in ['check','run-all']:cpu_check();precheck(w);smoke(w)
        if a.action in ['train','run-all']:
            assert sum(r['updates'] for r in read(OUT/'smoke_checks.json').values())==36
            train_all(w)
        if a.action in ['evaluate','run-all']:predict_all(w);score()
        if a.action in ['verify','run-all']:verify(w)
        status(execution='COMPLETE_COMPUTE' if (OUT/'VERIFICATION.json').exists() else 'PARTIAL',automatic_followup=False)
    except BaseException as e:
        save(OUT/'last_execution_error.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),at=time.time()))
        status(execution='PARTIAL_RESOURCE' if isinstance(e,ResourceError) else 'INVALID_IMPLEMENTATION',reason=str(e));raise
    finally:w.close()

if __name__=='__main__':main()
