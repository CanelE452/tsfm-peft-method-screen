"""One isolated fixed stage; exceptions produce evidence and nonzero exit status."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import traceback
import torch
from .util import read_json,write_json,redact
from .backend import resolve_backend
from .model import ModelSpec
from .lifecycle import run_arm,run_restore
from .numerical import check_peft_parity,check_loss_reduction


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('arm','restore','preflight'),required=True)
    p.add_argument('--backend-json',type=Path,required=True)
    p.add_argument('--arm',choices=('L0','FI','FM','LI','LM'))
    p.add_argument('--public',type=Path,required=True)
    p.add_argument('--private',type=Path,required=True)
    p.add_argument('--run-config',type=Path,required=True)
    a=p.parse_args();a.public.mkdir(parents=True,exist_ok=True)
    cfg=read_json(a.run_config)
    torch.set_num_threads(cfg['threads']);torch.set_num_interop_threads(1)
    try:
        if a.mode=='preflight':
            requested=read_json(a.backend_json)['kind']
            backend=resolve_backend(requested)
            write_json(a.public/'backend_resolved.json',backend)
            result={'reduction':check_loss_reduction(),
                    'standard_lora_peft':check_peft_parity() if requested=='chronos' else {'status':'NOT_RUN_PEFT_UNAVAILABLE_LOCAL_TEST_DOUBLE'}}
            write_json(a.public/'preflight.json',result);return 0
        backend=read_json(a.backend_json)
        if a.mode=='arm':
            spec=ModelSpec(a.arm,rank=cfg['rank'],alpha=cfg['alpha'],tau_init=tuple(cfg['tau_init']),seed=cfg['seed'])
            return 0 if run_arm(backend,spec,a.public,a.private,steps=cfg['steps'],lr=cfg['lr']) else 2
        run_restore(a.private,a.public);return 0
    except BaseException as e:
        error={'status':'FAILED_NO_RETRY','mode':a.mode,'arm':a.arm,'exception_type':type(e).__name__,
               'message':redact(str(e)),'traceback':redact(traceback.format_exc())}
        write_json(a.public/(a.mode+'_FAILURE.json'),error)
        print(error['traceback'],file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
