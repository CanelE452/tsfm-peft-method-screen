"""Required checkpoint-zero diagnostics; inference only, before any TEST scoring."""
import sys,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from common import *
from runner import check_selection
from model import ModelA,setup
from A.data import Solar
from evaluation import predict,path

if __name__=='__main__':
    assert read(RESULTS/'RUN_COMPLETE.json')['status']=='COMPLETE'
    assert not (RESULTS/'ANALYSIS_STARTED.json').exists()
    setup();check_selection()
    with executor_lock():
        if 'A' in read(RESULTS/'EXECUTION_PLAN.json')['completed_candidates']:
            d=Solar();packet=d.batch(d.pairs['TEST'])
            for seed in SEEDS:
                for arm in ARMS_A:
                    states=torch.load(CACHE/'A'/'fits'/f'{arm}_{seed}'/'states.pt',weights_only=True)
                    m=ModelA(arm,seed);m.load_learned(states[0])
                    predict(m,packet,'A',path('A',seed,arm,'TEST',0))
                    del m,states;gc.collect();torch.cuda.empty_cache()
            save(RESULTS/'A'/'ZERO_DIAGNOSTIC_MANIFEST.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in sorted((CACHE/'A'/'predictions').rglob('TEST_0.npz'))},checkpoint_zero_raw_only=True,used_for_selection=False))
        save(RESULTS/'ALL_DIAGNOSTIC_PREDICTIONS_COMPLETE.json',dict(status='COMPLETE',additional_optimizer_updates=0,test_scoring_started=False))
        event('all_diagnostic_predictions_complete')
