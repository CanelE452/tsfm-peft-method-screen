import subprocess,unittest,io
from .common import *
from experiments.persistence_evidence_extension_20260918 import common as ext

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    if (OUT/'SEAL.json').exists():return check_seal()
    assert not (OUT/'UPDATE_LEDGER.jsonl').exists()
    hashes={}
    def add(p):
        p=Path(p);hashes[str(p.relative_to(ROOT))]=sha(p)
    for folder in [EXP,ROOT/'experiments/additive_persistence_validation_v1_20260917',ROOT/'experiments/additive_b0_adapter_v1_20260917',ROOT/'experiments/outlier_signal_followup_v2_20260917',ROOT/'experiments/outlier_signal_peft_v1_20260917',ROOT/'experiments/persistence_evidence_extension_20260918',ROOT/'experiments/c3_weakness_controls_20260918',ROOT/'experiments/c3_training_factorial_20260918']:
        for p in folder.glob('*.py'):add(p)
    add(EXP/'PROTOCOL.md');add(ROOT/'scripts/priority12/common.py')
    data=read(old.OUT/'DATA_MANIFEST.json');save(OUT/'DATA_MANIFEST.json',{p:data[p] for p in ['electricity','electricity_transfer','ettm1']});add(OUT/'DATA_MANIFEST.json')
    for source in SOURCES:
        for name in ['train_x.npy','train_y.npy','train_sigma.npy','V_SELECT_x.npy','V_SELECT_y.npy','V_SELECT_sigma.npy']:
            p=old.CACHE/'conditions'/source/name
            previous=ROOT/'.cache/additive_b0_adapter_v1_20260917/conditions'/source/name
            assert sha(p)==sha(previous),('PLAIN_DATA_MISMATCH',p);add(p)
    for panel in ['electricity','electricity_transfer','ettm1']:
        for kind in ['standard','shape']:
            for p in ext.panel_path(panel,kind).iterdir():
                if p.suffix in ['.npy','.json'] and (p.name.startswith('E_DISCOVERY') or kind=='shape'):add(p)
        for p in ext.data_path(panel).glob('E_DISCOVERY*.npz'):add(p)
    reused=[]
    for source in SOURCES:
        for seed in SEEDS:
            b=old.baseline_row(source,seed);assert sha(ROOT/b['checkpoint'])==b['sha256'];add(ROOT/b['checkpoint'])
            p=ROOT/'results/additive_b0_adapter_v1_20260917/fits'/f'{source}_C2_s{seed}_lr0.0003'/'receipt.json'
            r=read(p);assert r['status']=='COMPLETE' and r['updates']==1024 and r['microbatch']==32 and r['lr']==LR and r['frozen_unchanged'];add(p)
            assert sorted(c['step'] for c in r['checkpoints'])==[0,256,512,768,1024]
            assert min(r['checkpoints'],key=lambda c:(c['objective'],c['step']))==r['selected']
            for c in r['checkpoints']:assert sha(ROOT/c['checkpoint'])==c['sha256'];add(ROOT/c['checkpoint'])
            row=dict(r,arm='PLAIN',fit=fit_id(source,'PLAIN',seed),reused=True,original_receipt=str(p.relative_to(ROOT)),new_updates=0)
            save(OUT/'fits'/row['fit']/'receipt.json',row);reused.append(row)
    snapshot=read(ROOT/'results/outlier_signal_followup_v2_20260917/download_receipts.json')['amazon/chronos-bolt-small']['snapshot']
    for p in (ROOT/snapshot).rglob('*'):
        if p.is_file() and p.suffix in ['.json','.safetensors','.bin']:add(p)
    for p in [old.OUT/'BASELINE_MANIFEST.json',ROOT/'results/additive_b0_adapter_v1_20260917/LR_SELECTION.json']:add(p)
    grid=[dict(source=s,seed=z,arm=a,lr=LR,fit=fit_id(s,a,z)) for s in SOURCES for z in SEEDS for a in NEW_ARMS]
    save(OUT/'GRID.json',grid);add(OUT/'GRID.json')
    save(OUT/'REUSE_AUDIT.json',dict(paths=4,all_training_and_validation_arrays_exact=True,all_checkpoint_hashes_verified=True,initial_model_parity_to_be_checked_on_GPU=True,receipts=[r['original_receipt'] for r in reused]));add(OUT/'REUSE_AUDIT.json')
    save(OUT/'SEAL.json',dict(at=time.time(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),hashes=hashes,new_main_cap=MAIN_CAP,smoke_cap=SMOKE_CAP,actual_gpu_checked=False,protocol=str((EXP/'PROTOCOL.md').relative_to(ROOT))))
    status(execution='SEALED_NOT_TRAINED')

def cpu_checks():
    s=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromNames(['experiments.temporal_response_peft_20260919.test_core','experiments.temporal_response_peft_20260919.test_reuse']);r=unittest.TextTestRunner(stream=s,verbosity=2).run(suite)
    (OUT/'CPU_TESTS.txt').write_text(s.getvalue());save(OUT/'CPU_TESTS.json',dict(tests=r.testsRun,success=r.wasSuccessful(),actual_model=False));assert r.wasSuccessful()
if __name__=='__main__':setup();prepare();cpu_checks()
