"""Predeclare one additive candidate and immutable warm-start/data provenance."""
import subprocess,shutil
from .common import *
PREV=ROOT/'results/outlier_signal_followup_v2_20260917'
PREVC=ROOT/'.cache/outlier_signal_followup_v2_20260917'

def prepare():
    if (OUT/'SOURCE_MANIFEST.json').exists():
        for path,h in read(OUT/'SOURCE_MANIFEST.json')['immutable_hashes'].items():assert sha(ROOT/path)==h,path
        return
    assert read(PREV/'verification.json')['status']=='VERIFIED'
    immutable={}
    for base in ['results/outlier_signal_followup_v2_20260917','experiments/outlier_signal_followup_v2_20260917','results/outlier_signal_peft_v1_20260917','experiments/outlier_signal_peft_v1_20260917']:
        for p in (ROOT/base).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:immutable[str(p.relative_to(ROOT))]=sha(p)
    for info in read(OUT/'download_receipts.json').values():
        for p,h in info['files'].items():assert sha(ROOT/p)==h;immutable[p]=h
    for p,h in read(PREV/'AUGMENTATION_MANIFEST.json')['hashes'].items():assert sha(ROOT/p)==h;immutable[p]=h
    for info in read(OUT/'DATA_MANIFEST.json').values():
        assert sha(ROOT/info['path'])==info['sha256'];immutable[info['path']]=info['sha256']
        for files in info['packet_hashes'].values():
            for p,h in files.items():assert sha(ROOT/p)==h;immutable[p]=h
    models={}
    for source in SOURCES:
        chosen=read(PREV/'LR_SELECTION.json')[source]['B0'];r=read(PREV/'fits'/chosen['fit']/'receipt.json')
        models[f'{source}_81550']=dict(source=source,seed=81550,old_fit=r['fit'],**r['selected'])
    for r in read(PREV/'MODEL_SELECTION.json'):
        if r['arm']=='B0':models[f"{r['source']}_{r['seed']}"]=r
    for r in models.values():assert sha(ROOT/r['checkpoint'])==r['sha256'];immutable[r['checkpoint']]=r['sha256']
    predictions={}
    for r in read(PREV/'predictions_manifest.json').values():
        if r['arm']!='B0':continue
        assert sha(ROOT/r['path'])==r['sha256'];immutable[r['path']]=r['sha256'];predictions[f"{r['source']}_C0_{r['seed']}"]=r
        v=PREVC/'predictions'/f"{r['source']}_B0_{r['seed']}_V.npy";immutable[str(v.relative_to(ROOT))]=sha(v)
    assert len(models)==6 and len(predictions)==4
    save(OUT/'BASELINE_MANIFEST.json',dict(models=models,predictions=predictions,selection='v2 V-only chosen B0, paired optimizer seed; never choose baseline by E',upstream_training_is_sunk_shared_cost=True))
    save(OUT/'SOURCE_MANIFEST.json',dict(initial_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),immutable_hashes=immutable,previous_results_preserved=True))
    save(OUT/'MACHINE_CONTRACT.json',dict(status='DESIGN_SEALED_BEFORE_NEW_UPDATES',arms={'C0':'stored B0 unchanged','C1':'continue existing B0 LoRA, 294912 trainable','C2':'freeze B0 + generic residual adapter, 8712 trainable','C3':'freeze B0 + persistence-masked residual adapter, 8712 trainable'},
        sources=SOURCES,seeds=dict(select=81550,repeat=[81551,81552]),lrs=[1e-4,3e-4],epochs=32,updates_per_fit=1024,fit_cap=24,main_updates_cap=24576,smoke_updates_cap=12,
        optimizer=dict(name='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=0,clip_norm=1,scheduler=None,reset_moments_at_warmstart=True),
        batch=32,microbatch_fallback=[32,16,8,4],precision='FP32, TF32 off, dropout0',checkpoints=[0,256,512,768,1024],
        V_objective='equal mean nMAE REFERENCE/POINT8/BURST8/SHIFT4/SHIFT8',tie='smaller LR, then earlier checkpoint',
        gate='1 - mean(p_t over native 16-slot patch)',persistence='same-sign abs deviation>3 in trailing8 observed slots; truncated boundary; observed median/MAD floor0.1TRAINsigma',
        residual='s_detached*tanh(W_up GELU(W_down e)), 512->8->512, biases, W_up and b_up zero',
        adapter_seed_offset=200000,shuffle_key=84100,p_permutation_key=84400,bootstrap_key=84500,
        cpu_tolerance=dict(rtol=1e-10,atol=1e-12),gpu_tolerance='normalized max1e-5 OR rtol1e-4; starting and off-path B0 parity must be bitwise',
        evaluation='previously exposed development E; all new E predictions saved before scoring; no independent test claim',
        main_contrasts=['C3/C0','C3/C1','C3/C2','C2/C0','C2/C1','C1/C0'],performance_gate=False,automatic_followup=False))
    print('AUDIT_AND_DESIGN_READY',flush=True)
