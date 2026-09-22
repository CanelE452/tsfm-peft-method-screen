from common import *
import math
import subprocess
import csv
import re


def main():
    ledger=read(RESULTS/'OPTIMIZER_LEDGER.json')
    assert (ledger['main'],ledger['smoke'],ledger['Q'],ledger['T'],ledger['F'])==(8192,24,3072,2560,2560)
    assert sum(ledger['runs'].values())==8216
    successful_smoke=sum(read(p).get('updates',0) for p in (RESULTS/'smoke').glob('*.json'))
    assert successful_smoke==23
    failure=read(RESULTS/'F_SMOKE_FAILURE.json')
    assert failure['failed_smoke_updates_charged']==1
    assert sha(EXP/'forensics/F_smoke_original.py')==failure['original_source_sha256']
    fits=list((RESULTS/'fits').glob('*/FIT.json'))
    assert len(fits)==32
    all_fit=[]
    for path in fits:
        result=read(path);run=result['run']
        assert ledger['runs'][run]==256 and result['updates']==256 and result['frozen_unchanged']
        lines=[json.loads(s) for s in (path.parent/'training.jsonl').read_text().splitlines()]
        assert len(lines)==256
        for line in lines:
            for k,v in line.items():
                if isinstance(v,float):assert math.isfinite(v),(run,k)
        for item in result['validation']:
            prefix='round' if run.startswith('F_') else 'step'
            checkpoint=CACHE/'fits'/run/f"{prefix}{item['step']}.pt"
            assert sha(checkpoint)==item['checkpoint_sha256']
        if run.startswith('T_') and not run.startswith('T_A1'):
            access=read(path.parent/'ACCESS.json')
            assert not access['teacher_modules_in_student']
            assert all(not s.replace('\\','/').startswith('data/') for s in access['cache_reads'])
        if run.startswith('F_'):
            if not run.startswith('F_LOCAL'):
                assert len(result['communication'])==16
                for info in result['communication']:
                    assert info['numpy_mean_verified'] and not info['raw_sigma_private_uploaded']
                    assert all('.lora_B.' in n for n in info['uploaded_keys'])
            counts={c:sum(x['client']==c for x in lines) for c in range(4)}
            assert set(counts.values())=={64}
        all_fit.append(run)
    sources={}
    for candidate in ['Q','T','F']:
        sealed=read(RESULTS/f'{candidate}_SOURCE_SEAL.json')
        for relative,digest in sealed['source'].items():assert sha(ROOT/relative)==digest,relative
        sources[candidate]=sha(RESULTS/f'{candidate}_SOURCE_SEAL.json')
    assert read(RESULTS/'ARTIFACT_RELOAD_PARITY.json')['status']=='PASS'
    evaluation=read(RESULTS/'EVALUATION_SOURCE_SEAL.json')
    for relative,digest in evaluation['source'].items():assert sha(ROOT/relative)==digest,relative
    qera=read(RESULTS/'QERA_FACTOR_BALANCE_AUDIT.json')
    assert qera['before_test_scoring'] and qera['new_training']==0
    seal=read(RESULTS/'TEST_PREDICTIONS_SEAL.json')
    assert sha(RESULTS/'SELECTIONS.json')==seal['selection_sha256']
    for key,item in seal['predictions'].items():assert sha(ROOT/item['path'])==item['sha256']
    all_files=[]
    for directory in [EXP,RESULTS]:
        all_files.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix!='.pyc')
    forbidden=[p for p in all_files if p.suffix in ['.pt','.npz','.npy','.parquet','.safetensors','.whl']]
    assert not forbidden
    final=read(RESULTS/'FINAL_CLASSIFICATION.json')
    assert final['recommendation'] is None
    for category in ['Q','T','F']:
        for name in ['REPORT_KO.md','FINAL_DECISION.md']:
            assert final[category]['decision'] in (RESULTS/category/name).read_text(encoding='utf-8')
    with (RESULTS/'resource_table.csv').open(newline='',encoding='utf-8') as handle:
        assert len(list(csv.DictReader(handle)))==68
    for name in ['triage_effects','Q_results','T_results','F_results']:
        for extension in ['png','svg']:
            assert (RESULTS/'figures'/f'{name}.{extension}').stat().st_size>1000
    link_count=0
    for path in [p for p in all_files if p.suffix=='.md']:
        for destination in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in destination or destination.startswith('#'):continue
            target=(path.parent/destination.split('#',1)[0]).resolve()
            assert target.exists(),(str(path),destination)
            link_count+=1
    record={'status':'SCOPED_CHECKS_PASS_WITH_DISCLOSED_SMOKE_REPAIR','main_updates':8192,'smoke_updates':24,
            'successful_smoke_updates':23,'failed_smoke_updates':1,'failure_record':'F_SMOKE_FAILURE.json',
            'fits_or_workflows':32,
            'units':'Q12 fits + T10 fits + F10 workflows (F_LOCAL comprises 4 independent client fits per seed)',
            'fits':sorted(all_fit),'source_seals':sources,'all_checkpoint_hashes_verified':True,
            'all_training_logs_finite':True,'all_frozen_persistent_state_hashes_unchanged':True,
            'nonpersistent_buffer_scope':'serialized/original selected Q inference buffers match; training frozen hash uses state_dict persistent state',
            'qera_implementation_scope':'balanced-factor local QERA-diag variant; official runtime factor scaling differs, product equivalence only',
            'qera_factor_difference_disclosed_before_test':True,
            'all_test_predictions_before_scoring':seal['all_saved_before_scoring'],
            'prediction_count':len(seal['predictions']),'no_raw_weights_checkpoints_in_publish_scope':True,
            'report_local_links_verified':link_count,'resource_table_rows':68,'figure_files':8,
            'automatic_followup_experiments':0,'time':time.time()}
    write(RESULTS/'VERIFICATION.json',record)
    manifest={}
    for directory in [EXP,RESULTS]:
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in str(p) and p.name!='MANIFEST.json' and p.suffix!='.pyc':
                manifest[str(p.relative_to(ROOT)).replace('\\','/')]={'sha256':sha(p),'bytes':p.stat().st_size}
    local={}
    for directory in [CACHE/'fits',CACHE/'teacher',CACHE/'test_predictions',CACHE/'metrics',CACHE/'bridge']:
        for p in directory.rglob('*'):
            if p.is_file():local[str(p.relative_to(ROOT)).replace('\\','/')]={'sha256':sha(p),'bytes':p.stat().st_size}
    write(RESULTS/'MANIFEST.json',{'published':manifest,'local_only':local,
          'raw_and_official_model_manifest':'DATA_AND_SPLIT_AUDIT.json / UPSTREAM.json',
          'replay_scope':'GitHub stores code, scalar tables and verification; numerical replay also requires local caches or re-execution',
          'manifest_self_excluded':True})
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
