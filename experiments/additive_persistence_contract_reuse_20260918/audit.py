"""Read-only reuse audit of the identical MASTER_CLI; never invokes a trainer."""
import io,unittest,subprocess,collections,traceback
from experiments.additive_persistence_validation_v1_20260917 import common as old
from experiments.additive_persistence_validation_v1_20260917 import verify as verifier
from experiments.additive_persistence_validation_v1_20260917.reference_checks import ReferenceChecks
from experiments.additive_persistence_validation_v1_20260917.common import ROOT,read,save,sha
PARENT=old.OUT
OUT=ROOT/'results/additive_persistence_contract_reuse_20260918'
class Watch(old.Watch):
    def __init__(self):
        self.last_disk=0
        old.OriginalWatch.__init__(self,OUT,'restore',wall_cap=3600)

def run():
    OUT.mkdir(parents=True,exist_ok=True);old.setup()
    paths=[old.Path('/home/minjae/Downloads/additive_persistence_paper_cli_20260917.txt'),old.Path('/home/minjae/Downloads/additive_persistence_paper_cli_20260917 (1).txt'),PARENT/'MASTER_CLI.txt'];digest=sha(paths[0]);assert digest=='409af4da5ae2fb77f9c25f24081631a064083fe0012b646cace20f01ed4a16f5'
    assert all(sha(p)==digest for p in paths)
    # All old output bytes are preserved, including intermediate status snapshots.
    before={str(p.relative_to(ROOT)):sha(p) for p in PARENT.rglob('*') if p.is_file()}
    save(OUT/'PRE_AUDIT.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),contract_sha256=digest,matched_paths=[str(p) for p in paths],parent_hashes=before,new_training_authorized=False,additional_optimizer_updates_cap=0,reuse_reason='same complete contract; repeated request does not reset its consumed budget',excluded_studies=['NESO2025/2026 extensions','paper_readiness','C3 redesign']))
    seal=old.check_seal();checked={}
    def check(path,h):
        p=old.Path(path);p=p if p.is_absolute() else ROOT/p
        assert p.is_file(),('MISSING',str(p));key=str(p.relative_to(ROOT));actual=checked.get(key)
        if actual is None:actual=sha(p);checked[key]=actual
        assert actual==h,('HASH_MISMATCH',key)
    for bucket in ['source_hashes','data_hashes']:
        for p,h in seal[bucket].items():check(p,h)
    for fname,key in [('SOURCE_MANIFEST.json','immutable_hashes'),('DATA_READY.json','hashes'),('TRAINING_COMPLETION_AUDIT.json','hashes')]:
        for p,h in read(PARENT/fname)[key].items():check(p,h)
    for fname in ['FINAL_ARTIFACT_AUDIT.json','PUBLICATION_AUDIT.json']:
        d=read(PARENT/fname)
        for p,h in d['required_artifact_hashes'].items():check(PARENT/p,h)
        for p,h in d.get('reporting_source_hashes',{}).items():check(p,h)
    fitcounts=collections.Counter();receipts=[];checkpoints=[]
    for p in sorted((PARENT/'fits').glob('*/receipt.json')):
        r=read(p);receipts.append(r);assert r['status']=='COMPLETE' and r['updates']==1024 and r['frozen_unchanged'];assert [c['step'] for c in r['checkpoints']]==[0,256,512,768,1024]
        fitcounts['ETTm2' if r['source']=='ettm2' else 'controls' if r['arm'].startswith('M_') else 'third_seed']+=1
        for c in r['checkpoints']:check(c['checkpoint'],c['sha256']);checkpoints.append(c['checkpoint'])
    assert dict(fitcounts)=={'third_seed':8,'controls':30,'ETTm2':20} or fitcounts==collections.Counter(third_seed=8,controls=30,ETTm2=20)
    models=read(PARENT/'MODEL_SELECTION.json');assert len(models)==54
    for r in models:check(r['checkpoint'],r['sha256'])
    ev=read(PARENT/'GLOBAL_EVALUATION_SEAL.json');assert ev['master_seal_sha256']==sha(PARENT/'MASTER_SEAL.json');assert ev['model_selection_sha256']==sha(PARENT/'MODEL_SELECTION.json');assert ev['calibration_selection_sha256']==sha(PARENT/'CALIBRATION_SELECTION.json')
    marker=read(PARENT/'ALL_PREDICTIONS_SAVED.json');assert ev['at']<marker['at'];assert marker['manifest_sha256']==sha(PARENT/'PREDICTIONS_MANIFEST.json')
    preds=read(PARENT/'PREDICTIONS_MANIFEST.json');assert len(preds)==246
    for r in preds.values():check(r['path'],r['sha256'])
    # Contract reference tests are local mathematical tests, not actual Chronos execution.
    stream=io.StringIO();test=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceChecks));(OUT/'CPU_REFERENCE_LOG.txt').write_text(stream.getvalue());assert test.wasSuccessful() and test.testsRun==20
    save(OUT/'CPU_REFERENCE_CHECKS.json',dict(status='VERIFIED',tests=test.testsRun,actual_model=False,reference_origin='existing locally authored reference_checks.py; original supplied PY was unavailable'))
    # Execute the existing full verifier, redirecting only its three output receipts.
    # It replays metrics and fixed checkpoint forwards; it has no optimizer call.
    original_save,original_csv=verifier.save,verifier.csvwrite
    def redirected_save(path,value):
        path=old.Path(path);assert path==PARENT/'VERIFICATION.json';save(OUT/'REPLAY_VERIFICATION.json',value)
    def redirected_csv(path,rows):
        path=old.Path(path);assert path.parent==PARENT and path.name in ['FIT_LEDGER.csv','SCALAR_REPLAY_SCOPE.csv'];original_csv(OUT/path.name,rows)
    verifier.save,verifier.csvwrite=redirected_save,redirected_csv;watch=None
    try:
        watch=Watch();watch.boundary(startup=True);verifier.verify(watch)
    finally:
        verifier.save,verifier.csvwrite=original_save,original_csv
        if watch:watch.close()
    after={str(p.relative_to(ROOT)):sha(p) for p in PARENT.rglob('*') if p.is_file()};assert before==after,'PARENT_CHANGED'
    v=read(OUT/'REPLAY_VERIFICATION.json');assert v['status']=='VERIFIED' and v['total_updates']==59428 and len(v['restored_selected_models'])==54
    logs=[old.json.loads(x) for x in (OUT/'gpu_restore.jsonl').read_text().splitlines()];assert not any(any(not a['own'] and a['name']!='/usr/share/rustdesk/rustdesk' for a in r['apps']) for r in logs)
    save(OUT/'VERIFICATION.json',dict(status='VERIFIED_REUSE',contract_sha256=digest,original_main_fits=58,original_main_updates=59392,original_smoke_updates=36,original_total_updates=59428,original_block_fits=dict(fitcounts),this_request_new_fits=0,this_request_optimizer_updates=0,this_request_new_prediction_datasets=0,checkpoint_entries_checked=len(checkpoints),unique_checkpoint_paths_checked=len(set(checkpoints)),selected_models_replayed=54,prediction_views_hash_checked=246,independent_scalar_rows=v['independent_scalar_rows'],scalar_metrics_per_row=2,full_origin_vector_reaggregation=v['all_origin_vector_reaggregation'],unique_files_hash_checked=len(checked),all_parent_outputs_unchanged=len(before),cpu_reference_checks=20,minimum_free_gpu_mib=min(r['free_mib'] for r in logs),unapproved_compute_samples=0,remaining_in_contract_training=0,remaining_in_contract_evaluation=0,unexecuted_out_of_contract=['formal COSA/TATO full comparison','independent-source certification','actual-event labels','other-backbone experiments'],scientific_result='see immutable original FINAL_DECISION.md; not paper PASS',automatic_successor=False))
    save(OUT/'CHECKED_HASHES.json',checked);print('VERIFIED_REUSE: 58 existing fits, 54 restored models, 246 prediction views; 0 new updates',flush=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        save(OUT/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),new_updates=0));raise
