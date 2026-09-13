"""Audit only JSON/metadata/history; never loads model, array values or GPU."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/forecast_query_equal_time_plan.json'
OUT = ROOT/'research/forecast_query_equal_time_plan'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def origins(spec):
    assert spec['stride'] > 0 and spec['stop_inclusive'] >= spec['start']
    values=list(range(spec['start'],spec['stop_inclusive']+1,spec['stride']))
    assert values[-1] == spec['stop_inclusive']
    return values


def target_set(oo,h):
    return {i for o in oo for i in range(o,o+h)}


def main():
    cfg=json.loads(CONFIG.read_text())
    tr,va,ev=[origins(cfg[k+'_origins']) for k in ('train','validation','evaluation')]
    h=cfg['horizon']
    assert min(tr) >= cfg['context']
    assert max(tr)+h <= min(va) and max(va)+h <= min(ev)
    assert not target_set(tr,h)&target_set(va,h)
    assert not target_set(va,h)&target_set(ev,h)
    assert cfg['train_scale_interval_half_open'] == [min(tr),max(tr)+h]
    assert len(tr)==193 and len(va)==8 and len(ev)==16
    assert cfg['arms']==['standard','head','side','query']
    assert all(len(cfg['learning_rates'][a])==2 for a in cfg['arms'])
    count=len(cfg['datasets'])*len(cfg['seeds'])*sum(len(cfg['learning_rates'][a]) for a in cfg['arms'])
    assert count==cfg['fit_attempt_cap']==32
    assert count*cfg['active_training_seconds_per_recipe']==cfg['nominal_total_active_training_seconds']==960
    assert cfg['checkpoint_training_seconds']==[0,10,20,30]
    pf=cfg['storage_preflight']
    n=len(cfg['datasets'])*len(cfg['arms'])*len(pf['checkpoint_options'])
    assert n*pf['bf16_repeats']==pf['bf16_measured_optimizer_updates']==48
    assert n==pf['fp32_equivalence_optimizer_updates']==16
    warm=len(cfg['arms'])*pf['state_warmup_updates_per_arm']+len(cfg['arms'])*2*pf['kernel_warmup_updates_per_variant']
    assert warm==pf['warmup_optimizer_updates']==16
    assert 48+16+warm==pf['total_optimizer_update_cap']==80
    old_query=json.loads((ROOT/'configs/forecast_query_pilot.json').read_text())
    old_block=json.loads((ROOT/'configs/block_shape_pilot.json').read_text())
    assert cfg['learning_rates']==old_query['recipes']
    sources={str(CONFIG.relative_to(ROOT)):sha(CONFIG), str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    for f in ['configs/forecast_query_pilot.json','configs/block_shape_pilot.json',
              'docs/FORECAST_QUERY_EQUAL_TIME_PLAN.md']:
        sources[f]=sha(ROOT/f)
    dataset_rows=[]
    for name in cfg['datasets']:
        path=ROOT/'data/processed'/name/'manifest.json'
        meta=json.loads(path.read_text())
        sources[str(path.relative_to(ROOT))]=sha(path)
        assert max(ev)+h <= meta['bounds']['end']
        past={
            'original_screen':meta['origins']['evaluation'],
            'forecast_query':origins(old_query['evaluation_origins']),
            'block_shape':old_block['evaluation_origins']
        }
        for label,oo in past.items():
            assert not target_set(oo,h)&target_set(ev,h),(name,label)
        # Hash bytes only; no numpy/dataframe/model code or target statistics.
        data_files={}
        for file,expected in meta['files'].items():
            data_path=path.parent/file
            actual=sha(data_path)
            assert actual==expected,(name,file)
            data_files[file]=actual
        dataset_rows.append(dict(dataset=name,rows=meta['bounds']['end'],
                                 frequency=meta['frequency'],evaluation_target_end_exclusive=max(ev)+h,
                                 previous_scored_target_end_exclusive=max(max(oo)+h for oo in past.values()),
                                 data_file_hashes=data_files,
                                 scope='No overlap with recorded scoring origins in this repository; not a pretraining or external-history audit.'))
    manifest=[dict(dataset=d,seed=s,arm=a,lr=lr,nominal_active_training_seconds=30)
              for d in cfg['datasets'] for s in cfg['seeds'] for a in cfg['arms']
              for lr in cfg['learning_rates'][a]]
    # Fixed hash order avoids importing a random library/version-dependent shuffle.
    manifest.sort(key=lambda r:hashlib.sha256(('forecast-query-time-v1:'+json.dumps(r,sort_keys=True)).encode()).hexdigest())
    for i,row in enumerate(manifest):
        row['attempt_order']=i
    report=dict(status='DESIGN_AUDIT_PASS_NOT_EXECUTED',
                evidence_base_commit='6225a995332f980095e16df1085750f80b59dbbc',
                source_hashes=sources,datasets=dataset_rows,
                origins=dict(train=tr,validation=va,evaluation=ev),
                fit_attempt_cap=count,nominal_active_training_seconds=960,
                preflight_optimizer_update_cap=80,
                completed_fits=0,new_gpu_jobs=0,new_optimizer_updates=0,
                target_array_values_loaded=False,new_evaluation_scores=0,
                pending_implementation=['Head/Side checkpoint wrapper','equal-time runner','isolated E loader and seals','GPU parity/preflight','independent result finalizer'])
    OUT.mkdir(parents=True,exist_ok=True)
    for filename,obj in [('design_audit.json',report),('fit_manifest.json',manifest)]:
        (OUT/filename).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('DESIGN AUDIT PASS; 32 planned attempts / 960 nominal active seconds / 80 preflight updates; executed0.')


if __name__=='__main__':
    main()
