"""CPU-only independent replay of all saved pilot predictions and V choices."""
import argparse
import csv
import json
import subprocess
import time
import numpy as np
from tsfm_peft_screen.reproducibility import ROOT, sha, digest, write_json
from tsfm_peft_screen.metrics import independent, score
from tsfm_peft_screen.overnight.methods import mixture_quantiles, teacher_weights
from tsfm_peft_screen.calibration_anchor.method import TOPICS, PROPOSED, verdict_from_rows, calibration_weights
from run_calibration_anchor_followup import CFG, OUT, CACHE, oo


def read(path):
    return json.loads(path.read_text())


def check_seal(path):
    s = read(path)
    checksum = s.pop('seal_sha256')
    assert digest(s) == checksum
    return s


def finalize(verify_only=False):
    contract = read(OUT/'contract.json')
    assert contract['config'] == CFG
    for f,h in contract['source_hashes'].items():
        assert sha(ROOT/f) == h, f
        import hashlib
        original = subprocess.check_output(['git','show',contract['execution_commit']+':'+f],cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == h, ('execution source',f)
    for f,h in contract['historical_result_hashes'].items():
        assert sha(ROOT/f) == h, f
    barrier = read(OUT/'evaluation_barrier.json')
    checksum = barrier.pop('sha256')
    assert digest(barrier) == checksum
    assert barrier['contract_sha256'] == sha(OUT/'contract.json')
    assert set(barrier['topics']) == set(CFG['topics'])
    for topic, entry in barrier['topics'].items():
        for f,h in entry['receipts'].items():
            assert sha(OUT/topic/f) == h
    cache_count = 0
    max_error = 0.
    seen = set()

    def replay(row):
        nonlocal cache_count, max_error
        f = CACHE/row['prediction_file']
        assert sha(f) == row['prediction_sha256']
        if str(f) not in seen:
            with np.load(f,allow_pickle=False) as z:
                actual = independent(z['prediction'],z['target'],z['scale'])
                assert np.isfinite(actual)
            err = abs(actual-row['metrics']['scaled_2pinball'])
            assert err <= 1e-10, f
            max_error = max(max_error,err)
            cache_count += 1
            seen.add(str(f))

    summaries, totals = [], dict(fits_attempted=0,fits_completed=0,training_updates=0,active_seconds=0.,teacher_cache_wall_seconds=0.)
    for topic in CFG['topics']:
        folder = OUT/topic
        train_state = read(folder/'train_status.json') if (folder/'train_status.json').exists() else {}
        totals['fits_attempted'] += train_state.get('fits_attempted',0)
        totals['training_updates'] += train_state.get('training_updates',0)
        decision = barrier['topics'][topic]['decision']
        if decision == 'INCONCLUSIVE_EXECUTION':
            summaries.append(dict(topic=topic,verdict=decision,error=train_state.get('error','Worker did not finish training')))
            continue
        seal = check_seal(folder/'selection_seal.json')
        assert seal['sealed_unix'] <= barrier['sealed_unix']
        assert sha(folder/'fits.json') == seal['fits_sha256']
        fits = read(folder/'fits.json')
        expected_arms = ['raw'] if decision == 'STOP_NO_TEACHER_HEADROOM' else TOPICS[topic]
        expected = {(d,s,a,r) for d in CFG['datasets'] for s in CFG['seeds'] for a in expected_arms for r in range(2)}
        assert {(f['dataset'],f['seed'],f['arm'],f['recipe']) for f in fits} == expected
        assert len(fits) == len(expected) == train_state['fits_completed'] == train_state['fits_attempted']
        totals['fits_completed'] += len(fits)
        for fit in fits:
            full_path = folder/(fit['fit_id']+'_fit.json')
            assert sha(full_path) == fit['resource_receipt_sha256']
            full = read(full_path)
            assert {k:v for k,v in full.items() if k != 'resources'} == {
                k:v for k,v in fit.items() if k != 'resource_receipt_sha256'}
            fit['resources'] = full['resources']
            assert fit['updates'] == 900
            assert [r['step'] for r in fit['resources']] == list(range(1,901))
            assert fit['initial_parameters_sha256'] != fit['final_parameters_sha256']
            assert abs(sum(r['seconds'] for r in fit['resources'])-fit['active_seconds']) < 1e-7
            assert max(r['peak_allocated_bytes'] for r in fit['resources']) <= 8*2**30
            assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in fit['resources'])
            rng = np.random.default_rng(fit['seed'])
            origins = oo('train')
            sequence = [[origins[i] for i in rng.integers(len(origins),size=2)] for _ in range(900)]
            assert digest(sequence) == fit['sample_sequence_sha256']
            rows = read(folder/(fit['fit_id']+'_trajectory.json'))
            assert [r['step'] for r in rows] == CFG['checkpoints']
            for row in rows:
                replay(row)
                assert sha(CACHE/row['checkpoint_file']) == row['checkpoint_sha256']
            assert min(rows,key=lambda r:(r['metrics']['scaled_2pinball'],r['step'],r['lr'])) == fit['best']
            totals['active_seconds'] += fit['active_seconds']
        assert sum(f['updates'] for f in fits) == train_state['training_updates']
        assert len(seal['selections']) == len(expected)//2
        for selected in seal['selections']:
            pool = [f['best'] for f in fits if all(f[k] == selected[k] for k in ['dataset','seed','arm'])]
            assert min(pool,key=lambda r:(r['metrics']['scaled_2pinball'],r['step'],r['lr'])) == selected
        if (folder/'teachers.json').exists():
            teachers = read(folder/'teachers.json')
            for dataset, record in teachers.items():
                totals['teacher_cache_wall_seconds'] += record['wall_seconds']
                record_cal = record['calibration']
                weight_path = CACHE/record_cal['weights_file']
                assert sha(weight_path) == record_cal['weights_sha256']
                train_row = record['train'][0]
                with np.load(CACHE/train_row['prediction_file'],allow_pickle=False) as z:
                    weights, expected_cal = calibration_weights(z['prediction'],z['target'])
                with np.load(weight_path,allow_pickle=False) as z:
                    assert np.array_equal(weights,z['weights'])
                assert all(record_cal[k]==v for k,v in expected_cal.items())
                for split in ('train','validation'):
                    rows = record[split]
                    for row in rows:
                        replay(row)
                    if topic == 'distill':
                        arrays = []
                        for row in rows:
                            if row['context'] != 'mixture':
                                with np.load(CACHE/row['prediction_file'],allow_pickle=False) as f:
                                    arrays.append(f['prediction'])
                        mixture_row = next(r for r in rows if r['context'] == 'mixture')
                        with np.load(CACHE/mixture_row['prediction_file'],allow_pickle=False) as f:
                            assert np.array_equal(mixture_quantiles(np.stack(arrays)),f['prediction'])
                            assert np.array_equal(teacher_weights(np.stack(arrays),f['scale']),f['weights'])
            if topic == 'distill':
                headroom = read(folder/'teacher_headroom.json')
                for row in headroom:
                    d = row['dataset']
                    rr = teachers[d]['validation']
                    mixture = next(r['metrics']['scaled_2pinball'] for r in rr if r['context'] == 'mixture')
                    best = min(r['metrics']['scaled_2pinball'] for r in rr if r['context'] != 'mixture')
                    raw = np.mean([min(f['best']['metrics']['scaled_2pinball'] for f in fits
                        if f['dataset']==d and f['seed']==s and f['arm']=='raw') for s in CFG['seeds']])
                    assert row['teacher'] == mixture and row['best_component'] == best
                    assert abs(row['raw_lora_mean']-raw) < 1e-12
                    assert row['passed'] == (mixture <= (1-CFG['teacher_headroom_gain'])*min(best,raw))
                assert (decision == 'STOP_NO_TEACHER_HEADROOM') == (not all(r['passed'] for r in headroom))
        if decision == 'STOP_NO_TEACHER_HEADROOM':
            assert not (folder/'evaluation.json').exists()
            summaries.append(dict(topic=topic,verdict=decision,headroom=read(folder/'teacher_headroom.json')))
            continue
        estate = read(folder/'evaluate_status.json') if (folder/'evaluate_status.json').exists() else {}
        if estate.get('status') != 'COMPLETE':
            summaries.append(dict(topic=topic,verdict='INCONCLUSIVE_EXECUTION',error=estate.get('error','Evaluation incomplete')))
            continue
        assert estate['evaluation_open_unix'] >= barrier['sealed_unix']
        rows = read(folder/'evaluation.json')
        expected_rows = {(d,s,a) for d in CFG['datasets'] for s in CFG['seeds'] for a in expected_arms}
        assert {(r['dataset'],r['seed'],r['arm']) for r in rows if r['seed'] is not None} == expected_rows
        assert len([r for r in rows if r['seed'] is not None]) == len(expected_rows)
        for r in rows:
            replay(r)
        reloads = read(folder/'selected_reloads.json')
        assert len(reloads) == len(seal['selections'])
        for r in reloads:
            replay(r['replay'])
            with np.load(CACHE/r['selected_prediction'],allow_pickle=False) as a, np.load(CACHE/r['replay']['prediction_file'],allow_pickle=False) as b:
                assert np.array_equal(a['prediction'],b['prediction'])
        result = verdict_from_rows(rows,seal['selections'],topic,CFG)
        recorded = read(folder/'summary.json')
        assert all(recorded[k] == result[k] for k in result)
        # Paired chronological block bootstrap, descriptive only (4 blocks).
        intervals=[]
        for dataset in CFG['datasets']:
            dr=[r for r in rows if r['dataset']==dataset]
            means={a:np.mean([r['metrics']['scaled_2pinball'] for r in dr if r['arm']==a]) for a in {r['arm'] for r in dr}}
            baseline=min((a for a in means if a!=PROPOSED[topic]),key=means.get)
            losses={}
            for arm in (PROPOSED[topic],baseline):
                values=[]
                for r in dr:
                    if r['arm']==arm:
                        with np.load(CACHE/r['prediction_file'],allow_pickle=False) as f:
                            values.append(np.array([independent(f['prediction'][i:i+1],f['target'][i:i+1],f['scale']) for i in range(len(oo('evaluation')))]))
                losses[arm]=np.mean(values,axis=0)
            differences=losses[PROPOSED[topic]]-losses[baseline]
            blocks=differences.reshape(4,-1).mean(1)
            rng=np.random.default_rng(61000)
            draws=blocks[rng.integers(4,size=(1000,4))].mean(1)
            intervals.append(dict(dataset=dataset,baseline=baseline,mean_loss_difference=float(differences.mean()),
                descriptive_95_interval=np.quantile(draws,[.025,.975]).tolist(),chronological_blocks=4))
        result.update(topic=topic,paired_intervals=intervals)
        summaries.append(result)
    assert totals['fits_attempted'] <= CFG['max_total_fit_attempts']
    assert totals['training_updates'] <= CFG['max_training_updates']
    verification=dict(status='VERIFIED',replayed_prediction_files=cache_count,max_primary_replay_error=max_error,
        all_historical_files_unchanged=len(contract['historical_result_hashes']),totals=totals,
        topics=summaries,scope='PILOT_PASS means follow-up signal only; novelty, replication and publication claims remain unresolved.')
    if verify_only:
        assert read(OUT/'verification.json') == verification
    else:
        write_json(OUT/'verification.json',verification)
        text=['# Calibration-aware anchoring follow-up','',
              'Development continuation screen; no publication PASS or established novelty. Historical results are unchanged.',
              '',f"Completed fits: {totals['fits_completed']}; attempted: {totals['fits_attempted']}; optimizer updates: {totals['training_updates']}.",
              '', '| Topic | Verdict | ETTh1 gain vs best | Traffic gain vs best |',
              '| --- | --- | ---: | ---: |']
        table=[]
        for r in summaries:
            gains={c['dataset']:100*c['gain_vs_best'] for c in r.get('checks',[])}
            left=f"{gains['etth1']:.4f}%" if 'etth1' in gains else '—'
            right=f"{gains['traffic']:.4f}%" if 'traffic' in gains else '—'
            text.append(f"| {r['topic']} | {r['verdict']} | {left} | {right} |")
            table.append(dict(topic=r['topic'],verdict=r['verdict'],etth1_gain_percent=gains.get('etth1'),traffic_gain_percent=gains.get('traffic')))
        text.extend(['',f"All {cache_count} recorded prediction files independently replayed; maximum primary error {max_error:.3g}.",
                     '', 'New eight-origin E tail per dataset; previous train/V reused adaptively. Adjacent same-series development holdout, not independent-source replication. All V decisions sealed before E.',
                     'Two seeds and four chronological bootstrap blocks give development evidence, not a strong significance claim.',
                     'Equal updates are enforced; unequal wall time and all teacher preparation costs are reported.',
                     '', 'Known ingredients: output anchoring, empirical calibration weighting and LoRA. A positive pilot still needs a distinct method contribution and independent replication.',''])
        (OUT/'REPORT.md').write_text('\n'.join(text))
        with open(OUT/'ranking.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(table)
    print(json.dumps(dict(status='VERIFIED',totals=totals,topics=[(r['topic'],r['verdict']) for r in summaries]),indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--verify-only',action='store_true')
    finalize(p.parse_args().verify_only)
