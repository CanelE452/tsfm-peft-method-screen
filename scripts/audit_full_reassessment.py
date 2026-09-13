"""Read-only historical analysis; writes only a separate retrospective report folder.

Run the historical verify-only scripts separately to replay predictions/gradients.
This script recomputes comparisons, not new models, significance or PASS decisions.
"""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/full_reassessment_20260914'
sources = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    path = ROOT / name
    sources[name] = sha(path)
    return json.loads(path.read_text())


def csv_read(name):
    path = ROOT / name
    sources[name] = sha(path)
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def csv_write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def gain(proposed, baseline):
    return 100 * (baseline - proposed) / baseline


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    historical = {str(p.relative_to(ROOT)): sha(p)
                  for p in sorted((ROOT / 'results').rglob('*')) if p.is_file()}
    comparisons, inventory, endpoints, sensitivity = [], [], [], []
    for i in range(1, 8):
        name = f'candidate_{i:02}'
        v = read(f'results/{name}/status.json')
        inventory.append(dict(experiment=name, fits=v.get('fit_count', 0),
                              stream_attempts=v.get('stream_count', 0),
                              historical_verdict=v['verdict']))
        if i == 5:
            name = 'candidate_05_repaired'
            v = read(f'results/{name}/status.json')
            inventory.append(dict(experiment=name, fits=0,
                                  stream_attempts=v['stream_count'],
                                  historical_verdict=v['verdict']))
        if v.get('proposed_primary') is not None:
            p, b, f0 = (v[k] for k in ('proposed_primary', 'baseline_primary', 'f0_primary'))
            assert abs(100 * (b-p)/f0 - v['gain_percent_f0']) < 1e-10
            comparisons.append(dict(experiment=name, dataset='original_primary',
                proposed=v['proposed'], baseline=v['strongest_baseline'],
                proposed_loss=p, baseline_loss=b, gain_baseline_percent=gain(p,b),
                gain_f0_percent=100*(b-p)/f0, scope='historical_selected_comparator'))
        if i not in (5, 6):
            selected = csv_read(f'results/candidate_{i:02}/selections.csv')
            trajectory = csv_read(f'results/candidate_{i:02}/trajectories.csv')
            for s in selected:
                rr = [r for r in trajectory if r['arm']==s['arm'] and float(r['lr'])==float(s['lr'])]
                last = max(int(r['step']) for r in rr)
                endpoints.append(dict(experiment=f'candidate_{i:02}', arm=s['arm'],
                    selected_step=int(s['step']), final_step=last,
                    at_budget_end=int(s['step'])==last,
                    selected_lr=float(s['lr'])))

    v = read('results/candidate_01_v2/status.json')
    inventory.append(dict(experiment='candidate_01_v2', fits=v['fit_count'],
                          stream_attempts=0, historical_verdict=v['verdict']))
    for s in v['seeds']:
        p,b,f0 = s['proposed_primary'],s['baseline_primary'],v['f0_primary']
        assert abs(100*(b-p)/f0-s['gain_percent_f0']) < 1e-10
        comparisons.append(dict(experiment='candidate_01_v2',dataset=f"jena_seed{s['seed']}",
            proposed='affine', baseline=s['strongest_baseline'], proposed_loss=p,
            baseline_loss=b,gain_baseline_percent=gain(p,b),gain_f0_percent=100*(b-p)/f0,
            scope='per_seed_historical_selected_comparator'))

    for name, verdict in [('memory_feasibility','STOP'),('local_backward_feasibility','STOP'),
                          ('forecast_query_checkpoint_diagnostic','TRADEOFF_ONLY')]:
        read(f'results/{name}/verification.json')
        inventory.append(dict(experiment=name, fits=0, stream_attempts=0, historical_verdict=verdict))
    for name, verdict in [('forecast_query_pilot','FAIL'),('block_shape_pilot','STOP'),
                          ('forecast_query_equal_time','STOP_CURRENT_QUERY')]:
        fits = read(f'results/{name}/fits.json')
        inventory.append(dict(experiment=name, fits=len(fits), stream_attempts=0,
                              historical_verdict=verdict))

    groups = [('overnight_20260913/anchor','prediction_anchor'),
              ('overnight_20260913/drift','drift_gate'),
              ('calibration_anchor_20260914/calibration','calibration_anchor'),
              ('forecast_query_equal_time','query'),('block_shape_pilot','block')]
    derived = {}
    for name, proposed in groups:
        rows = read(f'results/{name}/evaluation.json')
        if name.startswith(('overnight_', 'calibration_')):
            fits=read(f'results/{name}/fits.json')
            inventory.append(dict(experiment=name, fits=len(fits),stream_attempts=0,
                                  historical_verdict='PILOT_STOP'))
        derived[name] = {}
        for ds in sorted({r['dataset'] for r in rows}):
            rr = [r for r in rows if r['dataset']==ds]
            means = {a:mean(r['metrics']['scaled_2pinball'] for r in rr if r['arm']==a)
                     for a in {r['arm'] for r in rr}}
            derived[name][ds] = means
            # Explicit pairwise rows; no cross-dataset or cross-study pooling.
            for baseline in sorted(means.keys()-{proposed,'query_initial'}):
                p,b,f0 = means[proposed],means[baseline],means['F0']
                comparisons.append(dict(experiment=name,dataset=ds,proposed=proposed,
                    baseline=baseline,proposed_loss=p,baseline_loss=b,
                    gain_baseline_percent=gain(p,b),gain_f0_percent=100*(b-p)/f0,
                    scope='pairwise_seed_means_not_original_gate'))
            uniform = 'prediction_anchor' if 'prediction_anchor' in means else 'full_anchor'
            if uniform in means:
                for base in ('native','raw'):
                    comparisons.append(dict(experiment=name,dataset=ds,proposed=uniform,
                        baseline=base,proposed_loss=means[uniform],baseline_loss=means[base],
                        gain_baseline_percent=gain(means[uniform],means[base]),
                        gain_f0_percent=100*(means[base]-means[uniform])/means['F0'],
                        scope='uniform_anchor_followup_comparison'))

    distill = read('results/overnight_20260913/distill/fits.json')
    assert all(f['arm']=='raw' for f in distill)
    inventory.append(dict(experiment='overnight_20260913/distill',fits=len(distill),
                          stream_attempts=0,historical_verdict='STOP_NO_TEACHER_HEADROOM'))
    for name in ('overnight_20260913','calibration_anchor_20260914'):
        v = read(f'results/{name}/verification.json')
        for topic in v['topics']:
            if not topic.get('checks'):
                continue
            for threshold in (1., .5, 0.):
                flags = [c['gain_vs_best']*100 >= threshold for c in topic['checks']]
                seeds_ok = all(all(c['seed_checks']) for c in topic['checks'])
                sensitivity.append(dict(experiment=name+'/'+topic['topic'],
                    hypothetical_min_gain_percent=threshold,
                    all_dataset_means_meet_threshold=all(flags),
                    unchanged_seed_checks=seeds_ok,
                    combined_counterfactual=all(flags) and seeds_ok,
                    scope='retrospective_sensitivity_only_not_a_new_PASS'))
    maturity = read('results/candidate_05_repaired/origin_losses.json')
    maturity_means = {label:{arm:mean(losses[span]) for arm,losses in maturity.items()}
                     for label,span in [('all',slice(None)),('first25',slice(0,25)),('last5',slice(25,30))]}
    total_fits = sum(r['fits'] for r in inventory)
    total_streams = sum(r['stream_attempts'] for r in inventory)
    assert total_fits == 270 and total_streams == 9
    assert len(distill)==8  # All distillation fits were controls, no student training.
    assert all(sha(ROOT/name)==value for name,value in historical.items())
    csv_write('comparisons.csv',comparisons)
    csv_write('fit_inventory.csv',inventory)
    csv_write('validation_endpoints.csv',endpoints)
    csv_write('threshold_sensitivity.csv',sensitivity)
    receipt=dict(status='VERIFIED_RETROSPECTIVE_ARITHMETIC',
        input_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        total_completed_fits=total_fits,total_stream_attempts=total_streams,
        completed_streams=8,historical_aborted_streams=1,new_fits=0,new_optimizer_updates=0,
        historical_result_files_unchanged=len(historical),source_hashes=sources,
        group_means=derived,maturity_chronological_means=maturity_means,
        limitations=['No new selection or PASS verdict.','All E observations are development evidence.',
                     'Pairwise seed means differ from per-seed E-oracle gates.',
                     'Prediction and gradient replay is recorded separately in verification_commands.json.'])
    (OUT/'audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    (OUT/'historical_result_hashes.json').write_text(json.dumps(historical,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','total_completed_fits','total_stream_attempts',
                      'new_fits','historical_result_files_unchanged']}))


if __name__ == '__main__':
    main()
