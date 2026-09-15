"""Publication audit; no model execution, training or setting changes."""
import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments/channel_attention_prior_20260916'
OUT = ROOT / 'results/channel_attention_prior_20260916'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    seal = read(OUT / 'seal.json')
    history = read(OUT / 'historical_hashes.json')
    assert sha(OUT / 'historical_hashes.json') == seal['historical_manifest_sha256']
    for mapping in (history, seal['source_hashes'], seal['model_files']):
        for p, h in mapping.items():
            assert sha(ROOT / p) == h, p
    for data in seal['data'].values():
        for p, h in data['staged'].items():
            assert sha(ROOT / p) == h, p

    status = read(OUT / 'status.json')
    fits = read(OUT / 'fits.json')
    assert status['status'] == 'COMPLETE'
    assert len(fits) == status['fit_attempts'] == status['fits_completed'] == 8
    updates = sum(f['updates'] for f in fits)
    assert updates == status['training_updates'] <= seal['update_cap'] == 10160
    assert status['smoke_updates'] == seal['smoke_cap'] == 8
    trajectory = read(OUT / 'trajectory.json')
    for f in fits:
        assert f['status'] == 'COMPLETE'
        rows = [r for r in trajectory if r['fit'] == f['fit']]
        assert [r['epoch'] for r in rows] == list(range(f['epochs'] + 1))
        assert min(rows, key=lambda r: (r['metrics']['mse'], r['epoch'])) == f['best']
        best, bad = rows[0]['metrics']['mse'], 0
        for row in rows[1:]:
            if row['metrics']['mse'] < best - 1e-4:
                best, bad = row['metrics']['mse'], 0
            else:
                bad += 1
            assert bad < 5 or row is rows[-1]
        assert f['early_stopped'] == (bad >= 5)
        assert f['epochs'] == 20 or bad >= 5
        changes = read(OUT / (f['fit'] + '_changes.json'))
        assert all(math.isfinite(v) for v in changes.values())
        for prefix in ['head.'] + [
            f'side.{i}.{part}.' for i in range(8)
            for part in ['q', 'k', 'v', 'up', 'norm']
        ]:
            assert any(v > 0 for n, v in changes.items() if n.startswith(prefix)), prefix
        steps = read(OUT / (f['fit'] + '_steps.json'))
        assert len(steps) == f['updates']
        assert [r['update'] for r in steps] == list(range(1, f['updates'] + 1))
        for r in steps:
            assert r['lr'] == .001 * .5 ** ((r['epoch'] - 1) // 5)
            assert not r['external_compute_contaminated']
            assert all(math.isfinite(r[k]) for k in ['loss', 'gradient_norm', 'seconds'])
        assert f['frozen_and_buffers_unchanged'] and f['replay_max_abs'] == 0
    verify = read(OUT / 'verification.json')
    assert verify['report_source_sha256'] == sha(EXP / 'report.py')
    assert verify['selected_checkpoint_replays_exact'] == 8
    assert verify['unapproved_compute_samples'] == 0
    decision = read(OUT / 'decision.json')
    comps = list(csv.DictReader((OUT / 'comparisons.csv').open()))
    conditions = []
    for d in ['electricity', 'traffic']:
        for baseline in ['SIDE', 'BALANCED_LH']:
            rr = [r for r in comps if r['dataset'] == d
                  and r['arm'] == 'PRIOR' and r['role'] == 'selected'
                  and r['baseline'] == baseline]
            assert len(rr) == 2
            base = sum(float(r['baseline_mse']) for r in rr) / 2
            cand = sum(float(r['mse']) for r in rr) / 2
            gain = 100 * (base - cand) / base
            positive = all(float(r['gain_percent']) > 0 for r in rr)
            rec = next(r for r in decision['component_comparisons']
                       if r['dataset'] == d and r['baseline'] == baseline)
            assert math.isclose(gain, rec['gain_percent'], rel_tol=1e-12, abs_tol=1e-12)
            assert positive == rec['both_seeds_positive']
            conditions.append(gain >= 1 and positive)
    assert decision['component_signal'] == all(conditions)
    assert decision['independent_screen_pass'] == 'NOT_EVALUATED'
    assert decision['new_method_topic'] == 'NOT_CONFIRMED'
    files = sorted(list(EXP.glob('*.py')) + [
        p for p in OUT.iterdir() if p.is_file() and p.name != 'publication_audit.json'
    ] + [ROOT / 'docs/RESULTS_INDEX.md'])
    links = 0
    for p in files:
        assert p.suffix in {'.py', '.json', '.jsonl', '.md', '.csv'}
        assert b'\r\n' not in p.read_bytes(), p
        if p.suffix == '.py':
            ast.parse(p.read_text())
        elif p.suffix == '.json':
            read(p)
        elif p.suffix == '.jsonl':
            for line in p.read_text().splitlines():
                json.loads(line)
        elif p.suffix == '.md':
            assert len(p.read_text().splitlines()) > 1, p
            for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
                if target.startswith(('https://', 'http://', '#')):
                    continue
                assert (p.parent / target.split('#')[0]).exists(), (p, target)
                links += 1
    audit = dict(
        files={str(p.relative_to(ROOT)): sha(p) for p in files},
        training_fits=8, training_updates=updates, smoke_updates=8,
        unused_update_cap=10160-updates,
        historical_files_preserved=len(history),
        training_seal_preserved=True, policy_checks_replayed=True,
        parameter_groups_changed_in_all_fits=True,
        component_decision_recomputed=True, local_markdown_links_verified=links,
        total_published_bytes=sum(p.stat().st_size for p in files),
        note='Reporting scripts and final interpretation added after training seal. '
             'No additional fits. Raw data, weights and prediction arrays remain local.')
    (OUT / 'publication_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'files'}))


if __name__ == '__main__':
    main()
