"""Post-run publication checks; no training or scoring-policy changes."""
import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/channel_capacity_match_20260916'
EXP = ROOT / 'experiments/channel_capacity_match_20260916'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    seal = read(OUT / 'seal.json')
    history = read(OUT / 'historical_hashes.json')
    for mapping in (history, seal['source_hashes']):
        for path, expected in mapping.items():
            assert sha(ROOT / path) == expected, path
    fits = read(OUT / 'fits.json')
    trajectory = read(OUT / 'trajectory.json')
    assert len(fits) == 4
    for fit in fits:
        rows = [r for r in trajectory if r['fit'] == fit['fit']]
        best, bad = rows[0]['metrics']['mse'], 0
        for row in rows[1:]:
            if row['metrics']['mse'] < best - 1e-4:
                best, bad = row['metrics']['mse'], 0
            else:
                bad += 1
            assert bad < 5 or row == rows[-1]
        assert fit['early_stopped'] == (bad >= 5)
        assert fit['epochs'] == 20 or bad >= 5
        changes = read(OUT / (fit['fit'] + '_changes.json'))
        assert all(math.isfinite(v) and v > 0 for v in changes.values())
        for step in read(OUT / (fit['fit'] + '_steps.json')):
            assert step['lr'] == .001 * .5 ** ((step['epoch'] - 1) // 5)
            assert all(math.isfinite(step[k]) for k in ('loss', 'gradient_norm', 'seconds'))
            assert not step['external_compute_contaminated']
    completion = read(OUT / 'completion_audit.json')
    assert sum(f['updates'] for f in fits) == completion['training_updates'] == 4632
    assert completion['unused_update_cap'] == seal['update_cap'] - 4632 == 448
    for smoke in read(OUT / 'smoke.json'):
        assert smoke['initial_bf16_LH_exact'] and smoke['frozen_and_buffers_unchanged']
        assert len(smoke['steps']) == 2
        assert all(math.isfinite(v) and v > 0 for v in smoke['changes'].values())
    records = list(csv.DictReader((OUT / 'comparisons.csv').open()))
    savings = [100 * (1 - float(r['peak_mib']) / float(r['baseline_peak_mib']))
               for r in records if r['baseline'] == 'BALANCED_LH' and r['role'] == 'selected']
    assert savings == completion['peak_allocated_saving_percent_vs_LH']
    files = sorted(list(EXP.glob('*.py')) + [p for p in OUT.iterdir()
                   if p.is_file() and p.name != 'publication_audit.json']
                   + [ROOT / 'docs/RESULTS_INDEX.md'])
    links = 0
    for path in files:
        assert path.suffix in {'.py', '.json', '.jsonl', '.md', '.csv'}
        assert b'\r\n' not in path.read_bytes(), path
        if path.suffix == '.py':
            ast.parse(path.read_text())
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
                if target.startswith(('https://', 'http://', '#')):
                    continue
                assert (path.parent / target.split('#')[0]).exists(), (path, target)
                links += 1
    verification = read(OUT / 'verification.json')
    assert sha(EXP / 'report.py') == verification['report_source_sha256']
    audit = dict(files={str(p.relative_to(ROOT)): sha(p) for p in files},
                 historical_files_preserved=len(history), training_seal_preserved=True,
                 local_markdown_links_verified=links, policy_checks_replayed=True,
                 total_published_bytes=sum(p.stat().st_size for p in files),
                 note='Final documents appended after report generation. Local raw data, weights and prediction arrays excluded. No additional fits.')
    (OUT / 'publication_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'files'}))


if __name__ == '__main__':
    main()
