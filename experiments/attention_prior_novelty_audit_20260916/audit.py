"""Zero-fit algebra and existing-validation-trajectory audit; no model loading."""
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'results/channel_attention_prior_20260916'
OUT = ROOT / 'research/attention_prior_novelty_audit_20260916'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def csvsave(p, rows):
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(61701)
    assert not (OUT / 'verification.json').exists(), 'Completed audit already exists'
    history = {str(p.relative_to(ROOT)): sha(p)
               for folder in ['results', 'research']
               for p in (ROOT / folder).rglob('*')
               if p.is_file() and OUT not in p.parents}
    save(OUT / 'historical_hashes.json', history)
    old_seal = read(OLD / 'seal.json')
    for p, h in old_seal['source_hashes'].items():
        assert sha(ROOT / p) == h
    assert read(OLD / 'status.json')['status'] == 'COMPLETE'
    assert read(OLD / 'status.json')['training_updates'] == 9143

    q0 = torch.randn(2, 5, 4, dtype=torch.float64)
    k0 = torch.randn(2, 5, 4, dtype=torch.float64)
    q = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 5, 2, dtype=torch.float64, requires_grad=True)
    a0 = q0 @ k0.transpose(-1, -2) / math.sqrt(4)
    delta = q @ k.transpose(-1, -2) / math.sqrt(3)
    direct = (a0 + delta).softmax(-1)
    prior = (a0.log_softmax(-1) + delta).softmax(-1)
    qcat = torch.cat((q0 / 4**.25, q / 3**.25), dim=-1)
    kcat = torch.cat((k0 / 4**.25, k / 3**.25), dim=-1)
    concatenated = (qcat @ kcat.transpose(-1, -2)).softmax(-1)
    errors = [float((prior - direct).abs().max().detach()),
              float((concatenated - direct).abs().max().detach())]
    grads = [torch.autograd.grad((p @ v).square().sum(), (q, k, v), retain_graph=True)
             for p in (direct, prior, concatenated)]
    grad_errors = [max(float((x-y).abs().max()) for x, y in zip(grads[0], g))
                   for g in grads[1:]]
    assert max(errors + grad_errors) < 1e-10

    logits = torch.tensor([[[0., 0.], [0., 0.]],
                           [[10., 0.], [10., 0.]]], dtype=torch.float64)
    extra = torch.tensor([[.2, -.1], [.2, -.1]], dtype=torch.float64)
    mix = logits.softmax(-1).mean(0)
    via_mix = (mix.log() + extra).softmax(-1)
    via_logit_mean = (logits.mean(0) + extra).softmax(-1)
    non_equivalence = float((via_mix - via_logit_mean).abs().max())
    assert non_equivalence > .1
    algebra = dict(
        seed=61701, dtype='float64', tolerance=1e-10,
        single_head_forward_errors=errors, single_head_gradient_errors=grad_errors,
        probability_mean_vs_logit_mean_counterexample_error=non_equivalence,
        scope='Operator identities only; not an equivalence proof of complete LiSA and PRIOR models.')
    save(OUT / 'algebra.json', algebra)

    trajectory = read(OLD / 'trajectory.json')
    fits = read(OLD / 'fits.json')
    rows, summaries = [], []
    for source in ['electricity', 'traffic']:
        for seed in [41000, 41001]:
            arm_rows = {a: {r['epoch']: r for r in trajectory
                           if (r['dataset'], r['seed'], r['arm']) == (source, seed, a)}
                        for a in ['SIDE', 'PRIOR']}
            shared = sorted(set(arm_rows['SIDE']) & set(arm_rows['PRIOR']))
            assert shared == list(range(shared[-1] + 1))
            for epoch in shared:
                side, prior_row = [arm_rows[a][epoch] for a in ['SIDE', 'PRIOR']]
                assert side['updates'] == prior_row['updates']
                rows.append(dict(
                    dataset=source, seed=seed, epoch=epoch, updates=side['updates'],
                    SIDE_V_mse=side['metrics']['mse'], PRIOR_V_mse=prior_row['metrics']['mse'],
                    PRIOR_gain_percent=100*(side['metrics']['mse']-prior_row['metrics']['mse'])/side['metrics']['mse']))
            minima = {a: min((arm_rows[a][e] for e in shared),
                            key=lambda r: (r['metrics']['mse'], r['epoch']))
                      for a in ['SIDE', 'PRIOR']}
            selected = {a: next(f['best'] for f in fits
                               if (f['dataset'], f['seed'], f['arm']) == (source, seed, a))
                        for a in ['SIDE', 'PRIOR']}
            end = rows[-1]
            source_rows = [r for r in rows if (r['dataset'], r['seed']) == (source, seed) and r['epoch'] > 0]
            summaries.append(dict(
                dataset=source, seed=seed, common_last_epoch=shared[-1],
                shared_trained_epochs=len(source_rows),
                prior_better_epochs=sum(r['PRIOR_gain_percent'] > 0 for r in source_rows),
                common_last_SIDE_V_mse=end['SIDE_V_mse'],
                common_last_PRIOR_V_mse=end['PRIOR_V_mse'],
                common_last_gain_percent=end['PRIOR_gain_percent'],
                common_range_SIDE_best_epoch=minima['SIDE']['epoch'],
                common_range_PRIOR_best_epoch=minima['PRIOR']['epoch'],
                common_range_SIDE_best_V=minima['SIDE']['metrics']['mse'],
                common_range_PRIOR_best_V=minima['PRIOR']['metrics']['mse'],
                common_range_best_gain_percent=100*(minima['SIDE']['metrics']['mse']-minima['PRIOR']['metrics']['mse'])/minima['SIDE']['metrics']['mse'],
                original_SIDE_selected_epoch=selected['SIDE']['epoch'],
                original_PRIOR_selected_epoch=selected['PRIOR']['epoch'],
                original_PRIOR_selected_V=selected['PRIOR']['metrics']['mse']))
    csvsave(OUT / 'all_common_validation_epochs.csv', rows)
    csvsave(OUT / 'validation_summary.csv', summaries)

    lisa = read(Path('/tmp/attention_prior_review/source.json'))
    assert sha(Path('/tmp/attention_prior_review/modeling_llama_256.py')) == lisa['sha256']
    save(OUT / 'lisa_source.json', lisa)
    inputs = [OLD / n for n in ['seal.json', 'status.json', 'trajectory.json', 'fits.json',
                                'REPORT.md', 'INTERPRETATION.md', 'decision.json']]
    inputs += [Path(__file__), OUT / 'SCOPE.md']
    for p, h in history.items():
        assert sha(ROOT / p) == h, p
    verification = dict(
        baseline_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        input_hashes={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        historical_files_preserved=len(history), source_hashes_unchanged=True,
        new_fits=0, optimizer_updates=0, model_forwards=0,
        new_predictions=0, common_validation_records=len(rows),
        source_seed_pairs=len(summaries), all_common_epochs_disclosed=True,
        E_reselection_or_rescoring=False, GPU_used=False,
        note='Synthetic CPU algebra plus stored V metrics; no independent predictive PASS.')
    save(OUT / 'verification.json', verification)
    print(json.dumps(dict(algebra=algebra, validation=summaries, verification=verification), ensure_ascii=False))


if __name__ == '__main__':
    main()
