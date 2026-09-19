"""Quantify a known observational ambiguity; no new prediction or selection.

For identical inputs with two targets a,b and an identical median p,
  (|p-a|+|p-b|)/2 = |a-b|/2 + distance(p,[min(a,b),max(a,b)]).
The label-dependent interval is an audit device, never a predictor input.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
PRIOR = ROOT / 'results/temporal_response_peft_20260919'
CACHE = ROOT / '.cache/additive_persistence_validation_v1_20260917'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def components(p, a, b):
    lower, upper = np.minimum(a, b), np.maximum(a, b)
    floor = (upper - lower) / 2
    excess = np.maximum(lower - p, 0) + np.maximum(p - upper, 0)
    risk = (np.abs(p - a) + np.abs(p - b)) / 2
    return risk, floor, excess


def cpu_checks():
    rng = np.random.default_rng(91931)
    a, b, p = rng.normal(size=(3, 1024))
    risk, floor, excess = components(p, a, b)
    np.testing.assert_allclose(risk, floor + excess, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(components((a+b)/2, a, b)[2], 0, atol=1e-13)
    # Independent scalar calculation, both signs, degenerate and outside cases.
    for aa, bb in [(-3., 7.), (7., -3.), (2., 2.)]:
        for pp in [-10., 0., 2., 5., 12.]:
            rr, ff, ee = components(np.array(pp), np.array(aa), np.array(bb))
            expected = (abs(pp-aa) + abs(pp-bb)) / 2
            assert abs(float(rr)-expected) < 1e-12
            assert abs(float(ff+ee)-expected) < 1e-12
    return {'random_identity_cases': 1024, 'scalar_edge_cases': 15}


def main():
    checks = cpu_checks()
    checked = {}

    def verify(path, expected=None):
        key = str(path.relative_to(ROOT))
        h = sha(path)
        if expected is not None:
            assert h == expected, key
        checked[key] = h
        return h

    marker = json.loads((PRIOR/'ALL_PREDICTIONS_SAVED.json').read_text())
    verify(PRIOR/'PREDICTIONS.json', marker['manifest_sha256'])
    manifest = json.loads((PRIOR/'PREDICTIONS.json').read_text())
    raw = pd.read_csv(PRIOR/'RAW_SCORES.csv')
    verify(PRIOR/'RAW_SCORES.csv')
    rows, panel_rows = [], []
    for panel in ['electricity', 'electricity_transfer', 'ettm1']:
        folder = CACHE/'shapes'/panel
        data = CACHE/'data'/panel
        for path in [folder/'manifest.json', folder/'x.npy', folder/'offset.npy',
                     data/'E_DISCOVERY_inputs.npz', data/'E_DISCOVERY_labels.npz']:
            verify(path)
        meta = json.loads((folder/'manifest.json').read_text())
        ids = np.asarray(meta['origin_indices'])
        inputs = np.load(data/'E_DISCOVERY_inputs.npz')
        y = np.load(data/'E_DISCOVERY_labels.npz')['y'][ids]
        n, nc, horizon = y.shape
        assert horizon == 64 and n == 64
        shape = (len(meta['states']), 2, n, nc)
        x = np.load(folder/'x.npy', mmap_mode='r').reshape(*shape, 512)
        offsets = np.load(folder/'offset.npy').reshape(*shape, 1)
        i = meta['states'].index('PULSE8_D32')
        j = meta['states'].index('PAIRED_SHIFT8_D32')
        assert np.array_equal(x[i], x[j]), panel
        assert np.count_nonzero(offsets[i]) == 0
        assert (offsets[j] != 0).all()
        a, b = y[None, ...] + offsets[i], y[None, ...] + offsets[j]
        sigma = inputs['sigma'][None, None, :, None]
        panel_rows.append(dict(panel=panel, paired_inputs=2*n*nc,
                               identical_input_values=int(x[i].size),
                               input_bitwise_equal=True, target_equal=False,
                               distinct_origins=len(np.unique(inputs['origins'][ids])),
                               normalized_MAE_lower_bound=float((np.abs(a-b)/sigma/2).mean())))
        for key, record in manifest.items():
            if record['panel'] != panel or record['kind'] != 'shape':
                continue
            path = ROOT/record['path']
            if str(path.relative_to(ROOT)) not in checked:
                verify(path, record['sha256'])
            else:
                assert checked[str(path.relative_to(ROOT))] == record['sha256']
            pp = np.load(path, mmap_mode='r').reshape(*shape, 9, 64)
            assert np.array_equal(pp[i], pp[j]), key
            p = pp[i, ..., 4, :].astype(float)
            risk, floor, excess = components(p, a, b)
            err = float(np.max(np.abs(risk-floor-excess)))
            np.testing.assert_allclose(risk, floor+excess, rtol=1e-12, atol=1e-10)
            pulse = float((np.abs(p-a)/sigma).mean())
            shift = float((np.abs(p-b)/sigma).mean())
            rr, ff, ee = [float((v/sigma).mean()) for v in [risk, floor, excess]]
            old = raw[(raw.panel == panel) & (raw.kind == 'shape') &
                      (raw.stage == record['stage']) & (raw.arm == record['arm']) &
                      (raw.seed == record['seed'])]
            assert len(old[old.condition == 'PULSE8_D32']) == 1
            np.testing.assert_allclose([pulse, shift], [
                old[old.condition == 'PULSE8_D32'].nmae.iloc[0],
                old[old.condition == 'PAIRED_SHIFT8_D32'].nmae.iloc[0]],
                rtol=1e-11, atol=1e-11)
            rows.append(dict(panel=panel, arm=record['arm'], seed=record['seed'],
                             stage=record['stage'], step=record['step'],
                             pulse_nmae=pulse, paired_shift_nmae=shift,
                             pair_nmae=rr, oracle_pair_floor_nmae=ff,
                             avoidable_excess_nmae=ee,
                             floor_fraction_pct=100*ff/rr,
                             oracle_joint_improvement_upper_bound_pct=100*ee/rr,
                             prediction_inside_target_interval_fraction=float(((p>=np.minimum(a,b)) & (p<=np.maximum(a,b))).mean()),
                             same_prediction_bitwise=True, max_identity_error=err))
    assert len(rows) == 96
    frame = pd.DataFrame(rows)
    frame.to_csv(DEST/'PAIRED_RISK.csv', index=False)
    numeric = ['pulse_nmae', 'paired_shift_nmae', 'pair_nmae', 'oracle_pair_floor_nmae',
               'avoidable_excess_nmae', 'prediction_inside_target_interval_fraction']
    grouped = frame.groupby(['panel','stage','arm'], as_index=False)[numeric].mean()
    grouped['floor_fraction_pct'] = 100*grouped.oracle_pair_floor_nmae/grouped.pair_nmae
    grouped['oracle_joint_improvement_upper_bound_pct'] = 100*grouped.avoidable_excess_nmae/grouped.pair_nmae
    grouped.to_csv(DEST/'PAIRED_RISK_MEAN.csv', index=False)
    pd.DataFrame(panel_rows).to_csv(DEST/'PAIRED_INPUT_AUDIT.csv', index=False)
    # The ambiguity is a shape stress test, NOT part of TRP's primary family.
    effects = pd.read_csv(PRIOR/'EFFECTS.csv')
    verify(PRIOR/'EFFECTS.csv')
    primary = effects[effects.primary_family]
    assert len(primary) == 4 and (primary.kind == 'standard').all()
    assert set(primary.condition) == {'SHIFT8'}
    audit = dict(status='VERIFIED_EXISTING_PAIR_DIAGNOSTIC',
                 base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 new_fits=0, optimizer_updates=0, new_inference=0,
                 cpu_checks=checks, prediction_views=len(frame),
                 scalar_metric_replays=2*len(frame), source_hashes=checked,
                 code_sha256=sha(Path(__file__)),
                 known_limitation_not_new_discovery=True,
                 affects_TRP_primary_criterion=False,
                 cannot_explain_TRP_primary_failure=True,
                 oracle_interval_used_as_model_input=False,
                 independent_test=False, paper_pass=False)
    (DEST/'PAIR_AUDIT.json').write_text(json.dumps(audit, indent=2, sort_keys=True)+'\n')
    print(grouped[(grouped.panel=='electricity_transfer') & (grouped.stage=='selected')].to_string(index=False))


if __name__ == '__main__':
    main()
