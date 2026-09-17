"""Read-only-data, CPU model audit for the outlier/signal execution contract.

This is not the training runner and does not replace the missing supplied tests.
"""
import csv
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
NAME = 'outlier_signal_peft_v1_20260917'
OUT = ROOT / 'results' / NAME
CACHE = ROOT / '.cache' / NAME
FILES = {
    'electricity': ('data/raw/electricity.txt.gz', '3c4c069588198c1fcc95cace7bb69c99922129edfd673b7286661dad20badefa', 24),
    'ettm1': ('data/raw/overnight_20260913/ETTm1.csv', '6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e', 96),
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save(name, value):
    p = OUT / name
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csvwrite(name, rows):
    with open(OUT / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def select_origins(legal, period, count, seed):
    legal = np.asarray(legal, dtype=np.int64)
    days, counts = np.unique(legal // period, return_counts=True)
    full = days[counts == period]
    if len(full) < count:
        raise ValueError(f'INSUFFICIENT_FULL_DAYS: {len(full)} < {count}')
    chosen = full[np.floor((np.arange(count) + .5) * len(full) / count).astype(int)]
    phases = np.floor(np.arange(count) * period / count).astype(int)
    np.random.default_rng(seed).shuffle(phases)
    origins = chosen * period + phases
    assert len(np.unique(chosen)) == count and np.isin(origins, legal).all()
    hist = np.bincount(phases, minlength=period)
    assert hist.max() - hist.min() <= 1
    return origins, full


def data_audit():
    rows, manifest, audits = [], {}, {}
    for source_id, (source, (file, expected, period)) in enumerate(FILES.items()):
        path = ROOT / file
        assert sha(path) == expected, ('RAW_HASH_MISMATCH', file)
        if source == 'electricity':
            frame = pd.read_csv(path, header=None)
            grid = dict(type='index-day', timestamp_available=False, timezone=None)
            times = None
        else:
            frame = pd.read_csv(path)
            times = pd.to_datetime(frame.pop('date'))
            assert times.is_monotonic_increasing and not times.duplicated().any()
            assert (times.diff().dropna() == pd.Timedelta(minutes=15)).all()
            assert times.iloc[0].hour == 0 and times.iloc[0].minute == 0
            grid = dict(type='timestamp-naive', timestamp_available=True, timezone=None,
                        first=str(times.iloc[0]), last=str(times.iloc[-1]), step_minutes=15,
                        monotonic=True, duplicates=0, regular=True)
        n = len(frame)
        bounds = [0, int(.6*n), int(.8*n), n]
        columns = [c for c in frame.select_dtypes(include=np.number).columns
                   if np.isfinite(frame[c].iloc[:bounds[1]]).all()
                   and frame[c].iloc[:bounds[1]].std(ddof=0) > 1e-6][:4]
        assert len(columns) == 4
        a = frame[columns].to_numpy(dtype=np.float64)
        sigma = a[:bounds[1]].std(axis=0, ddof=0)
        # Future is inspected here ONLY for finiteness, not magnitudes or scores.
        invalid_prefix = np.r_[0, np.cumsum(~np.isfinite(a).all(axis=1))]
        audits[source] = {}
        packets = {}
        for role_id, (role, count) in enumerate([('TRAIN', 256), ('V_SELECT', 64), ('E_DISCOVERY', 128)]):
            lo, hi = bounds[role_id:role_id+2]
            candidates = np.arange(max(512, lo), hi-64+1)
            legal = candidates[invalid_prefix[candidates+64] == invalid_prefix[candidates-512]]
            seed = 81700 + 10*source_id + role_id
            origins, full = select_origins(legal, period, count, seed)
            days = origins // period
            phases = origins % period
            target_indices = (origins[:, None] + np.arange(64)).flatten()
            _, multiplicities = np.unique(target_indices, return_counts=True)
            mult, freq = np.unique(multiplicities, return_counts=True)
            audits[source][role] = dict(
                eligible_origins=len(legal), full_eligible_days=len(full),
                required_days=count, distinct_days=len(np.unique(days)),
                distinct_weeks=len(np.unique(days//7)), phase_seed=seed,
                phase_histogram=np.bincount(phases, minlength=period).tolist(),
                date_decile_counts=np.histogram(days, bins=np.linspace(full[0], full[-1]+1, 11))[0].tolist(),
                first_origin=int(origins.min()), last_origin=int(origins.max()),
                origin_span_slots=int(np.ptp(origins)), day_span=int(np.ptp(days)+1),
                unique_target_slots=len(multiplicities), total_target_slots=len(target_indices),
                target_overlap_histogram={str(m): int(f) for m, f in zip(mult, freq)},
                day_phase_correlation=float(np.corrcoef(days, phases)[0, 1]),
            )
            for origin in origins:
                rows.append(dict(source=source, role=role, origin=int(origin),
                                 day=int(origin//period), phase=int(origin%period),
                                 timestamp=str(times.iloc[origin]) if times is not None else '',
                                 target_end_exclusive=int(origin+64)))
            folder = CACHE / 'data' / source
            folder.mkdir(parents=True, exist_ok=True)
            inp = folder / f'{role}_inputs.npz'
            lab = folder / f'{role}_labels.npz'
            np.savez_compressed(inp, x=np.stack([a[o-512:o].T for o in origins]).astype(np.float32),
                                origins=origins, sigma=sigma)
            np.savez_compressed(lab, y=np.stack([a[o:o+64].T for o in origins]))
            packets[role] = {str(p.relative_to(ROOT)): sha(p) for p in [inp, lab]}
        manifest[source] = dict(path=file, sha256=expected, rows=n, original_columns=len(frame.columns),
                                selected_columns=[str(c) for c in columns], bounds=bounds, grid=grid,
                                period=period, sigma_train_population=sigma.tolist(), packet_hashes=packets,
                                evaluation_role='reused development source; not a new independent test')
    csvwrite('origins.csv', rows)
    save('DATA_MANIFEST.json', manifest)
    save('origin_audit.json', audits)
    return manifest


def model_audit():
    from chronos import ChronosBoltPipeline
    import chronos.chronos_bolt as bolt_module
    from .model import attach_lora, ForecastModel, state_hash
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    downloads = json.loads((OUT/'download_receipts.json').read_text())
    for receipt in downloads.values():
        for file, expected in receipt['files'].items():
            assert sha(ROOT/file) == expected
    snapshot = ROOT / downloads['amazon/chronos-bolt-small']['snapshot']
    cfg = json.loads((snapshot/'config.json').read_text())
    expected = dict(context_length=2048, input_patch_size=16, input_patch_stride=16,
                    prediction_length=64, quantiles=[i/10 for i in range(1, 10)], use_reg_token=True)
    assert cfg['chronos_config'] == expected, 'BLOCKED_MODEL_CONTRACT'
    for k,v in dict(d_model=512, num_layers=6, num_decoder_layers=6, num_heads=8, d_kv=64).items():
        assert cfg[k] == v, ('BLOCKED_MODEL_CONTRACT', k)
    base = ChronosBoltPipeline.from_pretrained(str(snapshot), device_map='cpu', torch_dtype=torch.float32).model
    base.eval().requires_grad_(False)
    for m in base.modules():
        if isinstance(m, torch.nn.Dropout): m.p = 0
        elif isinstance(getattr(m, 'dropout', None), float): m.dropout = 0.
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in base.modules())
    data = np.load(CACHE/'data/electricity/TRAIN_inputs.npz')
    x = torch.from_numpy(data['x'][:2].reshape(8, 512))
    sigma = torch.tensor(np.tile(data['sigma'], 2), dtype=torch.float32)
    original_hash = state_hash(dict(base.named_parameters()))
    with torch.no_grad(): native = base(context=x).quantile_preds
    modules = attach_lora(base, 81500)
    models = {arm: ForecastModel(base, arm, 81500) for arm in ['A0','A2','A4','A5']}
    counts = {arm: sum(p.numel() for p in model.parameters() if p.requires_grad)
              for arm, model in models.items()}
    assert counts == {'A0':294912, 'A2':294912, 'A4':299784, 'A5':299784}
    checks = {}
    with torch.no_grad():
        wrapped = models['A0'](x, sigma)
        normalized = float(((native-wrapped)/sigma[:,None,None]).abs().max())
        assert normalized <= 1e-5 or torch.allclose(native, wrapped, rtol=1e-4, atol=0)
        checks['native_wrapper_max_normalized_error'] = normalized
        clip = models['A2'](x, sigma)
        for arm in ['A4','A5']:
            pred = models[arm](x, sigma)
            diff = float(((clip-pred)/sigma[:,None,None]).abs().max())
            assert diff <= 1e-5
            checks[arm+'_step0_max_normalized_error'] = diff
        assert state_hash(dict(models['A4'].adapter.named_parameters())) == state_hash(dict(models['A5'].adapter.named_parameters()))
        checks['generic_residual_initial_parameter_hash_equal'] = True
        assert native.shape == (8,9,64) and torch.isfinite(native).all()
        checks['finite_native_output_shape'] = list(native.shape)
    # The original weight names are mapped back through the LoRA wrappers.
    frozen = {n.replace('.base.', '.'):p for n,p in base.named_parameters() if not p.requires_grad}
    assert state_hash(frozen) == original_hash
    checks['original_base_and_head_hash_unchanged'] = True
    checks['original_base_parameter_sha256'] = original_hash
    checks['optimizer_updates'] = 0
    checks['scope'] = 'CPU actual pretrained forward and initial LoRA/adapter attachment only; not smoke training'
    checks['supplied_reference_tests'] = 'BLOCKED_MISSING_REFERENCE_FILES'
    checks['unverified'] = ['supplied reference parity', 'two-update gradients', 'GPU microbatch',
                            'checkpoint resume', 'full generators and label streams', 'independent metric verification']
    save('implementation_checks.json', checks)
    save('PARAMETER_RECEIPT.json', dict(modules=modules, trainable_counts=counts,
                                       expected_A1_A3=294912, actual_A1_A3_construction_not_yet_tested=True))
    save('MODEL_RECEIPT.json', dict(config=cfg, snapshot=downloads['amazon/chronos-bolt-small'],
                                   device_of_this_check='cpu', dtype='float32', TF32=False, dropout=0,
                                   installed_source=str(Path(bolt_module.__file__).relative_to(ROOT)),
                                   installed_source_sha256=sha(bolt_module.__file__),
                                   versions={p:importlib.metadata.version(p) for p in
                                             ['torch','transformers','chronos-forecasting','peft','huggingface-hub']}))


def main():
    if (OUT/'DATA_MANIFEST.json').exists():
        manifest = json.loads((OUT/'DATA_MANIFEST.json').read_text())
        for info in manifest.values():
            assert sha(ROOT/info['path']) == info['sha256']
            for files in info['packet_hashes'].values():
                for path, expected in files.items():
                    assert sha(ROOT/path) == expected, ('EXISTING_PACKET_CHANGED',path)
        print('Existing prepared packets verified and reused', flush=True)
    else:
        manifest = data_audit()
    print('DATA_AUDIT', {s:{r:v['distinct_days'] for r,v in a.items()}
                         for s,a in json.loads((OUT/'origin_audit.json').read_text()).items()}, flush=True)
    model_audit()
    print('CPU_NATIVE_MODEL_AUDIT_COMPLETE; supplied reference tests and learning remain unexecuted', flush=True)


if __name__ == '__main__': main()
