import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("data.py")
SPEC = importlib.util.spec_from_file_location("branch_mixture_data", MODULE_PATH)
data = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(data)


def _write_toy_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE", tmp_path / ".cache" / data.NAME)
    monkeypatch.setattr(data, "RESULTS", tmp_path / "results" / data.NAME)
    ddir = data._data_dir()
    values = np.arange(3000 * 7, dtype=np.float32).reshape(3000, 7)
    sigma = values[: int(0.60 * len(values))].astype(np.float64).std(axis=0, ddof=0)
    origins = data.role_origins(len(values))
    np.savez_compressed(
        ddir / "ETTh2.npz",
        values=values,
        sigma=sigma,
        ids=np.asarray(data.ETT_COLUMNS[1:], dtype=object),
        **{f"origins_{role}": origins[role] for role in data.ROLES},
    )
    return values, sigma, origins


def test_role_origins_keep_targets_inside_role_and_daily_phase():
    T = 17420
    origins = data.role_origins(T)
    edges = data._role_edges(T)
    for role, role_origins in origins.items():
        lo, hi = edges[role]["start"], edges[role]["end"]
        assert len(role_origins) > 0
        assert np.all(role_origins % data.STRIDE == 0)
        assert np.all(role_origins >= max(lo, data.CONTEXT))
        assert np.all(role_origins + data.HORIZON <= hi)
        assert np.all(np.diff(role_origins) == data.STRIDE)


def test_schedule_shape_train_membership_and_determinism(tmp_path, monkeypatch):
    _, _, origins = _write_toy_dataset(tmp_path, monkeypatch)
    packet = data.schedule(92231)
    assert packet.shape == (512, 4, 2)
    assert packet[..., 0].min() >= 0
    assert packet[..., 0].max() < 7
    assert set(np.unique(packet[..., 1])).issubset(set(origins["TRAIN"].tolist()))
    np.testing.assert_array_equal(packet, data.schedule(92231))


def test_batch_returns_context_target_without_future_leakage(tmp_path, monkeypatch):
    values, sigma, origins = _write_toy_dataset(tmp_path, monkeypatch)
    loaded = data.load()
    pairs = np.asarray([[2, origins["TRAIN"][3]], [6, origins["TRAIN"][7]]], dtype=np.int64)
    x, y, s = data.batch(loaded, pairs)
    assert x.shape == (2, data.CONTEXT)
    assert y.shape == (2, data.HORIZON)
    assert s.shape == (2,)
    for row, (series_idx, origin) in enumerate(pairs):
        np.testing.assert_array_equal(x[row], values[origin - data.CONTEXT : origin, series_idx])
        np.testing.assert_array_equal(y[row], values[origin : origin + data.HORIZON, series_idx])
        assert not np.intersect1d(
            np.arange(origin - data.CONTEXT, origin), np.arange(origin, origin + data.HORIZON)
        ).size
        assert s[row] == sigma[series_idx]


def test_origin_audit_flags_window_finiteness():
    values = np.ones((3000, 7), dtype=np.float32)
    origins = data.role_origins(len(values))
    audit = data._origin_audit(values, origins)
    for role in data.ROLES:
        assert audit[role]["target_windows_inside_role"]
        assert audit["all_windows_finite"][role]
