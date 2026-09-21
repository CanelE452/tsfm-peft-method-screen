import numpy as np

import data


def test_role_origins_keep_future_inside_role_and_daily_phase():
    T = 17420
    origins = data.role_origins(T)
    edges = data._role_edges(T)
    for role, role_origins in origins.items():
        lo, hi = edges[role]["start"], edges[role]["end"]
        assert len(role_origins) > 0
        assert np.all(role_origins % 24 == 0)
        assert np.all(role_origins >= max(lo, data.CONTEXT))
        assert np.all(role_origins + data.HORIZON <= hi)
        assert np.all(np.diff(role_origins) == 24)


def test_electricity_selection_hash_order_is_performance_blind():
    matrix = np.arange(1000 * 40, dtype=float).reshape(1000, 40)
    selected, audit = data._select_electricity_columns(matrix, 600)
    expected = sorted(range(40), key=lambda i: data.hashlib.sha256(f"rollout-v1|ECL|{i}".encode("utf-8")).hexdigest())[:32]
    assert selected == expected
    assert "zero-rate/performance" in audit["selection_rule"]


def test_schedule_shape_and_role_membership(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE", tmp_path / ".cache" / data.CACHE.name)
    monkeypatch.setattr(data, "RESULTS", tmp_path / "results" / data.RESULTS.name)
    ddir = data._data_dir()
    values = np.ones((3000, 3), dtype=np.float32)
    sigma = np.ones(3, dtype=np.float64)
    origins = data.role_origins(len(values))
    np.savez_compressed(ddir / "ETTh1.npz", values=values, sigma=sigma, ids=np.array(["a", "b", "c"], dtype=object), **{f"origins_{role}": origins[role] for role in data.ROLES})
    packet = data.schedule("ETTh1", 92120)
    assert packet.shape == (512, 8, 2)
    assert packet[..., 0].min() >= 0 and packet[..., 0].max() < 3
    assert set(np.unique(packet[..., 1])).issubset(set(origins["TRAIN"].tolist()))
    np.testing.assert_array_equal(packet, data.schedule("ETTh1", 92120))
