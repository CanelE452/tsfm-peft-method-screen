from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("triage_data_module", DATA_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_schema_selection_and_hashes():
    data = _load_module()
    packet = data.load()
    assert set(packet) == {"values", "sigma", "ids", "origins"}
    values = packet["values"]
    sigma = packet["sigma"]
    ids = packet["ids"]
    origins = packet["origins"]
    assert values.dtype == np.float32
    assert values.ndim == 2
    assert values.shape[1] == data.N_SERIES
    assert sigma.dtype == np.float64
    assert sigma.shape == (data.N_SERIES,)
    assert len(ids) == data.N_SERIES
    assert list(origins) == list(data.ROLE_KEYS)
    assert np.isfinite(values).all()
    assert np.isfinite(sigma).all()
    assert np.all(sigma > 0)

    audit = json.loads(data.AUDIT_PATH.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["schema"]["values_shape"] == list(values.shape)
    assert audit["hashes"]["values_float32_hash"] == data._array_sha256(values)
    assert audit["hashes"]["sigma_float64_hash"] == data._array_sha256(sigma)

    raw_cols = audit["schema"]["raw_shape"][1]
    selected = np.array(audit["selection"]["selected_original_columns"], dtype=np.int64)
    ranked = sorted(
        (
            data.hashlib.sha256(f"{data.SELECT_SALT}{col}".encode("utf-8")).hexdigest(),
            col,
        )
        for col in range(raw_cols)
    )
    expected_order_for_selected_set = np.array(
        [col for _, col in ranked if col in set(selected.tolist())],
        dtype=np.int64,
    )
    assert np.array_equal(selected, expected_order_for_selected_set)


def test_origin_boundaries_and_t_bridge_isolation():
    data = _load_module()
    packet = data.load()
    origins = packet["origins"]
    n_rows = packet["values"].shape[0]
    split = data._splits(n_rows)
    specs = data._role_spec(split)

    for role in data.ROLE_KEYS:
        arr = origins[role]
        spec = specs[role]
        assert arr.dtype == np.int64
        assert arr.size > 0
        assert np.all(arr >= int(spec["target_start"]))
        assert np.all(arr + data.HORIZON <= int(spec["target_end"]))
        assert np.all(arr - data.CONTEXT >= int(spec["context_min_start"]))
        if spec["mod6"]:
            assert np.all(arr % 6 == 0)
        expected = data._legal_origins(spec)
        assert np.array_equal(arr, expected)

    assert np.min(origins["T_BRIDGE_TRAIN"] - data.CONTEXT) >= split["t_bridge_start"]
    assert np.min(origins["T_BRIDGE_VAL"] - data.CONTEXT) >= split["t_bridge_start"]
    assert np.min(origins["T_TEST"] - data.CONTEXT) >= split["t_bridge_start"]
    assert np.min(origins["Q_VAL"] - data.CONTEXT) < split["q_val_start"]
    assert np.min(origins["Q_TEST"] - data.CONTEXT) < split["q_test_start"]


def test_schedule_global_and_f_client_determinism():
    data = _load_module()
    packet = data.load()
    q_train = packet["origins"]["Q_TRAIN"]

    a = data.schedule("Q_TRAIN", 92201)
    b = data.schedule("Q_TRAIN", 92201)
    c = data.schedule("Q_TRAIN", 92202)
    assert a.shape == (256, 8, 2)
    assert a.dtype == np.int64
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all((0 <= a[..., 0]) & (a[..., 0] < data.N_SERIES))
    assert set(np.unique(a[..., 1])).issubset(set(q_train.tolist()))

    client = data.schedule("Q_TRAIN", 92201, client=2)
    client_explicit = data.schedule("Q_TRAIN", 92201, steps=64, client=2)
    assert client.shape == (64, 8, 2)
    assert np.array_equal(client, client_explicit)
    assert np.all(client[..., 0] == 2)
    assert set(np.unique(client[..., 1])).issubset(set(q_train.tolist()))


def test_probe_is_train_only_and_deterministic():
    data = _load_module()
    packet = data.load()
    q_train = packet["origins"]["Q_TRAIN"]
    first = data.probe()
    second = data.probe()
    assert first.shape == (32, 2)
    assert first.dtype == np.int64
    assert np.array_equal(first, second)
    assert np.all((0 <= first[:, 0]) & (first[:, 0] < data.N_SERIES))
    assert set(np.unique(first[:, 1])).issubset(set(q_train.tolist()))


def test_audit_files_are_data_only_and_complete():
    data = _load_module()
    data._seal_default_artifacts()
    audit = json.loads(data.AUDIT_PATH.read_text(encoding="utf-8"))
    assert audit["source"]["url"] == data.RAW_URL
    assert audit["source"]["upstream_revision"]
    assert len(audit["source"]["upstream_revision"]) == 40
    assert audit["finite"]["raw_all_finite"] is True
    assert audit["finite"]["selected_all_finite"] is True
    assert audit["std"]["sigma_train50_only"] is True
    assert "canonical_schedules_hash" in audit["hashes"]
    assert data.PACKET_PATH.exists()
    assert data.VALUES_PATH.exists()
    assert data.SIGMA_PATH.exists()
    assert data.ORIGINS_PATH.exists()
    assert data.PROBE_PATH.exists()
