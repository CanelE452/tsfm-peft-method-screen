from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BRIDGE_PATH = HERE / "bridge_data.py"
DATA_PATH = HERE / "data.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_writes_independent_bridge_packets():
    bridge = _load_module(BRIDGE_PATH, "bridge_data_for_build_test")
    data = _load_module(DATA_PATH, "full_data_for_bridge_build_test")
    audit = bridge.build()
    full = data.load()

    assert audit["status"] == "PASS"
    split = json.loads(data.AUDIT_PATH.read_text(encoding="utf-8"))["split"]
    offset = int(split["t_old_val_end"])
    bridge_end = int(split["t_bridge_end"])
    expected_values = full["values"][offset:bridge_end]

    for seed in bridge.DEFAULT_SEEDS:
        packet_path = bridge.CACHE_DIR / f"t_bridge_seed{seed}.npz"
        manifest_path = bridge.CACHE_DIR / f"t_bridge_seed{seed}.json"
        assert packet_path.exists()
        assert manifest_path.exists()
        with np.load(packet_path, allow_pickle=False) as packet:
            assert np.array_equal(packet["values"], expected_values)
            assert np.array_equal(packet["sigma"], full["sigma"])
            assert packet["offset"].shape == ()
            assert int(packet["offset"]) == offset
            assert np.array_equal(packet["origins"], full["origins"]["T_BRIDGE_VAL"])
            assert packet["schedule"].shape == (256, 8, 2)
            assert [str(x) for x in packet["ids"].tolist()] == full["ids"]


def test_load_bridge_reads_only_independent_npz(monkeypatch):
    bridge = _load_module(BRIDGE_PATH, "bridge_data_for_file_audit_test")
    bridge.build()
    allowed = bridge.CACHE_DIR / "t_bridge_seed92201.npz"
    opened_by_np_load: list[str] = []
    original_np_load = bridge.np.load
    original_open = builtins.open

    blocked_fragments = (
        "electricity_selected_packet.npz",
        "values_float32.npy",
        "origins.npz",
        "sigma_float64.npy",
        "electricity.txt.gz",
        "data.py",
    )

    def checked_np_load(path, *args, **kwargs):
        path_text = str(path)
        opened_by_np_load.append(path_text)
        assert Path(path_text) == allowed
        return original_np_load(path, *args, **kwargs)

    def checked_open(file, *args, **kwargs):
        file_text = str(file)
        if any(fragment in file_text for fragment in blocked_fragments):
            raise AssertionError(f"load_bridge opened forbidden full-data path: {file_text}")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(bridge.np, "load", checked_np_load)
    monkeypatch.setattr(builtins, "open", checked_open)
    packet = bridge.load_bridge(92201)

    assert opened_by_np_load == [str(allowed)]
    assert set(packet) == {"values", "sigma", "offset", "origins", "schedule", "ids"}
    assert packet["values"].dtype == np.float32
    assert packet["sigma"].dtype == np.float64
    assert packet["origins"].dtype == np.int64
    assert packet["schedule"].dtype == np.int64
    assert len(packet["ids"]) == packet["values"].shape[1]


def test_student_train_and_validation_access_bounds():
    bridge = _load_module(BRIDGE_PATH, "bridge_data_for_bounds_test")
    bridge.build()
    for seed in bridge.DEFAULT_SEEDS:
        packet = bridge.load_bridge(seed)
        offset = int(packet["offset"])
        bridge_end = offset + packet["values"].shape[0]
        bridge_train_end = offset + int(packet["values"].shape[0] * 0.75)
        schedule = packet["schedule"]
        train_origins = schedule[..., 1]
        val_origins = packet["origins"]

        assert schedule.shape == (256, 8, 2)
        assert np.all((0 <= schedule[..., 0]) & (schedule[..., 0] < packet["values"].shape[1]))
        assert np.all(train_origins - bridge.CONTEXT >= offset)
        assert np.all(train_origins >= offset)
        assert np.all(train_origins + bridge.HORIZON <= bridge_train_end)

        assert np.all(val_origins % 6 == 0)
        assert np.all(val_origins >= bridge_train_end)
        assert np.all(val_origins + bridge.HORIZON <= bridge_end)
        assert np.all(val_origins - bridge.CONTEXT >= offset)
