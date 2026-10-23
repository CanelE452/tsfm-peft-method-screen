#!/usr/bin/env python3
"""Inspect or explicitly run a preserved M5-dependent script after path relocation.

The default only verifies the vendored dependency and prints the path translation.
Original experiment files remain byte-for-byte unchanged. No training is started
unless the caller supplies --execute, which retains the target's original effects.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ORIGINAL_ROOT = "/home/minjae/Documents/github/tsfm-peft-method-screen"
ORIGINAL_M5_ROOT = "/home/minjae/Documents/github/m5dataset"
ORIGINAL_HF_ROOT = "/home/minjae/.cache/huggingface"
SUPPORTED_DIRECTORIES = {
    "intermittent_gonogo_20260922",
    "service_axis_v2_20260922",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_dependency(root: Path) -> dict:
    manifest_path = root / "DEPENDENCY_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported M5 dependency manifest version")
    for record in manifest["files"]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Dependency manifest path escapes its root")
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Dependency size mismatch: {record['path']}")
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Dependency SHA-256 mismatch: {record['path']}")
    return manifest


def translated_source(target: Path, project: Path, m5: Path, hf: Path) -> tuple[bytes, dict]:
    original = target.read_bytes()
    mapping = {ORIGINAL_ROOT: str(project), ORIGINAL_M5_ROOT: str(m5), ORIGINAL_HF_ROOT: str(hf)}
    translated = original
    for old, new in mapping.items():
        translated = translated.replace(old.encode(), new.encode())
    return translated, {
        "target": str(target.relative_to(project)),
        "original_source_sha256": hashlib.sha256(original).hexdigest(),
        "translated_source_sha256": hashlib.sha256(translated).hexdigest(),
        "root_map": mapping,
        "source_file_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="Preserved experiment .py path relative to this project")
    parser.add_argument("--m5-root", type=Path, default=None)
    parser.add_argument("--hf-root", type=Path, default=None)
    parser.add_argument("--check-loader", action="store_true", help="Import only data_split definitions; no data/model execution")
    parser.add_argument("--execute", action="store_true", help="Execute the translated target, including its original writes/training")
    raw_args = sys.argv[1:]
    separator = raw_args.index("--") if "--" in raw_args else len(raw_args)
    args = parser.parse_args(raw_args[:separator])
    target_args = raw_args[separator + 1:] if separator < len(raw_args) else []
    project = Path(__file__).resolve().parents[1]
    target = (project / args.target).resolve()
    if not target.is_relative_to(project):
        parser.error("Target must be inside this project")
    relative = target.relative_to(project)
    if (len(relative.parts) < 3 or relative.parts[0] != "experiments"
            or relative.parts[1] not in SUPPORTED_DIRECTORIES or target.suffix != ".py"):
        parser.error("Target must be a preserved intermittent_gonogo or service_axis_v2 experiment script")
    configured = args.m5_root or os.environ.get("TSFM_M5_ROOT")
    m5 = Path(configured).expanduser().resolve() if configured else (project / ".reproduction-dependencies/m5dataset").resolve()
    configured_hf = args.hf_root or os.environ.get("TSFM_HF_ROOT")
    hf = Path(configured_hf).expanduser().resolve() if configured_hf else (project / ".reproduction-dependencies/huggingface").resolve()
    sys.dont_write_bytecode = True
    manifest = verify_dependency(m5)
    translated, receipt = translated_source(target, project, m5, hf)
    # Syntax validation does not execute any statement in the target.
    compiled = compile(translated, str(target), "exec")
    receipt.update({
        "dependency_manifest_sha256": file_sha256(m5 / "DEPENDENCY_MANIFEST.json"),
        "dependency_files_verified": len(manifest["files"]),
        "dependency_source_head": manifest["source_head"],
        "mode": "execute" if args.execute else "inspect",
        "execution_environment": {"HF_HOME": str(hf), "HF_HUB_CACHE": str(hf / "hub")},
    })
    if args.check_loader:
        spec = importlib.util.spec_from_file_location("preserved_m5_data_split_check", m5 / "common/data_split.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load preserved data_split module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.DATA_DIR.resolve() != (m5 / "data").resolve():
            raise ValueError("Preserved data_split.DATA_DIR resolved incorrectly")
        receipt["loader_check"] = {
            "passed": True,
            "data_dir": str(module.DATA_DIR),
            "train_end": module.TRAIN_END,
            "horizon": module.H,
            "get_split_called": False,
        }
    print(json.dumps(receipt, indent=2), flush=True)
    if not args.execute:
        return 0
    os.environ["HF_HOME"] = str(hf)
    os.environ["HF_HUB_CACHE"] = str(hf / "hub")
    os.chdir(project)
    sys.path[:0] = [str(target.parent), str(project), str(project / "src")]
    sys.argv = [str(target), *target_args]
    namespace = {"__name__": "__main__", "__file__": str(target), "__package__": None}
    exec(compiled, namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
