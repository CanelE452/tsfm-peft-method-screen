#!/usr/bin/env python3
"""Inspect preserved scripts with runtime-only receipt-path remapping.

Only --execute runs the target and its original side effects. Inspection compiles
the target, maps the MOMENT receipt, and optionally parses its model configuration.
It does not construct a model, load weights, infer, score or train.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys


def path_mapping(project, hf, m5):
    return {
        "/home/minjae/Documents/github/tsfm-peft-method-screen": str(project),
        "/home/minjae/.cache/huggingface": str(hf),
        "/home/minjae/Documents/github/m5dataset": str(m5),
    }


def relocate_string(value, mapping, observed):
    for before, after in mapping.items():
        if value == before or value.startswith(before + "/"):
            result = after + value[len(before):]
            if result != value:
                observed[value] = result
            return result
    return value


def relocate_json(value, mapping, observed):
    """Map exact prefixes in JSON values and file-hash dictionary keys."""
    if isinstance(value, str):
        return relocate_string(value, mapping, observed)
    if isinstance(value, list):
        return [relocate_json(item, mapping, observed) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            mapped = relocate_string(key, mapping, observed) if isinstance(key, str) else key
            if mapped in result:
                raise ValueError("Runtime JSON relocation would create a duplicate key")
            result[mapped] = relocate_json(item, mapping, observed)
        return result
    return value


@contextmanager
def mapped_json_reads(mapping, observed):
    """Scope the adapter to this process; json.load delegates to json.loads."""
    original = json.loads
    def loads(source, *args, **kwargs):
        return relocate_json(original(source, *args, **kwargs), mapping, observed)
    json.loads = loads
    try:
        yield
    finally:
        json.loads = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="Project-relative script or experiment .py")
    parser.add_argument("--check-model-config", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--hf-root", type=Path)
    parser.add_argument("--m5-root", type=Path)
    raw_args = sys.argv[1:]
    separator = raw_args.index("--") if "--" in raw_args else len(raw_args)
    args = parser.parse_args(raw_args[:separator])
    forwarded = raw_args[separator + 1:] if separator < len(raw_args) else []
    project = Path(__file__).resolve().parents[1]
    target = (project / args.target).resolve()
    if not target.is_relative_to(project):
        parser.error("Target must stay inside this project")
    relative = target.relative_to(project)
    if relative.parts[0] not in {"scripts", "experiments"} or target.suffix != ".py":
        parser.error("Expected a project script or experiment .py file")
    if target == Path(__file__).resolve():
        parser.error("Cannot launch this launcher recursively")
    hf = Path(args.hf_root or os.environ.get("TSFM_HF_ROOT") or project / ".reproduction-dependencies/huggingface").expanduser().resolve()
    m5 = Path(args.m5_root or os.environ.get("TSFM_M5_ROOT") or project / ".reproduction-dependencies/m5dataset").expanduser().resolve()
    mapping = path_mapping(project, hf, m5)
    source = target.read_bytes()
    translated = source
    for before, after in mapping.items():
        translated = translated.replace(before.encode(), after.encode())
    compiled = compile(translated, str(target), "exec")
    sys.dont_write_bytecode = True
    observed = {}
    receipt_path = project / "results/priority12_20260915/moment_source.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = relocate_json(json.loads(receipt_bytes), mapping, observed)
    snapshot = Path(receipt["snapshot"])
    if not snapshot.is_relative_to(hf) or not (snapshot / "config.json").is_file():
        raise ValueError("MOMENT receipt must resolve to the preserved model configuration")
    report = dict(target=str(relative), mode="execute" if args.execute else "inspect",
                  original_source_sha256=hashlib.sha256(source).hexdigest(),
                  translated_source_sha256=hashlib.sha256(translated).hexdigest(),
                  source_file_modified=False, json_files_modified=False, root_map=mapping,
                  moment_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
                  moment_snapshot=str(snapshot),
                  execution_environment={"HF_HOME": str(hf), "HF_HUB_CACHE": str(hf / "hub")})
    if args.check_model_config:
        import momentfm
        from transformers import T5Config
        config_path = snapshot / "config.json"
        config_data = json.loads(config_path.read_text())
        config = T5Config.from_dict(config_data["t5_config"])
        if config.d_model != 512:
            raise ValueError("Restored MOMENT configuration does not match the small model")
        report["model_config_check"] = dict(passed=True, momentfm_imported=momentfm is not None,
            configuration_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            d_model=config.d_model, model_constructed=False, weights_loaded=False)
    report["observed_json_path_relocations"] = dict(observed)
    print(json.dumps(report, indent=2), flush=True)
    if not args.execute:
        return 0
    os.environ["HF_HOME"] = str(hf)
    os.environ["HF_HUB_CACHE"] = str(hf / "hub")
    os.chdir(project)
    sys.path[:0] = [str(target.parent), str(project), str(project / "src")]
    sys.argv = [str(target), *forwarded]
    namespace = {"__name__": "__main__", "__file__": str(target), "__package__": None}
    try:
        with mapped_json_reads(mapping, observed):
            exec(compiled, namespace)
    finally:
        print(json.dumps({"runtime_json_path_relocations": observed}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
