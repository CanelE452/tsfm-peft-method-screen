"""Small deterministic I/O helpers; no network, package installation, or code generation."""
from __future__ import annotations
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def redact(text: str) -> str:
    text = re.sub(r'hf_[A-Za-z0-9]{10,}', '[REDACTED_HF_TOKEN]', text)
    text = re.sub(r'(?i)(authorization\s*[:=]\s*bearer\s+)\S+', r'\1[REDACTED]', text)
    text = re.sub(r'https://[^/\s@]+@github\.com', 'https://[REDACTED]@github.com', text)
    return text


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(t) -> str:
    import torch
    x = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode())
    h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_hash(state: dict) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(tensor_hash(value).encode())
    return h.hexdigest()


def git(repo: Path, *args: str, timeout: int = 30, check: bool = True) -> str:
    r = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise ContractError(f"git {args[0]} failed ({r.returncode}): {redact(r.stderr)}")
    return r.stdout.strip()


def verify_manifest(root: Path) -> dict:
    manifest = read_json(root / 'PACKAGE_MANIFEST.json')
    expected = manifest['files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'PACKAGE_MANIFEST.json'}
    # Runtime receipts are never written into the installed source directory.
    require(actual == set(expected), f'Package file set mismatch: extra={sorted(actual-set(expected))}; missing={sorted(set(expected)-actual)}')
    for rel, sha in expected.items():
        require(file_hash(root / rel) == sha, f'Package modified: {rel}')
    return manifest
