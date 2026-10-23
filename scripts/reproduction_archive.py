#!/usr/bin/env python3
"""Inventory, verify, and safely restore the 2026-10-23 reproduction assets.

Only the standard library is required, plus zstd for the compressed archive.
This tool does not run experiments or deserialize model/pickle files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile
import time

INCLUDES = [
    '.cache', 'data', 'runs', 'external', '.reproduction-dependencies',
    'results/intermittent_gonogo_20260922/curves.npz',
    'results/paper_readiness_20260918/run.log',
    'results/transient_peft_readiness_v3_20260923/data_access_raw.log',
    'experiments/transient_difficulty_screen_v1_20260924/AUTHOR_VALIDATION/unit_tests.log',
    'results/delta_adapter_comparison_20260919/completion_hook.log',
]
EXCLUDES = [
    '.cache/tsfm_three_candidate_setup_20260925/envs',
    'external/icdm-2026-reproduction/src/.venv',
]
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'}
OLD_ROOT = '/home/minjae/Documents/github/tsfm-peft-method-screen'
OLD_HF = '/home/minjae/.cache/huggingface'
OLD_M5 = '/home/minjae/Documents/github/m5dataset'

def digest(stream):
    value = hashlib.sha256()
    while data := stream.read(4 * 1024 * 1024):
        value.update(data)
    return value.hexdigest()

def safe_name(name):
    value = PurePosixPath(name)
    if value.is_absolute() or '..' in value.parts or not value.parts:
        raise ValueError('Unsafe archive path: ' + name)
    return value.as_posix()

def walk(root):
    def visit(path):
        name = path.relative_to(root).as_posix()
        if any(name == x or name.startswith(x + '/') for x in EXCLUDES):
            return
        if path.name in SKIP_DIRS or path.name.endswith(('.pyc', '.pyo')):
            return
        if path.name.endswith(('.pid', '.lock')) and path.is_file() and path.stat().st_size < 4096:
            return
        if path.is_symlink():
            yield path
        elif path.is_dir():
            for child in sorted(path.iterdir()):
                yield from visit(child)
        elif path.is_file():
            if path.name in {'.env', '.env.local', 'credentials', 'id_rsa', 'id_ed25519'}:
                raise ValueError('Review sensitive filename before archiving: ' + name)
            yield path
    for prefix in INCLUDES:
        path = root / prefix
        if path.exists() or path.is_symlink():
            yield from visit(path)

def inventory(args):
    root = args.root.resolve()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    total = count = 0
    last = time.monotonic()
    with args.manifest.open('w') as manifest, args.file_list.open('wb') as names:
        for path in walk(root):
            name = path.relative_to(root).as_posix()
            value = path.lstat()
            record = {'path': name, 'mode': stat.S_IMODE(value.st_mode)}
            if path.is_symlink():
                record.update(type='symlink', target=os.readlink(path))
            else:
                with path.open('rb') as source:
                    checksum = digest(source)
                after = path.stat()
                if (after.st_size, after.st_mtime_ns) != (value.st_size, value.st_mtime_ns):
                    raise ValueError('File changed during inventory: ' + name)
                record.update(type='file', size=value.st_size, sha256=checksum)
                total += value.st_size
            manifest.write(json.dumps(record, sort_keys=True) + '\n')
            names.write(os.fsencode(name) + b'\0')
            count += 1
            if time.monotonic() - last > 10:
                print(f'inventory: {count} files, {total / 1e9:.2f} GB hashed', flush=True)
                last = time.monotonic()
    summary = {'source_root': str(root), 'includes': INCLUDES, 'excludes': EXCLUDES,
               'skipped_dirs': sorted(SKIP_DIRS), 'count': count, 'file_bytes': total,
               'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    args.manifest.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)

def archive_stream(folder):
    parts = sorted(folder.glob('tsfm-reproduction.tar.zst.part[0-9][0-9][0-9][0-9]'))
    if not parts:
        raise ValueError('No archive parts found in ' + str(folder))
    for index, part in enumerate(parts):
        if part.name != f'tsfm-reproduction.tar.zst.part{index:04d}':
            raise ValueError('Missing or incorrectly ordered archive part')
    # cat and zstd are kept as separate processes so decompression cannot invoke a shell.
    cat = subprocess.Popen(['cat', '--', *map(str, parts)], stdout=subprocess.PIPE)
    zstd = subprocess.Popen(['zstd', '-q', '-d', '-c'], stdin=cat.stdout, stdout=subprocess.PIPE)
    cat.stdout.close()
    return cat, zstd

def relocate(target, root):
    mapping = [(OLD_ROOT, str(root)),
               (OLD_HF, str(root / '.reproduction-dependencies/huggingface')),
               (OLD_M5, str(root / '.reproduction-dependencies/m5dataset'))]
    for before, after in mapping:
        if target == before or target.startswith(before + '/'):
            return after + target[len(before):]
    return target

def check_or_restore(args):
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    expected = {r['path']: r for r in records}
    if len(expected) != len(records):
        raise ValueError('Duplicate inventory paths')
    root = args.root.resolve()
    seen = {}
    total = 0
    last = time.monotonic()
    cat, zstd = archive_stream(args.parts)
    try:
        with tarfile.open(fileobj=zstd.stdout, mode='r|') as archive:
            for member in archive:
                name = safe_name(member.name)
                if member.isdir():
                    continue
                record = expected.get(name)
                if record is None or name in seen:
                    raise ValueError('Unexpected or duplicate archive member: ' + name)
                destination = root / name
                output = None
                if args.command == 'restore':
                    if not destination.parent.resolve().is_relative_to(root):
                        raise ValueError('Archive parent escapes destination: ' + name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists() or destination.is_symlink():
                        raise ValueError('Refusing to overwrite an existing artifact: ' + name)
                if member.issym():
                    if record['type'] != 'symlink' or member.linkname != record['target']:
                        raise ValueError('Symlink does not match inventory: ' + name)
                    if args.command == 'restore':
                        target = relocate(member.linkname, root)
                        if not (destination.parent / target).resolve().is_relative_to(root):
                            raise ValueError('Unmapped external symlink: ' + name)
                        destination.symlink_to(target)
                    seen[name] = None
                    continue
                if record['type'] != 'file':
                    raise ValueError('Member type mismatch: ' + name)
                if member.islnk():
                    linked = safe_name(member.linkname)
                    if linked not in seen or seen[linked] != record['sha256']:
                        raise ValueError('Invalid hardlink: ' + name)
                    checksum = seen[linked]
                    if args.command == 'restore':
                        os.link(root / linked, destination)
                elif member.isfile():
                    if member.size != record['size']:
                        raise ValueError('File size mismatch: ' + name)
                    value = hashlib.sha256()
                    if args.command == 'restore':
                        output = destination.open('xb')
                    try:
                        with archive.extractfile(member) as stream:
                            while block := stream.read(4 * 1024 * 1024):
                                value.update(block)
                                if output:
                                    output.write(block)
                    finally:
                        if output:
                            output.close()
                    checksum = value.hexdigest()
                    if args.command == 'restore':
                        destination.chmod(record['mode'])
                else:
                    raise ValueError('Unsupported archive member: ' + name)
                if checksum != record['sha256']:
                    raise ValueError('SHA256 mismatch: ' + name)
                seen[name] = checksum
                total += record['size']
                if time.monotonic() - last > 10:
                    print(f'{args.command}: {len(seen)}/{len(expected)} artifacts, {total / 1e9:.2f} GB verified', flush=True)
                    last = time.monotonic()
        # tar readers stop at the end marker; drain zstd so both subprocesses can finish.
        while zstd.stdout.read(4 * 1024 * 1024):
            pass
        if zstd.wait() or cat.wait():
            raise ValueError('Archive concatenation or decompression failed')
    finally:
        for process in (cat, zstd):
            if process.poll() is None:
                process.terminate()
                process.wait()
    if set(seen) != set(expected):
        raise ValueError('Missing inventory members: ' + str(sorted(set(expected) - set(seen))[:10]))
    print(json.dumps({'verified': True, 'artifacts': len(seen), 'file_bytes': total,
                      'operation': args.command}), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inv = commands.add_parser('inventory')
    inv.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    inv.add_argument('--manifest', required=True, type=Path)
    inv.add_argument('--file-list', required=True, type=Path)
    for name in ('verify', 'restore'):
        command = commands.add_parser(name)
        command.add_argument('--manifest', required=True, type=Path)
        command.add_argument('--parts', required=True, type=Path)
        command.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.command == 'inventory':
        inventory(args)
    else:
        check_or_restore(args)

if __name__ == '__main__':
    main()
