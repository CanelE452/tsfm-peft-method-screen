from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--publish', action='store_true')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    for rel, expected in manifest['files'].items():
        got = sha(root / rel)
        if got != expected:
            raise SystemExit(f'Package hash mismatch: {rel}')
    repo = Path(args.repo).resolve()
    py = repo / '.venv' / 'bin' / 'python'
    if not py.exists():
        raise SystemExit('Repository .venv missing; expected previously used Chronos-2 environment')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
    tests = subprocess.run([str(py), '-B', '-m', 'unittest', 'gonogo.selftest', '-v'], cwd=root, env=env)
    if tests.returncode:
        raise SystemExit(tests.returncode)
    cmd = [str(py), '-B', '-m', 'gonogo.suite', '--repo', str(repo), '--package-root', str(root)]
    if args.publish:
        cmd.append('--publish')
    raise SystemExit(subprocess.run(cmd, cwd=root, env=env).returncode)


if __name__ == '__main__':
    main()
