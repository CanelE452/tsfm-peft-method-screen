from __future__ import annotations
import argparse, json, shutil, time, traceback
from pathlib import Path
import torch
from .util import read_json, write_json, sha256, require, git
from .data import ensure_weather, load_weather
from .candidate_a import run as run_a
from .candidate_b import run as run_b
from .candidate_c import run as run_c


def publish(repo, package_root, result_dir):
    exp = repo / 'experiments' / 'actual_tsfm_gonogo_v1_20260925'
    require(not exp.exists(), f'Experiment destination already exists: {exp}')
    shutil.copytree(package_root, exp, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    rels = [str(exp.relative_to(repo)), str(result_dir.relative_to(repo))]
    a = git(['add', '--', *rels], repo)
    require(a['returncode'] == 0, 'git add failed: ' + a['stderr'])
    ck = git(['diff', '--cached', '--check'], repo)
    require(ck['returncode'] == 0, 'git diff --check failed: ' + ck['stdout'] + ck['stderr'])
    c = git(['commit', '-m', f'Run actual TSFM candidate Go-NoGo ({result_dir.name})'], repo)
    require(c['returncode'] == 0, 'commit failed: ' + c['stderr'])
    sha = git(['rev-parse', 'HEAD'], repo)['stdout'].strip()
    p = git(['push', 'origin', 'HEAD:main'], repo)
    return {'commit': sha, 'push_returncode': p['returncode'], 'push_stdout': p['stdout'], 'push_stderr': p['stderr']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--package-root', required=True)
    ap.add_argument('--publish', action='store_true')
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    package_root = Path(args.package_root).resolve()
    cfg = read_json(package_root / 'RUN_CONFIG.json')
    require((repo / '.git').exists(), 'Not a git repository')
    require('tsfm-peft-method-screen' in repo.name, 'Wrong repository')
    require(not (repo / 'experiments' / 'actual_tsfm_gonogo_v1_20260925').exists(), 'This fixed package was already installed/executed')
    status = git(['status', '--porcelain'], repo)
    require(status['returncode'] == 0 and status['stdout'].strip() == '', 'Working tree must be clean before run')

    run_id = time.strftime('run_%Y%m%dT%H%M%SZ', time.gmtime())
    out = repo / 'results' / 'actual_tsfm_gonogo_v1_20260925' / run_id
    out.mkdir(parents=True)
    head = git(['rev-parse', 'HEAD'], repo)['stdout'].strip()
    planned_steps = 4 * cfg['candidate_a']['steps'] + 6 * cfg['candidate_b']['steps']
    require(planned_steps <= cfg['limits']['max_total_optimizer_steps'], 'Planned optimizer budget exceeds contract')
    manifest = {
        'head_at_start': head,
        'run_id': run_id,
        'config': cfg,
        'optimizer_steps_planned': planned_steps,
        'test_split_numeric_values_parsed': False,
        'test_split_scored': False,
        'downloads_explicitly_authorized': {'weather': True, 'chronos_2': True},
        'package_install_allowed': False,
    }
    write_json(out / 'RUN_MANIFEST.json', manifest)

    final = None
    try:
        require(torch.cuda.is_available(), 'CUDA unavailable')
        weather_path = ensure_weather(repo, cfg)
        data = load_weather(weather_path, cfg)
        write_json(out / 'DATA.json', {
            'path': str(weather_path),
            'sha256': sha256(weather_path),
            'rows_total_file': data['n_total'],
            'rows_numerically_parsed': data['n'],
            'channels': data['c'],
            'headers': data['header'],
            'train_end': data['train_end'],
            'cal_mid': data['cal_mid'],
            'val_end': data['val_end'],
            'test_start': data['test_start'],
            'test_numeric_values_parsed': False,
            'test_scored': False,
        })
        A, payload = run_a(data, cfg, out)
        C = run_c(payload, cfg, out)
        B = run_b(data, cfg, out)
        final = {
            'status': 'COMPLETED_ACTUAL_TRAINING_GONOGO',
            'A_DECISION': A['decision'],
            'B_DECISION': B['decision'],
            'C_DECISION': C['decision'],
            'actual_optimizer_steps': planned_steps,
            'test_split_numeric_values_parsed': False,
            'test_split_scored': False,
            'second_seed': False,
            'interpretation': 'One-seed pilot. GO means confirmation-worthy, not paper-level evidence.',
        }
        write_json(out / 'FINAL_DECISION.json', final)
    except Exception as e:
        final = {
            'status': 'TECHNICAL_FAILURE',
            'error': repr(e),
            'test_split_numeric_values_parsed': False,
            'test_split_scored': False,
        }
        write_json(out / 'FAILURE.json', {
            **final,
            'traceback': traceback.format_exc(),
        })

    if args.publish:
        receipt = publish(repo, package_root, out)
        # Publication receipt is printed because writing it after commit would create an uncommitted file.
        print(json.dumps({'publication': receipt}, indent=2))
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return 0 if final['status'] != 'TECHNICAL_FAILURE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
