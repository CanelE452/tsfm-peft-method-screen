"""Final publication checks and manifest; never invokes training or inference."""
from pathlib import Path
import csv
import hashlib
import json
import re
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'experiments' / HERE.name


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def command(args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True, encoding='utf-8').strip()


def allowed(path):
    return path.startswith(f'experiments/{HERE.name}/') or path.startswith(f'results/{HERE.name}/') or path == 'scripts/run_rollout_uncertainty_peft.py'


def main():
    verification = read(HERE / 'VERIFICATION.json')
    assert verification['status'] == 'PASS'
    assert verification['scientific_decision'] == 'STATE_OR_CALIBRATION_SUFFICIENT'
    assert verification['execution_status'] == 'COMPLETED_WITH_DISCLOSED_PROTOCOL_DEVIATION'
    ledger = [json.loads(line) for line in (HERE / 'UPDATE_LEDGER.jsonl').read_text().splitlines()]
    counts = {}
    for kind, expected in [('main', 12288), ('smoke', 12)]:
        records = [r for r in ledger if r['kind'] == kind]
        keys = [(r['fit_id'], r['step']) for r in records]
        assert len(keys) == len(set(keys)) == expected
        assert all(r['status'].startswith('committed') for r in records)
        counts[kind] = len(keys)
    assert sha(HERE / 'UPDATE_LEDGER.jsonl') == verification['final_ledger_sha256']
    assert not list((ROOT / '.cache' / HERE.name / 'fits').glob('*/pending.json'))
    audit_names = ['MODEL_AND_SMOKE_AUDIT.json', 'VALIDATION_AUDIT_Electricity.json', 'VALIDATION_AUDIT_ETTh1.json',
                   'CALIBRATION_INDEPENDENT_AUDIT.json', 'SUPPLEMENTARY_MODEL_VERIFICATION.json']
    assert all(read(HERE / name)['status'] == 'PASS' for name in audit_names)
    for name, value in verification['additional_audit_sha256'].items():
        assert sha(HERE / name) == value
    for name, value in verification['reports_sha256'].items():
        assert sha(HERE / name) == value
    for record in read(HERE / 'SCORE_ARTIFACTS_MANIFEST.json').values():
        path = ROOT / record['path']
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
    figures = read(HERE / 'FIGURES_MANIFEST.json')
    assert figures['figure_types'] == 3 and len(figures['artifacts']) == 3
    assert sha(HERE / 'make_figures.py') == figures['script_sha256']
    for figure in figures['artifacts']:
        for record in figure['files']:
            assert sha(HERE / record['path']) == record['sha256']
    assert sha(EXP / 'contract' / 'MASTER_CLI.txt') == '8b4a50ed2ca3215b4a2b9ea45cada6e3e7d6febef17a72b55f440448e826bfac'
    original = read(HERE / 'TRAINING_CONTRACT_SEALED.json')['source_hashes']
    current = {str(p.relative_to(ROOT)): sha(p) for p in EXP.rglob('*.py')}
    assert current.keys() == original.keys()
    changes = {k for k in current if current[k] != original[k]}
    amendments = read(HERE / 'SOURCE_AMENDMENTS.json')['amendments']
    assert changes == {a['path'] for a in amendments}
    for a in amendments:
        assert original[a['path']] == sha(ROOT / a['original_path']) == a['original_sha256']
        assert current[a['path']] == a['current_sha256']
        assert (ROOT / a['diff_path']).stat().st_size > 0
    xml = ET.parse(HERE / 'CPU_TESTS.xml')
    assert len(xml.findall('.//testcase')) == 24
    assert not xml.findall('.//failure') and not xml.findall('.//error')
    with (HERE / 'SCORES_SUMMARY.csv').open(newline='') as f:
        scored = list(csv.DictReader(f))
    assert len(scored) == 780 and {int(r['seed']) for r in scored} == {0, 92121, 92122}
    assert {r['variant'] for r in scored} == {'raw', 'ordered', 'affine'}
    baseline = 'e84ef88579382574d9c82985c2c0c4672e262b21'
    paths = command(['git', 'diff', '--name-only', baseline]).splitlines()
    paths += command(['git', 'ls-files', '--others', '--exclude-standard']).splitlines()
    assert all(allowed(p) for p in paths), [p for p in paths if not allowed(p)]
    assert not command(['git', 'ls-files', f'.cache/{HERE.name}'])
    old = Path('E:/CODING/proj/hierarchical-tsfm-peft')
    old_head = command(['git', 'rev-parse', 'HEAD'], old)
    assert old_head == 'ab5ef0d5392c62d38beb6c2839192b0227c83e66'
    assert not command(['git', 'status', '--porcelain'], old)
    links = []
    for name in ['REPORT_KO.md', 'FINAL_DECISION.md']:
        for target in re.findall(r'\]\(([^)]+)\)', (HERE / name).read_text(encoding='utf-8')):
            if target.startswith(('https://', 'http://', '#')):
                continue
            path = (HERE / target).resolve()
            if path.name != 'FINAL_ARTIFACT_MANIFEST.json':
                assert path.is_file(), (name, target)
            links.append({'document': name, 'target': target})
    receipt = {'status': 'PASS', 'scope': 'publication integrity; not a scientific PASS',
               'execution_status': verification['execution_status'], 'scientific_decision': verification['scientific_decision'],
               'updates': counts, 'cpu_tests_passed': 24, 'score_rows': len(scored),
               'score_and_figure_artifacts_hashes_verified': True,
               'scoped_paths_only': True, 'raw_cache_untracked': True,
               'prior_hier_tracked_tree_clean_and_head_unchanged': old_head,
               'empty_old_hier_directory': 'Deletion was blocked by automatic approval policy; left untouched. No tracked files changed.',
               'local_report_links': links, 'source_amendments_verified': len(amendments),
               'script_sha256': sha(__file__), 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (HERE / 'PUBLICATION_VERIFICATION.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    files = [*EXP.rglob('*'), *HERE.rglob('*'), ROOT / 'scripts/run_rollout_uncertainty_peft.py']
    records = []
    for p in sorted(files):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix == '.pyc' or p.name == 'FINAL_ARTIFACT_MANIFEST.json':
            continue
        assert p.suffix not in {'.pt', '.safetensors', '.bin', '.npy', '.npz'}
        assert p.stat().st_size < 95_000_000
        records.append({'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size})
    manifest = {'excludes': ['this manifest itself', '__pycache__', 'ignored raw/model/checkpoint/prediction cache'],
                'files': records, 'file_count': len(records), 'bytes': sum(r['bytes'] for r in records)}
    (HERE / 'FINAL_ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    assert (HERE / 'FINAL_ARTIFACT_MANIFEST.json').is_file()
    print(json.dumps({'status': 'PASS', 'artifact_files': len(records), 'bytes': manifest['bytes'], 'report_links': len(links), 'updates': counts}))


if __name__ == '__main__':
    main()
