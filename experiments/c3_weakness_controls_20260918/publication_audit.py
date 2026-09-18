"""Check published artifacts and local links after scientific verification."""
import ast
import hashlib
import json
import re
from pathlib import Path

from .common import OUT, ROOT, check_seal, read, save


def audit():
    assert read(OUT / 'VERIFICATION.json')['status'] == 'VERIFIED'
    assert read(OUT / 'status.json')['execution'] == 'VERIFIED'
    assert not (OUT / 'ERROR.json').exists()
    check_seal()
    contract = Path('/home/minjae/Downloads/c3_weakness_resolution_plan_20260918.md')
    assert contract.read_bytes() == (OUT / 'EXECUTION_CONTRACT.md').read_bytes()
    code = ROOT / 'experiments/c3_weakness_controls_20260918'
    for p in code.glob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    links = 0
    for p in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^\s)]+)\)', p.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            assert (p.parent / target.split('#')[0]).exists(), (p, target)
            links += 1
    for name in ['01_primary_controls', '02_tradeoffs', '03_selected_position_gates', '04_all_shapes']:
        for suffix in ['png', 'pdf', 'svg']:
            assert (OUT / 'figures' / (name + '.' + suffix)).stat().st_size > 1000
    artifacts = {}
    for base in [OUT, code]:
        for p in sorted(base.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.name == 'PUBLICATION_AUDIT.json':
                continue
            assert p.stat().st_size < 95 * 2**20, (p, 'GitHub file size')
            artifacts[str(p.relative_to(ROOT))] = dict(
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size
            )
    save(OUT / 'PUBLICATION_AUDIT.json', dict(
        status='VERIFIED', checked_markdown_links=links,
        same_download_contract=True, scientific_seal_unchanged=True,
        figures=12, artifact_hashes=artifacts,
        local_data_weights_predictions_required_for_replay=True,
        automatic_successor=False,
    ))
    print('PUBLICATION_AUDIT_VERIFIED', len(artifacts), flush=True)


if __name__ == '__main__':
    audit()
