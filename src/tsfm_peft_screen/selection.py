from pathlib import Path
import json
from .reproducibility import digest,write_json,sha

def choose(records):
    if not records:raise ValueError('No validation candidates')
    return min(records,key=lambda r:(r['validation_loss'],r['step'],r['lr']))
def seal(path,winners,contract):
    path=Path(path)
    if path.exists():raise FileExistsError('Selection is immutable')
    record={'winners':winners,'contract_hash':digest(contract)}
    record['seal_hash']=digest(record);write_json(path,record)
    return record

def require_seal(path,contract):
    s=json.loads(Path(path).read_text());h=s.pop('seal_hash')
    assert h==digest(s) and s['contract_hash']==digest(contract),'Selection seal mismatch'
    return s
