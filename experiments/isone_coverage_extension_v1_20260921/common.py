"""Coverage-only utilities. No training/model dependency."""
from pathlib import Path
import hashlib,json,os
ROOT=Path(__file__).resolve().parents[2]
NAME='isone_coverage_extension_v1_20260921'
OUT=ROOT/'results'/NAME
CACHE=ROOT/'.cache'/NAME
OLD=ROOT/'results/mag_real_levelshift_confirmation_v1_20260921'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,obj):
    tmp=Path(path).with_suffix(Path(path).suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,path)
def check_seal():
    seal=read(OUT/'PROTOCOL_SEAL.json')
    assert sha(OUT/'COVERAGE_EXTENSION_PROTOCOL.md')==seal['protocol_sha256']
    code=seal['frozen_score_code'];assert sha(ROOT/code['path'])==code['sha256']
    return seal
def check_parent():
    files=read(OUT/'PARENT_PRESERVATION.json')['files']
    for path,digest in files.items():assert sha(ROOT/path)==digest,path
    return len(files)
