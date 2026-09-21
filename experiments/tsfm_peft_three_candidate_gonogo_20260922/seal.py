from common import *
import platform
import subprocess


def seal(candidate):
    path=RESULTS/f'{candidate}_SOURCE_SEAL.json'
    assert not path.exists()
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in EXP.rglob('*')
             if p.is_file() and p.suffix in ['.py','.md','.txt']}
    write(path,{'time':time.time(),'candidate':candidate,'source':sources,
                'main_ledger_before':read(RESULTS/'OPTIMIZER_LEDGER.json'),
                'data_audit_sha256':sha(RESULTS/'DATA_AND_SPLIT_AUDIT.json'),
                'allocation_sha256':sha(RESULTS/'Q_ALLOCATION.json'),
                'upstream_sha256':sha(RESULTS/'UPSTREAM.json')})


if __name__=='__main__':seal(sys.argv[1])
