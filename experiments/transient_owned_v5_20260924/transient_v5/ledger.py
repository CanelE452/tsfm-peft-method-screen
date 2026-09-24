from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import json
from .util import now,require

class Ledger:
    def __init__(self,path: Path):
        self.path=path; path.parent.mkdir(parents=True,exist_ok=True)
        self.sequence=0
    def record(self,operation,state,**extra):
        self.sequence+=1
        with self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'seq':self.sequence,'utc':now(),'operation':operation,'state':state,**extra},ensure_ascii=False,allow_nan=False)+'\n')
            f.flush()
    @contextmanager
    def operation(self,name,**extra):
        self.record(name,'START',**extra)
        try:
            yield
        except Exception as e:
            self.record(name,'FAILED',error=type(e).__name__,**extra)
            raise
        else:
            self.record(name,'COMPLETED',**extra)


def summarize(path: Path):
    counts={}
    if path.exists():
        for line in path.read_text().splitlines():
            r=json.loads(line); counts.setdefault(r['operation'],{'START':0,'COMPLETED':0,'FAILED':0})[r['state']]+=1
    return counts
