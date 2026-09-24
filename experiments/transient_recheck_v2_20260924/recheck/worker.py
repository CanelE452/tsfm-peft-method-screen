from __future__ import annotations
import argparse,sys
from pathlib import Path
from .engine import run_engine

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--package',required=True)
    p.add_argument('--public',required=True);p.add_argument('--private',required=True)
    a=p.parse_args()
    run_engine(Path(a.repo),Path(a.package),Path(a.public),Path(a.private))

if __name__=='__main__':
    main()
