from __future__ import annotations
import argparse,traceback
from pathlib import Path
from .common import read_json,write_json,Blocked,utc
from .engine import main_topic

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--topic',choices=['wind','respiration','coordinates'],required=True)
    ap.add_argument('--repo',required=True);ap.add_argument('--cache',required=True);ap.add_argument('--public',required=True);ap.add_argument('--config',required=True)
    ap.add_argument('--wind-csv');ap.add_argument('--wind-locations');ap.add_argument('--wind-archive')
    a=ap.parse_args();out=Path(a.public);out.mkdir(parents=True,exist_ok=True)
    cfg=read_json(a.config);cfg['paths']={k:v for k,v in vars(a).items() if k.startswith('wind_') and v}
    try:
        r=main_topic(a.topic,Path(a.repo),Path(a.cache),out,cfg);print(r['axis_status']);return 0
    except Blocked as e:
        write_json(out/'RESULT.json',{'topic':a.topic,'axis_status':'BLOCKED_DATA_OR_ENVIRONMENT','reason':str(e),
              'scientific_no_go':False,'neural_fit':0,'utc':utc()});return 3
    except Exception as e:
        (out/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8',newline='\n')
        write_json(out/'RESULT.json',{'topic':a.topic,'axis_status':'IMPLEMENTATION_OR_CONTRACT_ERROR','reason':str(e),
              'scientific_no_go':False,'utc':utc()});return 4
if __name__=='__main__':raise SystemExit(main())
