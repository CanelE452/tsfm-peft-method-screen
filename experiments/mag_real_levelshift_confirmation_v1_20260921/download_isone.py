"""Authenticated official Jan-Aug 2026 system-load download; no model imports."""
from pathlib import Path
from datetime import date, timedelta, datetime, timezone
import hashlib
import json
import os
import time
import requests

ROOT=Path(__file__).resolve().parents[2]
NAME='mag_real_levelshift_confirmation_v1_20260921'
CACHE=ROOT/'.cache'/NAME
OUT=ROOT/'results'/NAME/'isone_acquisition'
BASE='https://webservices.iso-ne.com/api/v1.1/'

def sha(b):return hashlib.sha256(b).hexdigest()
def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)

def main():
    OUT.mkdir(exist_ok=True)
    folder=CACHE/'isone_raw';folder.mkdir(exist_ok=True)
    credentials=json.loads((CACHE/'private_auth/isone.json').read_text())
    session=requests.Session();session.auth=(credentials['username'],credentials['password'])
    session.headers.update({'Accept':'application/json'})
    receipt_path=OUT/'DOWNLOAD_RECEIPTS.json'
    receipts=json.loads(receipt_path.read_text()) if receipt_path.exists() else []
    byday={r['day']:r for r in receipts}
    day=date(2026,1,1)
    while day<date(2026,9,1):
        key=day.strftime('%Y%m%d');path=folder/(key+'.json')
        if key in byday:
            assert sha(path.read_bytes())==byday[key]['sha256']
            day+=timedelta(days=1);continue
        url=BASE+f'hourlysysload/day/{key}/location/32.json'
        response=None
        for attempt in range(3):
            try:
                response=session.get(url,timeout=(15,60),allow_redirects=False)
            except requests.RequestException:
                if attempt==2:raise SystemExit(f'NETWORK_BLOCK day={key}; completed days preserved')
                time.sleep(2);continue
            if response.status_code in [429,500,502,503,504] and attempt<2:
                time.sleep(5);continue
            break
        if response.status_code!=200:
            raise SystemExit(f'HTTP_BLOCK day={key} status={response.status_code}; completed days preserved')
        packet=response.json();rows=packet['HourlySystemLoads']['HourlySystemLoad']
        assert isinstance(rows,list) and len(rows)>0
        for row in rows:
            assert row['Location']=={'@LocId':'32','$':'NEPOOL AREA'}
            assert row['BeginDate'].startswith(day.isoformat())
            assert 'Load' in row
        tmp=path.with_suffix('.tmp');tmp.write_bytes(response.content);os.replace(tmp,path)
        item={'day':key,'url':url,'download_utc':datetime.now(timezone.utc).isoformat(),'http_status':200,'bytes':len(response.content),'sha256':sha(response.content),'rows':len(rows),'path':str(path.relative_to(ROOT))}
        receipts.append(item);save(receipt_path,receipts)
        print('DOWNLOADED',key,'rows',len(rows),'completed',len(receipts),flush=True)
        day+=timedelta(days=1)
        time.sleep(.15)
    assert len(receipts)==243
    print('DOWNLOAD_COMPLETE_243_DAYS',flush=True)

if __name__=='__main__':main()
