"""Fixed dates, four bounded HTTP workers, immutable reuse, no score/model calls."""
from .common import *
from datetime import date,datetime,timedelta,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import threading,time,requests
local=threading.local()

def main():
    seal=check_seal();check_parent()
    folder=CACHE/'raw';folder.mkdir(exist_ok=True)
    credentials=read(ROOT/'.cache/mag_real_levelshift_confirmation_v1_20260921/private_auth/isone.json')
    def fetch(day):
        if not hasattr(local,'session'):
            local.session=requests.Session();local.session.auth=(credentials['username'],credentials['password'])
            local.session.headers.update({'Accept':'application/json'})
        key=day.strftime('%Y%m%d');url=f'https://webservices.iso-ne.com/api/v1.1/hourlysysload/day/{key}/location/32.json'
        for attempt in range(3):
            try:r=local.session.get(url,timeout=(15,60),allow_redirects=False)
            except requests.RequestException:
                if attempt==2:raise RuntimeError(f'NETWORK_BLOCK {key}') from None
                time.sleep(3);continue
            if r.status_code in [429,500,502,503,504] and attempt<2:time.sleep(10);continue
            break
        if r.status_code!=200:raise RuntimeError(f'HTTP_BLOCK {key} {r.status_code}')
        payload=r.json();container=payload.get('HourlySystemLoads')
        if not isinstance(container,dict):raise RuntimeError(f'SCHEMA_BLOCK {key}')
        rows=container.get('HourlySystemLoad',[])
        if isinstance(rows,dict):rows=[rows]
        assert isinstance(rows,list)
        for row in rows:
            assert row['Location']=={'@LocId':'32','$':'NEPOOL AREA'}
            assert row['BeginDate'].startswith(day.isoformat()) and 'Load' in row
        path=folder/(key+'.json');assert not path.exists(),'ORPHAN_RAW_REQUIRES_AUDIT'
        tmp=path.with_suffix('.tmp');tmp.write_bytes(r.content);os.replace(tmp,path)
        item={'day':key,'url':url,'download_utc':datetime.now(timezone.utc).isoformat(),'http_status':200,'bytes':len(r.content),'sha256':sha(path),'rows':len(rows),'path':str(path.relative_to(ROOT)),'origin':'new_download'}
        save(folder/(key+'.receipt.json'),item)
        time.sleep(.25)
        return item
    receiptpath=OUT/'DOWNLOAD_RECEIPTS.json'
    byday={r['day']:r for r in read(receiptpath)} if receiptpath.exists() else {}
    # Recover only complete per-day file+receipt pairs after interrupted downloads.
    for p in sorted(folder.glob('*.receipt.json')):
        item=read(p);assert sha(ROOT/item['path'])==item['sha256'];byday[item['day']]=item
    for r in read(OLD/'isone_acquisition/DOWNLOAD_RECEIPTS.json'):
        assert sha(ROOT/r['path'])==r['sha256'];byday[r['day']]={**r,'origin':'b3761e5_immutable_reuse'}
    probe=next(r for r in read(OUT/'AVAILABILITY_PROBES.json') if r['label']=='20200101')
    if '20200101' not in byday:
        byday['20200101']={**probe,'day':'20200101','origin':'availability_probe_reuse','download_utc':probe['utc']}
    for row in byday.values():assert sha(ROOT/row['path'])==row['sha256']
    first=date.fromisoformat(seal['first_day']);last=date.fromisoformat(seal['last_day_inclusive'])
    days=[first+timedelta(days=i) for i in range((last-first).days+1)]
    missing=[d for d in days if d.strftime('%Y%m%d') not in byday]
    def checkpoint():save(receiptpath,[byday[k] for k in sorted(byday)])
    checkpoint();print('START',len(days),'days; reused',len(byday),'new',len(missing),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,d):d for d in missing}
        for future in as_completed(futures):
            try:r=future.result()
            except BaseException:
                for f in futures:f.cancel()
                checkpoint();raise
            byday[r['day']]=r
            if len(byday)%25==0:
                checkpoint();print('PROGRESS',len(byday),'/',len(days),'last',r['day'],flush=True)
    checkpoint();assert len(byday)==len(days)
    check_parent();check_seal()
    save(OUT/'DOWNLOAD_STATUS.json',{'status':'COMPLETE','daily_files':len(byday),'reused_2026_days':243,'new_or_probe_days':len(byday)-243,'optimizer_updates':0,'model_predictions':0,'target_error_scores':0})
    print('DOWNLOAD_COMPLETE',len(byday),flush=True)
if __name__=='__main__':main()
