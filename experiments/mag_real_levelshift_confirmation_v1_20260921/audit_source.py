"""Replay NYISO raw schema/quality audit. No model, training or forecast scoring."""
from pathlib import Path
import hashlib, json, zipfile, io
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
NAME='mag_real_levelshift_confirmation_v1_20260921'
OUT=ROOT/'results'/NAME
CACHE=ROOT/'.cache'/NAME

def main():
 receipts=json.loads((OUT/'SOURCE_RECEIPT.json').read_text()); frames=[]; schemas=set()
 for receipt in receipts:
  path=CACHE/receipt['url'].split('/')[-1]; raw=path.read_bytes()
  assert hashlib.sha256(raw).hexdigest()==receipt['sha256']
  with zipfile.ZipFile(io.BytesIO(raw)) as z:
   for item in receipt['csvs']:
    b=z.read(item['name']);assert hashlib.sha256(b).hexdigest()==item['sha256']
    df=pd.read_csv(io.BytesIO(b));schemas.add(tuple(df.columns));frames.append(df)
 d=pd.concat(frames,ignore_index=True)
 assert schemas=={('Time Stamp','Time Zone','Name','PTID','Load')},schemas
 local=pd.to_datetime(d['Time Stamp'],format='%m/%d/%Y %H:%M:%S')
 offsets=d['Time Zone'].map({'EST':5,'EDT':4});assert offsets.notna().all()
 utc=(local+pd.to_timedelta(offsets,unit='h')).dt.tz_localize('UTC')
 roundtrip=utc.dt.tz_convert('America/New_York').dt.tz_localize(None)
 per_entity={}
 for name,idx in d.groupby('Name').groups.items():
  t=utc.loc[idx].sort_values().drop_duplicates()
  per_entity[name]={'rows':len(idx),'interval_seconds_counts':{str(k):int(v) for k,v in t.diff().dt.total_seconds().value_counts().items()}}
 audit={'status':'BLOCKED_SCHEMA_NO_SYSTEM_TOTAL','monthly_archives':len(receipts),'csv_files':len(frames),'rows':len(d),'schema':[list(s) for s in schemas], 'entities':sorted(d.Name.unique()),'entity_ptids':d[['Name','PTID']].drop_duplicates().to_dict('records'),'timezone_counts':d['Time Zone'].value_counts().to_dict(),'duplicate_timestamp_entity':int(d.duplicated(['Time Stamp','Time Zone','Name']).sum()),'nonfinite_load':int((~np.isfinite(d.Load)).sum()),'missing_cells':int(d.isna().sum().sum()),'dst_roundtrip_mismatches':int((roundtrip!=local).sum()),'utc_min':str(utc.min()),'utc_max':str(utc.max()),'per_entity_sampling':per_entity,'system_total_present':False,'zone_sum_created':False,'new_model_predictions':0,'target_scores':0,'origins_constructed':0}
 assert set(d.Name)=={'CAPITL','CENTRL','DUNWOD','GENESE','HUD VL','LONGIL','MHK VL','MILLWD','N.Y.C.','NORTH','WEST'}
 (OUT/'NYISO_SCHEMA_AUDIT.json').write_text(json.dumps(audit,indent=2))
 print(json.dumps({k:v for k,v in audit.items() if k!='per_entity_sampling'},indent=2))
if __name__=='__main__':main()
