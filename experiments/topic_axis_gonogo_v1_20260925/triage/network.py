"""Bounded public-data retrieval only. Never execute downloaded code or replace data."""
from __future__ import annotations
import hashlib,json,os,time,urllib.request,urllib.parse
from pathlib import Path
from .common import Blocked,require,write_json,hash_file,utc

class Fetcher:
    def __init__(self,cache,public,max_bytes=1100*1024**2,seconds=600):
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.public=Path(public);self.used=0;self.limit=max_bytes;self.deadline=time.monotonic()+seconds;self.events=[]
    def get(self,url,name,cap=16*1024**2,sha=None,md5=None):
        p=self.cache/name;p.parent.mkdir(parents=True,exist_ok=True)
        def valid():
            if not p.is_file() or p.stat().st_size>cap:return False
            if sha and hash_file(p)!=sha:return False
            if md5:
                h=hashlib.md5()
                with p.open('rb') as f:
                    for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
                if h.hexdigest()!=md5:return False
            return True
        if not valid():
            if time.monotonic()>self.deadline:raise Blocked('DOWNLOAD_TIME_BUDGET')
            req=urllib.request.Request(url,headers={'User-Agent':'ResearchReadiness/1.0'})
            part=p.with_suffix(p.suffix+'.part')
            try:
                with urllib.request.urlopen(req,timeout=30) as r,part.open('wb') as out:
                    n=int(r.headers.get('Content-Length','0') or 0)
                    if n>cap or self.used+n>self.limit:raise Blocked('DOWNLOAD_SIZE_BUDGET')
                    total=0
                    while True:
                        if time.monotonic()>self.deadline:raise Blocked('DOWNLOAD_TIME_BUDGET')
                        b=r.read(1048576)
                        if not b:break
                        total+=len(b);self.used+=len(b)
                        if total>cap or self.used>self.limit:raise Blocked('DOWNLOAD_SIZE_BUDGET')
                        out.write(b)
                part.replace(p)
            except Exception as e:
                self.events.append({'url':url,'status':'FAILED','error':str(e),'utc':utc()});self.flush()
                raise Blocked('DATA_DOWNLOAD_FAILED: '+str(e)) from e
            if not valid():raise Blocked('DATA_CHECKSUM_MISMATCH')
        self.events.append({'url':url,'file':name,'bytes':p.stat().st_size,'sha256':hash_file(p),'status':'AVAILABLE'})
        self.flush();return p
    def json(self,url,name):return json.loads(self.get(url,name,cap=4*1024**2).read_text())
    def flush(self):write_json(self.public/'DOWNLOADS.json',{'events':self.events,'downloaded_bytes':self.used})
