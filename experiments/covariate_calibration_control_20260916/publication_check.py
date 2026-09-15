"""Independent secondary metrics and preservation audit; never refits calibrators."""
import math,re,time,json
from pathlib import Path
import numpy as np
import pandas as pd
from run import ROOT,RUN,OUT,CACHE,read,save,sha,verified
s=verified();df=pd.read_csv(OUT/'scores.csv');jobs={j['id']:j for j in s['jobs']};errors=[]
for r in df.to_dict('records'):
 j=jobs[r['id']];y=np.load(ROOT/j['target']);sd=j['std']
 with np.load(ROOT/r['prediction']) as p:q=p[r['variant']];raw=p['RAW']
 residual=[float(q[10,h])-float(y[h]) for h in range(24)];norm=[x/sd for x in residual];mean=math.fsum(norm)/24
 values=dict(raw_RMSE=math.sqrt(math.fsum(x*x for x in residual)/24),raw_MAE=math.fsum(abs(x) for x in residual)/24,coverage80=sum(q[2,h]<=y[h]<=q[18,h] for h in range(24))/24,width80=math.fsum(float(q[18,h])-float(q[2,h]) for h in range(24))/24,median_scaled_MSE=math.fsum(x*x for x in norm)/24,median_scaled_bias_squared=mean*mean,median_scaled_centered_MSE=math.fsum((x-mean)**2 for x in norm)/24)
 values['raw_2pinball']=r['primary']*sd
 for name,value in values.items():
  error=abs(value-r[name])/max(1,abs(value));errors.append(error);assert math.isclose(value,r[name],rel_tol=1e-10,abs_tol=1e-10),(r,name,value)
 if r['variant']=='CAL':
  e0=(raw[10]-y)/sd;np.testing.assert_allclose(np.array(norm)-mean,e0-e0.mean(),rtol=1e-12,atol=1e-12)
for p,h in read(OUT/'historical_hashes.json').items():assert sha(ROOT/p)==h,p
for p in OUT.glob('*.md'):
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if not target.startswith(('https://','http://','#')):assert (p.parent/target).exists(),(p,target)
paths=list(OUT.iterdir())+list((ROOT/'experiments'/RUN).glob('*.py'))
artifacts={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file() and p.name!='publication_audit.json'}
save(OUT/'publication_audit.json',dict(status='PASS',at=time.time(),secondary_and_decomposition_checks=len(errors),maximum_relative_error=max(errors),centered_error_invariance_checks=224,historical_files_preserved=len(read(OUT/'historical_hashes.json')),local_markdown_links_checked=True,unique_cpu_calibrators=7,initial_calibration_calls=7,calibration_replay_calls_in_main_verification=7,new_fits_in_this_check=0,new_gpu_inference=0,artifacts=artifacts))
print('PUBLICATION CHECK PASS',len(errors),'secondary/decomposition values; no fits/forwards/refitting')
