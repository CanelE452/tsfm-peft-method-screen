"""Independent scalar audit of final caches and published aggregates; no fitting."""
import ast,csv,math
from common import *
from run import verify_seal

def main():
 seal=verify_seal();rows=list(csv.DictReader(open(OUT/'scores.csv')));lookup={(r['id'],r['method']):r for r in rows};checked=0;max_relative=0.
 finals=read(OUT/'final_predictions.json');assert len(finals)==28
 for j,p in zip(seal['jobs'],finals):
  assert j['id']==p['id'] and sha(ROOT/p['path'])==p['sha256'];y=np.load(ROOT/j['target'])
  with np.load(ROOT/p['path']) as pred:
   assert len(pred.files)==12
   for name in pred.files:
    q=pred[name];n=len(y);sq=ab=width=inside=0.;pin=0.
    for h in range(n):
     d=float(q[10,h])-float(y[h]);sq+=d*d;ab+=abs(d);width+=float(q[18,h])-float(q[2,h]);inside+=int(q[2,h]<=y[h]<=q[18,h])
     for i,tau in enumerate(TAUS):
      e=float(y[h])-float(q[i,h]);pin+=2*(tau*e if e>=0 else (tau-1)*e)
    raw=pin/(n*len(TAUS));v=dict(primary=raw/j['std'],raw_2pinball=raw,raw_RMSE=math.sqrt(sq/n),raw_MAE=ab/n,coverage80=inside/n,width80=width/n)
    for k,z in v.items():
     r=float(lookup[j['id'],name][k]);assert math.isclose(z,r,rel_tol=1e-10,abs_tol=1e-10),(j['id'],name,k,z,r);max_relative=max(max_relative,abs(z-r)/max(1,abs(z)))
    checked+=1
 assert checked==336 and len(lookup)==336
 with open(OUT/'macro.csv') as f:
  for r in csv.DictReader(f):
   sel=[s for s in rows if s['role']==r['role'] and s['method']==r['method']];assert len(sel)==int(r['n'])
   for k in ['primary','raw_2pinball','raw_RMSE','raw_MAE','coverage80','width80']:assert math.isclose(sum(float(s[k]) for s in sel)/len(sel),float(r[k]),rel_tol=1e-12,abs_tol=1e-12)
 state=read(OUT/'status.json');calls=read(OUT/'calls.json');assert len(calls)==state['pipeline_calls']==128;assert sum(r['target_forecasts'] for r in calls)==state['target_forecasts']==233
 assert not any(r['contaminated'] for r in calls);assert state['fits']==state['updates']==0
 old=read(OUT/'historical_hashes.json');assert all(sha(ROOT/p)==h for p,h in old.items())
 for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
 save(OUT/'publication_audit.json',dict(status='VERIFIED',meaning='execution and numeric audit, not scientific PASS',final_cache_files=28,scalar_metric_rows=checked,scalar_metric_values=checked*6,max_relative_error=max_relative,macro_aggregates_verified=True,neural_fits=0,pipeline_calls=128,target_forecasts=233,contaminated_calls=0,source_data_model_seal_verified=True,historical_files_preserved=len(old),report_sha256=sha(OUT/'REPORT.md'),interpretation_sha256=sha(OUT/'INTERPRETATION.md'),posthoc_scripts={p.name:sha(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['verify.py','interpret.py']}))
 print((OUT/'publication_audit.json').read_text())
if __name__=='__main__':main()
