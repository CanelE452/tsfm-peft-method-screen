"""Post-hoc forecast reliability audit; no model import or deployment policy."""
import csv,hashlib,json,math,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];RUN='covariate_reliability_diagnostic_20260916';OUT=ROOT/'research'/RUN
OLD=ROOT/'results/covariate_vintage_reference_20260916'
FEATURES=['temperature_2m','wind_speed_10m','shortwave_radiation']
TAUS=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def primary(q,y,sd):
 e=y[None]-q;return float(np.maximum(TAUS[:,None]*e,(TAUS[:,None]-1)*e).mean()*2/sd)
def scalar(q,y,sd):
 s=0.
 for i,t in enumerate(TAUS):
  for h in range(24):
   e=float(y[h])-float(q[i,h]);s+=2*(t*e if e>=0 else (t-1)*e)
 return s/(21*24*sd)
def ranks(x):return np.array([1+sum(y<v for y in x)+(sum(y==v for y in x)-1)/2 for v in x],dtype=float)
def corr(x,y):
 a=ranks(x);b=ranks(y);a-=a.mean();b-=b.mean();den=math.sqrt(sum(v*v for v in a)*sum(v*v for v in b));return float(sum(v*w for v,w in zip(a,b))/den) if den else None

def main():
 assert not (OUT/'verification.json').exists(),'No repeated diagnostic overwrite'
 tracked=subprocess.check_output(['git','ls-files','results','research'],cwd=ROOT,text=True).splitlines();old={p:sha(ROOT/p) for p in tracked}
 seal=read(OLD/'seal.json');ps=read(OLD/'predictions.json');finals=read(OLD/'final_predictions.json');oldscore=pd.read_csv(OLD/'scores.csv').set_index(['id','method'])
 wm=ROOT/'.cache/covariate_availability_audit_20260916/weather_measurements/mv_feeder/OS Gorredijk.parquet'
 dm=read(ROOT/'research/covariate_availability_audit_20260916/download_manifest.json');expected=next(r['sha256'] for r in dm if r['path'].startswith('weather_measurements/'));assert sha(wm)==expected
 weather=pd.read_parquet(wm)
 if 'timestamp' not in weather.columns:weather=weather.reset_index()
 weather['timestamp']=pd.to_datetime(weather.timestamp,utc=True);weather=weather.set_index('timestamp');assert weather.index.is_unique
 origins=[];pathrows=[];maxloss=0.;maxweather=0.
 for j,p,f in zip(seal['jobs'],ps,finals):
  assert j['id']==p['id']==f['id'] and sha(ROOT/p['path'])==p['sha256'] and sha(ROOT/f['path'])==f['sha256']
  for key in ['input','target']:assert sha(ROOT/j[key])==j[key+'_sha256']
  x=np.load(ROOT/j['input']);raw=np.load(ROOT/p['path']);final=np.load(ROOT/f['path']);y=np.load(ROOT/j['target'])
  wsd=np.maximum(x['past'].astype(float).std(1,ddof=0),1e-6);paths=x['paths'].astype(float);q=np.sort(raw['PATHS'],axis=1)
  idx=pd.date_range(pd.Timestamp(j['origin']),periods=96,freq='15min');truth=weather.reindex(idx)[FEATURES].to_numpy(dtype=float)
  assert np.isfinite(truth).all(),'WEATHER_PROXY_MISSING: no date replacement'
  truth=truth.reshape(24,4,3).mean(1).T
  input_spread=float((paths.std(0,ddof=0)/wsd[:,None]).mean());output_spread=float(q[:,10].std(0,ddof=0).mean()/j['std']);errs=[];loss=[]
  for k in range(4):
   e=float((abs(paths[k]-truth)/wsd[:,None]).mean());es=sum(abs(float(paths[k,a,h])-float(truth[a,h]))/float(wsd[a]) for a in range(3) for h in range(24))/72
   assert math.isclose(e,es,rel_tol=1e-12,abs_tol=1e-12);maxweather=max(maxweather,abs(e-es));errs.append(e)
   value=primary(q[k],y,j['std']);sv=scalar(q[k],y,j['std']);assert math.isclose(value,sv,rel_tol=1e-10,abs_tol=1e-10);maxloss=max(maxloss,abs(value-sv));loss.append(value)
   pathrows.append(dict(id=j['id'],role=j['role'],month=j['month'],path=k,primary=value,weather_error_proxy=e))
  f0=primary(final['F0'],y,j['std']);mix=primary(final['MIXTURE'],y,j['std'])
  assert math.isclose(loss[0],float(oldscore.loc[(j['id'],'LATEST'),'primary']),rel_tol=1e-12)
  assert math.isclose(f0,float(oldscore.loc[(j['id'],'F0'),'primary']),rel_tol=1e-12)
  assert math.isclose(mix,float(oldscore.loc[(j['id'],'MIXTURE'),'primary']),rel_tol=1e-12)
  origins.append(dict(id=j['id'],role=j['role'],month=j['month'],input_spread=input_spread,output_spread=output_spread,weather_error_proxy=errs[0],latest=loss[0],f0=f0,mixture=mix,damage=loss[0]-f0,mixture_gain=loss[0]-mix,oldest_minus_latest=loss[3]-loss[0],oldest_weather_minus_latest=errs[3]-errs[0]))
  x.close();raw.close();final.close()
 df=pd.DataFrame(origins);pf=pd.DataFrame(pathrows);df.to_csv(OUT/'origins.csv',index=False);pf.to_csv(OUT/'path_scores.csv',index=False)
 thresholds={v:float(df[df.role=='C'][v].median()) for v in ['input_spread','output_spread']};save('thresholds.json',dict(role='C',definition='median, high strictly greater',values=thresholds,not_a_deployment_policy=True))
 cohorts=[];correlations=[];summaries=[]
 for role in ['C','D']:
  part=df[df.role==role]
  for signal in ['input_spread','output_spread','weather_error_proxy']:
   for outcome in ['damage','mixture_gain']:
    rho=corr(part[signal].to_numpy(),part[outcome].to_numpy());ref=float(part[signal].rank(method='average').corr(part[outcome].rank(method='average')))
    assert rho is not None and abs(rho-ref)<1e-12
    correlations.append(dict(role=role,n=len(part),signal=signal,outcome=outcome,spearman=rho,signal_available_at_prediction=signal!='weather_error_proxy',requires_four_model_paths=signal=='output_spread'))
  for signal,threshold in thresholds.items():
   for name,mask in [('low',part[signal]<=threshold),('high',part[signal]>threshold)]:
    sel=part[mask];assert len(sel)>0
    cohorts.append(dict(role=role,signal=signal,cohort=name,n=len(sel),threshold=threshold,latest_primary=float(sel.latest.mean()),f0_primary=float(sel.f0.mean()),mixture_primary=float(sel.mixture.mean()),damage_mean=float(sel.damage.mean()),harmful_origins=int((sel.damage>0).sum()),mixture_gain_mean=float(sel.mixture_gain.mean()),weather_error_proxy_mean=float(sel.weather_error_proxy.mean())))
  for k in range(4):
   sel=pf[(pf.role==role)&(pf.path==k)];summaries.append(dict(role=role,path=k,n=len(sel),primary=float(sel.primary.mean()),weather_error_proxy=float(sel.weather_error_proxy.mean())))
 pd.DataFrame(cohorts).to_csv(OUT/'cohorts.csv',index=False);pd.DataFrame(correlations).to_csv(OUT/'correlations.csv',index=False);pd.DataFrame(summaries).to_csv(OUT/'path_summary.csv',index=False)
 save('paired_age_summary.json',[dict(role=role,origins=len(part),oldest_weather_worse=int((part.oldest_weather_minus_latest>0).sum()),oldest_prediction_worse=int((part.oldest_minus_latest>0).sum()),mean_weather_change=float(part.oldest_weather_minus_latest.mean()),mean_primary_change=float(part.oldest_minus_latest.mean()),latest_harmful_vs_f0=int((part.damage>0).sum())) for role,part in df.groupby('role')])
 assert all(sha(ROOT/p)==h for p,h in old.items());save('historical_hashes.json',old)
 save('verification.json',dict(status='COMPLETE',fits=0,updates=0,model_forwards=0,gpu_used=False,new_deployment_policies=0,origin_count=28,path_score_count=112,reused_primary_checks=84,weather_scalar_checks=112,correlation_checks=12,max_primary_scalar_error=maxloss,max_weather_scalar_error=maxweather,baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=sha(__file__),scope_sha256=sha(OUT/'SCOPE.md'),weather_measurements_sha256=sha(wm),historical_files_preserved=len(old),exposure='POSTHOC_C_AND_REUSED_D_ONLY',weather_truth_role='EX_POST_PROXY_NOT_MODEL_INPUT',scientific_pass='NOT_EVALUATED'))
 print(pd.DataFrame(correlations).to_string(index=False));print(pd.DataFrame(summaries).to_string(index=False));print(pd.DataFrame(cohorts).to_string(index=False))
if __name__=='__main__':main()
