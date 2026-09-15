"""Cached-forecast calibration control. No torch/model/GPU imports or training."""
import argparse,importlib.util,hashlib,json,math,time,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];RUN='covariate_calibration_control_20260916';OUT=ROOT/'results'/RUN;CACHE=ROOT/'.cache'/RUN;OLD=ROOT/'results/covariate_lora_controls_20260916'
SOURCE=ROOT/'experiments/covariate_vintage_reference_20260916/common.py'
spec=importlib.util.spec_from_file_location('reference_math',SOURCE);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
read,save,sha=ref.read,ref.save,ref.sha

def prepare():
 assert not (OUT/'seal.json').exists(),'Do not overwrite preparation'
 old=read(OLD/'seal.json');assert read(OLD/'status.json')['status']=='COMPLETE'
 for p,h in read(OLD/'publication_audit.json')['artifacts'].items():assert sha(ROOT/p)==h,p
 predictions=read(OLD/'predictions.json')
 for p in predictions:assert sha(ROOT/p['path'])==p['sha256']
 for j in old['jobs']:
  for key in ['input','target']:assert sha(ROOT/j[key])==j[key+'_sha256']
 tracked=subprocess.check_output(['git','ls-files','results','research'],cwd=ROOT,text=True).splitlines()
 save(OUT/'historical_hashes.json',{p:sha(ROOT/p) for p in tracked})
 files=list(Path(__file__).parent.glob('*.py'))+[SOURCE,OUT/'PROTOCOL.md',OLD/'seal.json',OLD/'predictions.json',OLD/'scores.csv',OLD/'fits.json',OLD/'publication_audit.json']
 save(OUT/'seal.json',dict(at=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files={str(p):sha(p) for p in files},jobs=old['jobs'],predictions=predictions,planned_neural_fits=0,planned_cpu_calibrators=7,planned_scale_settings=42,planned_scores=448,prior_control='FIXED120 only; original selected policies unchanged'))
 print('PREPARED: 7 CPU calibrators, 42 scale settings, zero model inference')

def verified():
 s=read(OUT/'seal.json')
 for p,h in s['files'].items():assert sha(p)==h,p
 for p in s['predictions']:assert sha(ROOT/p['path'])==p['sha256']
 for j in s['jobs']:
  for k in ['input','target']:assert sha(ROOT/j[k])==j[k+'_sha256']
 return s

def groups():return [('INIT0',-1)]+[(a,k) for a in ['STD','EXODROP','VINTAGE'] for k in [61710,61711]]
def prediction(s,arm,seed,j,path):
 if j['split']=='V':label='STD_61710_CP0' if arm=='INIT0' else f'{arm}_{seed}_CP120'
 else:label='INIT0' if arm=='INIT0' else f'{arm}_{seed}_EVAL120'
 r=next(r for r in s['predictions'] if r['id']==j['id'] and r['path_index']==path and r['label']==label)
 with np.load(ROOT/r['path']) as x:q=x['q'].copy();np.testing.assert_array_equal(q,np.sort(x['raw'],axis=0))
 return q,r

def analyze():
 s=verified();assert not (OUT/'calibration.json').exists(),'Existing calibrated attempt; no overwrite'
 t0=time.perf_counter();v=[j for j in s['jobs'] if j['split']=='V'];d=[j for j in s['jobs'] if j['split']=='D'];assert len(v)==4 and len(d)==16
 calibrations=[]
 for arm,seed in groups():
  qs=[prediction(s,arm,seed,j,0)[0] for j in v];ys=[np.load(ROOT/j['target']) for j in v];sd=[j['std'] for j in v]
  c=ref.calibrate(qs,ys,sd);calibrations.append(dict(arm=arm,seed=seed,**c))
 save(OUT/'calibration.json',dict(at=time.time(),records=calibrations,source='V4, k0 only; D arrays not passed to fitting'))
 rows=[];manifest=[]
 for c in calibrations:
  for j in d:
   y=np.load(ROOT/j['target'])
   for path in [0,3]:
    q,source=prediction(s,c['arm'],c['seed'],j,path);cal=ref.apply(q,j['std'],c)
    p=CACHE/f"{c['arm']}_{c['seed']}_{j['id']}_k{path}.npz";assert not p.exists();np.savez_compressed(p,RAW=q,CAL=cal)
    manifest.append(dict(arm=c['arm'],seed=c['seed'],id=j['id'],path_index=path,path=str(p.relative_to(ROOT)),sha256=sha(p),source_path=source['path'],source_sha256=source['sha256']))
    for variant,z in [('RAW',q),('CAL',cal)]:
     error=(z[10]-y)/j['std'];bias=float(error.mean());centered=float(np.mean((error-bias)**2));mse=float(np.mean(error**2))
     assert abs(mse-bias*bias-centered)<=1e-10*max(1,mse)
     rows.append(dict(arm=c['arm'],seed=c['seed'],variant=variant,id=j['id'],month=j['month'],path_index=path,prediction=str(p.relative_to(ROOT)),**ref.metrics(z,y,j['std']),median_scaled_MSE=mse,median_scaled_bias_squared=bias*bias,median_scaled_centered_MSE=centered))
 save(OUT/'predictions.json',manifest);pd.DataFrame(rows).to_csv(OUT/'scores.csv',index=False)
 save(OUT/'status.json',dict(status='COMPLETE',neural_fits=0,updates=0,gpu_inference=0,cpu_calibrators=7,scale_settings=42,validation_grid_scores=168,D_scores=len(rows),transformed_forecasts=224,seconds=time.perf_counter()-t0,errors=[],calibration_sha256=sha(OUT/'calibration.json')))
 print('ANALYSIS COMPLETE',len(rows),'score rows')

def verify_and_report():
 s=verified();state=read(OUT/'status.json');assert state['status']=='COMPLETE'
 cal=read(OUT/'calibration.json');assert sha(OUT/'calibration.json')==state['calibration_sha256'];df=pd.read_csv(OUT/'scores.csv');manifest=read(OUT/'predictions.json');jobs={j['id']:j for j in s['jobs']}
 assert len(df)==len(df.drop_duplicates(['arm','seed','variant','id','path_index']))==448
 old=pd.read_csv(OLD/'scores.csv');maxerr=0.;oldchecks=0;gridchecks=0;transformchecks=0
 for c in cal['records']:
  v=[j for j in s['jobs'] if j['split']=='V'];res=[]
  for j in v:
   q,_=prediction(s,c['arm'],c['seed'],j,0);res.extend(((np.load(ROOT/j['target'])-q[10])/j['std']).tolist())
  ordered=sorted(res);b=(ordered[47]+ordered[48])/2;assert abs(b-c['b'])<=1e-12
  candidates=[]
  for grid in c['grid']:
   scores=[]
   for j in v:
    q,_=prediction(s,c['arm'],c['seed'],j,0);z=np.array([[q[10,h]+b*j['std']+grid['s']*(q[i,h]-q[10,h]) for h in range(24)] for i in range(21)])
    scores.append(ref.scalar_primary(z,np.load(ROOT/j['target']),j['std']));gridchecks+=1
   value=math.fsum(scores)/4;assert math.isclose(value,grid['primary'],rel_tol=1e-10,abs_tol=1e-10);candidates.append((value,grid['s']))
  assert min(candidates)[1]==c['s']
 # All old results and transformed forecasts, independent scalar primary.
 for r in df.to_dict('records'):
  j=jobs[r['id']];c=next(c for c in cal['records'] if c['arm']==r['arm'] and c['seed']==r['seed'])
  with np.load(ROOT/r['prediction']) as p:z=p[r['variant']];q=p['RAW']
  assert np.isfinite(z).all() and (np.diff(z,axis=0)>=0).all()
  scalar=ref.scalar_primary(z,np.load(ROOT/j['target']),j['std']);error=abs(scalar-r['primary']);maxerr=max(maxerr,error);assert math.isclose(scalar,r['primary'],abs_tol=1e-10,rel_tol=1e-10)
  if r['variant']=='CAL':
   expected=np.array([[q[10,h]+c['b']*j['std']+c['s']*(q[i,h]-q[10,h]) for h in range(24)] for i in range(21)])
   np.testing.assert_allclose(z,expected,rtol=1e-12,atol=1e-10);np.testing.assert_allclose(z[18]-z[2],c['s']*(q[18]-q[2]),rtol=1e-12,atol=1e-9);transformchecks+=1
  else:
   od=old[(old.arm==r['arm'])&(old.seed==r['seed'])&(old.id==r['id'])&(old.path_index==r['path_index'])]
   if r['arm']!='INIT0':od=od[od.source=='FIXED120']
   assert len(od)==1
   for key in ['primary','raw_2pinball','raw_RMSE','raw_MAE','coverage80','width80']:assert math.isclose(r[key],float(od.iloc[0][key]),rel_tol=1e-10,abs_tol=1e-10)
   oldchecks+=1
 for r in manifest:
  assert sha(ROOT/r['path'])==r['sha256'] and sha(ROOT/r['source_path'])==r['source_sha256']
  assert (ROOT/r['path']).stat().st_mtime>=cal['at']
 for p,h in read(OUT/'historical_hashes.json').items():assert sha(ROOT/p)==h,p
 # Fit boundary takes only V; changing a D-label mapping leaves every V argument identical.
 ymap={j['id']:np.load(ROOT/j['target']) for j in s['jobs'] if j['split']!='TRAIN'};poison={k:(np.full_like(y,123456789.) if jobs[k]['split']=='D' else y) for k,y in ymap.items()}
 for c in cal['records']:
  v=[j for j in s['jobs'] if j['split']=='V'];qs=[prediction(s,c['arm'],c['seed'],j,0)[0] for j in v];ds=[j['std'] for j in v]
  fitted=ref.calibrate(qs,[poison[j['id']] for j in v],ds);assert fitted=={k:c[k] for k in ['b','s','grid']}
 verify=dict(status='PASS',raw_score_replays=oldchecks,scalar_primary_checks=len(df),max_primary_error=maxerr,grid_scalar_checks=gridchecks,transform_checks=transformchecks,prediction_hashes=len(manifest),D_poison_fit_checks=7,historical_files_preserved=len(read(OUT/'historical_hashes.json')),neural_fits=0,gpu_inference=0)
 save(OUT/'verification.json',verify)
 fields=['primary','raw_2pinball','raw_RMSE','raw_MAE','coverage80','width80','median_scaled_MSE','median_scaled_bias_squared','median_scaled_centered_MSE']
 df.groupby(['arm','seed','variant','path_index'],as_index=False)[fields].mean().to_csv(OUT/'macro_scores.csv',index=False)
 df.groupby(['arm','variant','path_index','month'],as_index=False)[fields].mean().to_csv(OUT/'monthly_scores.csv',index=False)
 comparisons=[('VINTAGE','STD','RAW','RAW'),('VINTAGE','STD','CAL','CAL'),('VINTAGE','EXODROP','CAL','CAL'),('VINTAGE','INIT0','CAL','CAL'),('STD','VINTAGE','CAL','RAW')];effects=[]
 for candidate,base,cv,bv in comparisons:
  for path in [0,3]:
   for seed in [-1,61710,61711]:
    def values(arm,var):
     d=df[(df.arm==arm)&(df.variant==var)&(df.path_index==path)]
     if seed!=-1 and arm!='INIT0':d=d[d.seed==seed]
     return d.groupby('month').primary.mean().reindex([6,7,8,9]).to_numpy()
    x=values(candidate,cv);b=values(base,bv);ix=np.random.default_rng(61712).integers(0,4,(2000,4));bb=b[ix].mean(1);xx=x[ix].mean(1);boot=100*(bb-xx)/bb
    effects.append(dict(candidate=candidate,baseline=base,candidate_variant=cv,baseline_variant=bv,path_index=path,seed=seed,candidate_primary=x.mean(),baseline_primary=b.mean(),gain_percent=100*(b.mean()-x.mean())/b.mean(),positive_months=int((x<b).sum()),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
 pd.DataFrame(effects).to_csv(OUT/'effects.csv',index=False)
 lines=['# 과거 예보 LoRA 이득은 단순 출력 보정으로 대체되는가?','',
 '[확인] 기존 6-fit 예측의 후속 분석을 완료했다. 신규 neural fits/updates/GPU 추론은 모두0이다. 같은 V4로 편향·분위수 폭 보정기7개(각6개 scale 후보, 총42설정)를 구하고, 이미 노출된 D16의 최신k0/과거k3에서 RAW/CAL448행을 비교했다. 새방법 PASS·독립확증이 아니다.','',
 '## 고정한 방법과 정보','',
 '기준 e9c9859. 모든 학습군의 FIXED120을 사용하며 원래 V 선택결과는 변경하지 않는다. INIT0도 같은 보정 기회를 갖는다. b는 V의 normalized median residual, s는 기존6값 grid의 V primary 최소다. q′=q50+b·context_std+s·(q−q50). b를 primary에 대해 최적화한 것은 아니며 이 보정이 가능한 모든 단순 보정의 최선이라는 주장도 아니다. 모든 계수를 먼저 봉인한 뒤 D를 계산했다. 단, D는 이미 연구에 노출된 자료다.','',
 '| 군 | seed | b | s |','| --- | ---: | ---: | ---: |']
 for c in cal['records']:lines.append(f"| {c['arm']} | {c['seed']} | {c['b']:.6f} | {c['s']} |")
 lines+=['','## 원점수 — 낮을수록 좋음','', '| 보정 | 입력 | INIT0 | STD | EXODROP | VINTAGE |','| --- | --- | ---: | ---: | ---: | ---: |']
 for variant in ['RAW','CAL']:
  for path in [0,3]:
   vals=[df[(df.arm==a)&(df.variant==variant)&(df.path_index==path)].primary.mean() for a in ['INIT0','STD','EXODROP','VINTAGE']]
   lines.append(f'| {variant} | k{path} | '+' | '.join(f'{v:.9f}' for v in vals)+' |')
 lines+=['','## 사전에 정한 비교','', '| 비교 | 입력 | 개선율 % | 개선월/4 | 월bootstrap95% % |','| --- | --- | ---: | ---: | --- |']
 for e in effects:
  if e['seed']==-1:lines.append(f"| {e['candidate']}_{e['candidate_variant']} vs {e['baseline']}_{e['baseline_variant']} | k{e['path_index']} | {e['gain_percent']:.4f} | {e['positive_months']} | [{e['ci_low']:.4f}, {e['ci_high']:.4f}] |")
 lines+=['','양의 개선율은 첫 방법의 오차 감소다. 두seed별 효과는 [effects.csv](effects.csv), 모든 raw/보정 지표는 [원점수](scores.csv)와 [macro](macro_scores.csv)에 공개한다. 월block2000회(seed61712), 네월과 한타깃의 기술적 구간이다. 부호에 따라 데이터를 교체하거나 새기준 PASS를 부여하지 않는다.','',
 '## 검증·비용·미실행','',
 f"원본224행 재생, scalar primary448회(max error {maxerr:.3g}), V grid168회, affine변환224개, D label poison 7개와 예측hash224개를 검사했다. 기존 결과 {verify['historical_files_preserved']}개가 변하지 않았다. [검증](verification.json).", 
 f"주 분석 CPU wall {state['seconds']:.3f}초(독립검산·문헌 검토 제외). 신경망학습0, GPU추론0, 통계보정7개. 계획된 점수의 미실행0, 추가학습·새날짜·새타깃은 없음.",
 'MSE를 bias²+중심화오차분산으로 분해한 값은 사후 설명 통계다. 정답을 알고 평균오차를 빼는 예측기를 배포했다고 주장하지 않는다. 보정계수·변환은 모두 V에서 구한다. 원자료와 예측cache는 로컬이며 원격에는 코드·해시·점수를 올린다.','',
 '## 연구 판단','',
 'EXECUTION=COMPLETE, NOVELTY=KNOWN_POSTPROCESSING_CONTROL. 성능 해석과 선행 검토는 [INTERPRETATION.md](INTERPRETATION.md)에 별도로 기록한다. 이번 결과를 근거로 원래 학습실험의 체크포인트나 판정을 소급 변경하지 않는다.','']
 (OUT/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(verify,indent=2));print(pd.DataFrame(effects).query('seed == -1').to_string(index=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','analyze','verify']);a=p.parse_args()
 {'prepare':prepare,'analyze':analyze,'verify':verify_and_report}[a.action]()
