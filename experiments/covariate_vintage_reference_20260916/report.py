"""C-only postprocessing selection, then D scoring and independent validation."""
import csv,time,math
from common import *
from run import verify_seal

def writecsv(p,rows):
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def scalar_cdf(x,a):
 values=[float(a[0])]+list(map(float,a))+[float(a[-1])];ps=[0.]+list(TAUS)+[1.];total=0.
 for j in range(len(values)-1):
  left,right=values[j:j+2];mass=ps[j+1]-ps[j]
  if right<=x:total+=mass
  elif left<=x and right>left:total+=mass*(x-left)/(right-left)
 return total

def main():
 seal=verify_seal();status=read(OUT/'status.json');assert status['status']=='COMPLETE';assert not (OUT/'calibration.json').exists(),'Do not reselect calibration'
 cache={};mix_seconds=0.;mix_checks=0;raw_crossings=0
 predrecords=read(OUT/'predictions.json');assert len(predrecords)==28
 for j,p in zip(seal['jobs'],predrecords):
  assert j['id']==p['id'] and sha(ROOT/p['path'])==p['sha256']
  with np.load(ROOT/p['path']) as raw:
   for name in raw.files:raw_crossings+=int(np.sum(np.diff(raw[name],axis=1)<0))
   paths=np.sort(raw['PATHS'],axis=1);t0=time.perf_counter();mix=mixture(paths);mix_seconds+=time.perf_counter()-t0
   for h in range(24):
    for qi,t in enumerate(TAUS):
     v=mix[qi,h];epsilon=max(1.,float(np.max(abs(paths[:,:,h]))))*1e-9
     at=sum(scalar_cdf(v,a[:,h]) for a in paths)/4
     below=sum(scalar_cdf(v-epsilon,a[:,h]) for a in paths)/4
     assert at>=t-1e-8 and below<=t+1e-8,('MIXTURE_CDF_CHECK',j['id'],h,qi,at,below)
     mix_checks+=1
   cache[j['id']]={name:np.sort(raw[name][0],axis=0) for name in ['F0','PAST','MEAN_INPUT']}
   cache[j['id']].update(LATEST=paths[0],MIXTURE=mix)
  with np.load(ROOT/j['input']) as x:
   cache[j['id']]['SEASONAL24_POINT']=x['target'][-24:].astype(float)
   cache[j['id']]['SEASONAL168_POINT']=x['target'][-168:-144].astype(float)
 calibration_jobs=[j for j in seal['jobs'] if j['role']=='C'];cal={};t0=time.perf_counter()
 ys=[np.load(ROOT/j['target']) for j in calibration_jobs];stds=[j['std'] for j in calibration_jobs]
 for name in BASE:cal[name]=calibrate([cache[j['id']][name] for j in calibration_jobs],ys,stds)
 for lag in [24,168]:
  name=f'SEASONAL{lag}';res=np.concatenate([(y-cache[j['id']][name+'_POINT'])/j['std'] for j,y in zip(calibration_jobs,ys)])
  cal[name]=dict(residual_quantiles=np.quantile(res,TAUS,method='linear').tolist(),calibration_samples=len(res))
 save(OUT/'calibration.json',dict(at=time.time(),selection_role='C',neural_fits=0,statistical_postprocessors=7,width_grid_evaluations=30,methods=cal))
 calhash=sha(OUT/'calibration.json');postprocessing_seconds=time.perf_counter()-t0
 # D labels are read for scoring only after the immutable C-only calibration file exists.
 scores=[];checks=[];finalmanifest=[]
 for j in seal['jobs']:
  y=np.load(ROOT/j['target']);pred={k:cache[j['id']][k] for k in BASE}
  for name in BASE:pred[name+'_CAL']=apply(pred[name],j['std'],cal[name])
  for lag in [24,168]:
   name=f'SEASONAL{lag}';pred[name]=cache[j['id']][name+'_POINT'][None]+np.array(cal[name]['residual_quantiles'])[:,None]*j['std']
  for name,q in pred.items():
   assert np.isfinite(q).all() and np.all(np.diff(q,axis=0)>=-1e-7)
   m=metrics(q,y,j['std']);sc=scalar_primary(q,y,j['std']);assert math.isclose(m['primary'],sc,rel_tol=1e-10,abs_tol=1e-10)
   scores.append(dict(id=j['id'],role=j['role'],month=j['month'],method=name,**m))
   checks.append(abs(sc-m['primary']))
  path=CACHE/f"{j['id']}_final.npz";np.savez_compressed(path,**pred)
  finalmanifest.append(dict(id=j['id'],path=str(path.relative_to(ROOT)),sha256=sha(path)))
 writecsv(OUT/'scores.csv',scores);save(OUT/'final_predictions.json',finalmanifest)
 methods=list(pred);macro=[];monthly=[]
 for role in ['C','D']:
  for name in methods:
   rows=[r for r in scores if r['role']==role and r['method']==name];macro.append(dict(role=role,method=name,n=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in metrics(np.zeros((21,24)),np.zeros(24),1)}))
 for month in [6,7,8,9]:
  for name in methods:
   rows=[r for r in scores if r['month']==month and r['method']==name];monthly.append(dict(month=month,method=name,primary=float(np.mean([r['primary'] for r in rows]))))
 writecsv(OUT/'macro.csv',macro);writecsv(OUT/'monthly.csv',monthly)
 dev={r['method']:r for r in macro if r['role']=='D'};monthlymap={(r['method'],r['month']):r['primary'] for r in monthly};effects=[]
 rng=np.random.default_rng(61701);samples=rng.integers(0,4,size=(2000,4));candidate='MIXTURE_CAL'
 for name in methods:
  if name==candidate:continue
  b=np.array([monthlymap[name,k] for k in [6,7,8,9]]);c=np.array([monthlymap[candidate,k] for k in [6,7,8,9]])
  gains=100*(b[samples].mean(1)-c[samples].mean(1))/b[samples].mean(1)
  effects.append(dict(baseline=name,baseline_primary=dev[name]['primary'],candidate_primary=dev[candidate]['primary'],gain_percent=100*(dev[name]['primary']-dev[candidate]['primary'])/dev[name]['primary'],positive_months=int(np.sum(b>c)),ci_low=float(np.quantile(gains,.025)),ci_high=float(np.quantile(gains,.975))))
 writecsv(OUT/'effects.csv',effects)
 eligible=[r for r in effects if r['baseline']!='MIXTURE'];strongest=min(eligible,key=lambda x:x['baseline_primary'])
 signal=all(r['gain_percent']>=2 and r['positive_months']>=3 for r in eligible)
 evidence='TEACHER_SIGNAL_TO_INVESTIGATE' if signal else 'POSITIVE_BUT_UNCERTAIN' if all(r['gain_percent']>0 for r in eligible) else 'NO_ADDED_VALUE_IN_THIS_REFERENCE'
 calls=read(OUT/'calls.json');timing={}
 for name in ['LATEST','PATHS']:
  rows=[r for r in calls if r['label'].startswith('TIMING/'+name+'/') and not r['label'].endswith('/0')]
  assert len(rows)==5
  timing[name]=dict(n=5,median_seconds=float(np.median([r['seconds'] for r in rows])),min_seconds=min(r['seconds'] for r in rows),max_seconds=max(r['seconds'] for r in rows),peak_allocated=max(r['peak_allocated'] for r in rows),peak_reserved=max(r['peak_reserved'] for r in rows),contaminated=any(r['contaminated'] for r in rows))
 old=read(OUT/'historical_hashes.json');assert all(sha(ROOT/p)==h for p,h in old.items());assert sha(OUT/'calibration.json')==calhash
 assert status['target_forecasts']==233 and status['pipeline_calls']==128 and status['primitive_forwards']==128
 verification=dict(status='COMPLETE',neural_fits=0,optimizer_updates=0,target_forecasts=233,pipeline_calls=128,primitive_forwards=128,statistical_postprocessors=7,width_grid_evaluations=30,score_rows=len(scores),independent_scalar_max_error=max(checks),mixture_cdf_checks=mix_checks,raw_quantile_crossings=raw_crossings,sorted_for_all_methods=True,calibration_sha256=calhash,selection_before_D_scores=True,frozen_unchanged=status['frozen_unchanged'],buffers_unchanged=status['buffers_unchanged'],replay=read(OUT/'replay.json'),historical_files_preserved=len(old),timing=timing,mixture_cpu_seconds=mix_seconds,postprocessor_selection_seconds=postprocessing_seconds)
 save(OUT/'verification.json',verification)
 decision=dict(execution='COMPLETE',predictive_evidence=evidence,novelty='KNOWN_VINTAGE_MIXTURE_REFERENCE_NOT_NEW_PEFT',topic='NOT_CONFIRMED',strongest_baseline=strongest,teacher_signal=signal,student_fits=0,scope='ONE_TARGET_SIMULATED_AS_OF_DISCOVERY',automatic_student_training=False)
 save(OUT/'decision.json',decision)
 wall=read(OUT/'predict_wall.json');peak=max(r['peak_allocated'] for r in calls)/2**20
 lines=['# 예보 vintage 참조 — 실제 예측 결과','',f"[확인] 판정은 **{evidence}**다. 새 PEFT 주제는 아직 확보하지 못했다. 신경망 학습 0fits·업데이트0, Chronos-2 target forecast **233개 / pipeline 128호출**을 완료했다. 동일 모델의 통계 보정7건과 폭 후보 비교30개를 별도로 수행했다.",'','## 질문과 고정 범위','',
 '네 개의 적법한 과거 예보 버전을 여러 미래 경로로 처리하는 것이 최신 경로 및 같은 보정 기회의 단순 대조보다 나은지 확인했다. OS Gorredijk 한 타깃의 C12원점(3~5월)에서 보정을 고정하고 D16원점(6~9월)을 평가했다. 10~12월은 점수 계산에 쓰지 않았다. 데이터 가용성 점검 이후 고정한 탐색 구간이며 독립 외부 검증으로 주장하지 않는다. [사전 프로토콜](PROTOCOL.md), [입력·코드·모델 봉인](seal.json).','',
 '기상 경로는 원점에 가용한 최신 네 행을 각 valid time에서 모은 것으로, 같은 발행 cycle의 확률적 ensemble이 아니다. 실제 기상 정답을 입력하거나 보정에 쓰지 않았다. 과거 기상도 versioned 파일의 모사된 가용시각으로 선택했다. 부하 및 기상 미래 행의 오염에 28개 원점 입력이 모두 불변이었다. 시간별 평균은 정확히 네 15분 값으로 만들었고 결측 때문에 원점을 교체하지 않았다.','',
 'Chronos-2의 알려진 공변량 입력 기능을 사용했다. CDF 혼합과 편향·폭 보정도 알려진 참조 연산이며, 이번에 새 PEFT 모듈을 구현하거나 학습하지 않았다. [공식 Chronos 코드](https://github.com/amazon-science/chronos-forecasting).','',
 '## 개발 원점수','',
 'Primary는 21개 native 분위수의 평균2-pinball / 각 원점의 과거336h 표준편차다. 낮을수록 좋다. 연속 CRPS 정확값이 아니다. 모두 같은 원점·target·scale로 비교했다. CAL은 C만으로 추정한 편향과 폭이다. 계절 대조는 C잔차 분위수를 더했다. 원래 분위수 crossing을 정렬하는 정책도 모두 동일하다.','',
 '| 방법 | Primary | raw 2-pinball | median RMSE | 80% 포함률 |','|---|---:|---:|---:|---:|']
 for name in methods:
  r=dev[name];lines.append(f"| {name} | {r['primary']:.9f} | {r['raw_2pinball']:.4f} | {r['raw_RMSE']:.4f} | {r['coverage80']:.4f} |")
 lines += ['',f"MIXTURE_CAL 대비 가장 낮은 비-mixture 대조는 **{strongest['baseline']}**다. 후보 상대 개선율은 **{strongest['gain_percent']:.3f}%**, 개선한 달은 **{strongest['positive_months']}/4**다. 월 단위 탐색적 bootstrap95% 구간은 [{strongest['ci_low']:.3f}, {strongest['ci_high']:.3f}]%다. 네 달뿐이므로 정밀한 불확실성 추정이나 모집단 유의성으로 해석하지 않는다. [전체 원점수](scores.csv), [모든 대조 효과](effects.csv), [월별](monthly.csv), [C 보정 기록](calibration.json).",'',
 'MIXTURE는 분포의 CDF를 혼합한 뒤 분위수를 역산했다. 각 native 분위수 함수를 선형 보간하고 양끝은 고정한 유계 분포라는 가정이 있다. 단순 분위수 평균과 다르다. 여러 경로를 처리해도 부하의 joint trajectory distribution을 얻었다고 주장하지 않는다.','',
 '## 비용과 검산','',
 f"GPU controller {wall['seconds']:.2f}초(시작 안정 대기 포함), 최대 allocated {peak:.2f}MiB, 최저 여유 {wall['minimum_free_mib']}MiB다. 실행 원장에 외부 compute 및 오염 여부를 기록했다. 아래 시간은 동일 batch_size16, 첫 C원점에서 warmup1회 뒤5회 측정한 pipeline 호출이다. 4PATHS는 네 경로를 배치로 처리한다.",'',
 '| 처리 | 중간값 ms | 최대 allocated MiB |','|---|---:|---:|']
 for name in ['LATEST','PATHS']:r=timing[name];lines.append(f"| {name} | {r['median_seconds']*1000:.3f} | {r['peak_allocated']/2**20:.2f} |")
 lines += ['',f"CDF 혼합 CPU 총 {mix_seconds:.3f}초/28원점, 보정 설정 선택 {postprocessing_seconds:.3f}초는 별도다. 전처리·입출력·모델 적재를 전부 포함한 서비스 latency가 아니며, 학생을 실행하지 않았으므로 PEFT 자원 이득은 미평가다. [호출 원장](calls.json), [GPU controller](predict_wall.json).",'',
 f"모든 {len(scores)}개 원점/방법 점수를 독립 scalar float64로 확인했고 최대 오차는 {max(checks):.3g}다. 혼합 분위수 {mix_checks}개의 CDF 조건을 별도 스칼라 코드로 점검했다. 첫 C원점 7개 예측을 같은 shape로 재생했고 검산 결과는 [verification.json](verification.json)에 있다. 파라미터·버퍼 hash와 기존 {len(old)}개 결과 파일 hash도 그대로다.",'',
 '## 다음 결정과 남은 한계','',
 f"이번 사전 teacher 신호 조건의 충족 여부는 {signal}다. 이 결과만으로 모든 불확실성 처리 방법을 기각하거나 새 PEFT의 성공을 선언하지 않는다. 동일 가중의 나이 다른 예보 경로는 보정된 미래 입력 분포가 아니며, 한 지역의 여름 개발 구간이다. 일반 LoRA·동일 정보와 감독의 증류·학생 PEFT·독립 원천 검증은 미실행이다. 원점수 확인 뒤 입력·기간·경로수·문턱을 바꾸지 않았다.",'',
 '예보 버전 자체의 가치와, 그 가치를 학생으로 옮기는 새 적응 규칙의 가치는 별도다. 강한 단순 대조가 이 참조를 대체한다면 해당 참조를 teacher로 삼아 학생부터 학습할 근거가 부족하다. 개선이 확인돼도 알려진 ensemble distillation 이상의 차별점을 따로 입증해야 한다. [최종 상태](decision.json). 자동 학생 학습은 실행하지 않았다.','',
 '## 재현','',
 '코드·provenance·원점수·검증은 Git에 보관하고 입력/target/예측 npz·모델 가중치는 로컬 캐시에 둔다. 준비/예측은 같은 run을 덮어쓰지 않으며 기존 완료 결과를 다시 학습하지 않는다.','',
 '```bash','scripts/with_cuda.sh .venv/bin/python experiments/covariate_vintage_reference_20260916/run.py prepare','scripts/with_cuda.sh .venv/bin/python experiments/covariate_vintage_reference_20260916/run.py predict','scripts/with_cuda.sh .venv/bin/python experiments/covariate_vintage_reference_20260916/report.py','```','']
 (OUT/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':main()
