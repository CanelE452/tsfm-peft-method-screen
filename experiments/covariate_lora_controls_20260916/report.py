"""Independent CPU recomputation and descriptive comparisons; no model selection."""
import csv,subprocess,math
import pandas as pd
from common import *


def independent(q,y,sd):
 # Explicit scalar implementation, independent of common.score.
 values=[]
 for i,t in enumerate(QUANTILES):
  for h in range(24):
   residual=float(y[h])-float(q[i,h]);values.append(2*(t*residual if residual>=0 else (t-1)*residual))
 pin=math.fsum(values)/504
 median=QUANTILES.index(.5);lo=QUANTILES.index(.1);hi=QUANTILES.index(.9)
 squared=[(float(q[median,h])-float(y[h]))**2 for h in range(24)]
 absolute=[abs(float(q[median,h])-float(y[h])) for h in range(24)]
 return dict(primary=pin/sd,raw_2pinball=pin,raw_RMSE=math.sqrt(math.fsum(squared)/24),raw_MAE=math.fsum(absolute)/24,coverage80=sum(float(q[lo,h])<=float(y[h])<=float(q[hi,h]) for h in range(24))/24,width80=math.fsum(float(q[hi,h])-float(q[lo,h]) for h in range(24))/24)


def main():
 s=read(OUT/'seal.json');state=read(OUT/'status.json');assert state['status']=='COMPLETE',state
 jobs={j['id']:j for j in s['jobs']};fits=read(OUT/'fits.json');selection=read(OUT/'selection.json');pre=read(OUT/'preflight.json')
 assert sha(OUT/'fits.json')==selection['fits_sha256'] and sha(OUT/'selection.json')==state['selection_sha256']
 for p,h in {**s['files'],**s['model_files']}.items():assert sha(p)==h,p
 for j in jobs.values():
  for key in ['input','target']:assert sha(ROOT/j[key])==j[key+'_sha256']
 for p,h in read(OUT/'historical_hashes.json').items():assert sha(ROOT/p)==h,p
 predictions=read(OUT/'predictions.json');by_path={r['path']:r for r in predictions}
 for p,r in by_path.items():assert sha(ROOT/p)==r['sha256']
 df=pd.read_csv(OUT/'scores.csv');assert len(df)==416;errors=[];vchecks=0
 for r in df.to_dict('records'):
  assert r['prediction'] in by_path;j=jobs[r['id']]
  with np.load(ROOT/r['prediction']) as x:
   q=x['q'];np.testing.assert_array_equal(q,np.sort(x['raw'],axis=0));values=independent(q,np.load(ROOT/j['target']),j['std'])
  for k,v in values.items():
   e=abs(r[k]-v);assert math.isclose(r[k],v,rel_tol=1e-10,abs_tol=1e-10),(r,k,v);errors.append(e)
 assert len(fits)==6 and sum(f['updates'] for f in fits)==state['updates']==720
 schedule=read(OUT/'schedule.json');assert len(schedule)==720
 for f,sel in zip(fits,selection['selections']):
  assert f['status']=='COMPLETE' and f['updates']==120 and len(f['curve'])==120
  assert f['trainable_parameters']==147456 and f['trainable_tensors']==192
  assert f['frozen_unchanged'] and f['buffers_unchanged'] and f['changed_tensors']>0
  stream=[r for r in schedule if r['arm']==f['arm'] and r['seed']==f['seed']]
  assert [(r['step'],r['id'],r['path_index']) for r in f['curve']]==[(r['step'],r['id'],r['path']) for r in stream]
  for epoch in range(15):assert sorted(r['id'] for r in stream if r['epoch']==epoch)==sorted(j['id'] for j in jobs.values() if j['split']=='TRAIN')
  if f['arm']=='VINTAGE':
   for jid in set(r['id'] for r in stream):assert [sum(r['id']==jid and r['path']==k for r in stream) for k in range(4)]==[4,4,4,3]
  for cp in f['checkpoints']:
   assert sha(ROOT/cp['path'])==cp['sha256'];vv=[]
   for j in jobs.values():
    if j['split']=='V':
     path=CACHE/f"{f['id']}_CP{cp['step']}_{j['id']}_k0.npz"
     with np.load(path) as x:vv.append(independent(x['q'],np.load(ROOT/j['target']),j['std'])['primary'])
     vchecks+=1
   assert math.isclose(np.mean(vv),cp['validation_primary'],rel_tol=1e-10,abs_tol=1e-10)
  assert min(f['checkpoints'],key=lambda c:(c['validation_primary'],c['step']))==sel['checkpoint']
 for seed in SEEDS:assert len({f['initial_hash'] for f in fits if f['seed']==seed})==1
 replay=read(OUT/'replay.json');assert len(replay)==24 and max(r['normalized_max_abs'] for r in replay)<=1e-5
 assert len(read(OUT/'smoke.json'))==3 and state['smoke_updates']==6
 assert state['training_forwards']==726 and state['primitive_forwards']==state['training_forwards']+state['inference_forwards']
 # Independent TRAIN-only scaling recomputation and augmentation feature checks.
 sc=read(OUT/'scaling.json');train=[j for j in jobs.values() if j['split']=='TRAIN'];arrays=[]
 for j in train:
  with np.load(ROOT/j['input']) as x:arrays.append(x['past'].astype(float))
 a=np.concatenate(arrays,axis=1);np.testing.assert_array_equal(a.mean(1),sc['mean']);np.testing.assert_array_equal(np.maximum(a.std(1),1e-6),sc['std'])
 for r in schedule:
  path,mask=augmentation(r['arm'],r['seed'],r['step']-1);assert path==r['path']
  if mask is not None:np.testing.assert_array_equal(mask,r['mask'])
 verification=dict(status='PASS',D_rows=416,independently_recomputed_metrics=len(errors),max_absolute_error=max(errors),validation_prediction_checks=vchecks,checkpoint_hashes=24,prediction_hashes=len(predictions),replays=24,replays_exact=sum(r['exact'] for r in replay),matched_initialization=True,matched_origin_order=True,train_only_scaling=True,vintage_exposure_counts=[4,4,4,3],historical_files_preserved=len(read(OUT/'historical_hashes.json')),actual_fits=6,actual_updates=720,smoke_updates=6)
 save(OUT/'verification.json',verification)
 metrics=list(independent(np.zeros((21,24)),np.ones(24),1))
 macro=df.groupby(['arm','seed','source','path_index'],as_index=False)[metrics].mean();macro.to_csv(OUT/'macro_scores.csv',index=False)
 monthly=df.groupby(['arm','source','path_index','month'],as_index=False)[metrics].mean();monthly.to_csv(OUT/'monthly_scores.csv',index=False)
 effects=[]
 for source in ['SELECTED','FIXED120']:
  for path in [0,3]:
   for candidate,base in [('STD','INIT0'),('EXODROP','STD'),('VINTAGE','STD'),('EXODROP','INIT0'),('VINTAGE','INIT0')]:
    for seed in [-1]+SEEDS:
     def series(arm):
      d=df[(df.arm==arm)&(df.path_index==path)]
      if arm!='INIT0':d=d[(d.source==source)&((d.seed==seed) if seed!=-1 else True)]
      return d.groupby('id').primary.mean().sort_index()
     x=series(candidate);b=series(base);assert x.index.equals(b.index)
     xm=np.array([x[[i for i in x.index if jobs[i]['month']==m]].mean() for m in [6,7,8,9]])
     bm=np.array([b[[i for i in b.index if jobs[i]['month']==m]].mean() for m in [6,7,8,9]])
     sample=np.random.default_rng(61712).integers(0,4,size=(2000,4));bx=bm[sample].mean(1);xx=xm[sample].mean(1);boot=100*(bx-xx)/bx
     effects.append(dict(source=source,path_index=path,seed=seed,candidate=candidate,baseline=base,candidate_primary=float(x.mean()),baseline_primary=float(b.mean()),gain_percent=float(100*(b.mean()-x.mean())/b.mean()),positive_months=int((xm<bm).sum()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
 csvwrite(OUT/'effects.csv',effects)
 # Difference from old inference is documented, never substituted for matched INIT0.
 ref=ROOT/'results/covariate_vintage_reference_20260916';oldp={r['id']:r for r in read(ref/'predictions.json')};diff=[]
 for r in df[df.arm=='INIT0'].to_dict('records'):
  j=jobs[r['id']];orig=oldp[r['id']];assert sha(ROOT/orig['path'])==orig['sha256']
  with np.load(ROOT/orig['path']) as x:old=np.sort(x['PATHS'][r['path_index']],axis=0)
  with np.load(ROOT/r['prediction']) as x:new=x['q']
  diff.append(dict(id=j['id'],path_index=r['path_index'],normalized_max_abs=float(np.max(abs(new-old))/j['std']),old_primary=independent(old,np.load(ROOT/j['target']),j['std'])['primary'],new_primary=r['primary']))
 csvwrite(OUT/'initial_reference_comparison.csv',diff)
 resources=[dict(arm=f['arm'],seed=f['seed'],trainable_parameters=f['trainable_parameters'],full_trajectory_seconds=f['seconds'],optimizer_seconds=f['optimizer_seconds'],peak_allocated_MiB=f['peak_allocated_bytes']/2**20,peak_reserved_MiB=f['peak_reserved_bytes']/2**20,selected_updates=sel['checkpoint']['step'],selected_checkpoint_elapsed_seconds=sel['checkpoint']['elapsed_seconds']) for f,sel in zip(fits,selection['selections'])]
 csvwrite(OUT/'resources.csv',resources)
 wall=read(OUT/'run_wall.json');poll=[json.loads(line) for line in (OUT/'gpu_run.jsonl').read_text().splitlines()]
 contamination=sum(r['contaminated'] for f in fits for r in f['curve'])
 unauthorised=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in r['apps']) for r in poll)
 save(OUT/'resource_audit.json',dict(wall=wall,unauthorized_compute_samples=unauthorised,contaminated_optimizer_steps=contamination,peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in resources),minimum_free_MiB=min(r['free_mib'] for r in poll),primitive_forwards=state['primitive_forwards'],inference_forwards=state['inference_forwards'],training_forwards=state['training_forwards']))
 lines=['# 실제 예보 입력에서 LoRA 학습 대조 — 완료 보고','',
 '[확인] 최대 6 fits를 모두 실행했다. 이 실행은 표준 LoRA, 공변량 드롭아웃, 과거 예보 augmentation이라는 **알려진 대조군의 유용성**을 비교한다. 새로운 PEFT 구조나 독립 방법론 PASS는 아니다.','',
 '## 연구 질문과 고정 비교','',
 '기존 최신 예보 입력은 target-only보다 평균적으로 유용했으나, 경로 혼합·단순 신뢰도 신호에서 새 방법의 추가 가치를 확보하지 못했다. 이번 질문은 같은 입력의 LoRA 적응에 학습 가능한 여지가 있는가, 그리고 드롭아웃 또는 이전 예보 학습이 그 위에 무엇을 더하는가이다.','',
 'OS Gorredijk 한 타깃, TRAIN 3–4월 8원점 / V 5월 4원점 / D 6–9월 16원점이다. 모두 이미 이전 실험에 노출된 DISCOVERY 자료다. 미래 기상 정답은 읽지 않았고 기존 simulated-as-of 예보 파일을 재사용했다. context336h, horizon24h, feature3, TRAIN 과거만으로 공통 표준화했다. 실제 배포 공개시각의 증거는 아니다.','',
 '세 방법 모두 rank1/alpha2, 96 q/k/v/o projection, 147456개 학습 파라미터, frozen backbone/native head, lr1e-4, 120updates, seed61710/61711이다. STD는 최신 예보, EXODROP은 과거·미래 전체 채널에 같은 p=.3 inverted dropout, VINTAGE는 epoch별 네 예보 경로를 순환한다. VINTAGE에는 추가 예보 정보가 있으므로 같은 계산량과 같은 정보량을 혼동하지 않는다.','',
 '모든 6 fits의 V 선택을 봉인한 다음 D를 채점했다. SELECTED는 V 최소 체크포인트, FIXED120은 사전 고정 진단이다. D에서 유리한 체크포인트나 조건으로 주 비교를 바꾸지 않았다.','',
 '## 실제 실행과 미실행 범위','',
 f"- 본학습 **6/6 fits, 720/720 updates**, 별도 smoke 3회·6updates. 오류 {len(state['errors'])}건.",
 f"- 모델 forward {state['primitive_forwards']}회: gradient-enabled 학습 {state['training_forwards']}회, no-grad 추론 {state['inference_forwards']}회. 저장 예측 {len(predictions)}개, D 원점수416행, V검산{vchecks}개, 새 모델 체크포인트 재생24개.",
 '- 계획된 세 방법·두 seed·최신/과거 경로·selected/fixed120 평가의 미실행 0. 추가 후보·학생 모델·다른 타깃·10–12월 평가·학습률 탐색은 실행하지 않았다.','',
 '## 원점수','',
 'Primary는 정렬된 21분위수의 mean 2-pinball / 부하 context 표준편차이며 낮을수록 좋다. 아래는 두 seed 평균이다. 최신/과거 경로를 합쳐 새 지표를 만들지 않았다.','',
 '| 평가 | 입력 | INIT0 | STD | EXODROP | VINTAGE |','| --- | --- | ---: | ---: | ---: | ---: |']
 for source in ['SELECTED','FIXED120']:
  for path in [0,3]:
   values=[]
   for arm in ['INIT0']+ARMS:
    d=df[(df.arm==arm)&(df.path_index==path)]
    if arm!='INIT0':d=d[d.source==source]
    values.append(d.primary.mean())
   lines.append(f"| {source} | {'최신 k0' if path==0 else '과거 k3'} | "+' | '.join(f'{v:.9f}' for v in values)+' |')
 lines+=['','보조 지표(raw pinball·RMSE·MAE·80% 포함률/폭), 각 seed와 월 원점수는 [macro](macro_scores.csv), [월별](monthly_scores.csv), [전체](scores.csv)에 보존했다. 원본 무학습 예측과 이번 표준화 INIT0의 차이는 [직접 비교](initial_reference_comparison.csv)에 남겼다. 직접 이득의 기준은 이번 INIT0이다.','',
 '## 각 구성요소의 추가 가치','',
 '| 선택 | 입력 | 비교 | 개선율 % | 개선 월/4 | 월 block bootstrap95% % |','| --- | --- | --- | ---: | ---: | --- |']
 for e in effects:
  if e['seed']==-1 and (e['baseline']=='STD' or e['candidate']=='STD'):
   lines.append(f"| {e['source']} | k{e['path_index']} | {e['candidate']} vs {e['baseline']} | {e['gain_percent']:.4f} | {e['positive_months']} | [{e['ci_low']:.4f}, {e['ci_high']:.4f}] |")
 lines+=['','양의 개선율은 오차 감소다. 월 단위 2000회(seed61712) 재표집이며 seed·원점을 독립 타깃처럼 세지 않았다. 4개월·한 타깃·재사용 D에 대한 기술적 불확실성이지 독립 일반화 검정이 아니다. 두 seed 방향은 [효과 전체](effects.csv)에 별도로 공개한다.','',
 '## 자원과 재현 범위','',
 f"전체 controller {wall['seconds']:.2f}초(안전 대기 {wall['wait_seconds']:.2f}초 포함), 최소 GPU free {wall['minimum_free_mib']}MiB. 본학습 peak allocated 최대 {max(r['peak_allocated_MiB'] for r in resources):.2f}MiB. 비승인 compute 표본 {unauthorised}, 오염 기록 update {contamination}.",
 '모든 군의 파라미터 예산은 동일하다. 메모리·시간의 [실측](resources.csv)을 공개하며 구조적 자원 절감은 주장하지 않는다. full trajectory 시간에는 중간 검증·저장 비용이 들어가고 optimizer 시간은 그 합을 별도로 표시했다. 선택 지점 elapsed 역시 연구 중간 검증을 포함하여 순수 배포 시간을 뜻하지 않는다.','',
 f"[독립 검산](verification.json): D {len(errors)}개 지표, V {vchecks}개 primary, checkpoint24개와 저장 예측 해시, TRAIN 전용 표준화, 동일 초기화·순서, 과거 결과 {verification['historical_files_preserved']}개 보존. D 지표 최대 절대오차 {max(errors):.3g}; 재생24개 중 exact {verification['replays_exact']}개.",
 f"초기 native pipeline normalized max error {pre['pipeline_normalized_max_abs']:.3g}, label poison 예측 불변, native loss mask/reduction 검사 통과. native loss는 target 한 행의 24시점만 관측하고 32시점·4변수의 native reduction을 유지한다. 초기 raw/표준화 입력 예측 차이는 {pre['raw_vs_standardized_normalized_max_abs']:.3g}(context std 단위)로 별도 측정했다.",
 '원자료·예측·weights는 로컬 ignored cache에 있다. GitHub에는 코드·해시·원점수·보고서를 올리므로 원격 파일만으로 수치 재생이 완결되지는 않는다.','',
 '## 신규성과 남은 한계','',
 'LoRA 자체와 dropout/vintage augmentation은 알려진 기법이다. Exogenous Dropout의 채널 masking 규칙을 적용했지만 원 논문의 backbone·데이터·전체 실험을 재현한 것은 아니다. CoRA/UniCA/TFMAdapter 대비 방법 우위는 직접 비교하지 않았다. [선행 경계](../../research/covariate_reliability_diagnostic_20260916/LITERATURE_BOUNDARY.md)에 실제 읽은 범위를 기록했다.','',
 'NOVELTY=KNOWN_METHOD_CONTROLS_ONLY. 이 결과가 좋아도 새 PEFT 방법론 PASS가 아니다. 결과가 나빠도 모든 PEFT나 주제가 불가능하다는 결론은 아니다. 한 타깃, 적은 TRAIN/V, 고정 rank/lr 하나, 과거 개발 자료 재사용, simulated availability, 기상 vintage가 진짜 ensemble이 아니라는 한계가 남는다.','',
 'EXECUTION=COMPLETE. PREDICTIVE_EVIDENCE=POSITIVE_BUT_UNCERTAIN(최신 예보), 과거 예보에서는 INIT0 대비 TRADEOFF. NOVELTY=KNOWN_METHOD_CONTROLS_ONLY. 새 방법론 주제는 미확보이며, 알려진 vintage 학습에서 좁은 양성 신호를 확인했다. [구성요소·다음 연구 판단](INTERPRETATION.md)에 해석한다. 새로운 후보 학습은 이 실행에서 자동으로 추가하지 않는다.','']
 (OUT/'REPORT.md').write_text('\n'.join(lines))
 print(json.dumps(verification,indent=2));print(pd.DataFrame(effects).query('seed == -1').to_string(index=False))

if __name__=='__main__':main()
