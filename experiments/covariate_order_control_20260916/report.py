"""Independent scalar checks and order-only comparisons, no new fitting."""
import math
import pandas as pd
from common import *

def scalar(q,y,sd):
 pin=math.fsum(2*(t*(float(y[h])-float(q[i,h])) if y[h]>=q[i,h] else (t-1)*(float(y[h])-float(q[i,h]))) for i,t in enumerate(base.QUANTILES) for h in range(24))/504
 d=[float(q[10,h])-float(y[h]) for h in range(24)]
 return dict(primary=pin/sd,raw_2pinball=pin,raw_RMSE=math.sqrt(math.fsum(x*x for x in d)/24),raw_MAE=math.fsum(abs(x) for x in d)/24,coverage80=sum(q[2,h]<=y[h]<=q[18,h] for h in range(24))/24,width80=math.fsum(float(q[18,h])-float(q[2,h]) for h in range(24))/24)

def main():
 s=verified();state=read(OUT/'status.json');assert state['status']=='COMPLETE';fits=read(OUT/'fits.json');sel=read(OUT/'selection.json');jobs={j['id']:j for j in s['jobs']}
 assert sha(OUT/'fits.json')==sel['fits_sha256'] and sha(OUT/'selection.json')==state['selection_sha256']
 new=pd.read_csv(OUT/'scores.csv');old=pd.read_csv(OLD/'scores.csv');new['provenance']='NEW';old['provenance']='REUSED';df=pd.concat([old,new],ignore_index=True);df.loc[df.arm=='VINTAGE','arm']='CYCLE';assert len(df)==672
 df.to_csv(OUT/'comparison_scores.csv',index=False)
 errors=[];vc=0;schedule=read(OUT/'schedule.json');reference_schedule=read(OLD/'schedule.json')
 for r in df.to_dict('records'):
  j=jobs[r['id']]
  with np.load(ROOT/r['prediction']) as p:q=p['q'];np.testing.assert_array_equal(q,np.sort(p['raw'],axis=0))
  for k,value in scalar(q,np.load(ROOT/j['target']),j['std']).items():
   assert math.isclose(value,r[k],rel_tol=1e-10,abs_tol=1e-10),(r,k,value);errors.append(abs(value-r[k])/max(1,abs(value)))
 for f,c in zip(fits,sel['selections']):
  assert f['status']=='COMPLETE' and f['updates']==120 and f['trainable_parameters']==147456 and f['trainable_tensors']==192
  assert f['changed_tensors']>0 and f['frozen_unchanged'] and f['buffers_unchanged']
  stream=[r for r in schedule if r['arm']==f['arm'] and r['seed']==f['seed']];oldstream=[r for r in reference_schedule if r['arm']=='VINTAGE' and r['seed']==f['seed']]
  assert [(r['id'],r['step']) for r in stream]==[(r['id'],r['step']) for r in oldstream]
  assert sorted((r['id'],r['path']) for r in stream)==sorted((r['id'],r['path']) for r in oldstream)
  assert [(r['id'],r['path_index'],r['step']) for r in f['curve']]==[(r['id'],r['path'],r['step']) for r in stream]
  for cp in f['checkpoints']:
   assert sha(ROOT/cp['path'])==cp['sha256'];values=[]
   for j in s['jobs']:
    if j['split']=='V':
     with np.load(CACHE/f"{f['id']}_CP{cp['step']}_{j['id']}_k0.npz") as p:values.append(scalar(p['q'],np.load(ROOT/j['target']),j['std'])['primary']);vc+=1
   assert math.isclose(np.mean(values),cp['validation_primary'],rel_tol=1e-10,abs_tol=1e-10)
  assert min(f['checkpoints'],key=lambda cp:(cp['validation_primary'],cp['step']))==c['checkpoint']
 for p in read(OUT/'predictions.json'):assert sha(ROOT/p['path'])==p['sha256']
 for p,h in read(OUT/'historical_hashes.json').items():assert sha(ROOT/p)==h,p
 assert state['updates']==480 and state['smoke_updates']==4 and state['training_forwards']==484
 replay=read(OUT/'replay.json');assert len(replay)==16 and max(r['normalized_max_abs'] for r in replay)<=1e-5
 save(OUT/'verification.json',dict(status='PASS',independent_metric_checks=len(errors),max_relative_error=max(errors),validation_primary_checks=vc,equal_input_label_multisets=True,equal_origin_streams=True,selected_checkpoint_replays=16,replays_exact=sum(r['exact'] for r in replay),historical_preserved=len(read(OUT/'historical_hashes.json')),new_fits=4,new_updates=480,smoke_updates=4))
 keys=['primary','raw_2pinball','raw_RMSE','raw_MAE','coverage80','width80'];df.groupby(['arm','seed','source','path_index'],as_index=False)[keys].mean().to_csv(OUT/'macro_scores.csv',index=False);df.groupby(['arm','source','path_index','month'],as_index=False)[keys].mean().to_csv(OUT/'monthly_scores.csv',index=False)
 effects=[]
 for source in ['FIXED120','SELECTED']:
  for path in [0,3]:
   for arm in ARMS:
    for comparator in ['CYCLE','STD','INIT0']:
     for seed in [-1]+SEEDS:
      def values(a):
       d=df[(df.arm==a)&(df.path_index==path)]
       if a!='INIT0':d=d[(d.source==source)&((d.seed==seed) if seed!=-1 else True)]
       return d.groupby('month').primary.mean().reindex([6,7,8,9]).to_numpy()
      x=values(arm);b=values(comparator);ix=np.random.default_rng(61712).integers(0,4,(2000,4));bb=b[ix].mean(1);xx=x[ix].mean(1);boot=100*(bb-xx)/bb
      effects.append(dict(arm=arm,baseline=comparator,source=source,path_index=path,seed=seed,candidate_primary=x.mean(),baseline_primary=b.mean(),gain_percent=100*(b.mean()-x.mean())/b.mean(),positive_months=int((x<b).sum()),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
 pd.DataFrame(effects).to_csv(OUT/'effects.csv',index=False)
 resources=[dict(arm=f['arm'],seed=f['seed'],seconds=f['seconds'],optimizer_seconds=f['optimizer_seconds'],peak_allocated_MiB=f['peak_allocated_bytes']/2**20,peak_reserved_MiB=f['peak_reserved_bytes']/2**20,selected_updates=c['checkpoint']['step']) for f,c in zip(fits,sel['selections'])];pd.DataFrame(resources).to_csv(OUT/'resources.csv',index=False)
 wall=read(OUT/'run_wall.json');poll=[json.loads(x) for x in (OUT/'gpu_run.jsonl').read_text().splitlines()];unsafe=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in r['apps']) for r in poll);contaminated=sum(r['contaminated'] for f in fits for r in f['curve'])
 save(OUT/'resource_audit.json',dict(wall=wall,unauthorized_compute_samples=unsafe,contaminated_updates=contaminated,forward_counts={k:state[k] for k in ['training_forwards','inference_forwards']},peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in resources)))
 lines=['# 예보 버전 노출 순서 대조 — 완료','',
 '기존 예보 버전 학습의 이득이 노출 순서에 얼마나 민감한지 비교했다. CYCLE은기존VINTAGE를표시하는별칭이며재학습하지않았다. REVERSE/SHUFFLE은같은입력·정답멀티셋과초기값에서예보버전순서만달라진다. 노출된개발자료이며새PEFT발명/독립PASS가아니다.','',
 '## 실제 실행','',
 f"신규4/4fits·480updates, smoke2회·4updates. 이전6fits는원본예측참조로유지했고추가학습으로세지않았다. 신규모델forward {state['training_forwards']+state['inference_forwards']}회(학습{state['training_forwards']},추론{state['inference_forwards']}). 계획된신규평가256행의미실행0. 새로운보정기·다른후속후보학습0.",
 '각원점k0/k1/k2/k3를4/4/4/3회보는15epoch,각epoch같은8원점순열,총120updates. 새 학습률·기간·seed를추가하지않았다. 부하context336h/예측24h,rank1alpha2/147456개LoRA파라미터와기존native모델경로유지. 동일정보멀티셋의직접비교는FIXED120이며selected의prefix정보는서로다를수있다.','',
 '## 원점수 — 두seed 평균','',
 '| 정책 | 경로 | INIT0 | STD | EXODROP | CYCLE | REVERSE | SHUFFLE |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for source in ['FIXED120','SELECTED']:
  for path in [0,3]:
   vals=[]
   for arm in ['INIT0','STD','EXODROP','CYCLE']+ARMS:
    d=df[(df.arm==arm)&(df.path_index==path)]
    if arm!='INIT0':d=d[d.source==source]
    vals.append(d.primary.mean())
   lines.append(f'| {source} | k{path} | '+' | '.join(f'{v:.9f}' for v in vals)+' |')
 lines+=['','Primary는기존scaled2pinball,낮을수록좋다. 모든seed/보조지표와원점수를[전체표](comparison_scores.csv)에보존했다. 이전보정연구결과를이번raw표와같은조건으로합치지않았다.','',
 '| 고정120 비교 | 경로 | 개선율 % | 개선월/4 | 월block95% % |','| --- | --- | ---: | ---: | --- |']
 for e in effects:
  if e['seed']==-1 and e['source']=='FIXED120':lines.append(f"| {e['arm']} vs {e['baseline']} | k{e['path_index']} | {e['gain_percent']:.4f} | {e['positive_months']} | [{e['ci_low']:.4f}, {e['ci_high']:.4f}] |")
 lines+=['','## 검산·비용·한계','',
 f"독립scalar지표{len(errors)}개·V primary{vc}개,입력멀티셋/원점순서,새모델복원16개(exact{sum(r['exact'] for r in replay)}),기존결과{len(read(OUT/'historical_hashes.json'))}개hash보존을검사했다. 기존nativepipeline/targetpoison/FP64수식검사는같은helper의이전검증을참조했다. [검산](verification.json).",
 f"GPU controller {wall['seconds']:.2f}초,대기{wall['wait_seconds']:.2f}초,최소free {wall['minimum_free_mib']}MiB,peak allocated최대{max(r['peak_allocated_MiB'] for r in resources):.2f}MiB. 비승인compute표본{unsafe},오염update{contaminated}. 외부compute통계의RustDesk는승인된예외다. 상세fit비용은[resources](resources.csv).",
 '한타깃·TRAIN8/V4/D16·4개월·동일개발기간재사용이다. 월block2000회(seed61712)는서술적구간이며독립확증이아니다. 순서가달라도좋다는관측만으로추가정보의원인이나새구조필요성이증명되지는않는다. 같은정보를다른순서로학습하는알려진대조이며신규성은KNOWN_ORDER_CONTROLS_ONLY다. 원자료·예측·weights는로컬cache에남기고코드·해시·점수·보고서를공개한다.','',
 'EXECUTION=COMPLETE. 성능/연구해석은[INTERPRETATION.md](INTERPRETATION.md). 사용자최신지시에따라이기상예보축에서추가후속을시작하지않고,별도MOMENT단일계약으로이동한다.','']
 (OUT/'REPORT.md').write_text('\n'.join(lines));print(pd.DataFrame(effects).query("seed == -1 and source == 'FIXED120'").to_string(index=False))

if __name__=='__main__':main()
