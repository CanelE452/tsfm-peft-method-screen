"""Independent scalar/selection/training-record audit and Korean interpretation."""
import math
import torch
from .common import *
def scalar(pred,y,sigma):
 # Explicit float64 scalar reduction, independent of vectorized training/selection metric.
 return sum(math.sqrt(math.fsum(((float(pred[i,c,h])-float(y[i,c,h]))/float(sigma[c]))**2 for i in range(len(y)) for h in range(48))/(len(y)*48)) for c in range(4))/4

def verify():
 validate_seal();s=read(OUT/'state.json');assert s['status']=='FINISHED' and s['main_updates']==10240 and s['smoke_updates']==10 and s['completed']==20
 history=read(OUT/'historical_hashes.json')
 for p,h in history.items():assert sha(ROOT/p)==h,('HISTORICAL_CHANGE',p)
 d=Data();fits=read(OUT/'fits.json');logs=[json.loads(l) for l in (OUT/'optimizer.jsonl').read_text().splitlines()];assert len(logs)==10250
 assert not any(r['contaminated'] for r in logs),'RESOURCE_CONTAMINATION'
 Vchecks=[];records=[]
 for f in fits:
  rr=[r for r in logs if r['id']==f['id']];assert len(rr)==512
  for j,r in enumerate(rr):assert r['step']==j+1 and r['origin_ordinal']==int(rng('ORDER',f['seed'],j//64).permutation(64)[j%64]) and r['origin']==int(d.x['TRAIN']['origins'][r['origin_ordinal']])
  state=torch.load(CACHE/'resume'/f"{f['id']}.pt",map_location='cpu',weights_only=False);assert state['step']==512 and state['seal']==sha(OUT/'SEAL.json')
  final=torch.load(ROOT/f['checkpoints'][-1]['path'],map_location='cpu',weights_only=True);assert tensor_hash(state['parameters'])==tensor_hash(final)
  assert all(int(v['step'])==512 and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in state['optimizer']['state'].values())
  assert [c['step'] for c in f['checkpoints']]==[0,256,512]
  for cp in f['checkpoints']:
   assert sha(ROOT/cp['path'])==cp['sha256'];pp=np.load(ROOT/cp['prediction']);assert np.array_equal(pp['origins'],d.x['V_SELECT']['origins']);v=scalar(pp['pred'],d.y['V_SELECT'],d.sigma);assert abs(v-cp['accuracy'])<=1e-10;Vchecks.append(dict(id=f['id'],step=cp['step'],scalar=v))
  records.append(dict(id=f['id'],exact_order=True,updates=512,checkpoint_resume_exact=True,optimizer_steps=512))
 def best(f):return min(f['checkpoints'],key=lambda c:(c['accuracy'],c['step']))
 choices=read(OUT/'LR_selection.json')['selections'];ss=read(OUT/'selections.json')['selections']
 for sel in choices+ss:
  candidates=[f for f in fits if f['seed']==sel['seed'] and f['arm']==sel['arm']]
  expected=min(candidates,key=lambda f:(best(f)['accuracy'],f['lr'],best(f)['step']))
  assert sel['id']==expected['id'] and sel['selected']==best(expected)
 for f in fits:
  if f['seed']!=73100:assert f['lr']==next(c['lr'] for c in choices if c['arm']==f['arm'])
 assert sha(OUT/'selections.json')==read(OUT/'EVALUATION_SEAL.json')['selection_hash']
 # E is opened for scoring only after selection and prediction completion.
 y=np.load(OC/'E_DISCOVERY_labels.npz')['y'];manifest=read(OUT/'predictions_manifest.json');reuse=read(OUT/'REUSE_RECEIPT.json')
 allpred=manifest+[dict(r,arm={'B0':'SHORT','B1':'LONG'}[r['arm']],reused=True) for r in reuse['predictions']]
 assert len(allpred)==28;rows=[];origins=[]
 for row in allpred:
  assert sha(ROOT/row['path'])==row['sha256'];p=np.load(ROOT/row['path']);q=p['pred'];q=q[:,0] if q.ndim==4 else q;assert q.shape==y.shape and np.array_equal(p['origins'],d.x['E_DISCOVERY']['origins'])
  v=nrms(q,y,d.sigma);vv=scalar(q,y,d.sigma);assert abs(v-vv)<=1e-10
  rows.append(dict(arm=row['arm'],seed=row['seed'],policy=row['policy'],selected_step=row['selected_step'],NRMSE=v,rawMAE=float(abs(q-y).mean()),normalizedMAE=float((abs(q-y)/d.sigma[None,:,None]).mean()),prediction=row['path'],reused=row.get('reused',False)))
  for i,o in enumerate(p['origins']):
   for c in range(4):origins.append(dict(arm=row['arm'],seed=row['seed'],policy=row['policy'],origin=int(o),channel=c,normalizedMSE=float((((q[i,c]-y[i,c])/d.sigma[c])**2).mean())))
 csvwrite(OUT/'scores.csv',rows);csvwrite(OUT/'scores_by_origin.csv',origins);save(OUT/'independent_selection_verification.json',Vchecks);save(OUT/'training_record_verification.json',records)
 verify_result=dict(passed=True,actual_new_fits=20,reused_fits=8,main_updates=10240,smoke_updates=10,forward_calls=s['forwards'],old_files_preserved=len(history),V_scalar_checks=len(Vchecks),E_scalar_checks=len(rows),all_selections_replayed=True,all_resume_optimizer_states_checked=True,contaminated_updates=0,seal_intact=True,at=time.time())
 save(OUT/'verification.json',verify_result);print('VERIFIED',verify_result,flush=True)

def contrasts():
 d=Data();y=np.load(OC/'E_DISCOVERY_labels.npz')['y'];man=read(OUT/'predictions_manifest.json')+[dict(r,arm={'B0':'SHORT','B1':'LONG'}[r['arm']]) for r in read(OUT/'REUSE_RECEIPT.json')['predictions']]
 preds={}
 for r in man:
  if r['policy']!='selected':continue
  p=np.load(ROOT/r['path'])['pred'];preds[r['arm'],r['seed']]=p[:,0] if p.ndim==4 else p
 w=block_counts(d.x['E_DISCOVERY']['origins'],168);den=w.sum(1);assert (den>0).all()
 def values(a,seeds):
  vals=[];boots=[]
  for seed in seeds:
   e=((preds[a,seed]-y)/d.sigma[None,:,None])**2;vals.append(nrms(preds[a,seed],y,d.sigma));boots.append(np.sqrt(np.einsum('bi,ic->bc',w,e.mean(-1))/den[:,None]).mean(1))
  return np.mean(vals),np.mean(boots,axis=0)
 pairs=[('LEARN_KD','POOL_KD'),('LEARN_KD','LEARN'),('LEARN','POOL'),('POOL_KD','POOL'),('LEARN_KD','LONG'),('LEARN_KD','STATS_SHORT'),('LEARN_KD','SHORT'),('LONG','SHORT'),('STATS_SHORT','SHORT'),('POOL','STATS_SHORT')]
 rows=[]
 for a,b in pairs:
  for seeds,key in [([73101],73101),([73102],73102),([73101,73102],'MEAN')]:
   av,ab=values(a,seeds);bv,bb=values(b,seeds);g=100*(bv-av)/bv;boot=100*(bb-ab)/bb;lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(proposed=a,baseline=b,seed=key,proposed_NRMSE=float(av),baseline_NRMSE=float(bv),gain_percent=float(g),CI_low=float(lo),CI_high=float(hi)))
 csvwrite(OUT/'contrasts.csv',rows);return rows

def report():
 assert read(OUT/'verification.json')['passed'];cs=contrasts();rows=list(csv.DictReader((OUT/'scores.csv').open()));res=read(OUT/'inference_resources.json');fits=read(OUT/'fits.json');reuse=read(OUT/'REUSE_RECEIPT.json');verify_info=read(OUT/'verification.json')
 def c(a,b):return next(r for r in cs if r['proposed']==a and r['baseline']==b and r['seed']=='MEAN')
 components=all(c('LEARN_KD',b)['CI_low']>0 for b in ['POOL_KD','LEARN'])
 ar=['SHORT','LONG']+ARMS;means={a:np.mean([float(r['NRMSE']) for r in rows if r['arm']==a and r['policy']=='selected']) for a in ar}
 mem={a:np.mean([r['peak_allocated_bytes'] for r in res if r['arm']==a]) for a in ar};lat={a:np.mean([r['median_seconds'] for r in res if r['arm']==a]) for a in ar}
 memory_gain=100*(mem['LONG']-mem['LEARN_KD'])/mem['LONG'];latency_gain=100*(lat['LONG']-lat['LEARN_KD'])/lat['LONG'];tradeoff=c('LEARN_KD','LONG')['CI_low']>=-1 and max(memory_gain,latency_gain)>=10
 status='EXPLORATORY_COMPONENT_SIGNAL' if components else 'COMPONENT_NOT_ESTABLISHED';decision='0개: 새 구성요소를 다음 투자 후보로 확정할 근거 부족' if not components else '최대1개: 고정 토큰 예산의 예측 보존 압축, 탐색 근거에 한정'
 text='# 긴 이력 압축 PEFT — 실행 결과\n\n**새20/20fits와 평가·검산을 완료했다. 기존SHORT/LONG8fits는재학습없이검증후재사용했다.** 이번은Traffic4채널의재사용개발평가이며독립test도신규방법PASS도아니다.\n\n'
 text+='[봉인 계획](PROTOCOL.md), [세부 조건](PROTOCOL.json), [재사용 감사](REUSE_RECEIPT.json), [검산](verification.json).\n\n'
 text+='## 원점수와 실제 추론 자원\n\n| 군 | 두seed평균 NRMSE↓ | 추론 peak MiB | 원점당 median평균 ms |\n|---|---:|---:|---:|\n'
 for a in ar:text+=f'| {a} | {means[a]:.6f} | {mem[a]/2**20:.2f} | {lat[a]*1000:.2f} |\n'
 text+='\n추론 자원은V의같은4원점,seed별같은모델load절차,2warmup과12회측정이다. 압축·native전처리·정규화·출력복원을포함하고,데이터파일I/O와교사비용은별도다. peak는PyTorch allocated,전체장치메모리가아니다. RustDesk와순서에따른시스템변동으로작은속도차이는제한적으로해석한다.\n\n'
 text+='## 모든 seed와 선택\n\n| 군 | seed | selected step | selected NRMSE | fixed512 NRMSE |\n|---|---:|---:|---:|---:|\n'
 for a in ar:
  for seed in [73101,73102]:
   s=next(r for r in rows if r['arm']==a and int(r['seed'])==seed and r['policy']=='selected');f=next(r for r in rows if r['arm']==a and int(r['seed'])==seed and r['policy']=='fixed512');text+=f"| {a} | {seed} | {s['selected_step']} | {float(s['NRMSE']):.6f} | {float(f['NRMSE']):.6f} |\n"
 text+='\n## 구성요소의 직접 추가 가치\n\n양수는제안쪽이좋음. 두seed NRMSE평균의상대차이이며,같은원점을168h시간block으로2000회재표집한95%구간이다. seed2개로seed모집단불확실성을충분히추정한것이아니다. 채널4개를독립dataset4개로세지않는다.\n\n| 제안 / 대조 | 평균 gain% | 95% 구간 |\n|---|---:|---|\n'
 for r in cs:
  if r['seed']=='MEAN':text+=f"| {r['proposed']} / {r['baseline']} | {r['gain_percent']:+.4f} | [{r['CI_low']:+.4f}, {r['CI_high']:+.4f}] |\n"
 text+=f'\n구성요소판정: **{status}**. LEARN_KD대POOL_KD는학습압축,LEARN_KD대LEARN은증류를검사한다. POOL_KD대POOL로단순증류효과도분리한다. STATS_SHORT는긴이력의정규화통계와최근토큰만사용하므로압축의평균/표준편차정보교란을확인하는대조다.\n\n'
 text+=f'LONG대비LEARN_KD 추론peak절감{memory_gain:+.3f}%,latency절감{latency_gain:+.3f}%. 사전정의정확도1%비열등상한과자원10%절감의동시신호는 **{tradeoff}**. 단순토큰84→42 감소를실제50%속도향상으로부르지않는다.\n\n'
 text+='## 학습·교사 비용과 검산\n\n'
 text+=f"새본학습20fits/10,240updates,폐기smoke10updates. 재사용8fits/4,096updates. 교사는그8fits중LONG선택seed의기존2LR실험에서선택했다. 교사fit2개의optimizer시간합{sum(f['optimizer_seconds'] for f in reuse['fits'] if f['arm']=='B1' and f['seed']==73100):.2f}초를배포recipe비용에포함해야하며,저장돼있다는이유로0비용이라하지않는다. 이번추가교사훈련0,TRAIN교사예측생성{read(OUT/'teacher.json')['seconds']:.2f}초다.\n\n"
 text+=f"새20fits optimizer실측합{sum(f['optimizer_seconds'] for f in fits):.2f}초,선택포함경로시간합{sum(f['seconds'] for f in fits):.2f}초. 학습자원원점수는[fits](fits.json),추론자원은[inference_resources](inference_resources.json). 교사E예측이나숨긴정답입력은없고교사선택에는기존V가쓰였다. 동일TRAIN에대한교사의in-sample예측을사용한증류라는한계를남긴다.\n\n"
 text+=f"모든20resume의최종가중치·Adam512step·정확한origin순서를검사했다. V60scalar/E28scalar독립검산,14?가아닌신규20개정책별복원검사를[restore](restore.json)에기록했다. frozen파라미터와buffer보존,POOL/LEARN초기동일성,학습압축gradient,전체토큰native경로동일성도검사했다. 과거{verify_info['old_files_preserved']}개파일은hash불변이다. 비허용GPU작업오염0updates.\n\n"
 text+='## 한계와 종료\n\n단일Traffic의기존4채널·2반복·64개날짜E를재사용했다. LONG의개발양성신호를보고정한후보여서독립확증으로볼수없다. 추가source/seed/LR와후속학습을자동실행하지않는다.\n\nConvex attention pooling과teacher MSE는알려진구성이다. [TS-Memory](https://arxiv.org/abs/2602.11550)는검색교정의증류,[압축메모리연구](https://arxiv.org/abs/2409.13530)는TSFM의채널문맥확장을다룬다. 이번최근/과거토큰배치가그들과완전히같다는뜻은아니지만,조합만으로신규성을확보하지못하며정식선행재현은미수행이다. 이는PEFT실험의타당성과논문기여를구분한판정이다.\n\n'
 text+='원시데이터·예측배열·가중치는로컬ignoredcache,공개검토에는원점수·해시·코드·보고서를남긴다. GitHub파일만으로전체수치재생이된다고주장하지않는다.\n'
 text=text.replace('14?가아닌신규20개','신규20개')
 write(OUT/'REPORT.md',text);write(OUT/'FINAL_DECISION.md',f'# 최종 결정\n\n실행완료20/20신규fits,8기존fits재사용,평가·검산완료.\n\n구성요소: **{status}**. 정확도/자원절충신호: **{tradeoff}**. 신규성: **NOT_ESTABLISHED**.\n\n다음투자후보 **{decision}**. 이는논문성공선언이아니다. 선행직접비교와독립source검증은아직없다.\n\n날짜분산입력,단순POOL/STATS_SHORT/SHORT/LONG대조와모든LEARN/KD실험결과는보존한다. 불리한seed나구간을제외하지않는다. 추가학습을시작하지않고이번승인범위에서종료한다. [전체보고서](REPORT.md).\n')
 save(OUT/'DECISION.json',dict(execution='COMPLETE',component_signal=components,resource_tradeoff_signal=bool(tradeoff),novelty='NOT_ESTABLISHED',followup_candidates=int(components),no_followup_started=True))
 print('REPORT',status,flush=True)
