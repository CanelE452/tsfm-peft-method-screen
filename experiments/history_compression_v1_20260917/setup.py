import subprocess,shutil
from .common import *
def prepare():
 if (OUT/'SEAL.json').exists():validate_seal();return
 d=Data();spec=read(OLD/'N02/PROTOCOL.json');div=read(OLD/'N02/origin_diversity.json');assert div['passed']
 # Every information packet and original frozen selection is validated, without opening E labels.
 inputs={}
 for p,h in spec['data']['packet_hashes'].items():assert sha(ROOT/p)==h;inputs[p]=h
 for f in ['PROTOCOL.json','train_statistics.json','origin_diversity.json','fits.json','selections.json','LR_selection.json','predictions_manifest.json']:
  p=OLD/'N02'/f;inputs[str(p.relative_to(ROOT))]=sha(p)
 fits=[v for v in read(OLD/'N02/fits.json') if v['arm'] in ['B0','B1']];assert len(fits)==8 and all(f['status']=='COMPLETE' and f['updates']==512 for f in fits)
 for f in fits:
  assert f['lr'] in LRS and f['seed'] in SEEDS
  for cp in f['checkpoints']:
   assert sha(ROOT/cp['path'])==cp['sha256'];inputs[cp['path']]=cp['sha256'];inputs[cp['prediction']]=sha(ROOT/cp['prediction'])
 selections=[s for s in read(OLD/'N02/selections.json')['selections'] if s['arm'] in ['B0','B1']]
 oldpred=[r for r in read(OLD/'N02/predictions_manifest.json')['predictions'] if r['arm'] in ['B0','B1']]
 for p in oldpred:assert sha(ROOT/p['path'])==p['sha256'];inputs[p['path']]=p['sha256']
 teacher=next(s for s in read(OLD/'N02/LR_selection.json')['selections'] if s['arm']=='B1')
 for role,n in [('TRAIN',64),('V_SELECT',32),('E_DISCOVERY',64)]:
  x=d.x[role];oo=x['origins'];assert len(oo)==n and len(set(oo//24))==n and x['context'].shape==(n,4,1344) and np.isfinite(x['context']).all()
  assert all(div['roles'][role]['checks'].values())
 history={str(p.relative_to(ROOT)):sha(p) for folder in ['results','research'] for p in (ROOT/folder).rglob('*') if p.is_file() and OUT not in p.parents}
 save(OUT/'historical_hashes.json',history)
 save(OUT/'REUSE_RECEIPT.json',dict(fits=fits,selections=selections,predictions=oldpred,teacher=teacher,roles=div['roles'],historical_files=len(history),reused_fits=8,reused_updates=4096,duplicate_new_fits=0))
 protocol=dict(name=NAME,source='traffic existing first4 columns',model='pinned Chronos2 FP32 rank8; all old LoRA targets; dropout0; frozen head',roles={r:len(d.x[r]['origins']) for r in ROLES},new_arms=ARMS,reused_arms=['SHORT','LONG'],lrs=LRS,choice_seed=73100,repeat_seeds=[73101,73102],steps=512,checkpoints=[0,256,512],order='old rng ORDER(seed,epoch);64 origins repeated8epochs',optimizer='AdamW betas.9/.999 eps1e-8 wd0 globalclip1',new_fit_cap=20,new_main_update_cap=10240,smoke_update_cap=10,teacher_extra_fits=0,resource_optimizer_updates=0,forward_cap=40000,wall_cap_seconds=14400,kd_lambda=.25,compression=dict(old_patches=63,group_size=3,summary_tokens=21,recent_patches=21,rank=8,rule='weights=softmax(u^T tanh(V(h-mean_h))), u=0 init, pooled=sum(weights*h); mean native positions; native absolute time embeddings preserved'),selection='each arm seed73100 twoLR best V; lock LR; repeats choose checkpoint on V; no calibration; score E after all selections sealed',primary='mean_channel sqrt(mean_origin,h ((pred-y)/TRAIN_sigma)^2)',bootstrap='168hour index blocks,2000replicates seed73300; paired sameorigins; paired weights per seed; mean seed NRMSE for both bootstrap and point estimates',comparisons=['LEARN_KD vs POOL_KD','LEARN_KD vs LEARN','LEARN vs POOL','POOL_KD vs POOL','LEARN_KD vs LONG','LEARN_KD vs STATS_SHORT','LEARN_KD vs SHORT'],decision='Exploratory component evidence needs positive mean and paired95%CI lower>0 versus POOL_KD and LEARN. Resource tradeoff: >=10% lower measured peak or latency than LONG with relative error upperCI<=1%. Not independent replication or novelty PASS. No automatic followup.',E_exposure='already used development E; no new independent test',limits='one source4channels2repeat seeds; fixed teacher chosen using historical V;teacher has same TRAIN labels but privileged long model training;charge full teacher costs, no E teacher; pooled native embeddings is not claimed novel')
 save(OUT/'PROTOCOL.json',protocol)
 write(OUT/'PROTOCOL.md','''# 긴 이력 압축 PEFT — 봉인 개발 파일럿\n\n사용자의 2026-09-17 “그렇게 해서 실험해줘” 요청으로 실행한다. 앞서 종료한 트랙은 재개하지 않고, 교정된 N02의 실제 입력·원점·기준선만 읽기 전용 재사용한다.\n\nSHORT/LONG 완료8fits는 검산 후 참조한다. 새5군 × (선택seed 두LR + 반복2seed 고정LR) =20fits,512updates씩, 총10,240본업데이트. 군별2smoke=10폐기업데이트. 다른 데이터·후속학습 자동 실행 없음. 모든 군은 성능 gate 없이 정해진 비교를 마친다. 공통 GPU 위험은 정지/재개 상태로 보존한다.\n\n최근336시간21patch 유지, 오래된1008시간63patch를 인접3개씩21token으로 요약한다. 전체 context token84→42, REG/미래 포함88→46. 과거 원시값을 가짜 동일간격 시계열로 재해석하지 않고 native patch embedding을 풀링하며, native 시간 feature와 원래 patch 중심 position을 보존한다. STATS_SHORT는 긴 이력 native normalization만 계산하고 최근21patch만 encoder에 넣는다. 각 압축군은 같은 full-context normalization을 쓴다.\n\nPOOL은균등convex평균, LEARN은공유rank8비선형score로같은3개 내convex가중. 초기u=0으로POOL과동일. POOL_KD/LEARN_KD는같은label MSE에0.25×교사예측MSE 추가. 교사는E와무관하게기존B1의선택seed73100,V선택체크포인트로고정, TRAIN예측만생성한다. 학생도동일1344관측을사용한다. TRAIN포함교사fit비용과cache생성비용을공개한다.\n\n주 비교는LEARN_KD대POOL_KD(학습압축),LEARN_KD대LEARN(증류),LEARN대POOL,POOL_KD대POOL. SHORT/LONG/STATS_SHORT도모두공개한다. 낮은NRMSE가좋고,두반복평균과시간block bootstrap을보고한다. V에서만LR/checkpoint선택, 모든선택봉인후E채점, 출력보정없음. E는재사용개발기간이다.\n\n추가가치의탐색근거는두핵심비교의평균양수및95%CI하한>0일때만기록한다. 자원절충신호는LONG대비측정peak또는latency10%이상감소와상대오차95%상한1%이내를같이요구한다. 현재결과에맞춰문턱을수정하지않는다. 이는논문PASS기준이아니다. 한쪽seed악화도보존한다.\n\n선행의압축메모리와TS-Memory증류와겹치므로풀링/증류조합의신규성을주장하지않는다. https://arxiv.org/abs/2409.13530 , https://arxiv.org/abs/2602.11550 . 정식재현비교와독립원천은범위밖이며다음연구가자동승인된것이아니다.\n''')
 inputs[str((OUT/'PROTOCOL.json').relative_to(ROOT))]=sha(OUT/'PROTOCOL.json');inputs[str((OUT/'PROTOCOL.md').relative_to(ROOT))]=sha(OUT/'PROTOCOL.md')
 save(OUT/'SEAL.json',dict(at=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=all_source_hashes(),inputs=inputs,history_manifest=sha(OUT/'historical_hashes.json')))
 print('PREPARED sealed20newfits8reused',flush=True)
