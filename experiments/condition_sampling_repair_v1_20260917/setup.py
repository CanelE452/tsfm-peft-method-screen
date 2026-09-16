"""Audit old artifacts; prepare all six unscored packets; seal eligibility before GPU use."""
import sys,subprocess,copy,ast,shutil,time
import pandas as pd
from .common import *
from .sampling import *
from .prepare import FILES,prepare_track
BASE='b80e4a4af34c34827a24319258f334d9a171cf86'

def prepare_all(contract):
 if (OUT/'MASTER_SEAL.json').exists():print('ALREADY_SEALED_NO_DUPLICATE');return
 assert not (OUT/'controller_state.json').exists()
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 (OUT/'MASTER_PROTOCOL.md').write_bytes(Path(contract).read_bytes())
 hist={str(p.relative_to(ROOT)):sha(p) for folder in ['results','research'] for p in (ROOT/folder).rglob('*') if p.is_file() and OUT not in p.parents}
 oldcode={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/condition_studies_v1_20260916').glob('*.py')};save(OUT/'historical_hashes.json',{**hist,**oldcode})
 audit=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),base=BASE,dirty=subprocess.check_output(['git','status','--porcelain'],text=True),diff_since_base=subprocess.check_output(['git','diff',BASE+'..HEAD','--stat'],text=True),contract_path=str(contract),contract_sha256=sha(contract),old_results_sha256={str(p.relative_to(ROOT)):sha(p) for p in OLD.rglob('*') if p.is_file()},old_model_sha256=read(OLD/'MASTER_SEAL.json')['model_files'])
 for path,h in audit['old_model_sha256'].items():assert sha(path)==h
 for name in ['MASTER_REPORT.md','FINAL_DECISION.md','origin_dispersion.csv']:assert (OLD/name).exists()
 for t in ORDER:
  for name in ['REPORT.md','origins.json','origins.csv','selections.json','fit_manifest.csv','predictions_manifest.json','verification.json']:assert (OLD/t/name).exists()
 save(OUT/'repository_audit.json',audit)
 (OUT/'REPOSITORY_AUDIT.md').write_text(f'# 저장소·완료 상태 감사\n\nHEAD `{audit["head"]}`, 기준 `{BASE}`. 차이: {audit["diff_since_base"] or "없음"}. 이번 새 경로 외의 기존 사용자 변경은 없었다. 기존 코드·결과는 read-only이며 이전 결과와 연구 {len(hist)}개 파일 및 기존 실험 소스 hash를 보존한다. 기존 날짜 우선 비교는 없어 신규 비교이고 이전 가중치를 재학습해 복원하지 않는다. pinned Chronos-2 revision과 모델 파일 hash를 확인했다. 세 원자료도 아래 사전 영수증에서 고정 hash를 검증한다.\n\nR05/N06/R09는 UNAFFECTED_REFERENCE이며 실행하지 않는다. 기존 E 및 이번 E 모두 재사용 공개 원천의 개발 평가다. 새 독립 test가 아니다.\n')
 arrays={};receipts={}
 for source,(path,h) in FILES.items():
  assert sha(ROOT/path)==h
  if source=='ettm1':df=pd.read_csv(ROOT/path);a=df.iloc[:,1:].to_numpy(float);columns=list(df.columns[1:]);assert pd.to_datetime(df.iloc[:,0]).diff().dropna().eq(pd.Timedelta(minutes=15)).all()
  else:a=np.loadtxt(ROOT/path,delimiter=',');columns=[f'column_{i:03}' for i in range(a.shape[1])]
  arrays[source]=(a,columns);receipts[source]=dict(path=path,sha256=h,raw_shape=list(a.shape),bytes=(ROOT/path).stat().st_size,download_bytes=0)
 specs=[];rows=[];origin_tests=test_selector();cross=[]
 for t in ORDER:
  out=OUT/t;out.mkdir(exist_ok=True);oldspec=read(OLD/t/'PROTOCOL.json');a,columns=arrays[SOURCES[t]];di=prepare_track(t,a,columns,receipts[SOURCES[t]]);spec=copy.deepcopy(oldspec);spec.update(data=di,repair_id='S'+str(ORDER.index(t)+1).zfill(2),origin_selection='day_first_v1; sole scientific change');save(out/'PROTOCOL.json',spec);specs.append(spec)
  rec=read(out/'data_receipt.json');assert rec['selected_columns']==read(OLD/t/'data_receipt.json')['selected_columns'];indices=[columns.index(c) for c in rec['selected_columns']];aa=a[:,indices];candidates=legal_candidates(t,aa,rec['bounds']);orig=read(out/'origins.json');oldorig=read(OLD/t/'origins.json');div={};bad=[]
  poison=np.where(np.isfinite(aa),1e8-aa,aa);permuted=np.where(np.isfinite(aa),np.roll(np.nan_to_num(aa),1,axis=1),aa)
  assert legal_candidates(t,poison,rec['bounds'])==candidates==legal_candidates(t,permuted,rec['bounds'])
  for role in ROLES:
   n=64 if role in ['TRAIN','E_DISCOVERY'] else 32;v=candidates[role];again=pick(v,n,[],rec['period']);assert orig[role]==again==pick(v[::-1],n,[],rec['period']);d=dispersion(t,orig[role],v);checks,failed=failures(t,role,d);div[role]=dict(**d,checks=checks,failed=failed,status='BLOCKED_DIVERSITY' if failed else 'PASS');bad.extend([role+':'+k for k in failed])
   for version,oo in [('old',oldorig[role]),('repaired',orig[role])]:
    ds=dispersion(t,oo,v);rows.append(dict(track=t,role=role,version=version,**{k:x for k,x in ds.items() if not isinstance(x,dict)},phase_histogram=json.dumps(ds['phase_histogram']),gap_slots_quantiles=json.dumps(ds['gap_slots_quantiles']),target_overlap_multiplicity_histogram=json.dumps(ds['target_overlap_multiplicity_histogram'])))
  if t=='N03':
   from experiments.condition_studies_v1_20260916.prepare import pick as oldpick
   phases=np.floor(np.linspace(0,95,64)).astype(int).tolist();reproduced=oldpick(candidates['E_DISCOVERY'],64,phases,96);assert reproduced==oldorig['E_DISCOVERY'];assert (max(reproduced)-min(reproduced))*.25==23.75;assert div['E_DISCOVERY']['distinct_days']==64 and div['E_DISCOVERY']['span_ratio']>=.95;origin_tests['N03_actual_old_pathology_reproduced']=True;origin_tests['N03_actual_repaired_multi_day']=True
  save(out/'origin_diversity.json',dict(track=t,roles=div,passed=not bad,failures=bad));shutil.copy2(OLD/t/'origins.csv',out/'old_origins.csv');shutil.copy2(out/'origins.csv',out/'new_origins.csv');save(OUT/'new_origins'/f'{t}.json',orig);shutil.copy2(out/'origins.csv',OUT/'new_origins'/f'{t}.csv')
  ns=read(out/'train_statistics.json');os=read(OLD/t/'train_statistics.json');assert all(ns[k]==v for k,v in os.items()),('TRAIN_STATS_CHANGED',t);assert set(ns)-set(os)<={'split_train_end'}
  if 'split_train_end' in ns:assert ns['split_train_end']==ns['train_end']
  if t!='N07':assert read(out/'feature_or_transform_manifest.json')==read(OLD/t/'feature_or_transform_manifest.json'),('METHOD_AUX_CHANGED',t)
  save(out/'STATUS.json',dict(EXECUTION='BLOCKED_DIVERSITY' if bad else 'PREPARED',EVIDENCE='NOT_MEASURED',NOVELTY='UNVERIFIED_VARIANT',reason=bad,neural_fits=0))
  for sel in read(OLD/t/'selections.json')['selections']:
   cp=sel['selected'];path=ROOT/cp['path'];valid=path.exists() and sha(path)==cp['sha256'];cross.append(dict(track=t,arm=sel['arm'],seed=sel['seed'],checkpoint=cp['path'],sha256=cp['sha256'],hash_valid=valid,status='AVAILABLE' if valid else 'BLOCKED_OLD_CHECKPOINT'))
  (out/'TOPIC_ONEPAGE.md').write_text(f'# {t} {NAMES[t]} 표본 교정\n\n질문: {QUESTIONS[t]}. 기존 동일 군 {ARMS[t]}, 핵심 대조 {PROPOSED[t]} 대 {CONTRASTS[t]}. 이번 과학적 변경은 날짜 우선 원점 선정 하나다. 방법 정의는 기준 PROTOCOL의 보존된 contract_section과 implementation_choices에 기록했다. 현재 표본 조건: {"통과" if not bad else "BLOCKED_DIVERSITY: "+str(bad)}. 조건 미충족을 성능 실패로 해석하지 않는다.\n')
  shutil.copy2(OLD/t/'LITERATURE_BOUNDARY.md',out/'LITERATURE_BOUNDARY.md');(out/'LIMITATIONS.md').write_text('# 한계\n\n기존과 같은 공개 원천·개발 기간, 네 채널·두 반복 seed다. 더 넓은 표본은 독립 source 검증이 아니다. 정식 선행 재현을 하지 않는다. old/new 교차 평가는 TRAIN/V/선택/평가의 완전한 인과 분해가 아니다. BLOCKED_DIVERSITY 트랙은 repaired 성능 근거가 없으며 이전 부정적 결론의 강도를 높이지 않는다.\n')
 save(OUT/'old_checkpoint_audit.json',cross);save(OUT/'origin_selector_verification.json',dict(**origin_tests,all_value_poison_and_column_shuffle_invariant=True,all_candidate_reverse_invariant=True,all_training_updates_before_audit=0));csvwrite(OUT/'old_vs_new_origin_dispersion.csv',rows)
 save(OUT/'MASTER_MANIFEST.json',dict(queue=ORDER,specs=specs,budget=dict(main_fits=96,main_updates=49152,discarded_updates=48,total_updates=49200,forwards=200000,controller_seconds=86400,wait_seconds=1800,cache_bytes=100*2**30,disk_free_bytes=10*2**30),unaffected=['R05','N06','R09'],prepared_at=time.time(),contract_sha256=sha(OUT/'MASTER_PROTOCOL.md'),closest_baselines=CLOSEST,sampling_sensitivity='sign reversal primary closest contrast; no post-result magnitude threshold'))
 from .checks import check_all
 check_all()
 parity={}
 for name in ['model.py','evaluate.py']:
  parity[name]=sha(EXP/name)==sha(ROOT/'experiments/condition_studies_v1_20260916'/name);assert parity[name]
 # CPU packet builders/statistics/formulas unchanged; only pick provider differs.
 oa=ast.parse((ROOT/'experiments/condition_studies_v1_20260916/prepare.py').read_text());na=ast.parse((EXP/'prepare.py').read_text())
 for name in ['prepare_track','spectral','observation_map']:
  oldfun=next(v for v in oa.body if isinstance(v,ast.FunctionDef) and v.name==name);newfun=next(v for v in na.body if isinstance(v,ast.FunctionDef) and v.name==name);assert ast.dump(oldfun)==ast.dump(newfun);parity[name]=True
 oldruntime=ast.parse((ROOT/'experiments/condition_studies_v1_20260916/runtime.py').read_text());newruntime=ast.parse((EXP/'runtime.py').read_text());oc=next(v for v in oldruntime.body if isinstance(v,ast.ClassDef) and v.name=='Controller');nc=next(v for v in newruntime.body if isinstance(v,ast.ClassDef) and v.name=='Controller')
 for name in ['predict','frozen_cache','smoke','checkpoint','best','select_seed']:
  assert ast.dump(next(v for v in oc.body if isinstance(v,ast.FunctionDef) and v.name==name))==ast.dump(next(v for v in nc.body if isinstance(v,ast.FunctionDef) and v.name==name));parity[name]=True
 save(OUT/'implementation_parity.json',dict(passed=True,exact=parity,other_changes='origin selector, scope/caps, phase orchestration, cross-evaluation and reporting only',old_code_sha256=oldcode))
 from .report import origin_report,report_all
 origin_report();save(OUT/'QUEUE_STATUS.json',dict(queue=ORDER,current=None,status='PREPARED',completed=[]));report_all()
 oldseal=read(OLD/'MASTER_SEAL.json');shutil.copy2(OLD/'environment_receipt.json',OUT/'environment_receipt.json')
 files={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name in ['MASTER_PROTOCOL.md','MASTER_MANIFEST.json','origins.json','origins.csv','new_origins.csv','old_origins.csv','origin_diversity.json','data_receipt.json','permissions.json','train_statistics.json','feature_or_transform_manifest.json','PROTOCOL.json','old_checkpoint_audit.json','origin_selector_verification.json','historical_hashes.json','implementation_parity.json']}
 for spec in specs:files.update(spec['data']['packet_hashes'])
 implementation={str(p.relative_to(ROOT)):sha(p) for p in EXP.glob('*.py')};implementation['scripts/run_condition_sampling_repair.py']=sha(ROOT/'scripts/run_condition_sampling_repair.py')
 save(OUT/'MASTER_SEAL.json',dict(at=time.time(),files=files,implementation=implementation,installed_code=oldseal['installed_code'],model_files=oldseal['model_files'],all_six_specs_fixed=True,training_updates_before_seal=0,E_scored_before_seal=False,active_tracks=[t for t in ORDER if read(OUT/t/'origin_diversity.json')['passed']]))
 print('REPAIR_SEALED',read(OUT/'MASTER_SEAL.json')['active_tracks'],flush=True)
