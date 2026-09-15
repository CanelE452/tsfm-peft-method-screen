"""Audit finished pilot and enrich human report. No inference or optimizer calls."""
import csv,hashlib,json,math,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/building_coldstart_coverage_v1_20260915'))
from core import *
def main():
    status=read(OUT/'status.json');assert status['status'].startswith(('STOP_','HELDOUT_','EXECUTION_','INSUFFICIENT_'))
    v=read(OUT/'independent_verification.json');fs=read(OUT/'fit_attempts.json') if (OUT/'fit_attempts.json').exists() else [];smoke=read(OUT/'smoke_updates.json') if (OUT/'smoke_updates.json').exists() else []
    seal=read(OUT/'prepare_seal.json') if (OUT/'prepare_seal.json').exists() else {}
    for p,h in seal.get('model_files',{}).items():assert sha(p)==h,p
    if seal:check_seal()
    cfg=read(CONFIG);sp=read(OUT/'building_split.json');episodes=read(OUT/'episodes.json');by={e['id']:e for e in episodes}
    prediction_rows=read(OUT/'prediction_manifest.json') if (OUT/'prediction_manifest.json').exists() else []
    assert len({r['id'] for r in prediction_rows})==len(prediction_rows)
    if (OUT/'recipe_seal.json').exists() and 'lr' in read(OUT/'recipe_seal.json'):
        r=read(OUT/'recipe_seal.json');rr=csvread(OUT/'recipe_selection.csv');choices=[]
        for lr in cfg['lr_grid']:
            for s in cfg['checkpoint_updates']:
                sub=[x for x in rr if float(x['lr'])==lr and int(x['updates'])==s];assert {x['building_id'] for x in sub}==set(sp['recipe']);choices.append((sum(float(x['primary']) for x in sub)/4,s,lr))
        best=min(choices);assert r['updates']==best[1] and r['lr']==best[2]
        for f in fs:
            if f['phase']!='recipe':assert f['lr']==r['lr'] and f['planned_updates']==r['updates']
    if (OUT/'stageA_gate.json').exists():
        ints=[];lookup={(r['episode'],r['method']):r for r in prediction_rows if r['role']=='dev'}
        for bid in sp['dev']:
            g={}
            for e in episodes:
                if e['building_id']!=bid:continue
                f=lookup[e['id'],'F0']['scaled_RMSE'];l=lookup[e['id'],'STANDARD']['scaled_RMSE'];g[e['history_days'],e['forecast_type']]=100*(1-l/f)
            ints.append((g[3,'Wednesday']-g[3,'Saturday'])-(g[14,'Wednesday']-g[14,'Saturday']))
        gate=read(OUT/'stageA_gate.json');assert math.isclose(sum(ints)/4,gate['mean_interaction_pp'],rel_tol=1e-10,abs_tol=1e-10);assert sum(x>0 for x in ints)==gate['positive_buildings'];passed=sum(x>0 for x in ints)>=3 and sum(ints)/4>=.5;assert passed==(gate['status']=='COVERAGE_PROBLEM_SIGNAL')
        if not passed:assert not any(f['phase']=='heldout' for f in fs);assert not any(r['method'].startswith(('COVERAGE','UNIFORM','BINARY')) for r in prediction_rows)
    rules_verified=0
    if (OUT/'stageB_selection.json').exists() and read(OUT/'stageB_selection.json')['status']!='NOT_RUN':
        from analysis import scalar_metrics
        sel=read(OUT/'stageB_selection.json');rr=csvread(OUT/'stageB_rules.csv');parents={(r['episode'],r['method']):r for r in prediction_rows if r['role']=='dev'}
        for row in rr:
            method=row['method']
            if method in ['F0','STANDARD','AFFINE']:continue
            eid=row['episode'];e=by[eid]
            alpha=0. if method=='BINARY_FALLBACK' and e['coverage_count']==0 else 1. if method=='BINARY_FALLBACK' else float(method.split('_')[1]) if method.startswith('UNIFORM_') else e['coverage_count']/(e['coverage_count']+float(method.split('_')[1]))
            with np.load(ROOT/parents[eid,'F0']['path']) as d:q0=d['q'];y=d['target'];h=d['history']
            with np.load(ROOT/parents[eid,'STANDARD']['path']) as d:ql=d['q']
            val=scalar_metrics((1-alpha)*q0+alpha*ql,y,h)['scaled_RMSE'];assert math.isclose(val,float(row['scaled_RMSE']),rel_tol=1e-10,abs_tol=1e-10);rules_verified+=1
        def avg(method,h=None):
            vals=[float(row['scaled_RMSE']) for row in rr if row['method']==method and (h is None or int(row['history_days'])==h)];return sum(vals)/len(vals)
        alpha=min(cfg['alpha_grid'],key=lambda x:(avg(f'UNIFORM_{x}'),x));tau=min(cfg['tau_grid'],key=lambda x:(avg(f'COVERAGE_{x}'),x));assert (alpha,tau)==(sel['alpha'],sel['tau'])
        cov=f'COVERAGE_{tau}';simple=min([f'UNIFORM_{alpha}','BINARY_FALLBACK','AFFINE'],key=lambda x:(avg(x,3),x));gstd=100*(1-avg(cov,3)/avg('STANDARD',3));gsimple=100*(1-avg(cov,3)/avg(simple,3));degrade=100*(avg(cov,14)/avg('STANDARD',14)-1);positives=0
        for bid in sp['dev']:
            def cell(method):return next(float(row['scaled_RMSE']) for row in rr if row['method']==method and row['building_id']==bid and int(row['history_days'])==3 and row['forecast_type']=='Saturday')
            positives+=cell(cov)<cell(simple)
        assert (gstd>=.3 and gsimple>=.3 and positives>=3 and degrade<=.2)==(sel['status']=='METHOD_SIGNAL')
        if sel['status']!='METHOD_SIGNAL':assert not any(f['phase']=='heldout' for f in fs)
    if not (OUT/'heldout_seal.json').exists():assert not any(r['role']=='heldout' for r in prediction_rows)
    quantile_crossings=0;median_changes={};descriptive={}
    for row in prediction_rows:
        if row['method'] not in ['F0','STANDARD','RECIPE']:continue
        with np.load(ROOT/row['path']) as d:
            assert np.array_equal(d['q'],np.sort(d['raw'],axis=0));quantile_crossings+=int((np.diff(d['raw'],axis=0)<0).sum());median_changes[row['role']]=median_changes.get(row['role'],0)+int((d['q'][10]!=d['raw'][10]).sum())
    dev=[row for row in prediction_rows if row['role']=='dev']
    if len(dev)==48:
        byep={(r['episode'],r['method']):r['scaled_RMSE'] for r in dev};dev_ids=sorted({r['episode'] for r in dev});f0=[byep[e,'F0'] for e in dev_ids];lora=[byep[e,'STANDARD'] for e in dev_ids];descriptive=dict(tag='확인',F0_macro=sum(f0)/16,LoRA_macro=sum(lora)/16,LoRA_macro_gain_percent=100*(1-sum(lora)/sum(f0)),LoRA_positive_episodes=sum(l<f for l,f in zip(lora,f0)),episodes=16,native_quantile_crossings=quantile_crossings,median_changes_by_role=median_changes)
        save(OUT/'descriptive_summary.json',descriptive)
    if fs:
        assert len({f['initial_lora_hash'] for f in fs if 'initial_lora_hash' in f})==1
        assert len({f['frozen_hash_before'] for f in fs if 'frozen_hash_before' in f})==1
        assert all(f['trainable_count']==1179648 for f in fs if 'trainable_count' in f)
    # Independently verify per-update recipe sample selection and temporal boundaries.
    for f in fs:
        e=by[f['episode']];h=np.load(ROOT/e['history_file']);origin=pd.Timestamp(e['origin']);start=origin-pd.Timedelta(hours=len(h));n=len(h)//24-1
        train_n=n-2 if f['phase']=='recipe' else n
        rr=csvread(OUT/(f['fit']+'_trajectory.csv'));assert len(rr)==f['actual_updates']
        rng=np.random.default_rng(61601);order=[]
        while len(order)<f['planned_updates']:order.extend(rng.permutation(train_n).tolist())
        for j,row in enumerate(rr):
            k=int(row['window']);assert k==order[j] and 0<=k<train_n;target_start=start+pd.Timedelta(hours=24*(k+1));assert start<=target_start-pd.Timedelta(hours=24) and target_start+pd.Timedelta(hours=24)<=origin
    gpu={}
    for p in OUT.glob('gpu_*.jsonl'):
        rr=[json.loads(l) for l in p.read_text().splitlines()];gpu[p.name]=dict(samples=len(rr),min_free_mib=min(r['free_mib'] for r in rr),unapproved_external_samples=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in r['apps']) for r in rr),busy_samples=sum(r['busy'] for r in rr))
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip();own=[]
    for line in apps.splitlines():
        p=Path('/proc')/line.split(',')[0].strip()/'cmdline'
        if p.exists() and 'run_building_coldstart_coverage_v1' in p.read_bytes().decode(errors='replace'):own.append(line)
    assert not own
    p=OUT/'REPORT.md';s=p.read_text();marker='## 최종 감사와 실제 점수 요약'
    if marker in s:s=s.split(marker)[0].rstrip()+'\n'
    lines=['',marker,'',f"[확인] Smoke 실제 {sum(r['updates'] for r in smoke)} updates와 LoRA 학습 {sum(f['actual_updates'] for f in fs)} updates를 분리했다. 실제 학습 시도 {len(fs)}/48, 완료 {sum(f['status']=='COMPLETE' for f in fs)}. 미실행 허용 범위 {48-len(fs)} fits는 중단 gate 이후 투자 한도이며 수행한 실험으로 세지 않는다."]
    if descriptive:
        lines += ['',f"[확인] 공통 출력 정렬을 적용한 Standard LoRA는 개발16개 중 {descriptive['LoRA_positive_episodes']}개에서 F0보다 정확했다. 전체 macro scaled_RMSE는 F0 {descriptive['F0_macro']:.6f} → LoRA {descriptive['LoRA_macro']:.6f}, 개선 {descriptive['LoRA_macro_gain_percent']:.4f}%다. 따라서 일반 LoRA 적응 이득과 coverage 가설의 재현성은 다른 결론이다.",'','[확인] H3 Saturday에서는 Kristine·Rebecca가 LoRA로 개선됐고 Cora·Hugh는 악화됐다. Affine은 Hugh에서 F0 대비 +3.0505%로 LoRA 손해를 없앴지만 Cora에서는 -33.1382%로 더 나빴다. 단일 baseline이 모든 건물 문제를 해결했다는 근거도 없다.','',f"[확인] Native quantile crossing {quantile_crossings}개를 기록했고 공통 사전 정렬로 median이 달라진 hourly 지점 수는 role별 {median_changes}다. 정렬 없는 원출력도 cache에 보존했다.",'','[추정] 평균 I가 양수인 것만으로 건물 간 반복성을 입증할 수 없다. 이번 규칙의 후속 조건은 2/4로 미충족이었다. Stage B를 실행하지 않았으므로 coverage scaling 자체의 성능 실패나 성공은 아직 측정하지 않았다.']
    if (OUT/'preparation_repair.json').exists():lines+=['','[확인] 첫 smoke는 공식 pipeline에 CUDA 입력을 전달해 CPU pin-memory 오류로 종료됐다. 이때 optimizer0·fits0이었다. 같은 입력값을 CPU tensor로 전달하도록 API 연결만 수정한 뒤 smoke를 재검사했다. [원래 오류](preparation_attempt1/execution_error.json), [수정 기록](preparation_repair.json). 데이터·학습률·기준 변경은 없었다.']
    rr=[r for r in prediction_rows if r['role']=='dev'];methods=['F0','STANDARD','AFFINE']
    if len(rr)==48:
        lines+=['','[확인] 개발16개 episode의 실제 scaled_RMSE(낮을수록 좋음):','', '| 건물 | 요일 | H일 | F0 | LoRA | Affine |','| --- | --- | ---: | ---: | ---: | ---: |']
        for e in episodes:
            if e['role']!='dev':continue
            lookup={r['method']:r['scaled_RMSE'] for r in rr if r['episode']==e['id']};lines.append(f"| {e['building_id']} | {e['forecast_type']} | {e['history_days']} | {lookup['F0']:.6f} | {lookup['STANDARD']:.6f} | {lookup['AFFINE']:.6f} |")
        lines+=['','[확인] H×요일별 건물 macro scaled_RMSE:','', '| H일 | 요일 | F0 | LoRA | Affine |','| --- | --- | ---: | ---: | ---: |']
        for h in [3,14]:
            for day in ['Wednesday','Saturday']:
                vals=[np.mean([r['scaled_RMSE'] for r in rr if r['history_days']==h and r['forecast_type']==day and r['method']==m]) for m in methods];lines.append(f"| {h} | {day} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} |")
    lines+=['','[확인] Stage A 원점수 CSV에는 각 건물·episode·방법의 raw RMSE, raw MAE, scaled 2-pinball도 포함된다. 서로 다른 건물의 raw 단위 오차를 같은 의미로 평균하지 않았다. Official-style NRMSE는 별도 secondary로 100×RMSE/평가 target 평균을 기록했다. 이 미래 평균은 평가 표시용이며 학습·선택·primary 분모에는 사용하지 않았다. 공식 전체 기간 benchmark 집계와 동일한 결과라고 하지 않는다.']
    if fs:
        lines+=['','[확인] 실제 자원 합계와 fit 최대값:','', '| phase | active 학습초 합계 | fit wall초 합계 | 최대 allocated GiB | 최대 reserved GiB |','| --- | ---: | ---: | ---: | ---: |']
        for phase in ['recipe','screen','heldout']:
            rr=[f for f in fs if f['phase']==phase]
            if rr:lines.append(f"| {phase} | {sum(f['active_train_seconds'] for f in rr):.3f} | {sum(f['wall_seconds'] for f in rr):.3f} | {max(f['peak_allocated'] for f in rr)/2**30:.4f} | {max(f['peak_reserved'] for f in rr)/2**30:.4f} |")
    lines+=['',f"[확인] GPU 원본 로그의 승인되지 않은 외부 compute 샘플 합계 {sum(x['unapproved_external_samples'] for x in gpu.values())}. RustDesk만 기존 예외로 인정했다. 현재 이 파일럿 GPU worker는 없다.",'','[확인] [publication 감사](publication_audit.json)에서 기존 파일 hash, 선택 recipe, Stage A 산식, window 순서·시간 경계, model/source/data hash를 추가 검사했다.']
    p.write_text(s+'\n'.join(lines).rstrip()+'\n')
    result=dict(tag='확인',at=time.time(),status=status,smoke_updates=sum(r['updates'] for r in smoke),fits=len(fs),updates=sum(f['actual_updates'] for f in fs),model_hashes_verified=len(seal.get('model_files',{})),source_and_data_verified=bool(seal),window_orders_and_temporal_boundaries_verified=True,gpu_logs=gpu,current_compute_apps=apps,own_workers_remaining=own,historical_files_preserved=v['historical_files_preserved'],prediction_records_verified=v['prediction_records'],stageB_rule_rows_verified=rules_verified,no_gpu_work_during_audit=True)
    save(OUT/'publication_audit.json',result)
    artifacts=[p for p in OUT.rglob('*') if p.is_file() and p.name!='publication_audit.json']+list((EXP).glob('*.py'))+[CONFIG,Path(__file__)]
    result['artifacts']={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in artifacts};save(OUT/'publication_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts'},indent=2))
if __name__=='__main__':main()
