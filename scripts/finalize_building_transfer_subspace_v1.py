"""Independent score/selection audit and Korean report for the finite transfer study."""
import csv,importlib.util,json,math,sys,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN='building_transfer_subspace_v1_20260915';OUT=ROOT/'results'/RUN
sys.path.insert(0,str(ROOT/'experiments'/RUN));import run as r

def main():
    seal=r.read(OUT/'seal.json');patch=r.read(OUT/'post_run_patch.json')
    for kind in ['sources','data','manifests']:
        for name,expected in seal[kind].items():
            if name==patch['original']:
                snapshot=ROOT/patch['executed_snapshot'];assert r.sha(snapshot)==expected==patch['executed_sha256']
                assert r.sha(ROOT/name)==patch['current_sha256']
                assert (ROOT/name).read_text()==snapshot.read_text().replace(patch['exact_old'],patch['exact_new'])
            else:assert r.sha(ROOT/name)==expected,(kind,name)
    terminal=r.read(OUT/'terminal.json')
    if terminal['status'] in ['EXECUTION_ERROR','BLOCKED_COMMON_RESOURCE']:
        fits=[r.read(p) for p in sorted((OUT/'fits').glob('*.json'))]
        updates=sum(f.get('updates',f.get('completed_updates',0)) for f in fits)
        history=r.read(OUT/'historical_hashes.json')
        unchanged=all((ROOT/p).exists() and r.sha(ROOT/p)==h for p,h in history.items())
        record=dict(status=terminal['status'],fit_records=len(fits),completed_fits=sum(f['status']=='COMPLETED' for f in fits),optimizer_updates=updates,historical_unchanged=unchanged,performance_failure=False,no_automatic_retry=True,error=terminal['error'])
        r.save(OUT/'execution_summary.json',record)
        (OUT/'REPORT.md').write_text('# 새 건물 전이 실험 — 실행 중단 기록\n\n'
            +f"상태 **{terminal['status']}**. 완료 fits {record['completed_fits']}, 실제 updates {updates}. 이는 성능 실패가 아니다.\n\n"
            +f"오류: `{terminal['error']}`. 자동 재시도·설정 변경은 하지 않았다. [상세 traceback](terminal.json), [실행 집계](execution_summary.json), [사전 프로토콜](PROTOCOL.md), [원장](fit_ledger.jsonl)을 보존한다.\n\n"
            +'미완료 비교군의 성능·자원 이득·새 방법의 추가 가치는 판정하지 않는다. 일부 scores.csv가 있더라도 미완료 범위를 전체 결과로 일반화하지 않는다. 기존 heldout은 미실행이며 새 방향의 신규성은 프로토콜의 선행연구 대비 한계를 그대로 갖는다.\n')
        print(json.dumps(record));return
    episodes=r.read(OUT/'episodes.json');emap={e['id']:e for e in episodes};split=r.read(OUT/'split.json')
    assert not (set(split['source'])|set(split['tune'])|set(split['dev']))&set(split['excluded_previous_14'])
    assert len(set(split['source']+split['tune']+split['dev']))==24
    rows=list(csv.DictReader((OUT/'scores.csv').open()));fits=[r.read(p) for p in sorted((OUT/'fits').glob('*.json'))]
    events=[json.loads(s) for s in (OUT/'fit_ledger.jsonl').read_text().splitlines()]
    assert len(fits)==sum(e['event']=='START' for e in events)==sum(e['event']=='COMPLETE' for e in events)<=93
    assert all(f['status']=='COMPLETED' and f['frozen_unchanged'] and f['buffers_unchanged'] and f['reload_exact'] for f in fits)
    updates=sum(f['updates'] for f in fits);assert updates<=18400
    # Independent scalar implementation: do not call production metrics.
    maxdiff=0.;metric_count=0
    for row in rows:
        p=ROOT/row['replay_file'];assert r.sha(p)==row['replay_sha256'];z=np.load(p);q=z['q'];y=z['y'];h=z['h']
        assert np.array_equal(h,r.load(emap[row['episode']]['history'])) and np.array_equal(y,r.load(emap[row['episode']]['target']))
        mean=sum(float(v) for v in h)/len(h);sd=max(math.sqrt(sum((float(v)-mean)**2 for v in h)/len(h)),1e-6)
        rmse=math.sqrt(sum((float(q[10,j])-float(y[j]))**2 for j in range(24))/24)
        mae=sum(abs(float(q[10,j])-float(y[j])) for j in range(24))/24
        pin=0.
        for i,tau in enumerate(r.pure.QUANTILES):
            for j in range(24):
                delta=float(y[j])-float(q[i,j]);pin+=2*max(tau*delta,(tau-1)*delta)
        values=dict(scaled_RMSE=rmse/sd,raw_RMSE=rmse,raw_MAE=mae,scaled_2pinball=pin/(504*sd),history_std=sd)
        for k,v in values.items():
            error=abs(v-float(row[k]));maxdiff=max(maxdiff,error);assert error<1e-9,(row['episode'],row['arm'],k,error);metric_count+=1
    crossings=0;cpcount=0;median_changes=0
    for f in fits:
        assert len(f['losses'])==f['updates']
        for c in f['checkpoints'].values():
            p=ROOT/c['prediction_file'];assert r.sha(p)==c['prediction_sha256'];z=np.load(p)
            assert np.array_equal(z['q'],np.sort(z['raw'],axis=1));crossings+=int(np.sum(np.diff(z['raw'],axis=1)<0));median_changes+=int(np.sum(z['q'][:,10]!=z['raw'][:,10]))
            p=ROOT/c['checkpoint_file'];assert r.sha(p)==c['checkpoint_sha256'];cpcount+=1
    selection_audit={}
    for prefix in ['local_','warm_','coeff_']:
        path=OUT/f'{prefix}selection.json'
        if not path.exists():continue
        rec=r.read(path);v=[z for z in rows if z['role']=='tune' and z['arm'].startswith(prefix)];arms=sorted({z['arm'] for z in v})
        summaries=[]
        for arm in arms:
            rr=[z for z in v if z['arm']==arm];assert len(rr)==8
            summaries.append(dict(arm=arm,macro_scaled_RMSE=sum(float(z['scaled_RMSE']) for z in rr)/8,mean_steps=sum(int(z['step']) for z in rr)/8,rank=int(arm.split('_r')[1].split('_')[0])))
        selected=min(summaries,key=lambda z:(z['macro_scaled_RMSE'],z['mean_steps'],z['rank'],z['arm']))
        assert selected['arm']==rec['selected']['arm'];selection_audit[prefix]=selected
    # Independent source validation selection replay from checkpoint forecasts.
    sourcechecks={};sources=r.read(OUT/'sources.json');smap={s['id']:s for s in sources}
    for f in fits:
        if not (f['id']=='source_pooled' or f['id'].startswith('direction_')):continue
        ss=sources if f['id']=='source_pooled' else [smap[f['id'][10:]]];curve=[]
        for step,c in f['checkpoints'].items():
            qs=np.load(ROOT/c['prediction_file'])['q'];idx=0;vals=[]
            for s in ss:
                h=r.load(s['train']);vy=r.load(s['validation']).reshape(14,24);std=float(h.std())
                vals.append(sum(math.sqrt(sum((float(qs[idx+j,10,k])-float(y[k]))**2 for k in range(24))/24)/std for j,y in enumerate(vy))/14);idx+=14
            curve.append((sum(vals)/len(vals),int(step)))
        chosen=min(curve)[1];assert chosen==r.read(OUT/(f['id']+'_selection.json'))['selected']['step'];sourcechecks[f['id']]=chosen
    historical=r.read(OUT/'historical_hashes.json')
    assert all((ROOT/p).exists() and r.sha(ROOT/p)==h for p,h in historical.items())
    # Fresh model reconstruction checks; includes saved CPU tensors, beyond in-place training reload.
    r.setup();w=r.Watch(OUT,'audit',wall_cap=10800);fresh=[]
    try:
        w.boundary(startup=True)
        for prefix in ['local_dev_','warm_dev_','coeff_dev_']:
            matches=[f for f in fits if f['id'].startswith(prefix)]
            if not matches:continue
            f=matches[0];e=emap[f['id'][len(prefix):]];k=f['updates'];h=r.load(e['history']);pool=None;bank=None
            if prefix=='coeff_dev_':
                poolrec=next(z for z in fits if z['id']=='source_pooled');pool=r.state(poolrec,sourcechecks['source_pooled']);bank=[]
                for d in r.read(OUT/'direction_selection.json')['selected']:
                    src=next(z for z in fits if z['id']=='direction_'+d['building']);bank.append(r.state(src,d['step']))
            m=r.make(f['rank'],pool=pool,sources=bank);r.restore(m,r.state(f,k));raw=r.predict(m,h[-24:],w)[1];expected=np.load(ROOT/f['checkpoints'][str(k)]['prediction_file'])['raw'][0]
            err=float(np.max(np.abs(raw-expected)));assert err==0,(prefix,err);fresh.append(dict(fit=f['id'],step=k,max_abs=err));del m;r.cleanup()
    finally:w.close()
    gpu=[json.loads(s) for s in (OUT/'gpu_study.jsonl').read_text().splitlines()]
    external=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in x['apps']) for x in gpu)
    assert external==0,'Resource contamination; report without clean benchmark claims'
    verification=dict(status='PASS',executed_sources_verified=True,sealed_data_unchanged=True,post_run_patch=patch,score_rows=len(rows),scalar_metric_values=metric_count,max_abs_error=maxdiff,checkpoint_files=cpcount,raw_quantile_crossings=crossings,rearranged_median_values=median_changes,recipe_replay=selection_audit,source_selection_replay=sourcechecks,fresh_model_reload=fresh,historical_files_unchanged=len(historical),old_heldout_values_loaded=False,previous_buildings_excluded=14,unapproved_external_compute_samples=external,scope='Numerical replay uses ignored local npz and checkpoint cache; GitHub alone does not contain raw data/model/cache.')
    r.save(OUT/'independent_verification.json',verification)
    typed=[]
    for z in rows:
        z=dict(z)
        for k in ['scaled_RMSE','raw_RMSE','raw_MAE','scaled_2pinball','train_seconds']:z[k]=float(z[k])
        z['days']=int(z['days']);z['step']=int(z['step']);typed.append(z)
    d=[z for z in typed if z['role']=='dev'];arms=sorted({z['arm'] for z in d});means={a:np.mean([z['scaled_RMSE'] for z in d if z['arm']==a]) for a in arms};f0=means['F0']
    summary=[]
    for a in arms:
        v=[z for z in d if z['arm']==a];assert len(v)==16
        summary.append(dict(arm=a,episodes=16,scaled_RMSE=float(means[a]),raw_RMSE=float(np.mean([z['raw_RMSE'] for z in v])),raw_MAE=float(np.mean([z['raw_MAE'] for z in v])),scaled_2pinball=float(np.mean([z['scaled_2pinball'] for z in v])),gain_vs_F0_percent=float(100*(f0-means[a])/f0),target_updates=sum(z['step'] for z in v),target_training_seconds=sum(z['train_seconds'] for z in v),H3=float(np.mean([z['scaled_RMSE'] for z in v if z['days']==3])),H14=float(np.mean([z['scaled_RMSE'] for z in v if z['days']==14]))))
    r.csvwrite(OUT/'development_summary.csv',summary)
    group=[];lookup={(z['episode'],z['arm']):z for z in d}
    for bid in split['dev']:
        for a in arms:
            vals=[lookup[(f'{bid}_H{h}',a)]['scaled_RMSE'] for h in [3,14]];base=[lookup[(f'{bid}_H{h}','F0')]['scaled_RMSE'] for h in [3,14]]
            group.append(dict(building=bid,arm=a,H3=vals[0],H14=vals[1],mean=float(np.mean(vals)),gain_vs_F0_percent=float(100*(np.mean(base)-np.mean(vals))/np.mean(base))))
    r.csvwrite(OUT/'building_summary.csv',group)
    resource_rows=[]
    for rank in [1,8]:
        rr=[f for f in fits if f['id'].startswith('local_tune_') and f['rank']==rank]
        resource_rows.append(dict(rank=rank,trainable_parameters=147456*rank,fits=len(rr),median_peak_allocated_bytes=float(np.median([f['peak_allocated_bytes'] for f in rr])),median_gradient_seconds_per_update=float(np.median([f['training_seconds']/f['updates'] for f in rr]))))
    r.csvwrite(OUT/'rank_resource_comparison.csv',resource_rows)
    budget_rows=[]
    for bid in split['dev']:
        for day in [3,14]:
            local=lookup[(f'{bid}_H{day}','LOCAL')];fixed=lookup[(f'{bid}_H{day}','LOCAL_FIXED120')]
            budget_rows.append(dict(building=bid,history_days=day,selected_steps=local['step'],fixed_steps=120,selected_scaled_RMSE=local['scaled_RMSE'],fixed_scaled_RMSE=fixed['scaled_RMSE'],gain_selected_vs_fixed_percent=100*(fixed['scaled_RMSE']-local['scaled_RMSE'])/fixed['scaled_RMSE']))
    r.csvwrite(OUT/'budget_control_comparison.csv',budget_rows)

    gate=r.read(OUT/'transfer_gate.json');rng=np.random.default_rng(20260915);draw=rng.integers(0,8,size=(10000,8));b=np.array([np.mean([lookup[(f'{bid}_H{h}',gate['simple'])]['scaled_RMSE'] for h in [3,14]]) for bid in split['dev']]);t=np.array([np.mean([lookup[(f'{bid}_H{h}',gate['transfer'])]['scaled_RMSE'] for h in [3,14]]) for bid in split['dev']]);boot=100*(b[draw].mean(1)-t[draw].mean(1))/b[draw].mean(1);ci=np.percentile(boot,[2.5,97.5]).tolist()
    r.save(OUT/'descriptive_building_bootstrap.json',dict(seed=20260915,resamples=10000,paired_building_percentile95=ci,not_used_for_gate=True,limitation='Only eight development buildings from one dataset; descriptive interval, no independent confirmation.'))
    replay_gain=100*(float(b.mean())-float(t.mean()))/float(b.mean())
    replay_wins=int(np.sum(t<b));replay_history={}
    for h in [3,14]:
        bb=[lookup[(f'{bid}_H{h}',gate['simple'])]['scaled_RMSE'] for bid in split['dev']]
        tt=[lookup[(f'{bid}_H{h}',gate['transfer'])]['scaled_RMSE'] for bid in split['dev']]
        replay_history[str(h)]=100*(sum(bb)-sum(tt))/sum(bb)
    replay_pass=bool(replay_gain>=1 and replay_wins>=6 and replay_history['3']>0 and replay_history['14']>=-1)
    assert abs(replay_gain-gate['macro_gain_percent'])<1e-10 and replay_wins==gate['positive_buildings'] and replay_pass==gate['pass']
    for h,g in replay_history.items():assert abs(g-gate['history_gain_percent'][h])<1e-10
    deployment=r.read(OUT/'deployment_selection.json');lr=r.read(OUT/'local_selection.json')['selected']['arm'];wr=r.read(OUT/'warm_selection.json')['selected']['arm']
    for key,arms0 in [('simple',['F0','AFFINE',lr]),('transfer',['POOLED','POOLED_AFFINE',wr])]:
        chosen=min(arms0,key=lambda a:(sum(z['scaled_RMSE'] for z in typed if z['role']=='tune' and z['arm']==a)/8,a));assert chosen==deployment[key]
    verification['deployment_and_transfer_gate_replay']=dict(status='PASS',macro_gain_percent=replay_gain,positive_buildings=replay_wins,history_gain_percent=replay_history,gate_pass=replay_pass)
    r.save(OUT/'independent_verification.json',verification)
    stages={}

    for f in fits:
        stage='source' if f['id']=='source_pooled' else 'source_directions' if f['id'].startswith('direction_') else '_'.join(f['id'].split('_')[:2])
        z=stages.setdefault(stage,dict(fits=0,updates=0,training_seconds=0.,wall_seconds=0.,peak_allocated_bytes=0))
        z['fits']+=1;z['updates']+=f['updates'];z['training_seconds']+=f['training_seconds'];z['wall_seconds']+=f['wall_seconds'];z['peak_allocated_bytes']=max(z['peak_allocated_bytes'],f['peak_allocated_bytes'])
    r.save(OUT/'execution_summary.json',dict(status=terminal['status'],attempted_fits=len(fits),completed_fits=len(fits),failed_fits=0,optimizer_updates=updates,full_model_smoke_optimizer_updates=0,max_fits=93,max_updates=18400,stages=stages,minimum_gpu_free_mib=min(z['free_mib'] for z in gpu),peak_torch_allocated_bytes=max(f['peak_allocated_bytes'] for f in fits),wall=r.read(OUT/'study_wall.json'),heldout_prediction_count=0))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.5));ordered=sorted(arms,key=lambda a:means[a]);ax.bar(ordered,[means[a] for a in ordered],color=['#346a95' if a!='F0' else '#999999' for a in ordered]);ax.set_ylabel('Mean history-scaled RMSE (lower is better)');ax.tick_params(axis='x',rotation=25);ax.set_title('8 unseen development buildings, 2 history lengths, 1 seed');fig.tight_layout();fig.savefig(OUT/'development_scores.png',dpi=150);plt.close(fig)
    recipe=r.read(OUT/'stage1_recipe.json');warm=r.read(OUT/'stage2_recipe.json');deploy=r.read(OUT/'deployment_selection.json');source=r.read(OUT/'source_pooled_selection.json')
    lines=['# 새 건물 짧은 이력 PEFT 전이 실험 — 결과 보고','','설계·실행 시작: 2026-09-15 KST. 실행 종료·검산: 2026-09-16 KST. 실험 ID의 날짜는 시작일을 유지한다.','',f"**최종 상태: `{terminal['status']}`.** 실제 {len(fits)} fits, {updates:,} optimizer updates를 완료했다. 학습 오류와 성능 판정은 구분했으며 오류 fits는 0이다. 이번 상태는 개발 단계의 고정 기준 결과이고 논문 PASS를 뜻하지 않는다.",'','## 무엇이 좋아졌는가','',f"개발 평균 scaled RMSE 기준 가장 낮은 비교군은 **{min(means,key=means.get)}**이다. F0는 {f0:.8f}, 최저값은 {min(means.values()):.8f}다. 모든 건물·이력에서의 일률적 개선을 요구하지 않는다. 아래 점수는 새 개발 건물 8개×H3/H14=16개 episode 평균이다.",'','| 비교군 | scaled RMSE ↓ | raw RMSE ↓ | raw MAE ↓ | scaled pinball ↓ | F0 대비 개선 % | H3 | H14 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for z in summary:lines.append(f"| {z['arm']} | {z['scaled_RMSE']:.8f} | {z['raw_RMSE']:.6f} | {z['raw_MAE']:.6f} | {z['scaled_2pinball']:.8f} | {z['gain_vs_F0_percent']:+.3f} | {z['H3']:.6f} | {z['H14']:.6f} |")
    lines+=['','![개발 점수 비교](development_scores.png)','','원점수는 [전체 episode scores.csv](scores.csv), [건물별 요약](building_summary.csv), [개발 평균](development_summary.csv)에 남겼다. 서로 다른 건물의 raw RMSE 단위를 합친 평균은 대형 건물의 영향을 받으며, 선택 지표는 노출 이력으로만 스케일한 RMSE다. H3/H14의 scaling 자체가 다르므로 서로 다른 H의 절대값 차이를 순수한 이력 효과라고 해석하지 않는다.','','## 학습량과 전이의 추가 가치','',f"Local은 tuning4에서 rank **{recipe['rank']}**, **{recipe['policy']}**를 선택했다. Warm은 동일 rank의 **{warm['policy']}**를 선택했다. Source pooled는 0/660/1320/2640 중 **{warm['source_step']} updates**를 source 미래 validation으로 선택했다. Source 학습은 target dev를 쓰지 않았다.",'',f"배포 비교군도 tune에서 고정했다: simple=`{deploy['simple']}`, transfer=`{deploy['transfer']}`. Dev에서 transfer는 선택 simple 대비 **{gate['macro_gain_percent']:+.3f}%**, 개선 건물 **{gate['positive_buildings']}/8**, H3 **{gate['history_gain_percent']['3']:+.3f}%**, H14 **{gate['history_gain_percent']['14']:+.3f}%**다. 고정 전이 조건 충족: **{gate['pass']}**. [기준 및 건물별 판정](transfer_gate.json). 건물 bootstrap의 서술적 95% 구간은 [{ci[0]:+.3f}, {ci[1]:+.3f}]%이며 판정 변경에 사용하지 않았다.",'',f"LOCAL 대 LOCAL_FIXED120은 같은 rank·초기화 trajectory에서 선택 학습량과 기존120 updates의 차이를 보여준다([창별 비교](budget_control_comparison.csv)). 선택 LOCAL의 평균은 {means['LOCAL']:.8f}, fixed120은 {means['LOCAL_FIXED120']:.8f}이며, fixed120 대비 개선은 {100*(means['LOCAL_FIXED120']-means['LOCAL'])/means['LOCAL_FIXED120']:+.3f}%다. F0→LOCAL은 표준 LoRA 적응, F0→AFFINE은 단순 출력 보정, LOCAL→WARM은 추가 source 학습 및 초기화 전이, POOLED→POOLED_AFFINE은 두 target 출력 계수의 가치다. 이 차이들을 새 PEFT 구조의 성과로 합쳐 주장하지 않는다.",'','[선택 순위의 역전과 원인 해석](INTERPRETATION.md): fixed120의 약8% 개선은 주지표에 한정되고 H3에서는 악화했다. 설정 선택용 건물에서의 순위가 개발 건물에서 뒤집혔으므로, 더 짧은 학습이 일반적인 해법이라는 가설도 확인되지 않았다. 원인에 관한 해석과 직접 입증된 사실을 구분했다.\n\n## 실제 비용과 미실행 범위','','| 단계 | fits | optimizer updates | 실제 학습 초 | fit wall 초 | peak allocated GiB |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,z in stages.items():lines.append(f"| {name} | {z['fits']} | {z['updates']} | {z['training_seconds']:.2f} | {z['wall_seconds']:.2f} | {z['peak_allocated_bytes']/2**30:.3f} |")
    lines+=['','| 배포 비교군 | 16 target episode의 선택 updates 합계 | 선택 시점까지 gradient 학습 초 합계 |','| --- | ---: | ---: |']
    for z in summary:lines.append(f"| {z['arm']} | {z['target_updates']} | {z['target_training_seconds']:.3f} |")
    lines+=['','0초는 target gradient 학습이 없다는 뜻이며 inference·OLS·source pretraining·탐색이 공짜라는 뜻이 아니다. 같은 trajectory의 checkpoint 비용을 arm마다 합쳐 전체 실험 비용이라고 부르면 중복 계산된다. 실제 전체 비용은 위 단계별 표를 따른다. 시간에는 GPU 안전 대기, 모델 로드·hash·저장·평가가 일부 분리되어 있으므로 gradient seconds와 전체 wall을 혼동하지 않는다. Source 정보 및 source 학습 비용을 추가한 WARM과 LOCAL은 동일 총 데이터·compute 비교가 아니다. 작은 trainable 수를 곧바로 메모리·속도 이득으로 주장하지 않는다. [동일 tuning episode의 rank별 측정](rank_resource_comparison.csv)에 trainable 수, peak allocation과 update당 시간을 함께 남겼다.','',f"최대93 fits 중 **{93-len(fits)} fits 미실행**. 사유와 범위: {terminal.get('unrun','')}. 기존 heldout6은 이번에도 예측0이며 추가 후보·seed·재튜닝을 실행하지 않았다.",'']
    if (OUT/'candidate_gate.json').exists():
        cg=r.read(OUT/'candidate_gate.json');lines+=['## 소수 계수 방식의 추가 가치','',f"Source direction + coefficient 방법의 최강 고정 대조군 `{cg['best_reported_control']}` 대비 개선은 **{cg['macro_gain_percent']:+.3f}%**, 건물 {cg['positive_buildings']}/8이다. 추가 가치 조건 충족: **{cg['pass']}**. 이는 단순 전이 효과를 넘어서는지 확인한 개발 결과이며, source bank의 추가12 fits 비용과 선행연구 충돌은 별도 남는다.",'']
    else:lines+=['## 소수 계수 방식의 추가 가치','','계수 방식의 target 학습·평가는 실행하지 않았다. 전이 또는 source direction 사전 조건 미충족이므로 **계수 방법의 성능 실패라고 기록하지 않는다**. 미실행 방법을 PASS/FAIL 또는 새로운 논문 기여로 주장하지 않는다.','']
    lines+=['## 신규성 한계와 이번 실행으로 결정할 수 있는 범위','','표준 LoRA·학습량 조절·source-pooled 초기화·공통 source delta bank는 그 자체로 신규 PEFT 방법이 아니다. [PhysioPFM (ICML2025)](https://proceedings.mlr.press/v267/wu25ah.html)은 개인화 low-rank prior/generation을 다루고, [LoRA Recycle (CVPR2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_LoRA_Recycle_Unlocking_Tuning-Free_Few-Shot_Adaptability_in_Visual_Foundation_Models_CVPR_2025_paper.pdf), [Meta-LoRA personalization](https://arxiv.org/abs/2608.12389), [MTA](https://arxiv.org/abs/2511.20072)와도 개념적 중복이 있다. 이번 개발 신호만으로 새로운 방법론 논문의 성공을 선언하지 않는다. 동일 source 정보·총 예산의 강한 대조군과 다른 원리, 새로운 건물/원천·복수 seed의 독립 확인이 남는다.','','[추가 해석 경계](BOUNDARIES.md): source에는 target origin보다 늦은 달력 시점도 포함되므로 offline building-disjoint transfer이며 당시 시점의 온라인 배포 가능성을 뜻하지 않는다. Chronos-2 사전학습 corpus 중복 부재도 입증하지 않았다. Source12/tune4/dev8 모두 하나의 공개 dataset이며 physical building은 다르지만 site가 겹친다. 건물당 한 origin, 단일 seed, 관측 연속성과 기존 eligibility로 제한한 표본이라 계절·site·결측·모든 건물로 일반화할 수 없다. 새 결과로 grid·허용오차·학습률·범위를 바꾸지 않았다. 종료 후 미실행 coefficient 분기의 비교군 목록에서 fixed120 대조군 누락을 발견해 한 줄 수정했다([수정 기록](post_run_patch.json), [실제 실행 소스](executed_source/run.py)). 실행한 단계·판정·원점수에는 영향이 없고 재학습하지 않았다. 봉인 검산은 보존된 실제 실행 소스와 수정의 정확한 한 줄 차이를 모두 확인한다. 과거 결과를 지우거나 새 기준으로 소급 PASS 처리하지 않았다.','','## 검증과 재현 범위','',f"사전 CPU 계약 검사6개, 종료 후 미실행 분기의 대조군 누락 수정 회귀 검사를 포함한 CPU7/7 및 native pipeline/target poison/rank parity/bank 식·gradient smoke를 통과했다. {len(fits)} fits의 frozen parameter 및 buffer 불변과 저장 tensor 복원을 확인했다. 별도 fresh model 재구성 {len(fresh)}건도 예측이 정확히 같았다. 저장 예측 {len(rows)}행의 독립 scalar metric {metric_count}개 최대 오차는 {maxdiff:.3g}, source/tune 선택 재현도 일치했다. 이전 파일 **{len(historical)}개 해시 유지**. [독립 검산](independent_verification.json), [실행 집계](execution_summary.json), [사전 프로토콜](PROTOCOL.md), [봉인](seal.json).",'',f"감시 중 비승인 외부 compute sample0, 최소 GPU free {min(z['free_mib'] for z in gpu)}MiB. RustDesk만 기존 승인 예외다. 원시 wall.external_compute_samples는 승인된 RustDesk 표본까지 포함하며, 비승인 표본 수와 구분한다. raw quantile crossing {crossings}개, rearrangement로 달라진 median 값 {median_changes}개를 기록했다(여러 체크포인트/학습창 반복 예측 포함, 독립 test 사례 수가 아님).",'','데이터·모델·prediction npz와 checkpoint는 로컬 ignored cache다. GitHub에는 코드, protocol, manifest, 원점수, 검산 및 본 보고서를 보관한다. GitHub 파일만으로 로컬 numerical replay가 모두 가능하다고 주장하지 않는다.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(status=terminal['status'],fits=len(fits),updates=updates,summary=summary,gate=gate,verification=verification),indent=2))
if __name__=='__main__':main()
