"""Independent scalar checks, paired block uncertainty and finite decision report."""
from runtime import *

def scalar_metrics(p,y,std):
    ms=[];ma=[];raw=[]
    for ch in range(y.shape[1]):
        valid=np.isfinite(y[:,ch]);a=p[:,ch][valid].astype(np.float64);b=y[:,ch][valid].astype(np.float64)
        if len(b):
            errors=a-b;ms.append(math.fsum(float(v)*float(v) for v in errors)/len(b));ma.append(math.fsum(abs(float(v)) for v in errors)/len(b));raw.append(ma[-1]*float(std[ch]))
    return dict(mse=math.fsum(ms)/len(ms),mae=math.fsum(ma)/len(ma),raw_mae=math.fsum(raw)/len(raw))

def verify_selection(curves,selected):
    checks=0
    for d in ['electricity','traffic']:
        for a in ARMS:
            records=[r for r in curves if r['dataset']==d and r['arm']==a]
            minima={}
            for lr in LRS:
                for seed in SEEDS:
                    for panel in ['V_FIXED','V_MIXED']:
                        rr=sorted((r for r in records if r['lr']==lr and r['seed']==seed and r['panel']==panel),key=lambda x:x['epoch'])
                        assert [r['epoch'] for r in rr]==list(range(21))
                        vals=np.array([r['metrics']['mse'] for r in rr]);minima[(lr,seed,panel)]=rr[int(np.argmin(vals))]
            ranked=sorted((math.fsum(minima[(lr,s,'V_MIXED')]['metrics']['mse'] for s in SEEDS)/3,lr) for lr in LRS);lr=ranked[0][1]
            for s in SEEDS:
                expected={'P_MAIN':minima[(lr,s,'V_MIXED')],'P_VFIXED':minima[(lr,s,'V_FIXED')],'P_LR_FIXED':minima[(.001,s,'V_MIXED')],'P_LAST20':next(r for r in records if r['lr']==lr and r['seed']==s and r['panel']=='V_MIXED' and r['epoch']==20)}
                for policy,r in expected.items():
                    actual=next(x for x in selected if x['dataset']==d and x['arm']==a and x['seed']==s and x['policy']==policy)
                    for k in ['lr','epoch','checkpoint_path','checkpoint_sha256']:assert actual[k]==r[k]
                    checks+=1
    return checks

def verify_report():
    c=contract();s=state_json();assert s['status'] in ['RESOURCES_COMPLETE','COMPLETE']
    ss=read(OUT/'selection_seal.json');curves=read(OUT/'training_curves.json');ev=read(OUT/'evaluation.json');fits=read(OUT/'fits.json')
    assert len(fits)==36 and all(f['status']=='COMPLETE' and f['epochs']==20 for f in fits)
    assert sum(f['updates'] for f in fits)==45720 and len(curves)==1512
    assert sha(OUT/'training_curves.json')==ss['curves_sha256'] and sha(OUT/'fits.json')==ss['fits_sha256']
    nsel=verify_selection(curves,ss['selections']);check_hashes(c['source_hashes']);check_hashes(c['model_files'])
    check_hashes(read(OUT/'historical_hashes.json'))
    pred_records=curves+ev+[r['replay'] for r in read(OUT/'replays.json')]+read(OUT/'prior_dependency.json')
    unique={r['prediction_path']:r for r in pred_records};max_error=0.;verified=0
    for path,r in unique.items():
        assert sha(ROOT/path)==r['prediction_sha256']
        with np.load(ROOT/path) as z:actual=scalar_metrics(z['prediction'],z['target'],z['std'])
        for key,value in actual.items():
            expected=r['metrics'][key];assert math.isclose(value,expected,rel_tol=1e-10,abs_tol=1e-10),(path,key,value,expected)
            max_error=max(max_error,abs(value-expected));verified+=1
    checkpoints={r['checkpoint_path']:r['checkpoint_sha256'] for r in curves}
    check_hashes(checkpoints)
    # Verify all steps and schedules independently, including reuse cost provenance.
    steps_total=0;new_steps=0;contaminated=0;gradmax=0.;resource_main=[]
    for f in fits:
        d=f['dataset'];dc=c['data'][d];n=len(dc['panels']['TRAIN'])//8
        steps=read(ROOT/f['historical_steps_path']) if f['mode']=='REUSE' else read(OUT/(f['fit']+'_steps.json'))
        assert len(steps)==20*n
        for i,r in enumerate(steps):
            assert r['update']==i+1 and r['epoch']==i//n+1 and r['origins']==8
            assert r['lr']==f['lr']*.5**((r['epoch']-1)//5)
            assert all(math.isfinite(r[k]) for k in ['loss','gradient_norm','seconds'])
            contaminated+=int(r['external_compute_contaminated']);gradmax=max(gradmax,r['gradient_norm'])
        steps_total+=len(steps);new_steps+=len(steps) if f['mode']=='NEW' else 0
        resource_main.append(dict(fit=f['fit'],dataset=d,arm=f['arm'],seed=f['seed'],lr=f['lr'],mode=f['mode'],updates=len(steps),active_seconds=sum(r['seconds'] for r in steps),median_step_seconds=float(np.median([r['seconds'] for r in steps])),step_peak_allocated=max(r['peak_allocated'] for r in steps),step_peak_reserved=max(r['peak_reserved'] for r in steps),current_run_wall_seconds=f['wall_seconds'],current_validation_seconds=f['validation_seconds'],current_load_peak=f['load_peak'],current_validation_peak=f['validation_peak_allocated'],current_checkpoint_seconds=f['checkpoint_seconds'],current_resume_save_seconds=f['resume_save_seconds']))
    csvwrite(OUT/'main_resources.csv',resource_main)
    assert steps_total==45720 and new_steps==s['training_updates'];assert s['smoke_updates']==12 and s['resource_updates']<=120
    schedules=read(OUT/'schedules.json')
    for d,dc in c['data'].items():
        for seed in SEEDS:
            rng=np.random.default_rng(seed);assert schedules[f'{d}_{seed}']==[rng.permutation(dc['panels']['TRAIN']).tolist() for _ in range(20)]
    # Shared block weights for every arm, policy, seed and channel in a given panel.
    effects=[];uncertainty=[];phase_errors=[];seed_scores=[];macro=[]
    for di,d in enumerate(c['data']):
        for panel in ['E_FIXED','E_MIXED']:
            rows=[r for r in ev if r['dataset']==d and r['panel']==panel]
            n=len(c['data'][d]['panels'][panel]);blocks=np.array_split(np.arange(n),8);rng=np.random.default_rng(65016+di)
            weights=[]
            for _ in range(2000):
                indices=[]
                while len(indices)<n:indices.extend(blocks[int(rng.integers(8))].tolist())
                weights.append(np.bincount(indices[:n],minlength=n))
            weights=np.asarray(weights,np.float64)
            np.savez_compressed(CACHE/f'bootstrap_{d}_{panel}.npz',weights=weights,block_sizes=[len(b) for b in blocks])
            distribution={};score={}
            for r in rows:
                with np.load(ROOT/r['prediction_path']) as z:p=z['prediction'].astype(np.float64);y=z['target'].astype(np.float64);oo=z['origins']
                valid=np.isfinite(y);err=np.where(valid,p-y,0.);se=(err**2).sum(2);counts=valid.sum(2)
                numerator=weights@se;denominator=weights@counts
                boot=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
                key=(r['arm'],r['policy'],r['seed']);distribution[key]=np.nanmean(boot,axis=1);score[key]=r['metrics']['mse']
                seed_scores.append(dict(dataset=d,panel=panel,arm=r['arm'],policy=r['policy'],seed=r['seed'],epoch=r['epoch'],lr=r.get('lr'),mse=r['metrics']['mse'],mae=r['metrics']['mae'],raw_mae=r['metrics']['raw_mae']))
                for phase in sorted(set((oo%24).tolist())):
                    ii=oo%24==phase;v=metrics(p[ii],y[ii],np.ones(32));phase_errors.append(dict(dataset=d,panel=panel,arm=r['arm'],policy=r['policy'],seed=r['seed'],phase=phase,origins=int(ii.sum()),mse=v['mse'],mae=v['mae']))
            for a,pol in sorted(set((r['arm'],r['policy']) for r in rows)):
                rr=[r for r in rows if r['arm']==a and r['policy']==pol]
                macro.append(dict(dataset=d,panel=panel,arm=a,policy=pol,mse=float(np.mean([r['metrics']['mse'] for r in rr])),mae=float(np.mean([r['metrics']['mae'] for r in rr])),raw_mae=float(np.mean([r['metrics']['raw_mae'] for r in rr]))))
            contrasts=[(a,'P_MAIN',a,p,k) for a in ARMS for p,k in [('P_LR_FIXED','LR_SELECTION'),('P_VFIXED','V_PANEL_SELECTION')]]
            contrasts += [(a,pol,b,pol,'COMPONENT' if (a,b)==('PRIOR','SIDE') else 'ACCURACY_TRADEOFF') for pol in ['P_MAIN','P_LAST20'] for a,b in [('PRIOR','SIDE'),('PRIOR','LH'),('SIDE','LH')]]
            for a,ap,b,bp,kind in contrasts:
                av=np.array([score[(a,ap,z)] for z in SEEDS]);bv=np.array([score[(b,bp,z)] for z in SEEDS]);gain=100*(bv.mean()-av.mean())/bv.mean();gseed=100*(bv-av)/bv
                ab=np.mean([distribution[(a,ap,z)] for z in SEEDS],axis=0);bb=np.mean([distribution[(b,bp,z)] for z in SEEDS],axis=0);bg=100*(bb-ab)/bb;lo,hi=np.quantile(bg,[.025,.975])
                r=dict(dataset=d,panel=panel,kind=kind,arm=a,policy=ap,baseline=b,baseline_policy=bp,mse=float(av.mean()),baseline_mse=float(bv.mean()),gain_pct=float(gain),ci_low=float(lo),ci_high=float(hi),all_seed_positive=bool((gseed>0).all()),**{f'gain_{z}':float(g) for z,g in zip(SEEDS,gseed)})
                effects.append(r);uncertainty.append(dict(**r,replicates=2000,seed=65016+di,blocks=8,origin_count=n,denominator='resampled finite target counts per channel and seed, then equal-channel mean; three-seed score mean before gain',interpretation='descriptive reused-development block CI; overlapping targets; phase-time correlation; selection bias remains'))
    csvwrite(OUT/'effects.csv',effects);csvwrite(OUT/'uncertainty.csv',uncertainty);csvwrite(OUT/'seed_scores.csv',seed_scores);csvwrite(OUT/'macro_scores.csv',macro);csvwrite(OUT/'phase_errors.csv',phase_errors)
    # Full per-channel raw scores are retained without source aggregation.
    csvwrite(OUT/'channel_scores.csv',[dict(dataset=r['dataset'],panel=r['panel'],arm=r['arm'],policy=r['policy'],seed=r['seed'],channel=ch,mse=r['metrics']['channel_mse'][ch],mae=r['metrics']['channel_mae'][ch],raw_mae=r['metrics']['channel_raw_mae'][ch],valid_count=r['metrics']['valid_counts'][ch]) for r in ev for ch in range(32)])
    all_gpu=[]
    for p in OUT.glob('gpu_*.jsonl'):
        all_gpu.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())
    unauthorized=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in x['apps']) for x in all_gpu)
    verification=dict(status='PASS',trajectories=36,new_fits=sum(f['mode']=='NEW' for f in fits),reused_fits=sum(f['mode']=='REUSE' for f in fits),new_updates=new_steps,reused_updates=steps_total-new_steps,smoke_updates=s['smoke_updates'],resource_updates=s['resource_updates'],unique_predictions=len(unique),independent_scalar_checks=verified,max_abs_scalar_error=max_error,checkpoints=len(checkpoints),selection_checks=nsel,historical_files_preserved=len(read(OUT/'historical_hashes.json')),unauthorized_compute_samples=unauthorized,contaminated_training_steps=contaminated,max_gradient_norm_before_clip=gradmax,minimum_free_mib=min(x['free_mib'] for x in all_gpu),selected_replays=len(read(OUT/'replays.json')),selected_replays_bitwise=sum(r['bitwise_equal'] for r in read(OUT/'replays.json')),comparison_scope='DISCOVERY_REUSED_PERIODS')
    save(OUT/'independent_verification.json',verification)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,d in enumerate(c['data']):
        for j,lr in enumerate(LRS):
            ax=axes[i,j]
            for arm in ARMS:
                matrix=np.array([[next(r['metrics']['mse'] for r in curves if r['dataset']==d and r['arm']==arm and r['lr']==lr and r['seed']==seed and r['epoch']==e and r['panel']=='V_MIXED') for e in range(21)] for seed in SEEDS])
                ax.plot(range(21),matrix.mean(0),label=arm);ax.fill_between(range(21),matrix.min(0),matrix.max(0),alpha=.12)
            ax.set(title=f'{d} / LR={lr} / V_MIXED',xlabel='Epoch (0-20)',ylabel='Equal-channel standardized MSE');ax.legend()
    fig.tight_layout();fig.savefig(OUT/'validation_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,d in zip(axes,c['data']):
        rr=[r for r in effects if r['dataset']==d and r['panel']=='E_MIXED' and r['policy']=='P_MAIN' and r['kind'] in ['COMPONENT','ACCURACY_TRADEOFF']]
        for i,r in enumerate(rr):ax.bar(np.arange(3)+i*.23,[r[f'gain_{z}'] for z in SEEDS],width=.23,label=r['arm']+' vs '+r['baseline'])
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(np.arange(3)+.23,[str(z) for z in SEEDS]);ax.set(title=d,ylabel='Gain % (negative = worse)');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'seed_gains.png',dpi=160);plt.close(fig)
    rr=read(OUT/'resources.json');fig,axes=plt.subplots(2,2,figsize=(11,8))
    for i,d in enumerate(c['data']):
        for r in rr:
            if r['dataset']!=d or r['status']!='MEASURED':continue
            arm=r['option'].split('-')[0];score=next(x['mse'] for x in seed_scores if x['dataset']==d and x['panel']=='E_MIXED' and x['arm']==arm and x['policy']=='P_MAIN' and x['seed']==41000)
            for j,x in enumerate([r['peak_allocated']/2**20,r['median_seconds']*1000]):
                axes[i,j].scatter(x,score);axes[i,j].annotate(r['option'],(x,score),fontsize=8)
        axes[i,0].set(title=d+' selected seed41000',xlabel='Measured training peak MiB',ylabel='E_MIXED MSE');axes[i,1].set(title=d+' selected seed41000',xlabel='Measured optimizer step ms',ylabel='E_MIXED MSE')
    fig.tight_layout();fig.savefig(OUT/'accuracy_resources.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(16,6))
    for i,(d,dc) in enumerate(c['data'].items()):
        for j,panel in enumerate(['TRAIN','V_FIXED','V_MIXED','E_FIXED','E_MIXED']):
            axes[i,j].bar(range(24),np.bincount(np.array(dc['panels'][panel])%24,minlength=24));axes[i,j].set(title=f'{d} {panel}',xlabel='Origin row modulo24')
    fig.tight_layout();fig.savefig(OUT/'phase_distributions.png',dpi=160);plt.close(fig)
    component=[r for r in effects if r['panel']=='E_MIXED' and r['kind']=='COMPONENT' and r['policy']=='P_MAIN']
    signal=[r for r in component if r['gain_pct']>=1 and r['all_seed_positive'] and r['ci_low']>0]
    lhbest=all(next(r['mse'] for r in macro if r['dataset']==d and r['panel']=='E_MIXED' and r['policy']=='P_MAIN' and r['arm']=='LH')<=min(r['mse'] for r in macro if r['dataset']==d and r['panel']=='E_MIXED' and r['policy']=='P_MAIN' and r['arm'] in ['SIDE','PRIOR']) for d in c['data'])
    decision='A' if signal else 'C' if lhbest else 'B'
    reason={'A':'PRIOR에 제한된 개발 신호가 있어 recipe를 고정해 보존한다. 별도 독립 자료 검증을 제안하되 자동 실행하지 않는다.','B':'PRIOR의 사전 구성요소 문턱을 확보하지 못했다. SIDE를 유효한 저메모리 대조군으로 보존하고 prior 확장을 중단한다. 정확도 기준 LH도 보존한다.','C':'LH가 두 원천의 주 비교에서 가장 정확하다. 교정된 LH를 작동 기준으로 보존한다. SIDE는 측정된 저메모리 대조군으로만 보관하며 PRIOR 확장과 새 attention 후보 자동 생성을 중단한다.'}[decision]
    (OUT/'FINAL_DECISION.md').write_text(f'# 최종 결정 {decision}\n\n{reason}\n\n실행은 EXECUTION_COMPLETE이며 성능 또는 논문 PASS와 다르다. '+('PRIOR_COMPONENT_SIGNAL: '+', '.join(r['dataset'] for r in signal) if signal else 'PRIOR_COMPONENT_SIGNAL 미충족; 평균 양성은 WEAK_OR_UNCERTAIN_SIGNAL, 정확도·자원 손익은 TRADEOFF_ONLY로 구분한다.')+'\n\n기존 구현·원점수·부정적인 seed와 원천을 모두 보존한다. 신규성은 확보되지 않았다. 사용한 E_FIXED/E_MIXED는 개발 기간 재사용이며 독립 test가 아니다. 새 연구·새 seed·추가 학습은 자동 실행하지 않는다.\n')
    lines=['# 교정된 표본 위 MOMENT PEFT 비교 — 실행 완료','',
        '교정된 TRAIN 표본에서 LH(LoRA+HEAD), SIDE, PRIOR에 같은 최적화 기회를 주고, 학습률 선택과 검증 패널 선택 효과를 분리했다. 새로운 PEFT 발명 실험이 아니다. 건물·예보·Query·Censor·BASIS 결과를 합치지 않았다.','',
        f"36/36개 20-epoch 경로를 완료했다. 신규 {verification['new_fits']} fits·{new_steps:,} updates, 기존 {verification['reused_fits']} fits·{steps_total-new_steps:,} updates를 감사 후 재사용했다. 재사용 점수는 복사하지 않고 두 V 패널을 다시 예측했다. smoke 6회·12 updates, 자원 검사 {s['resource_updates']} 폐기 updates. 본학습·평가 미실행 0. 재개 {s.get('resume_count',0)}회. INIT와 매 epoch를 보존했다.", '',
        'TRAIN 원점, 모델 revision, 채널32, L96/H96, 761,952 trainable, BF16 forward/FP32 loss·Adam, micro/effective8을 고정했다. LR {0.001,0.0003}, seeds {41000,41001,41002}, StepLR5/0.5, epochs20을 모든 방법에 적용했다. V patience는 사용하지 않았다. 마지막 epoch가 최소 V인 경로는 예산 제한으로 해석하며 연장하지 않았다.','',
        'V_MIXED에서 3seed 평균 최소 V로 공통 LR을 고르고 seed별 epoch를 선택했다. P_VFIXED는 그 LR에서 검증 패널만, P_LR_FIXED는 V_MIXED에서 LR 허용만, P_LAST20은 반환 epoch만 바꾼다. 전체 학습 및 선택 봉인 뒤 E를 열었다.','',
        '## 원점수와 선택 효과','',
        '| 원천 | 평가 | 방법 | 정책 | 평균 MSE | 평균 MAE | raw MAE |','|---|---|---|---|---:|---:|---:|']
    for r in macro:lines.append(f"| {r['dataset']} | {r['panel']} | {r['arm']} | {r['policy']} | {r['mse']:.9f} | {r['mae']:.9f} | {r['raw_mae']:.6f} |")
    lines += ['', '세 seed 원점수와 선택 epoch/LR은 [seed_scores.csv](seed_scores.csv), 채널 원점수는 [channel_scores.csv](channel_scores.csv)에 모두 보존했다. INIT는 무작위 forecast head의 무학습 값이며 유효한 pretrained F0가 아니다. Seasonal-naive/last-value는 같은 정보의 기준이다. 원천 간 raw 오차를 하나로 평균하지 않았다.', '',
        '| 원천/평가 | 방법·대조 | 선택 효과 | 개선율 % | 블록95% 구간 |','|---|---|---|---:|---|']
    for r in effects:
        if r['kind'] not in ['LR_SELECTION','V_PANEL_SELECTION']:continue
        lines.append(f"| {r['dataset']}/{r['panel']} | {r['arm']} P_MAIN vs {r['baseline_policy']} | {r['kind']} | {r['gain_pct']:+.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] |")
    lines += ['', '양수는 개선, 음수는 악화이며 0도 그대로 기록했다. P_MAIN 이외 정책이 더 좋아도 주 정책으로 승격하지 않았다.', '', '## PRIOR의 SIDE 대비 추가 가치와 LH 대비 정확도', '', '| 원천/평가 | 정책 | 비교 | 개선율 % | seed41000 / 41001 / 41002 % | 블록95% 구간 |','|---|---|---|---:|---|---|']
    for r in effects:
        if r['kind'] in ['LR_SELECTION','V_PANEL_SELECTION']:continue
        lines.append(f"| {r['dataset']}/{r['panel']} | {r['policy']} | {r['arm']} vs {r['baseline']} | {r['gain_pct']:+.4f} | {r['gain_41000']:+.4f} / {r['gain_41001']:+.4f} / {r['gain_41002']:+.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] |")
    lines += ['', 'PRIOR를 저장한 상태에서 prior항만 끈 V_MIXED 진단은 [prior_dependency.csv](prior_dependency.csv)에 있다. 총12 논리적 예측 중 on6개는 기존 V 예측을 참조했고 off6개만 추가했다. 제거 시 손해는 공동 적응 의존성이지 SIDE보다 우월하다는 증거가 아니다.', '', '## 실제 자원 및 정확도 절충', '', '| 원천 | 선택 seed41000 옵션 | 상태 | 학습 peak MiB | step 중앙값 ms |','|---|---|---|---:|---:|']
    for r in rr:lines.append(f"| {r['dataset']} | {r['option']} | {r['status']} | {r.get('peak_allocated',0)/2**20:.2f} | {r.get('median_seconds',0)*1000:.3f} |")
    lines += ['', '위 표는 이번 선택 상태에서 고정 TRAIN batch로 실제 측정한 값이다. 지원·동등성 미확보 옵션은 NOT_MEASURED이고 0은 실제 비용 0을 뜻하지 않는다. 각 측정은 같은 arm의 저장 가중치·빈 Adam·RNG를 복원하고 warmup2+timed9 updates를 폐기했다. 과거 Adam 상태를 복원했다고 주장하지 않는다. 두 LR 탐색의 전체 비용과 선택 prefix 비용은 구분해야 하며, 모든 fits의 실제 비용은 [main_resources.csv](main_resources.csv), 선택 updates는 selection_seal.json에 있다. 재사용 fit의 과거 비용은 provenance를 표시했다.', '',
        '본학습 SIDE는 기존처럼 쓰지 않는 prior도 추출했다. SIDE-fast는 자원 전용 경로이며 같은 형태의 출력·gradient·update 검사를 통과한 경우에만 비용을 제시했다. LH native checkpoint도 동일한 자원 전용 기준이다. 메모리 이득을 PRIOR 고유 기여나 동등한 정확도로 해석하지 않는다. 학습 allocated/reserved, 평가/로드 peak, NVML 전체 사용량은 별도이며 비동시 peak를 더하지 않았다.', '',
        f"GPU 최소 여유 {verification['minimum_free_mib']} MiB, 비승인 compute 표본 {unauthorized}, 오염 학습 step {contaminated}. RustDesk만 승인 예외다. 같은 epoch 기회이며 같은 wall-time 실험이 아니다.", '',
        '## 검산·한계·결정', '',
        f"독립 FP64 scalar {verified:,}개(최대 절대 차이 {max_error:.3g}), {len(checkpoints)}개 checkpoint hash, 선택 규칙 {nsel}개, 새 모델 복원 {verification['selected_replays']}개(비트 일치 {verification['selected_replays_bitwise']}), 기존 파일 {verification['historical_files_preserved']}개 보존을 확인했다. [검증 기록](independent_verification.json).", '',
        '두 V/E 패널은 일부 다른 target을 포함한다. MIXED 생성은 값과 예측을 보지 않았지만 순수 위상 인과효과는 아니다. Traffic V는 18개 위상만 포함한다. E의 예측 target 중첩, phase와 시간의 상관, 이미 사용한 기간의 선택 편향이 남는다. 8개 연속 origin 블록을 복원추출해 N개까지 이어 붙여 자르고, 채널별 유효 target 분모를 다시 계산했다. 2,000개 paired resample의 동일 인덱스를 모든 방법·정책·seed에 사용했다. 3seed는 독립 source3개가 아니며 CI는 개발자료의 서술적 구간이다.', '',
        '작은 Q/K/V adapter는 Tiny-Attention, 백본 역전파를 줄이는 side 원리는 LST, 기존 attention score에 보정을 더하는 가까운 사례는 LiSA에 있다. 현 PRIOR의 확률 평균 위치·동결 side 연결 차이는 필요성과 고유 효용을 추가 입증해야 한다. 공식 알고리즘 전체 직접 재현은 이번 범위에서 실행하지 않았다. [보존된 선행 감사](../../research/attention_prior_novelty_audit_20260916/REPORT.md). 수치 신호와 신규성은 별개다.', '',
        f'최종 결정 **{decision}**: {reason} 실행 완료를 논문 PASS라고 부르지 않는다. [FINAL_DECISION.md](FINAL_DECISION.md). 추가 연구 자동 실행 없음.', '',
        '![검증 학습곡선](validation_curves.png)\n\n![seed별 이득](seed_gains.png)\n\n![정확도·자원](accuracy_resources.png)\n\n![원점 위상](phase_distributions.png)', '',
        '원시 자료·모델 가중치·예측 배열은 무시된 로컬 cache에 보관한다. GitHub의 코드·점수·manifest만으로 전체 수치를 재생할 수 있다는 뜻은 아니다.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');status(status='COMPLETE',decision=decision,execution='EXECUTION_COMPLETE')
    print('COMPLETE',decision,verification,flush=True)
