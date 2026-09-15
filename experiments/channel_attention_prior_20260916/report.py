"""Independent scoring and predeclared decision audit for two side-attention arms."""
import json,math,time
from pathlib import Path
import numpy as np
from model import ROOT,ARMS
from common import read,save,sha,csvwrite
from rank2_data import independent
OUT=ROOT/'results/channel_attention_prior_20260916'
BAL=ROOT/'results/channel_phase_balance_20260916'
LP=ROOT/'results/channel_head_only_20260916'

def hashes(m):
    for p,h in m.items():assert sha(ROOT/p)==h,('HASH_CHANGED',p)

def main():
    status=read(OUT/'status.json');assert status['status']=='COMPLETE'
    seal=read(OUT/'seal.json');hashes(seal['source_hashes']);history=read(OUT/'historical_hashes.json');hashes(history)
    fits=read(OUT/'fits.json');traj=read(OUT/'trajectory.json');ev=read(OUT/'evaluation.json');ctrl=read(OUT/'reused_controls.json')
    for dc in seal['data'].values():hashes(dc['staged'])
    hashes(seal['model_files'])
    selection=read(OUT/'selection_seal.json');assert selection['selections']==[f['best'] for f in fits]
    assert selection['source_seal_sha256']==sha(OUT/'seal.json')
    assert status['fit_attempts']==status['fits_completed']==len(fits)==8
    assert status['training_updates']==sum(f['updates'] for f in fits)<=10160 and status['smoke_updates']==8
    assert read(OUT/'schedules.json')==read(BAL/'schedules.json') and seal['data']==read(BAL/'seal.json')['data']
    cache={};errors=[]
    for r in traj+ev+ctrl+[f['replay'] for f in fits]:
        path=r['prediction_path'];assert sha(ROOT/path)==r['prediction_sha256']
        if path not in cache:
            with np.load(ROOT/path) as z:a={k:z[k].copy() for k in z.files}
            mse=independent(a['prediction'],a['target']);maes=[]
            for c in range(a['target'].shape[1]):
                pairs=[(float(x),float(y)) for x,y in zip(a['prediction'][:,c].flat,a['target'][:,c].flat) if math.isfinite(float(y))]
                maes.append(math.fsum(abs(x-y) for x,y in pairs)/len(pairs))
            cache[path]=dict(arrays=a,mse=mse,mae=math.fsum(maes)/len(maes),raw_mae=math.fsum(v*float(s) for v,s in zip(maes,a['std']))/len(maes))
        x=cache[path]
        for k in ['mse','mae']:assert math.isclose(x[k],r['metrics'][k],rel_tol=1e-12,abs_tol=1e-12)
        assert math.isclose(x['raw_mae'],float(np.mean(r['metrics']['channel_raw_mae'])),rel_tol=1e-12,abs_tol=1e-12)
        errors.append(abs(x['mse']-r['metrics']['mse']))
        if 'checkpoint_path' in r:assert sha(ROOT/r['checkpoint_path'])==r['checkpoint_sha256']
    schedules=read(OUT/'schedules.json')
    for f in fits:
        rows=[r for r in traj if r['fit']==f['fit']]
        assert min(rows,key=lambda r:(r['metrics']['mse'],r['epoch']))==f['best']
        steps=read(OUT/(f['fit']+'_steps.json'))
        assert len(steps)==f['updates'] and [r['update'] for r in steps]==list(range(1,f['updates']+1))
        assert f['updates']==f['epochs']*{'electricity':64,'traffic':63}[f['dataset']]
        assert f['trainable']==761952 and f['frozen_and_buffers_unchanged'] and f['replay_max_abs']==0
        assert np.array_equal(cache[f['best']['prediction_path']]['arrays']['prediction'],cache[f['replay']['prediction_path']]['arrays']['prediction'])
        assert all(len(oo)==sum(r['origins'] for r in steps if r['epoch']==i+1) for i,oo in enumerate(schedules[f"{f['dataset']}_{f['seed']}"][:f['epochs']]))
    allrows=ev+ctrl;resources=[]
    for r in ev:
        f=next(f for f in fits if (f['dataset'],f['seed'],f['arm'])==(r['dataset'],r['seed'],r['arm']))
        steps=read(OUT/(f['fit']+'_steps.json'))
        if r['role']=='selected':assert (r['epoch'],r['updates'])==(f['best']['epoch'],f['best']['updates'])
        assert (ROOT/r['prediction_path']).stat().st_mtime>=selection['at']
        resources.append(dict(dataset=r['dataset'],seed=r['seed'],arm=r['arm'],role=r['role'],epoch=r['epoch'],updates=r['updates'],trainable=f['trainable'],mse=r['metrics']['mse'],mae=r['metrics']['mae'],raw_mae=cache[r['prediction_path']]['raw_mae'],peak_mib=f['peak_allocated']/2**20,adaptation_step_seconds=sum(x['seconds'] for x in steps if x['update']<=r['updates']),research_step_seconds=f['active_seconds']))
    csvwrite(OUT/'scores.csv',resources)
    comparisons=[];channels=[];parts=[]
    for r in ev:
        baselines=['BALANCED_LH','HEAD_ONLY','CAPACITY_MATCHED']+(['SIDE'] if r['arm']=='PRIOR' else [])
        for baseline in baselines:
            b=next(x for x in allrows if (x['dataset'],x['seed'],x['role'],x['arm'])==(r['dataset'],r['seed'],r['role'],baseline))
            a=cache[r['prediction_path']];ba=cache[b['prediction_path']]
            for k in ['target','origins','std']:assert np.array_equal(a['arrays'][k],ba['arrays'][k],equal_nan=True)
            if r['role']=='matched_old_epoch':assert r['epoch']==b['epoch']==seal['old_selected_epochs'][f"{r['dataset']}_{r['seed']}"] and r['updates']==b['updates']
            comparisons.append(dict(dataset=r['dataset'],seed=r['seed'],arm=r['arm'],role=r['role'],baseline=baseline,epoch=r['epoch'],baseline_epoch=b['epoch'],updates=r['updates'],baseline_updates=b['updates'],mse=a['mse'],baseline_mse=ba['mse'],gain_percent=100*(ba['mse']-a['mse'])/ba['mse']))
        arr=cache[r['prediction_path']]['arrays'];err=arr['prediction'].astype(np.float64)-arr['target'];assert np.isfinite(err).all()
        level=err.mean(-1,keepdims=True);daily=np.tile(err.reshape(-1,32,4,24).mean(2)-level,(1,1,4));rem=err-level-daily
        pp=[float(np.mean(x*x)) for x in [level,daily,rem]];assert math.isclose(sum(pp),r['metrics']['mse'],rel_tol=1e-12,abs_tol=1e-12)
        parts.append(dict(dataset=r['dataset'],seed=r['seed'],arm=r['arm'],role=r['role'],level_mse=pp[0],daily_mse=pp[1],remainder_mse=pp[2]))
        for i,cid in enumerate(seal['data'][r['dataset']]['channel_ids']):channels.append(dict(dataset=r['dataset'],seed=r['seed'],arm=r['arm'],role=r['role'],channel=cid,mse=float(np.mean(err[:,i]**2))))
    csvwrite(OUT/'comparisons.csv',comparisons);csvwrite(OUT/'channel_scores.csv',channels);csvwrite(OUT/'residual_components.csv',parts)
    macro=[]
    for d in seal['data']:
        for role in ['selected','matched_old_epoch']:
            for arm in ARMS+['BALANCED_LH','HEAD_ONLY','CAPACITY_MATCHED']:
                rr=[r for r in allrows if (r['dataset'],r['role'],r['arm'])==(d,role,arm)];assert len(rr)==2
                macro.append(dict(dataset=d,role=role,arm=arm,mse=float(np.mean([r['metrics']['mse'] for r in rr])),mae=float(np.mean([r['metrics']['mae'] for r in rr]))))
    csvwrite(OUT/'macro_scores.csv',macro)
    uncertainty=[]
    # Descriptive paired block resampling; not a new success gate or independent-test claim.
    for d in seal['data']:
        for role in ['selected','matched_old_epoch']:
            cand=[next(r for r in ev if (r['dataset'],r['seed'],r['role'],r['arm'])==(d,seed,role,'PRIOR')) for seed in [41000,41001]]
            def by_origin(records):
                return np.mean([np.mean((cache[r['prediction_path']]['arrays']['prediction'].astype(np.float64)-cache[r['prediction_path']]['arrays']['target'])**2,axis=(1,2)) for r in records],axis=0)
            cc=by_origin(cand);n=len(cc);blocks=np.array_split(np.arange(n),min(8,n))
            for baseline in ['SIDE','BALANCED_LH','HEAD_ONLY','CAPACITY_MATCHED']:
                bb=by_origin([next(r for r in allrows if (r['dataset'],r['seed'],r['role'],r['arm'])==(d,seed,role,baseline)) for seed in [41000,41001]])
                rng=np.random.default_rng(9019);gains=[]
                for _ in range(2000):
                    ids=np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks)+1)])[:n]
                    a=float(np.mean(cc[ids]));b=float(np.mean(bb[ids]));gains.append(100*(b-a)/b)
                lo,hi=np.percentile(gains,[2.5,97.5])
                uncertainty.append(dict(dataset=d,role=role,baseline=baseline,gain_percent=100*(float(bb.mean())-float(cc.mean()))/float(bb.mean()),descriptive_ci_low=float(lo),descriptive_ci_high=float(hi),origin_count=n,blocks=len(blocks),bootstrap_draws=2000,bootstrap_seed=9019))
    csvwrite(OUT/'descriptive_uncertainty.csv',uncertainty)
    # Independent replay of stopping and learning-rate policies.
    for f in fits:
        rr=[r for r in traj if r['fit']==f['fit']];best=rr[0]['metrics']['mse'];bad=0
        for r in rr[1:]:
            if r['metrics']['mse']<best-1e-4:best=r['metrics']['mse'];bad=0
            else:bad+=1
            assert bad<5 or r==rr[-1]
        assert f['early_stopped']==(bad>=5) and (f['epochs']==20 or bad>=5)
        for step in read(OUT/(f['fit']+'_steps.json')):
            assert step['lr']==.001*.5**((step['epoch']-1)//5)
            assert not step['external_compute_contaminated']
            assert all(math.isfinite(step[k]) for k in ['loss','gradient_norm','seconds'])
    smokes=read(OUT/'smoke.json');assert len(smokes)==4
    for sm in smokes:
        assert sm['initial_bf16_LH_exact'] and sm['frozen_and_buffers_unchanged'] and len(sm['steps'])==2
        for prefix in ['head.']+[f'side.{i}.{part}.' for i in range(8) for part in ['q','k','v','up','norm']]:
            assert any(v>0 for n,v in sm['changes'].items() if n.startswith(prefix))
    component=[]
    for d in seal['data']:
        for base in ['SIDE','BALANCED_LH']:
            rr=[r for r in comparisons if r['dataset']==d and r['arm']=='PRIOR' and r['role']=='selected' and r['baseline']==base]
            a=float(np.mean([r['mse'] for r in rr]));b=float(np.mean([r['baseline_mse'] for r in rr]));g=100*(b-a)/b
            component.append(dict(dataset=d,baseline=base,gain_percent=g,both_seeds_positive=all(r['gain_percent']>0 for r in rr),meets_predeclared_condition=g>=1 and all(r['gain_percent']>0 for r in rr)))
    memory=[];bfits=read(BAL/'fits.json')
    for f in fits:
        bf=next(b for b in bfits if (b['dataset'],b['seed'])==(f['dataset'],f['seed']))
        steps=read(OUT/(f['fit']+'_steps.json'))
        memory.append(dict(dataset=f['dataset'],seed=f['seed'],arm=f['arm'],epochs=f['epochs'],updates=f['updates'],trainable=f['trainable'],peak_allocated_mib=f['peak_allocated']/2**20,peak_reserved_mib=max(x['peak_reserved'] for x in steps)/2**20,LH_peak_allocated_mib=bf['peak_allocated']/2**20,peak_allocated_saving_percent=100*(1-f['peak_allocated']/bf['peak_allocated']),active_step_seconds=f['active_seconds'],early_stopped=f['early_stopped']))
    csvwrite(OUT/'resources.csv',memory)
    wall=read(OUT/'finite_diagnostic_wall.json');events=[json.loads(x) for x in (OUT/'gpu_finite_diagnostic.jsonl').read_text().splitlines()]
    external=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in e['apps']) for e in events);assert external==0
    verify=dict(at=time.time(),unique_prediction_caches=len(cache),scalar_mse_mae_raw_mae_records=len(errors),max_mse_abs_error=max(errors),selected_checkpoint_replays_exact=8,source_data_model_hashes_valid=True,schedules_identical=True,all_E_target_origin_std_identical=True,fixed_epoch_updates_equal=True,selection_sealed_before_E=True,stopping_and_LR_replayed=True,smoke_all_groups_changed=True,frozen_and_buffers_unchanged=True,historical_files_preserved=len(history),unapproved_compute_samples=external,report_source_sha256=sha(Path(__file__)))
    save(OUT/'verification.json',verify)
    decision=dict(execution='COMPLETE',component_signal=all(r['meets_predeclared_condition'] for r in component),component_comparisons=component,exposure='DISCOVERY_REUSED_E',novelty='UNRESOLVED_PRIOR_COMPONENTS_KNOWN',independent_screen_pass='NOT_EVALUATED',new_method_topic='NOT_CONFIRMED',additional_fits_launched=0)
    save(OUT/'decision.json',decision)
    lines=['# 동결 attention prior의 추가 가치 — 8-fit 결과','', '**후보PRIOR와 같은 용량의SIDE 직접 비교를 완료했다. 이번 E는 반복 노출된 개발 자료이며 독립 SCREEN_PASS가 아니다.**','', f"실제8/8fits, {status['training_updates']}/10160 본학습updates, smoke8updates. 기존 LH/head-only/동일용량pointwise 각4fits는 재학습하지 않았다. 미완료fit0, 추가후속학습0, 실행오류/재시도0.",'', '## 무엇을 바꿨는가','', '두 군 모두8층 중간표현을 받는 저랭크side attention과 같은head를 학습한다. PRIOR만 각층 frozen attention의head평균분포를logit기준으로 사용한다. SIDE는그기준없이연결을학습한다. 둘 다 Q/K/V와출력변환을학습하고backbone으로역전파하지않는다. 각층폭10은LoRA와같은761,952 trainables를맞추는등식으로정했다. 학습률·표본·초기head·선택정책은고정했다. [사전프로토콜](PROTOCOL.md).','', '## 원점수','', '| 원천 | recipe | SIDE MSE | PRIOR MSE | LH MSE | head-only MSE | pointwise168 MSE |','|---|---|---:|---:|---:|---:|---:|']
    for d in seal['data']:
        for role in ['selected','matched_old_epoch']:
            vals=[next(r['mse'] for r in macro if (r['dataset'],r['role'],r['arm'])==(d,role,a)) for a in ARMS+['BALANCED_LH','HEAD_ONLY','CAPACITY_MATCHED']]
            lines.append('| '+d+' | '+role+' | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
    lines+=['','각방법의 V선택과사전에정한동일epoch/updates를함께공개했다. E점수로epoch를바꾸지않았다. [seed별MSE·MAE·rawMAE](scores.csv), [모든채널](channel_scores.csv), [비교별효과](comparisons.csv).','', '| 원천 | PRIOR의 대조 | MSE 개선율 | 두seed 양수 | 사전1% 개발조건 |','|---|---|---:|---|---|']
    for r in component:lines.append(f"| {r['dataset']} | {r['baseline']} | {r['gain_percent']:+.3f}% | {r['both_seeds_positive']} | {r['meets_predeclared_condition']} |")
    lines+=['','[설명용95%구간](descriptive_uncertainty.csv)은 같은원점에서두seed를평균하고시간순8블록paired bootstrap2000회(seed9019)로계산했다. 독립확증이나탐색편향보정이아니다. [레벨/일주기/나머지잔차분해](residual_components.csv)는정답을사용한설명용이며배포가능한보정기가아니다.','', '## 자원과 실제 실행량','', '| fit | epochs | updates | peak allocated MiB | peak reserved MiB | LH 대비allocated절감 |','|---|---:|---:|---:|---:|---:|']
    for r in memory:lines.append(f"| {r['dataset']}/{r['seed']}/{r['arm']} | {r['epochs']} | {r['updates']} | {r['peak_allocated_mib']:.3f} | {r['peak_reserved_mib']:.3f} | {r['peak_allocated_saving_percent']:.2f}% |")
    lines+=['',f"Controller {wall['seconds']:.1f}초, 최소GPU여유{wall['minimum_free_mib']}MiB, 비승인compute0표본. 두군trainables는LH와같아학습파라미터절감0이다. 원래0-LoRA가상주한채side모듈이추가되므로총모델이작아졌다고주장하지않는다. 학습peak와전체GPU사용량은다르다. prior추출을두군모두수행해비교조건을맞췄으며SIDE를최적화한최소비용벤치마크는아니다. [상세자원](resources.csv). 선택checkpoint까지step시간과총연구step시간은scores.csv에구분했다.",'','## 검산과 남은 한계','',f"고유예측{len(cache)}개/{len(errors)}개 MSE·MAE·rawMAE를독립float64 scalar로검산했다. 최대MSE차{max(errors):.3g}. 선택checkpoint8개의새모델재생exact,동일target/origin/std·순열·source/model/data hash·선택봉인·LR/조기종료정책을확인했다. 기존{len(history)}개결과파일보존. [검산](verification.json).",'',f"사전개발조건충족={decision['component_signal']}. 신규성UNRESOLVED_PRIOR_COMPONENTS_KNOWN, 독립SCREEN_PASS=NOT_EVALUATED, 새방법론주제=NOT_CONFIRMED. 양성이라도현재E에대한사후개발신호이며별도미노출평가와선행대비차별성확인이남아있다.",'', 'LST/LAST의side경로·저차원attention, prior가중정규화의수학은알려진원리다. 현재차이는고정backbone의층별attention을side기준으로활용하는선택이며최초성은확정하지않았다. 원논문LAST전체재현도아니다. [선행·반례검토](RESEARCH_REVIEW.md). 후보가실패한조건의설정재탐색·추가backbone·미노출평가·후속후보학습은미실행이다. 원자료/weights/예측배열은로컬,코드·원점수·manifest·보고서는GitHub에보존한다.']
    probe=read(OUT/'prior_probe.json')
    assert probe['source_sha256']==sha(Path(__file__).with_name('prior_probe.py'))
    assert probe['new_fits']==probe['optimizer_updates']==0
    lines += ['', '## 학습 입력에서 확인한 attention 기준분포', '',
              '봉인 후 보조 관찰로 각 원천의 기존 학습 입력 8개만 CPU FP32로 읽었다. 학습과 V/E 채점은 추가하지 않았다. GPU BF16 분포와 수치적으로 같다는 검사는 아니다.',
              '', '| 원천 | 층별 평균 정규화 엔트로피 범위 | 균일분포와의 평균 L1 거리 범위 |',
              '|---|---:|---:|']
    for d in seal['data']:
        rr=[r for r in probe['records'] if r['dataset']==d]
        assert len(rr)==8 and all(r['origins']==seal['data'][d]['origins']['train'][:8] for r in rr)
        entropy=[r['mean_normalized_entropy'] for r in rr]; distance=[r['mean_L1_to_uniform'] for r in rr]
        lines.append(f'| {d} | {min(entropy):.3f}–{max(entropy):.3f} | {min(distance):.3f}–{max(distance):.3f} |')
    lines += ['', '이 입력들에서 기준분포는 균일하지 않았다. 따라서 추가 prior가 단순히 균일분포여서 SIDE와 같은 연산이 됐다는 설명은 맞지 않는다. 그렇다고 이 분포가 예측에 유용하거나 모든 입력에서 같은 특성을 가진다는 뜻은 아니다. [관찰 원기록](prior_probe.json).',
              '', f'본학습 update 상한 중 미사용 {10160-status["training_updates"]}회는 고정한 조기 종료 정책에 따른 것이다. 여유분으로 대체 학습을 실행하지 않았다.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verification=verify,decision=decision),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
