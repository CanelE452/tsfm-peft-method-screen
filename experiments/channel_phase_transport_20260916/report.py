"""Independent scoring and predeclared decision audit for three transport arms."""
import json,math,time
from pathlib import Path
import numpy as np
from model import ROOT,ARMS
from common import read,save,sha,csvwrite
from rank2_data import independent
OUT=ROOT/'results/channel_phase_transport_20260916'
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
    assert status['fit_attempts']==status['fits_completed']==len(fits)==12
    assert status['training_updates']==sum(f['updates'] for f in fits)<=15240 and status['smoke_updates']==12
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
        assert f['trainable']==606304 and f['frozen_and_buffers_unchanged'] and f['replay_max_abs']==0
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
        baselines=['BALANCED_LH','HEAD_ONLY']+(['POINTWISE','UNIFORM'] if r['arm']=='CONDITIONED' else [])
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
            for arm in ARMS+['BALANCED_LH','HEAD_ONLY']:
                rr=[r for r in allrows if (r['dataset'],r['role'],r['arm'])==(d,role,arm)];assert len(rr)==2
                macro.append(dict(dataset=d,role=role,arm=arm,mse=float(np.mean([r['metrics']['mse'] for r in rr])),mae=float(np.mean([r['metrics']['mae'] for r in rr]))))
    csvwrite(OUT/'macro_scores.csv',macro)
    uncertainty=[]
    # Descriptive paired block resampling; not a new success gate or independent-test claim.
    for d in seal['data']:
        for role in ['selected','matched_old_epoch']:
            cand=[next(r for r in ev if (r['dataset'],r['seed'],r['role'],r['arm'])==(d,seed,role,'CONDITIONED')) for seed in [41000,41001]]
            def by_origin(records):
                return np.mean([np.mean((cache[r['prediction_path']]['arrays']['prediction'].astype(np.float64)-cache[r['prediction_path']]['arrays']['target'])**2,axis=(1,2)) for r in records],axis=0)
            cc=by_origin(cand);n=len(cc);blocks=np.array_split(np.arange(n),min(8,n))
            for baseline in ['POINTWISE','UNIFORM','BALANCED_LH','HEAD_ONLY']:
                bb=by_origin([next(r for r in allrows if (r['dataset'],r['seed'],r['role'],r['arm'])==(d,seed,role,baseline)) for seed in [41000,41001]])
                rng=np.random.default_rng(9018);gains=[]
                for _ in range(2000):
                    ids=np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks)+1)])[:n]
                    a=float(np.mean(cc[ids]));b=float(np.mean(bb[ids]));gains.append(100*(b-a)/b)
                lo,hi=np.percentile(gains,[2.5,97.5])
                uncertainty.append(dict(dataset=d,role=role,baseline=baseline,gain_percent=100*(float(bb.mean())-float(cc.mean()))/float(bb.mean()),descriptive_ci_low=float(lo),descriptive_ci_high=float(hi),origin_count=n,blocks=len(blocks),bootstrap_draws=2000,bootstrap_seed=9018))
    csvwrite(OUT/'descriptive_uncertainty.csv',uncertainty)
    inventory=read(OUT/'parameter_inventory.json')['rows']
    for r in inventory:
        assert sum(p['numel'] for p in r['parameters'])==r['total_parameters']
        assert sum(p['numel'] for p in r['parameters'] if p['trainable'])==r['trainable_parameters']
    assert next(r['trainable_parameters'] for r in inventory if r['arm']=='CONDITIONED')==606304
    probe=read(OUT/'train_operator_probe.json');assert sha(ROOT/'experiments/channel_phase_transport_20260916/probe.py')==probe['source_sha256']
    component=[];resource=[]
    bfits=read(BAL/'fits.json')
    for d in seal['data']:
        for base in ['POINTWISE','UNIFORM']:
            rr=[r for r in comparisons if r['dataset']==d and r['arm']=='CONDITIONED' and r['role']=='selected' and r['baseline']==base]
            a=float(np.mean([r['mse'] for r in rr]));b=float(np.mean([r['baseline_mse'] for r in rr]));g=100*(b-a)/b
            component.append(dict(dataset=d,baseline=base,gain_percent=g,both_seeds_positive=all(r['gain_percent']>0 for r in rr),meets_predeclared_component_condition=g>=1 and all(r['gain_percent']>0 for r in rr)))
        rr=[r for r in comparisons if r['dataset']==d and r['arm']=='CONDITIONED' and r['role']=='selected' and r['baseline']=='BALANCED_LH']
        ratio=float(np.mean([r['mse'] for r in rr])/np.mean([r['baseline_mse'] for r in rr]));peakratios=[]
        for f in fits:
            if f['dataset']==d and f['arm']=='CONDITIONED':
                bf=next(b for b in bfits if b['dataset']==d and b['seed']==f['seed']);peakratios.append(f['peak_allocated']/bf['peak_allocated'])
        resource.append(dict(dataset=d,mse_ratio_to_LH=ratio,trainable_reduction_percent=100*(761952-606304)/761952,worst_peak_ratio_to_LH=max(peakratios),meets_predeclared_resource_condition=ratio<=1.01 and (761952-606304)/761952>=.2 and max(peakratios)<=.5))
    wall=read(OUT/'finite_diagnostic_wall.json');events=[json.loads(x) for x in (OUT/'gpu_finite_diagnostic.jsonl').read_text().splitlines()];external=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in e['apps']) for e in events);assert external==0
    verification=dict(at=time.time(),unique_prediction_caches=len(cache),scalar_mse_mae_raw_mae_records=len(errors),max_mse_abs_error=max(errors),selected_replays_exact=12,source_and_data_hashes_valid=True,schedules_identical=True,all_E_target_origin_std_identical=True,fixed_epoch_updates_equal=True,frozen_and_buffers_unchanged=True,historical_files_preserved=len(history),unapproved_compute_samples=external,report_source_sha256=sha(Path(__file__)))
    save(OUT/'verification.json',verification)
    decision=dict(execution='COMPLETE',exposure='DISCOVERY_REUSED_E',component_signal=all(x['meets_predeclared_component_condition'] for x in component),resource_signal=all(x['meets_predeclared_resource_condition'] for x in resource),component_comparisons=component,resource_comparisons=resource,novelty='UNRESOLVED_PRIOR_COMPONENTS_KNOWN',independent_screen_pass='NOT_EVALUATED',new_method_topic='NOT_CONFIRMED',additional_fits_launched=0)
    save(OUT/'decision.json',decision)
    lines=['# 같은 시간대 표현 집계의 입력 조건화 — 실행 결과','', '**한 후보(CONDITIONED)와 두 동일 용량 대조(POINTWISE/UNIFORM)의 유한 비교를 완료했다. 노출된 개발 E이며 독립 SCREEN_PASS가 아니다.**','', f"실제12/12fits, {status['training_updates']}/15240 본학습updates, smoke12updates. 기존balanced LH4fits와head-only4fits는재학습하지않았다. 미완료fit0,추가후속학습0. 실행오류/재시도0.",'','## 무엇을 왜 바꿨는가','', '동결한MOMENT encoder의마지막표현h에16차원bottleneck residual을더했다. POINTWISE는각patch를따로변환하고, UNIFORM은24시간간격의4patch를같게평균하며, CONDITIONED는같은위상patch의실제입력거리에따라가중평균한다. 후보의추가가치는일반비선형adapter와uniform집계를각각넘는지로판단한다. 세arm 모두606304개를학습하고head·초기예측·sample순열·학습률·평가를같게맞췄다. E의추가정답이나온라인적응은쓰지않았다. [완전한수식·사전정책](PROTOCOL.md).','', '## 원점수','', '| 원천 | recipe | POINTWISE MSE | UNIFORM MSE | CONDITIONED MSE | LoRA+head MSE | head-only MSE |','|---|---|---:|---:|---:|---:|---:|']
    for d in seal['data']:
        for role in ['selected','matched_old_epoch']:
            vals=[next(r['mse'] for r in macro if (r['dataset'],r['role'],r['arm'])==(d,role,a)) for a in ARMS+['BALANCED_LH','HEAD_ONLY']]
            lines.append('| '+d+' | '+role+' | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
    lines+=['','각방법의V최저checkpoint비교와사전고정epoch의같은updates비교를모두공개했다. source안에서두seed MSE를평균하며서로다른source의raw단위를섞지않는다. [seed별MSE/MAE/rawMAE·자원](scores.csv), [개별효과](comparisons.csv), [모든채널](channel_scores.csv).','', '| 원천 | 후보의대조 | 평균MSE개선 | 두seed개선 | 사전1%조건 |','|---|---|---:|---|---|']
    for r in component:lines.append(f"| {r['dataset']} | {r['baseline']} | {r['gain_percent']:+.3f}% | {r['both_seeds_positive']} | {r['meets_predeclared_component_condition']} |")
    lines+=['','참고 [paired block bootstrap 구간](descriptive_uncertainty.csv)은 두seed를같은원점에서평균하고, E원점을시간순8블록으로묶어2000회재표집했다(seed9018). 후보·대조에같은블록을적용했다. 노출된E의설명용민감도이며독립검증·선택편향보정이나새PASS gate가아니다. 이진조건이충족되지않아도효과의부재를확정하지않는다.','', '## 자원과예측을함께본판단','', '| 원천 | 후보/LH MSE비 | 학습파라미터절감 | 최대peak비 | 사전자원·정확도동시조건 |','|---|---:|---:|---:|---|']
    for r in resource:lines.append(f"| {r['dataset']} | {r['mse_ratio_to_LH']:.6f} | {r['trainable_reduction_percent']:.3f}% | {r['worst_peak_ratio_to_LH']:.4f} | {r['meets_predeclared_resource_condition']} |")
    lines+=['','1%허용폭/파라미터20%절감/peak50%절감은본실험전에고정했다. 이를통계적동등성이나독립성공으로해석하지않는다. 작은메모리만으로성능손실을성공으로바꾸지않는다. peak는각fit실제학습최대allocated이며상주GPU사용량과다르다. 고정된LoRA모듈forward가남아있어최소LP실행의최적화벤치마크도아니다.','', '총 모델 파라미터는 LH 36,099,360개, 후보 36,115,744개로 후보가 16,384개 더 많다. 감소한 것은 학습 파라미터와 역전파 저장량이다. [실제 shape·학습 대상 목록](parameter_inventory.json). 실행 중 짧은 CPU 입력·파라미터 검사도 병행했으므로 wall 시간은 전용 성능 벤치마크로 일반화하지 않는다.','', '| fit | epochs | updates | peak MiB | 학습step합계초 |','|---|---:|---:|---:|---:|']
    for f in fits:lines.append(f"| {f['fit']} | {f['epochs']} | {f['updates']} | {f['peak_allocated']/2**20:.3f} | {f['active_seconds']:.2f} |")
    lines+=['',f"Controller {wall['seconds']:.1f}초,최소GPU여유{wall['minimum_free_mib']}MiB,비승인compute{external}표본. RustDesk만허용했다. 총연구비용과선택checkpoint까지의적응비용은구분하며후자는scores.csv에있다. epoch상한도달은수렴의증명이아니다.",'','## 검산·한계·미실행','',f"고유예측{len(cache)}개/{len(errors)}개 MSE·MAE·rawMAE기록을독립float64 scalar로검산했다. 최대MSE차{max(errors):.3g}. 선택checkpoint12개를새모델에서재생해차이0. FP64수식/gradient,초기identity,반례와uniform동치,학습대상실제변경·동결가중치/버퍼보존,미래값poison불변을확인했다. 기존{len(history)}개결과·연구파일보존. [검산](verification.json).",'',f"구성요소개발신호={decision['component_signal']},자원·정확도동시신호={decision['resource_signal']}. 독립SCREEN_PASS는NOT_EVALUATED,신규성은UNRESOLVED_PRIOR_COMPONENTS_KNOWN,새방법론주제는NOT_CONFIRMED다. 좋은seed/채널만선택하거나실패한조건을재튜닝하지않았다.",'', '잔차분해는정답을쓴사후설명용이며배포가능한oracle가아니다. 한E시작위상·두공개source·두seed의개발비교라다른기간/데이터에대한일반화는미검증이다. Autoformer주기집계·일반kernel attention·bottleneck adapter의원리자체는알려져있고,TimePEFT/CoRA전체재현이나우위를본실험이확인하지않는다. [선행검토](RESEARCH_REVIEW.md). 독립미노출평가·추가backbone·신규성확정은미실행이다. 원자료/weights/예측배열은로컬cache에보존하고GitHub에는코드·원점수·hash·검산을보관한다.']
    lines+=['', '봉인된 train의 첫64개 원점에서 입력 조건화 가중치는 모든 검사 행에서 uniform과 달랐다. [입력만 사용한 연산 검사](train_operator_probe.json). 이 활동성은 예측 이득의 증거가 아니다.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verification=verification,decision=decision,macro=macro),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
