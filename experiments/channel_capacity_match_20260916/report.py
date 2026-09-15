"""Independent audit for the fixed capacity-matching control."""
import json,math,time
from pathlib import Path
import numpy as np
from model import ROOT,ARMS
from common import read,save,sha,csvwrite
from rank2_data import independent
OUT=ROOT/'results/channel_capacity_match_20260916'
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
    assert status['fit_attempts']==status['fits_completed']==len(fits)==4
    assert status['training_updates']==sum(f['updates'] for f in fits)<=5080 and status['smoke_updates']==4
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
    allrows=ev+ctrl;comparisons=[];scores=[];channels=[]
    external_fits={'BALANCED_LH':read(BAL/'fits.json'),'HEAD_ONLY':read(LP/'fits.json'),'POINTWISE':[f for f in read(ROOT/'results/channel_phase_transport_20260916/fits.json') if f['arm']=='POINTWISE']}
    for r in ev:
        f=next(f for f in fits if (f['dataset'],f['seed'])==(r['dataset'],r['seed']))
        steps=read(OUT/(f['fit']+'_steps.json'));a=cache[r['prediction_path']]
        if r['role']=='selected':assert (r['epoch'],r['updates'])==(f['best']['epoch'],f['best']['updates'])
        assert (ROOT/r['prediction_path']).stat().st_mtime>=selection['at']
        scores.append(dict(dataset=r['dataset'],seed=r['seed'],role=r['role'],epoch=r['epoch'],updates=r['updates'],trainable=761952,mse=a['mse'],mae=a['mae'],raw_mae=a['raw_mae'],peak_mib=f['peak_allocated']/2**20,adaptation_step_seconds=sum(x['seconds'] for x in steps if x['update']<=r['updates']),research_step_seconds=f['active_seconds']))
        for baseline in ['BALANCED_LH','HEAD_ONLY','POINTWISE']:
            b=next(b for b in ctrl if (b['dataset'],b['seed'],b['role'],b['arm'])==(r['dataset'],r['seed'],r['role'],baseline));ba=cache[b['prediction_path']]
            for key in ['target','origins','std']:assert np.array_equal(a['arrays'][key],ba['arrays'][key],equal_nan=True)
            if r['role']=='matched_old_epoch':assert r['epoch']==b['epoch']==seal['old_selected_epochs'][f"{r['dataset']}_{r['seed']}"] and r['updates']==b['updates']
            bf=next(f for f in external_fits[baseline] if (f['dataset'],f['seed'])==(r['dataset'],r['seed']))
            comparisons.append(dict(dataset=r['dataset'],seed=r['seed'],role=r['role'],baseline=baseline,epoch=r['epoch'],baseline_epoch=b['epoch'],updates=r['updates'],baseline_updates=b['updates'],mse=a['mse'],baseline_mse=ba['mse'],gain_percent=100*(ba['mse']-a['mse'])/ba['mse'],peak_mib=f['peak_allocated']/2**20,baseline_peak_mib=bf['peak_allocated']/2**20,trainable=761952,baseline_trainable={'BALANCED_LH':761952,'HEAD_ONLY':589920,'POINTWISE':606304}[baseline]))
        err=a['arrays']['prediction'].astype(np.float64)-a['arrays']['target'];assert np.isfinite(err).all()
        for i,cid in enumerate(seal['data'][r['dataset']]['channel_ids']):channels.append(dict(dataset=r['dataset'],seed=r['seed'],role=r['role'],channel=cid,mse=float(np.mean(err[:,i]**2))))
    csvwrite(OUT/'scores.csv',scores);csvwrite(OUT/'comparisons.csv',comparisons);csvwrite(OUT/'channel_scores.csv',channels)
    macros=[];uncertainty=[]
    for d in seal['data']:
        for role in ['selected','matched_old_epoch']:
            for baseline in ['BALANCED_LH','HEAD_ONLY','POINTWISE']:
                rr=[r for r in comparisons if (r['dataset'],r['role'],r['baseline'])==(d,role,baseline)];assert len(rr)==2
                a=float(np.mean([r['mse'] for r in rr]));b=float(np.mean([r['baseline_mse'] for r in rr]))
                macros.append(dict(dataset=d,role=role,baseline=baseline,capacity_mse=a,baseline_mse=b,gain_percent=100*(b-a)/b,both_seeds_improved=all(r['gain_percent']>0 for r in rr)))
                def errors_by_origin(records):
                    return np.mean([np.mean((cache[r['prediction_path']]['arrays']['prediction'].astype(np.float64)-cache[r['prediction_path']]['arrays']['target'])**2,axis=(1,2)) for r in records],axis=0)
                cc=errors_by_origin([r for r in ev if r['dataset']==d and r['role']==role]);bb=errors_by_origin([r for r in ctrl if r['dataset']==d and r['role']==role and r['arm']==baseline]);n=len(cc);blocks=np.array_split(np.arange(n),min(8,n));rng=np.random.default_rng(9018);gg=[]
                for _ in range(2000):
                    ids=np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks)+1)])[:n];base=float(np.mean(bb[ids]));gg.append(100*(base-float(np.mean(cc[ids])))/base)
                lo,hi=np.percentile(gg,[2.5,97.5]);uncertainty.append(dict(dataset=d,role=role,baseline=baseline,gain_percent=100*(b-a)/b,descriptive_ci_low=float(lo),descriptive_ci_high=float(hi),blocks=8,draws=2000,seed=9018))
    csvwrite(OUT/'macro_comparisons.csv',macros);csvwrite(OUT/'descriptive_uncertainty.csv',uncertainty)
    wall=read(OUT/'finite_diagnostic_wall.json');events=[json.loads(x) for x in (OUT/'gpu_finite_diagnostic.jsonl').read_text().splitlines()];external=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in e['apps']) for e in events);assert external==0
    verification=dict(at=time.time(),unique_prediction_caches=len(cache),scalar_mse_mae_raw_mae_records=len(errors),max_mse_abs_error=max(errors),selected_replays_exact=4,source_and_data_hashes_valid=True,schedules_identical=True,all_E_target_origin_std_identical=True,fixed_epoch_updates_equal=True,selection_seal_and_evaluation_timing_verified=True,frozen_and_buffers_unchanged=True,trainable_count_equal_to_LH=761952,historical_files_preserved=len(history),unapproved_compute_samples=external,report_source_sha256=sha(Path(__file__)))
    save(OUT/'verification.json',verification)
    save(OUT/'decision.json',dict(execution='COMPLETE',exposure='DISCOVERY_REUSED_E',novelty='KNOWN_CAPACITY_CONTROL',new_method_topic='NOT_CONFIRMED',independent_screen_pass='NOT_EVALUATED',macro_comparisons=macros,additional_fits_launched=0))
    lines=['# LoRA와 같은 파라미터 수의 마지막 adapter 대조','', '**동일 용량 대조4회와 평가를 완료했다. 알려진 adapter의 폭을 한 번 고정한 진단이며 새로운 방법론 PASS가 아니다.**','', f"실제4/4fits, {status['training_updates']}/5080 본학습updates, smoke4updates. 기존 LH/head-only/POINTWISE16 각4fits는 재학습하지 않았다. 미완료fit0, 실행오류/재시도0. 추가후속학습0.",'','## 무엇을 확인했는가','', '공통 head589,920개를 제외한 적응 용량을172,032개로 맞췄다. 마지막 residual h+B GELU(Ah)의 폭은172032/(2×512)=168이다. 점수를 보면서 폭을 선택하지 않았다. 기존 POINTWISE16과는 폭만 달라지고, 모델·head초기함수·sampling·원점·정규화·학습률·epoch 상한을 유지했다. encoder와 초기0-update LoRA는 동결했다. [사전 프로토콜](PROTOCOL.md).','', '## 원점수와 효과','', '| 원천 | recipe | 폭168 MSE | LH MSE | 폭16 MSE | head-only MSE |','|---|---|---:|---:|---:|---:|']
    for d in seal['data']:
        for role in ['selected','matched_old_epoch']:
            rr=[r for r in macros if r['dataset']==d and r['role']==role];lookup={r['baseline']:r for r in rr}
            lines.append(f"| {d} | {role} | {rr[0]['capacity_mse']:.6f} | {lookup['BALANCED_LH']['baseline_mse']:.6f} | {lookup['POINTWISE']['baseline_mse']:.6f} | {lookup['HEAD_ONLY']['baseline_mse']:.6f} |")
    lines+=['','각방법의 V 최저 checkpoint와, 사전에 정한 같은epoch/updates 비교를 함께 공개한다. 모든 E는 이미 노출된 개발 자료다. [seed별 MSE·MAE·rawMAE](scores.csv), [대조별 효과](comparisons.csv), [전체 채널](channel_scores.csv).','', '| 원천 | V 선택 대조 | 폭168 개선율 | 두seed 개선 | 설명용95% 구간 |','|---|---|---:|---|---|']
    for r in macros:
        if r['role']!='selected':continue
        ci=next(x for x in uncertainty if (x['dataset'],x['role'],x['baseline'])==(r['dataset'],r['role'],r['baseline']))
        lines.append(f"| {r['dataset']} | {r['baseline']} | {r['gain_percent']:+.3f}% | {r['both_seeds_improved']} | [{ci['descriptive_ci_low']:.3f}%, {ci['descriptive_ci_high']:.3f}%] |")
    lines+=['','구간은 두seed를 동일원점에서 평균하고 시간순8블록으로2000회 paired resample한 설명용 결과다(seed9018). 독립 검증이나 새로운 PASS gate가 아니다. 추가 hyperparameter 선택이나 노출 편향을 보정하지 않는다.','', '## 자원·검산','', '| fit | epochs | updates | 학습peak MiB | 학습step 합계 초 |','|---|---:|---:|---:|---:|']
    for f in fits:lines.append(f"| {f['fit']} | {f['epochs']} | {f['updates']} | {f['peak_allocated']/2**20:.3f} | {f['active_seconds']:.2f} |")
    lines+=['',f"Controller {wall['seconds']:.1f}초, 최소GPU여유 {wall['minimum_free_mib']}MiB, 비승인compute {external}표본. 새모델 trainables761,952개가 LH와 정확히 같다. 따라서 LH 대비 학습파라미터 절감은0이다. 총 모델은 고정LoRA를 보존한채 residual adapter를 더하므로 작아졌다고 주장하지 않는다. 역전파 경로와 활성 저장량이 달라 학습 peak는 별도로 측정했다. 선택checkpoint까지 step 시간과 총 연구step시간은 scores.csv에서 구분한다.",'',f"고유예측 {len(cache)}개/{len(errors)}개 MSE·MAE·rawMAE 기록을 독립 float64 scalar로 검산했다. 최대MSE차 {max(errors):.3g}. 선택checkpoint4개의 새모델 V 재생은exact. 전체frozen/buffer불변·intended parameters 실제변경·초기예측exact·미래poison불변, source/model/staged data/hash 및 기존{len(history)}개 결과 파일 보존을 확인했다. [검산](verification.json).",'', '## 해석 한계와 남은 작업','', '같은 파라미터 수가 같은 표현력이나 최적화 기하를 보장하지 않는다. 폭 변경은 초기 파라미터 수와 학습 동역학도 바꾸므로 차이를 순수한 삽입 위치 인과효과로 단정하지 않는다. 단일lr/폭/20epoch cap의 개발 대조다. cap도달은 수렴 증명이 아니다. 이전 입력 조건화 후보의 추가 가치 없음 판정은 그대로 보존한다.','', 'NOVELTY=KNOWN_CAPACITY_CONTROL, 독립SCREEN_PASS=NOT_EVALUATED, 새방법론주제=NOT_CONFIRMED. 후보 설계와 미노출 평가·다른backbone 일반화는 아직 미실행이다. [연구 검토](RESEARCH_REVIEW.md). 코드·원점수·manifest·검산은GitHub에, raw data/weights/예측배열은로컬cache에보존한다.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verification=verification,macros=macros),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
