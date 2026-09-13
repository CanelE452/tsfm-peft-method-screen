"""Independent CPU replay and preregistered equal-time continuation decision."""
import csv
import hashlib
import json
import subprocess
import sys

import numpy as np
import torch

from tsfm_peft_screen.metrics import score, independent
from tsfm_peft_screen.reproducibility import ROOT, sha, digest, write_json

OUT=ROOT/'results/forecast_query_equal_time'
CACHE=ROOT/'.cache/forecast_query_equal_time'


def read(name):
    return json.loads((OUT/name).read_text())


def verify_seal(name):
    seal=read(name)
    saved=seal.pop('seal_sha256')
    assert saved==digest(seal)
    return seal


def array_difference(a,b):
    a,b=a.numpy().astype(np.float64).reshape(-1),b.numpy().astype(np.float64).reshape(-1)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    e=a-b
    return dict(max_absolute=float(np.abs(e).max()),
                relative_l2=float(np.sqrt(np.sum(e*e))/max(np.sqrt(np.sum(a*a)),1e-30)))


def main():
    status=read('status.json')
    assert status['status']=='COMPLETE',status
    contract=read('contract.json')
    cfg=contract['config']
    counts=status['counts']
    assert counts['completed_fits']==counts['fit_attempts']==32
    assert (counts['preflight_warmup_updates'],counts['preflight_bf16_updates'],counts['preflight_fp32_updates'])==(16,48,16)
    assert counts['evaluation_open_events']==2
    execution_sources=dict(contract['source_hashes'])
    execution_sources['scripts/run_forecast_query_equal_time.py']=contract['runner_sha256']
    execution_sources['scripts/run_forecast_query_checkpoint_diagnostic.py']=contract['shared_helper_sha256']
    for path,expected in execution_sources.items():
        payload=subprocess.check_output(['git','show',f"{contract['execution_commit']}:{path}"],cwd=ROOT)
        assert hashlib.sha256(payload).hexdigest()==expected
    for path,expected in contract['historical_result_hashes'].items():
        assert sha(ROOT/path)==expected
    for name,receipt in contract['data'].items():
        for kind in ('development','heldout'):
            assert sha(CACHE/(name+'_'+kind+'.npz'))==receipt[kind+'_sha256']
    storage=verify_seal('storage_seal.json')
    assert storage['measurements_sha256']==sha(OUT/'preflight_measurements.json')
    pre=read('preflight_measurements.json')
    eqs=read('preflight_parity.json')
    assert len(pre)==64 and len(eqs)==32 and len(storage['choices'])==8
    parity_error=0.
    for eq in eqs:
        pair=[]
        initial=[]
        for cp in (False,True):
            row=next(r for r in pre if all(r[k]==eq[k] for k in ('arm','dataset','precision','repeat')) and r['checkpoint']==cp)
            path=CACHE/(row['tag']+'.pt')
            assert sha(path)==row['cache_sha256']
            initial.append(row['initial_state_sha256'])
            pair.append(torch.load(path,weights_only=True,map_location='cpu'))
        assert initial[0]==initial[1]
        for key in ('z','raw','loss','gradient','update','adam'):
            delta=array_difference(pair[0][key],pair[1][key])
            for kind,value in delta.items():
                assert abs(value-eq['metrics'][key][kind])<=1e-10
                assert value<=cfg['numerical_equivalence_tolerances'][eq['precision']][kind]
                parity_error=max(parity_error,value)
        assert pair[0]['rng_after_sha256']==pair[1]['rng_after_sha256']
        assert abs(pair[0]['gradient_norm']-pair[1]['gradient_norm'])<=cfg['numerical_equivalence_tolerances'][eq['precision']]['max_absolute']
    # Independent storage selection, including the predeclared 2% time tie.
    for selected in storage['choices']:
        options=[]
        for cp in (False,True):
            rr=[r for r in pre if r['dataset']==selected['dataset'] and r['arm']==selected['arm'] and r['precision']=='bf16' and r['checkpoint']==cp]
            assert len(rr)==3
            peak=max(r['peak_allocated_bytes'] for r in rr)
            if peak<=cfg['storage_preflight']['max_peak_allocated_bytes']:
                options.append(dict(checkpoint=cp,median_seconds=float(np.median([r['seconds'] for r in rr])),peak_allocated_bytes=peak))
        fastest=min(r['median_seconds'] for r in options)
        winner=min([r for r in options if r['median_seconds']<=fastest*1.02],key=lambda r:(r['peak_allocated_bytes'],r['checkpoint']))
        assert all(selected[k]==v for k,v in winner.items())
    seal=verify_seal('selection_seal.json')
    assert seal['contract_sha256']==sha(OUT/'contract.json')
    assert seal['storage_seal_sha256']==sha(OUT/'storage_seal.json')
    assert seal['trajectories_sha256']==sha(OUT/'trajectories.json')
    fits=read('fits.json')
    trajectories=read('trajectories.json')
    ev=read('evaluation.json')
    reloads=read('checkpoint_replay.json')
    assert len(fits)==32 and len(trajectories)==128 and len(ev)==22 and len(reloads)==16
    assert sum(r['updates'] for r in fits)==counts['training_optimizer_updates']
    for r in fits:
        actual=sum(s['seconds'] for s in r['step_resources'])
        assert abs(actual-r['active_seconds'])<=1e-10 and 30<=actual<=30.5
        rr=[t for t in trajectories if t['fit']==r['fit']]
        assert sorted(t['nominal_seconds'] for t in rr)==[0,10,20,30]
        for t in rr:
            assert sha(CACHE/t['checkpoint_file'])==t['checkpoint_sha256']
            assert t['nominal_seconds']<=t['active_seconds']<=t['nominal_seconds']+.5
        winner=min(rr,key=lambda t:(t['metrics']['scaled_2pinball'],t['nominal_seconds'],t['lr']))
        assert winner['prediction_sha256']==r['best']['prediction_sha256']
        assert sha(CACHE/r['best']['checkpoint_file'])==r['best']['checkpoint_sha256']
    for s in seal['selections']:
        candidates=[r['best'] for r in fits if all(r[k]==s[k] for k in ('dataset','seed','arm'))]
        assert min(candidates,key=lambda r:(r['metrics']['scaled_2pinball'],r['nominal_seconds'],r['lr']))==s
    max_metric_error=0.
    for r in trajectories+ev+reloads:
        path=CACHE/r['prediction_file']
        assert sha(path)==r['prediction_sha256']
        with np.load(path,allow_pickle=False) as z:
            a=independent(z['prediction'],z['target'],z['scale'])
            metrics=score(z['prediction'],z['target'],z['scale'])
        assert abs(a-r['metrics']['scaled_2pinball'])<=1e-10
        assert all(abs(v-r['metrics'][k])<=1e-10 for k,v in metrics.items())
        max_metric_error=max(max_metric_error,abs(a-r['metrics']['scaled_2pinball']))
    for r in reloads:
        chosen=next(s for s in seal['selections'] if s['fit']==r['fit'])
        with np.load(CACHE/r['prediction_file']) as a,np.load(CACHE/chosen['prediction_file']) as b:
            assert np.array_equal(a['prediction'],b['prediction'])
    summaries=[]
    decisions=[]
    paired=[]
    for name in cfg['datasets']:
        lookup={(r['arm'],r['seed']):r for r in ev if r['dataset']==name}
        means={}
        for arm in ['F0']+cfg['arms']:
            losses=[lookup[('F0',None)]['metrics']['scaled_2pinball']]*2 if arm=='F0' else [lookup[(arm,s)]['metrics']['scaled_2pinball'] for s in cfg['seeds']]
            means[arm]=float(np.mean(losses))
            summaries.append(dict(dataset=name,arm=arm,seed30000_loss=losses[0],seed30001_loss=losses[1],mean_loss=means[arm]))
        strongest=min(cfg['research_continuation']['baseline_arms'],key=lambda a:means[a])
        per_seed=[]
        for seed in cfg['seeds']:
            q=lookup[('query',seed)]
            baseline=min([lookup[('F0',None)]['metrics']['scaled_2pinball']]+[lookup[(a,seed)]['metrics']['scaled_2pinball'] for a in ['standard','head','side']])
            per_seed.append(dict(seed=seed,query_loss=q['metrics']['scaled_2pinball'],
                                 best_baseline_loss=baseline,loss_ratio=q['metrics']['scaled_2pinball']/baseline,
                                 nonzero_selection=q['nominal_seconds']>0,
                                 improves_own_initial=q['metrics']['scaled_2pinball']<lookup[('query_initial',seed)]['metrics']['scaled_2pinball']))
        gain=1-means['query']/means[strongest]
        passed=means['query']<=.995*means[strongest] and all(r['loss_ratio']<=1.01 and r['nonzero_selection'] and r['improves_own_initial'] for r in per_seed)
        decisions.append(dict(dataset=name,strongest_baseline=strongest,query_gain_vs_best_seed_mean=gain,
                              query_mean=means['query'],baseline_mean=means[strongest],seeds=per_seed,
                              continuation_gate=passed))
        for arm in cfg['research_continuation']['baseline_arms']:
            diffs=[]
            for seed in cfg['seeds']:
                q=lookup[('query',seed)]
                b=lookup[('F0',None)] if arm=='F0' else lookup[(arm,seed)]
                with np.load(CACHE/q['prediction_file']) as qa,np.load(CACHE/b['prediction_file']) as ba:
                    assert np.array_equal(qa['target'],ba['target'])
                    diff=[independent(qa['prediction'][i:i+1],qa['target'][i:i+1],qa['scale'])-
                          independent(ba['prediction'][i:i+1],ba['target'][i:i+1],ba['scale']) for i in range(16)]
                diffs.append(diff)
            by_origin=np.mean(diffs,axis=0)
            blocks=by_origin.reshape(4,4).mean(1)
            rng=np.random.default_rng(51000)
            draws=blocks[rng.integers(0,4,size=(1000,4))].mean(1)
            paired.append(dict(dataset=name,baseline=arm,seed_origin_loss_differences=diffs,
                               seed_mean_difference=float(by_origin.mean()),
                               descriptive_95_percent_interval=np.quantile(draws,[.025,.975]).tolist(),
                               independent_blocks=4,scope='Descriptive only; overlapping contexts and tiny number of blocks.'))
    verdict='CONTINUE_CANDIDATE_VALIDATION' if all(d['continuation_gate'] for d in decisions) else 'STOP_CURRENT_QUERY'
    monitor=read('gpu_monitor.json')
    active=[r for r in monitor if r['phase']!='startup_wait']
    verification=dict(passed=True,execution_source_files=len(execution_sources),parity_pairs=32,
                      max_parity_error=parity_error,prediction_caches_replayed=166,
                      checkpoint_reloads=16,max_primary_replay_error=max_metric_error,
                      historical_files_unchanged=len(contract['historical_result_hashes']),
                      counts=counts,gpu_samples=len(monitor),
                      min_observed_active_free_mib=min(r['free_mib'] for r in active),
                      external_compute_seen=any(r['external_pids'] for r in monitor))
    summary=dict(verdict=verdict,original_forecast_query_verdict='FAIL',decisions=decisions,metrics=summaries,
                 counts=counts,active_training_seconds=sum(r['active_seconds'] for r in fits),
                 total_fit_wall_seconds=sum(r['total_fit_wall_seconds'] for r in fits),
                 complete_experiment_wall_seconds=status['elapsed_wall_seconds'])
    if '--verify-only' in sys.argv:
        assert read('summary.json')==summary
        assert read('verification.json')==verification
        assert read('paired_differences.json')==paired
        print('EQUAL TIME VERIFIED',verdict,verification)
        return
    write_json(OUT/'summary.json',summary)
    write_json(OUT/'verification.json',verification)
    write_json(OUT/'paired_differences.json',paired)
    with (OUT/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(summaries)
    lines=['# Forecast-query 동일 학습 시간 비교 결과','',f'판정: **{verdict}**. 기존 메모리·품질 gate의 FAIL은 유지한다.','',
           '## 범위와 실행','',f"실행 commit: {contract['execution_commit']}. 고정 계획 commit: {contract['plan_commit']}.",
           f"32/32 fits 완료, 실제 학습 updates={counts['training_optimizer_updates']}. Preflight80 updates는 별도다.",
           f"총 active training={summary['active_training_seconds']:.2f}s, fit wall 합계={summary['total_fit_wall_seconds']:.2f}s, "
           f"전체 실행 wall={summary['complete_experiment_wall_seconds']:.2f}s.",
           '각 fit은 명목30초+최대0.5초 경계 초과 이내이며, 선택은 V의0/10/20/30초와2LR만 사용했다.',
           'V/E/저장/대기 시간은 active clock 밖이고 wall 장부에 포함한다. end-to-end 동일시간 비교로 해석하지 않는다.','',
           '## Primary loss (낮을수록 좋음)','',
           '| 데이터 | 방식 | seed30000 | seed30001 | 평균 |','|---|---|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['dataset']} | {r['arm']} | {r['seed30000_loss']:.8f} | {r['seed30001_loss']:.8f} | {r['mean_loss']:.8f} |")
    lines+=['','## 데이터셋별 고정 판정','']
    for d in decisions:
        lines.append(f"- {d['dataset']}: 최선 기준선={d['strongest_baseline']}, Query 평균 개선율={100*d['query_gain_vs_best_seed_mean']:.4f}%, 지속 gate={d['continuation_gate']}.")
        for r in d['seeds']:
            lines.append(f"  - seed{r['seed']}: 최선 기준선 대비 loss ratio={r['loss_ratio']:.6f}, 선택 시간>0={r['nonzero_selection']}, 자체 초기 출력보다 개선={r['improves_own_initial']}.")
    lines+=['','## 저장 방식과 비용','',
            '| 데이터 | 방식 | checkpoint | preflight median step s | preflight peak MiB |',
            '|---|---|---|---:|---:|']
    for r in storage['choices']:
        lines.append(f"| {r['dataset']} | {r['arm']} | {r['checkpoint']} | {r['median_seconds']:.5f} | {r['peak_allocated_bytes']/2**20:.2f} |")
    lines+=['','학습 중 각 fit의 실제 업데이트 수·peak·validation/checkpoint/wall 비용은 fits.json에 전부 기록했다.',
            'Preflight3회 시간 차이는 공유 데스크톱의 작은 표본이며 확정적 시스템 우위가 아니다.','',
            '## 무결성','',
            f"32 preflight on/off 쌍 독립 재계산, 최대 수치 차이={parity_error:.3g}.",
            f"166개 prediction cache와16개 선택 checkpoint replay; primary 최대 오차={max_metric_error:.3g}.",
            f"기존 결과 {verification['historical_files_unchanged']}개 파일과 frozen backbone 파라미터 불변.",
            f"GPU 점검 {len(monitor)}회, 실행 단계 최소 관측 free={verification['min_observed_active_free_mib']:.0f}MiB, "
            f"외부 compute 관측(대기 포함)={verification['external_compute_seen']}.",
            '모든16 선택을 봉인한 뒤에만 E 평가를 열었다. raw 파일 mechanical staging과 E scoring은 구분한다.','',
            '## 해석의 한계','',
            'Train/V 일부는 과거 실험의 개발 데이터다. E는 기록상 이전 scoring과 겹치지 않는 이후 구간이며 같은 원천의16 origins뿐이다.',
            'ETTm2와 Electricity는 시간 해상도와 달력상 horizon이 다르다. seed2개는 optimization 반복이며 독립 데이터2개가 아니다.',
            'paired_differences.json의4 chronological block bootstrap 구간은 참고용이며 통계적 유의성이나 논문 PASS의 근거가 아니다.',
            '이번 gate는 같은 명목 최적화 시간의 품질 개선에 관한 개발 지속 조건이다. 메모리20% gate를 바꿔 실패를 성공으로 재분류하지 않는다.',
            '조건을 만족해도 신규성·독립 데이터 일반화·논문 기여는 별도 검증 대상이다. 미충족이면 이 Query의 추가 확장을 중단한다.','',
            '## 재현','',
            'scripts/with_cuda.sh .venv/bin/python scripts/finalize_forecast_query_equal_time.py --verify-only',
            '원시 예측과 파라미터/Adam cache는 ignored .cache/forecast_query_equal_time에 있다.',
            'GPU runner는 기존 결과 디렉터리를 덮어쓰지 않는다.','']
    (OUT/'RESULT.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
