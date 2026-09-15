"""Independent scalar scoring and bookkeeping for the head-only ablation."""
from pathlib import Path
import json,math,time
import numpy as np
from model import ROOT
from common import read,save,sha,csvwrite
from rank2_data import independent
OUT=ROOT/'results/channel_head_only_20260916'
BAL=ROOT/'results/channel_phase_balance_20260916'

def hashes(mapping):
    for p,h in mapping.items():assert sha(ROOT/p)==h,('HASH_CHANGED',p)

def main():
    status=read(OUT/'status.json');assert status['status']=='COMPLETE'
    seal=read(OUT/'seal.json');hashes(seal['source_hashes']);history=read(OUT/'historical_hashes.json');hashes(history)
    fits=read(OUT/'fits.json');traj=read(OUT/'trajectory.json');ev=read(OUT/'evaluation.json');ctrl=read(OUT/'reused_controls.json');bfits=read(BAL/'fits.json')
    assert status['fit_attempts']==status['fits_completed']==len(fits)==4
    assert status['training_updates']==sum(f['updates'] for f in fits)<=5080 and status['smoke_updates']==4
    assert read(OUT/'schedules.json')==read(BAL/'schedules.json')
    assert seal['data']==read(BAL/'seal.json')['data']
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
    for f in fits:
        rows=[r for r in traj if r['fit']==f['fit']]
        assert min(rows,key=lambda r:(r['metrics']['mse'],r['epoch']))==f['best']
        steps=read(OUT/(f['fit']+'_steps.json'))
        assert len(steps)==f['updates'] and [r['update'] for r in steps]==list(range(1,f['updates']+1))
        assert f['updates']==f['epochs']*{'electricity':64,'traffic':63}[f['dataset']]
        assert f['trainable']==589920 and f['frozen_and_buffers_unchanged'] and f['replay_max_abs']==0
        a=cache[f['best']['prediction_path']]['arrays'];b=cache[f['replay']['prediction_path']]['arrays']
        assert np.array_equal(a['prediction'],b['prediction'])
        schedule=read(OUT/'schedules.json')[f"{f['dataset']}_{f['seed']}"]
        assert all(len(oo)==sum(r['origins'] for r in steps if r['epoch']==i+1) for i,oo in enumerate(schedule[:f['epochs']]))
    comparisons=[];component_rows=[];channel_rows=[]
    for r in ev:
        b=next(x for x in ctrl if (x['dataset'],x['seed'],x['role'])==(r['dataset'],r['seed'],r['role']))
        f=next(x for x in fits if (x['dataset'],x['seed'])==(r['dataset'],r['seed']))
        bf=next(x for x in bfits if (x['dataset'],x['seed'])==(r['dataset'],r['seed']))
        a=cache[r['prediction_path']];ba=cache[b['prediction_path']]
        for k in ['target','origins','std']:assert np.array_equal(a['arrays'][k],ba['arrays'][k],equal_nan=True)
        if r['role']=='matched_old_epoch':assert r['updates']==b['updates'] and r['epoch']==b['epoch']==seal['old_selected_epochs'][f"{r['dataset']}_{r['seed']}"]
        steps=read(OUT/(f['fit']+'_steps.json'));bstep=read(BAL/(bf['fit']+'_steps.json'))
        row=dict(dataset=r['dataset'],seed=r['seed'],role=r['role'],head_epoch=r['epoch'],lora_head_epoch=b['epoch'],head_updates=r['updates'],lora_head_updates=b['updates'],head_mse=a['mse'],lora_head_mse=ba['mse'],lora_added_gain_percent=100*(a['mse']-ba['mse'])/a['mse'],head_mae=a['mae'],lora_head_mae=ba['mae'],head_raw_mae=a['raw_mae'],lora_head_raw_mae=ba['raw_mae'],head_trainable=589920,lora_head_trainable=761952,head_peak_mib=f['peak_allocated']/2**20,lora_head_peak_mib=bf['peak_allocated']/2**20,head_adaptation_step_seconds=sum(x['seconds'] for x in steps if x['update']<=r['updates']),lora_head_adaptation_step_seconds=sum(x['seconds'] for x in bstep if x['update']<=b['updates']))
        comparisons.append(row)
        for arm,arrays in [('HEAD_ONLY',a['arrays']),('LORA_HEAD',ba['arrays'])]:
            error=arrays['prediction'].astype(np.float64)-arrays['target']
            assert np.isfinite(error).all()
            level=error.mean(-1,keepdims=True)
            daily=np.tile(error.reshape(-1,32,4,24).mean(2)-level,(1,1,4))
            remainder=error-level-daily
            parts=[float(np.mean(v*v)) for v in [level,daily,remainder]]
            assert math.isclose(sum(parts),float(np.mean(error*error)),rel_tol=1e-12,abs_tol=1e-12)
            component_rows.append(dict(dataset=r['dataset'],seed=r['seed'],role=r['role'],arm=arm,level_mse=parts[0],daily_mse=parts[1],remainder_mse=parts[2],mse=sum(parts)))
        eh=a['arrays']['prediction'].astype(np.float64)-a['arrays']['target']
        el=ba['arrays']['prediction'].astype(np.float64)-ba['arrays']['target']
        for c,cid in enumerate(seal['data'][r['dataset']]['channel_ids']):
            h=float(np.mean(eh[:,c]**2));l=float(np.mean(el[:,c]**2))
            channel_rows.append(dict(dataset=r['dataset'],seed=r['seed'],role=r['role'],channel=cid,head_mse=h,lora_head_mse=l,lora_added_gain_percent=100*(h-l)/h if h else None))
    csvwrite(OUT/'residual_components.csv',component_rows)
    csvwrite(OUT/'channel_comparisons.csv',channel_rows)
    csvwrite(OUT/'comparisons.csv',comparisons)
    source=[]
    for d in seal['data']:
        for role in ['selected','matched_old_epoch']:
            rr=[r for r in comparisons if r['dataset']==d and r['role']==role];assert len(rr)==2
            h=float(np.mean([r['head_mse'] for r in rr]));l=float(np.mean([r['lora_head_mse'] for r in rr]))
            source.append(dict(dataset=d,role=role,head_mean_mse=h,lora_head_mean_mse=l,lora_added_gain_percent=100*(h-l)/h,both_seeds_lora_better=all(r['lora_added_gain_percent']>0 for r in rr)))
    wall=read(OUT/'finite_diagnostic_wall.json');events=[json.loads(x) for x in (OUT/'gpu_finite_diagnostic.jsonl').read_text().splitlines()]
    external=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in e['apps']) for e in events)
    assert external==0
    verify=dict(at=time.time(),unique_prediction_caches=len(cache),scalar_mse_mae_raw_mae_records=len(errors),max_mse_abs_error=max(errors),selected_replays_exact=4,source_and_data_hashes_valid=True,schedules_identical=True,all_E_target_origin_std_identical=True,fixed_epoch_updates_equal=True,frozen_and_buffers_unchanged=True,historical_files_preserved=len(history),unapproved_compute_samples=external,report_source_sha256=sha(Path(__file__)))
    save(OUT/'verification.json',verify)
    save(OUT/'decision.json',dict(execution='COMPLETE',evidence='EXPOSED_DEVELOPMENT_COMPONENT_ABLATION',novelty='DIRECT_EQUIVALENCE_LINEAR_PROBING',new_method_topic='NOT_ESTABLISHED',source_comparisons=source,additional_fits_launched=0))
    lines=['# 균형 표본에서 head-only와 LoRA 추가 가치','', '**head-only 4회 학습과 고정 평가를 완료했다. 이것은 구성요소 대조이며 새 PEFT 방법론 PASS가 아니다.**','',f"신규4/4fits·{status['training_updates']}/5080updates, smoke4updates. 기존 balanced LoRA+head4fits는 재학습하지 않았다. 미완료fit0. 추가 후보·seed·재튜닝·재시도0.",'','## 문제와 비교','', '직전 sampling 변경으로 회복된 성능에 encoder LoRA가 추가로 기여하는지 확인했다. 모델 forward와 초기 head는 같고, 이번에는 encoder를 동결해 head589920개만 학습했다. LoRA+head는761952개다. 두 방법 모두 균형 train 창, 동일 순열·학습률·V/E·정규화를 쓴다. INIT을 포함한 V 최저오차 checkpoint를 각각 선택한다. 별도로 사전 고정 epoch(전력seed41000=1/41001=3, 교통둘=1)에서 동일 updates 비교를 한다.','', '## 원점수','', '양수인 LoRA 추가 이득은 LoRA+head가 head-only보다 낮은 MSE임을 뜻한다. 백분율은100×(head-only−LoRA+head)/head-only다.','', '| 원천 | seed | recipe | head epoch | LoRA epoch | head-only MSE | LoRA+head MSE | LoRA 추가 이득 |','|---|---:|---|---:|---:|---:|---:|---:|']
    for r in comparisons:lines.append(f"| {r['dataset']} | {r['seed']} | {r['role']} | {r['head_epoch']} | {r['lora_head_epoch']} | {r['head_mse']:.6f} | {r['lora_head_mse']:.6f} | {r['lora_added_gain_percent']:+.3f}% |")
    lines+=['','| 원천 | recipe | head 평균 MSE | LoRA+head 평균 MSE | LoRA 추가 이득 | 두seed 모두 LoRA 우세 |','|---|---|---:|---:|---:|---|']
    for r in source:lines.append(f"| {r['dataset']} | {r['role']} | {r['head_mean_mse']:.6f} | {r['lora_head_mean_mse']:.6f} | {r['lora_added_gain_percent']:+.3f}% | {r['both_seeds_lora_better']} |")
    lines+=['','MAE·원단위MAE·선택epoch까지 optimizer step 시간과 같은 epoch 비교의 updates는 [comparisons.csv](comparisons.csv)에 모두 있다. 두 원천의 원단위 지표를 섞지 않는다. E는 이미 사용한 개발 구간으로 독립 시험이 아니다. E를 보고 새 checkpoint를 고르거나 기준을 바꾸지 않았다.','', '사후 진단 [채널별 오차](channel_comparisons.csv)와 [잔차 분해](residual_components.csv)는 모든 채널을 보존한다. 잔차의 평균·24시간 반복 성분·나머지는 정답을 사용하는 설명용 분해이며, 배포 보정이나 새 채널 선택 규칙이 아니다. RMSE를 가산 분해한 것이 아니라 MSE의 직교 분해다.','', '## 자원과 실제 실행','', '| fit | epochs | updates | peak MiB | 학습 step 합계 초 |','|---|---:|---:|---:|---:|']
    for f in fits:lines.append(f"| {f['fit']} | {f['epochs']} | {f['updates']} | {f['peak_allocated']/2**20:.3f} | {f['active_seconds']:.2f} |")
    lines+=['',f"전체 controller {wall['seconds']:.1f}초, 최소 GPU 여유 {wall['minimum_free_mib']}MiB. 비승인compute {external}개; RustDesk만 허용했다. head-only는 학습 파라미터172032개({100*172032/761952:.3f}%)를 줄였다. encoder backward도 없어지지만 고정된0-update LoRA forward 모듈을 남겼으므로 최소 LP 구현의 자원 최적화는 아니다. 기존 실행과의 시간 차이를 모든 환경의 속도 개선으로 일반화하지 않는다. 총 연구비용과 선택 checkpoint까지 적용비용을 분리한다.",'', '## 검산과 연구 판단','',f"고유 예측 {len(cache)}개·{len(errors)}개 MSE/MAE/원단위MAE 기록을 독립 scalar 계산으로 검산했다. 최대 MSE차 {max(errors):.3g}. 선택 checkpoint4개의 새 모델 V 재생 차이0, 초기 head/실모델 초기 예측 일치, 학습 대상 변경·전체동결가중치/버퍼 보존·미래 poison 불변을 확인했다. 이전 결과·연구 {len(history)}개 파일을 보존했다. [검산](verification.json).",'', 'EXECUTION=COMPLETE, EVIDENCE=EXPOSED_DEVELOPMENT_COMPONENT_ABLATION, NOVELTY=DIRECT_EQUIVALENCE_LINEAR_PROBING. 이 비교는 알려진 LP와 LoRA의 가치 분해다. 어느 쪽이 좋아도 신규 알고리즘 성공으로 바꾸지 않는다. 두 원천·두seed·한 E 시작 위상에 한정되고 건물/Query 실패 원인으로 확대할 수 없다. 새 PEFT 주제 확보에는 강한 단순 대조를 넘는 구체적 변경, 가까운 선행과의 차이 및 노출되지 않은 평가가 여전히 필요하다.','', '[프로토콜](PROTOCOL.md), [사전 연구 검토](RESEARCH_REVIEW.md), [판단](decision.json). 원자료·weights·예측배열은 로컬cache에 보존하고 GitHub에는 코드·원점수·manifest·검산을 공개한다.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verification=verify,source_comparisons=source),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
