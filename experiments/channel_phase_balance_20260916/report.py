"""Independent bookkeeping and scalar scoring audit for the single sampling control."""
from pathlib import Path
import json,math,time
import numpy as np
from model import ROOT
from common import read,save,sha,csvwrite
from rank2_data import independent
OUT=ROOT/'results/channel_phase_balance_20260916';OLD=ROOT/'results/channel_sharing_screen_v1_20260915'

def hashes(m):
    for p,h in m.items():assert sha(ROOT/p)==h,('HASH_CHANGED',p)

def main():
    status=read(OUT/'status.json');assert status['status']=='COMPLETE'
    seal=read(OUT/'seal.json');hashes(seal['source_hashes']);history=read(OUT/'historical_hashes.json');hashes(history)
    fits=read(OUT/'fits.json');trajectory=read(OUT/'trajectory.json');evaluation=read(OUT/'evaluation.json');controls=read(OUT/'reused_controls.json');oldfits=read(OLD/'fits.json')
    assert status['fit_attempts']==status['fits_completed']==len(fits)==4
    assert status['training_updates']==sum(f['updates'] for f in fits)<=5080 and status['smoke_updates']==4
    checks=[];cache={}
    for r in trajectory+evaluation+[f['replay'] for f in fits]+controls:
        p=r['prediction_path'];assert sha(ROOT/p)==r['prediction_sha256']
        if p not in cache:
            with np.load(ROOT/p) as z:a={k:z[k].copy() for k in z.files}
            mse=independent(a['prediction'],a['target'])
            channels=[]
            for c in range(a['target'].shape[1]):
                pairs=[(float(x),float(y)) for x,y in zip(a['prediction'][:,c].flat,a['target'][:,c].flat) if math.isfinite(float(y))]
                channels.append(math.fsum(abs(x-y) for x,y in pairs)/len(pairs))
            cache[p]=dict(mse=mse,mae=math.fsum(channels)/len(channels),arrays=a)
        scal=cache[p]
        assert math.isclose(scal['mse'],r['metrics']['mse'],rel_tol=1e-12,abs_tol=1e-12)
        assert math.isclose(scal['mae'],r['metrics']['mae'],rel_tol=1e-12,abs_tol=1e-12)
        checks.append(abs(scal['mse']-r['metrics']['mse']))
        if 'checkpoint_path' in r:assert sha(ROOT/r['checkpoint_path'])==r['checkpoint_sha256']
    schedule=read(OUT/'schedules.json');oldcontract=read(OLD/'contract.json');old_schedules=read(OLD/'schedules.json')
    for d,dc in seal['data'].items():
        for field in ['validation','evaluation']:assert dc['origins'][field]==oldcontract['data'][d]['origins'][field]
        for field in ['mean','std','staged','channel_ids','t1','t2']:assert dc[field]==oldcontract['data'][d][field]
        mapping=dict(zip(oldcontract['data'][d]['origins']['train'],dc['origins']['train']))
        for seed in [41000,41001]:
            assert all([mapping[o] for o in before]==after for before,after in zip(old_schedules[f'{d}_{seed}'],schedule[f'{d}_{seed}']))
            assert len(schedule[f'{d}_{seed}'])==20
            for oo in schedule[f'{d}_{seed}']:assert sorted(oo)==dc['origins']['train']
    exposure=[]
    for d,dc in seal['data'].items():
        before=oldcontract['data'][d]['origins']['train'];after=dc['origins']['train']
        a={t for o in before for t in range(o,o+96)};b={t for o in after for t in range(o,o+96)}
        exposure.append(dict(dataset=d,old_unique_target_rows=len(a),new_unique_target_rows=len(b),new_target_rows_not_in_old=len(b-a),old_target_rows_not_in_new=len(a-b)))
        assert not b-a
    save(OUT/'target_exposure_audit.json',exposure)
    for f in fits:
        rr=[r for r in trajectory if r['fit']==f['fit']];assert min(rr,key=lambda r:(r['metrics']['mse'],r['epoch']))==f['best']
        steps=read(OUT/(f['fit']+'_steps.json'));assert len(steps)==f['updates'] and [r['update'] for r in steps]==list(range(1,f['updates']+1))
        assert f['updates']==f['epochs']*{'electricity':64,'traffic':63}[f['dataset']]
        assert f['frozen_and_buffers_unchanged'] and f['replay_max_abs']==0
    rows=[];decomp=[]
    for r in evaluation:
        d,seed=r['dataset'],r['seed'];f=next(f for f in fits if f['dataset']==d and f['seed']==seed)
        oldf=next(f for f in oldfits if f['dataset']==d and f['seed']==seed and f['arm']=='LH')
        if r['role']=='matched_old_epoch':assert r['epoch']==oldf['best']['epoch'] and r['updates']==oldf['best']['updates']
        for base in ['LH','SEASONAL_NAIVE']:
            b=next(b for b in controls if b['dataset']==d and b['arm']==base and (b['seed']==seed or base=='SEASONAL_NAIVE') and b['role']!='INIT')
            pa=cache[r['prediction_path']]['arrays'];ba=cache[b['prediction_path']]['arrays']
            assert np.array_equal(pa['target'],ba['target'],equal_nan=True) and np.array_equal(pa['origins'],ba['origins']) and np.array_equal(pa['std'],ba['std'])
            rows.append(dict(dataset=d,seed=seed,role=r['role'],epoch=r['epoch'],updates=r['updates'],baseline=base,mse=r['metrics']['mse'],baseline_mse=b['metrics']['mse'],gain_percent=100*(b['metrics']['mse']-r['metrics']['mse'])/b['metrics']['mse'],mae=r['metrics']['mae'],baseline_mae=b['metrics']['mae'],raw_mae=float(np.mean(r['metrics']['channel_raw_mae'])),baseline_raw_mae=float(np.mean(b['metrics']['channel_raw_mae'])),peak_mib=f['peak_allocated']/2**20,baseline_peak_mib=oldf['peak_allocated']/2**20 if base=='LH' else 0,fit_active_seconds=f['active_seconds']))
        # Descriptive decomposition, same as the already completed residual diagnostic.
        a=cache[r['prediction_path']]['arrays'];err=a['prediction'].astype(np.float64)-a['target'];level=err.mean(-1,keepdims=True);daily=np.tile(err.reshape(-1,32,4,24).mean(2)-level,(1,1,4));remaining=err-level-daily
        parts=[float(np.mean(v**2)) for v in [level,daily,remaining]];total=float(np.mean(err**2));assert math.isclose(sum(parts),total,rel_tol=1e-12,abs_tol=1e-12)
        decomp.append(dict(dataset=d,seed=seed,role=r['role'],mse=total,level_mse=parts[0],daily_mse=parts[1],remainder_mse=parts[2],daily_share_percent=100*parts[1]/total))
    csvwrite(OUT/'comparisons.csv',rows);csvwrite(OUT/'decomposition.csv',decomp)
    verification=dict(at=time.time(),unique_prediction_caches=len(cache),scalar_mse_and_mae_records=len(checks),max_mse_abs_error=max(checks),all_fixed_epoch_updates_matched=True,all_selected_checkpoint_replays_exact=True,selected_replay_count=4,all_V_E_origins_and_normalization_unchanged=True,schedules_verified=True,old_sample_index_permutations_preserved=True,no_new_unique_target_rows=True,historical_files_preserved=len(history),source_seal_unchanged=True,report_source_sha256=sha(Path(__file__)))
    save(OUT/'verification.json',verification)
    decisions=[]
    for d in seal['data']:
        dd=dict(dataset=d)
        for role in ['selected','matched_old_epoch']:
            rr=[r for r in rows if r['dataset']==d and r['role']==role and r['baseline']=='LH'];assert len(rr)==2
            dd[role]=dict(both_seeds_improved=all(r['gain_percent']>0 for r in rr),balanced_mean_mse=float(np.mean([r['mse'] for r in rr])),old_mean_mse=float(np.mean([r['baseline_mse'] for r in rr])))
            dd[role]['gain_percent']=100*(dd[role]['old_mean_mse']-dd[role]['balanced_mean_mse'])/dd[role]['old_mean_mse']
        decisions.append(dd)
    save(OUT/'decision.json',dict(execution='COMPLETE',evidence='EXPOSED_DEVELOPMENT_SAMPLING_CONTROL',novelty='KNOWN_SAMPLING_CONTROL',new_peft_topic='NOT_ESTABLISHED',sources=decisions,additional_fits_launched=0))
    lines=['# 시작 위상 균형 LoRA 대조 결과','', '**단일 sampling 변경의 개발 대조를 완료했다. 알려진 표본 설계이므로 결과가 좋아도 새로운 PEFT 방법론 PASS는 아니다.**','',f"신규4/4 fits, {status['training_updates']}/5080 optimizer updates, smoke4 updates. 기존 LH4fits는 재학습하지 않았다. 모든 E 평가는 이미 노출된 같은 구간이다. 모델·원자료·채널·split·정규화·V/E 원점·학습률·epoch 상한은 유지했다.",'', '## 검증한 변경','', '기존 날짜별 원점 a_i에 i mod24 시간을 더하고, train 끝을 넘는 경우24시간을 뺐다. 기존 시점에서 최대23시간 이동하며512/504개 창 수를 유지한다. 모델은 같은761952개 trainable parameters의 LH다. 각seed의 초기 학습 파라미터는 기존 epoch0 checkpoint와 동일했고, BF16 smoke의 초기 V 예측도 기존 저장값과 정확히 같았다.','', '선택 checkpoint 비교와 별개로 기존 V 선택 epoch에 맞춘 비교를 사전에 고정했다. 후자는 optimizer updates 수가 정확히 같다. 창의 개별 값과 순서도 바뀌므로 전체 sampling 변경의 효과이며 시간 위상 하나의 순수 인과효과를 증명한 것은 아니다.','', '## 원점수와 효과','', '| 원천 | seed | 평가 recipe | epoch / updates | 새 MSE | 기존 LH MSE | LH 대비 개선 |','|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        if r['baseline']=='LH':lines.append(f"| {r['dataset']} | {r['seed']} | {r['role']} | {r['epoch']} / {r['updates']} | {r['mse']:.6f} | {r['baseline_mse']:.6f} | {r['gain_percent']:+.3f}% |")
    lines+=['','| 원천 | recipe | 새 평균 MSE | 기존 평균 MSE | 평균 개선율 | 두 seed 모두 개선 |','|---|---|---:|---:|---:|---|']
    for d in decisions:
        for role in ['selected','matched_old_epoch']:
            r=d[role];lines.append(f"| {d['dataset']} | {role} | {r['balanced_mean_mse']:.6f} | {r['old_mean_mse']:.6f} | {r['gain_percent']:+.3f}% | {r['both_seeds_improved']} |")
    lines+=['','계절 반복과의 비교, normalized MAE, 원단위MAE, 자원 사용량은 [comparisons.csv](comparisons.csv)에 있다. 효과는같은원천의 macro에서100×(기존−새값)/기존으로 계산했다. 서로 다른 원천의 raw MAE를 섞지 않는다. 이전 online 일별 잔차 보정은 E 정답을 추가 사용하므로 이 표의 동일정보 대조에 포함하지 않았다.','', '## 실제 자원·검산·미실행','', '| fit | epochs | updates | peak MiB | optimizer active seconds |','|---|---:|---:|---:|---:|']
    for f in fits:lines.append(f"| {f['fit']} | {f['epochs']} | {f['updates']} | {f['peak_allocated']/2**20:.3f} | {f['active_seconds']:.2f} |")
    wall=read(OUT/'finite_diagnostic_wall.json');events=[json.loads(x) for x in (OUT/'gpu_finite_diagnostic.jsonl').read_text().splitlines()];external=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in e['apps']) for e in events)
    lines+=['',f"Controller wall {wall['seconds']:.1f}초, 최소GPU여유 {wall['minimum_free_mib']}MiB, 비승인compute 표본 {external}개. 모델 파라미터 수는 변하지 않았으므로 파라미터 절감은0이다. 20epoch 상한 도달을 수렴으로 간주하지 않는다. 선택된epoch와 실제 종료epoch가 달라 총연구비용과 배포 적응비용을 구분해야 한다.",'',f"고유예측cache {len(cache)}개/{len(checks)}개 MSE·MAE 기록을 독립 scalar로 검산했다. 최대 MSE차 {max(checks):.3g}. 선택 checkpoint4개를 새 모델에서 재생해 차이0, 같은epoch 비교의updates 일치, 기존 {len(history)}개 결과 파일 보존을 확인했다. 모든 새 train 창은 train 경계 안에 있고 미래값 오염의 영향을 받지 않았다.",'', '미완료 fits0. 추가 후보·추가seed·학습률 조정·평가원점 교체0. 다른 시작 위상/새 원천의 독립 확인과 새로운 PEFT 방법의 추가 가치는 이 실행에서 검증하지 않았다. 동일 sampler를 사용한 Time-PEFT 채널 모듈 재평가도 아직 하지 않았다.','', '## 연구 판단의 범위','', '실행은 COMPLETE, 근거는 EXPOSED_DEVELOPMENT_SAMPLING_CONTROL, 신규성은 KNOWN_SAMPLING_CONTROL이다. sampling 대조가 좋아졌다면 이전 제한된 학습 시작점에서 얻은 모든 모듈 순위를 일반적인 학습 조건으로 넓히지 않아야 한다. 좋아지지 않은 조건도 함께 공개한다. 새 sampling의 고유 target 시점은 기존 target 시점의 부분집합으로, 새로운 정답 시점 추가는0이었다. [정답 노출 감사](target_exposure_audit.json). 새 알고리즘의 유용성은 이 강한 baseline과 같은 정보량의 단순 대조를 넘을 때 별도로 검토해야 한다.','', '[봉인 프로토콜](PROTOCOL.md), [표본 감사](sampler_audit.json), [검산](verification.json), [판단](decision.json). 원자료·가중치·예측 배열은 로컬cache에 남고 GitHub에는 source·원점수·hash·검증 기록을 보관한다.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verification=verification,decisions=decisions),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
