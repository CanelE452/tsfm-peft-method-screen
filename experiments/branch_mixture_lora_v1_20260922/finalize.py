import json
import math
import numpy as np
import pandas as pd
from common import *
from metrics import point_loss, empirical, apply, summarize, bootstrap
import data

WINDOWS={'block1':(0,64),'block2':(64,128),'full128':(0,128)}

def main():
    seal=read(RESULTS/'SELECTION_SEAL.json')
    assert seal['selections_sha']==sha(RESULTS/'SELECTIONS.json')
    assert seal['calibration_sha']==sha(RESULTS/'CALIBRATION.json')
    training=read(RESULTS/'TRAINING_SEAL.json')
    assert training['source_hashes']==source_hashes() and training['protocol_sha']==sha(EXP/'PROTOCOL.md')
    manifest=read(RESULTS/'PREDICTIONS_MANIFEST.json')
    for f in manifest['files']: assert sha(ROOT/f['path'])==f['sha256']
    fits=read(RESULTS/'SELECTIONS.json')
    assert len(fits)==4 and sum(r['updates'] for r in fits)==2048
    d=data.load(); sig=d['sigma']; origins=d['origins']['TEST']
    y=np.stack([d['values'][o:o+128].T for o in origins]).astype(np.float64)
    params=read(RESULTS/'CALIBRATION.json')
    summaries=[]; channels=[]; origin_rows=[]; lead_rows=[]; distributions=[]
    losses={}; scalar_error=0.; crps_error=0.; identity_min=1e30
    for receipt in [r for r in manifest['files'] if r['role']=='TEST']:
        key=receipt['key']
        with np.load(ROOT/receipt['path']) as z:
            raw=z['q'].astype(np.float64)
            components=z['components'].astype(np.float64) if 'components' in z else None
        seed=int(key.split('_s')[1].split('_')[0]) if '_s' in key else 0
        arm=key.split('_s')[0] if seed else key
        selected=not key.endswith('_fixed512')
        variants={'raw':raw,'ordered':np.sort(raw,axis=-1)}
        if key in params: variants['affine']=apply(variants['ordered'],sig,params[key])
        for variant,q in variants.items():
            pl=point_loss(y,q,sig)
            # Independent algebraic score, full array, not a second call to the scorer.
            e=y[...,None]-q
            ref=(np.abs(e)+(2*np.arange(1,10)/10-1)*e).mean(-1)/sig[None,:,None]
            scalar_error=max(scalar_error,float(np.abs(ref-pl).max()))
            np.testing.assert_allclose(pl,ref,atol=2e-13,rtol=2e-13)
            for window,(lo,hi) in WINDOWS.items():
                sl=slice(lo,hi)
                record=dict(key=key,arm=arm,seed=seed,selected=selected,variant=variant,window=window)
                summaries.append(dict(**record,**summarize(y[:,:,sl],q[:,:,sl],sig)))
                losses[(key,variant,window)]=pl[:,:,sl].mean((1,2))
                for s,ident in enumerate(d['ids']):
                    channels.append(dict(**record,series=ident,**summarize(y[:,s:s+1,sl],q[:,s:s+1,sl],sig[s:s+1])))
                for i,o in enumerate(origins):
                    origin_rows.append(dict(**record,origin=int(o),scaled_pinball=float(pl[i,:,sl].mean())))
            for h in range(128):
                lead_rows.append(dict(key=key,arm=arm,seed=seed,selected=selected,variant=variant,lead=h+1,
                                      scaled_pinball=float(pl[:,:,h].mean())))
        if components is not None:
            atoms=components.transpose(0,1,3,2,4).reshape(len(origins),7,64,81)
            mix=empirical(atoms,y[:,:,64:])/sig[None,:,None]
            comp=empirical(components,y[:,:,None,64:]).mean(2)/sig[None,:,None]
            difference=comp-mix
            identity_min=min(identity_min,float(difference.min()))
            assert difference.min()>-1e-10
            # Explicit O(M^2) scalar implementation on fixed deterministic cells.
            for i,s,h in [(0,0,0),(13,3,31),(len(origins)-1,6,63)]:
                z=atoms[i,s,h]; target=y[i,s,h+64]
                scalar=sum(abs(float(v)-target) for v in z)/81
                scalar-=sum(abs(float(a)-float(b)) for a in z for b in z)/(2*81*81)
                crps_error=max(crps_error,abs(scalar/sig[s]-mix[i,s,h]))
            distributions.append(dict(key=key,arm=arm,seed=seed,selected=selected,
                 tail_mixture_crps=float(mix.mean()),tail_component_crps=float(comp.mean()),
                 tail_disagreement=float(difference.mean()),representation='equal-weight empirical atoms; uncalibrated'))
    summary=pd.DataFrame(summaries); channel=pd.DataFrame(channels)
    summary.to_csv(RESULTS/'scores.csv',index=False)
    channel.to_csv(RESULTS/'channel_scores.csv',index=False)
    pd.DataFrame(origin_rows).to_csv(RESULTS/'origin_scores.csv',index=False)
    pd.DataFrame(lead_rows).to_csv(RESULTS/'lead_scores.csv',index=False)
    pd.DataFrame(distributions).to_csv(RESULTS/'distribution_scores.csv',index=False)
    effects=[]; seed_effects=[]
    for fixed in (False,True):
        suffix='_fixed512' if fixed else ''
        for variant in ('ordered','affine'):
            if fixed and variant=='affine': continue
            for window in WINDOWS:
                a=[losses[(f'COMPONENT_s{s}{suffix}',variant,window)] for s in SEEDS]
                b=[losses[(f'MIXTURE_s{s}{suffix}',variant,window)] for s in SEEDS]
                effects.append(dict(selected=not fixed,variant=variant,window=window,**bootstrap(a,b)))
                for idx,s in enumerate(SEEDS):
                    seed_effects.append(dict(seed=s,selected=not fixed,variant=variant,window=window,
                        component=float(a[idx].mean()),mixture=float(b[idx].mean()),**bootstrap([a[idx]],[b[idx]])))
    effect_df=pd.DataFrame(effects); seeds_df=pd.DataFrame(seed_effects)
    effect_df.to_csv(RESULTS/'effects.csv',index=False)
    seeds_df.to_csv(RESULTS/'seed_effects.csv',index=False)
    ce=channel[(channel.selected)&(channel.variant=='ordered')&(channel.window=='block2')&(channel.seed>0)]
    ce=ce.groupby(['arm','series']).scaled_pinball.mean().unstack('arm')
    ce['relative_gain_percent']=100*(ce.COMPONENT-ce.MIXTURE)/ce.COMPONENT
    ce.to_csv(RESULTS/'channel_effects.csv')
    resources=[]
    for r in fits:
        resources.append(dict(key=r['key'],phase='training',seconds=r['training_seconds'],units=r['updates'],
            ms_per_unit=1000*r['training_seconds']/r['updates'],peak_allocated_mib=r['peak_allocated_bytes']/2**20,
            trainable_parameters=r['trainable_parameters'],selected_step=r['selected_step']))
    for r in manifest['files']:
        if r['role']=='TEST': resources.append(dict(key=r['key'],phase='inference',seconds=r['seconds'],units=r['examples'],
              ms_per_unit=1000*r['seconds']/r['examples'],peak_allocated_mib=r['peak_allocated_bytes']/2**20,
              peak_reserved_mib=r['peak_reserved_bytes']/2**20))
    resource_df=pd.DataFrame(resources); resource_df.to_csv(RESULTS/'resources.csv',index=False)
    selected=seeds_df[(seeds_df.selected)&(seeds_df.variant=='ordered')&(seeds_df.window=='block2')]
    primary=effect_df[(effect_df.selected)&(effect_df.variant=='ordered')&(effect_df.window=='block2')].iloc[0].to_dict()
    calibrated=effect_df[(effect_df.selected)&(effect_df.variant=='affine')&(effect_df.window=='block2')].iloc[0].to_dict()
    positives=int((selected.effect>0).sum())
    decision=['NO_POSITIVE_SIGNAL','INCONCLUSIVE_MIXED_SEEDS','POSITIVE_PILOT_SIGNAL'][positives]
    save(RESULTS/'decision.json',dict(execution='COMPLETE',scientific_decision=decision,positive_seeds=positives,
         primary=primary,calibrated=calibrated,novelty='NOT_ESTABLISHED',paper_pass=False,automatic_followup=False))
    # Recompute CAL scores via the independent absolute-error pinball identity.
    cal_y=np.stack([d['values'][o:o+128].T for o in d['origins']['CALIBRATION']]).astype(np.float64)
    calibration_error=0.
    for key,blocks in params.items():
        with np.load(CACHE/'predictions/CALIBRATION'/f'{key}.npz') as z: q=np.sort(z['q'].astype(np.float64),axis=-1)
        for k,block in enumerate(blocks):
            sl=slice(64*k,64*(k+1)); base=q[:,:,sl]; m=base[...,4:5]
            checks=[]
            for r in block['grid']:
                corrected=m+r['beta']*sig[None,:,None,None]+r['alpha']*(base-m)
                e=cal_y[:,:,sl,None]-corrected
                score=float(((np.abs(e)+(2*np.arange(1,10)/10-1)*e)/sig[None,:,None,None]).mean())
                calibration_error=max(calibration_error,abs(score-r['score']))
                checks.append(dict(alpha=r['alpha'],beta=r['beta'],score=score))
            # Match tie policy using original score precision after verifying all numeric values.
            winner=min(block['grid'],key=lambda r:(r['score'],(r['alpha']-1)**2+r['beta']**2,r['alpha'],abs(r['beta']),r['beta']))
            assert (winner['alpha'],winner['beta'])==(block['alpha'],block['beta'])
    assert scalar_error<1e-10 and crps_error<1e-10 and calibration_error<1e-10
    verification=dict(status='PASS',scalar_pinball_max_error=scalar_error,scalar_81atom_crps_max_error=crps_error,
         calibration_grid_max_error=calibration_error,calibration_points_verified=sum(len(b['grid']) for blocks in params.values() for b in blocks),
         minimum_component_minus_mixture_score=identity_min,prediction_files_sha_checked=len(manifest['files']),
         protocol_and_code_seal_unchanged=True,main_updates=2048,smoke_updates=4,paired_initialization_and_schedules=True,
         future_targets_used_for_context=False,selection_uses_test=False,trainable_parameter_count=fits[0]['trainable_parameters'],
         old_experiments_modified=False,independent_verification_scope='algebraic pinball + explicit pairwise empirical CRPS; not an independent second training run')
    save(RESULTS/'VERIFICATION.json',verification)
    figures(summary,seeds_df,ce)
    report(summary,resource_df,selected,primary,calibrated,decision,fits,verification)
    event('finalized',decision=decision,relative_percent=primary['relative_percent'])

def figures(summary,seeds,ce):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    colors={'COMPONENT':'#4477AA','MIXTURE':'#EE7733'}
    for arm in ARMS:
        rows=summary[(summary.selected)&(summary.arm==arm)&(summary.variant=='ordered')&(summary.window=='block2')].sort_values('seed')
        axes[0].plot([0,1],rows.scaled_pinball,'o-',color=colors[arm],label=arm)
    axes[0].set(xticks=[0,1],xticklabels=[str(s) for s in SEEDS],xlabel='Training seed',ylabel='Scaled twice-pinball (lower better)',title='ETTh2: deployed block 65-128')
    axes[0].legend(frameon=False)
    vals=ce.relative_gain_percent
    axes[1].barh(vals.index,vals.values,color=['#228833' if x>0 else '#CC6677' for x in vals])
    axes[1].axvline(0,color='black',lw=.8)
    axes[1].set(xlabel='Mixture gain over component (%)',title='Equal-seed mean, by channel')
    use=summary[(summary.selected)&(summary.window=='block2')&(summary.variant=='affine')]
    means=use.groupby('arm').scaled_pinball.mean().reindex(['F0_NATIVE','COMPONENT','MIXTURE','CHRONOS2_DIRECT'])
    axes[2].bar(range(4),means.values,color=['#BBBBBB','#4477AA','#EE7733','#66CCEE'])
    axes[2].set(xticks=range(4),xticklabels=['F0 native','Component','Mixture','Chronos-2'],ylabel='Scaled twice-pinball',title='Same CAL-only affine opportunity')
    axes[2].tick_params(axis='x',labelrotation=20)
    fig.savefig(RESULTS/'RESULTS.png',dpi=180)
    fig.savefig(RESULTS/'RESULTS.pdf')
    plt.close(fig)

def report(summary,resources,selected,primary,calibrated,decision,fits,verification):
    table=summary[(summary.selected)&(summary.window=='block2')&(summary.variant.isin(['ordered','affine']))].groupby(['arm','variant'])[['scaled_pinball','raw_mae','coverage80','scaled_width80']].mean().reset_index()
    training=resources[resources.phase=='training'][['key','seconds','peak_allocated_mib','trainable_parameters','selected_step']]
    means=table[table.variant=='affine'].set_index('arm').scaled_pinball
    practical='MIXTURE가 F0의 단순 보정보다 낮은 점수다.' if means['MIXTURE']<means['F0_NATIVE'] else 'MIXTURE가 F0의 단순 보정보다 낮은 점수를 확보하지 못했다.'
    text=f'''# Branch-mixture LoRA 제한 실험 결과

**실행 COMPLETE. 판단 {decision}.** ETTh2 하나에서 공식 branching의 혼합분포를 학습하는 LoRA를 경로별 CRPS 학습과 비교했다. 새 방법의 성능 신호와 구현 성공을 구분하며, 신규성·논문 PASS를 선언하지 않는다.

4/4 fits, 2,048/2,048 main updates, 4/4 smoke updates를 완료했다. 추가 LR/rank/seed/자료 탐색0회다. 두 반복 모두 비교에 포함했고 선택용 seed는 없다. 평균은 seed별 점수 평균이며 예측 ensemble이 아니다.

## 무엇을 바꾸었나

Chronos-Bolt-small 동결 + 동일 q/v rank8 LoRA({fits[0]['trainable_parameters']:,}개 학습 파라미터). 첫64 예측은 공통 pinball, 다음64 예측은 COMPONENT가 아홉 9-atom 분포의 CRPS를 평균하고 MIXTURE가 합친81-atom 분포의 CRPS를 사용한다. 두 loss의 차이는 경로 사이 CDF 불일치에 대한 비음수 항이다. 이 항을 제거하면 성능이 좋아질지는 가설이며 유용한 불확실성을 자동 보장하지 않는다.

현재 공식 Bolt는 중앙값-only가 아니다. 실제 공식9경로를 모두 사용하고 다음9×9를9분위수로 축약한다. 학습과 평가 모두 같은 branching이다. 생성한 context는 detach하며 미래 정답을 넣지 않는다. 공식 raw quantile-labelled path를 유지하고, 최종 공통 정렬 결과를 주 평가한다. 경험분포 CRPS는 연속분포 CRPS와 다르고 지지점은 IID sample이 아니다.

## 평가 범위와 선택

ETTh2 공식7계열, 17,420시간, context512/horizon128. 시간순 TRAIN60%/CAL10%/VAL10%/TEST20%, 일24시간 간격, target은 각 구간 안에 완전히 포함된다. TEST140원점, VAL/CAL각67원점이다. 과거 저장소 계획서에 ETTh2가 언급돼 완전히 새로운 자료로 주장하지 않는다. 이번 TEST는 이 실행의 선택·보정에 사용하지 않았다. 공개 benchmark/pretraining 노출 가능성까지 배제한 것은 아니다.

고정LR1e-4, batch4,512updates. 각 arm은 VAL에서 INIT/128/256/512 중 후반 scaled twice-pinball 최저 checkpoint를 선택했다. TEST를 보기 전에 checkpoint와 CAL 보정 계수를 봉인했다. 동일35점 affine grid를 각64시간 블록에 CAL만으로 적용했다.

## 결과

양수 개선율은 MIXTURE가 COMPONENT보다 좋다는 뜻이다. 주 지표는 공식9분위수 축약 후 후반65–128의 TRAIN-표준편차 정규화 mean twice-pinball이다. 임의의1% 기준을 사용하지 않았다.

```text
{selected[['seed','component','mixture','relative_percent','ci95_low','ci95_high']].to_string(index=False,float_format=lambda x:f'{x:.7f}')}
```

평균 상대 개선 **{primary['relative_percent']:+.4f}%**. 점수 차이 {primary['effect']:+.7f}, 원점 block-bootstrap95%구간 [{primary['ci95_low']:+.7f}, {primary['ci95_high']:+.7f}]. 동일 CAL 보정 이후 상대 개선 **{calibrated['relative_percent']:+.4f}%**, 구간 [{calibrated['ci95_low']:+.7f}, {calibrated['ci95_high']:+.7f}]. CI는 절대 점수 차이의 구간이다.

```text
{table.to_string(index=False,float_format=lambda x:f'{x:.6f}')}
```

{practical} Chronos-2 직접128 예측은 별도 practical reference이며 같은 모델/학습자원 대조는 아니다. 본4-fit 실험에는 별도 ordinary/median-rollout LoRA 학습군이 없으므로 모든 LoRA recipe 대비 우위나 일반 rollout 학습의 독립 효과를 주장하지 않는다.

![seed·channel·calibration 결과](RESULTS.png)

그림의 seed 점들은 두 독립 초기화/학습순서 반복이며 오차막대가 아니다. channel 효과는 두seed 평균이고 사후 채널 선택을 하지 않았다. 전체128/첫64/후반64, raw/ordered/affine, fixed512 민감도는 [scores.csv](scores.csv), [effects.csv](effects.csv), [seed_effects.csv](seed_effects.csv)에 있다. [channel](channel_scores.csv), [origin](origin_scores.csv), [lead](lead_scores.csv), [81-atom mixture와 disagreement](distribution_scores.csv)도 모두 남겼다.

Bootstrap은7개 일원점 블록·2000회로 모든 채널과seed를 함께 재표집한다. 중첩된 horizon을 독립 표본으로 세지 않는다. 단일 자료·두seed 조건부 시간변동 구간이며 데이터셋 모집단이나 충분한 seed 모집단 불확실성을 뜻하지 않는다.

## 자원과 검산

```text
{training.to_string(index=False,float_format=lambda x:f'{x:.3f}')}
```

시간은 본학습 forward/backward/update 합계이며 검증·checkpoint 쓰기·개발 시간은 포함하지 않는다. 두 arm의 main conditional batch forward/backward는 총4096회, series-context 계산81920개다. 모든 경우 같은branch9개와batch4를 사용했다. 추론 시간/allocated/reserved memory는 [resources.csv](resources.csv)에 있다. 실제 wall time과 GPU 상태는 events와 환경 기록에 보존한다.

[VERIFICATION.json](VERIFICATION.json): 실제 TRAIN 입력으로 native64/128 및 smoke 학습 후 공식API parity, zero-LoRA 초기 일치, frozen가중치 불변, paired 초기hash/schedule, finite gradient를 통과했다. 독립 절대오차식 pinball 최대차 {verification['scalar_pinball_max_error']:.3g}, 직접81×81 pairwise CRPS 최대차 {verification['scalar_81atom_crps_max_error']:.3g}, CAL {verification['calibration_points_verified']}개 점수 최대차 {verification['calibration_grid_max_error']:.3g}. 데이터SHA/시간축/누수·코드/선택/예측hash 검증을 남겼다. CPU 테스트는 [CPU_TESTS.json](CPU_TESTS.json).

재현 entry: `python experiments/branch_mixture_lora_v1_20260922/preflight.py`, `runner.py all`, `finalize.py`. 기존 완료경로에 smoke를 다시 실행하면 차단한다. 재실행은 새 예산의 연구 작업이며 자동 수행하지 않는다. manifest의 raw/weights/checkpoint/prediction 배열은 ignored 로컬 cache에 있고 GitHub에는 포함하지 않는다. requirements lock은 실제 재사용Python3.11 환경이다.

## 해석과 종료

사전 정의한 방향 반복 기준의 결과는 **{decision}**이다. 이 결과는 mixture-loss LoRA의 한정된pilot이며 TSFM PEFT 전체를 지지하거나 반증하지 않는다. CAL 대비 손익·F0 단순보정·direct모델·추가자원을 함께 판단해야 한다. 별도 architecture/seed/자료를 자동 추가하지 않고 종료했다.

[고정 설계와 선행연구](../../experiments/{NAME}/PROTOCOL.md). Ensemble CRPS rollout 학습 자체는 AIFS-CRPS/Aurora1.5 등의 선행이 있어 신규성은 미확정이다. 구현·자료·자원 실패와 성능 음성은 분리한다.
'''
    (RESULTS/'REPORT_KO.md').write_text(text,encoding='utf-8')
    (RESULTS/'FINAL_DECISION.md').write_text(f'''# Final decision

EXECUTION: COMPLETE
SCIENTIFIC_SIGNAL: {decision}
PRIMARY_RELATIVE_GAIN_PERCENT: {primary['relative_percent']:+.7f}
CALIBRATED_RELATIVE_GAIN_PERCENT: {calibrated['relative_percent']:+.7f}
FITS: 4 / 4
MAIN_UPDATES: 2048 / 2048
SMOKE_UPDATES: 4 / 4
NOVELTY: NOT_ESTABLISHED
PAPER_PASS: NOT_CLAIMED
AUTOMATIC_FOLLOWUP: NONE

The result concerns ETTh2 512→128, frozen Chronos-Bolt-small q/v rank8 LoRA, component versus mixture empirical CRPS on native branches. It does not isolate generic rollout benefit or compare all standard LoRA recipes. Both arm selection and output calibration exclude TEST. See REPORT_KO.md for uncertainty, practical references and data exposure.
''',encoding='utf-8')

if __name__=='__main__': main()
