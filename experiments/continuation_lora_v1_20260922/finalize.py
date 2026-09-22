import numpy as np
import pandas as pd
from common import *
from metrics import point_loss, empirical, apply, summarize, bootstrap
import data


WINDOWS = {'block1': (0, 64), 'block2': (64, 128), 'full128': (0, 128)}


def independent_pinball(y, q, sigma):
    e = np.asarray(y, dtype=np.float64)[..., None] - np.asarray(q, dtype=np.float64)
    return (np.abs(e) + (2 * np.arange(1, 10) / 10 - 1) * e).mean(-1) / sigma[None, :, None]


def load_q(role, key):
    with np.load(CACHE / 'predictions' / role / f'{key}.npz') as z:
        return z['q'].astype(np.float64)


def verify_inputs():
    seal = read(RESULTS / 'SELECTION_SEAL.json')
    assert seal['selections_sha'] == sha(RESULTS / 'SELECTIONS.json')
    assert seal['calibration_sha'] == sha(RESULTS / 'CALIBRATION.json')
    assert seal['training_seal_sha'] == sha(RESULTS / 'TRAINING_SEAL.json')
    training = read(RESULTS / 'TRAINING_SEAL.json')
    assert training['source_hashes'] == source_hashes()
    assert training['protocol_sha'] == sha(EXP / 'PROTOCOL.md')
    assert training['data_sha'] == sha(RESULTS / 'DATA_AUDIT.json')
    assert training['attenuation_sha'] == sha(RESULTS / 'ATTENUATION_DIAGNOSTIC.json')
    attenuation = read(RESULTS / 'ATTENUATION_DIAGNOSTIC.json')
    assert attenuation['complete'] and attenuation['optimizer_updates'] == 0
    assert attenuation['old_test_read_or_scored'] is False
    assert attenuation['score_csv_sha'] == sha(RESULTS / 'attenuation_scores.csv')
    assert read(RESULTS / 'PREFLIGHT.json')['status'] == 'PASS'
    assert read(RESULTS / 'CPU_TESTS.json')['returncode'] == 0
    audit = read(RESULTS / 'DATA_AUDIT.json')
    source = audit['source']
    assert source['quality_decision'] == 'PASS'
    assert sha(ROOT / source['raw_path']) == source['raw_sha256']
    assert sha(ROOT / source['prepared_npz']['path']) == source['prepared_npz']['sha256']
    for r in source['train_schedules'].values():
        assert sha(ROOT / r['path']) == r['sha256']
    manifest = read(RESULTS / 'PREDICTIONS_MANIFEST.json')
    assert manifest['selection_seal_sha'] == sha(RESULTS / 'SELECTION_SEAL.json')
    for f in manifest['files']:
        assert sha(ROOT / f['path']) == f['sha256']
    assert sum(f['role'] == 'TEST' for f in manifest['files']) == 10
    fits = read(RESULTS / 'SELECTIONS.json')
    ledger = read(RESULTS / 'OPTIMIZER_LEDGER.json')
    assert len(fits) == ledger['fits'] == 4
    assert sum(r['updates'] for r in fits) == ledger['main_updates'] == 2048
    assert ledger['smoke_updates'] == read(RESULTS / 'PREFLIGHT.json')['smoke_updates'] == 4
    assert ledger['selections_sha'] == sha(RESULTS / 'SELECTIONS.json')
    for seed in SEEDS:
        pair = [r for r in fits if r['seed'] == seed]
        assert {r['arm'] for r in pair} == set(ARMS)
        assert len({r['initial_hash'] for r in pair}) == len({r['schedule_sha'] for r in pair}) == 1
    for r in fits:
        assert r['frozen_unchanged']
        assert [v['step'] for v in r['loss_records']] == list(range(1, 513))
        for checkpoint in r['validation'].values():
            assert sha(ROOT / checkpoint['path']) == checkpoint['sha256']
    return audit, manifest, fits, ledger


def verify_selection_and_calibration(d, fits, params):
    sig = d['sigma']
    val_y = np.stack([d['values'][o:o + 128].T for o in d['origins']['VALIDATION']])
    cal_y = np.stack([d['values'][o:o + 128].T for o in d['origins']['CALIBRATION']])
    val_error = 0.
    val_rows = []
    for r in fits:
        checked = {}
        assert set(r['validation']) == {'0', '128', '256', '512'}
        for step, v in r['validation'].items():
            q = np.sort(load_q('VALIDATION', f"{r['key']}_step{step}"), axis=-1)
            score = float(independent_pinball(val_y[:, :, 64:], q[:, :, 64:], sig).mean())
            val_error = max(val_error, abs(score - v['score']))
            checked[step] = score
            val_rows.append(dict(key=r['key'],step=int(step),sealed_score=v['score'],independent_score=score))
        winner = min(checked, key=lambda k: (checked[k], int(k)))
        assert int(winner) == r['selected_step']
        assert r['checkpoint'] == r['validation'][winner]
    calibration_error = 0.
    calibration_points = 0
    f0 = load_q('CALIBRATION', 'F0_NATIVE')
    for key, blocks in params.items():
        raw = load_q('CALIBRATION', key)
        if key.startswith('CONTINUATION_s'):
            assert np.array_equal(raw[:, :, :64], f0[:, :, :64])
            assert blocks[0] == params['F0_NATIVE'][0]
        q = np.sort(raw, axis=-1)
        for k, block in enumerate(blocks):
            sl = slice(64 * k, 64 * (k + 1))
            base = q[:, :, sl]
            med = base[..., 4:5]
            checks = []
            for trial in block['grid']:
                corrected = med + trial['beta'] * sig[None, :, None, None] + trial['alpha'] * (base - med)
                score = float(independent_pinball(cal_y[:, :, sl], corrected, sig).mean())
                calibration_error = max(calibration_error, abs(score - trial['score']))
                calibration_points += 1
                checks.append(dict(alpha=trial['alpha'], beta=trial['beta'], score=score))
            winner = min(checks, key=lambda v: (v['score'], (v['alpha'] - 1) ** 2 + v['beta'] ** 2, v['alpha'], abs(v['beta']), v['beta']))
            assert (winner['alpha'], winner['beta']) == (block['alpha'], block['beta'])
    assert val_error < 1e-10 and calibration_error < 1e-10
    pd.DataFrame(val_rows).to_csv(RESULTS / 'validation_verification.csv', index=False)
    return dict(validation_checkpoint_count=len(val_rows),validation_score_max_error=val_error,
                validation_selection_matches=len(fits),calibration_grid_max_error=calibration_error,
                calibration_points_verified=calibration_points,calibration_first64_exact_f0=True,
                calibration_first64_parameters_exact_f0=True)


def main():
    audit, manifest, fits, ledger = verify_inputs()
    d = data.load()
    sig = d['sigma']
    origins = d['origins']['TEST']
    nseries = len(d['ids'])
    assert nseries == 8
    y = np.stack([d['values'][o:o + 128].T for o in origins]).astype(np.float64)
    params = read(RESULTS / 'CALIBRATION.json')
    selection_check = verify_selection_and_calibration(d, fits, params)
    f0 = load_q('TEST', 'F0_NATIVE')
    f0_calibrated = apply(np.sort(f0, axis=-1), sig, params['F0_NATIVE'])
    summaries, channels, origin_rows, lead_rows, distributions = [], [], [], [], []
    losses = {}
    scalar_error, crps_error, first64_error = 0., 0., 0.
    identity_min = float('inf')
    for receipt in [r for r in manifest['files'] if r['role'] == 'TEST']:
        key = receipt['key']
        with np.load(ROOT / receipt['path']) as z:
            raw = z['q'].astype(np.float64)
            components = z['components'].astype(np.float64) if 'components' in z else None
        seed = int(key.split('_s')[1].split('_')[0]) if '_s' in key else 0
        arm = key.split('_s')[0] if seed else key
        selected = not key.endswith('_fixed512')
        variants = {'raw': raw, 'ordered': np.sort(raw, axis=-1)}
        if key in params:
            variants['affine'] = apply(variants['ordered'], sig, params[key])
        if arm == 'CONTINUATION':
            first64_error = max(first64_error, float(np.abs(raw[:, :, :64] - f0[:, :, :64]).max()))
            assert np.array_equal(raw[:, :, :64], f0[:, :, :64])
            if selected:
                assert np.array_equal(variants['affine'][:, :, :64], f0_calibrated[:, :, :64])
        for variant, q in variants.items():
            pl = point_loss(y, q, sig)
            independent = independent_pinball(y, q, sig)
            scalar_error = max(scalar_error, float(np.abs(pl - independent).max()))
            np.testing.assert_allclose(pl, independent, atol=2e-13, rtol=2e-13)
            for window, (lo, hi) in WINDOWS.items():
                sl = slice(lo, hi)
                record = dict(key=key,arm=arm,seed=seed,selected=selected,variant=variant,window=window)
                summaries.append(dict(**record, **summarize(y[:, :, sl], q[:, :, sl], sig)))
                losses[(key, variant, window)] = pl[:, :, sl].mean((1, 2))
                for s, ident in enumerate(d['ids']):
                    channels.append(dict(**record,series=ident,**summarize(y[:, s:s+1, sl], q[:, s:s+1, sl], sig[s:s+1])))
                for i, origin in enumerate(origins):
                    origin_rows.append(dict(**record,origin=int(origin),scaled_pinball=float(pl[i, :, sl].mean())))
            for h in range(128):
                lead_rows.append(dict(key=key,arm=arm,seed=seed,selected=selected,variant=variant,lead=h+1,scaled_pinball=float(pl[:, :, h].mean())))
        if components is not None:
            atoms = components.transpose(0, 1, 3, 2, 4).reshape(len(origins), nseries, 64, 81)
            mix = empirical(atoms, y[:, :, 64:]) / sig[None, :, None]
            comp = empirical(components, y[:, :, None, 64:]).mean(2) / sig[None, :, None]
            difference = comp - mix
            identity_min = min(identity_min, float(difference.min()))
            assert difference.min() > -1e-10
            for i, s, h in [(0, 0, 0), (min(13, len(origins)-1), 3, 31), (len(origins)-1, nseries-1, 63)]:
                z = atoms[i, s, h]
                target = y[i, s, h+64]
                scalar = sum(abs(float(v)-target) for v in z) / 81
                scalar -= sum(abs(float(a)-float(b)) for a in z for b in z) / (2*81*81)
                crps_error = max(crps_error, abs(scalar/sig[s] - mix[i, s, h]))
            distributions.append(dict(key=key,arm=arm,seed=seed,selected=selected,tail_mixture_crps=float(mix.mean()),
                                     tail_component_crps=float(comp.mean()),tail_disagreement=float(difference.mean()),
                                     representation='equal-weight empirical atoms; uncalibrated'))
    summary = pd.DataFrame(summaries)
    channel = pd.DataFrame(channels)
    summary.to_csv(RESULTS / 'scores.csv', index=False)
    channel.to_csv(RESULTS / 'channel_scores.csv', index=False)
    pd.DataFrame(origin_rows).to_csv(RESULTS / 'origin_scores.csv', index=False)
    pd.DataFrame(lead_rows).to_csv(RESULTS / 'lead_scores.csv', index=False)
    pd.DataFrame(distributions).to_csv(RESULTS / 'distribution_scores.csv', index=False)
    effects, references, seed_effects = [], [], []
    for fixed in (False, True):
        suffix = '_fixed512' if fixed else ''
        for variant in ('ordered', 'affine'):
            if fixed and variant == 'affine':
                continue
            for window in WINDOWS:
                comparisons = [('CONTINUATION_vs_SHARED', 'SHARED', 'CONTINUATION')]
                comparisons += [(f'{arm}_vs_{base}', base, arm) for arm in ARMS for base in ('F0_NATIVE', 'CHRONOS2_DIRECT')]
                for comparison, baseline, candidate in comparisons:
                    a = [losses[(f'{baseline}_s{s}{suffix}' if baseline in ARMS else baseline,variant,window)] for s in SEEDS]
                    b = [losses[(f'{candidate}_s{s}{suffix}',variant,window)] for s in SEEDS]
                    record = dict(comparison=comparison,baseline=baseline,candidate=candidate,selected=not fixed,variant=variant,window=window)
                    row = dict(**record,baseline_score=float(np.mean(a)),candidate_score=float(np.mean(b)),**bootstrap(a,b))
                    (effects if baseline == 'SHARED' else references).append(row)
                    for idx, seed in enumerate(SEEDS):
                        seed_effects.append(dict(**record,seed=seed,baseline_score=float(a[idx].mean()),candidate_score=float(b[idx].mean()),**bootstrap([a[idx]],[b[idx]])))
    effect_df = pd.DataFrame(effects)
    reference_df = pd.DataFrame(references)
    seeds_df = pd.DataFrame(seed_effects)
    effect_df.to_csv(RESULTS / 'effects.csv', index=False)
    reference_df.to_csv(RESULTS / 'reference_effects.csv', index=False)
    seeds_df.to_csv(RESULTS / 'seed_effects.csv', index=False)
    channel_effects = []
    for variant in ('ordered', 'affine'):
        means = channel[channel.selected & (channel.variant == variant) & (channel.window == 'block2')].groupby(['series','arm']).scaled_pinball.mean().unstack('arm')
        for ident, row in means.iterrows():
            for baseline in ('SHARED', 'F0_NATIVE', 'CHRONOS2_DIRECT'):
                channel_effects.append(dict(series=ident,variant=variant,baseline=baseline,candidate='CONTINUATION',
                    baseline_score=row[baseline],candidate_score=row['CONTINUATION'],
                    relative_gain_percent=100*(row[baseline]-row['CONTINUATION'])/row[baseline]))
    ce = pd.DataFrame(channel_effects)
    ce.to_csv(RESULTS / 'channel_effects.csv', index=False)
    resources = []
    for r in fits:
        resources.append(dict(key=r['key'],phase='training',seconds=r['training_seconds'],units=r['updates'],
            ms_per_unit=1000*r['training_seconds']/r['updates'],peak_allocated_mib=r['peak_allocated_bytes']/2**20,
            trainable_parameters=r['trainable_parameters'],selected_step=r['selected_step'],
            conditional_forward_calls=r['batch_conditional_forward_calls'],backward_calls=r['batch_backward_calls']))
    for r in manifest['files']:
        if r['role'] == 'TEST':
            resources.append(dict(key=r['key'],phase='inference',seconds=r['seconds'],units=r['examples'],
                ms_per_unit=1000*r['seconds']/r['examples'],peak_allocated_mib=r['peak_allocated_bytes']/2**20,
                peak_reserved_mib=r['peak_reserved_bytes']/2**20))
    resource_df = pd.DataFrame(resources)
    resource_df.to_csv(RESULTS / 'resources.csv', index=False)
    primary = reference_df[(reference_df.comparison == 'CONTINUATION_vs_F0_NATIVE') & reference_df.selected & (reference_df.variant == 'affine') & (reference_df.window == 'block2')].iloc[0].to_dict()
    practical_seeds = seeds_df[(seeds_df.comparison == 'CONTINUATION_vs_F0_NATIVE') & seeds_df.selected & (seeds_df.variant == 'affine') & (seeds_df.window == 'block2')]
    structural = {variant: effect_df[effect_df.selected & (effect_df.variant == variant) & (effect_df.window == 'block2')].iloc[0].to_dict() for variant in ('ordered','affine')}
    positives = int((practical_seeds.effect > 0).sum())
    decision = 'PRACTICAL_DIRECTION_POSITIVE_PILOT_SIGNAL' if positives == 2 else 'STOP_CURRENT_CONTINUATION_REFINEMENT_NO_REPEATABLE_PRACTICAL_ADVANTAGE'
    assert scalar_error < 1e-10 and crps_error < 1e-10 and first64_error == 0
    verification = dict(status='PASS',**selection_check,scalar_pinball_max_error=scalar_error,
        scalar_81atom_crps_max_error=crps_error,minimum_component_minus_mixture_score=identity_min,
        test_first64_raw_max_error=first64_error,test_first64_raw_and_affine_exact_f0=True,
        prediction_files_sha_checked=len(manifest['files']),raw_prepared_schedule_hashes_checked=True,
        protocol_and_code_seal_unchanged=True,main_updates=2048,smoke_updates=4,
        paired_initialization_and_schedules=True,optimizer_steps_1_to_512_verified=True,
        trainable_parameter_count=fits[0]['trainable_parameters'],selection_uses_test=False,
        old_test_reevaluated=False,independent_verification_scope='independent algebraic pinball, explicit pairwise CRPS, VAL selection and CAL recomputation; no second training run')
    save(RESULTS / 'VERIFICATION.json', verification)
    save(RESULTS / 'decision.json', dict(execution='COMPLETE',scientific_decision=decision,
        primary_practical=primary,practical_positive_seeds=positives,structural=structural,
        novelty='NOT_ESTABLISHED_KNOWN_GATING_TRANSFER',paper_pass=False,automatic_followup=False))
    figures(summary, ce)
    report(summary, resource_df, practical_seeds, primary, structural, reference_df, decision, fits, verification, d, audit, ledger)
    event('finalized',decision=decision,practical_relative_percent=primary['relative_percent'])
    save(RESULTS / 'ARTIFACT_MANIFEST.json', dict(files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(RESULTS.rglob('*')) if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json'],
        local_only='Raw data, model weights, checkpoint and prediction arrays are ignored local caches; public receipts alone do not reproduce every number.'))


def figures(summary, ce):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
    colors = {'SHARED': '#4477AA', 'CONTINUATION': '#EE7733'}
    use = summary[summary.selected & (summary.variant == 'affine') & (summary.window == 'block2')]
    for arm in ARMS:
        rows = use[use.arm == arm].sort_values('seed')
        axes[0].plot(range(len(SEEDS)), rows.scaled_pinball, 'o-', color=colors[arm], label=arm)
    for arm, color, style in [('F0_NATIVE','#555555','--'), ('CHRONOS2_DIRECT','#66AA99',':')]:
        axes[0].axhline(use[use.arm == arm].scaled_pinball.iloc[0], color=color, ls=style, label=arm)
    axes[0].set(xticks=range(len(SEEDS)),xticklabels=[str(s) for s in SEEDS],xlabel='Training seed',ylabel='Scaled twice-pinball (lower better)',title='Solar: calibrated continuation 65-128')
    axes[0].legend(frameon=False,fontsize=8)
    rows = ce[(ce.variant == 'affine') & (ce.baseline == 'F0_NATIVE')]
    axes[1].barh(rows.series.astype(str),rows.relative_gain_percent,color=['#228833' if x > 0 else '#CC6677' for x in rows.relative_gain_percent])
    axes[1].axvline(0,color='black',lw=.8)
    axes[1].set(xlabel='Continuation gain over F0 (%)',ylabel='Preselected column ID',title='Practical effect, all eight series')
    means = use.groupby('arm').scaled_pinball.mean().reindex(['F0_NATIVE','SHARED','CONTINUATION','CHRONOS2_DIRECT'])
    axes[2].bar(range(4),means.values,color=['#BBBBBB','#4477AA','#EE7733','#66CCEE'])
    axes[2].set(xticks=range(4),xticklabels=['F0 native','Shared','Continuation','Chronos-2'],ylabel='Scaled twice-pinball',title='Same CAL-only affine opportunity')
    axes[2].tick_params(axis='x',labelrotation=20)
    fig.savefig(RESULTS / 'RESULTS.png',dpi=180)
    fig.savefig(RESULTS / 'RESULTS.pdf')
    plt.close(fig)


def report(summary, resources, practical_seeds, primary, structural, references, decision, fits, verification, d, audit, ledger):
    table = summary[summary.selected & (summary.window == 'block2') & summary.variant.isin(['ordered','affine'])].groupby(['arm','variant'])[['scaled_pinball','raw_mae','coverage80','scaled_width80']].mean().reset_index()
    training = resources[resources.phase == 'training'][['key','seconds','peak_allocated_mib','trainable_parameters','selected_step','backward_calls']]
    attenuation = pd.read_csv(RESULTS / 'attenuation_scores.csv')
    attenuation_receipt = read(RESULTS / 'ATTENUATION_DIAGNOSTIC.json')
    fixed = references[(references.comparison == 'CONTINUATION_vs_F0_NATIVE') & ~references.selected & (references.variant == 'ordered') & (references.window == 'block2')].iloc[0]
    shared = references[(references.comparison == 'SHARED_vs_F0_NATIVE') & references.selected & (references.variant == 'affine') & (references.window == 'block2')].iloc[0]
    direction = '두 seed에서 보정 F0 대비 개선 방향이 반복됐다. 제한된 practical-direction pilot 신호이며 확정적인 우위 판정은 아니다.' if (practical_seeds.effect > 0).all() else '보정 F0 대비 개선 방향이 두 seed에서 반복되지 않았다. 현재 continuation refinement는 종료하며 새 후보를 자동 실행하지 않는다.'
    text = f'''# Continuation-only LoRA 제한 실험 결과

**실행 COMPLETE. 판단 {decision}.** {direction} 이 판단의 주 비교는 후반64에서 **CONTINUATION+CAL 대 F0+같은 CAL**이다. SHARED보다 좋은 것만으로 실용적 우위를 선언하지 않는다.

4/4 fits, 2,048/2,048 main updates, 4/4 smoke updates. 선택용 seed 없이 두 반복을 모두 포함했다. 추가 LR/rank/loss/seed 탐색과 이전 TEST 재평가는 없었다. 평균은 seed별 점수 평균이며 예측 ensemble이 아니다.

## 설계와 범위

Solar-energy 공식10분 간격137열 원자료를 겹치지 않는6행 평균으로 시간 단위로 집계했다. TRAIN 유한·양의 분산 조건을 통과한 열을 고정SHA 순서로 정렬한 첫8열({', '.join(d['ids'])})이다. 예측 성능이나 0비율로 열을 고르지 않았다. 실제 달력·시간대는 만들지 않았다. 전체{len(d['values']):,}시간, context512/horizon128, 시간순 TRAIN60/CAL10/VAL10/TEST20%, 일24슬롯 간격이다. CAL/VAL/TEST 원점 수는 각각 {len(d['origins']['CALIBRATION'])}/{len(d['origins']['VALIDATION'])}/{len(d['origins']['TEST'])}개다. 모든 target window는 해당 역할 구간 안에 있으며 표준편차는 TRAIN만 사용한다.

저장소 노출 감사: {audit['repository_exposure'].get('conclusion', 'DATA_AUDIT.json 참조')} 공개 benchmark의 upstream pretraining 노출을 배제한 자료는 아니다. 원자료·집계·결측·선택·hash는 [DATA_AUDIT.json](DATA_AUDIT.json)에 있다.

동일 pretrained Chronos-Bolt-small을 동결하고 q/v rank8 LoRA({fits[0]['trainable_parameters']:,}개 파라미터), FP32, LR1e-4, batch4,512updates를 사용했다. SHARED는 두 블록에 같은 LoRA를 적용하며 .5 첫64 pinball + .5 후반 혼합분포 CRPS로 학습한다. CONTINUATION은 첫64에서 LoRA를 끄고 F0의9개 경로를 고정한 뒤 다음64에서만 LoRA를 켠다. 후반 loss 계수 .5는 유지한다.

이 비교는 **gating + F0 branch context 고정 + 첫64 loss 제거**의 묶음 효과다. 각 요소의 독립 효과를 식별하지 않는다. 둘째 호출에서는 관측 토큰에도 LoRA가 작동하므로 생성 토큰만 수정하는 방법이 아니다. 첫64 보존은 검증할 수 있지만 후반 성능 개선을 보장하지 않는다. Aurora의 기존 [from_second LoRA](https://microsoft.github.io/aurora/api.html)가 있어 gating 자체는 새 원리라고 주장하지 않는다.

공식 Bolt는 중앙값-only가 아니다. raw 분위수-labelled9경로,9×9후반 예측,공식9분위수 축약을 동일하게 사용했다. 생성 context는 detach하고 미래 정답은 넣지 않는다. CRPS는81개 동일질량 지지점의 정확한 경험분포 점수이며 연속분포의 정확한 CRPS 또는 IID 표본 보정을 주장하지 않는다.

각 모델은 INIT/128/256/512에서 VAL 후반 점수가 최소인 checkpoint를 선택했다. 선택과 CAL-only35점 위치·폭 보정은 TEST 예측 전에 봉인했다. 보정은 출력에만 적용하며 context에 되먹이지 않는다. Chronos-2 direct는 다른 크기·사전학습의 실용 참고 모델이며 같은 계산량의 인과 대조가 아니다.

## 학습 없는 감쇠 진단

이전 ETTh2의 선택 MIXTURE 가중치에 W=W0+lambda*DeltaW를 적용하고 lambda{{0,.25,.5,1}}를 기존 VALIDATION에서만 비교했다. LoRA B만 감쇠하며 출력 보간이 아니다. 이 진단은 노출된 개발 자료에 대한 것이고 독립 확인 결과가 아니다. Solar에서는 진단 결과와 관계없이 두 arm 모두 lambda1을 사용했다.

```text
{attenuation.to_string(index=False,float_format=lambda x:f'{x:.7f}')}
```

[ATTENUATION_DIAGNOSTIC.json](ATTENUATION_DIAGNOSTIC.json)에 endpoint parity와 선택·추론 검산이 있다. VAL 평균으로 선택한 lambda는 {attenuation_receipt['selected_lambda']:g}다. 두 seed 모두 lambda1의 점수가 가장 낮았으므로, 현재 진단은 가중치 변경을 줄이는 것이 해결책이라는 설명을 지지하지 않는다. 이 값으로 과적합 원인이 확인됐다고 해석하지 않는다. 진단 optimizer updates는0이다.

## 실용 주 비교와 구조 비교

양수 개선율은 CONTINUATION이 기준선보다 좋다는 뜻이다. 주 지표는 공식9분위수 축약·공통정렬·동일CAL 후반65–128의 TRAIN표준편차 정규화 twice-pinball이다. 임의의1% 또는 모든 자료 승리 조건은 사용하지 않았다.

```text
{practical_seeds[['seed','baseline_score','candidate_score','relative_percent','ci95_low','ci95_high']].to_string(index=False,float_format=lambda x:f'{x:.7f}')}
```

F0+CAL 대비 평균 상대 개선 **{primary['relative_percent']:+.4f}%**, 절대 점수 차이 {primary['effect']:+.7f},95%구간 [{primary['ci95_low']:+.7f}, {primary['ci95_high']:+.7f}]. CI는 상대%가 아니라 기준선−후보의 절대 점수 차이다. 두 seed의 방향 기준과 CI를 구분한다.

SHARED 대비 구조 비교: 공통정렬 **{structural['ordered']['relative_percent']:+.4f}%**, 동일CAL **{structural['affine']['relative_percent']:+.4f}%**. 보정 후 절대 차이95%구간 [{structural['affine']['ci95_low']:+.7f}, {structural['affine']['ci95_high']:+.7f}]. SHARED 자체의 F0+CAL 대비 개선은 {shared['relative_percent']:+.4f}%다. 두 학습법 모두 좋아지면 일반 적응의 이득과 gating의 추가 이득을 구분해야 한다.

```text
{table.to_string(index=False,float_format=lambda x:f'{x:.6f}')}
```

checkpoint 선택 없는 fixed512 CONTINUATION의 보정 전 F0 대비 후반 개선은 {fixed['relative_percent']:+.4f}%이며, 이를 주 판정으로 바꾸지 않는다. 첫64 raw 예측과 선택 모델의 CAL 출력은 F0와 완전히 일치했다(최대오차 {verification['test_first64_raw_max_error']:.1f}). 이 보존 성질이 후반의 추가 예측 가치를 자동 입증하지는 않는다.

![seed·전체열·동일보정 결과](RESULTS.png)

그림의 seed 점은 두 독립 초기화·학습순서 결과이며 오차막대가 아니다. 전 열을 포함하고 사후 유리한 열만 선택하지 않았다. [scores.csv](scores.csv), [seed_effects.csv](seed_effects.csv), [effects.csv](effects.csv), [reference_effects.csv](reference_effects.csv)에 첫64/후반64/전체128, raw/ordered/affine와 fixed512 결과를 남겼다. [channel](channel_scores.csv), [channel 효과](channel_effects.csv), [origin](origin_scores.csv), [lead](lead_scores.csv), [경험분포 CRPS](distribution_scores.csv)를 함께 제공한다.

7개 일원점 moving-block bootstrap2000회,seed92249로 모든 열·seed를 함께 표집했다. 중첩 horizon을 독립 표본으로 세지 않았다. 구간은 이 자료·두seed에 조건부인 시간 원점 불확실성이며 데이터셋 모집단이나 충분한 seed 모집단에 대한 신뢰구간이 아니다. 작은 Solar 표본과 한정된 기간의 결과다.

## 자원과 검산

```text
{training.to_string(index=False,float_format=lambda x:f'{x:.3f}')}
```

학습 시간은 forward/backward/update 합계이고 검증·checkpoint 쓰기·개발은 제외한다. main conditional batch forwards {ledger['main_conditional_batch_forwards']:,}회, backwards {ledger['main_batch_backwards']:,}회다. SHARED는update당2backwards,CONTINUATION은1backward이므로 같은FLOP 학습이라 부르지 않는다. 두 방법 모두 같은9경로·batch4·update수다. [resources.csv](resources.csv)에 추론 시간·allocated/reserved memory도 저장했다.

[VERIFICATION.json](VERIFICATION.json): 독립 절대오차식 pinball 최대차 {verification['scalar_pinball_max_error']:.3g},명시적81×81 CRPS 최대차 {verification['scalar_81atom_crps_max_error']:.3g},VAL checkpoint16개 재검산 최대차 {verification['validation_score_max_error']:.3g},선택4/4 일치,CAL {verification['calibration_points_verified']}점 최대차 {verification['calibration_grid_max_error']:.3g}. CAL/TEST 첫64 F0완전일치·보정계수 동일,데이터/예측/checkpoint/seal SHA,optimizer1–512 순서,paired초기화·배치도 확인했다. 실제모델 parity/유한gradient/backbone동결은 [PREFLIGHT.json](PREFLIGHT.json)과 SMOKE기록,CPU검사는 [CPU_TESTS.json](CPU_TESTS.json)에 있다. [validation_verification.csv](validation_verification.csv)에 독립 선택 점수를 남겼다.

원자료·model weights·checkpoint·prediction arrays는 ignored 로컬 cache에 있고 GitHub에는 검산 manifest만 제공한다. GitHub만으로 모든 수치를 즉시 재생할 수 있다는 뜻은 아니다. 실행 진입점은 `experiments/{NAME}/preflight.py`, `runner.py all`, `finalize.py`다. 완료경로를 재학습하지 않으며 재실행은 별도 예산 작업이다.

## 종료

현재 결과는 **{decision}**이다. 신규성·논문PASS를 선언하지 않는다. 일반 one-block/median LoRA 대비 독립 이득을 식별한 실험도 아니다. 구현·자료·자원 문제와 과학적 성능 결과를 구분한다. 다른 LR/rank/loss/자료나 deferred head를 자동 실행하지 않고 종료했다. [PROTOCOL.md](../../experiments/{NAME}/PROTOCOL.md),[FINAL_DECISION.md](FINAL_DECISION.md).
'''
    (RESULTS / 'REPORT_KO.md').write_text(text, encoding='utf-8')
    (RESULTS / 'FINAL_DECISION.md').write_text(f'''# 최종 판단

EXECUTION: COMPLETE
SCIENTIFIC_DECISION: {decision}
PRIMARY: CONTINUATION affine versus F0_NATIVE affine, block65–128
PRIMARY_RELATIVE_GAIN_PERCENT: {primary['relative_percent']:+.7f}
PRIMARY_ABSOLUTE_EFFECT_CI95: [{primary['ci95_low']:+.7f}, {primary['ci95_high']:+.7f}]
PRACTICAL_POSITIVE_SEEDS: {int((practical_seeds.effect > 0).sum())} / 2
STRUCTURAL_AFFINE_GAIN_PERCENT: {structural['affine']['relative_percent']:+.7f}
FITS: 4 / 4
MAIN_UPDATES: 2048 / 2048
SMOKE_UPDATES: 4 / 4
ATTENUATION_OPTIMIZER_UPDATES: 0
NOVELTY: NOT_ESTABLISHED; Aurora already supports from_second LoRA
PAPER_PASS: NOT_CLAIMED
AUTOMATIC_FOLLOWUP: NONE

{direction}

Solar의 사전고정8열·512→128·두seed에 한정한다. 단순 보정 F0 대비 실용 비교와 SHARED 대비 구조 비교는 별도다. gating+고정F0경로+첫loss제거의 묶음 효과이며 각 요소의 독립 인과효과나 새로운 원리의 증명이 아니다. 첫64 보존이 후반 개선을 보장하지 않는다. 기존 ETTh2 감쇠 진단은 VAL만 사용하며 이전TEST는 재평가하지 않았다. 새 설정·후보를 자동 실행하지 않고 종료했다.
''', encoding='utf-8')


if __name__ == '__main__':
    main()
