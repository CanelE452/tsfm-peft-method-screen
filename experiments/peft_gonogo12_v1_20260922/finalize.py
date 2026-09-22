"""CPU-only scoring of the fixed, sealed experiment; never trains or selects."""
import csv
import math
import numpy as np
import pandas as pd
from common import *
from evaluation import readpred,predpath,adjusted
from metrics import point_loss
from runner import check_seal,H1_FINAL,H2_METHODS


def write_csv(name,rows):
    rows=list(rows)
    with (RESULTS/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def collect():
    rows=[];invariants=[];references=read(RESULTS/'REFERENCE_SELECTION.json')
    for candidate,arms in [('h1',H1_FINAL),('h2',H2_METHODS)]:
        selection=read(RESULTS/f'SELECTION_{candidate.upper()}.json')
        cals=read(RESULTS/f'CALIBRATION_{candidate.upper()}.json')
        ref=references.get('candidates',{}).get(candidate,{})
        methods=arms+(['CHRONOS2_DIRECT'] if ref.get('status')=='AVAILABLE' else [])
        conditions=['CLEAN'] if candidate=='h1' else ['BLOCK48','IID48','ALIGNED48_LAST','CLEAN']
        for arm in methods:
            for seed in ([0] if arm=='CHRONOS2_DIRECT' else CONFIG[candidate]['seeds']):
                cal=ref['eval_calibration'] if seed==0 else cals[str(seed)][arm]
                chosen=ref['mode'] if seed==0 else selection['details']['U_SHRINK' if arm=='PERMUTED_G' else arm]['mode']
                for condition in conditions:
                    part='test' if candidate=='h1' else 'test_'+condition
                    pred=readpred(predpath(candidate,'eval',seed,arm,2,part))
                    assert np.isfinite(pred['q']).all() and np.isfinite(pred['y']).all()
                    if candidate=='h2' and arm in CONFIG['h2']['arms'] and arm!='QV_LORA' and (condition=='CLEAN' or (arm=='SET_BIAS' and condition=='ALIGNED48_LAST')):
                        native=readpred(predpath(candidate,'eval',seed,'F0_NATIVE',2,part))
                        np.testing.assert_array_equal(pred['pairs'],native['pairs'])
                        scaled=np.max(np.abs(pred['q']-native['q'])/pred['sigma'][:,None,None])
                        assert scaled<=1e-5
                        invariants.append(dict(seed=seed,arm=arm,condition=condition,max_scaled_difference=float(scaled)))
                    for mode in ['RAW','CAL']:
                        q=adjusted(pred,cal,mode);ordered=np.sort(q,axis=-1);y=pred['y'].astype(float)
                        errors=point_loss(q,y,pred['sigma'])
                        mae=np.abs(y-ordered[...,4]).mean(axis=1)
                        coverage=((y>=ordered[...,0])&(y<=ordered[...,-1])).mean(axis=1)
                        width=(ordered[...,-1]-ordered[...,0]).mean(axis=1)
                        crossing=(np.diff(q,axis=-1)<0).mean(axis=(1,2))
                        # Independent scalar oracle, including quantile ordering and sigma.
                        oracle=sum(2*max(t*(float(y[0,h])-float(ordered[0,h,j])),(t-1)*(float(y[0,h])-float(ordered[0,h,j])))
                                   for h in range(24) for j,t in enumerate(np.arange(1,10)/10))/(24*9*float(pred['sigma'][0]))
                        assert abs(oracle-errors[0])<1e-12
                        for i,(sid,origin) in enumerate(pred['pairs']):
                            rows.append(dict(candidate=candidate,arm=arm,seed=seed,condition=condition,mode=mode,selected=mode==chosen,
                                 series=str(sid),origin=int(origin),mask_repeat=int(pred['repeats'][i]),weight=float(pred['weights'][i]),
                                 scaled_pinball=float(errors[i]),raw_mae=float(mae[i]),scaled_mae=float(mae[i]/pred['sigma'][i]),
                                 coverage80=float(coverage[i]),raw_width80=float(width[i]),scaled_width80=float(width[i]/pred['sigma'][i]),crossing=float(crossing[i])))
    write_csv('origin_mask_scores.csv',rows)
    save(RESULTS/'STRUCTURAL_INVARIANTS.json',dict(status='PASS',checks=invariants,scope='raw forecasts; affine calibration may change clean forecasts'))
    return pd.DataFrame(rows)


METRICS=['scaled_pinball','raw_mae','scaled_mae','coverage80','raw_width80','scaled_width80','crossing']


def aggregate(frame,keys):
    return frame.groupby(keys,sort=False)[METRICS].mean().reset_index()


def cube(frame,arm,seeds):
    parts=[]
    for seed in seeds:
        subset=frame[(frame.arm==arm)&(frame.seed==(0 if arm=='CHRONOS2_DIRECT' else seed))]
        table=subset.groupby(['series','origin']).scaled_pinball.mean().unstack('origin').sort_index().sort_index(axis=1)
        assert table.shape==(8,14) and table.notna().all().all()
        parts.append(table.to_numpy())
    return np.stack(parts)


def paired_effect(candidate,baseline):
    return 100*(baseline-candidate)/baseline


def compare(a,b):
    per_seed=paired_effect(a.mean(axis=(1,2)),b.mean(axis=(1,2)))
    rng=np.random.default_rng(92270);boot=[]
    for _ in range(2000):
        starts=rng.integers(0,12,size=5)
        dates=(starts[:,None]+np.arange(3)).ravel()[:14]
        boot.append(float(paired_effect(a[:,:,dates].mean(axis=(1,2)),b[:,:,dates].mean(axis=(1,2))).mean()))
    return dict(seed_effects=per_seed.tolist(),mean_effect_pct=float(per_seed.mean()),
                winning_clients=int((a.mean(axis=(0,2))<b.mean(axis=(0,2))).sum()),
                conditional_ci95=np.quantile(boot,[.025,.975]).tolist())


def decide(frame):
    decisions={};seedrows=[];seriesrows=[];effectrows=[]
    for candidate in ['h1','h2']:
        s=read(RESULTS/f'SELECTION_{candidate.upper()}.json');seeds=CONFIG[candidate]['seeds']
        condition='CLEAN' if candidate=='h1' else 'BLOCK48'
        f=frame[(frame.candidate==candidate)&frame.selected&(frame.condition==condition)]
        proposal=s['proposal'];a=cube(f,proposal,seeds);comparisons={}
        for arm in f.arm.unique():
            if arm==proposal:continue
            b=cube(f,arm,seeds);comparison=compare(a,b);comparisons[arm]=comparison
            effectrows.append(dict(candidate=candidate,proposal=proposal,baseline=arm,mean_effect_pct=comparison['mean_effect_pct'],
                 ci_low=comparison['conditional_ci95'][0],ci_high=comparison['conditional_ci95'][1],winning_clients=comparison['winning_clients']))
            for j,seed in enumerate(seeds):
                seedrows.append(dict(candidate=candidate,proposal=proposal,baseline=arm,seed=seed,
                         proposal_score=float(a[j].mean()),baseline_score=float(b[j].mean()),effect_pct=comparison['seed_effects'][j]))
            for j,sid in enumerate(sorted(f.series.unique())):
                seriesrows.append(dict(candidate=candidate,proposal=proposal,baseline=arm,series=sid,
                       proposal_score=float(a[:,j].mean()),baseline_score=float(b[:,j].mean()),
                       effect_pct=float(paired_effect(a[:,j].mean(),b[:,j].mean()))))
        required=['FIXED_SHRINK','UNSHRUNK'] if candidate=='h1' else ['KEY_BIAS','CENTROID_BIAS','GENERIC_BIAS']
        primary=comparisons[s['baseline']]
        mechanisms=all(comparisons[arm]['mean_effect_pct']>=.5 for arm in required)
        if candidate=='h1':mechanisms &= comparisons['PERMUTED_G']['mean_effect_pct']>0
        bolt=[arm for arm in comparisons if arm not in ['CHRONOS2_DIRECT','PERMUTED_G']]
        # A baseline is >1% better when its reduction relative to the proposal is >1%.
        dominates=[arm for arm in bolt if float(paired_effect(cube(f,arm,seeds).mean(axis=(1,2)),a.mean(axis=(1,2))).mean())>1]
        go=all(v>0 for v in primary['seed_effects']) and primary['mean_effect_pct']>=1 and primary['winning_clients']>=5 and mechanisms and not dominates
        nonpositive=[arm for arm in dict.fromkeys([s['baseline']]+required+(['PERMUTED_G'] if candidate=='h1' else [])) if all(v<=0 for v in comparisons[arm]['seed_effects'])]
        decision='HOLD_BUDGET_LIMITED' if s['undertrained'] else ('GO_SCREEN' if go else ('NO_GO_CURRENT_RECIPE' if nonpositive else 'HOLD_INCONCLUSIVE'))
        ref=comparisons.get('CHRONOS2_DIRECT')
        practical='PRACTICAL_REFERENCE_GAIN' if ref and ref['mean_effect_pct']>=1 and all(v>0 for v in ref['seed_effects']) else ('MECHANISM_SIGNAL_ONLY' if ref else 'PRACTICAL_UNVERIFIED')
        safety=None
        if candidate=='h1':
            shared=cube(f,'SHARED',seeds)
            safety={arm:float((cube(f,arm,seeds).mean(axis=(0,2))>shared.mean(axis=(0,2))).mean()) for arm in ['U_SHRINK','UNSHRUNK','FIXED_SHRINK','LOCAL_LORA','RIDGE']}
        decisions[candidate]=dict(decision=decision,proposal=proposal,baseline=s['baseline'],primary=primary,comparisons=comparisons,
              mechanisms_pass=bool(mechanisms),dominating_bolt_baselines=dominates,nonpositive_controls=nonpositive,undertrained=s['undertrained'],
              practical_comparison=practical,go_label=practical if decision=='GO_SCREEN' else None,h1_fraction_worse_than_shared=safety,
              scope='Previously exposed Electricity; 8 evaluation series, 14 dates, 2 training seeds; no novelty or paper PASS claim')
    write_csv('effect_summary.csv',effectrows);write_csv('seed_effects.csv',seedrows);write_csv('series_effects.csv',seriesrows)
    save(RESULTS/'DECISIONS.json',decisions)
    return decisions


def resources():
    rows=[]
    for p in sorted((RESULTS/'fits').glob('*.json')):
        r=read(p);rows.append({k:r[k] for k in ['key','candidate','phase','seed','steps','seconds','trainable_parameters','peak_allocated_bytes','peak_reserved_bytes']})
    write_csv('fit_resources.csv',rows)
    grouped={}
    for row in rows:
        if row['phase']!='main':continue
        key=row['key']
        arm=next(a for a in ['SHARED','BLOCK_A','BLOCK_B','RIDGE','LOCAL_LORA','QV_LORA','SET_BIAS','CENTROID_BIAS','KEY_BIAS','GENERIC_BIAS'] if a in key)
        grouped.setdefault((row['candidate'],arm),[]).append(row)
    write_csv('resource_summary.csv',[dict(candidate=c,arm=a,fits=len(v),updates=sum(r['steps'] for r in v),
              fit_seconds=sum(r['seconds'] for r in v),median_seconds_per_update=float(np.median([r['seconds']/r['steps'] for r in v])),
              trainable_parameters=v[0]['trainable_parameters']) for (c,a),v in grouped.items()])
    phase=read(RESULTS/'GPU_PHASE_TIME.json')
    write_csv('phase_resources.csv',[dict(candidate=c,**row) for c,v in phase.items() for row in v])
    return rows,phase


def verify(frame):
    check_seal();start=read(RESULTS/'MAIN_START.json')
    assert start['design_sha']==sha(DESIGN_DIR/'DESIGN.json')
    for n,h in start['source_hashes'].items():assert sha(EXP/n)==h
    shared_parity=[]
    for seed in CONFIG['h1']['seeds']:
        for part in ['cal','validation']:
            a=readpred(predpath('h1','dev',seed,'SHARED',2,part))
            b=readpred(predpath('h1','dev',seed,'UNSHRUNK',0,part))
            np.testing.assert_array_equal(a['pairs'],b['pairs'])
            delta=float(np.max(np.abs(a['q']-b['q'])/a['sigma'][:,None,None]))
            np.testing.assert_allclose(a['q']/a['sigma'][:,None,None],b['q']/b['sigma'][:,None,None],atol=1e-5,rtol=1e-5)
            shared_parity.append(dict(seed=seed,part=part,max_scaled_difference=delta))
    ledger=read(RESULTS/'OPTIMIZER_LEDGER.json');assert len(ledger['fits'])==117
    for c in ['h1','h2']:
        assert ledger['counts'][c]==dict(main_updates=CONFIG[c]['main_updates_cap'],smoke_updates=CONFIG[c]['smoke_updates_cap'],main_fits=CONFIG[c]['fit_cap'])
    for key,r in ledger['fits'].items():
        assert r['status']=='COMPLETE' and r['completed']==r['reserved']==r['planned_steps']
        receipt=read(RESULTS/'fits'/f'{key}.json');assert receipt['frozen_unchanged']
        with np.load(CACHE/'fits'/key/'schedule.npz') as z:pairs=z['pairs']
        assert sha(CACHE/'fits'/key/'schedule.npz')==receipt['schedule_sha256']
        if r['phase']=='main':
            origins=pairs[...,1].astype(int);ids=set(pairs[...,0].ravel())
            if '_SHARED_' in key or key.startswith('h2_'):
                assert ids==set(map(str,CONFIG['data']['donor_ids'])) and origins.max()+24<=15576
            else:
                group='dev' if '_dev_' in key else 'eval';lo,hi=CONFIG['data']['ranges'][group+'_adapt'];mid=lo+384
                assert origins.min()-192>=lo and origins.max()+24<=hi
                assert np.all((origins+24<=mid)|(origins-192>=mid))
    assert frame.origin.min()==16104 and frame.origin.max()+24==16440
    assert set(frame.series)==set(map(str,CONFIG['data']['evaluation_ids']))
    assert not frame[METRICS].isna().any().any()
    # All same-seed H2 arms see byte-identical pair schedules.
    for seed in CONFIG['h2']['seeds']:
        schedules=[]
        for arm in CONFIG['h2']['arms']:
            with np.load(CACHE/'fits'/f'h2_{arm}_s{seed}'/'schedule.npz') as z:schedules.append(z['pairs'])
        for a in schedules[1:]:np.testing.assert_array_equal(a,schedules[0])
    events=[json.loads(line) for line in (RESULTS/'events.jsonl').read_text().splitlines()]
    assert sum(e['kind']=='all_selections_sealed' for e in events)==1
    result=dict(status='PASS',main_fits=108,main_updates=7680,smoke_updates=18,completed_fit_receipts=117,
          primary_rows=len(frame),scalar_metric_oracle=True,all_prediction_hashes=True,all_frozen_weights_unchanged=True,
          source_design_and_selection_seals_unchanged=True,schedule_and_chronology_checks=True,
          actual_shared_basis_zero_coefficient_parity=shared_parity,
          structural_invariants=read(RESULTS/'STRUCTURAL_INVARIANTS.json')['status'],automatic_followup=False,
          limitation='Independent scalar metric and schedule checks within the same codebase; not external replication. Raw/model/prediction caches are local.')
    save(RESULTS/'VERIFICATION.json',result)


def figures(frame,decisions,resource_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,(c,d) in zip(axes,decisions.items()):
        comps=d['comparisons'];names=list(comps);y=np.arange(len(names))
        for j in range(2):ax.scatter([comps[n]['seed_effects'][j] for n in names],y+(-.1 if j==0 else .1),label=f'Seed {j+1}',s=32)
        ax.axvline(0,color='black',lw=1);ax.axvline(1,color='gray',ls=':',lw=1)
        ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlabel('Candidate improvement over control (%)')
        ax.set_title(f'{c.upper()}: {d["proposal"]}\n{d["decision"]}');ax.legend();ax.grid(axis='x',alpha=.2)
    fig.savefig(RESULTS/'seed_effects.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,c in zip(axes,['h1','h2']):
        s=read(RESULTS/f'SELECTION_{c.upper()}.json')
        for arm in dict.fromkeys([s['proposal'],s['learned_baseline'],s['baseline']]):
            ax.plot([0,.5,1],[s['curves'][arm][str(j)] for j in range(3)],marker='o',label=arm)
        ax.set_title(c.upper()+' fixed-checkpoint DEV diagnostics');ax.set_xlabel('Fraction of fixed update budget');ax.set_ylabel('Scaled twice-pinball (lower better)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(RESULTS/'development_curves.png',dpi=180);plt.close(fig)
    f=frame[(frame.candidate=='h2')&frame.selected];v=f.groupby(['arm','condition']).scaled_pinball.mean().unstack('condition')
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True);im=ax.imshow(v.to_numpy(),cmap='viridis')
    ax.set_xticks(range(len(v.columns)),v.columns);ax.set_yticks(range(len(v.index)),v.index)
    for i in range(len(v)):
        for j in range(len(v.columns)):ax.text(j,i,f'{v.iloc[i,j]:.4f}',ha='center',va='center',color='white' if v.iloc[i,j]<(v.max().max()+v.min().min())/2 else 'black')
    ax.set_title('H2 selected-mode score; mask families kept separate');fig.colorbar(im,ax=ax,label='Scaled twice-pinball')
    fig.savefig(RESULTS/'mask_conditions.png',dpi=180);plt.close(fig)


def report(decisions,phase):
    lines=['# 두 PEFT 가설의 고정 GO/NO-GO 실험','', '[확인] 학습·예측·검산을 완료했다. 기존 실험 결과와 합치지 않은 두 후보의 제한적 개발 screen이다.','']
    for c,d in decisions.items():
        p=d['primary'];lines += [f'## {c.upper()} — {d["decision"]}','',
          f'후보 `{d["proposal"]}`, DEV에서 고정한 대조 `{d["baseline"]}`. TEST 개선율은 seed별 **{p["seed_effects"][0]:+.3f}% / {p["seed_effects"][1]:+.3f}%**, 평균 **{p["mean_effect_pct"]:+.3f}%**다. 8계열 중 {p["winning_clients"]}개에서 평균 점수가 좋았다.',
          f'날짜 block 조건부 95% CI는 [{p["conditional_ci95"][0]:+.3f}, {p["conditional_ci95"][1]:+.3f}]%다. 이 구간은 고정 계열과 두 seed에 조건부이며 seed 모집단 신뢰구간이 아니다. CI의 0 포함만으로 기각하지 않았다.','',
          f'메커니즘 조건 통과: {d["mechanisms_pass"]}. 두 seed 모두 비양수인 필수 대조: {", ".join(d["nonpositive_controls"]) or "없음"}. 1% 초과 우세한 다른 Bolt 대조: {", ".join(d["dominating_bolt_baselines"]) or "없음"}.',
          f'사전고정 UNDERTRAINED flag: {d["undertrained"]}. 이 flag가 없다고 수렴을 증명하지는 않는다.','']
        for arm,v in d['comparisons'].items():lines.append(f'- {arm} 대비 평균 {v["mean_effect_pct"]:+.3f}% (seed {v["seed_effects"][0]:+.3f}% / {v["seed_effects"][1]:+.3f}%).')
        lines+=['',f'Chronos-2 비교 분류: `{d["practical_comparison"]}`. GO가 아니면 이 분류만으로 후보 채택이나 실용 우위를 주장하지 않는다.','']
        if c=='h1':
            lines+=['H1은 shared LoRA를 고정 SVD 기저로 분해한 뒤 계열당 288개 계수만 개인화했다. 두 block의 차이를 불안정성 proxy로 사용했다. 정확한 posterior uncertainty 추정이 아니다.',
                    '공유 모델보다 나빠진 계열 비율: '+', '.join(f'{a} {v:.1%}' for a,v in d['h1_fraction_worse_than_shared'].items())+'.','']
            gates=read(RESULTS/'H1_EVAL_VARIANCE.json')
            for seed,values in gates.items():
                gs=[v['g'] for v in values.values()]
                lines.append(f'- seed {seed}: EVAL gate 범위 {min(gs):.4f}–{max(gs):.4f}, 평균 {np.mean(gs):.4f}. g의 큰/작은 값만으로 uncertainty가 정확히 추정됐다고 해석하지 않는다.')
            lines.append('')
        else:lines+=['H2는 마지막 encoder attention의 24개 계수로 관측 위치 집합을 반영했다. CLEAN에서 모든 bias arm, patch 정렬 결측에서 SET은 학습 후에도 raw F0 예측을 보존했다. 보정 모드가 다르면 최종 점수는 다를 수 있다. BLOCK48이 primary이며 IID48/CLEAN/ALIGNED는 별도로 보고한다.','']
    lines+=['## 그림','', '![Seed effects](seed_effects.png)','', '![Fixed DEV curves](development_curves.png)','', '![Mask conditions](mask_conditions.png)','',
      '## 실행·선택·비용','',
      '총 108 main fits / 7,680 main optimizer updates, 별도 18 smoke updates를 사용했다. H1은 98 fits / 5,120 updates, H2는 10 fits / 2,560 updates다. H1의 두 block 불안정성 추정 비용을 숨기지 않고 모두 포함했다. 선택 전용 seed는 없으며 지정된 두 seed만 EVAL 평균에 넣었다.',
      'checkpoint는 고정 마지막 단계다. LR·rank·mask family 추가 탐색은 없었다. raw/CAL, H1 고정 shrink 계수, DEV baseline과 EVAL의 CAL 계수를 모두 봉인한 뒤 TEST 예측을 만들었다. PERMUTED_G는 U와 같은 raw/CAL 선택을 사용했다.',
      'GPU 단계 장부: '+', '.join(f'{c.upper()} {sum(r["seconds"] for r in v)/60:.2f}분' for c,v in phase.items())+'. 초기 P0·환경 검사 시간은 별도 overhead이며 이 수치의 학습 단계와 혼동하지 않는다.',
      'fit 자원표의 GPU peak는 동시에 상주한 모델을 포함하는 process 관측값이다. 공정한 단독 모델 latency 또는 메모리 우위로 해석하지 않는다. 288/24개 계수 감소와 실제 시스템 비용은 구분한다.',
      'q/v LoRA는 294,912개, H1 개인화는 288개, H2 bias는 24개의 학습 파라미터를 사용한다. H1 배포는 pretrained backbone 외에 공유 SVD 기저가 필요하므로 개인화 계수 수를 전체 저장량으로 표현하지 않는다. [arm별 비용 합계](resource_summary.csv)에 공유 학습과 BLOCK_A/B 추정 비용을 구분했다.',
      '', '## 검산과 적용 범위','',
      '[검산](VERIFICATION.json), [고정 선택](ALL_SELECTIONS_SEALED.json), [결정 상세](DECISIONS.json), [P0 검사](P0_COMPLETE.json), [실행 중 전처리 구현 사건](IMPLEMENTATION_INCIDENT.json)을 함께 남겼다. 무효인 선행 P0는 원점/반복 설정 오류로 폐기했으며 학습 0회였다. 올바른 P0와 smoke 후 본학습을 시작했다. 구현 문제를 성능 실패로 세지 않았다.',
      '이전 연구에 노출된 Electricity 한 source, EVAL 8계열 ×14일, seed 두 개의 개발 screen이다. 가려진 context truth는 모델·보간·normalization에 쓰지 않았다. 새 source 검증, 자연 결측 일반화, 신규성, 논문 PASS는 확인하지 않았다. 부정결과는 현재 고정 recipe의 판단이며 rollout PEFT나 TSFM PEFT 전체의 반증이 아니다.',
      '공식 Chronos-2는 native direct 24-step API를 사용하고 cross-series learning을 끈 별도 참조다. 본 실험은 rollout 비교가 아니며 현재 Chronos 공식 장기 branching 방식에 관한 새 주장을 하지 않는다.',
      '', '## 자료','',
      '- [모든 origin·mask raw/CAL 점수](origin_mask_scores.csv), [계열 점수](series_scores.csv), [seed 점수](seed_scores.csv)',
      '- [seed 효과](seed_effects.csv), [계열 효과](series_effects.csv), [효과·CI 요약](effect_summary.csv)',
      '- [fit 자원](fit_resources.csv), [GPU 단계 자원](phase_resources.csv), [학습 장부](OPTIMIZER_LEDGER.json)',
      '- [파일 provenance](MANIFEST.json), [native 구조 보존](STRUCTURAL_INVARIANTS.json)',
      '', 'raw data, model weights, optimizer/checkpoint, prediction arrays는 Git 제외 로컬 cache에 보존한다. GitHub의 코드·점수·hash만으로 원본 cache 없이 모든 수치를 재생할 수 있다고 주장하지 않는다. 두 후보 판정 후 자동 후속 실험 없이 종료한다.']
    (RESULTS/'REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (RESULTS/'FINAL_DECISION.md').write_text('# FINAL DECISION\n\n'+'\n'.join(f'- {c.upper()}: **{d["decision"]}** — {d["proposal"]}; primary 평균 {d["primary"]["mean_effect_pct"]:+.3f}%.' for c,d in decisions.items())+'\n\n[전체 한국어 보고서](REPORT_KO.md). 현재 고정 recipe의 자원 배분 판단이며 신규성·논문 PASS·PEFT 전체 반증을 뜻하지 않는다. 추가 학습/설정 탐색/자동 후속 없음.\n',encoding='utf-8')


def main():
    assert read(RESULTS/'RUN_COMPLETE.json')['status']=='COMPLETE';check_seal()
    frame=collect()
    keys=['candidate','arm','seed','condition','mode','selected']
    series=aggregate(frame,keys+['series']);series.to_csv(RESULTS/'series_scores.csv',index=False)
    aggregate(series,keys).to_csv(RESULTS/'seed_scores.csv',index=False)
    decisions=decide(frame);rows,phase=resources();verify(frame);figures(frame,decisions,rows);report(decisions,phase)
    manifest=dict(source_hashes=source_hashes(),design_sha256=sha(DESIGN_DIR/'DESIGN.json'),
        results={str(p.relative_to(RESULTS)):sha(p) for p in sorted(RESULTS.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'},
        local_caches={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(CACHE.rglob('*')) if p.is_file() and p.suffix in ['.npz','.pt']},
        upstream=read(RESULTS/'ENVIRONMENT.json') if (RESULTS/'ENVIRONMENT.json').exists() else 'See preflight provenance records',automatic_followup=False)
    save(RESULTS/'MANIFEST.json',manifest)
    print(json.dumps(decisions,indent=2))


if __name__=='__main__':main()
