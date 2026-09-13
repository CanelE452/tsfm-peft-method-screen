"""Render completed train/V diagnostics without model fitting or E access."""
import csv
import json
import os
from pathlib import Path
from statistics import mean
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tsfm_peft_screen.metrics import score
from tsfm_peft_screen.reproducibility import sha

OUT=ROOT/'results/reassessment_diagnostics_20260914'
read=lambda p:json.loads(p.read_text())
verification=read(OUT/'verification.json')
assert verification['status']=='VERIFIED_V_DIAGNOSTICS'
assert read(OUT/'queue_status.json')['status']=='COMPLETE'
contract=read(OUT/'contract.json')
comparisons=list(csv.DictReader((OUT/'comparisons.csv').open()))

def percent(a,b):return 100*(b-a)/b
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

selected={t:read(OUT/t/'selection_seal.json')['selections'] for t in ('anchor','dualclock')}
summary=[]
for dataset in ('etth1','traffic'):
    rows=[r for r in selected['anchor'] if r['dataset']==dataset]
    means={a:mean(r['metrics']['scaled_2pinball'] for r in rows if r['arm']==a)
           for a in ('native','native_anchor','raw','raw_anchor')}
    summary.append(dict(dataset=dataset,means=means,
        native_anchor_gain_percent=percent(means['native_anchor'],means['native']),
        raw_anchor_gain_percent=percent(means['raw_anchor'],means['raw']),
        raw_anchor_vs_native_gain_percent=percent(means['raw_anchor'],means['native'])))
write('anchor_summary.json',summary)

long_rows=[]
for seed in (30000,30001):
    means={r['arm']:r['metrics']['scaled_2pinball'] for r in selected['dualclock'] if r['seed']==seed}
    long_rows.append(dict(seed=seed,means=means,
        dualclock_vs_standard_gain_percent=percent(means['dualclock'],means['standard']),
        dualclock_vs_summary_gain_percent=percent(means['dualclock'],means['summary'])))
write('dualclock_summary.json',long_rows)

with np.load(ROOT/'data/processed/m5/fit.npz',allow_pickle=False) as z:
    heavy=z['zero_fraction']>=np.quantile(z['zero_fraction'],.75)
series=[]
for seed in (30000,30001):
    s={r['arm']:r for r in selected['dualclock'] if r['seed']==seed}
    cached={}
    for arm in ('dualclock','summary','standard'):
        path=ROOT/s[arm]['prediction_file'];assert sha(path)==s[arm]['prediction_sha256']
        with np.load(path,allow_pickle=False) as z:
            cached[arm]=np.array([score(z['prediction'][:,i:i+1],z['target'][:,i:i+1],z['scale'][i:i+1])['scaled_2pinball'] for i in range(256)])
        assert abs(cached[arm].mean()-s[arm]['metrics']['scaled_2pinball'])<1e-10
    for baseline in ('standard','summary'):
        effects=cached[baseline]-cached['dualclock']
        series.append(dict(seed=seed,baseline=baseline,mean_loss_difference=float(effects.mean()),
            positive_series_fraction=float((effects>0).mean()),median_loss_difference=float(np.median(effects)),
            trimmed_mean_without_best26=float(np.sort(effects)[:-26].mean()),zero_heavy_mean=float(effects[heavy].mean()),
            effects=effects.tolist(),scope='V-only post-training diagnostic; items are not independent replications'))
write('dualclock_series_diagnostics.json',series)

fits=[];costs=[]
for topic in ('anchor','dualclock'):
    ff=read(OUT/topic/'fits.json');fits.extend(ff)
    gpu=read(OUT/topic/'gpu.json')
    resources=[r for f in ff for r in read(OUT/topic/(f['fit_id']+'_fit.json'))['resources']]
    costs.append(dict(topic=topic,completed_fits=len(ff),training_updates=sum(f['updates'] for f in ff),
        active_seconds=sum(f['active_seconds'] for f in ff),fit_wall_seconds=sum(f['wall_seconds'] for f in ff),
        peak_allocated_mib=max(r['peak_allocated_bytes'] for r in resources)/2**20,
        gpu_samples=len(gpu),external_compute_samples=sum(bool(r['external_pids']) for r in gpu),
        min_observed_free_mib=min(r['free_mib'] for r in gpu)))
write('resource_summary.json',costs)

lines=['# Anchoring / DualClock 재진단 결과', '',
    '**44/44 fits, 46,080 training updates 완료. 이번 결과는 train/V 진단이며 새 E 평가나 논문 PASS가 아니다.**', '',
    f"실행 commit `{contract['execution_commit']}`. 기존 270 fits에 이번44 fits를 추가해 누적314 fits다. Stream 시도는9회(8완료/과거1중단) 그대로다. 별도 GPU smoke14 updates는 fits에 포함하지 않는다.", '',
    f"저장 예측 {verification['prediction_caches_replayed']}개 독립 재계산, 최대 primary 오차 {verification['max_primary_error']:.3g}. 기존 결과 {verification['historical_files_unchanged']}개 파일 불변. 실제 parameter update와 frozen weights 검사를 통과했고 각 fit의 선택 checkpoint를 다시 읽어 V 예측이 정확히 일치함을 확인했다.", '',
    '**Anchoring: 같은 목적함수 내 보존항 효과**', '',
    '| 데이터 | Native | Native+anchor | Raw | Raw+anchor | Native 보존 이득 | Raw 보존 이득 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for r in summary:
    m=r['means'];lines.append(f"| {r['dataset']} | {m['native']:.9f} | {m['native_anchor']:.9f} | {m['raw']:.9f} | {m['raw_anchor']:.9f} | {r['native_anchor_gain_percent']:+.4f}% | {r['raw_anchor_gain_percent']:+.4f}% |")
lines+=['','각 arm의 두 seed V-selected loss 평균이다. 양수 이득은 보존항을 넣은 쪽의 오차 감소다. 목적함수 간 보존항의 상대적 gradient 강도가 같다고 가정하지 않는다. 고정 LR·step의 비교는 comparisons.csv에 모두 포함했다.','','| 데이터 | Seed | Arm | LR | 선택 step | V loss |','|---|---:|---|---:|---:|---:|']
for r in selected['anchor']:
    lines.append(f"| {r['dataset']} | {r['seed']} | {r['arm']} | {r['lr']:g} | {r['step']} | {r['metrics']['scaled_2pinball']:.9f} |")
lines+=['','**DualClock: 추가 예산과 비교군**','','| Seed | Standard | Summary | DualClock | vs Standard | vs Summary |','|---|---:|---:|---:|---:|---:|']
for r in long_rows:
    m=r['means'];lines.append(f"| {r['seed']} | {m['standard']:.9f} | {m['summary']:.9f} | {m['dualclock']:.9f} | {r['dualclock_vs_standard_gain_percent']:+.4f}% | {r['dualclock_vs_summary_gain_percent']:+.4f}% |")
lines+=['','| Seed | Arm | LR | 선택 step | V loss |','|---|---|---:|---:|---:|']
for r in selected['dualclock']:
    lines.append(f"| {r['seed']} | {r['arm']} | {r['lr']:g} | {r['step']} | {r['metrics']['scaled_2pinball']:.9f} |")
lines+=['','| Seed | Arm / 비교 | 구분 | 360 또는 기준 loss | 1440 또는 비교 loss | 개선율 |','|---|---|---|---:|---:|---:|']
for r in comparisons:
    if r['topic']=='dualclock' and 'long_vs_short' in r['contrast']:
        lines.append(f"| {r['seed']} | {r['contrast']} | {r['scope']} | {float(r['reference_loss']):.9f} | {float(r['method_loss']):.9f} | {float(r['improvement_percent']):+.4f}% |")
lines+=['','V-selected long-vs-short 비교는 후보 checkpoint 수가 늘면 최솟값이 나빠질 수 없다는 선택 효과를 포함한다. 고정 LR의360→1440 endpoint 변화, 최종 선택 위치, 비교군과의 간격을 함께 해석해야 한다.', '',
    f"역사적 seed30000의 첫360-step V trajectory 최대 차이: {verification['historical_360_prefix_max_validation_difference']:.3g}. 이 수치는 historical_prefix.json에서 모든 LR·arm·checkpoint별로 확인할 수 있다.",'',
    '**Event branch 진단**','','| Seed | Arm | 개입 | 원래 대비 V loss 변화 |','|---|---|---|---:|']
for r in comparisons:
    if r['scope']=='V_ablation':
        lines.append(f"| {r['seed']} | {r['contrast'].split('_')[0]} | {r['contrast']} | {-float(r['improvement_percent']):+.5f}% |")
lines+=['','양수는 개입으로 손실이 증가했다는 뜻이다. Zero-event는 event embedding만0, rotate는32-item 평가 batch 안에서 event 정보를 회전, drop은 hidden residual adapter 전체 제거다. OOD 개입 가능성이 있으므로 event의 고유 인과 기여나 재학습 대조의 대체물로 해석하지 않는다. 이 결과로 모델을 다시 선택하지 않았다.','','| Seed | 대조 | 이긴 item 비율 | 평균 차이 | Median 차이 | 상위26개 효과 제외 평균 | Zero-heavy 평균 |','|---|---|---:|---:|---:|---:|---:|']
for r in series:
    lines.append(f"| {r['seed']} | {r['baseline']} | {r['positive_series_fraction']:.1%} | {r['mean_loss_difference']:+.8f} | {r['median_loss_difference']:+.8f} | {r['trimmed_mean_without_best26']:+.8f} | {r['zero_heavy_mean']:+.8f} |")
lines+=['','item 효과는 비교군−DualClock, 양수가 유리하다. 기존 train-only item 선택 및 zero-heavy 정의를 사용했다. V 관측의 사후 진단이며 유의성·독립 재현 주장은 아니다.','','**계산량과 한계**','','| 작업 | Fits | Updates | Active seconds | Fit wall seconds | Peak MiB |','|---|---:|---:|---:|---:|---:|']
for r in costs:lines.append(f"| {r['topic']} | {r['completed_fits']} | {r['training_updates']} | {r['active_seconds']:.1f} | {r['fit_wall_seconds']:.1f} | {r['peak_allocated_mib']:.1f} |")
lines+=['','Active time은 실제 학습 step이며 V 평가·checkpoint·GPU 대기·teacher cache 준비는 포함하지 않는다. 총 queue wall은 queue_status.json, 각 F0 cache 준비 시간은 각 작업 baselines.json에 있다. 동일 updates와 minibatch를 제공했으며 동일 end-to-end 비용을 주장하지 않는다.','',
    '두 실험 모두 기존 train/V를 재사용했다. Anchor는 두 데이터셋의 네 채널, DualClock은 한 M5 subset이다. Seed는 optimizer 반복이고 독립 데이터 표본이 아니다. 성능이 좋은 조건만 선택해 새 PASS를 만들지 않았다. 후속 방법/적용 조건의 변경에는 별도 미노출 평가와 사전에 고정한 비교가 필요하다.','',
    '이 실행은 끝났으며 새 E 평가나 또 다른 연구 주제를 자동 실행하지 않는다. 과거 FAIL/STOP은 유지한다.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

os.environ.setdefault('MPLCONFIGDIR','/tmp/tsfm-diagnostic-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
colors={'standard':'#333333','summary':'#1f77b4','dualclock':'#d62728'}
for i,seed in enumerate((30000,30001)):
    for j,lr in enumerate((3e-5,1e-4)):
        ax=axes[i,j]
        for f in read(OUT/'dualclock/fits.json'):
            if f['seed']==seed and f['lr']==lr:
                rr=read(OUT/'dualclock'/(f['fit_id']+'_trajectory.json'))
                ax.plot([r['step'] for r in rr],[r['metrics']['scaled_2pinball'] for r in rr],
                        label=f['arm'],color=colors[f['arm']],marker='.',linewidth=1.2)
        ax.axvline(360,color='#777777',linestyle='--',linewidth=.8)
        ax.set(title=f'M5 seed {seed}, LR {lr:g}',xlabel='Training updates',ylabel='Validation scaled 2-pinball')
        ax.grid(alpha=.2);ax.legend()
fig.suptitle('DualClock budget diagnosis: reused validation data, no test evaluation')
fig.savefig(OUT/'dualclock_learning.png',dpi=160);plt.close(fig)
print(json.dumps(dict(anchor=summary,dualclock=long_rows,resources=costs),indent=2))
