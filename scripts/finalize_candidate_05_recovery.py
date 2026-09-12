"""Audit and present the fixed-recipe recovery; never train or tune."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tsfm_peft_screen.reproducibility import ROOT,sha,write_json
from tsfm_peft_screen.metrics import score,replay
from tsfm_peft_screen.selection import require_seal
out=ROOT/'results/candidate_05_repaired';cache=ROOT/'.cache/candidate_05_repaired'
old=ROOT/'results/candidate_05';summary=ROOT/'results/screening_summary'
def read(p):return json.loads(p.read_text())
status=read(out/'status.json');assert status['verdict']!='IMPLEMENTATION_BLOCKED'
assert read(out/'job_exit.json')['exit_code']==0
contract=read(out/'contract.json');original=read(old/'contract.json');source=read(out/'execution_source.json')
assert {k:v for k,v in contract.items() if k!='source_hashes'}=={k:v for k,v in original.items() if k!='source_hashes'},'Recovery recipe changed'
assert sha(old/'status.json')==source['original_status_sha256']
assert sha(old/'contract.json')==source['original_contract_sha256']
require_seal(out/'selection.json',contract)
assert sha(out/'selection.json')==read(out/'evaluation_open.json')['selection_sha256']
for name,h in contract['source_hashes'].items():
    assert hashlib.sha256(subprocess.check_output(['git','show',f"{source['execution_commit']}:{name}"],cwd=ROOT)).hexdigest()==h,('Execution source mismatch',name)
# The original artifacts remain byte-identical to the prior final screen commit.
historical_commit='fbb37d31391f34f3cfab71c31528b34e381d4cd1'
for folder in [ROOT/'results'/f'candidate_{i:02}' for i in range(1,8)]:
    for p in folder.rglob('*'):
        if p.is_file():
            reference=subprocess.check_output(['git','show',f'{historical_commit}:{p.relative_to(ROOT)}'],cwd=ROOT)
            assert hashlib.sha256(reference).hexdigest()==sha(p),('Historical artifact changed',p)
attempts=read(out/'attempts.json');arms=contract['arms']
assert [a['arm'] for a in attempts]==arms and all(a['status']=='COMPLETE' for a in attempts)
integ=read(out/'integrity.json');usage=read(out/'resource_usage.json')
assert integ['status']=='PASS' and len(integ['streams'])==5
for item in integ['streams']:
    assert item['identity_max_abs']==0 and item['checkpoint_replay_max_abs']==0
    assert item['no_future_labels'] and item['issued_forecast_hashes_verified']==30
    if item['arm']=='MATURITY_PEFT':assert item['preservation_active_updates']>0
expected_steps={'F0':0,'IMMEDIATE_LORA':232,'WAIT_FULL':224,'TAFAS_LIKE':232,'MATURITY_PEFT':232}
assert {u['arm']:u['optimizer_steps'] for u in usage['streams']}==expected_steps
rows=list(csv.DictReader((out/'metrics.csv').open()));by={r['arm']:r for r in rows}
assert list(by)==arms
errors=[];reproduced={};hashes={};origin_losses={}
for arm in arms:
    path=cache/f'E_{arm}.npz';errors.append(replay(path));hashes[path.name]=sha(path)
    with np.load(path) as z:
        metric=score(z['prediction'],z['target'],z['scale'])
        assert abs(metric['scaled_2pinball']-float(by[arm]['scaled_2pinball']))<1e-12
        origin_losses[arm]=[score(z['prediction'][j:j+1],z['target'][j:j+1],z['scale'])['scaled_2pinball'] for j in range(30)]
        for j,origin in enumerate(contract['origins']):
            issued=cache/f'issued_{arm}_{j:02}.npz'
            with np.load(issued) as f:
                assert set(f.files)=={'prediction','origin'} and int(f['origin'])==origin
                np.testing.assert_array_equal(f['prediction'],z['prediction'][j])
            hashes[issued.name]=sha(issued)
        if arm in ['F0','IMMEDIATE_LORA','WAIT_FULL']:
            with np.load(ROOT/'.cache/candidate_05'/path.name) as prior:
                for key in ['prediction','target','scale']:np.testing.assert_array_equal(z[key],prior[key])
            reproduced[arm]=0.0
recorded=read(out/'origin_losses.json')
for arm in arms:np.testing.assert_allclose(origin_losses[arm],recorded[arm],rtol=0,atol=1e-12)
best=min(arms[1:4],key=lambda a:float(by[a]['scaled_2pinball']))
f0=float(by['F0']['scaled_2pinball']);proposed=float(by['MATURITY_PEFT']['scaled_2pinball'])
gain=100*(float(by[best]['scaled_2pinball'])-proposed)/f0
overhead=100*(float(by['MATURITY_PEFT']['adaptation_seconds'])/float(by[best]['adaptation_seconds'])-1)
early=100*float(np.mean(np.array(origin_losses[best][:25])-np.array(origin_losses['MATURITY_PEFT'][:25])))/f0
verdict='PASS' if gain>=1 and overhead<=20 and early>0 else 'WEAK' if gain>0 else 'FAIL'
assert status['verdict']==verdict and status['strongest_baseline']==best
assert abs(status['gain_percent_f0']-gain)<1e-12
assert abs(status['diagnostics']['compute_overhead_percent']-overhead)<1e-10
assert abs(status['diagnostics']['gain_excluding_last5_percent_f0']-early)<1e-12
trajectory=list(csv.DictReader((out/'trajectories.csv').open()))
for arm,steps in expected_steps.items():
    rr=[r for r in trajectory if r['arm']==arm]
    assert len(rr)==steps//8
    for j,r in enumerate(rr,1):
        assert int(r['updates'])==j*8 and r['issued_before_update']=='True'
        assert int(r['available_label_end'])==int(r['origin'])-1
if '--verify-only' in sys.argv:
    receipt=read(out/'recovery_verification.json')
    assert receipt['prediction_cache_sha256']==hashes
    assert receipt['verdict_independently_recomputed']==verdict
    latest=read(summary/'latest_review.json')
    assert latest['attempted_streams']==9 and latest['completed_streams']==8
    for item in latest['candidates']:
        assert sha(ROOT/item['source']/'status.json')==item['status_sha256']
    print('REPAIRED CANDIDATE05 VERIFICATION PASS; 5 completed streams; verdict',verdict)
    raise SystemExit(0)
write_json(out/'recovery_verification.json',dict(status='PASS',execution_commit=source['execution_commit'],original_recipe_unchanged=True,original_candidate_artifacts_unchanged=True,selection_seal_verified=True,completed_streams=5,metric_replay_max_abs=max(errors),prior_completed_prediction_max_abs=reproduced,optimizer_steps=expected_steps,verdict_independently_recomputed=verdict,issued_forecasts_verified=150,prediction_cache_sha256=hashes))
figdir=out/'figures';figdir.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':150,'savefig.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,4.5));values=[float(by[a]['scaled_2pinball']) for a in arms]
bars=ax.barh([a.replace('_',' ') for a in arms],values,color=['#157f78' if a=='MATURITY_PEFT' else '#7d91ad' for a in arms]);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.17)
for bar,v in zip(bars,values):ax.text(v+.006,bar.get_y()+bar.get_height()/2,f'{v:.6f}',va='center')
ax.set_xlabel('Issued forecast scaled 2-pinball (lower is better)');ax.set_title(f'Candidate05 repaired Round1 — {verdict}',loc='left');fig.tight_layout();fig.savefig(figdir/'primary.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5))
for arm in arms:ax.plot(np.arange(1,31),np.cumsum(origin_losses[arm])/np.arange(1,31),label=arm)
ax.set_xlabel('Issued origin');ax.set_ylabel('Cumulative prequential scaled 2-pinball');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figdir/'diagnostics.png');plt.close(fig)
table='| Method | Scaled 2-pinball ↓ | Unrevealed drift ↓ | Worst 5 loss ↓ | Adaptation seconds |\n| --- | ---: | ---: | ---: | ---: |\n'
for arm in arms:
    r=by[arm];table+=f"| {arm} | {float(r['scaled_2pinball']):.9f} | {float(r['unrevealed_drift']):.9f} | {float(r['worst5_origin_loss']):.9f} | {float(r['adaptation_seconds']):.3f} |\n"
text=f'''# Candidate05: repaired Round1 — {verdict}

[문제·방법]
부분적으로 도착한 정답으로 업데이트하면서 미공개 horizon의 업데이트 직전 예측을 보존하는 Maturity-PEFT. [고정 실험 계약](../../docs/CANDIDATE_05_RECOVERY.md).

[데이터·학습 파라미터]
Jena, 336 context / 48 horizon, 24시간 간격 30 origins, seed30000, LR1e-4, lambda1, eligible origin당 8 updates. 수정 전과 데이터·설정·평가 기준 모두 동일. TAFAS-like는 GCM만 재현하며 fixed scheduling/native probabilistic loss를 사용한 제한적 baseline이다.

[실행·원본 보존]
실행 commit `{source['execution_commit']}`. 사용자의 “나머지해줘” 요청에 따라 5개 스트림을 추가 실행했다. [기존 중단 결과](../candidate_05/RESULT.md)는 그대로 보존했다. 기존에 관측된 개발 E를 재사용한 구현 복구 실험이며 새로운 holdout 검증은 아니다. E 관측 후 설정 조정은 하지 않았다.

[raw 결과]
{table}
[강한 단순 baseline·relative 결과]
최강 online baseline: {best}. Proposed gain = 100 × (baseline loss − proposed loss) / F0 loss = **{gain:.6f}% F0** (필요 ≥1%). F0 대비 gain은 {100*(f0-proposed)/f0:.6f}%다.

[무결성]
5개 스트림 모두 30회 발행 완료. 150개 예측 파일과 최종 평가 캐시 일치. 정답은 당시 index < now인 구간만 업데이트에 사용했다. 초기 예측 및 checkpoint replay 오차 0. 독립 지표 replay 최대 오차 {max(errors):.3g}. 기존 완료 F0/Immediate/WaitFull 예측 배열과 모두 정확히 일치. Preservation이 실제 업데이트에서 활성화됐다. 검증 상세: recovery_verification.json.

[성공/실패 판정]
**{verdict}**. 최강 online baseline 대비 gain {gain:.6f}% (≥1%), adaptation overhead {overhead:.3f}% (≤20%), 마지막 5 origins 제외 gain {early:.6f}% (>0). 모든 조건을 만족해야 PASS다. 시간 수치는 이번 동일 GPU 순차 실행의 실측값이다.

[계산량]
추가 standard fits 0, streaming runs 5, optimizer updates {sum(expected_steps.values())}, candidate wall {usage['wall_seconds']:.3f}s, peak allocated GPU {max(u['gpu_peak_bytes'] for u in usage['streams'])} bytes. 누적 34 fits + 9 stream attempts (8 complete, 1 original abort). 기존 5-stream 예산 안에 든다고 주장하지 않는다.

[말할 수 없는 것]
단일 데이터·seed의 개발 결과이며 통계적 유의성, 다중 데이터 일반화, 신규성 확정 또는 미관측 holdout 성능을 주장할 수 없다. 겹치는 horizon이 있다. TAFAS의 전체 논문 재현 결과가 아니다.

[Round2 추천 여부]
{'수치 PASS로 후속 검토 가능하나 개발 E 재사용 사실을 고려해야 한다.' if verdict=='PASS' else '추천하지 않는다.'} Round2는 실행하지 않았다.
'''
(out/'RESULT.md').write_text(text)
latest=[]
for i in range(1,8):
    p=out if i==5 else ROOT/'results'/f'candidate_{i:02}'
    s=read(p/'status.json')
    latest.append(dict(candidate=i,verdict=s['verdict'],gain_percent_f0=s.get('gain_percent_f0'),source=str(p.relative_to(ROOT)),status_sha256=sha(p/'status.json')))
write_json(summary/'latest_review.json',dict(candidates=latest,editorial_candidate_order=[2,3,7,4,1,5,6],editorial_rationale='Qualitative follow-up priority, not cross-dataset statistical superiority. Candidate05 now has a completed negative comparison and stays below the other tested candidates. No weighted score sum.',candidate05_recovery_execution=source['execution_commit'],completed_standard_fits=34,attempted_streams=9,completed_streams=8,aborted_streams=1,additional_recovery_streams=5,round2_executed=False,round2_recommendations=[5] if verdict=='PASS' else [],scope='Original screen plus authorized fixed-recipe implementation recovery; original screen artifacts preserved'))
lines=['# Latest review: original screen + Candidate05 recovery','',f'Candidate05 now has a complete repaired comparison: **{verdict}**, gain vs {best} **{gain:.6f}% F0**. See [recovery report](../candidate_05_repaired/RESULT.md). Original results and ranking remain a historical record.','', '| Candidate | Latest verdict | Gain vs strongest baseline (% F0) |','| --- | --- | ---: |']
for r in latest:lines.append(f"| {r['candidate']:02} | {r['verdict']} | {r['gain_percent_f0'] if r['gain_percent_f0'] is not None else 'not evaluated'} |")
lines.extend(['','Editorial follow-up order: 02 → 03 → 07 → 04 → 01 → 05 → 06. Candidate05 now has a completed negative comparison. This order is qualitative, not cross-dataset statistical superiority.', '', '34 fits + 9 stream attempts cumulatively: 8 complete, 1 original abort. Recovery adds five streams with no tuning. No Round2 executed. Candidate06 remains stopped for novelty collision.', '', 'No PASS candidate; no Round2 recommendation.' if verdict!='PASS' else 'Candidate05 meets numerical pilot thresholds; further review must account for reused development E.'])
(summary/'latest_review.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(verification='PASS',verdict=verdict,gain_percent_f0=gain,compute_overhead_percent=overhead,early_gain_percent_f0=early,prior_completed_predictions_exact=True),indent=2))
