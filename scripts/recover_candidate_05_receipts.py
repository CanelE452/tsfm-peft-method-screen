"""Recover completed receipts from immutable issued forecasts after abort."""
import csv,json,re,shutil
import numpy as np
from tsfm_peft_screen.reproducibility import ROOT,write_json,sha
from tsfm_peft_screen.metrics import score,replay
from tsfm_peft_screen.runners.common_fit import csv_write
out=ROOT/'results/candidate_05';cache=ROOT/'.cache/candidate_05'
contract=json.loads((out/'contract.json').read_text());status=json.loads((out/'status.json').read_text());assert status['verdict']=='IMPLEMENTATION_BLOCKED'
rows=[];usage=[];checks=[];losses={};trajectory=[]
for line in (ROOT/'.cache/candidate_05.log').read_text().splitlines():
    if line.startswith('STREAM '):
        _,arm,index,updates,loss=line.split();now=contract['origins'][int(index)]
        trajectory.append(dict(arm=arm,origin=now,updates=int(updates),last_update_loss=float(loss),available_label_end=now-1,source='preserved worker stdout'))
for arm in ['F0','IMMEDIATE_LORA','WAIT_FULL']:
    path=cache/f'E_{arm}.npz'
    with np.load(path) as z:p=z['prediction'];y=z['target'];scale=z['scale']
    for i in range(30):
        with np.load(cache/f'issued_{arm}_{i:02}.npz') as issue:
            assert 'target' not in issue.files
            assert int(issue['origin'])==contract['origins'][i]
            assert np.array_equal(issue['prediction'],p[i])
    err=replay(path);row=score(p,y,scale);originloss=[score(p[i:i+1],y[i:i+1],scale)['scaled_2pinball'] for i in range(30)]
    losses[arm]=originloss;rows.append(dict(arm=arm,variant='issued_completed_stream',**row,worst5_origin_loss=float(np.mean(sorted(originloss)[-5:])),receipt='replayed from saved predictions after candidate abort'))
    updates=max([t['updates'] for t in trajectory if t['arm']==arm],default=0)
    usage.append(dict(arm=arm,optimizer_steps=updates,completed=True,measurement_limit='Individual wall/peak-memory measurements were not incrementally persisted by the aborted worker'))
    checks.append(dict(arm=arm,metric_replay_abs=err,issued_cache_matches_e_cache=True,issued_files_have_no_targets=True,issued_count=30,checkpoint_sha256=sha(cache/f'{arm}_final.pt')))
usage.append(dict(arm='TAFAS_LIKE',optimizer_steps=0,completed=False,error='Nonfinite gradient before first optimizer.step',issued_count=len(list(cache.glob('issued_TAFAS_LIKE_*.npz')))))
csv_write(out/'metrics.csv',rows);csv_write(out/'trajectories.csv',trajectory);csv_write(out/'selections.csv',[dict(arm=a,lr=1e-4 if a!='F0' else 0,recipe='sealed before E',execution='completed' if a in losses else 'failed' if a=='TAFAS_LIKE' else 'not started') for a in contract['arms']]);write_json(out/'origin_losses.json',losses)
write_json(out/'integrity.json',dict(status='FAIL',failed_gate='finite gradient in TAFAS input GCM with partial targets',completed_stream_checks=checks,metric_replay_max_abs=max(c['metric_replay_abs'] for c in checks),selection_seal_before_e=True,maturity_not_run=True,scope='Reconstruction of preserved receipts only; no new GPU inference, optimization, or E opening'))
job=json.loads((out/'job_exit.json').read_text());write_json(out/'resource_usage.json',dict(fit_count=0,stream_count=4,completed_stream_count=3,failed_stream_count=1,not_started=['MATURITY_PEFT'],wall_seconds=job['wall_seconds'],streams=usage,resource_limitation='Per-stream timing/peak memory lost on worker abort; only supervisor wall time and stdout-confirmed optimizer counts are recoverable'))
status.update(proposed='MATURITY_PEFT',proposed_primary=None,gain_percent_f0=None,strongest_baseline=None,strongest_completed_baseline=min(rows[1:],key=lambda x:x['scaled_2pinball'])['arm'],round2_recommended=False,reason='TAFAS-like partial-label gradient integrity failed; proposed stream not reached; no valid complete comparison',diagnostics=dict(completed_streams=3,failed_streams=1,proposed_stream_executed=False))
write_json(out/'status.json',status)
if not (out/'attempts_at_failure.json').exists():shutil.copyfile(out/'attempts.json',out/'attempts_at_failure.json')
a=json.loads((out/'attempts.json').read_text());a[-1]['status']='FAILED';a[-1]['error']='Nonfinite stream gradient';write_json(out/'attempts.json',a)
print('Recovered three completed stream metrics; no proposed metric manufactured.')
