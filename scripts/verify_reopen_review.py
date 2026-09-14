"""CPU artifact reaggregation; no model calls or training."""
import csv,json
from pathlib import Path
import numpy as np
from tsfm_peft_screen.metrics import score,independent
from tsfm_peft_screen.data import Panel
from tsfm_peft_screen.reproducibility import ROOT,sha,write_json
R=ROOT/'research/reopen_review_20260914'
def read(p):return json.loads(Path(p).read_text())
def close(a,b):assert np.isclose(a,b,rtol=1e-10,atol=1e-12),(a,b)
def huber(v):return np.where(abs(v)<1,.5*v*v,abs(v)-.5)

def main():
    checks=0
    p=ROOT/'results/reopen_fr_diagnostic_20260914';rr=list(csv.DictReader(open(p/'pair_diagnostics.csv')))
    for r in rr:
        f=ROOT/f".cache/reopen_fr_diagnostic_20260914/{r['dataset']}_{r['seed']}_{r['origin']}.npz"
        with np.load(f,allow_pickle=False) as z:
            pred=z['prediction'].astype(float);base=z['F0'].astype(float);target=z['target'];sc=z['scale']
        loss=score(pred,target,sc)['scaled_2pinball'];b=score(base,target,sc)['scaled_2pinball']
        close(loss,float(r['future_loss']));close(b,float(r['F0_future_loss']));close(loss,independent(pred,target,sc));checks+=3
        d=(pred-base)/sc[None,:,None,None]
        close(huber(d).mean(),float(r['F0_correction_magnitude']))
        close(huber(d[0,:,:,24:]-d[1,:,:,:24]).mean(),float(r['F0_correction_revision']))
        close(huber((pred[0,:,:,24:]-pred[1,:,:,:24])/sc[:,None,None]).mean(),float(r['raw_forecast_revision']));checks+=3
    panel=Panel('m5');schedule=read(ROOT/'results/candidate_07/sampling_manifest.json')
    p=ROOT/'results/reopen_censor_diagnostic_20260914';rr=list(csv.DictReader(open(p/'batch_diagnostics.csv')))
    for r in rr:
        s=schedule[int(r['batch'])];n=0
        for o,cs in zip(s['origins'],s['channels']):
            _,y=panel.window(o,cs);n+=int((y>panel.caps[np.array(cs),None]).sum())
        assert int(r['censored'])==n;checks+=1
    for state in read(p/'summary.json')['states']:
        a=[r for r in rr if r['arm']==state['arm']];n=sum(int(r['censored']) for r in a);assert n==state['total_censored_positions']
        for key,col in [('saturated_fraction','saturated'),('zero_gradient_fraction','zero_gradient'),('saturated_zero_gradient_fraction','saturated_zero_gradient')]:
            close(state[key],sum(int(r[col]) for r in a)/n);checks+=1
        close(state['saturated_zero_loss_fraction'],sum(float(r['saturated_zero_loss_sum']) for r in a)/sum(float(r['censor_loss_sum']) for r in a));checks+=1
    p=ROOT/'results/reopen_query_resource_20260914';front=read(p/'frontier.json');measure=read(p/'measurements.json');quality=read(ROOT/'results/forecast_query_equal_time/evaluation.json')
    assert len(measure)==18 and all(r['passed'] for r in read(p/'parity.json'))
    for r in front['points']:
        m=next(x for x in measure if (x['dataset'],x['arm'],x['checkpoint_blocks'])==(r['dataset'],r['arm'],r['cp']))
        close(r['memory'],max(x['peak_allocated'] for x in m['measured']));close(r['time'],np.median([x['seconds'] for x in m['measured']]))
        close(r['loss'],np.mean([x['metrics']['scaled_2pinball'] for x in quality if x['dataset']==r['dataset'] and x['arm']==r['arm']]));checks+=3
    for r in front['query_decisions']:
        q=r['query'];dom=[x for x in front['points'] if x['dataset']==q['dataset'] and x['arm'] in ['standard','side'] and all(x[k]<=q[k] for k in ['loss','memory','time']) and any(x[k]<q[k] for k in ['loss','memory','time'])]
        assert dom==r['dominators'];checks+=1
    history=read(ROOT/'results/patchphase_v2_support_complete/contract.json')['history']
    assert all(sha(ROOT/f)==h for f,h in history.items())
    for state in read(ROOT/'results/reopen_censor_diagnostic_20260914/manifest.json')['states']:assert sha(state['checkpoint'])==state['checkpoint_sha256']
    for state in read(ROOT/'results/reopen_fr_diagnostic_20260914/manifest.json')['states']:
        c=state['selected'];assert sha(ROOT/c['checkpoint_file'])==c['checkpoint_sha256']
    write_json(R/'verification.json',dict(status='VERIFIED',independent_checks=checks,historical_results_unchanged=len(history),
        FR_prediction_replays=32,censor_sampling_rows=720,query_resource_points=18,query_parity_all_passed=True,
        additional_model_forward=0,additional_backward=0,additional_fits=0,
        limitation='Censor counts/loss fractions independently reaggregated from batch records and train sampling; per-position gradient tensors were not persisted. Query equivalence based on saved full tensor comparison errors; no re-run.',
        query_peak_scope='receipt.resources.gpu_peak_bytes reflects last allocator reset; per-config measurements are authoritative; maximum measured allocated bytes='+str(max(x['peak_allocated'] for x in measure))))
    print('VERIFIED',checks,'independent checks;',len(history),'historical results unchanged')

if __name__=='__main__':main()
