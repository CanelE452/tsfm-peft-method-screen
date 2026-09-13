"""Read-only post-hoc parent diagnostics; no new forecasting claims."""
import json
from pathlib import Path
import numpy as np
import torch
from tsfm_peft_screen.reproducibility import ROOT,sha,write_json
from tsfm_peft_screen.metrics import score

PARENT=ROOT/'results/overnight_20260913'
CACHE=ROOT/'.cache/overnight_20260913'
OUT=ROOT/'research/calibration_anchor_20260914'

def main():
    OUT.mkdir(exist_ok=True)
    results=json.loads((PARENT/'verification.json').read_text())
    assert results['status']=='VERIFIED'
    learning=[]
    for topic in ['anchor','drift','distill']:
        fits=json.loads((PARENT/topic/'fits.json').read_text())
        for dataset in ['etth1','traffic']:
            for arm in sorted({f['arm'] for f in fits}):
                ff=[f for f in fits if f['dataset']==dataset and f['arm']==arm]
                regressions=[]; ratios=[]
                for f in ff:
                    tr=json.loads((PARENT/topic/(f['fit_id']+'_trajectory.json')).read_text())
                    regressions.append(100*(tr[-1]['metrics']['scaled_2pinball']/f['best']['metrics']['scaled_2pinball']-1))
                    ratios.append(float(np.median([x['regularizer']/max(x['loss'],1e-8) for x in f['resources'][-100:]])))
                learning.append(dict(topic=topic,dataset=dataset,arm=arm,
                    best_steps=[f['best']['step'] for f in ff],
                    median_final_V_regression_percent=float(np.median(regressions)),
                    median_last100_regularizer_to_loss=float(np.median(ratios))))
    sel=json.loads((PARENT/'anchor/selection_seal.json').read_text())['selections']
    counterfactual=[]
    for dataset in ['etth1','traffic']:
        with np.load(CACHE/'anchor'/f'{dataset}_teacher_validation_1024.npz') as z:
            f0=np.sort(z['prediction'],axis=2);target=z['target'];scale=z['scale']
        for arm in ['native','raw','prediction_anchor']:
            losses=[]
            for s in [s for s in sel if s['dataset']==dataset and s['arm']==arm]:
                with np.load(CACHE/s['prediction_file']) as z:
                    p=np.sort(z['prediction'],axis=2)
                shift=p[:,:,10:11]-f0[:,:,10:11]
                width=lambda a:np.maximum(a[:,:,18:19]-a[:,:,2:3],scale[None,:,None,None]*1e-5)
                variants=[p,f0+shift,p-shift,p[:,:,10:11]+(f0-f0[:,:,10:11])*width(p)/width(f0)]
                losses.append([score(v,target,scale)['scaled_2pinball'] for v in variants])
            v=np.mean(losses,axis=0)
            counterfactual.append(dict(dataset=dataset,arm=arm,original=float(v[0]),
                frozen_spread=float(v[1]),frozen_median=float(v[2]),frozen_standardized_shape=float(v[3]),
                frozen_spread_gain_percent=float(100*(1-v[1]/v[0])),
                frozen_standardized_shape_gain_percent=float(100*(1-v[3]/v[0]))))
    gate_weights=[]
    ds=json.loads((PARENT/'drift/selection_seal.json').read_text())['selections']
    for s in ds:
        if not s['arm'].endswith('_gate'):
            continue
        st=torch.load(CACHE/s['checkpoint_file'],map_location='cpu',weights_only=True)
        norm=sum(float(v.square().sum()) for k,v in st.items() if '.condition.weight' in k)**.5
        gate_weights.append(dict(dataset=s['dataset'],arm=s['arm'],seed=s['seed'],step=s['step'],condition_weight_L2=norm))
    record=dict(parent_verification_sha256=sha(PARENT/'verification.json'),parent_totals=results['totals'],
        verdicts={r['topic']:r['verdict'] for r in results['topics']},
        learning=learning,counterfactual_V_recombination=counterfactual,selected_gate_weights=gate_weights,
        scope='Post-hoc development diagnosis. Recombined V predictions are not trained models, new E results, or proof of causal mechanisms.',
        automatic_continuation=dict(detected_completion=True,triggered_once=True,exit_code=1,
            reason='Installed Codex CLI rejected gpt-6-astra: newer CLI required; no subsequent training was started automatically.',
            recovery='User returned; work continued directly in the existing conversation. Failed trigger and logs preserved.'))
    write_json(OUT/'failure_diagnostics.json',record)
    print('Wrote post-hoc diagnosis; parent experiments unchanged.')

if __name__=='__main__':
    main()
