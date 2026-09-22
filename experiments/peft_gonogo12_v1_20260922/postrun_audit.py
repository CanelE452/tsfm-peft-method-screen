"""Read-only numerical audit of saved selections, data and forecast receipts."""
import numpy as np
import torch
from common import *
from evaluation import readpred,predpath,calibration,score
from runner import check_seal,H1_METHODS,H2_METHODS
import data


def main():
    check_seal();d=data.load_data();checks=0
    for candidate,methods in [('h1',H1_METHODS),('h2',H2_METHODS)]:
        selection=read(RESULTS/f'SELECTION_{candidate.upper()}.json')
        for arm in methods:
            for mode in ['RAW','CAL']:
                values=[]
                for seed in CONFIG[candidate]['seeds']:
                    cal=readpred(predpath(candidate,'dev',seed,arm,2,'cal'))
                    val=readpred(predpath(candidate,'dev',seed,arm,2,'validation'))
                    values.append(score(val,calibration(cal),mode))
                assert abs(np.mean(values)-selection['details'][arm]['options'][mode])<1e-12
                checks+=1
        evalcal=read(RESULTS/f'CALIBRATION_{candidate.upper()}.json')
        for seed,arms in evalcal.items():
            for arm,coefficients in arms.items():
                actual=calibration(readpred(predpath(candidate,'eval',int(seed),arm,2,'cal')))
                for sid in coefficients:np.testing.assert_allclose(actual[sid],coefficients[sid],rtol=0,atol=1e-12)
                checks+=1
    forecasts=0
    for path in (CACHE/'predictions').glob('*/eval/*/*/2_test*.npz'):
        p=readpred(path)
        for i,(sid,origin) in enumerate(p['pairs']):
            origin=int(origin);sid=str(sid)
            np.testing.assert_array_equal(p['y'][i],d.values[origin:origin+24,d.index[sid]])
            expected=data.mask_for(sid,origin,'eval_test',str(p['conditions'][i]),repeat=int(p['repeats'][i]))
            np.testing.assert_array_equal(p['masks'][i],expected)
        forecasts+=1
    states=torch.load(CACHE/'H1_states.pt',weights_only=True)
    gates=read(RESULTS/'H1_EVAL_VARIANCE.json')
    for seed,state in states.items():
        dev=[]
        for group in ['dev','eval']:
            for sid,arms in state[group].items():
                a=np.concatenate([v.numpy().astype(float).ravel() for v in arms['BLOCK_A'][32].values()])
                b=np.concatenate([v.numpy().astype(float).ravel() for v in arms['BLOCK_B'][32].values()])
                variance=np.mean((a-b)**2)/4;energy=np.mean(((a+b)/2)**2)
                if group=='dev':dev.append(energy-variance)
                else:
                    tau=state['tau2'][2];expected=tau/(tau+variance) if tau+variance else 0
                    assert abs(expected-gates[str(seed)][sid]['g'])<1e-12
        # Production uses FP32 averaging of coefficients; this oracle averages in FP64.
        assert abs(max(float(np.mean(dev)),0)-state['tau2'][2])<1e-8
    manifest=read(RESULTS/'MANIFEST.json')
    for relative,record in manifest['local_caches'].items():assert sha(ROOT/relative)==record['sha256']
    for relative,digest in manifest['results'].items():assert sha(RESULTS/relative)==digest
    save(RESULTS/'POSTRUN_AUDIT.json',dict(status='PASS',saved_selection_and_calibration_checks=checks,
        evaluation_prediction_files=forecasts,test_truth_and_masks_match_source=True,h1_gate_numpy_oracle=True,
        manifest_local_files=len(manifest['local_caches']),manifest_result_files=len(manifest['results']),
        additional_model_forward=0,additional_optimizer_updates=0,scope='Post-run CPU cache audit; no new model or selection'))
    print('PASS',checks,forecasts)


if __name__=='__main__':main()
