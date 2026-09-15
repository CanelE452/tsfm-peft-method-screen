"""Descriptive prediction disagreement; no new inference or fitting."""
from pathlib import Path
import json,hashlib,csv,math
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/channel_phase_transport_20260916'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ev=json.loads((OUT/'evaluation.json').read_text());rows=[]
    for r in ev:
        if r['arm']!='CONDITIONED' or r['role']!='selected':continue
        assert sha(ROOT/r['prediction_path'])==r['prediction_sha256']
        with np.load(ROOT/r['prediction_path']) as z:p=z['prediction'].astype(np.float64);y=z['target'].astype(np.float64)
        for arm in ['POINTWISE','UNIFORM']:
            b=next(b for b in ev if b['arm']==arm and (b['dataset'],b['seed'],b['role'])==(r['dataset'],r['seed'],r['role']))
            assert sha(ROOT/b['prediction_path'])==b['prediction_sha256']
            with np.load(ROOT/b['prediction_path']) as z:q=z['prediction'].astype(np.float64);assert np.array_equal(y,z['target'])
            delta=p-q;e=q-y;cross=float(2*np.mean(e*delta));energy=float(np.mean(delta**2));change=float(np.mean((p-y)**2)-np.mean(e**2))
            assert math.isclose(cross+energy,change,abs_tol=1e-12,rel_tol=1e-12)
            rows.append(dict(dataset=r['dataset'],seed=r['seed'],baseline=arm,prediction_disagreement_rms=math.sqrt(energy),disagreement_mse=energy,error_cross_term=cross,mse_change=change))
    with (OUT/'prediction_disagreement.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (OUT/'diagnostic_verification.json').write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),prediction_pairs=len(rows),all_mse_change_identities_verified=True,additional_fits=0,additional_inference=0,scope='Already produced selected E caches; posthoc diagnostic, not a deployment correction.'),indent=2)+'\n')
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
