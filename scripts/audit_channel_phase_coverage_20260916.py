"""Manifest-only audit of already executed window sampling; no fits or new score selection."""
from pathlib import Path
import hashlib,json,collections
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/channel_residual_evidence_20260916'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    path=ROOT/'results/channel_sharing_screen_v1_20260915/contract.json';c=json.loads(path.read_text());rows=[]
    for d,dc in c['data'].items():
        r=dict(dataset=d,phase_modulus=24,splits={})
        for split,oo in dc['origins'].items():
            count=collections.Counter(o%24 for o in oo)
            r['splits'][split]=dict(origins=len(oo),origin_phase_counts=dict(sorted(count.items())),distinct_phases=len(count))
        assert set(o%24 for o in dc['origins']['train'])=={0}
        assert set(o%24 for o in dc['origins']['evaluation'])=={19}
        assert set(o%24 for o in dc['origins']['validation'])=={4 if d=='electricity' else 16}
        r['all_train_targets_cover_every_phase']=all(len({(o+h)%24 for h in range(96)})==24 for o in dc['origins']['train'])
        rows.append(r)
    (OUT/'phase_coverage_audit.json').write_text(json.dumps(dict(source=str(path.relative_to(ROOT)),source_sha256=sha(path),audit_code_sha256=sha(Path(__file__)),rows=rows,interpretation='Forecast origins, not all observed hours, have singleton phase coverage. Phase is row-index modulo24, not independently verified local civil time. Distribution mismatch is verified; causal performance impact not yet tested.',affects='Completed 24-fit R2 and 4-fit identity diagnostic reuse this exact manifest',neural_fits=0),indent=2)+'\n')
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
