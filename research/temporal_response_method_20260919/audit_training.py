"""Independent ledger/receipt/checkpoint audit; never reads E scores or labels."""
from pathlib import Path
import json,hashlib,collections,math,time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/temporal_response_peft_20260919'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ledger=[json.loads(l) for l in (OUT/'UPDATE_LEDGER.jsonl').read_text().splitlines()]
    assert len(ledger)==16384
    keys=[(r['fit'],r['step']) for r in ledger];assert len(set(keys))==len(keys)
    by=collections.defaultdict(list)
    for r in ledger:by[r['fit']].append(r)
    assert len(by)==16
    receipts=[read(p) for p in (OUT/'fits').glob('*/receipt.json')];assert len(receipts)==20
    assert len([r for r in receipts if not r.get('reused')])==16
    rows=[];hashes=0
    for r in receipts:
        assert r['status']=='COMPLETE' and r['updates']==1024 and r['frozen_unchanged'] and r['microbatch']==32
        assert r['lr']==.0003 and [c['step'] for c in r['checkpoints']]==[0,256,512,768,1024]
        assert r['selected']==min(r['checkpoints'],key=lambda c:(c['objective'],c['step']))
        for c in r['checkpoints']:
            assert sha(ROOT/c['checkpoint'])==c['sha256'];hashes+=1
        if not r.get('reused'):
            z=by[r['fit']];assert [x['step'] for x in z]==list(range(1,1025))
            assert all(x['lr']==r['lr'] and x['seed']==r['seed'] and x['labelled_examples']==32 and x['adapted_forwards']==x['teacher_forwards']==2 for x in z)
            assert all(math.isfinite(x[k]) for x in z for k in ['loss','supervised','regularizer','gradient_norm','seconds'])
            assert math.isclose(sum(x['seconds'] for x in z),r['optimizer_seconds'],rel_tol=1e-12)
            assert read(OUT/'fits'/r['fit']/'intent.json')==dict(status='JOURNALED',completed_step=1024)
            assert r['buffers_unchanged']
        rows.append(dict(fit=r['fit'],reused=bool(r.get('reused')),selected_step=r['selected']['step'],selected_v=r['selected']['objective'],fixed1024_v=r['checkpoints'][-1]['objective']))
    smoke=[json.loads(l) for p in OUT.glob('smoke_*.jsonl') for l in p.read_text().splitlines()];assert len(smoke)==16
    seal=read(OUT/'SEAL.json')
    for p,h in seal['hashes'].items():assert sha(ROOT/p)==h
    amend=read(OUT/'SEAL_AMENDMENT_01.json');assert sha(OUT/'SEAL_INITIAL.json')==amend['original_seal_sha256']
    audit=dict(at=time.time(),status='VERIFIED_TRAINING_ONLY',new_fits=16,reused_fits=4,main_updates=16384,smoke_updates=16,unique_update_keys=True,checkpoint_hash_checks=hashes,all_v_only_selections_replayed=True,all_frozen_hash_and_buffer_receipts_verified=True,all_optimizer_times_reconciled=True,scientific_settings_unchanged=True,source_hash_checks=len(seal['hashes']),new_E_scores_read=False,selection_details=rows,audit_code_sha256=sha(Path(__file__)))
    (OUT/'TRAINING_AUDIT.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print({k:v for k,v in audit.items() if k!='selection_details'})
if __name__=='__main__':main()
