"""Final artifact, numerical and document integrity audit; no new predictions."""
import csv,gzip,re,subprocess,zipfile
from datetime import datetime,timezone
import pandas as pd
from .common import *
PAPER=ROOT/'papers/persistence_adaptation'
def main():
    seal=check_seal();v=read(OUT/'VERIFICATION.json');assert v['status']=='VERIFIED'
    pred=read(OUT/'PREDICTIONS.json');assert len(pred)==114
    assert sha(OUT/'PREDICTIONS.json')==read(OUT/'ALL_PREDICTIONS_SAVED.json')['manifest_sha256']
    assert sum(x['reused'] for x in pred.values())==24
    for key,r in pred.items():assert sha(ROOT/r['path'])==r['sha256'],key
    checks=read(OUT/'MODEL_CHECKS.json');assert len(checks)==114
    for key,r in checks.items():
        assert r['optimizer_updates']==0
        if r.get('frozen'):assert r['state_unchanged'] and r['restore']
        else:assert r.get('reused_parent_verified') or r.get('point_baseline')
    with gzip.open(OUT/'NEW_ORIGIN_SCORES.csv.gz','rt') as f:
        z=csv.reader(f);next(z);count=sum(1 for _ in z)
    assert count==640320
    raw=pd.read_csv(OUT/'RAW_SCORES.csv');effect=pd.read_csv(OUT/'EFFECTS.csv');assert len(effect)==742
    for r in effect.to_dict('records'):
        z=raw[(raw.panel==r['panel'])&(raw.kind==r['kind'])&(raw.condition==r['condition'])];a=z[z.arm==r['new']].nmae.mean();b=z[z.arm==r['baseline']].nmae.mean()
        np.testing.assert_allclose([a,b,100*(1-a/b)],[r['new_nmae'],r['baseline_nmae'],r['gain_pct']],rtol=1e-10,atol=1e-10)
    assert (effect.groupby('panel').common_draw_sha256.nunique()==1).all()
    fac=pd.read_csv(OUT/'FACTORIAL.csv');assert len(fac)==56
    np.testing.assert_allclose(fac.gate+fac.weights,fac.D-fac.A,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(fac.gate,((fac.B-fac.A)+(fac.D-fac.C))/2,rtol=1e-10,atol=1e-12)
    gates=pd.read_csv(OUT/'GATE_IDENTIFIABILITY.csv');assert len(gates)==95
    g=gates[(gates.panel=='neso_2025')&(gates.kind=='standard')&(gates.condition=='SHIFT8')].iloc[0];assert g.contexts==256 and g.equal_context_fraction==1
    cost=read(OUT/'COST.json');assert cost['new_GPU_views']==70 and cost['new_point_baseline_views']==20 and cost['new_fits']==cost['optimizer_updates']==cost['unapproved_compute_samples']==0
    allraw=pd.read_csv(PAPER/'tables/ALL_RAW_SCORES.csv');full=pd.read_csv(OUT/'RAW_SCORES_WITH_NRMSE.csv');pd.testing.assert_frame_equal(allraw,full)
    # Recompute every synthetic operating-mixture point from the raw condition means.
    mix=pd.read_csv(PAPER/'tables/T5_mixture_curves.csv');means=raw[raw.kind=='standard'].groupby(['panel','arm','condition']).nmae.mean()
    for r in mix.itertuples():
        get=lambda arm,condition:means[r.panel,arm,condition]
        d=lambda cond:get(r.baseline,cond)-get('C3',cond)
        expected=(1-r.w)*((1-r.q)*d('REFERENCE')+r.q*d('FAULT'))+r.w*d('SHIFT8')
        assert abs(expected-r.absolute_nmae_benefit)<1e-12
    assert len(list((PAPER/'figures').glob('*.png')))==9
    for extn in ['pdf','svg']:assert len(list((PAPER/'figures').glob('*.'+extn)))==9
    with zipfile.ZipFile(PAPER/'EVIDENCE_BRIEF_KO.docx') as z:
        assert len([n for n in z.namelist() if n.startswith('word/media/')])==9
        assert '논문'.encode() in z.read('word/document.xml')
    pdftext=subprocess.check_output(['pdftotext',str(PAPER/'EVIDENCE_BRIEF_KO.pdf'),'-']).decode()
    assert '640,320' in pdftext and '256' in pdftext and '원자료' in pdftext
    for i in range(1,10):assert f'그림 {i}.' in pdftext
    links=0
    for folder in [PAPER,OUT]:
        for p in folder.glob('*.md'):
            for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
                if link.startswith(('http:','https:','#')):continue
                assert (p.parent/link).exists(),(p,link);links+=1
    paths=sorted(p for folder in [EXP,OUT,PAPER] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name not in ['FINAL_ARTIFACT_AUDIT.json','run.log'])
    assert all(p.stat().st_size<100*2**20 for p in paths)
    save(OUT/'FINAL_ARTIFACT_AUDIT.json',dict(status='VERIFIED',at_utc=datetime.now(timezone.utc).isoformat(),sealed_files_unchanged=len(seal['hashes']),prediction_hashes=114,model_checks=114,origin_rows_crc_verified=count,effects_independently_replayed=742,factorial_rows=56,mask_audit_rows=95,mixture_points_replayed=len(mix),figure_sets=9,docx_embedded_images=9,pdf_all_figures_and_Korean_text_verified=True,local_links_verified=links,new_fits=0,optimizer_updates=0,artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},scope='artifact and arithmetic replay; not a second full inference execution'))
    print('FINAL_ARTIFACT_VERIFIED',len(paths),flush=True)
if __name__=='__main__':main()
