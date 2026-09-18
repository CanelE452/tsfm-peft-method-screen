"""Validate report numbers, artifact links, and unchanged scientific inputs."""
import ast,re
import pandas as pd
from .common import *

def audit():
    check_seal();assert read(OUT/'INDEPENDENT_AUDIT.json')['status']=='VERIFIED'
    f=pd.read_csv(OUT/'DECOMPOSITION.csv');r=f[(f.panel=='electricity_transfer')&(f.kind=='standard')&(f.condition=='SHIFT8')&(f.family=='ALL')].iloc[0]
    np.testing.assert_allclose(r[['A','B','C','D']].to_numpy(float),[.358660224,.360609109,.356891993,.357760389],rtol=0,atol=5e-10)
    s=pd.read_csv(OUT/'SEED_DECOMPOSITION.csv');s=s[(s.panel=='electricity_transfer')&(s.kind=='standard')&(s.condition=='SHIFT8')&(s.family=='ALL')];seed_fraction=s[s.seed==81552].iloc[0].total/s.total.sum();assert round(100*seed_fraction,1)==85.4
    c=pd.read_csv(OUT/'CHANNEL_CONTRIBUTIONS.csv');c=c[(c.panel=='electricity_transfer')&(c.kind=='standard')&(c.condition=='SHIFT8')];assert (c.total>0).sum()==13 and (c.total<0).sum()==3 and c.leave_out_MAG_gain_pct.min()>0
    for p in EXP.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    links=0
    for p in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^\s)]+)\)',p.read_text()):
            if '://' in target or target.startswith('#'):continue
            assert (p.parent/target.split('#')[0]).exists(),(p,target);links+=1
    for name in ['01_gate_weights_decomposition','02_state_tradeoffs','03_channel_effects','04_shapes_decomposition']:
        for extn in ['png','pdf','svg']:assert (OUT/'figures'/f'{name}.{extn}').stat().st_size>1000
    hashes={}
    for base in [EXP,OUT]:
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.name!='PUBLICATION_AUDIT.json':
                assert p.stat().st_size<95*2**20;hashes[str(p.relative_to(ROOT))]=sha(p)
    save(OUT/'PUBLICATION_AUDIT.json',dict(status='VERIFIED',report_key_numbers=True,local_markdown_links=links,figures=12,artifact_hashes=hashes,posthoc=True,new_fits=0,optimizer_updates=0))
    print('PUBLICATION_AUDIT_VERIFIED',len(hashes),flush=True)
if __name__=='__main__':audit()
