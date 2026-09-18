"""Observed-only features. Condition names are reporting labels, never inputs."""
import pandas as pd
from .common import *
from experiments.outlier_signal_followup_v2_20260917.model import coordinates
from experiments.outlier_signal_peft_v1_20260917.model import robust_scale

@torch.no_grad()
def batch_features(x,s):
    m,r=robust_scale(x,s);d=(x-m)/r;_,_,extreme,p=coordinates(x,s)
    assert ((p>=0)&(p<=extreme)).all()
    signs=(torch.sign(d)*extreme).numpy().astype('int8');I=extreme.numpy();pp=p.numpy()
    run=np.zeros_like(signs,dtype=np.int16)
    for t in range(512):run[:,t]=np.where(signs[:,t]!=0,np.where((t>0)&(signs[:,t]==signs[:,max(0,t-1)]),run[:,max(0,t-1)]+1,1),0)
    gap=I-pp; assert np.all(gap[run>=8]==0)
    gm=1-I.reshape(-1,32,16).mean(-1);gc=1-pp.reshape(-1,32,16).mean(-1)
    np.testing.assert_allclose(gc,correction_gate(x,s).numpy(),rtol=0,atol=0)
    pos=np.arange(512);nI=I.sum(1);mass=gap.sum(1)
    df=pd.DataFrame(dict(peak_abs_d=d.abs().max(-1).values.numpy(),extreme_fraction=nI/512,longest_same_sign_run=run.max(1),recent128_extreme_fraction=I[:,-128:].mean(1),last_extreme_position=np.where(nI>0,np.where(I>0,pos,-1).max(1),-1),gate_equal=(gc==gm).all(1),mean_C3_gate=gc.mean(1),mean_MAG_gate=gm.mean(1),gate_gap_mean=(gc-gm).mean(1),gap_mass=mass,gap_mass_recent128=gap[:,-128:].sum(1),gap_mass_recent32=gap[:,-32:].sum(1),gap_mass_short_runs=(gap*(run<8)).sum(1),partial_extreme_count=((I>0)&(pp<1)).sum(1)))
    return df

def make():
    check_seal();allrows=[];scalarchecks=0
    for panel in PANELS:
        for kind in ['standard','shape']:
            d,names,ids,nc,_=metadata(panel,kind);x,s=ext.inputs(panel,kind);n=len(ids)
            blocks=[]
            for lo in range(0,len(x),256):blocks.append(batch_features(torch.tensor(np.array(x[lo:lo+256])),torch.tensor(np.array(s[lo:lo+256]))))
            f=pd.concat(blocks,ignore_index=True);assert len(f)==len(names)*2*n*nc
            f.insert(0,'channel',np.tile(np.arange(nc),len(f)//nc));f.insert(0,'origin',np.tile(np.repeat(d['origins'][ids],nc),len(names)*2));f.insert(0,'draw',np.tile(np.repeat(np.arange(2),n*nc),len(names)));f.insert(0,'condition',np.repeat(names,2*n*nc));f.insert(0,'kind',kind);f.insert(0,'panel',panel)
            # Scalar reference for real observed-input gates, no future labels.
            xx=torch.tensor(np.array(x[[0,-1]]));ss=torch.tensor(np.array(s[[0,-1]]));m,r=robust_scale(xx,ss);dd=((xx-m)/r).numpy();expected=[]
            for a in dd:
                z=[]
                for t in range(512):
                    z.append(float(abs(a[t])>3)*sum(abs(a[j])>3 and (a[j]>0)==(a[t]>0) for j in range(max(0,t-7),t+1))/min(8,t+1))
                expected.append(1-np.array(z).reshape(32,16).mean(1));scalarchecks+=32
            np.testing.assert_allclose(expected,correction_gate(xx,ss).numpy(),rtol=0,atol=1e-7)
            allrows.append(f);print('FEATURES',panel,kind,len(f),flush=True)
    allf=pd.concat(allrows,ignore_index=True);allf.to_csv(OUT/'INPUT_FEATURES.csv.gz',index=False,compression='gzip')
    sums=allf.groupby(['panel','kind','condition']).agg(contexts=('origin','size'),gate_equal_fraction=('gate_equal','mean'),mean_gate_gap=('gate_gap_mean','mean'),mean_C3_gate=('mean_C3_gate','mean'),mean_MAG_gate=('mean_MAG_gate','mean'),extreme_fraction=('extreme_fraction','mean'),mean_longest_run=('longest_same_sign_run','mean'),gap_mass=('gap_mass','sum'),gap_mass_short_runs=('gap_mass_short_runs','sum'),gap_recent128=('gap_mass_recent128','sum'),gap_recent32=('gap_mass_recent32','sum')).reset_index()
    sums['short_run_mass_fraction']=sums.gap_mass_short_runs/sums.gap_mass;sums['recent128_mass_fraction']=sums.gap_recent128/sums.gap_mass;sums['recent32_mass_fraction']=sums.gap_recent32/sums.gap_mass;sums.to_csv(OUT/'GATE_SUMMARY.csv',index=False)
    save(OUT/'FEATURE_VERIFICATION.json',dict(status='VERIFIED',contexts=len(allf),scalar_gate_patch_checks=scalarchecks,all_gap_on_first_7_of_run=True,observed_only=True,optimizer_updates=0))

if __name__=='__main__':make()
