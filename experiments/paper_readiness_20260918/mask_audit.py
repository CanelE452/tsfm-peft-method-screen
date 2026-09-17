"""Post-hoc CPU audit of gate distinguishability, without predictions or tuning."""
import pandas as pd
from .common import *
from experiments.additive_persistence_validation_v1_20260917.model import mechanism_gate

def audit():
    check_seal();assert read(OUT/'VERIFICATION.json')['status']=='VERIFIED';torch.set_num_threads(4);rows=[]
    for panel in PANELS:
        for kind in ['standard','shape']:
            x,s=inputs(panel,kind);names=STATES if kind=='standard' else read(panel_path(panel,kind)/'manifest.json')['states'];nc=len(read(ext.OUT/'DATA_MANIFEST.json')[panel]['selected_columns']);n=128 if kind=='standard' else 64;block=2*n*nc
            for k,name in enumerate(names):
                eq=one=0;distance=0.;maxdiff=0.
                for lo in range(k*block,(k+1)*block,256):
                    hi=min(lo+256,(k+1)*block);xx=torch.tensor(np.array(x[lo:hi]),dtype=torch.float32);ss=torch.tensor(np.array(s[lo:hi]),dtype=torch.float32)
                    a=mechanism_gate(xx,ss,'C3');b=mechanism_gate(xx,ss,'M_RECENCY');diff=(a-b).abs();eq+=int((a==b).all(-1).sum());one+=int((a==1).all(-1).sum());distance+=float(diff.double().sum());maxdiff=max(maxdiff,float(diff.max()))
                rows.append(dict(panel=panel,kind=kind,condition=name,contexts=block,C3_RECENCY_exact_equal_contexts=eq,equal_context_fraction=eq/block,C3_all_one_context_fraction=one/block,mean_abs_gate_difference=distance/(block*32),max_abs_gate_difference=maxdiff,posthoc=True,new_model_predictions=0,optimizer_updates=0))
    f=pd.DataFrame(rows);f.to_csv(OUT/'GATE_IDENTIFIABILITY.csv',index=False);f.to_csv(ROOT/'papers/persistence_adaptation/tables/T9_gate_identifiability.csv',index=False)
    save(OUT/'MASK_AUDIT.json',dict(status='VERIFIED',rows=len(f),posthoc_after_score=True,input_only=True,new_predictions=0,optimizer_updates=0,all_registered_conditions_included=True))
    print(f[(f.panel=='neso_2025')&(f.kind=='standard')][['condition','equal_context_fraction','C3_all_one_context_fraction','mean_abs_gate_difference']].to_string(index=False))
if __name__=='__main__':audit()
