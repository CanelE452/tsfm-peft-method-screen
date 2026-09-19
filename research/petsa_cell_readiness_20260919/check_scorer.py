from pathlib import Path
import tempfile,json,time
import numpy as np
import pandas as pd
from experiments.petsa_cell_comparison_20260919 import common as c
from experiments.petsa_cell_comparison_20260919 import score as sc

def main():
 with tempfile.TemporaryDirectory(prefix='petsa-synthetic-scorer-') as tmp:
  root=Path(tmp);out=root/'out';out.mkdir();preds={}
  arms=['B0','PLAIN','C3','MAG_ONLY','TOKEN_GATE','TOKEN_GATE_ENTROPY',*c.ARMS]
  for panel in ['electricity','electricity_transfer','ettm1',c.NEW]:
   data=root/panel/'data';data.mkdir(parents=True)
   period=96 if panel=='ettm1' else 24
   np.savez(data/'E_DISCOVERY_inputs.npz',origins=np.array([0,7*period,14*period]),sigma=np.ones(1))
   np.savez(data/'E_DISCOVERY_labels.npz',y=np.ones((3,1,64)))
   for kind in ['standard','shape']:
    folder=root/panel/kind;folder.mkdir()
    names=c.STATES if kind=='standard' else c.SHAPES
    if kind=='shape':c.save(folder/'manifest.json',dict(states=names,origin_indices=[0,1,2]))
    n=len(names)*2*3
    np.save(folder/('E_DISCOVERY_offset.npy' if kind=='standard' else 'offset.npy'),np.zeros(n))
    prediction=np.broadcast_to((1+np.arange(1,10)/10)[None,:,None],(n,9,64)).copy()
    path=folder/'synthetic_prediction.npy';np.save(path,prediction)
    for arm in arms:
     for seed in c.SEEDS:
      for stage in ['selected','fixed1024']:
       key=f'{panel}__{kind}__{arm}__s{seed}__{stage}'
       preds[key]=dict(panel=panel,kind=kind,arm=arm,seed=seed,stage=stage,step=1024,path=str(path),sha256=c.sha(path))
  assert len(preds)==224
  c.save(out/'PREDICTIONS.json',preds)
  c.save(out/'EVALUATION_SEAL.json',dict(at=1))
  c.save(out/'ALL_PREDICTIONS_SAVED.json',dict(at=2,manifest_sha256=c.sha(out/'PREDICTIONS.json')))
  # These overrides are in this disposable process only. All data and scores are synthetic.
  sc.OUT=out;sc.check_seal=lambda:None
  sc.data_path=lambda panel:root/panel/'data'
  sc.panel_path=lambda panel,kind:root/panel/kind
  sc.score()
  raw=pd.read_csv(out/'RAW_SCORES.csv');effects=pd.read_csv(out/'EFFECTS.csv')
  np.testing.assert_allclose(raw.nmae,.5,rtol=1e-12,atol=1e-12)
  np.testing.assert_allclose(effects[['gain_pct','ci_low','ci_high','bonf2_low','bonf2_high']],0,rtol=0,atol=1e-12)
  assert len(effects[effects.primary_family])==2
  protection=effects[effects.panel.isin(['electricity_transfer',c.NEW])&(effects.stage=='selected')&(effects.kind=='standard')&effects.condition.isin(['REFERENCE','FAULT'])]
  assert len(protection)==12
  result=dict(status='SYNTHETIC_SCORER_FIXTURE_VERIFIED_NOT_AN_EXPERIMENT',prediction_views=224,raw_rows=len(raw),primary_family_rows=2,protection_rows=12,equal_predictions_zero_effect=True,expected_nmae=.5,real_E_read=False,optimizer_updates=0)
  c.save(c.OUT/'SCORER_FIXTURE_CHECK.json',result)
  print(json.dumps(result,indent=2))

if __name__=='__main__':main()
