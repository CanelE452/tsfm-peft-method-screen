"""Complete the predeclared channel-mean nRMSE reporting from saved origin scores."""
import pandas as pd
from .common import *
def main():
    check_seal();cols=['panel','kind','arm','seed','condition','origin','channel','nmae','mae','normalized_mse','pinball','crossing']
    p=pd.read_csv(PRIOR/'SCORES_BY_ORIGIN.csv.gz',dtype={'channel':str});p=p[(p.stage=='selected')&p.kind.isin(['standard','shape'])][cols]
    f=pd.concat([p,pd.read_csv(ext.OUT/'NEW_ORIGIN_SCORES.csv.gz',dtype={'channel':str})[cols],pd.read_csv(OUT/'NEW_ORIGIN_SCORES.csv.gz',dtype={'channel':str})[cols]],ignore_index=True)
    f['condition']=f.condition.map(lambda c:'FAULT' if c.startswith(('POINT','BURST')) else c)
    keys=['panel','kind','condition','arm','seed']
    raw=f.groupby(keys)[['nmae','mae','normalized_mse','pinball','crossing']].mean()
    nrmse=f.groupby(keys+['channel']).normalized_mse.mean().pow(.5).groupby(level=list(range(len(keys)))).mean().rename('nrmse_channel_mean')
    z=raw.join(nrmse).reset_index();ref=pd.read_csv(OUT/'RAW_SCORES.csv');np.testing.assert_allclose(z[ref.columns[5:]],ref[ref.columns[5:]],rtol=1e-10,atol=1e-12)
    z.to_csv(OUT/'RAW_SCORES_WITH_NRMSE.csv',index=False);z.to_csv(ROOT/'papers/persistence_adaptation/tables/ALL_RAW_SCORES.csv',index=False)
    save(OUT/'NRMSE_REPORTING_CHECK.json',dict(status='VERIFIED',rows=len(z),definition='mean across channels of sqrt(mean origin/draw/condition normalized MSE), separately per model seed',other_metrics_match_sealed_scores=True,new_predictions=0,optimizer_updates=0))
if __name__=='__main__':main()
