"""Bounded checkpoint diagnostics only: no FR/Censor optimizer or fits."""
import argparse,fcntl,gc,json,time,csv
from pathlib import Path
import numpy as np
import torch
import run_temporal_transfer_gradient_v1 as G
from tsfm_peft_screen.reassessment import DiagnosticModel
from tsfm_peft_screen.data import Panel
from tsfm_peft_screen.runners.common_fit import build,batch as old_batch,predict
from tsfm_peft_screen.lora import restore as old_restore
from tsfm_peft_screen.candidates.censor import cdf
from tsfm_peft_screen.metrics import score,independent
from tsfm_peft_screen.reproducibility import ROOT,sha,write_json,seed_all,guard


def read(p):return json.loads(Path(p).read_text())
def csv_write(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def huber(v):return np.where(abs(v)<1,.5*v*v,abs(v)-.5)


def main(mode):
    out=ROOT/f'results/reopen_{mode}_diagnostic_20260914';cache=ROOT/f'.cache/reopen_{mode}_diagnostic_20260914'
    assert not out.exists() and not cache.exists(),'No overwrite/retry';out.mkdir();cache.mkdir()
    G.OUT=out;watch=G.Budget();lock=open(ROOT/'.cache/gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();forward=0;cdf_backward=0;rows=[];states=[]
    history={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results').rglob('*') if p.is_file() and out not in p.parents}
    def save(status='RUNNING',**kw):write_json(out/'receipt.json',dict(status=status,new_fits=0,optimizer_updates=0,model_backward=0,
        forward=forward,cdf_tensor_backward=cdf_backward,seconds=time.monotonic()-start,**kw))
    try:
        watch.resource('startup')
        if mode=='fr':
            parent=ROOT/'results/reassessment_diagnostics_20260914';contract=read(parent/'contract.json')
            fits=read(parent/'anchor/fits.json');base=read(parent/'anchor/baselines.json');allstates=[]
            for ds in ['etth1','traffic']:
                f0=next(r for r in base if r['dataset']==ds and r['split']=='validation')['metrics']['scaled_2pinball']
                for seed in [33000,33001]:
                    choices=[r['best'] for r in fits if r['dataset']==ds and r['seed']==seed and r['arm']=='native']
                    chosen=min(choices,key=lambda r:(r['metrics']['scaled_2pinball'],r['step'],r['lr']))
                    eligible=chosen['step']>0 and chosen['metrics']['scaled_2pinball']<f0
                    allstates.append(dict(dataset=ds,seed=seed,selected=chosen,F0_V=f0,eligible=eligible))
            write_json(out/'manifest.json',dict(states=allstates,origin_indices=[0,2,4,6,8,10,12,14],base_origins=list(range(10752,12193,96)),
                pair_delta=24,horizon=48,context=1024,new_fits=0,scope='Historical development V, actual aligned future targets within each pair, not independent E. Standard selected by old V only. First eight evenly spread pair bases; no performance-driven origin selection.'))
            teachers={}
            for state in allstates:
                if not state['eligible']:continue
                ds=state['dataset'];seed=state['seed'];c=state['selected'];seed_all(seed)
                inp=contract['inputs'][ds];assert sha(ROOT/inp['path'])==inp['sha256']
                with np.load(ROOT/inp['path'],allow_pickle=False) as f:values=f['values'];sc=f['scale']
                assert sha(ROOT/c['checkpoint_file'])==c['checkpoint_sha256']
                m=DiagnosticModel('anchor','native',seed);ps={n:p for n,p in m.named_parameters() if p.requires_grad}
                G.restore(ps,torch.load(ROOT/c['checkpoint_file'],map_location='cpu',weights_only=True));before=G.tensor_hash(dict(m.named_parameters()))
                for idx in [0,2,4,6,8,10,12,14]:
                    o=10752+96*idx;oo=[o,o+24];watch.resource('fr_pair')
                    x=torch.tensor(np.concatenate([values[v-1024:v].T for v in oo]),device='cuda');g=torch.arange(2,device='cuda').repeat_interleave(4)
                    y=np.stack([values[v:v+48].T for v in oo])
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        if (ds,o) not in teachers:
                            forward+=1;teachers[ds,o]=m(x,g,frozen=True)[1].cpu().numpy().reshape(2,4,21,48)
                        forward+=1;p=m(x,g)[1].cpu().numpy().reshape(2,4,21,48)
                    b=teachers[ds,o];d=(p.astype(float)-b.astype(float))/sc[None,:,None,None]
                    raw=(p[0,:,:,24:].astype(float)-p[1,:,:,:24].astype(float))/sc[:,None,None]
                    fr=d[0,:,:,24:]-d[1,:,:,:24]
                    assert np.array_equal(y[0,:,24:],y[1,:,:24],equal_nan=True)
                    loss=score(p,y,sc)['scaled_2pinball'];f0loss=score(b,y,sc)['scaled_2pinball'];assert abs(loss-independent(p,y,sc))<=1e-10
                    rows.append(dict(dataset=ds,seed=seed,step=c['step'],lr=c['lr'],origin=o,later_origin=o+24,
                        raw_forecast_revision=float(huber(raw).mean()),F0_correction_revision=float(huber(fr).mean()),
                        F0_correction_magnitude=float(huber(d).mean()),future_loss=loss,F0_future_loss=f0loss,adaptation_gain_percent=100*(f0loss-loss)/f0loss))
                    np.savez_compressed(cache/f'{ds}_{seed}_{o}.npz',prediction=p,F0=b,target=y,scale=sc,origins=np.array(oo))
                assert before==G.tensor_hash(dict(m.named_parameters())) and sha(ROOT/c['checkpoint_file'])==c['checkpoint_sha256']
                rr=[r for r in rows if r['dataset']==ds and r['seed']==seed]
                native=float(np.mean([r['future_loss'] for r in rr]));f0mean=float(np.mean([r['F0_future_loss'] for r in rr]));variation=float(np.ptp([r['F0_correction_revision'] for r in rr]))
                states.append(dict(dataset=ds,seed=seed,mean_future_loss=native,mean_F0_future_loss=f0mean,FR_range=variation,
                    verdict='FR_REOPEN_POSSIBLE' if native<f0mean and variation>0 else 'FR_NO_ENTRY',
                    reason='Positive retrospective adaptation with nonzero varying correction revision' if native<f0mean and variation>0 else 'Positive entry not established on the fixed diagnostic pairs; not a universal FR refutation'))
                csv_write(out/'pair_diagnostics.csv',rows);save(states=len(states));del m,ps,x,g;gc.collect();torch.cuda.empty_cache()
        else:
            panel=Panel('m5');schedule=read(ROOT/'results/candidate_07/sampling_manifest.json');selected=read(ROOT/'results/candidate_07/selection.json')['winners']
            assert len(schedule)==360
            for f,h in panel.meta['files'].items():assert sha(panel.root/f)==h
            chosen=[r for r in selected if r['arm'] in ['CENSORED_LOSS_LORA','CENSOR_PRESERVE_LORA']]
            write_json(out/'manifest.json',dict(states=chosen,batches=360,scope='Saved V-selected states replayed on all 360 actual original train sampling batches. Synthetic censoring uses original train-derived capacity; this is not the gradient trajectory observed during historical training.',
                sampling_manifest_sha256=sha(ROOT/'results/candidate_07/sampling_manifest.json'),new_fits=0))
            for c in chosen:
                assert sha(c['checkpoint'])==c['checkpoint_sha256'];m,a=build(7,c['arm']);old_restore(torch.load(c['checkpoint'],map_location='cpu',weights_only=True),m,a)
                before=G.tensor_hash(dict(m.named_parameters()));distances=[];allloss=0.;badloss=0.;total=0;saturated=0;zero=0;both=0
                for i,r in enumerate(schedule):
                    if i%10==0:watch.resource('censor_train_replay')
                    x,y,g,_,sc,caps=old_batch(panel,r['origins'],[np.array(v) for v in r['channels']],7,'clean')
                    with torch.no_grad():forward+=1;p=predict(m,a,7,x,g,'clean')[1]
                    raw=p.detach().requires_grad_();sale=torch.minimum(y,caps[:,None]);mask=y>caps[:,None]
                    prob=cdf(raw,sale);loss=-(1-prob).clamp_min(1e-6).log();cdf_backward+=1
                    grad=torch.autograd.grad((loss*mask).sum(),raw)[0]
                    sat=prob>=1-1e-6;zg=grad.abs().sum(1)==0;joint=mask&sat&zg
                    sortedp=raw.detach().sort(dim=1).values;support=sortedp[:,-1]+(sortedp[:,-1]-sortedp[:,-2]).clamp_min(1e-4)
                    gap=(sale-support)[mask].detach().cpu().numpy();distances.extend(gap.tolist())
                    n=int(mask.sum());total+=n;saturated+=int((sat&mask).sum());zero+=int((zg&mask).sum());both+=int(joint.sum())
                    ll=float(loss[mask].detach().sum());bl=float(loss[joint].detach().sum());allloss+=ll;badloss+=bl
                    rows.append(dict(arm=c['arm'],batch=i,censored=n,saturated=int((sat&mask).sum()),zero_gradient=int((zg&mask).sum()),
                        saturated_zero_gradient=int(joint.sum()),censor_loss_sum=ll,saturated_zero_loss_sum=bl,lower_bound_minus_upper_support_mean=float(np.mean(gap)) if n else None))
                    del x,y,g,sc,caps,p,raw,prob,loss,grad,sortedp,support
                assert before==G.tensor_hash(dict(m.named_parameters())) and sha(c['checkpoint'])==c['checkpoint_sha256']
                states.append(dict(arm=c['arm'],checkpoint_step=c['step'],total_censored_positions=total,saturated_fraction=saturated/total,
                    zero_gradient_fraction=zero/total,saturated_zero_gradient_fraction=both/total,saturated_zero_loss_fraction=badloss/allloss,
                    lower_bound_minus_upper_support_quantiles={str(q):float(np.quantile(distances,q)) for q in [0,.25,.5,.75,.9,.99,1]},
                    interpretation='Report continuous fractions; no fabricated materiality cutoff. Snapshot limitation, not historical gradient trajectory.'))
                csv_write(out/'batch_diagnostics.csv',rows);save(states=len(states));del m,a;gc.collect();torch.cuda.empty_cache()
        write_json(out/'summary.json',dict(states=states))
        assert all(sha(ROOT/p)==h for p,h in history.items())
        save('COMPLETED',historical_files_unchanged=len(history),resources=guard(start),exit_code=0)
    except BaseException as e:save('EXECUTION_ERROR',error=repr(e),exit_code=1);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fr','censor']);main(p.parse_args().mode)
