import gc,json,time
import numpy as np
import torch
from ..backbone import load_base,forecast,native_loss
from ..lora import attach,snapshot,restore,audit
from ..data import Panel
from ..metrics import score,replay
from ..selection import seal
from ..reproducibility import ROOT,seed_all,guard,write_json,sha,source_hashes
from ..candidates.maturity import matured_mask,preservation,Calibration
from .common_fit import csv_write
ARMS=['F0','IMMEDIATE_LORA','WAIT_FULL','TAFAS_LIKE','MATURITY_PEFT']

def stream_forecast(m,cal,x,groups):
    if cal is None:return forecast(m,x,groups)
    x=torch.cat([cal.input(chunk) for chunk in x.split(4)])
    z,p,l,s=forecast(m,x,groups)
    p=torch.cat([cal.output(chunk) for chunk in p.split(4)])
    z=((p-l[:,None,:])/s[:,None,:]).asinh()
    return z,p,l,s

def run(out=None,cache=None):
    out=ROOT/'results/candidate_05' if out is None else out
    cache=ROOT/'.cache/candidate_05' if cache is None else cache
    if (out/'selection.json').exists() or (out/'status.json').exists():
        raise FileExistsError('Existing stream results are immutable')
    out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    panel=Panel('jena');start=time.monotonic()
    contract=dict(candidate=5,dataset='jena',manifest_hash=sha(panel.root/'manifest.json'),source_hashes=source_hashes(),arms=ARMS,origins=panel.origins['evaluation'].tolist(),lr=1e-4,lambda_preserve=1.,updates_per_origin=8,seed=30000,order='issue immutable forecast; update only labels at indices < now; advance24',supervision='latest partial and latest complete issued forecast; WAIT_FULL only latest complete; no retroactive forecast replacement',TAFAS_scope='GCM input/output equation3; fixed24 schedule replaces PAAS; issued forecasts never replaced; native probabilistic loss replaces MSE',exception='WAIT_FULL necessarily has fewer available training examples in first two origins; same 8-update opportunities once eligible',selection='single predetermined recipe; no E tuning')
    write_json(out/'contract.json',contract);seal(out/'selection.json',[dict(arm=a,lr=1e-4 if a!='F0' else 0,step='online',validation_loss=None) for a in ARMS],contract)
    panel.open_e(out/'selection.json',contract);write_json(out/'evaluation_open.json',dict(selection_sha256=sha(out/'selection.json')))
    rows=[];trajectories=[];usage=[];integrity=[];per_origin={}
    attempts=[]
    for arm in ARMS:
        attempts.append(dict(arm=arm,status='RUNNING'));write_json(out/'attempts.json',attempts)
        seed_all(30000);guard();torch.cuda.reset_peak_memory_stats();job_start=time.monotonic();m=load_base();cal=None
        if arm=='TAFAS_LIKE':cal=Calibration(4).cuda()
        elif arm!='F0':attach(m)
        params=[p for model in [m,cal] if model is not None for p in model.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=0) if params else None
        issued=[];issued_hashes=[];updates=0;adapt_seconds=0.;drifts=[];p_losses=[]
        origins=panel.origins['evaluation'];scale=torch.tensor(panel.scale,dtype=torch.float32,device='cuda')
        identity=None;preserve_active=0
        for i,now in enumerate(origins):
            guard(job_start);x_np,_=panel.window(int(now),target=False);x=torch.tensor(x_np,device='cuda');groups=torch.zeros(4,device='cuda',dtype=torch.long)
            with torch.no_grad():z,p,l,s=stream_forecast(m,cal,x,groups)
            if i==0:
                with torch.no_grad():f0=forecast(m,x,groups)[1]
                identity=float((p-f0).abs().max());assert identity<=1e-6
            issued.append(p.cpu().numpy());file=cache/f'issued_{arm}_{i:02}.npz'
            np.savez_compressed(file,prediction=issued[-1],origin=int(now));issued_hashes.append((file,sha(file)))
            if arm=='F0' or i==0:continue
            past=[i-1]+([i-2] if i>=2 else [])
            if arm=='WAIT_FULL':past=[j for j in past if origins[j]+48<=now]
            if not past:continue
            xs=[];ys=[];masks=[]
            for j in past:
                previous=int(origins[j]);xx,_=panel.window(previous,target=False);mask=matured_mask(previous,int(now))
                # Read only matured slice; unrevealed labels never enter tensors.
                yy=np.full((4,48),np.nan,dtype=np.float32);known=int(mask.sum());yy[:,:known]=panel.values[previous:previous+known].T
                assert previous+known<=now
                xs.append(xx);ys.append(yy);masks.append(mask.to('cuda'))
            xx=torch.tensor(np.concatenate(xs),device='cuda');yy=torch.tensor(np.concatenate(ys),device='cuda');gg=torch.arange(len(past),device='cuda').repeat_interleave(4)
            # Pre-model anchor is required by proposed method; for other arms
            # this extra forward is diagnostic-only and excluded from adapt time.
            torch.cuda.synchronize();anchor_start=time.monotonic()
            with torch.no_grad():pre=stream_forecast(m,cal,xx,gg)[1].detach()
            torch.cuda.synchronize();anchor_seconds=time.monotonic()-anchor_start
            torch.cuda.synchronize();update_start=time.monotonic()
            for k in range(8):
                opt.zero_grad(set_to_none=True);zz,pp,ll,ss=stream_forecast(m,cal,xx,gg);loss=native_loss(zz,yy,ll,ss)
                reg=pp.sum()*0
                if arm=='MATURITY_PEFT':
                    regs=[preservation(pp[j*4:(j+1)*4],pre[j*4:(j+1)*4],mask,scale) for j,mask in enumerate(masks)]
                    reg=torch.stack(regs).mean();loss=loss+reg
                    preserve_active+=int(float(reg.detach())>0)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite streaming loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise FloatingPointError('Nonfinite stream gradient')
                torch.nn.utils.clip_grad_norm_(params,1);opt.step();updates+=1
            torch.cuda.synchronize();adapt_seconds+=time.monotonic()-update_start+(anchor_seconds if arm=='MATURITY_PEFT' else 0)
            with torch.no_grad():post=stream_forecast(m,cal,xx,gg)[1]
            values=[]
            for j,mask in enumerate(masks):
                if (~mask).any():values.append(float(abs((post[j*4:(j+1)*4]-pre[j*4:(j+1)*4])/scale[:,None,None])[... ,~mask].mean()))
            drift=float(np.mean(values)) if values else 0.;drifts.append(drift)
            trajectories.append(dict(arm=arm,origin=int(now),issued_before_update=True,available_label_end=int(now)-1,updates=updates,supervised_loss=float(loss.detach()-reg.detach()),preservation=float(reg.detach()),unrevealed_drift=drift))
            print('STREAM',arm,i,updates,float(loss.detach()),flush=True)
        assert all(sha(p)==h for p,h in issued_hashes),'Issued forecast changed retrospectively'
        if arm=='MATURITY_PEFT':assert preserve_active>0,'Preservation never active'
        final=snapshot(m,cal);torch.save(final,cache/f'{arm}_final.pt')
        with torch.no_grad():expected=stream_forecast(m,cal,x,groups)[1]
        restore(torch.load(cache/f'{arm}_final.pt',weights_only=True),m,cal)
        with torch.no_grad():actual=stream_forecast(m,cal,x,groups)[1]
        err=float((expected-actual).abs().max());assert err==0
        p=np.stack(issued);y=np.stack([panel.window(int(o))[1] for o in origins]);path=cache/f'E_{arm}.npz';np.savez_compressed(path,prediction=p,target=y,scale=panel.scale);metric_error=replay(path)
        metrics=score(p,y,panel.scale);origin_losses=[score(p[j:j+1],y[j:j+1],panel.scale)['scaled_2pinball'] for j in range(30)];per_origin[arm]=origin_losses
        rows.append(dict(arm=arm,variant='issued',**metrics,unrevealed_drift=float(np.mean(drifts)) if drifts else 0,worst5_origin_loss=float(np.mean(sorted(origin_losses)[-5:])),adaptation_seconds=adapt_seconds))
        usage.append(dict(arm=arm,wall_seconds=time.monotonic()-job_start,adaptation_seconds=adapt_seconds,optimizer_steps=updates,**guard()))
        integrity.append(dict(arm=arm,identity_max_abs=identity,checkpoint_replay_max_abs=err,metric_replay_abs=metric_error,issued_forecast_hashes_verified=30,preservation_active_updates=preserve_active,no_future_labels=True,trainable_count=sum(p.numel() for p in params)))
        # Persist completed-arm receipts before starting the next arm.
        csv_write(out/'metrics.csv',rows);csv_write(out/'trajectories.csv',trajectories)
        write_json(out/'origin_losses.json',per_origin)
        write_json(out/'completed_integrity.json',integrity)
        write_json(out/'completed_resources.json',usage)
        attempts[-1]['status']='COMPLETE';write_json(out/'attempts.json',attempts)
        del m,cal,opt,params,final;gc.collect();torch.cuda.empty_cache()
    csv_write(out/'metrics.csv',rows);csv_write(out/'trajectories.csv',trajectories);csv_write(out/'selections.csv',[dict(arm=a,lr=1e-4,recipe='fixed') for a in ARMS]);write_json(out/'origin_losses.json',per_origin)
    by={row['arm']:row for row in rows};proposed=by['MATURITY_PEFT'];best=min(['IMMEDIATE_LORA','WAIT_FULL','TAFAS_LIKE'],key=lambda a:by[a]['scaled_2pinball']);f0=by['F0']['scaled_2pinball'];gain=100*(by[best]['scaled_2pinball']-proposed['scaled_2pinball'])/f0
    overhead=100*(proposed['adaptation_seconds']/max(by[best]['adaptation_seconds'],1e-9)-1)
    early=100*float(np.mean(np.array(per_origin[best][:25])-np.array(per_origin['MATURITY_PEFT'][:25])))/f0
    passed=gain>=1 and overhead<=20 and early>0
    write_json(out/'status.json',dict(verdict='PASS' if passed else 'WEAK' if gain>0 else 'FAIL',problem_gate='PASS',novelty_collision=False,fit_count=0,stream_count=5,strongest_baseline=best,proposed='MATURITY_PEFT',proposed_primary=proposed['scaled_2pinball'],baseline_primary=by[best]['scaled_2pinball'],f0_primary=f0,gain_percent_f0=gain,diagnostics=dict(compute_overhead_percent=overhead,gain_excluding_last5_percent_f0=early,unrevealed_drift=proposed['unrevealed_drift'],worst5_origin_loss=proposed['worst5_origin_loss']),round2_recommended=passed))
    write_json(out/'integrity.json',dict(status='PASS',streams=integrity,selection_seal_before_e=True,metric_replay_max_abs=max(v['metric_replay_abs'] for v in integrity)));write_json(out/'resource_usage.json',dict(wall_seconds=time.monotonic()-start,streams=usage,fit_count=0,stream_count=5))
