"""Freeze all V choices, save all predictions, then permit E label scoring."""
import pandas as pd
from .common import *
from .prepare import STATES,SHAPES
from .train import predict
VSTATES=['REFERENCE','POINT8','BURST8','SHIFT4','SHIFT8']

def load_model(row,stage='selected'):
    model=build(row['arm'],row['seed'],row['source'])
    if row['arm']!='C0':
        r=checkpoint_row(row,stage);assert sha(ROOT/r['checkpoint'])==r['sha256'];restore(model,torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True))
    return model

def checkpoint_row(row,stage):
    if stage=='selected' or row['arm']=='C0':return row
    directory=OLD if row.get('reused') else OUT
    r=read(directory/'fits'/row['fit']/'receipt.json')
    return next(x for x in r['checkpoints'] if x['step']==1024)

def selected_key(source,arm,seed):return f'{source}_{arm}_{seed}'

def calibrate(watch):
    path=OUT/'CALIBRATION_SELECTION.json'
    if path.exists():return read(path)
    result={}
    rows=read(OUT/'MODEL_SELECTION.json')
    for source in SOURCES:
        seeds=[85551,85552,85553] if source=='ettm2' else [81551,81552,81553]
        f=CACHE/'conditions'/source;x=np.load(f/'V_SELECT_x.npy',mmap_mode='r');s=np.load(f/'V_SELECT_sigma.npy',mmap_mode='r');y=np.load(f/'V_SELECT_y.npy');n=len(y)//20
        idx=np.concatenate([np.arange(STATES.index(k)*2*n,(STATES.index(k)+1)*2*n) for k in VSTATES]);micro=read(OUT/'microbatch.json')[source]['microbatch']
        for seed in seeds:
            predictions={}
            for arm in ['C0','C2']:
                row=next(r for r in rows if (r['source'],r['arm'],r['seed'])==(source,arm,seed));model=load_model(row)
                predictions[arm]=predict(model,x,s,micro,watch).astype(float);del model;cleanup()
                p=CACHE/'V_predictions'/f'{source}_{arm}_{seed}.npy';p.parent.mkdir(exist_ok=True);np.save(p,predictions[arm])
            a,b=predictions['C0'],predictions['C2'];vals=[];metric_sigma=np.tile(read(OUT/'DATA_MANIFEST.json')[source]['sigma_train_population'],len(y)//4)
            for alpha in [0.,.25,.5,.75,1.]:
                p=a if alpha==0 else b if alpha==1 else a+alpha*(b-a)
                vals.append(dict(alpha=alpha,objective=float(np.mean(abs(p[idx,4]-y[idx])/metric_sigma[idx,None]))))
            alpha=min(vals,key=lambda v:(v['objective'],v['alpha']))['alpha']
            # Equal counts per condition imply equal-weight median equals ordinary pooled median.
            beta=float(np.median(((y[idx]-a[idx,4])/metric_sigma[idx,None]).reshape(-1)))
            result[f'{source}_{seed}']=dict(alpha=alpha,beta=beta,alpha_grid=vals,selection_conditions=VSTATES,condition_counts=[2*n*64]*5,optimizer_updates=0,origin_selection='original four source channels only',beta_definition='pooled equal-condition weighted median normalized residual; equal sample count')
    save(path,result);return result

def evaluation_seal():
    check_seal();rows=read(OUT/'MODEL_SELECTION.json');assert len(rows)==54
    journal=[json.loads(l) for l in open(OUT/'UPDATE_LEDGER.jsonl')];assert len(journal)==59392
    fits=[read(p) for p in (OUT/'fits').glob('*/receipt.json')];assert len(fits)==58 and all(r['status']=='COMPLETE' and r['updates']==1024 for r in fits)
    for r in rows:
        assert sha(ROOT/r['checkpoint'])==r['sha256']
        f=checkpoint_row(r,'fixed1024');assert sha(ROOT/f['checkpoint'])==f['sha256']
    if not (OUT/'GLOBAL_EVALUATION_SEAL.json').exists():
        save(OUT/'GLOBAL_EVALUATION_SEAL.json',dict(at=time.time(),model_selection_sha256=sha(OUT/'MODEL_SELECTION.json'),calibration_selection_sha256=sha(OUT/'CALIBRATION_SELECTION.json'),master_seal_sha256=sha(OUT/'MASTER_SEAL.json'),selected_model_count=54,main_updates=59392,smoke_updates=36,stage='all model/LR/checkpoint/alpha/beta fixed; E not newly scored'))
    else:
        d=read(OUT/'GLOBAL_EVALUATION_SEAL.json');assert d['model_selection_sha256']==sha(OUT/'MODEL_SELECTION.json') and d['calibration_selection_sha256']==sha(OUT/'CALIBRATION_SELECTION.json')

@torch.no_grad()
def profile(model,x,s,micro,watch):
    # Same 128 raw input series and FP32 batch per data panel. Three timed repeats; no updates.
    xx=x[:128];ss=s[:128];predict(model,xx,ss,micro,watch);seconds=[]
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        sync();t=time.perf_counter();predict(model,xx,ss,micro,watch);sync();seconds.append(time.perf_counter()-t)
    return dict(raw_seconds=seconds,median_seconds=float(np.median(seconds)),range_seconds=float(np.ptp(seconds)),profile_series=128,warmup_forwards_series=128,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),microbatch=micro,precision='FP32',forward_models=1)

def gate_hook(captured):
    def hook(module,args,out):
        e,g=args;cap=e.square().mean(-1).sqrt().quantile(.5,dim=-1,keepdim=True).clamp_min(1e-6).detach()[...,None]
        delta=cap*torch.tanh(module.up(torch.nn.functional.gelu(module.down(e))))
        vals=torch.stack([g,g*g,delta.square().mean(-1).sqrt(),(out-e).square().mean(-1).sqrt()],-1)
        captured.append(vals.detach().cpu().numpy())
    return hook

@torch.no_grad()
def embedding_stats(model,x,s,micro,watch):
    captured=[];hook=model.adapter.register_forward_hook(gate_hook(captured))
    from .model import mechanism_gate
    for lo in range(0,len(x),micro):
        watch.boundary();xx=torch.tensor(np.array(x[lo:lo+micro]),device='cuda');ss=torch.tensor(np.array(s[lo:lo+micro]),device='cuda');b=model.base
        z,_=b.instance_norm(xx);patch=b.patch(z.to(b.dtype));mask=b.patch(torch.ones_like(xx));e=b.input_patch_embedding(torch.cat([patch,mask],-1))
        g=torch.ones_like(patch[:,:,0]) if model.arm=='C2' else mechanism_gate(xx,ss,model.arm)
        model.adapter(e,g)
    hook.remove();a=np.concatenate(captured).reshape(10,2,128,-1,32,4)
    return {state:dict(mean_gate=float(a[i,...,0].mean()),mean_gate_square=float(a[i,...,1].mean()),mean_ungated_residual_rms=float(a[i,...,2].mean()),mean_applied_residual_rms=float(a[i,...,3].mean()),gate_by_patch=a[i,...,0].mean((0,1,2)).tolist(),applied_rms_by_patch=a[i,...,3].mean((0,1,2)).tolist()) for i,state in enumerate(STATES)}

def jobs():
    rows=read(OUT/'MODEL_SELECTION.json');result=[]
    for r in rows:
        panels=[r['source']]+(['electricity_transfer'] if r['source']=='electricity' else [])
        for panel in panels:
            for stage in ['selected','fixed1024']:
                # C0 is always the matched, V-selected B0. Fixed1024 refers only to additional adaptation.
                result.append(dict(row=r,panel=panel,stage=stage,kind='standard'))
            if r['arm'] in ['C0','C2','C3']:result.append(dict(row=r,panel=panel,stage='selected',kind='shape'))
    return result

def prediction_id(j):
    r=j['row'];return f"{j['panel']}__{j['kind']}__{j['stage']}__{r['arm']}__{r['seed']}"

def predict_all(watch):
    calibrate(watch);evaluation_seal()
    manifest=read(OUT/'PREDICTIONS_MANIFEST.json') if (OUT/'PREDICTIONS_MANIFEST.json').exists() else {}
    resources=read(OUT/'INFERENCE_RESOURCES.json') if (OUT/'INFERENCE_RESOURCES.json').exists() else {};gate_rows=read(OUT/'GATE_RESIDUAL_STATS.json') if (OUT/'GATE_RESIDUAL_STATS.json').exists() else {}
    previous=read(OLD/'predictions_manifest.json');plan=jobs();save(OUT/'PREDICTION_PLAN.json',[dict(id=prediction_id(j),panel=j['panel'],stage=j['stage'],kind=j['kind'],arm=j['row']['arm'],seed=j['row']['seed']) for j in plan])
    for j in plan:
        key=prediction_id(j);r=j['row'];panel=j['panel'];kind=j['kind'];stage=j['stage'];source=r['source'];arm=r['arm'];seed=r['seed']
        if key in manifest:assert sha(ROOT/manifest[key]['path'])==manifest[key]['sha256'];continue
        chosen=checkpoint_row(r,stage);f=CACHE/'conditions'/panel if kind=='standard' else CACHE/'shapes'/panel
        x=np.load(f/('E_DISCOVERY_x.npy' if kind=='standard' else 'x.npy'),mmap_mode='r');s=np.load(f/('E_DISCOVERY_sigma.npy' if kind=='standard' else 'sigma.npy'),mmap_mode='r')
        micro=read(OUT/'microbatch.json')[source]['microbatch'];cost={};reused=False
        selected_id=key.replace('__fixed1024__','__selected__')
        if stage=='fixed1024' and (arm=='C0' or r['sha256']==chosen['sha256']):
            receipt=manifest[selected_id];path=ROOT/receipt['path'];reused=True;cost=dict(reuse_of=selected_id,reason='identical checkpoint; no duplicate inference')
        elif stage=='selected' and kind=='standard' and panel==source and r.get('reused'):
            receipt=previous[selected_key(source,arm,seed)];path=ROOT/receipt['path'];assert sha(path)==receipt['sha256'];reused=True
            # Fresh common-size profiling and exact saved prediction replay, without full duplicate inference.
            model=load_model(r);cost=profile(model,x,s,micro,watch);p=predict(model,x[:micro],s[:micro],micro,watch)
            if model.adapter is not None:
                gate_rows[key]=embedding_stats(model,x,s,micro,watch);save(OUT/'GATE_RESIDUAL_STATS.json',gate_rows)
            assert np.array_equal(p,np.load(path,mmap_mode='r')[:micro]);del model;cleanup()
        else:
            model=load_model(r,stage);captured=[];hook=None
            if kind=='standard' and stage=='selected' and model.adapter is not None:hook=model.adapter.register_forward_hook(gate_hook(captured))
            torch.cuda.reset_peak_memory_stats();sync();t=time.perf_counter();p=predict(model,x,s,micro,watch);sync()
            cost=dict(full_prediction_seconds=time.perf_counter()-t,full_series=len(x),full_peak_allocated=torch.cuda.max_memory_allocated(),full_peak_reserved=torch.cuda.max_memory_reserved())
            if hook:
                hook.remove();a=np.concatenate(captured).reshape(10,2,128,-1,32,4)
                gate_rows[key]={state:dict(mean_gate=float(a[i,...,0].mean()),mean_gate_square=float(a[i,...,1].mean()),mean_ungated_residual_rms=float(a[i,...,2].mean()),mean_applied_residual_rms=float(a[i,...,3].mean()),gate_by_patch=a[i,...,0].mean((0,1,2)).tolist(),applied_rms_by_patch=a[i,...,3].mean((0,1,2)).tolist()) for i,state in enumerate(STATES)}
                save(OUT/'GATE_RESIDUAL_STATS.json',gate_rows)
            if kind=='standard' and stage=='selected':cost.update(profile(model,x,s,micro,watch))
            path=CACHE/'predictions'/f'{key}.npy';path.parent.mkdir(exist_ok=True);np.save(path,p);del model;cleanup()
        manifest[key]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),source=source,panel=panel,arm=arm,seed=seed,stage=stage,kind=kind,checkpoint_sha256=chosen['sha256'],shape=list(np.load(path,mmap_mode='r').shape),reused=reused)
        resources[key]=cost;save(OUT/'PREDICTIONS_MANIFEST.json',manifest);save(OUT/'INFERENCE_RESOURCES.json',resources)
        print('PREDICTION_SAVED',key,flush=True)
    # Linear output controls use already-saved paired selected predictions, V choices only.
    calibration=read(OUT/'CALIBRATION_SELECTION.json')
    for key,r in list(manifest.items()):
        if r['arm']!='C0' or r['stage']!='selected' or r['kind']!='standard':continue
        c2key=key.replace('__C0__','__C2__');a=np.load(ROOT/r['path']).astype(float);b=np.load(ROOT/manifest[c2key]['path']).astype(float);c=calibration[f"{r['source']}_{r['seed']}"]
        sig=np.load(CACHE/'conditions'/r['panel']/'E_DISCOVERY_sigma.npy')
        for arm in ['C2_SHRINK','C0_BIAS']:
            newkey=key.replace('__C0__','__'+arm+'__')
            if newkey in manifest:assert sha(ROOT/manifest[newkey]['path'])==manifest[newkey]['sha256'];continue
            p=(a if c['alpha']==0 else b if c['alpha']==1 else a+c['alpha']*(b-a)) if arm=='C2_SHRINK' else a+c['beta']*sig[:,None,None]
            path=CACHE/'predictions'/f'{newkey}.npy';np.save(path,p)
            manifest[newkey]=dict(r,path=str(path.relative_to(ROOT)),sha256=sha(path),arm=arm,reused=False,calibration=c,components=[key,c2key] if arm=='C2_SHRINK' else [key])
            resources[newkey]=dict(forward_models=2 if arm=='C2_SHRINK' else 1,profile_series=128,median_seconds=sum(resources[k]['median_seconds'] for k in ([key,c2key] if arm=='C2_SHRINK' else [key])),meaning='sum of actual component inference profiles; blend CPU overhead not included, no one-forward claim')
    # Exact legacy SHIFT8 past/draw matching for PULSE, additional to balanced +/- shape panel.
    ids=np.linspace(0,127,64,dtype=int)
    for key,r in list(manifest.items()):
        if r['arm'] not in ['C0','C2','C3'] or r['stage']!='selected' or r['kind']!='standard':continue
        newkey=key.replace('__standard__','__legacy_pulse__')
        if newkey in manifest:continue
        nc=len(read(OUT/'DATA_MANIFEST.json')[r['panel']]['selected_columns'])
        p=np.load(ROOT/r['path'],mmap_mode='r').reshape(10,2,128,nc,9,64)[8][:,ids].reshape(-1,9,64)
        path=CACHE/'predictions'/f'{newkey}.npy';np.save(path,p)
        manifest[newkey]=dict(r,path=str(path.relative_to(ROOT)),sha256=sha(path),kind='legacy_pulse',shape=list(p.shape),reused=True,exact_input_prediction_reuse_of=key)
    assert len(manifest)==246 # 150 standard views +36 balanced shape +24 output controls +36 exact legacy pulse views.
    save(OUT/'PREDICTIONS_MANIFEST.json',manifest);save(OUT/'INFERENCE_RESOURCES.json',resources)
    save(OUT/'ALL_PREDICTIONS_SAVED.json',dict(count=len(manifest),at=time.time(),manifest_sha256=sha(OUT/'PREDICTIONS_MANIFEST.json'),new_E_labels_scored=False))


def score():
    gate=read(OUT/'ALL_PREDICTIONS_SAVED.json');assert sha(OUT/'PREDICTIONS_MANIFEST.json')==gate['manifest_sha256'];evaluation_seal()
    rows=[];channel={}
    for key,r in read(OUT/'PREDICTIONS_MANIFEST.json').items():
        panel=r['panel'];data=read(OUT/'DATA_MANIFEST.json')[panel];nc=len(data['selected_columns']);d=np.load(CACHE/'data'/panel/'E_DISCOVERY_inputs.npz');base_y=np.load(CACHE/'data'/panel/'E_DISCOVERY_labels.npz')['y'];p=np.load(ROOT/r['path'],mmap_mode='r');assert sha(ROOT/r['path'])==r['sha256']
        if r['kind']=='standard':
            names=STATES;origins=d['origins'];offset=np.load(CACHE/'conditions'/panel/'E_DISCOVERY_offset.npy');y=np.tile(base_y.reshape(-1,64),(20,1))+offset[:,None]
        elif r['kind']=='shape':
            meta=read(CACHE/'shapes'/panel/'manifest.json');names=meta['states'];ids=meta['origin_indices'];origins=d['origins'][ids];offset=np.load(CACHE/'shapes'/panel/'offset.npy');y=np.tile(base_y[ids].reshape(-1,64),(len(names)*2,1))+offset[:,None]
        else:
            names=['PULSE_LEGACY_MATCH'];ids=np.linspace(0,127,64,dtype=int);origins=d['origins'][ids];y=np.tile(base_y[ids].reshape(-1,64),(2,1))
        sigma=np.tile(data['sigma_train_population'],len(y)//nc);norigin=len(origins)
        history=np.load(CACHE/'conditions'/panel/'E_history_subset.npy').reshape(128,nc) if panel in ['electricity','ettm1'] and r['kind']=='standard' else np.zeros((norigin,nc),dtype=bool)
        # Metric work in chunks avoids loading whole 16-channel quantile tensor twice.
        metrics=[]
        for lo in range(0,len(y),512):
            pp=p[lo:lo+512].astype(float);yy=y[lo:lo+512];ss=sigma[lo:lo+512];valid=np.isfinite(yy);assert valid.any(1).all()
            e=yy-pp[:,4];q=np.arange(1,10)[None,:,None]/10;qe=yy[:,None,:]-pp
            metrics.append(np.stack([np.nanmean(abs(e)/ss[:,None],1),np.nanmean(abs(e),1),np.nanmean(e*e/ss[:,None]**2,1),np.nanmean(2*np.maximum(q*qe,(q-1)*qe)/ss[:,None,None],(1,2)),(np.diff(pp,axis=1)<0).mean((1,2))],-1))
        m=np.concatenate(metrics).reshape(len(names),2,norigin,nc,5).mean(1)
        for ci,name in enumerate(names):
            for oi,origin in enumerate(origins):
                for ch,cid in enumerate(data['selected_columns']):rows.append(dict(prediction=key,panel=panel,source=r['source'],kind=r['kind'],stage=r['stage'],arm=r['arm'],seed=r['seed'],condition=name,origin=int(origin),channel=str(cid),history_subset=bool(history[oi,ch]),nmae=float(m[ci,oi,ch,0]),mae=float(m[ci,oi,ch,1]),normalized_mse=float(m[ci,oi,ch,2]),pinball=float(m[ci,oi,ch,3]),crossing=float(m[ci,oi,ch,4])))
        print('SCORED',key,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'SCORES_BY_ORIGIN.csv.gz',index=False,compression='gzip')
    keys=['panel','source','kind','stage','arm','seed','condition']
    grouped=frame.groupby(keys)[['nmae','mae','pinball','crossing']].mean();rmse=frame.groupby(keys+['channel']).normalized_mse.mean().pow(.5).groupby(level=list(range(7))).mean()
    grouped.join(rmse.rename('nrmse')).reset_index().to_csv(OUT/'RAW_SCORES.csv',index=False)
    from .statistics import analyze_effects
    analyze_effects(frame)
