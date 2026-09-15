"""Independent 24-attempt channel-sharing pilot; fixed data, seeds and update budget."""
import argparse,copy,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from priority12.common import ROOT,read,save,sha,digest,csvwrite,parameters,cpu_state,tensor_hash,frozen_hash,restore,preserve_rng,Watch,ResourceError,cleanup
from priority12.channel_model import ARMS,ChannelBlock,channel_count,groups,make_model,inventory,up
from priority12.channel_data import load,batch,metrics,independent
OUT=ROOT/'results/channel_basis_pilot_20260915';CACHE=ROOT/'.cache/channel_basis_pilot_20260915';COMBINED=ROOT/'results/priority12_20260915'
DATASETS=['electricity','traffic'];SEEDS=[40000,40001];CHECKPOINTS=[0,128,256,512,1024]
def write(name,x):save(OUT/name,x)
def configure():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
def source_hashes():
    files=[Path(__file__).resolve(),ROOT/'scripts/finalize_channel_basis_pilot.py',ROOT/'tests/test_priority12_channel.py',ROOT/'tests/test_priority12_channel_data.py',ROOT/'docs/PEFT_PRIORITY12_PROTOCOL_20260915.md',ROOT/'sources/RELATED_WORK.md',ROOT/'configs/channel_environment_overrides.txt']
    files+=list((ROOT/'scripts/priority12').glob('channel_*.py'))+[ROOT/'scripts/priority12/common.py']
    files+=[p for p in (ROOT/'sources/timepeft_ea4e7e1').rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}
def contract_check():
    c=read(OUT/'contract.json')
    for p,h in c['source_hashes'].items():assert sha(ROOT/p)==h,p
    for d in c['data'].values():
        assert sha(d['raw_path'])==d['raw_sha256']
        for p,h in d['staged'].items():assert sha(ROOT/p)==h,p
    for p,h in c['model_files'].items():assert sha(p)==h
    for p,h in c['historical_hashes'].items():assert sha(ROOT/p)==h,p
    assert sha(OUT/'schedules.json')==c['schedule_hash'] and sha(OUT/'fit_manifest.json')==c['fit_manifest_hash']
    return c

def prepare():
    assert not OUT.exists(),'No overwrite/retry';OUT.mkdir();data=read(COMBINED/'channel_data_manifest.json')
    (OUT/'PROTOCOL.md').write_bytes((ROOT/'docs/PEFT_PRIORITY12_PROTOCOL_20260915.md').read_bytes())
    t=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_priority12_channel.py','tests/test_priority12_channel_data.py'],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    write('cpu_tests.json',dict(exit_code=t.returncode,stdout=t.stdout,stderr=t.stderr));assert t.returncode==0,t.stdout+t.stderr
    schedules={f'{d}_{s}':np.random.default_rng(s).choice(data[d]['origins']['train'],(1024,2)).tolist() for d in DATASETS for s in SEEDS};write('schedules.json',schedules)
    cells=[dict(dataset=d,seed=s,arm=a) for d in DATASETS for s in SEEDS for a in ARMS];order=np.random.default_rng(60020).permutation(24);cells=[dict(attempt_order=i,**cells[int(j)]) for i,j in enumerate(order)];write('fit_manifest.json',cells)
    counts=[]
    for c in [16,32,64,128]:
        for arm in ARMS[1:]:
            m=ChannelBlock(arm,list(range(c)));actual=sum(p.numel() for p in m.parameters());assert actual==channel_count(arm,c=c);counts.append(dict(channels=c,arm=arm,channel_parameters=actual));del m
    csvwrite(OUT/'constructor_counts.csv',counts)
    inv=[];initial=[];masters={}
    # Actual pinned model CPU eval. No optimizer updates, no V/E scoring.
    values,_,_=load(CACHE,'electricity','train');x,y=batch(values,[data['electricity']['origins']['train'][0]],device='cpu')
    for seed in SEEDS:
        outputs={};common={}
        for arm in ARMS:
            m=make_model(arm,data['electricity']['channel_ids'],seed,False,'cpu').eval();row=inventory(m);inv.append(dict(seed=seed,**row))
            st=cpu_state(m);head_lora={n:p for n,p in st.items() if n.startswith('head.') or 'lora_' in n};h=tensor_hash(head_lora)
            if common:assert h==common['head_lora']
            else:common['head_lora']=h
            if arm!='LORA_HEAD':
                freq=tensor_hash({n:p for n,p in st.items() if n.startswith('frequency_adapter.')})
                if 'frequency' in common:assert freq==common['frequency']
                common['frequency']=freq
            with torch.no_grad():pred=m(x).forecast;assert pred.shape==(1,64,96) and torch.isfinite(pred).all();outputs[arm]=pred.clone()
            if arm=='SPECIFIC':
                official=up.ChannelAdapter(512,64,2);official.load_state_dict({k:v for k,v in m.channel_adapter.state_dict().items() if k!='group_indices'})
                reference=up.TimePEFTPipeline(m,copy.deepcopy(m.frequency_adapter),official).eval()
                with torch.no_grad():p=reference(x).forecast
                torch.testing.assert_close(p,pred,rtol=1e-5,atol=1e-5)
                loss=(p-y).square().mean();ourloss=(pred-y).square().mean();torch.testing.assert_close(loss,ourloss,rtol=1e-5,atol=1e-5)
                initial.append(dict(seed=seed,kind='upstream_specific',max_abs=float((p-pred).abs().max()),loss_abs=float((loss-ourloss).abs()),passed=True));del reference,official,p
            del m,st;import gc;gc.collect()
        for a in ARMS[1:]:
            torch.testing.assert_close(outputs[a],outputs['SPECIFIC'],rtol=1e-5,atol=1e-5);initial.append(dict(seed=seed,kind='common_initial_function',arm=a,max_abs=float((outputs[a]-outputs['SPECIFIC']).abs().max()),passed=True))
        masters[str(seed)]=common
    write('parameter_inventory.json',inv);write('initial_cpu_parity.json',initial);write('master_initial_hashes.json',masters)
    csvwrite(OUT/'parameter_counts.csv',[dict(seed=r['seed'],arm=r['arm'],trainable=r['trainable'],model_total=r['total'],**r['groups']) for r in inv])
    receipt=read(COMBINED/'moment_source.json');model_files={str(p):sha(p) for p in Path(receipt['snapshot']).iterdir() if p.is_file()}
    c=dict(base_commit='9051d03cebe23ae35d9689e7af576c63c4d5edd5',prepare_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes=source_hashes(),data=data,model=receipt,model_files=model_files,historical_hashes=read(COMBINED/'historical_hashes.json'),schedule_hash=sha(OUT/'schedules.json'),fit_manifest_hash=sha(OUT/'fit_manifest.json'),
        config=dict(arms=ARMS,seeds=SEEDS,context=512,horizon=96,channels=64,updates_per_fit=1024,fit_attempt_cap=24,check_update_cap=72,checkpoints=CHECKPOINTS,lr=.001,lr_after_update640=.0005,weight_decay=.01,betas=[.9,.999],eps=1e-8,clip=1,precision='FP32',group_seed=60010,fit_order_seed=60020,bootstrap_seed=60100,bootstrap_resamples=1000,bootstrap_block_size=4),
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,lock_sha256=sha(COMBINED/'channel_environment.lock'),matmul=torch.get_float32_matmul_precision(),TF32=False),
        resource_strategy_order=[dict(micro=2,checkpoint=False),dict(micro=1,checkpoint=False),dict(micro=1,checkpoint=True)],
        evaluation_scope='Previously used sources; train-only new channel selection. Development comparison, no independent confirmation',
        upstream_discrepancies=['common initial function','best-V state saved and restored','FP32 for all arms','fixed update LR schedule','explicit common encoder checkpoint policy','transformers metadata override to public requirements'],
        initial_prediction='Random forecasting head diagnostics; not pretrained F0 forecasts')
    write('contract.json',c);write('status.json',dict(status='PREPARED',cpu_toy_optimizer_updates=15,check_update_attempts=0,check_updates=0,fit_attempts=0,fits_completed=0,training_updates=0));write('leakage_checks.json',dict(train_only_selection=True,train_only_scaler=True,split_assertions_passed=True,group_assignment={d:groups(data[d]['channel_ids']) for d in DATASETS},E_scoring=0))
    print('C PREPARED CPU checks, upstream parity, 24 fixed cells',flush=True)

def optimizer(m):return torch.optim.AdamW(parameters(m).values(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.01)
def step(m,opt,values,oo,micro,w,applied):
    start=w.before();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();opt.zero_grad(set_to_none=True);lossval=0.
    for i in range(0,2,micro):
        x,y=batch(values,oo[i:i+micro]);assert torch.isfinite(y).all()
        p=m(x).forecast;assert p.shape==y.shape and torch.isfinite(p).all()
        loss=(p-y).square().mean()*(micro/2);assert torch.isfinite(loss);loss.backward();lossval+=float(loss.detach());del x,y,p,loss
    ps=list(parameters(m).values());assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps)
    norm=torch.nn.utils.clip_grad_norm_(ps,1.,error_if_nonfinite=True);opt.step();applied();assert all(torch.isfinite(p).all() for p in ps)
    torch.cuda.synchronize();seconds=time.perf_counter()-tick;end,contaminated=w.after(start)
    return dict(seconds=seconds,loss=lossval,gradient_norm=float(norm),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),free_mib=min(start['free_mib'],end['free_mib']),nvml_used_mib=end['used_mib'],external_compute_contaminated=contaminated,lr=opt.param_groups[0]['lr'])

def preflight():
    state=read(OUT/'status.json');assert state['status']=='PREPARED','No preflight retry';c=contract_check()
    qs=read(ROOT/'results/query_budget_numeric_v2_20260915/status.json')
    assert qs['status'] not in ['PREPARED','NUMERIC_WAITING_FOR_GPU','A_RUNNING','B_TRAINING','NUMERICS_BOUNDED','RESOURCE_SIGNAL'],'Q must terminate first'
    if any(s in qs.get('error','').lower() for s in ['illegal memory','device-side assert','data corruption']):
        state['status']='BLOCKED_COMMON_SAFETY';write('status.json',state);return
    w=Watch(OUT,'preflight');records=[];strategies=[]
    try:
        state['status']='PREFLIGHT_WAITING_FOR_GPU';write('status.json',state);w.boundary(startup=True)
        for strategy in c['resource_strategy_order']:
            success=True;current=[]
            for d in DATASETS:
                if not success:break
                values,_,_=load(CACHE,d,'train')
                for arm in ARMS:
                    w.boundary();m=opt=None
                    try:
                        m=make_model(arm,c['data'][d]['channel_ids'],40000,strategy['checkpoint'],'cuda');opt=optimizer(m);frozen=frozen_hash(m);initial=cpu_state(m);m.train()
                        for j in range(2):
                            assert state['check_update_attempts']+state.get('cpu_toy_optimizer_updates',0)<72;state['check_update_attempts']+=1;write('status.json',state)
                            def applied():state['check_updates']+=1
                            r=step(m,opt,values,c['data'][d]['origins']['train'][j:j+2],strategy['micro'],w,applied);records.append(dict(dataset=d,arm=arm,update=j,**strategy,**r));write('preflight_measurements.json',records)
                        assert frozen_hash(m)==frozen;assert tensor_hash(cpu_state(m))!=tensor_hash(initial)
                        active={}
                        if arm=='BASIS4':
                            active=dict(extra_basis_gradient=float(m.channel_adapter.up_projections[1].weight.grad.abs().sum()),coefficient_update=float((m.channel_adapter.coefficients.detach().cpu()-initial['channel_adapter.coefficients']).abs().sum()));assert min(active.values())>0
                        if arm=='SHARED_WIDE':
                            active=dict(extra_down_gradient=float(m.channel_adapter.down_projection.weight.grad[256:].abs().sum()));assert active['extra_down_gradient']>0
                        current.append(dict(dataset=d,arm=arm,status='PASS',frozen_unchanged=True,active_paths=active));print('C PREFLIGHT',d,arm,strategy,flush=True)
                    except torch.cuda.OutOfMemoryError as e:
                        current.append(dict(dataset=d,arm=arm,status='OOM',error=str(e)));success=False;break
                    finally:
                        m=opt=None;cleanup();write('status.json',state)
            strategies.append(dict(strategy=strategy,success=success,cells=current));write('preflight_strategies.json',strategies)
            if success:
                seal=dict(strategy=strategy,contract_hash=sha(OUT/'contract.json'),measurement_hash=sha(OUT/'preflight_measurements.json'),sealed_at=time.time(),no_forecast_selection=True);seal['seal_hash']=digest(seal);write('resource_strategy_seal.json',seal);state['status']='PREFLIGHT_PASSED';break
        else:state['status']='BLOCKED_RESOURCE_C'
    except BaseException as e:state.update(status='BLOCKED_GPU_BUSY' if 'BLOCKED_GPU_BUSY' in str(e) else 'INCONCLUSIVE_PREFLIGHT',error=repr(e),traceback=traceback.format_exc())
    finally:write('status.json',state);write('preflight_strategies.json',strategies);w.close();print('C PREFLIGHT END',state['status'],flush=True)

def evaluate(m,values,std,oo,tag,w):
    assert not (CACHE/(tag+'.npz')).exists(),'No prediction overwrite';pp=[];yy=[];peaks=[];started=time.monotonic();mode=m.training
    try:
        with preserve_rng(),torch.no_grad():
            m.eval()
            for o in oo:
                w.boundary();torch.cuda.reset_peak_memory_stats();x,y=batch(values,[o]);p=m(x).forecast;assert torch.isfinite(p).all();pp.append(p.cpu().numpy()[0]);yy.append(y.cpu().numpy()[0]);peaks.append(torch.cuda.max_memory_allocated())
    finally:m.train(mode)
    p=np.stack(pp);y=np.stack(yy);path=CACHE/(tag+'.npz');np.savez_compressed(path,prediction=p,target=y,std=std,origins=np.array(oo));mm=metrics(p,y,std)
    return dict(metrics=mm,prediction_path=str(path.relative_to(ROOT)),prediction_sha256=sha(path),wall_seconds=time.monotonic()-started,peak_allocated=max(peaks))

def train_evaluate():
    state=read(OUT/'status.json');assert state['status']=='PREFLIGHT_PASSED' and state['fit_attempts']==0
    c=contract_check();rs=read(OUT/'resource_strategy_seal.json');assert digest({k:v for k,v in rs.items() if k!='seal_hash'})==rs['seal_hash'];strategy=rs['strategy'];schedules=read(OUT/'schedules.json');w=Watch(OUT,'training');fits=[];trajectory=[];evaluation=[]
    try:
        w.boundary(startup=True)
        for cell in read(OUT/'fit_manifest.json'):
            d,a,seed=cell['dataset'],cell['arm'],cell['seed'];fid=f"{cell['attempt_order']:02}_{d}_{seed}_{a}";state['fit_attempts']+=1;assert state['fit_attempts']<=24
            fit=dict(fit=fid,**cell,status='RUNNING',updates=0);fits.append(fit);state.update(status='TRAINING',current_fit=fid);write('fits.json',fits);write('status.json',state)
            started=time.monotonic();records=[];m=opt=None;best=None;fr=[]
            try:
                w.boundary();torch.cuda.reset_peak_memory_stats();loadtick=time.monotonic();m=make_model(a,c['data'][d]['channel_ids'],seed,strategy['checkpoint'],'cuda');opt=optimizer(m);loading=time.monotonic()-loadtick;loadpeak=torch.cuda.max_memory_allocated();frozen=frozen_hash(m);initial=cpu_state(m)
                values,_,std=load(CACHE,d,'development');m.train()
                def checkpoint(stepno):
                    nonlocal best
                    r=evaluate(m,values,std,c['data'][d]['origins']['validation'],f'V_{fid}_{stepno}',w)
                    p=CACHE/f'{fid}_step{stepno}.pt';torch.save(cpu_state(m),p)
                    row=dict(fit=fid,dataset=d,arm=a,seed=seed,step=stepno,active_seconds=sum(r['seconds'] for r in records),checkpoint_path=str(p.relative_to(ROOT)),checkpoint_sha256=sha(p),**r);fr.append(row);trajectory.append(row);write('trajectories.json',trajectory)
                    if best is None or (r['metrics']['mse'],stepno)<(best['metrics']['mse'],best['step']):best=row
                    print('C V',fid,stepno,r['metrics']['mse'],flush=True)
                checkpoint(0)
                for update in range(1,1025):
                    opt.param_groups[0]['lr']=.001 if update<=640 else .0005
                    def applied():fit['updates']+=1;state['training_updates']+=1
                    r=step(m,opt,values,schedules[f'{d}_{seed}'][update-1],strategy['micro'],w,applied);records.append(dict(update=update,**r))
                    if update in CHECKPOINTS:checkpoint(update)
                    if update%32==0:write('status.json',state);write('fits.json',fits)
                assert frozen_hash(m)==frozen
                assert fit['updates']==1024 and state['training_updates']<=24576
                final=cpu_state(m);active={n:float((final[n]-initial[n]).double().norm()) for n in final};write(fid+'_parameter_updates.json',active)
                restore(m,torch.load(ROOT/best['checkpoint_path'],map_location='cpu',weights_only=True));replay=evaluate(m,values,std,c['data'][d]['origins']['validation'],f'RELOAD_{fid}',w)
                with np.load(ROOT/best['prediction_path']) as z,np.load(ROOT/replay['prediction_path']) as b:err=float(np.max(np.abs(z['prediction']-b['prediction'])))
                assert err==0 and replay['metrics']==best['metrics']
                fit.update(status='COMPLETE',best=best,reload_max_abs=err,reload=replay,wall_seconds=time.monotonic()-started,load_seconds=loading,active_seconds=sum(r['seconds'] for r in records),median_step_seconds=float(np.median([r['seconds'] for r in records])),peak_allocated=max([loadpeak]+[r['peak_allocated'] for r in records]+[r['peak_allocated'] for r in fr]),peak_reserved=max(r['peak_reserved'] for r in records),frozen_unchanged=True,budget_limited=best['step']==1024 and fr[-1]['metrics']['mse']<fr[-2]['metrics']['mse']);state['fits_completed']+=1
            except BaseException as e:
                fit.update(status='EXECUTION_ERROR',error=repr(e),wall_seconds=time.monotonic()-started)
                if isinstance(e,ResourceError) or any(v in str(e).lower() for v in ['illegal memory','device-side assert']):raise
            finally:write(fid+'_steps.json',records);write('fits.json',fits);write('status.json',state);m=opt=None;cleanup()
        complete_sources=[d for d in DATASETS if len([f for f in fits if f['dataset']==d and f['status']=='COMPLETE'])==12]
        if not complete_sources:state['status']='INCONCLUSIVE_EXECUTION';return
        selected=[f['best'] for f in fits if f['dataset'] in complete_sources];seal=dict(contract_hash=sha(OUT/'contract.json'),resource_strategy_hash=sha(OUT/'resource_strategy_seal.json'),complete_sources=complete_sources,selections=selected,sealed_at=time.time(),planned_attempts_finished=len(fits)==24);seal['seal_hash']=digest(seal);write('selection_seal.json',seal)
        contract_check();write('evaluation_access.json',dict(opened_at=time.time(),selection_hash=sha(OUT/'selection_seal.json'),scope='Only source blocks with all six arms and both seeds complete'))
        for choice in selected:
            d,a,seed=choice['dataset'],choice['arm'],choice['seed'];m=make_model(a,c['data'][d]['channel_ids'],seed,strategy['checkpoint'],'cuda');restore(m,torch.load(ROOT/choice['checkpoint_path'],map_location='cpu',weights_only=True));values,_,std=load(CACHE,d,'evaluation')
            r=evaluate(m,values,std,c['data'][d]['origins']['evaluation'],'E_'+choice['fit'],w);evaluation.append(dict(dataset=d,arm=a,seed=seed,role='selected',**r));write('evaluation.json',evaluation)
            if a=='BASIS4':
                coeff=m.channel_adapter.coefficients.detach().clone();weights=torch.stack([p.weight.detach() for p in m.channel_adapter.up_projections]);effective=torch.einsum('ck,kdr->cdr',coeff,weights)
                write(choice['fit']+'_mechanism.json',dict(coefficient_channel_variance=coeff.var(0,unbiased=False).cpu().tolist(),basis_coefficient_L1=coeff.abs().sum(0).cpu().tolist(),effective_weight_channel_variance=float(effective.var(0,unbiased=False).mean()),coefficients=coeff.cpu().tolist()))
                perm=np.random.default_rng(60110).permutation(64)
                with torch.no_grad():m.channel_adapter.coefficients.copy_(coeff[torch.tensor(perm,device='cuda')])
                r=evaluate(m,values,std,c['data'][d]['origins']['evaluation'],'E_COEFFICIENT_SWAP_'+choice['fit'],w);evaluation.append(dict(dataset=d,arm=a,seed=seed,role='posthoc_coefficient_swap',permutation=perm.tolist(),**r));write('evaluation.json',evaluation)
            del m;cleanup()
        state['status']='COMPLETE' if len(complete_sources)==2 else 'ONE_SOURCE_PILOT'
    except BaseException as e:state.update(status='BLOCKED_GPU_BUSY' if 'BLOCKED_GPU_BUSY' in str(e) else 'INCONCLUSIVE_EXECUTION',error=repr(e),traceback=traceback.format_exc())
    finally:write('status.json',state);w.close();print('C END',state['status'],state['fits_completed'],flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','preflight','train-evaluate','status']);a=p.parse_args();configure()
    if a.stage=='status':print(json.dumps(read(OUT/'status.json') if (OUT/'status.json').exists() else dict(status='NOT_PREPARED'),indent=2))
    else:globals()[a.stage.replace('-','_')]()
if __name__=='__main__':main()
