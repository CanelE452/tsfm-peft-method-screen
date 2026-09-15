"""Fixed MOMENT comparison. No imports or mutations of historical runner globals."""
import copy, importlib.util, json, math, os, random, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
NAME='channel_controlled_improvement_v1_20260916'
EXP=ROOT/'experiments'/NAME; OUT=ROOT/'results'/NAME; CACHE=ROOT/'.cache'/NAME
DATA=ROOT/'.cache/channel_sharing_screen_v1_20260915'
sys.path.insert(0,str(ROOT/'experiments/peft_rank12_20260915'))
from common import read,save,sha,digest,csvwrite,parameters,cpu_state,tensor_hash,frozen_hash,restore,preserve_rng,cleanup,Watch
from rank2_model import make as make_lh
from rank2_data import load,batch,loss,metrics,independent
from run_rank2 import configure,step
spec=importlib.util.spec_from_file_location('controlled_attention_model',ROOT/'experiments/channel_attention_prior_20260916/model.py')
attention=importlib.util.module_from_spec(spec);spec.loader.exec_module(attention)
ARMS=['LH','SIDE','PRIOR']; SEEDS=[41000,41001,41002]; LRS=[.001,.0003]

def make(arm,ids,seed,device='cpu'):
    return make_lh('LH',ids,seed,device) if arm=='LH' else attention.make(arm,ids,seed,device)

def check_hashes(h):
    for p,v in h.items():assert sha(ROOT/p)==v,('HASH_CHANGED',p)

def contract():
    c=read(OUT/'seal.json');check_hashes(c['source_hashes']);return c

def phase_panel(origins,start,stop,horizon,seed):
    base=np.asarray(origins,dtype=np.int64);n=len(base)
    assert n>0 and np.all(np.diff(base)>0)
    phases=np.floor(24*np.arange(n)/n).astype(np.int64)
    phases=np.random.default_rng(seed).permutation(phases);out=[]
    for b,p in zip(base,phases):
        o=int(b+(p-b)%24)
        if o+horizon>stop:o-=24
        if o<start:o+=24
        assert start<=o and o+horizon<=stop
        assert o%24==p and abs(o-b)<=23
        out.append(o)
    out=np.asarray(out,dtype=np.int64)
    assert len(np.unique(out))==n and np.all(np.diff(out)>0)
    counts=np.bincount(out%24,minlength=24);assert counts.max()-counts.min()<=1
    return out.tolist()

def optimizer(m,lr):
    return torch.optim.AdamW(parameters(m).values(),lr=lr,weight_decay=0,betas=(.9,.999),eps=1e-8)

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def rng_restore(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])

def atomic_torch(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def state_json():return read(OUT/'status.json')

def status(**kw):
    s=state_json();s.update(kw);save(OUT/'status.json',s);return s

def metric_record(p,y,std):
    met=metrics(p,y,std);met['raw_mae']=float(np.mean(met['channel_raw_mae']));return met

def predict(m,values,std,origins,tag,w,baseline=None):
    path=CACHE/(tag+'.npz');receipt=CACHE/(tag+'.json')
    if receipt.exists():
        r=read(receipt);assert sha(path)==r['prediction_sha256'];return r
    assert not path.exists(),('UNCOMMITTED_PREDICTION',tag)
    tick=time.monotonic();pp=[];yy=[];mode=m.training if m is not None else None
    torch.cuda.reset_peak_memory_stats()
    with preserve_rng(),torch.no_grad():
        if m is not None:m.eval()
        for i in range(0,len(origins),8):
            w.boundary();x,y=batch(values,origins[i:i+8])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                p=x[:,:,-24:].repeat(1,1,4) if baseline=='SEASONAL_NAIVE' else x[:,:,-1:].repeat(1,1,96) if baseline=='LAST_VALUE' else m(x).forecast
            assert torch.isfinite(p).all();pp.append(p.float().cpu().numpy());yy.append(y.cpu().numpy())
        if m is not None:m.train(mode)
    p=np.concatenate(pp);y=np.concatenate(yy)
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,prediction=p,target=y,std=std,origins=np.array(origins));os.replace(tmp,path)
    r=dict(metrics=metric_record(p,y,std),prediction_path=str(path.relative_to(ROOT)),prediction_sha256=sha(path),seconds=time.monotonic()-tick,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),exposure='DISCOVERY_REUSED_PERIODS')
    save(receipt,r);return r

def prepare():
    assert not (OUT/'seal.json').exists() and not CACHE.exists(),'No duplicate prepare'
    configure();CACHE.mkdir()
    balanced=ROOT/'results/channel_phase_balance_20260916';prior=ROOT/'results/channel_attention_prior_20260916'
    old=read(balanced/'seal.json');pc=read(prior/'seal.json')
    check_hashes(old['source_hashes']);check_hashes(pc['source_hashes']);check_hashes(old['model_files'])
    data=copy.deepcopy(old['data']);schedules=read(balanced/'schedules.json')
    assert schedules==read(prior/'schedules.json')
    save(OUT/'historical_hashes.json',{str(p.relative_to(ROOT)):sha(p) for folder in ['results','research'] for p in (ROOT/folder).rglob('*') if p.is_file() and OUT not in p.parents})
    cpu=[];panels=[];exposures=[]
    for di,(d,dc) in enumerate(data.items()):
        check_hashes(dc['staged']);assert sha(dc['raw_path'])==dc['raw_sha256']
        train=dc['origins']['train'];assert len(train)==(512 if d=='electricity' else 504)
        sampler=next(r for r in read(balanced/'sampler_audit.json') if r['dataset']==d);assert train==sampler['new_origins']
        rng=np.random.default_rng(41002);schedules[f'{d}_41002']=[rng.permutation(train).tolist() for _ in range(20)]
        for seed in SEEDS:
            rng=np.random.default_rng(seed)
            assert schedules[f'{d}_{seed}']==[rng.permutation(train).tolist() for _ in range(20)]
        dc['panels']={'TRAIN':train,'V_FIXED':dc['origins']['validation'],'E_FIXED':dc['origins']['evaluation']}
        for kind,seed,start,stop in [('V',62016+di,dc['t1'],dc['t2']),('E',63016+di,dc['t2'],dc['rows'])]:
            dc['panels'][kind+'_MIXED']=phase_panel(dc['panels'][kind+'_FIXED'],start,stop,96,seed)
        for panel,oo in dc['panels'].items():
            base=dc['panels'].get(panel.replace('MIXED','FIXED'),oo);targets=[o+j for o in oo for j in range(96)]
            hist=np.bincount(np.array(oo)%24,minlength=24).tolist()
            lo,hi=(96,dc['t1']) if panel=='TRAIN' else (dc['t1'],dc['t2']) if panel.startswith('V') else (dc['t2'],dc['rows'])
            assert all(lo<=o and o+96<=hi for o in oo)
            exposures.append(dict(dataset=d,panel=panel,origins=len(oo),target_occurrences=len(targets),unique_targets=len(set(targets)),duplicate_occurrences=len(targets)-len(set(targets)),target_union=sorted(set(targets)),phase_counts=hist))
            panels += [dict(dataset=d,panel=panel,index=i,origin=o,phase=o%24,base_origin=b,shift=o-b) for i,(o,b) in enumerate(zip(oo,base))]
        vals,std=load(DATA,d,'development');tr,st=load(DATA,d,'train')
        assert vals.shape[0]==dc['t2'] and tr.shape[0]==dc['t1'] and np.array_equal(tr,vals[:dc['t1']],equal_nan=True)
        assert np.array_equal(std,np.array(dc['std'])) and np.array_equal(st,std)
        # E poison is synthetic: no E array is opened before selection. Prefix-only data API.
        virtual=np.concatenate([vals,np.zeros((dc['rows']-len(vals),32),np.float32)])
        poisoned=virtual.copy();poisoned[dc['t2']:]=123456
        for panel in ['TRAIN','V_FIXED','V_MIXED']:
            a,b=batch(virtual,dc['panels'][panel],'cpu');x,y=batch(poisoned,dc['panels'][panel],'cpu')
            assert torch.equal(a,x) and torch.equal(b,y)
        assert np.array_equal(virtual[:dc['t1']],poisoned[:dc['t1']])
        p=np.random.default_rng(70000+di).normal(size=(8,32,96)).astype(np.float32);_,y=batch(tr,train[:8],'cpu');y[0,0,:3]=float('nan')
        actual=float(loss(torch.from_numpy(p),y));ref=independent(p,y.numpy());assert math.isclose(actual,ref,rel_tol=1e-6,abs_tol=1e-6)
        scalar=metric_record(p,y.numpy(),std)['mse'];assert math.isclose(scalar,ref,rel_tol=1e-10,abs_tol=1e-10)
        cpu.append(dict(dataset=d,prefix_isolation=True,synthetic_E_poison_invariant=True,normalization_prefix_invariant=True,loss_fp32=actual,loss_scalar_float64=ref,metric_float64=scalar))
    save(OUT/'schedules.json',schedules);csvwrite(OUT/'phase_panels.csv',panels);save(OUT/'exposure.json',exposures);save(OUT/'cpu_checks.json',cpu)
    (OUT/'exposure_ledger.md').write_text('# 노출 장부\n\n모든 결과는 DISCOVERY_REUSED_PERIODS. E 배열은 선택 봉인 뒤 scorer만 연다. 준비 단계에서 E 파일의 바이트 해시를 검증하는 것은 성능 선택이 아니다. V/E MIXED는 FIXED와 일부 다른 target이며 순수 위상 인과효과가 아니다. Traffic V는 18개 위상만 포함한다. 실제 target 행 union·중복·위상은 exposure.json, 이동량은 phase_panels.csv에 있다. 행 위상은 현지 시간 검증이 아니다.\n')
    # Verify initial states and every complete historical path. Partial paths never warm-start.
    inventories=[];initials={};reuse={};receipts=[]
    for d,dc in data.items():
        for seed in SEEDS:
            states={}
            for arm in ARMS:
                m=make(arm,dc['channel_ids'],seed);state=cpu_state(m);states[arm]=state
                assert sum(p.numel() for p in parameters(m).values())==761952
                inv=[dict(name=n,shape=list(p.shape),numel=p.numel(),trainable=p.requires_grad) for n,p in m.named_parameters()]
                inventories.append(dict(dataset=d,seed=seed,arm=arm,parameters=inv,initial_hash=tensor_hash(state),frozen_hash=frozen_hash(m),buffer_hash=tensor_hash(dict(m.named_buffers()))))
                if arm!='LH':
                    assert all(not p.requires_grad for p in m.encoder.parameters())
                    assert all(torch.count_nonzero(p)==0 for n,p in m.encoder.named_parameters() if 'lora_B' in n)
                initials[f'{d}_{seed}_{arm}']=tensor_hash(state)
                directory=balanced if arm=='LH' else prior
                f=next((f for f in read(directory/'fits.json') if f['dataset']==d and f['seed']==seed and (arm=='LH' or f['arm']==arm)),None)
                reason='No historical seed' if f is None else 'Partial path: no complete optimizer/scheduler/RNG state' if f['epochs']!=20 else None
                if f is not None and f['epochs']==20:
                    rows=sorted([r for r in read(directory/'trajectory.json') if r['fit']==f['fit']],key=lambda r:r['epoch'])
                    assert [r['epoch'] for r in rows]==list(range(21))
                    steps=read(directory/(f['fit']+'_steps.json'));n=len(dc['origins']['train'])//8
                    assert f['status']=='COMPLETE' and f['updates']==20*n and len(steps)==20*n and f['frozen_and_buffers_unchanged']
                    for i,r in enumerate(steps):
                        assert r['epoch']==i//n+1 and r['update']==i+1 and r['origins']==8
                        assert r['lr']==.001*.5**((r['epoch']-1)//5)
                        assert math.isfinite(r['loss']) and math.isfinite(r['gradient_norm'])
                    for r in rows:
                        assert sha(ROOT/r['checkpoint_path'])==r['checkpoint_sha256'] and r['updates']==r['epoch']*n
                    oldstate=torch.load(ROOT/rows[0]['checkpoint_path'],map_location='cpu',weights_only=True)
                    assert tensor_hash(oldstate)==tensor_hash(state)
                    source=(ROOT/'experiments'/directory.name/'run.py').read_text()
                    assert 'with preserve_rng(),torch.no_grad():' in source and 'scheduler.step()' in source and '8,w,applied' in source
                    key=f'{d}_{seed}_{arm}';reuse[key]=dict(old_fit=f['fit'],old_directory=str(directory.relative_to(ROOT)),checkpoints=rows,updates=f['updates'])
                    receipts.append(dict(key=key,status='REUSE_ACCEPTED',epochs=21,updates=f['updates'],initial_hash=tensor_hash(state),checkpoint_hashes=[r['checkpoint_sha256'] for r in rows],source=str((ROOT/'experiments'/directory.name/'run.py').relative_to(ROOT)),recipe_audited='AdamW .001 beta .9/.999 eps1e-8 wd0; clip1; StepLR5/.5 after V; BF16/FP32 batch8; preserve_rng; identical configure/step/model; complete sealed schedules',old_score_reused=False))
                else:receipts.append(dict(key=f'{d}_{seed}_{arm}',status='NEW_REQUIRED',reason=reason))
                del m;cleanup()
            head=lambda s:tensor_hash({n:v for n,v in s.items() if n.startswith('head.')})
            assert len({head(v) for v in states.values()})==1
            assert tensor_hash(states['SIDE'])==tensor_hash(states['PRIOR'])
    save(OUT/'parameter_inventory.json',inventories);save(OUT/'initial_hashes.json',initials);save(OUT/'REUSE_RECEIPT.json',receipts);save(OUT/'reuse.json',reuse)
    cells=[dict(dataset=d,arm=a,lr=lr,seed=s) for d in data for a in ARMS for lr in LRS for s in SEEDS]
    cells=[dict(order=i,fit=f'{i:02}_{cells[j]["dataset"]}_{cells[j]["arm"]}_{cells[j]["seed"]}_{cells[j]["lr"]}',**cells[j]) for i,j in enumerate(np.random.default_rng(64016).permutation(36))]
    for c in cells:c['reuse_key']=f'{c["dataset"]}_{c["seed"]}_{c["arm"]}' if c['lr']==.001 and f'{c["dataset"]}_{c["seed"]}_{c["arm"]}' in reuse else None
    save(OUT/'fit_manifest.json',cells);csvwrite(OUT/'fit_manifest.csv',cells)
    free=shutil.disk_usage(ROOT).free;estimate=36*21*761952*4+36*761952*12+3*2**30
    assert free>=10*2**30 and free>=estimate
    files=list(EXP.glob('*.py'))+[OUT/'PROTOCOL.md',OUT/'schedules.json',OUT/'fit_manifest.json',OUT/'initial_hashes.json',OUT/'reuse.json',OUT/'phase_panels.csv',OUT/'REUSE_RECEIPT.json',ROOT/'scripts/run_channel_controlled_improvement.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files};hashes.update(old['source_hashes']);hashes.update(pc['source_hashes'])
    save(OUT/'seal.json',dict(created_at=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes=hashes,data=data,model_files=old['model_files'],cells=cells,wall_cap=14400,fit_cap=36,update_cap=45720,smoke_cap=12,resource_cap=120,exposure='DISCOVERY_REUSED_PERIODS',estimated_storage_bytes=estimate,free_bytes=free))
    save(OUT/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,configure='same original configure; BF16 forward FP32 loss/Adam; micro=effective=8',rustdesk_exception='/usr/share/rustdesk/rustdesk only, previously approved',git_status=subprocess.check_output(['git','status','--short'],text=True),baseline_diff=subprocess.check_output(['git','diff','--stat','fa5be2988562249d45bd8609b84ecf14d0f8690c..HEAD'],text=True)))
    (OUT/'SCOPE.md').write_text('# 실행 범위\n\n첨부된 단일 PROTOCOL만 적용한다. 기준 fa5be29 이후 완료된 예보 대조는 별도로 보존했으며 입력·방법·결과를 합치지 않는다. 36개 고정 경로, 두 LR, 세 seed, 20 epochs. 기존 교정 TRAIN 유지. 재사용 심사는 REUSE_RECEIPT에 기록하며 중간 종료 경로는 재개하지 않는다. 새 모델 제안·후속 학습은 자동 실행하지 않는다.\n')
    save(OUT/'status.json',dict(status='PREPARED',training_updates=0,smoke_updates=0,resource_updates=0,new_fits_started=0,new_fits_completed=0,reused_fits_completed=0,reused_updates=0))
    print('PREPARED',len(reuse),'reuse;',36-len(reuse),'new trajectories',flush=True)
