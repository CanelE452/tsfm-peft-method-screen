import subprocess
from .common import *

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    if (OUT/'SEAL.json').exists():return check_seal()
    assert not (OUT/'UPDATE_LEDGER.jsonl').exists()
    hashes={}
    def add(p):hashes[str(p.relative_to(ROOT))]=sha(p)
    previous=ROOT/'results/learned_gate_comparison_20260919'
    assert read(previous/'AUDIT.json')['status']=='VERIFIED'
    for p,h in read(previous/'SEAL.json')['hashes'].items():
        assert sha(ROOT/p)==h,('PRIOR_SEAL_CHANGED',p);hashes[p]=h
    rec=snapshot()
    for p,h in rec['files'].items():assert sha(ROOT/p)==h;hashes[p]=h
    reuse={}
    for k,r in read(previous/'PREDICTIONS.json').items():
        if r['arm'] not in RENAMES:continue
        assert sha(ROOT/r['path'])==r['sha256'];add(ROOT/r['path'])
        rr=dict(r,arm=RENAMES[r['arm']],original_arm=r['arm'],original_key=k,original_manifest=str((previous/'PREDICTIONS.json').relative_to(ROOT)))
        reuse[k.replace('__'+r['arm']+'__','__'+rr['arm']+'__')]=rr
    assert len(reuse)==96
    models=[dict(r,original_arm=r['arm'],arm=RENAMES[r['arm']]) for r in read(ROOT/'results/temporal_response_peft_20260919/MODEL_SELECTION.json') if r['arm'] in RENAMES]
    assert len(models)==24
    for r in models:assert sha(ROOT/r['checkpoint'])==r['sha256'];add(ROOT/r['checkpoint'])
    sources={}
    for source in SOURCES:
        folder=parent.CACHE/'conditions'/source
        sources[source]={}
        for p in sorted(folder.glob('*.npy')):
            add(p);a=np.load(p,mmap_mode='r');sources[source][p.name]=dict(path=str(p.relative_to(ROOT)),shape=list(a.shape),dtype=str(a.dtype),sha256=sha(p))
        assert np.load(folder/'train_x.npy',mmap_mode='r').shape==(32,1024,512)
        # Verify train/V artifacts against historical seals, not just new hashes.
        oldhash=read(previous/'SEAL.json')['hashes']
        for name,r in sources[source].items():
            if name in ['train_x.npy','train_y.npy','train_sigma.npy','V_SELECT_x.npy','V_SELECT_y.npy','V_SELECT_sigma.npy']:
                assert r['path'] in oldhash and oldhash[r['path']]==r['sha256'],('TRAIN_V_HASH_MISMATCH',name)
    panels={}
    for panel in PANELS:
        paths=list(data_path(panel).glob('E_DISCOVERY*'))
        for kind in ['standard','shape']:
            paths+=list(panel_path(panel,kind).glob('*.npy'))+list(panel_path(panel,kind).glob('manifest.json'))
        panels[panel]={}
        for p in paths:
            if p.is_file():add(p);panels[panel][str(p.relative_to(ROOT))]=sha(p)
    baseline=dict(snapshot=rec,reused_models=models,reused_prediction_views=96,training_arrays=sources,evaluation_files=panels,
                  B0_lora_parameters=294912,adapter_parameters=8712,MAG_ONLY_is_no_lora=False,previous_results_unmodified=True)
    save(OUT/'BASELINE_MANIFEST.json',baseline);save(OUT/'REUSED_PREDICTIONS.json',reuse)
    checks=[]
    for source in SOURCES:
        x,y,s=batch(arrays(source),'F0_PLAIN',0,np.arange(4),device='cpu');raw=x.clone()
        for seed in [81550,*SEEDS]:
            base=build('F0',seed,source,'cpu');pmodel=build('F0_PLAIN',seed,source,'cpu');mmodel=build('F0_MAG',seed,source,'cpu')
            assert tensor_hash(cpu_state(pmodel))==tensor_hash(cpu_state(mmodel))
            with torch.no_grad():
                q=base(x,s);assert torch.equal(pmodel(x,s),q) and torch.equal(mmodel(x,s),q)
                assert torch.equal(pmodel(x,s,off=True),q) and torch.equal(mmodel(x,s,off=True),q)
                native=base.base(context=x).quantile_preds
                assert torch.equal(q,native),'NATIVE_F0_FORWARD_MISMATCH'
                assert torch.equal(raw,x)
            keys=list(dict(base.base.named_parameters()))
            for arm,m in [('F0',base),('F0_PLAIN',pmodel),('F0_MAG',mmodel)]:
                info=audit_model(m,arm);assert info['base_parameter_keys']==keys
                checks.append(dict(source=source,seed=seed,**info,initial_F0_exact=True,native_F0_exact=True,off_F0_exact=True,foundation_sha256=tensor_hash(m.base.state_dict()),adapter_initial_sha256=tensor_hash(cpu_state(m))))
            del base,pmodel,mmodel
    save(OUT/'NO_LORA_AUDIT.json',dict(stage='PRETRAIN_CPU',cases=checks,optimizer_updates=0,matched_adapter_initialization=True,native_F0_parity=True))
    save(OUT/'AUTHORIZATION.json',dict(experiment=NAME,contract_sha256=sha(EXP/'CONTRACT.txt'),main_cap=MAIN_CAP,smoke_cap=SMOKE_CAP,user_message='이번 요청은 저장소의 이전 새 학습 금지 지시를 이 실험 범위에 한해서만 해제하는 명시적 승인이다.',automatic_successor=False))
    (OUT/'PREFLIGHT_KO.md').write_text('# 실행 전 감사\n\n기준 HEAD: '+subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()+'\n\n단일 계약 CONTRACT.txt를 보존했다. 기존 MAG_ONLY는 no-LoRA가 아닌 B0+MAG이다. B0 q/v rank8 LoRA294912와 residual8712를 구분한다. 새 F0 경로는 pretrained snapshot을 직접 로드하며 attach_lora를 호출하지 않는다. 실제 CPU모델18경로의 parameter/module검사, 동일seed초기값, native F0/step0/off예측정확동일을 확인했다. 기존24개모델·96개예측view와TRAIN/V/E hash검증. 기존결과수정0. CPU검사는 실제학습smoke의대체가 아니다.\n\n새16fits/16384+8updates만승인. E는 모두기존노출개발패널이며NESO도독립source아님. GPU는실행시별도감시하고RustDesk만예외. 상세path/hash는BASELINE_MANIFEST.json. 원자료·가중치·prediction은로컬캐시다.\n')
    for folder in [EXP,ROOT/'experiments/learned_gate_comparison_20260919',ROOT/'experiments/additive_persistence_validation_v1_20260917',ROOT/'experiments/c3_weakness_controls_20260918']:
        for p in folder.glob('*'):
            if p.suffix in ['.py','.md','.txt']:add(p)
    for p in [previous/'PREDICTIONS.json',previous/'RAW_SCORES.csv',OUT/'BASELINE_MANIFEST.json',OUT/'REUSED_PREDICTIONS.json',OUT/'NO_LORA_AUDIT.json',OUT/'AUTHORIZATION.json',OUT/'PREFLIGHT_KO.md'] :add(p)
    save(OUT/'SEAL.json',dict(at=time.time(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),hashes=hashes,main_cap=MAIN_CAP,smoke_cap=SMOKE_CAP,all_evaluation_previously_exposed=True))
    status(execution='SEALED')

def smoke(watch):
    require_authorization();check_seal()
    if (OUT/'SMOKE.json').exists():
        rows=read(OUT/'SMOKE.json');assert len(rows)==4 and sum(r['updates'] for r in rows)==8;return
    assert not (OUT/'SMOKE_LEDGER.jsonl').exists() and not (OUT/'SMOKE_INTENT.json').exists(),'PARTIAL_SMOKE_NO_AUTOMATIC_REPLAY'
    rows=[]
    for source in SOURCES:
        x,y,s=batch(arrays(source),'F0_PLAIN',0,np.arange(32));raw=x.clone()
        for arm in ARMS:
            watch.boundary();m=build(arm,81550,source);base=build('F0',81550,source)
            initial=cpu_state(m);before=frozen_hash(m);buffers=tensor_hash(dict(m.named_buffers()))
            with torch.no_grad():
                pred=m(x,s);f0=base(x,s);assert torch.equal(pred,f0) and torch.equal(m(x,s,off=True),f0)
                hypothetical=y+10*s[:,None];assert not torch.equal(y,hypothetical);assert torch.equal(m(x,s),pred)
                if arm=='F0_MAG':
                    g=m.gate(x,s);assert torch.isfinite(g).all() and (g>=0).all() and (g<=1).all()
            opt=torch.optim.AdamW(parameters(m).values(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
            for step in range(2):
                begin=watch.before();opt.zero_grad(set_to_none=True);loss=loss_2pinball(m(x,s),y,s,m.base.quantiles);assert torch.isfinite(loss);loss.backward()
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters(m).values())
                norm=torch.nn.utils.clip_grad_norm_(parameters(m).values(),1.);assert torch.isfinite(norm)
                save(OUT/'SMOKE_INTENT.json',dict(source=source,arm=arm,step=step+1,status='BEFORE_OPTIMIZER'));opt.step();sync()
                append(OUT/'SMOKE_LEDGER.jsonl',dict(source=source,arm=arm,step=step+1,loss=float(loss.detach())))
                save(OUT/'SMOKE_INTENT.json',dict(source=source,arm=arm,step=step+1,status='JOURNALED'))
                _,bad=watch.after(begin)
                if bad:raise ResourceError('EXTERNAL_COMPUTE_DURING_SMOKE')
            state=cpu_state(m);changed=[n for n in state if not torch.equal(state[n],initial[n])]
            assert 'adapter.up.weight' in changed and 'adapter.down.weight' in changed
            assert frozen_hash(m)==before and tensor_hash(dict(m.named_buffers()))==buffers
            with torch.no_grad():
                trained=m(x,s);rev=m(x.flip(0),s.flip(0)).flip(0);diff=float(((trained-rev)/s[:,None,None]).abs().max())
                assert diff<=1e-5 or torch.allclose(trained,rev,rtol=1e-4,atol=0)
                assert torch.equal(m(x,s,off=True),f0) and torch.equal(x,raw)
            p=CACHE/'smoke'/f'{source}_{arm}.pt';atomic_torch(p,state)
            fresh=build(arm,81550,source);restore(fresh,torch.load(p,map_location='cpu',weights_only=True))
            with torch.no_grad():assert torch.equal(trained,fresh(x,s))
            rows.append(dict(source=source,arm=arm,updates=2,initial_F0_exact=True,off_F0_exact=True,restore_exact=True,frozen_unchanged=True,buffers_unchanged=True,raw_input_unchanged=True,target_substitution_prediction_unchanged=True,finite_gradients=True,changed_parameters=changed,batch_permutation_max_normalized_difference=diff,no_lora=audit_model(m,arm)))
            del m,base,opt,fresh;cleanup()
    assert len((OUT/'SMOKE_LEDGER.jsonl').read_text().splitlines())==SMOKE_CAP
    save(OUT/'SMOKE.json',rows)

if __name__=='__main__':setup();prepare()
