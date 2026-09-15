"""R1 CLI: fixed semantic/numeric diagnosis and conditional original resource study."""
import argparse,json,traceback
from contextlib import nullcontext
import numpy as np
import torch
from common import ROOT,EXP,RESEARCH,R1 as OUT,C1 as CACHE,save,read,sha,audit_history
import rank1_engine as E
from numerics import check_parity

def semantic_cpu():
    torch.manual_seed(61501);x=torch.randn(8,9,dtype=torch.float64);y=torch.randn(8,4,dtype=torch.float64);y[0,0]=float('nan');y[5,1]=float('nan');initial=torch.randn(9,21*4,dtype=torch.float64)*.02;results=[]
    def forward(x,p,groups):
        loc=x.mean(-1,keepdim=True);scale=x.std(-1,keepdim=True).clamp_min(1e-6);h=((x-loc)/scale).asinh()
        logits=h@h.T/3;logits=logits.masked_fill(groups[:,None]!=groups[None,:],float('-inf'));h=h+torch.softmax(logits,-1)@h
        return (h@p).reshape(len(x),21,4),loc,scale
    for optimizer in ['SGD','Adam']:
        outputs=[]
        for micro in [8,4]:
            p=torch.nn.Parameter(initial.clone());opt=(torch.optim.SGD([p],lr=.001) if optimizer=='SGD' else torch.optim.Adam([p],lr=.001));losses=[]
            for i in range(0,8,micro):
                groups=torch.arange(micro)//4;z,loc,scale=forward(x[i:i+micro],p,groups);loss=E.native_loss(z,y[i:i+micro],loc,scale)*(micro/8);loss.backward();losses.append(loss.detach())
            raw=p.grad.clone();torch.nn.utils.clip_grad_norm_([p],1.);clip=p.grad.clone();opt.step();outputs.append((sum(losses),raw,clip,p.detach().clone()))
        for a,b in zip(*outputs):torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-9)
        results.append(dict(optimizer=optimizer,passed=True))
    p=initial;g=torch.arange(8)//4;z=forward(x,p,g)[0];xx=x.clone();xx[:4]*=2;torch.testing.assert_close(z[4:],forward(xx,p,g)[0][4:],rtol=0,atol=0)
    torch.testing.assert_close(z,forward(x,p,g+10)[0],rtol=0,atol=0)
    return dict(checks=results,group_isolation=True,group_renumbering=True,missing_native_denominator=True,cpu_toy_optimizer_updates=4,actual_model_updates=0,scope='Small FP64 attention/normalization/native pinball; not full Chronos FP64 proof')

def numerical():
    c=E.check_contract();state=read(OUT/'status.json');assert state['status']=='PREPARED';save(OUT/'semantic_cpu.json',semantic_cpu());E.configure_backend();E.save('numeric_backend.json',E.backend_record());w=E.Watch('numeric');checks=[];index={};resources=[];semantic=[]
    def keep(tag,obj):
        p=CACHE/(tag+'.pt');assert not p.exists();torch.save(obj,p);index[tag]=dict(path=E.rel(p),sha256=sha(p));save(OUT/'numeric_artifacts.json',index)
    def update(m,opt,v,pair,micro,tag,precision='bf16',export=False):
        assert state['numeric_update_attempts']<96;state['numeric_update_attempts']+=1;E.save('status.json',state)
        def applied():state['numeric_updates']+=1
        r,a=E.step(m,opt,v,pair,micro,w,precision=precision,export=export,on_applied=applied);resources.append(dict(tag=tag,**r));E.save('numeric_resources.json',resources);E.save('status.json',state)
        if export:
            x,_,g=E.batch(v,c['config']['probe_pair'])
            with E.preserve_rng(),torch.no_grad(),(torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()):z,p,_,_=m(x,g)
            sc=torch.tensor(np.tile(c['data'][m.arm_dataset]['scale'],2),device='cuda',dtype=torch.float64)[:,None,None]
            a['probe_z']=z.cpu();a['probe_scaled_raw']=(p.double()/sc).cpu();keep(tag,a)
        return a
    try:
        state['status']='NUMERIC_WAITING_FOR_GPU';E.save('status.json',state);w.boundary(startup=True)
        for d in E.DATASETS:
            v,scale=E.load_data(d,'train')
            for arm in E.ARMS:
                m,opt=E.make_model(arm,39000,0);m.arm_dataset=d;frozen=E.frozen_hash(m)
                for j in range(2):update(m,opt,v,E.PAIRS[j],2,f'warm_{d}_{arm}_{j}')
                shared=E.S.snapshot(m,opt);keep(f'warm_{d}_{arm}',shared)
                x,_,g=E.batch(v,c['config']['numeric_pairs'][0]);xx=x.clone();xx[:4]=xx[:4]*1.5+1
                with E.preserve_rng(),torch.no_grad():original=m(x,g)[1].cpu();changed=m(xx,g)[1].cpu();renumber=m(x,g+7)[1].cpu()
                assert torch.equal(original[4:],changed[4:]);assert torch.equal(original,renumber)
                semantic.append(dict(dataset=d,arm=arm,input_isolation_exact=True,renumbering_exact=True,target_not_passed=True))
                del x,xx,g,original,changed,renumber
                for j,pair in enumerate(c['config']['numeric_pairs']):
                    assert all(16384<=o and o+48<=22576 for o in pair)
                    refs={}
                    for precision in ['fp32','bf16']:
                        for micro in [2,1]:
                            E.S.restore(m,opt,shared);m.set_storage(0);tag=f'{d}_{arm}_pair{j}_{precision}_mb{micro}_cp0';a=update(m,opt,v,pair,micro,tag,precision,True);refs[(precision,micro)]=(tag,a)
                        tag0,a=refs[(precision,2)];tag1,b=refs[(precision,1)];r=check_parity(a,b,precision,True,scale);checks.append(dict(dataset=d,arm=arm,pair=pair,kind='microbatch',reference=tag0,actual=tag1,**r));E.save('numeric_checks.json',checks)
                    for micro in [2,1]:
                        E.S.restore(m,opt,shared);m.set_storage(12);tag=f'{d}_{arm}_pair{j}_fp32_mb{micro}_cp12';a=update(m,opt,v,pair,micro,tag,'fp32',True);tag0,b=refs[('fp32',micro)];r=check_parity(b,a,'fp32',False,scale);checks.append(dict(dataset=d,arm=arm,pair=pair,kind='checkpoint',reference=tag0,actual=tag,**r));E.save('numeric_checks.json',checks)
                    del refs,a,b
                assert E.frozen_hash(m)==frozen;semantic[-1]['frozen_unchanged']=True;E.save('leakage_checks.json',semantic);del m,opt,shared;E.cleanup();print('R1 NUMERIC',d,arm,state['numeric_updates'],flush=True)
        assert state['numeric_updates']==84
        state['status']='NUMERICS_BOUNDED' if all(r['passed'] for r in checks) else 'INCONCLUSIVE_NUMERICS_V2'
        # One bounded localization from recorded tensors, no speculative training patch.
        failed=[]
        for r in checks:
            if r['passed']:continue
            top=sorted(r['per_tensor']['update'].items(),key=lambda item:item[1]['rms_error'],reverse=True)[:3]
            failed.append(dict(dataset=r['dataset'],arm=r['arm'],kind=r['kind'],precision=r['precision'],gates={k:v for k,v in r['gates'].items() if not v},largest_update_tensors=top,pinball_sign_flips=r['pinball_sign_flips']))
        E.save('bounded_localization.json',dict(failed_comparisons=failed,semantic_bug_identified=False,repair_count=0,reason='FP64 semantic and real group isolation pass; same-shape checkpoint and different-shape errors separated. No demonstrated semantic bug warrants a training-code patch; no tolerance changes or repeated N1.',remaining_numeric_updates=96-state['numeric_updates']))
    except BaseException as e:state.update(status='BLOCKED_GPU_BUSY' if 'BLOCKED_GPU_BUSY' in str(e) else 'INCONCLUSIVE_EXECUTION',error=repr(e),traceback=traceback.format_exc())
    finally:E.save('status.json',state);E.save('numeric_checks.json',checks);E.save('leakage_checks.json',semantic);w.close();print('R1 END',state['status'],state['numeric_updates'],flush=True)
    verify_numerical()

def verify_numerical():
    c=E.check_contract();index=read(OUT/'numeric_artifacts.json') if (OUT/'numeric_artifacts.json').exists() else {};checks=read(OUT/'numeric_checks.json') if (OUT/'numeric_checks.json').exists() else []
    for r in index.values():assert sha(ROOT/r['path'])==r['sha256']
    for r in checks:
        a=torch.load(ROOT/index[r['reference']]['path'],map_location='cpu',weights_only=False);b=torch.load(ROOT/index[r['actual']]['path'],map_location='cpu',weights_only=False);z=check_parity(a,b,r['precision'],r['kind']=='microbatch',c['data'][r['dataset']]['scale']);assert z=={k:r[k] for k in z}
    E.save('numeric_verification.json',dict(comparisons_replayed=len(checks),failed=sum(not r['passed'] for r in checks),new_gpu_forward=0,new_optimizer_updates=0))
    audit_history();s=read(OUT/'status.json')
    lines=['# R1 Query 수치 복구와 조건부 자원 비교','',f"상태 **{s['status']}**. 실제 수치 updates {s['numeric_updates']}/96, 자원 updates {s['A_updates']}/480, 본학습 {s['B_fits_completed']}/{s['B_fit_attempts']} attempts(상한12).",'', '새 배치와 RMS 기준은 이전 실패를 본 뒤 사용자 지시로 고정했다. 과거120-update/180-update 결과와 판정은 그대로 보존한다. 동일 shape checkpoint와 다른 shape microbatch 검사를 분리하고, FP64 변환 뒤 parameter delta를 계산했다.','', '| 종류 | 정밀도 | 통과/검사 |','| --- | --- | --- |']
    for k,p in [('checkpoint','fp32'),('microbatch','fp32'),('microbatch','bf16')]:
        rr=[r for r in checks if r['kind']==k and r['precision']==p];lines.append(f"| {k} | {p} | {sum(r['passed'] for r in rr)}/{len(rr)} |")
    lines+=['','미충족 검사는 수치 무결성 미확정이며 예측 성능 실패가 아니다. 의미 오류가 입증되지 않으면 코드나 허용치를 재조정하지 않는다. [수치 원기록](numeric_checks.json), [한정 위치 추적](bounded_localization.json), [독립 재계산](numeric_verification.json).','', '자원 또는 본학습 미실행은 NOT_RUN이며 성능0이나 자원 이점0을 측정한 것이 아니다. 조건을 통과하면 같은 계약의 자원·학습 절차만 이어간다. 별도 후보·재튜닝은 없다.','']
    (OUT/'NUMERICS_REVIEW.md').write_text('\n'.join(lines));(OUT/'REPORT.md').write_text('\n'.join(lines))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','numerical-check','profile','train-evaluate','verify','status']);a=p.parse_args();E.configure_backend()
    if a.stage=='prepare':E.prepare()
    elif a.stage=='numerical-check':numerical()
    elif a.stage=='verify':
        if (OUT/'resource_selection.json').exists():
            from finalize_rank1 import finalize
            finalize()
        else:verify_numerical()
    elif a.stage=='status':print(json.dumps(read(OUT/'status.json'),indent=2))
    else:getattr(E,a.stage.replace('-','_'))()
if __name__=='__main__':main()
