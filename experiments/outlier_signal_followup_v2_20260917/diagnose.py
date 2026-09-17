"""Zero optimizer steps; generator metadata restricted to descriptive diagnostics."""
import pandas as pd
from .common import *
from .prepare import OLD,OLDC,audit
from .model import robust_scale
from experiments.outlier_signal_peft_v1_20260917.common import build as old_build
from experiments.outlier_signal_peft_v1_20260917.model import preprocess

def exposure(x,s,metas,source,role):
    rows=[]
    for lo in range(0,len(x),512):
        a=np.asarray(x[lo:lo+512],dtype=float);ss=np.asarray(s[lo:lo+512],dtype=float)
        m=np.median(a,axis=-1);r=np.maximum(1.4826*np.median(np.abs(a-m[:,None]),axis=-1),.1*ss)
        clipped=np.clip(a,m[:,None]-6*r[:,None],m[:,None]+6*r[:,None]);disc=a-clipped;mask=disc!=0
        for j in range(len(a)):
            meta=metas[lo+j];run=maximum=0
            for bit in mask[j]:run=run+1 if bit else 0;maximum=max(maximum,run)
            duration=meta.get('shift_length',0);delta=meta.get('delta',0.)
            amp=abs(delta)/meta['r0'] if duration else abs(meta.get('fault_delta',[0.])[0])/meta['r0']
            shift=mask[j,-duration:] if duration else np.array([],bool)
            rows.append(dict(source=source,role=role,state=meta['state'],amplitude=round(amp,8),shift_duration=duration,
                fault_points=len(meta.get('positions',[])),example=meta['example'],epoch=meta.get('epoch',-1),generator=meta.get('generator',-1),
                median=m[j],robust_scale=r[j],clip_lower=m[j]-6*r[j],clip_upper=m[j]+6*r[j],clipped_points=int(mask[j].sum()),
                clipped_fraction=float(mask[j].mean()),discarded_mass=float(abs(disc[j]).sum()),normalized_mass=float(abs(disc[j]).sum()/(512*r[j])),
                maximum_discarded_over_r=float(abs(disc[j]).max()/r[j]),longest_run=maximum,
                last64_same_sign_clipped_points=int(max((disc[j,-64:]>0).sum(),(disc[j,-64:]<0).sum())),
                shift_positions=duration,shift_clipped_count=int(shift.sum()),shift_clipped_fraction=float(shift.mean()) if duration else np.nan,
                nonzero_residual=float(mask[j].any()),shift_nonzero_residual=float(shift.any()) if duration else np.nan,
                residual_mass_over_abs_delta=float(abs(disc[j,-duration:]).sum()/abs(delta)) if duration else np.nan))
    return pd.DataFrame(rows)

def cpu_diagnosis():
    audit()
    if (OUT/'clip_exposure_summary.csv').exists():return
    frames=[]
    for source in SOURCES:
        f=OLDC/'conditions'/source
        for role in ['TRAIN','V_SELECT','E_DISCOVERY']:
            prefix='train' if role=='TRAIN' else role
            x=np.load(f/f'{prefix}_x.npy',mmap_mode='r').reshape(-1,512);s=np.load(f/f'{prefix}_sigma.npy')
            if role=='TRAIN':s=np.tile(s,32)
            meta=read(f/f'{prefix}_generator_audit.json')
            frames.append(exposure(x,s,meta,source,role));print('CLIP_AUDIT',source,role,len(x),flush=True)
    df=pd.concat(frames,ignore_index=True);fields=['source','role','state','amplitude','shift_duration','fault_points']
    numeric=[c for c in df.columns if c not in fields+['example','epoch','generator']]
    grouped=df.groupby(fields,dropna=False)[numeric].mean().reset_index();sizes=df.groupby(fields).size().rename('examples').reset_index();grouped=grouped.merge(sizes,on=fields)
    grouped[grouped.role=='TRAIN'].to_csv(OUT/'train_clip_exposure.csv',index=False)
    grouped[grouped.role!='TRAIN'].to_csv(OUT/'eval_clip_exposure.csv',index=False)
    summary=df.groupby(['source','role','state'])[numeric].mean().reset_index();summary.to_csv(OUT/'clip_exposure_summary.csv',index=False)
    df.to_csv(CACHE/'clip_exposure_per_example.csv',index=False)

@torch.no_grad()
def adapter_diagnosis(watch):
    if (OUT/'adapter_usage.csv').exists():return
    rows=[];patchrows=[]
    for selected in read(OLD/'MODEL_SELECTION.json'):
        if selected['arm']!='A5':continue
        source=selected['source'];model=old_build('A5',selected['seed']);restore(model,torch.load(ROOT/selected['checkpoint'],weights_only=True,map_location='cpu'))
        f=OLDC/'conditions'/source;x=np.load(f/'V_SELECT_x.npy',mmap_mode='r');s=np.load(f/'V_SELECT_sigma.npy',mmap_mode='r');meta=read(f/'V_SELECT_generator_audit.json')
        values=[]
        for lo in range(0,len(x),32):
            watch.boundary();xx=torch.tensor(np.array(x[lo:lo+32]),device='cuda');ss=torch.tensor(np.array(s[lo:lo+32]),device='cuda')
            clipped=preprocess(xx,ss,'A5');z=model.base.patch(model.base.instance_norm(clipped)[0]);emb=model.base.input_patch_embedding(torch.cat([z,torch.ones_like(z)],-1))
            _,r=robust_scale(xx,ss);discard=model.base.patch((xx-clipped)/r);feat=torch.asinh(discard).clamp(-6,6)
            adjusted=model.adapter(emb,torch.cat([torch.asinh(z).clamp(-6,6),feat],-1));delta=adjusted-emb
            v=torch.stack([delta.square().mean(-1).sqrt(),emb.square().mean(-1).sqrt(),feat.square().mean(-1).sqrt(),discard.square().mean(-1).sqrt()],-1)
            values.append(v.cpu().numpy())
        a=np.concatenate(values);np.save(CACHE/f'{source}_A5_{selected["seed"]}_V_patch_usage.npy',a)
        for state in sorted({m['state'] for m in meta}):
            inds=np.array([i for i,m in enumerate(meta) if m['state']==state]);v=a[inds];flat=v.reshape(-1,4)
            corr=float(np.corrcoef(flat[:,0],flat[:,2])[0,1]) if np.std(flat[:,2])>0 and np.std(flat[:,0])>0 else None
            row=dict(source=source,seed=selected['seed'],state=state,examples=len(v),adapter_delta_rms=float(v[:,:,0].mean()),native_embedding_rms=float(v[:,:,1].mean()),
                     adapter_over_native=float((v[:,:,0]/v[:,:,1].clip(1e-12)).mean()),discarded_feature_rms=float(v[:,:,2].mean()),discarded_coordinate_rms=float(v[:,:,3].mean()),adapter_discarded_correlation=corr)
            rows.append(row)
            for patch in range(32):patchrows.append(dict(source=source,seed=selected['seed'],state=state,patch=patch,adapter_rms=float(v[:,patch,0].mean()),native_rms=float(v[:,patch,1].mean()),discarded_feature_rms=float(v[:,patch,2].mean())))
        del model;cleanup();print('ADAPTER_AUDIT',source,selected['seed'],flush=True)
    csvwrite(OUT/'adapter_usage.csv',rows);csvwrite(OUT/'adapter_usage_by_patch.csv',patchrows)

def finish_diagnosis():
    from experiments.outlier_signal_peft_v1_20260917.report import table
    f=pd.read_csv(OUT/'clip_exposure_summary.csv');a=pd.read_csv(OUT/'adapter_usage.csv');comparisons={}
    for source in SOURCES:
        train=f[(f.source==source)&(f.role=='TRAIN')&(f.state=='SHIFT')].iloc[0];ev=f[(f.source==source)&(f.role=='E_DISCOVERY')&(f.state=='SHIFT8')].iloc[0]
        comparisons[source]=dict(train_shift_nonzero=float(train.shift_nonzero_residual),E_shift8_nonzero=float(ev.shift_nonzero_residual),
             train_shift_fraction=float(train.shift_clipped_fraction),E_shift8_fraction=float(ev.shift_clipped_fraction),
             train_normalized_mass=float(train.normalized_mass),E_shift8_normalized_mass=float(ev.normalized_mass),
             descriptive_tag='TRAIN_EXPOSURE_GAP' if train.shift_clipped_fraction<ev.shift_clipped_fraction else 'NO_EXPOSURE_GAP',
             hard_clip='HARD_CLIP_INFORMATION_LOSS: quantitative exposure and historical shift error direction; not a causal isolation',
             underutilization='No arbitrary near-zero threshold; actual adapter ratios reported, no categorical zero-use claim')
    save(OUT/'failure_hypothesis.json',dict(optimizer_updates=0,comparisons=comparisons,descriptive_only=True,no_performance_gate=True,
         per_example_cache=dict(path=str((CACHE/'clip_exposure_per_example.csv').relative_to(ROOT)),sha256=sha(CACHE/'clip_exposure_per_example.csv')),
         old_residual_ablation_sha256=sha(OLD/'A5_validation_diagnostics.json')))
    selected=f[(f.state.str.startswith('SHIFT'))|((f.role=='TRAIN')&(f.state.isin(['POINT','BURST'])))][['source','role','state','clipped_fraction','normalized_mass','longest_run','shift_clipped_fraction','shift_nonzero_residual']]
    text='# 기존 실패 기전 진단 — optimizer update 0회\n\n기존 TRAIN/V/E 입력과 선택 A5 네 모델을 그대로 사용했다. E는 이미 사용한 개발 기간의 서술 감사이며 새 선택에 쓰지 않는다.\n\n'+table(selected)+'\n\n## V에서 기존 A5 사용량\n\n'+table(a[a.state.isin(['SHIFT4','SHIFT8','POINT8','BURST8'])])+'\n\nTRAIN은 4r0 shift만, E는 4/8r0를 포함한다. clipped fraction과 shift-affected fraction을 구분한다. nonzero_residual은 전체 입력, shift_nonzero_residual은 실제 shift 위치만 세었다. residual_mass_over_abs_delta는 shift 위치에서 버린 절대량 합/|delta|이며 위치 수로 나눈 비율은 아니다. last64_same_sign은 양/음 중 큰 count다.\n\n연속 clip 길이는 같은 부호 제약 없는 최장 run이다. 어댑터 RMS 비율은 patch별 비율의 평균이며 상관계수는 scenario 안의 example×patch 쌍이다. 기존 zero/permutation V 진단에서 경로 의존성은 관찰됐지만, 대조군보다 좋은 방법이라는 증거는 아니다. 노출 차이는 가능한 설명이지 단독 인과 증명이 아니다.\n\n'+json.dumps(comparisons,ensure_ascii=False,indent=2)+'\n\n이 진단의 크기를 성능 gate로 쓰지 않는다. B0–B5는 동일 고정 예산으로 직접 비교한다.\n'
    (OUT/'DIAGNOSTIC_REPORT.md').write_text(text)
