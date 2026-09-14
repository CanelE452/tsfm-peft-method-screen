"""One-shot CPU audit/selection/shrinkage; no training or automatic follow-up.

D is deferred when sharing safety cannot be established. Historical E is read
only by A for cached-score verification. B/C load V caches exclusively.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
from tsfm_peft_screen.metrics import score, independent
from tsfm_peft_screen.backbone import QUANTILES

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'results/anchor_window_study_20260914'
RUN='temporal_transfer_diagnostic_v1'
OUT=ROOT/'results'/RUN
RESEARCH=ROOT/'research'/RUN
Q=np.array(QUANTILES,dtype=np.float64)
TOL=1e-10


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def csvwrite(name,rows,fields=None):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def gain(method,reference):
    if not math.isfinite(method) or not math.isfinite(reference) or reference<=0:
        raise ValueError('Relative gain undefined for nonfinite values or nonpositive reference')
    return 100*(reference-method)/reference


def keyed(rows,fields):
    out={}
    for r in rows:
        k=tuple(r[f] for f in fields)
        if k in out:raise ValueError(f'Duplicate identifier: {k}')
        out[k]=r
    return out


def aggregate(rows,datasets,seeds):
    table=keyed(rows,('dataset','seed','arm'))
    expected={(d,s,a) for d in datasets for s in seeds for a in ('native','native_anchor')}
    if set(table)!=expected:raise ValueError('Missing or extra paired identifiers')
    seedrows=[];cells=[]
    for d in datasets:
        pp=[];aa=[]
        for s in seeds:
            p=table[d,s,'native']['loss'];a=table[d,s,'native_anchor']['loss']
            seedrows.append(dict(dataset=d,seed=s,plain_loss=p,anchor_loss=a,
                                 corrected_gain_percent=gain(a,p),old_ratio_percent=100*a/p))
            pp.append(p);aa.append(a)
        source,budget=d.rsplit('_',1)
        cells.append(dict(dataset=d,source=source,budget=int(budget),plain_mean=float(np.mean(pp)),
            anchor_mean=float(np.mean(aa)),gain_of_mean_losses=gain(float(np.mean(aa)),float(np.mean(pp))),
            mean_seed_gains=float(np.mean([gain(a,p) for a,p in zip(aa,pp)]))))
    sources=sorted({c['source'] for c in cells})
    get=lambda source,b:next(c['gain_of_mean_losses'] for c in cells if c['source']==source and c['budget']==b)
    interaction={s:get(s,32)-get(s,233) for s in sources}
    sparse=sum(get(s,32) for s in sources)/len(sources)
    macro=sum((get(s,32)+get(s,233))/2 for s in sources)/len(sources)
    mean_interaction=sum(interaction.values())/len(sources)
    support=sparse>0 and mean_interaction>0 and sum(v>0 for v in interaction.values())>=2
    return dict(seeds=seedrows,cells=cells,source_balanced_macro=macro,sparse_macro=sparse,
        interaction=interaction,mean_interaction=mean_interaction,
        decision='CONDITIONAL_FOLLOWUP_SIGNAL' if support else 'WINDOW_BUDGET_HYPOTHESIS_NOT_SUPPORTED')


def components(p,y,scale):
    p=np.sort(np.asarray(p,dtype=np.float64),axis=2)
    y=np.asarray(y,dtype=np.float64);scale=np.asarray(scale,dtype=np.float64)
    assert p.shape==(len(y),y.shape[1],21,y.shape[2]) and np.isfinite(p).all()
    assert np.isfinite(scale).all() and (scale>0).all()
    valid=np.isfinite(y);e=np.where(valid,y,0)[:,:,None,:]-p
    pin=np.where(valid[:,:,None,:],2*np.maximum(Q[None,None,:,None]*e,(Q[None,None,:,None]-1)*e),0.)
    return pin.sum((2,3))/21/scale[None,:],valid.sum(2)


def pooled(num,count,indices):
    n=count[indices].sum(0);active=n>0
    if not active.any():raise ValueError('No observed targets')
    return float((num[indices].sum(0)[active]/n[active]).mean())


def pick(candidates,scores):
    # Only caller-provided selection-window scores enter this function.
    return min(candidates,key=lambda c:(scores[c['id']],c['step'],c['lr'],c['id']))


def blend(f0,adapted,alpha):
    if alpha==0:return f0.copy()
    if alpha==1:return adapted.copy()
    return f0+alpha*(adapted-f0)


def split_origins(origins,train):
    origins=np.asarray(origins)
    if len(origins)!=16 or not np.all(np.diff(origins)>0):raise ValueError('Expected 16 increasing V origins')
    if max(train)+48>origins[0] or origins[7]+48>origins[8]:raise ValueError('Overlapping target intervals')
    return dict(S=list(range(8)),D_diag=list(range(8,16)),RECENT4=[4,5,6,7],SPREAD4=[0,2,4,7],ALL8=list(range(8)))


def rankcorr(x,y):
    def ranks(a):
        a=np.array(a);return np.array([sum(a<v)+(sum(a==v)+1)/2 for v in a])
    a,b=ranks(x),ranks(y)
    if np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


class Analysis:
    def __init__(self):
        self.cache={};self.errors=[];self.input_hashes={};self.parts_count=0;self.old_e=set()
        self.contract=read(PARENT/'contract.json');self.cfg=self.contract['config'];self.manifest=[]
    def load(self,row,split):
        path=ROOT/row['prediction_file']
        if split=='V' and '_E' in path.name:raise ValueError('E forbidden in B/C')
        if not path.exists():raise FileNotFoundError(path)
        assert sha(path)==row['prediction_sha256'],path
        self.input_hashes[str(path.relative_to(ROOT))]=sha(path)
        if split=='E':self.old_e.add(str(path))
        if str(path) not in self.cache:
            with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
            assert set(d)=={'prediction','target','scale','origins'}
            official=score(d['prediction'],d['target'],d['scale'])['scaled_2pinball']
            scalar=independent(d['prediction'],d['target'],d['scale'])
            err=max(abs(official-scalar),abs(official-row['metrics']['scaled_2pinball']))
            assert err<=TOL;self.errors.append(err)
            d['prediction']=np.sort(d['prediction'].astype(np.float64),axis=2)
            self.cache[str(path)]=d
        return self.cache[str(path)]
    def parts(self,p,y,sc):
        self.parts_count+=1
        n,c=components(p,y,sc)
        assert abs(pooled(n,c,np.arange(len(y)))-score(p,y,sc)['scaled_2pinball'])<=TOL
        return n,c
    def checkpoint(self,row):
        p=ROOT/row['checkpoint_file']
        if not p.exists():return False
        assert sha(p)==row['checkpoint_sha256'],p
        self.input_hashes[str(p.relative_to(ROOT))]=sha(p);return True


def audit(a):
    evals=read(PARENT/'evaluation.json');seal=read(PARENT/'evaluation_seal.json')
    table=keyed(evals,('dataset','seed','arm'))
    selected=keyed(seal['selections'],('dataset','seed','arm'))
    ds=a.cfg['topics']['anchor']['datasets'];seeds=a.cfg['topics']['anchor']['seeds']
    expected={(d,s,arm) for d in ds for s,arm in [(None,'F0')]+[(s,arm) for s in seeds for arm in ('native','native_anchor')]}
    assert set(table)==expected
    assert set(selected)=={k for k in table if k[2]!='F0'}
    missing=[]
    for r in evals:
        if r['arm']!='F0':assert r['selection']==selected[r['dataset'],r['seed'],r['arm']]
        try:a.load(r,'E')
        except FileNotFoundError as exc:missing.append(str(exc))
    result=aggregate([dict(dataset=r['dataset'],seed=r['seed'],arm=r['arm'],loss=r['metrics']['scaled_2pinball'])
                      for r in evals if r['arm']!='F0'],ds,seeds)
    old=read(PARENT/'summary.json')
    for c in result['cells']:
        prior=next(o for o in old['cells'] if o['dataset']==c['dataset'])
        assert abs(c['gain_of_mean_losses']-prior['anchor_gain_percent'])<=TOL
        c['f0_loss']=table[c['dataset'],None,'F0']['metrics']['scaled_2pinball']
        c['anchor_vs_f0_percent']=gain(c['anchor_mean'],c['f0_loss'])
        c['plain_vs_f0_percent']=gain(c['plain_mean'],c['f0_loss'])
        for i,s in enumerate(seeds):
            r=next(r for r in result['seeds'] if (r['dataset'],r['seed'])==(c['dataset'],s))
            assert abs(r['old_ratio_percent']-prior['seed_gains_percent'][i])<=TOL
    for new,prior in [('source_balanced_macro','macro_anchor_gain_percent'),('sparse_macro','sparse_macro_gain_percent'),('mean_interaction','mean_interaction_percentage_points')]:
        assert abs(result[new]-old[prior])<=TOL
    assert result['decision']==old['decision']
    for k,v in result['interaction'].items():assert abs(v-old['sparse_minus_dense_gain_percentage_points'][k])<=TOL
    source=(ROOT/'scripts/run_anchor_window_study.py').read_text()
    relevant='\n'.join(l for l in source.splitlines() if any(k in l for k in ['seed_gains_percent=','macro =','sparse =','interaction =','support =','decision=']))
    result.update(decision_unchanged=True,dependency_explanation='Decision uses sparse macro and source interactions computed from mean arm losses. Erroneous seed ratio column is not an input.',historical_code_excerpt=relevant,missing_E_caches=missing)
    csvwrite('corrected_seed_gains.csv',result['seeds']);write(OUT/'aggregate_replay.json',result)
    (OUT/'ERRATUM.md').write_text('# 파생 개선율 정정\n\n원래 seed 열은 100×anchor/plain이며 개선율이 아니다. 올바른 식은 100×(plain−anchor)/plain이다.\n\n30개 기존 E 원점수 → 식별자로 결합한 12 seed 대조 → 6 cell의 평균 손실 비율 → 3-source 균형 macro와 interaction → 원래 decision을 각각 재계산했다. seed 열의 오류는 현재 확인한 최종 판정을 바꾸지 않는다. 검산기는 원래 예측 primary만 검증했으므로 이 파생 열의 오류를 검출하지 못했다. 이번에는 파생 집계도 별도 단위 테스트로 검증한다.\n\n기존 보고/봉인/소스는 수정하지 않았다. ETTm2 동일 점수는 개선율 0%다. Beijing dense의 plain 대비 개선과 F0 대비 작은 이득을 구분한다.\n')
    return result


def temporal(a):
    fits=read(PARENT/'anchor/fits.json')
    ftab=keyed(fits,('dataset','seed','arm','lr'))
    cfg=a.cfg['topics']['anchor'];datasets=cfg['datasets']
    expected={(d,s,arm,lr) for d in datasets for s in cfg['seeds'] for arm in cfg['arms'] for lr in cfg['learning_rates']}
    assert set(ftab)==expected
    baselines=keyed(read(PARENT/'anchor/baselines.json'),('dataset','split'))
    inventory=[];perorigin=[];comparisons=[];curves=[];shrink=[];diags=[];blocks=[];probe=[]
    for ds in datasets:
        source,budget=ds.rsplit('_',1);spec=a.cfg['data'][source]
        dense=list(range(spec['train'][0],spec['train'][1]+1,spec['train'][2]))
        train=[dense[i] for i in np.linspace(0,len(dense)-1,int(budget),dtype=int)]
        for seed in cfg['seeds']:
            for arm in cfg['arms']:
                key=dict(dataset=ds,source=source,budget=int(budget),seed=seed,arm=arm)
                candidates=[];arrays={};partial=[]
                try:f0=a.load(baselines[ds,'validation'],'V')
                except FileNotFoundError:
                    blocks.append(dict(**key,status='BLOCKED_CACHE_INCOMPLETE',reason='F0 V cache missing'));continue
                origins=f0['origins'];parts=split_origins(origins,train);S,D=parts['S'],parts['D_diag']
                y,sc=f0['target'],f0['scale'];fnum,fcount=a.parts(f0['prediction'],y,sc)
                # Check cached targets/scale against the development-only staged file.
                receipt=a.contract['inputs'][source];dev=ROOT/receipt['development_path']
                assert sha(dev)==receipt['development_sha256'];a.input_hashes[str(dev.relative_to(ROOT))]=sha(dev)
                with np.load(dev,allow_pickle=False) as z:
                    assert np.array_equal(sc,z['scale'])
                    assert np.array_equal(y,np.stack([z['values'][o:o+48].T for o in origins]),equal_nan=True)
                for lr in cfg['learning_rates']:
                    fit=ftab[ds,seed,arm,lr]
                    receiptpath=PARENT/'anchor'/(fit['fit_id']+'_fit.json')
                    assert sha(receiptpath)==fit['fit_receipt_sha256']
                    trajectory=read(PARENT/'anchor'/(fit['fit_id']+'_trajectory.json'))
                    assert [r['step'] for r in trajectory]==[0,150,450,900]
                    for row in trajectory:
                        assert (row['dataset'],row['seed'],row['arm'],row['lr'])==(ds,seed,arm,lr)
                        cid=row['fit_id']+'_step'+str(row['step'])
                        cp=a.checkpoint(row)
                        if int(budget)==233 and lr==3e-5 and row['step']==450:
                            probe.append(dict(**key,checkpoint_file=row['checkpoint_file'],checkpoint_sha256=row['checkpoint_sha256'],available=cp,
                                train_origins=train[-2:],S_probe_origins=origins[6:8].tolist(),D_probe_origins=origins[14:16].tolist(),
                                train_context_intervals=[[o-1024,o] for o in train[-2:]],S_context_intervals=[[int(o)-1024,int(o)] for o in origins[6:8]],D_context_intervals=[[int(o)-1024,int(o)] for o in origins[14:16]],
                                target_horizon=48,channels=list(range(4)),scale=sc.tolist(),development_file=str(dev.relative_to(ROOT)),model_revision=a.contract['model_revision'],lr=lr,step=450))
                        inv=dict(**key,candidate_id=cid,step=row['step'],lr=lr,prediction_file=row['prediction_file'],checkpoint_available=cp,cache_available=False,duplicate_group='',canonical_id='')
                        try:data=a.load(row,'V')
                        except FileNotFoundError:
                            partial.append(cid);inventory.append(inv);continue
                        assert np.array_equal(data['origins'],origins) and np.array_equal(data['scale'],sc)
                        assert np.array_equal(data['target'],y,equal_nan=True)
                        ph=hashlib.sha256(data['prediction'].tobytes()).hexdigest()
                        canonical=next((c for c in candidates if c['step']==0),None) if row['step']==0 else None
                        if canonical:
                            assert np.array_equal(data['prediction'],arrays[canonical['id']]['p'])
                        else:
                            c=dict(id=cid,step=row['step'],lr=lr,prediction_hash=ph);candidates.append(c)
                            num,count=a.parts(data['prediction'],y,sc)
                            arrays[cid]=dict(p=data['prediction'],num=num,count=count)
                        inv.update(cache_available=True,duplicate_group=ph,canonical_id=canonical['id'] if canonical else cid)
                        inventory.append(inv)
                if partial:
                    blocks.append(dict(**key,status='BLOCKED_CACHE_INCOMPLETE',reason=';'.join(partial)));continue
                assert len(candidates)==7
                distinct=len({c['prediction_hash'] for c in candidates})
                if distinct==1:
                    blocks.append(dict(**key,status='NO_DISTINCT_CANDIDATES',reason='All seven predictions identical'))
                for c in candidates:
                    data=arrays[c['id']]
                    for i,o in enumerate(origins):
                        for channel in range(4):
                            perorigin.append(dict(**key,candidate_id=c['id'],origin=int(o),channel=channel,
                                scaled_quantile_mean_pinball_numerator=float(data['num'][i,channel]),valid_target_count=int(data['count'][i,channel]),
                                train_scale=float(sc[channel]),quantile_count=21,active=bool(data['count'][i,channel]>0)))
                # Include F0 sufficient statistics, not origin means, for recomposition.
                for i,o in enumerate(origins):
                    for channel in range(4):
                        perorigin.append(dict(**key,candidate_id='F0',origin=int(o),channel=channel,
                            scaled_quantile_mean_pinball_numerator=float(fnum[i,channel]),valid_target_count=int(fcount[i,channel]),train_scale=float(sc[channel]),quantile_count=21,active=bool(fcount[i,channel]>0)))
                fS,fD=pooled(fnum,fcount,S),pooled(fnum,fcount,D)
                sscore={c['id']:pooled(arrays[c['id']]['num'],fcount,S) for c in candidates}
                dscore={c['id']:pooled(arrays[c['id']]['num'],fcount,D) for c in candidates}
                selected={'FROZEN':dict(id='F0',step=0,lr=0.),'FIXED':next(c for c in candidates if c['lr']==3e-5 and c['step']==150)}
                for rule in ('RECENT4','SPREAD4','ALL8'):
                    scores={c['id']:pooled(arrays[c['id']]['num'],fcount,parts[rule]) for c in candidates}
                    selected[rule]=pick(candidates,scores)
                a.manifest.append(dict(**key,train_last_target_exclusive=max(train)+48,origins=origins.tolist(),splits=parts,
                    candidate_ids=[c['id'] for c in candidates],selections=selected,active_channels_by_block={n:np.flatnonzero(fcount[ii].sum(0)>0).tolist() for n,ii in dict(S=S,D_diag=D,D_front4=D[:4],D_back4=D[4:]).items()},
                    scope='D_diag reuses historical V; no independent holdout'))
                assert all((fcount[ii].sum(0)>0).all() for ii in [S,D,D[:4],D[4:],parts['RECENT4'],parts['SPREAD4']]),'Missing common channels; explicit handling required'
                for rule,c in selected.items():
                    num=fnum if c['id']=='F0' else arrays[c['id']]['num']
                    si=parts.get(rule,S);ss=pooled(num,fcount,si);sl=pooled(num,fcount,S);dl=pooled(num,fcount,D)
                    front,back=pooled(num,fcount,D[:4]),pooled(num,fcount,D[4:])
                    comparisons.append(dict(**key,rule=rule,candidate_id=c['id'],step=c['step'],lr=c['lr'],
                        selection_n_origins=len(si),selection_loss=ss,selection_gain_vs_f0_percent=gain(ss,pooled(fnum,fcount,si)),
                        S_loss=sl,D_loss=dl,F0_S=fS,F0_D=fD,S_gain_vs_f0_percent=gain(sl,fS),D_gain_vs_f0_percent=gain(dl,fD),
                        D_loss_difference_vs_f0=dl-fD,D_front4_loss=front,D_back4_loss=back,
                        D_front4_gain_percent=gain(front,pooled(fnum,fcount,D[:4])),D_back4_gain_percent=gain(back,pooled(fnum,fcount,D[4:])),
                        S_valid_targets=int(fcount[S].sum()),D_valid_targets=int(fcount[D].sum()),D_front4_valid_targets=int(fcount[D[:4]].sum()),D_back4_valid_targets=int(fcount[D[4:]].sum()),
                        S_improves_D_worsens=bool(sl<fS-TOL and dl>fD+TOL)))
                oracle=pick(candidates,dscore);sbest=pick(candidates,sscore)
                diags.append(dict(**key,distinct_predictions=distinct,rank_spearman=rankcorr(list(sscore.values()),list(dscore.values())),
                    S_ties=len(sscore)-len(set(sscore.values())),D_ties=len(dscore)-len(set(dscore.values())),
                    same_R2_R3_id=selected['RECENT4']['id']==selected['SPREAD4']['id'],
                    same_R2_R3_prediction=np.array_equal(arrays[selected['RECENT4']['id']]['p'],arrays[selected['SPREAD4']['id']]['p']),
                    S_best_id=sbest['id'],D_posthoc_oracle_id=oracle['id'],S_selected_D_loss=dscore[sbest['id']],D_posthoc_oracle_loss=dscore[oracle['id']],
                    optimistic_selection_gap=dscore[sbest['id']]-dscore[oracle['id']],
                    no_candidate_beats_F0=not any(v<fD-TOL for v in dscore.values()),
                    useful_candidate_missed_by_ALL8=any(v<fD-TOL for v in dscore.values()) and dscore[sbest['id']]>=fD-TOL))
                fixed=selected['RECENT4'];adapted=arrays[fixed['id']]['p'];local=[]
                for alpha in (0.,.25,.5,.75,1.):
                    p=blend(f0['prediction'],adapted,alpha)
                    assert np.all(np.diff(p,axis=2)>=0)
                    if alpha==0:assert np.array_equal(p,f0['prediction'])
                    if alpha==1:assert np.array_equal(p,adapted)
                    num,_=a.parts(p,y,sc);sl=pooled(num,fcount,S);dl=pooled(num,fcount,D)
                    r=dict(**key,checkpoint_id=fixed['id'],alpha=alpha,S_loss=sl,D_loss=dl,S_gain_vs_f0_percent=gain(sl,fS),D_gain_vs_f0_percent=gain(dl,fD))
                    local.append(r);curves.append(r)
                choice=min(local,key=lambda r:(r['S_loss'],r['alpha']))
                oracle_alpha=min(local,key=lambda r:(r['D_loss'],r['alpha']))
                original=local[-1]['D_loss']
                shrink.append(dict(**key,checkpoint_id=fixed['id'],S_selected_alpha=choice['alpha'],D_loss=choice['D_loss'],F0_D=fD,R2_original_D=original,
                    D_gain_vs_original_percent=gain(choice['D_loss'],original),D_difference_vs_original=choice['D_loss']-original,
                    D_gain_vs_f0_percent=gain(choice['D_loss'],fD),D_posthoc_oracle_alpha=oracle_alpha['alpha'],D_posthoc_oracle_loss=oracle_alpha['D_loss'],
                    interior_shrink_beats_original_and_F0=bool(0<choice['alpha']<1 and choice['D_loss']<min(original,fD)-TOL),
                    no_alpha_beats_F0=not any(r['D_loss']<fD-TOL for r in local)))
    csvwrite('candidate_inventory.csv',inventory)
    if perorigin:csvwrite('per_origin_scores.csv',perorigin)
    if comparisons:csvwrite('temporal_selection_comparisons.csv',comparisons)
    if curves:csvwrite('correction_scale_curves.csv',curves)
    if shrink:csvwrite('correction_scale_selection.csv',shrink)
    write(OUT/'temporal_selection_manifest.json',a.manifest)
    write(OUT/'selection_diagnostics.json',dict(cells=diags,blocked=blocks))
    write(OUT/'gradient_probe_manifest.json',dict(states=probe,max_states=12,max_backward=60,max_forward=256,max_optimizer_updates=0,status='PENDING_RESOURCE_DECISION'))
    return comparisons,shrink,diags,blocks


def gpu_decision():
    # Read telemetry only. No CUDA context, polling loop, reservation or deferred scheduler.
    base=[str(ROOT/'scripts/with_cuda.sh'),'nvidia-smi']
    gpu=subprocess.check_output(base+['--query-gpu=memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    processes=subprocess.check_output(base+['--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
    return dict(status='DEFERRED_GPU_BUSY' if processes else 'BLOCKED_D_NOT_EXECUTED',gpu_csv=gpu,compute_processes_csv=processes,
        reason='Concurrent CPU A-C is safe. External GPU job peak allocation/performance headroom is not established; the existing no-external-compute guard is preserved. No GPU probe is launched or scheduled.' if processes else 'GPU is now free but this CPU runner has no D implementation. Report honestly; D requires a separate implementation.',
        forward_attempts=0,backward_attempts=0,perturbations=0,optimizer_updates=0)


def main():
    assert not OUT.exists(),'Preserve old attempts; choose a new run suffix'
    start=time.monotonic();OUT.mkdir(parents=True)
    history={str(p.relative_to(ROOT)):sha(p) for folder in ('results','research','src','scripts','configs','tests','automation') for p in (ROOT/folder).rglob('*')
             if p.is_file() and '__pycache__' not in str(p) and OUT not in p.parents}
    a=Analysis()
    try:
        for name,h in {**a.contract['source_hashes'],**a.contract['historical_result_hashes']}.items():assert sha(ROOT/name)==h,name
        agg=audit(a);comparisons,shrink,diags,blocked=temporal(a)
        d=gpu_decision();write(OUT/'gradient_probe_status.json',d)
        manifest=read(OUT/'gradient_probe_manifest.json');manifest['status']=d['status'];write(OUT/'gradient_probe_manifest.json',manifest)
        csvwrite('gradient_geometry.csv',[],['dataset','seed','arm','gradient_a','gradient_b','dot','cosine','status'])
        csvwrite('local_perturbation_effects.csv',[],['dataset','seed','arm','direction','epsilon','region','predicted_delta','actual_delta','status'])
        write(OUT/'restoration_checks.json',dict(status='NOT_APPLICABLE_NO_GPU_PERTURBATIONS',checks_performed=0,permanent_weight_changes=0))
        (OUT/'gradient_probe_report.md').write_text('# D GPU 진단\n\n'+d['status']+'\n\n'+d['reason']+'\n\n'+d['gpu_csv']+'\n'+d['compute_processes_csv']+'\n\nGPU forward/backward 0, 교란 0, 신규 fit 0, optimizer update 0. 국소 목적/시간 충돌 가설을 판정하지 않는다.\n')
        write(OUT/'verification_scope.json',dict(A_E_caches_replayed=len(a.old_e),total_unique_prediction_caches_replayed=len(a.cache),
            max_score_absolute_error=max(a.errors),float64_primary_tolerance=TOL,CPU_sufficient_statistic_reaggregations=a.parts_count,
            B_C_source='historical V only',missing_A_E_caches=agg['missing_E_caches'],blocked_cells=blocked,
            source_count=3,seed_role='optimization repeat',budget_role='intervention on shared targets',new_E_forecasts=0))
        write(OUT/'analysis_summary.json',dict(selection_cells=diags,comparisons=comparisons,shrinkage=shrink))
        for name,h in {**history,**a.input_hashes}.items():assert sha(ROOT/name)==h,name
        write(OUT/'source_and_history_hashes.json',dict(historical_and_source_files=history,predictions_checkpoints_and_development_files=a.input_hashes,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),all_unchanged=True))
        write(OUT/'execution_receipt.json',dict(status='A_B_C_COMPLETE_D_'+d['status'],exit_code=0,new_fits=0,optimizer_updates=0,
            GPU_forward_attempts=0,GPU_backward_attempts=0,discarded_perturbations=0,old_cached_E_arrays_read=len(a.old_e),old_E_new_forecasts=0,new_E_forecasts=0,
            CPU_prediction_caches_replayed=len(a.cache),CPU_reaggregations=a.parts_count,historical_files_unchanged=len(history),
            weight_cache_files_hash_verified=sum(p.endswith('.pt') for p in a.input_hashes),wall_seconds=time.monotonic()-start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,D_status=d['status'],automatic_followup=False))
        print(json.dumps(read(OUT/'execution_receipt.json'),indent=2))
    except BaseException as exc:
        write(OUT/'execution_error.json',dict(status='EXECUTION_ERROR',error=f'{type(exc).__name__}: {exc}',exit_code=1,new_fits=0,optimizer_updates=0))
        raise


if __name__=='__main__':main()
