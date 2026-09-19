"""Finalize only the completed, sealed delta comparison. No model training."""
from pathlib import Path
import hashlib,json,math,sys,subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/delta_adapter_comparison_20260919'
HERE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def md_table(frame):
    columns=list(frame.columns)
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(f'{x:.6f}' if isinstance(x,float) else str(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))

def main():
    assert read(OUT/'status.json')['execution']=='COMPLETE_COMPUTE'
    verification=read(OUT/'VERIFICATION.json');assert verification['prediction_views']==144
    seal=read(OUT/'SEAL.json')
    for p,h in seal['hashes'].items():assert sha(ROOT/p)==h,p
    ledger=[json.loads(s) for s in (OUT/'UPDATE_LEDGER.jsonl').read_text().splitlines()]
    assert len(ledger)==16384 and len({(r['fit'],r['step']) for r in ledger})==16384
    assert len((OUT/'SMOKE_LEDGER.jsonl').read_text().splitlines())==8
    receipts=[read(p) for p in (OUT/'fits').glob('*/receipt.json')];assert len(receipts)==16
    resources=[];selection_rows=[];checkpoint_count=0
    for r in receipts:
        records=[v for v in ledger if v['fit']==r['fit']]
        assert [v['step'] for v in records]==list(range(1,1025))
        assert r['frozen_unchanged'] and r['buffers_unchanged']
        assert min(r['checkpoints'],key=lambda c:(c['objective'],c['step']))==r['selected']
        for c in r['checkpoints']:
            assert sha(ROOT/c['checkpoint'])==c['sha256'];checkpoint_count+=1
            selection_rows.append(dict(source=r['source'],arm=r['arm'],seed=r['seed'],lr=r['lr'],step=c['step'],validation_nmae=c['objective']))
        np.testing.assert_allclose(math.fsum(v['seconds'] for v in records),r['optimizer_seconds'],rtol=1e-11,atol=1e-10)
        resources.append({k:r[k] for k in ['fit','source','arm','seed','lr','trainable_parameters','optimizer_seconds','validation_seconds','invocation_wall_seconds','peak_allocated','peak_reserved']})
    choices=read(OUT/'LR_SELECTION.json')
    for source,arms in choices.items():
        for arm,c in arms.items():
            runs=[r for r in receipts if r['source']==source and r['arm']==arm and r['seed']==81550]
            assert len(runs)==2
            expected=min(runs,key=lambda r:(r['selected']['objective'],r['lr'],r['selected']['step']))
            assert c['lr']==expected['lr'] and c['selection_fit']==expected['fit']
    pred=read(OUT/'PREDICTIONS.json');assert len(pred)==144
    marker=read(OUT/'ALL_PREDICTIONS_SAVED.json');assert marker['manifest_sha256']==sha(OUT/'PREDICTIONS.json')
    eseal=read(OUT/'EVALUATION_SEAL.json');assert eseal['selection_sha256']==sha(OUT/'MODEL_SELECTION.json') and eseal['at']<marker['at']
    delta_new_inference=sum(r['arm'].startswith('DELTA') and not r['reused'] for r in pred.values())
    old=read(OUT/'REUSED_PREDICTIONS.json');assert len(old)==96
    for k,v in old.items():assert pred[k]==v
    checked=set();crossing=[]
    for k,r in pred.items():
        if r['path'] not in checked:assert sha(ROOT/r['path'])==r['sha256'];checked.add(r['path'])
        if r['arm'].startswith('DELTA'):
            assert r['source']==('electricity' if r['panel'].startswith('electricity') else 'ettm1')
            assert r['seed'] in [81551,81552]
            selected=[v for v in read(OUT/'MODEL_SELECTION.json') if all(v[f]==r[f] for f in ['source','arm','seed','stage'])]
            assert len(selected)==1 and selected[0]['B0_sha256']==r['B0_sha256'] and selected[0]['sha256']==r['checkpoint_sha256']
        p=np.load(ROOT/r['path'],mmap_mode='r');count=total=0
        for lo in range(0,len(p),256):
            chunk=p[lo:lo+256];mask=chunk[:,1:]<chunk[:,:-1];count+=int(mask.sum());total+=mask.size
        crossing.append(dict(view=k,arm=r['arm'],panel=r['panel'],kind=r['kind'],stage=r['stage'],seed=r['seed'],crossing_pairs=count,total_pairs=total,crossing_fraction=count/total))
    pd.DataFrame(crossing).to_csv(OUT/'QUANTILE_CROSSING.csv',index=False)
    pd.DataFrame(resources).to_csv(OUT/'RESOURCES.csv',index=False)
    pd.DataFrame(selection_rows).to_csv(OUT/'VALIDATION_GRID.csv',index=False)
    f=pd.read_csv(OUT/'ORIGIN_SCORES.csv.gz');raw=pd.read_csv(OUT/'RAW_SCORES.csv');effects=pd.read_csv(OUT/'EFFECTS.csv')
    keys=['panel','kind','arm','seed','stage','step','condition'];again=f.groupby(keys,as_index=False)[['nmae','mae','pinball']].mean()
    both=again.merge(raw,on=keys,suffixes=('_again','_saved'),validate='one_to_one');assert len(both)==len(raw)
    for field in ['nmae','mae','pinball']:np.testing.assert_allclose(both[field+'_again'],both[field+'_saved'],rtol=1e-10,atol=1e-11)
    for r in effects.itertuples():
        z=raw[(raw.panel==r.panel)&(raw.kind==r.kind)&(raw.stage==r.stage)&(raw.condition==r.condition)]
        a=z[z.arm==r.proposed].nmae.mean();b=z[z.arm==r.baseline].nmae.mean()
        np.testing.assert_allclose([a,b,100*(1-a/b)],[r.proposed_nmae,r.baseline_nmae,r.gain_pct],rtol=1e-10,atol=1e-10)
    # Check reused scalar outputs against their already published source.
    published=pd.read_csv(ROOT/'results/temporal_response_peft_20260919/RAW_SCORES.csv')
    z=raw[raw.arm.isin(['B0','PLAIN','C3','MAG_ONLY'])].merge(published,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(z)==len(raw[raw.arm.isin(['B0','PLAIN','C3','MAG_ONLY'])])
    for field in ['nmae','mae','pinball']:np.testing.assert_allclose(z[field+'_new'],z[field+'_old'],rtol=1e-10,atol=1e-10)
    old_replay_count=len(z)
    gpu=[json.loads(s) for s in (OUT/'gpu_delta.jsonl').read_text().splitlines()]
    unapproved=sum(any(not a['own'] and not a.get('allowed_desktop',False) for a in r['apps']) for r in gpu);assert unapproved==0
    primary=effects[effects.primary_family];assert len(primary)==4
    primary.to_csv(OUT/'PRIMARY_COMPARISONS.csv',index=False)
    tables=raw[raw.stage=='selected'].groupby(['panel','condition','arm'],as_index=False).nmae.mean()
    summary=tables[tables.condition.isin(['REFERENCE','FAULT','SHIFT8'])].pivot(index=['panel','condition'],columns='arm',values='nmae').reset_index()
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=False)
    arm_order=['B0','PLAIN','C3','MAG_ONLY','DELTA_XY_BUDGET','DELTA_XY_DEFAULT']
    for ax,panel in zip(axes,['electricity','electricity_transfer','ettm1']):
        z=tables[(tables.panel==panel)&(tables.condition=='SHIFT8')].set_index('arm').loc[arm_order]
        ax.bar(range(6),z.nmae,color=['#999999','#777777','#298bb6','#418b52','#dfbd59','#a95952'])
        ax.set_xticks(range(6),['B0','Plain','C3','Magnitude','Delta-7','Delta-512'],rotation=40,ha='right')
        ax.set_title(panel);ax.set_ylabel('SHIFT8 nMAE (lower is better)')
    fig.suptitle('Matched B0 / two repeat seeds / reused development E');fig.tight_layout()
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    param_summary=pd.DataFrame(resources).groupby('arm',as_index=False)[['trainable_parameters','optimizer_seconds','peak_allocated']].mean()
    param_summary['peak_allocated_MiB']=param_summary.pop('peak_allocated')/2**20
    primary_table=primary[['proposed','baseline','proposed_nmae','baseline_nmae','gain_pct','bonf4_low','bonf4_high','seed_gains']]
    narrow=raw[(raw.panel=='electricity_transfer')&(raw.kind=='standard')&(raw.stage=='selected')&(raw.condition=='SHIFT8')].groupby('arm').nmae.mean()
    c3_plain_gain=100*(1-narrow['C3']/narrow['PLAIN'])
    c3_mag_gain=100*(1-narrow['C3']/narrow['MAG_ONLY'])
    adverse=effects[(effects.proposed=='C3')&(effects.baseline=='DELTA_XY_DEFAULT')&(effects.panel=='electricity_transfer')&(effects.kind=='shape')&(effects.stage=='selected')&(effects.condition=='STEP12_D63')].iloc[0]
    lines=[]
    for r in primary.itertuples():
        strength='양의 구간' if r.bonf4_low>0 else '음의 구간' if r.bonf4_high<0 else '0 포함'
        lines.append(f'- {r.proposed} 대 {r.baseline}: {r.gain_pct:+.3f}%, family4 보정95% 구간 [{r.bonf4_low:.3f}, {r.bonf4_high:.3f}] ({strength}).')
    text=f'''# δ-Adapter 공개 XY cell과의 통제 비교

**16/16 본학습·선택·144개 예측 view의 채점·독립 검산 완료.** 이 완료 상태는 새 PEFT 방법론의 신규성이나 논문 PASS를 뜻하지 않는다. C3/MAG를 수정하지 않고 가까운 선행의 실제 비교를 추가했다.

## 실행과 구현 범위

새16fits = 두 원천×두δ구성×selection seed81550의두LR(8fits) + 선택LR의81551/81552 반복(8fits). 본학습16,384updates, 별도smoke8updates. 기존 B0/PLAIN/C3/MAG의96개 view는 hash 검증 후 재사용했다. 새δ48view(새 전체 추론{delta_new_inference}개, 같은 checkpoint 재사용{48-delta_new_inference}개)는 선택을 봉인한 뒤 저장했으며, 동일checkpoint alias를 제외한 고유 파일 수는 전체{len(checked)}개다. 전체 예측 저장 후 새 E정답을 채점했다. 원점별/채널별 원점수와 불리한 조건을 모두 보관한다.

공식 `Anoise/Adapter` commit0add06e의 additive XY cell을 독립 구현하고 입력 길이512/64×hidden7/512의4개 CPU 경우에서 output/input-gradient/batch permutation의 exact parity를 확인했다. 두δ×두원천의 실제 모델에서 학습·off 경로 B0 동일성·동결 가중치 보존·fresh checkpoint 복원을 검사했다. 원래 random residual 초기화를 유지해 초기 출력은 B0와 다를 수 있다.

공개 기본 폭512와8,712개 기존 어댑터에 가까운 폭7을 함께 사용했다. 작은δ는8,766개로0.62% 더 많고, 기본δ는1,116,736개다. 동일 parameter count라고 쓰지 않는다. 공개 cell을512→64 예측에 연결하면서 output 입력 차원을64로 지정했고, 관측 평균/기존 TRAIN sigma로 단위를 맞췄다. output cell은9개 분위수에 공유했다. MSE 대신 기존 normalized2pinball·동일 synthetic TRAIN/V를 사용하므로 **공식 논문 전체 재현이 아니라 현재 과제로 옮긴 통제 비교**다. 공식 Y-only/feature-selector/quantile-calibrator와 같은 실험이 아니다.

## 사전 주 비교

양수는 C3/MAG가δ보다 낮은 오차라는 뜻이다. 이 비교는 과거에 확인한 전력16계열 SHIFT8의 좁은 주장에 대응하며 모든 상태의 범용 우위를 뜻하지 않는다.

{md_table(primary_table)}

{chr(10).join(lines)}

같은 전력16계열 SHIFT8에서 C3는 PLAIN보다{c3_plain_gain:.3f}% 좋지만 MAG 대비 이득은{c3_mag_gain:+.3f}%다. 즉 **가까운 선행 대비 이득은 확인됐으나 지속성 규칙을 추가할 이유는 여전히 입증되지 않았다.** 모든 source·상태에서 이겨야 한다는 판정은 사용하지 않는다. 기여로 주장한 구성요소와 단순 대조의 차이를 별도로 요구하는 것이다.

## 원자료·오류·변화의 원점수

{md_table(summary)}

나머지 상태와9개 변화 형태, fixed1024, seed별 값은 [RAW_SCORES.csv](RAW_SCORES.csv), [EFFECTS.csv](EFFECTS.csv), [CHANNEL_SCORES.csv](CHANNEL_SCORES.csv)에 있다. 같은 과거의 PULSE 손해도 삭제하지 않았다. ETTm1 또는 원자료/오류 조건에서의 손해를 전체 평균으로 감추지 않는다. C3와MAG 사이의 과거 판정은 이 비교로 바뀌지 않는다. 특히 전력16계열의 더 긴 STEP12_D63 형태에서 C3 대 기본폭δ의 이득은{adverse.gain_pct:+.3f}%로 손해다. SHIFT8의 양성을 모든 변화 형태로 확대하지 않는다.

## 자원과 남은 한계

다음은16개경로의 군별 평균이며 LR선택 경로도 포함한다. 실제 시간은 [RESOURCES.csv](RESOURCES.csv)에 경로별로 남겼다. 같은update 수는 같은FLOPs나같은wall time과 같지 않다.

{md_table(param_summary)}

GPU guard 표본{len(gpu)}개, 비승인 외부 compute0개, 최소 여유{min(r['free_mib'] for r in gpu)}MiB. RustDesk만 허용했다. 별도 quantile crossing 비율은 [QUANTILE_CROSSING.csv](QUANTILE_CROSSING.csv)에 있으며 이는 사후 진단이고 선택 기준을 바꾸지 않는다.

가까운 선행을 추가했어도 C3 고유의 지속성 요소가 MAG보다 필요하다는 근거가 자동으로 생기지는 않는다. 작은 latent adapter와 boundary adapter의 차이는 위치·연산·초기화가 함께 다르므로 이 비교만으로 개별 원인의 인과기여를 정하지 않는다. 데이터는 이미 사용한 개발E이며 전력의16개 다른계열도 독립자료가 아니다. 실제 센서사건 label·독립확인·정식 선행의전체 설정 재현·충분한 신규성은 남아 있다. 확인된 좁은 이득은 보존하지만 방법론 논문의 전체 목표는 아직 완료로 표시하지 않는다.

모델·origin·조건·loss·LR·seed·update·checkpoint·선택 규칙은 사전 봉인을 유지했다. 새 후보·추가 학습을 자동 실행하지 않았다. [고정 계약](../../experiments/delta_adapter_comparison_20260919/PROTOCOL.md), [공식 코드 receipt](PRIOR_CODE_RECEIPTS.json), [독립 검산](PUBLICATION_AUDIT.json), [별도 bootstrap·비용·seed 검토](COMPARISON_REVIEW_KO.md), [재검산 절차](../../research/delta_adapter_review_20260919/REPRODUCTION_KO.md). 원자료·모델·예측 cache는 로컬 보관이며 GitHub에는 코드·해시·점수·그림이 있다.
'''
    (OUT/'REPORT.md').write_text(text)
    (OUT/'FINAL_DECISION.md').write_text('# 최종 결정\n\n실행은 **COMPLETE_CONTROLLED_PRIOR_COMPARISON**. 방법론 논문 목표 달성·논문 PASS와 구분한다.\n\n'+'\n'.join(lines)+f'\n\n같은 주 조건에서 C3는 PLAIN보다{c3_plain_gain:.3f}% 좋고 MAG 대비 이득은{c3_mag_gain:+.3f}%다. 전력의 좁은 효과를 보존하지만 지속성 규칙 고유의 가치를 주장할 근거는 미확보다. 전력16계열 STEP12_D63에서 기본폭δ보다{-adverse.gain_pct:.3f}% 나쁜 결과도 함께 남긴다.\n\n기존 C3/MAG/PLAIN의 가중치와 판정을 보존한다. δ의선행비교 결과를 추가했지만 C3 지속성 고유의가치나MAG의신규성·독립확인을 대신하지 않는다. 새 구조·추가 LR/seed/데이터·후속 학습은0개다. 실제 확인한 좁은 효과만 주장할 수 있으며 전체 목표는 미달이다.\n')
    # Execution metadata/logs change after this subprocess exits. The separate
    # comparison audit depends on this audit; exclude its outputs to avoid cycles.
    excluded={'PUBLICATION_AUDIT.json','gpu_delta.jsonl','gpu_budget.json','delta_wall.json',
              'COMPLETION_HOOK.json','RUN_PROGRESS_SNAPSHOT.json','COMPARISON_AUDIT.json',
              'BOOTSTRAP_SUPPORT.csv','MATCHED_REPEAT_RESOURCES.csv','DESCRIPTIVE_SEED_CONTRASTS.csv',
              'tradeoffs.png','tradeoffs.pdf','COMPARISON_REVIEW_KO.md'}
    audit=dict(status='VERIFIED',new_fits=16,main_updates=16384,smoke_updates=8,duplicate_updates=0,
               checkpoint_hashes_verified=checkpoint_count,prediction_views=144,new_delta_full_inference_views=delta_new_inference,delta_checkpoint_alias_views=48-delta_new_inference,unique_prediction_files=len(checked),
               scalar_metric_checks=verification['scalar_metrics'],raw_rows_independently_regrouped=len(both),
               old_score_rows_replayed=old_replay_count,effect_rows_recomputed=len(effects),
               GPU_samples=len(gpu),unapproved_external_compute_samples=unapproved,
               source_seal_files_verified=len(seal['hashes']),goal_achieved=False,paper_pass=False,
               result_hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in excluded and p.suffix!='.log'},
               result_hash_exclusions=sorted(excluded),
               hash_scope_note='Completed experiment artifacts; mutable hook/snapshot/logs and separately audited derivative comparison outputs excluded.',
               report_code_sha256=sha(Path(__file__)))
    (OUT/'PUBLICATION_AUDIT.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print('FINAL_REPORT_VERIFIED',len(receipts),len(pred),checkpoint_count,flush=True)

if __name__=='__main__':main()
