"""Post-completion CPU audit and descriptive evidence; no fitting or selection."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/delta_adapter_comparison_20260919'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(f):
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))
def main():
    assert read(OUT/'PUBLICATION_AUDIT.json')['status']=='VERIFIED'
    assert read(OUT/'status.json')['execution']=='COMPLETE_COMPUTE'
    origin=pd.read_csv(OUT/'ORIGIN_SCORES.csv.gz');effects=pd.read_csv(OUT/'EFFECTS.csv');raw=pd.read_csv(OUT/'RAW_SCORES.csv')
    checked=0;coverage=[]
    # Independent representation: resample whole-week sums, not origin weights.
    for key,g in origin.groupby(['panel','kind','stage','condition']):
        panel,kind,stage,condition=key
        assert sorted(g.seed.unique().tolist())==[81551,81552]
        t=g.groupby(['origin','arm']).nmae.mean().unstack()
        assert not t.isna().any().any()
        day_length=24 if panel.startswith('electricity') else 96
        week=t.index.to_numpy()//(7*day_length)
        summed=t.groupby(week).sum();sizes=t.groupby(week).size().to_numpy()
        n=len(summed);random=np.random.default_rng(91942)
        # Same sealed draws; aggregate by week before replicating those draws.
        draws=random.integers(0,n,size=(2000,n))
        samples=summed.to_numpy()[draws].sum(axis=1)/sizes[draws].sum(axis=1)[:,None]
        cols={a:i for i,a in enumerate(summed.columns)}
        rows=effects[(effects.panel==panel)&(effects.kind==kind)&(effects.stage==stage)&(effects.condition==condition)]
        assert len(rows)==8
        for r in rows.itertuples():
            boot=100*(1-samples[:,cols[r.proposed]]/samples[:,cols[r.baseline]])
            actual=np.quantile(boot,[.025,.975,.05/8,1-.05/8])
            np.testing.assert_allclose(actual,[r.ci_low,r.ci_high,r.bonf4_low,r.bonf4_high],rtol=1e-9,atol=1e-10)
            checked+=1
        coverage.append(dict(panel=panel,kind=kind,stage=stage,condition=condition,origins=len(t),distinct_index_days=len(np.unique(t.index.to_numpy()//day_length)),index_weeks=n,index_span_days=int((t.index.max()-t.index.min())//day_length+1),seed_count=2))
    pd.DataFrame(coverage).to_csv(OUT/'BOOTSTRAP_SUPPORT.csv',index=False)
    # Existing receipt provenance, tied to the reused checkpoint rather than names alone.
    manifest=read(OUT/'PREDICTIONS.json');resources=[];receipt_hashes={}
    sources={
        'PLAIN':('additive_b0_adapter_v1_20260917','C2'),
        'C3':('additive_b0_adapter_v1_20260917','C3'),
        'MAG_ONLY':('c3_weakness_controls_20260918','MAG_ONLY'),
        'DELTA_XY_BUDGET':('delta_adapter_comparison_20260919','DELTA_XY_BUDGET'),
        'DELTA_XY_DEFAULT':('delta_adapter_comparison_20260919','DELTA_XY_DEFAULT')}
    for source in ['electricity','ettm1']:
        for arm,(folder,original_arm) in sources.items():
            for seed in [81551,81552]:
                record=manifest[f'{source}__standard__{arm}__s{seed}__selected']
                matches=[]
                for path in (ROOT/'results'/folder/'fits').glob(f'{source}_{original_arm}_s{seed}_lr*/receipt.json'):
                    r=read(path)
                    if r['selected']['sha256']==record['checkpoint_sha256']:matches.append((path,r))
                assert len(matches)==1,(source,arm,seed,len(matches))
                path,r=matches[0];assert r['updates']==1024 and r['frozen_unchanged']
                receipt_hashes[str(path.relative_to(ROOT))]=sha(path)
                resources.append(dict(source=source,arm=arm,seed=seed,updates=1024,selected_step=record['step'],parameters=r.get('trainable_parameters',8712),optimizer_seconds=r['optimizer_seconds'],peak_allocated_MiB=r['peak_allocated']/2**20,receipt=str(path.relative_to(ROOT))))
    cost=pd.DataFrame(resources);cost.to_csv(OUT/'MATCHED_REPEAT_RESOURCES.csv',index=False)
    # All-arm point estimates, preserving source/state/seed/stage. No new success gate.
    contrast=[]
    for key,g in raw.groupby(['panel','kind','stage','condition','seed']):
        vals=g.set_index('arm').nmae
        for proposed in ['C3','MAG_ONLY']:
            for baseline in ['B0','PLAIN','C3','MAG_ONLY','DELTA_XY_BUDGET','DELTA_XY_DEFAULT']:
                if baseline==proposed:continue
                contrast.append(dict(zip(['panel','kind','stage','condition','seed'],key),proposed=proposed,baseline=baseline,proposed_nmae=vals[proposed],baseline_nmae=vals[baseline],gain_pct=100*(1-vals[proposed]/vals[baseline])))
    con=pd.DataFrame(contrast);con.to_csv(OUT/'DESCRIPTIVE_SEED_CONTRASTS.csv',index=False)
    states=['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT']
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),sharey=True)
    colors={'C3':'#287fa7','MAG_ONLY':'#36864c','DELTA_XY_BUDGET':'#bd9400','DELTA_XY_DEFAULT':'#a9554d'}
    for ax,panel in zip(axes,['electricity','electricity_transfer','ettm1']):
        g=raw[(raw.panel==panel)&(raw.kind=='standard')&(raw.stage=='selected')].groupby(['condition','arm']).nmae.mean().unstack()
        for arm,color in colors.items():
            gain=100*(1-g.loc[states,arm]/g.loc[states,'B0'])
            ax.plot(range(len(states)),gain,marker='o',label=arm,color=color)
        ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(range(len(states)),states,rotation=40,ha='right');ax.set_title(panel);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('nMAE reduction vs frozen B0 (%)');axes[-1].legend(fontsize=8)
    fig.suptitle('Descriptive means of two repeat seeds; reused development evaluation');fig.tight_layout()
    fig.savefig(OUT/'tradeoffs.png',dpi=180);fig.savefig(OUT/'tradeoffs.pdf');plt.close(fig)
    cost_summary=cost.groupby('arm',as_index=False)[['parameters','optimizer_seconds','peak_allocated_MiB']].mean()
    sub=con[(con.panel=='electricity_transfer')&(con.kind=='standard')&(con.stage=='selected')&(con.condition=='SHIFT8')]
    focus=sub[sub.baseline.isin(['B0','PLAIN','MAG_ONLY'])][['proposed','baseline','seed','gain_pct']]
    support=pd.DataFrame(coverage);support=support[(support.kind=='standard')&(support.stage=='selected')&(support.condition=='SHIFT8')]
    crossing=pd.read_csv(OUT/'QUANTILE_CROSSING.csv');crossing=crossing.groupby('arm',as_index=False)[['crossing_pairs','total_pairs']].sum();crossing['crossing_fraction']=crossing.crossing_pairs/crossing.total_pairs
    text='''# 직접 비교의 통계·비용·방법론 해석 보완

기존에 봉인된 비교의 CPU 재검산과 기술 통계다. 새 학습·모델 선택·성공 기준은 없다. 과거 분석 논문을 방법론 논문이라고 이름만 바꾸지 않는다.

## 신뢰구간이 다루는 불확실성

주 단위 합계를 먼저 구한 뒤 봉인된 2,000개 재표집에 적용하는 별도 구현으로 모든 효과행의95%/family4 구간을 재계산했다. 원 runner는 origin 가중치를 사용한다. 두 계산이 일치했다. family4 보정의 주장은 이번 단위의 사전 지정 네 비교에만 적용한다. 이전 후보 탐색 전체의 선택 편향을 보정하거나 이미 본 E를 확증 자료로 바꾸는 것은 아니다. 나머지 수치는 탐색적이다.

이는 두 학습 seed와 현재 채널·자료를 고정한 시간 블록 변동성이다. seed 모집단·독립 source·최적 설정 탐색 불확실성까지 포함하지 않는다. index-week는 표본 index 기준7일 블록이며 달력의ISO week라고 부르지 않는다. E 구간은 기존 개발 자료다.

'''+table(support[['panel','origins','distinct_index_days','index_weeks','index_span_days','seed_count']])+'''

## 기존 구성요소 대조를 삭제하지 않는다

다음 전력16계열 selected SHIFT8의 seed별 결과도 함께 읽어야 한다. δ 대조에서의 이득이 C3 지속성 규칙의 추가 가치를 대신 입증하지 않는다. MAG는 관측 크기만 사용하는 더 단순한 대조군이다. 모든 상태·seed·fixed1024 대조는 DESCRIPTIVE_SEED_CONTRASTS.csv에 있다.

'''+table(focus)+'''

## 동일 반복 경로의 자원

각 군의 두 원천×두 반복 seed, 총4개1024-update 경로 평균이다. 선택된 checkpoint hash를 원 receipt와 맞춰서 연결했다. optimizer 시간은 실제 기록이며 검증·선택·I/O의 전체 비용이 아니다. 서로 다른 시각에 기록했으므로 엄밀한 동시 성능 벤치마크나 하드웨어 불변 speedup으로 주장하지 않는다. allocator peak는 PyTorch 할당량이며 nvidia-smi 전체 사용량과 다르다. GPU 메모리와 파라미터 수를 혼동하지 않는다.

'''+table(cost_summary)+'''

검증에서 선택된 실제 step은 다음과 같다. step0은 경로 전체1,024 updates를 실행한 뒤 사전 규칙에 따라 초기 가중치를 선택했다는 뜻이다. δ의 초기 잔차는 random이므로 δ step0을 B0와 같다고 볼 수 없다. 추가 학습의 이득과 초기 구조의 영향을 혼동하지 않는다.

'''+table(cost[['source','arm','seed','selected_step']])+'''

이 파라미터 수는 추가 어댑터만 센다. 모두 기존 B0 LoRA294,912개와 frozen Chronos backbone을 따로 유지하므로 전체 저장·배포 크기가8,712개라는 뜻이 아니다. B0를 얻는 사전 적응 비용도 추가 학습 비용에 포함하지 않았다. δ는 입력까지 gradient를 전파하지만 기존 latent adapter는 다른 위치에 있어 파라미터 수만으로 훈련 메모리·시간을 예측할 수 없다. 위치·초기화·연산·출력 제한을 한 번에 바꾼 비교이므로 속도/효과 차이의 단일 원인을 분리한 실험은 아니다.

## 분위수 순서 진단

'''+table(crossing)+'''

인접 quantile 쌍의 crossing 비율을 저장된 모든 view에서 합산했다. selected/fixed 동일 checkpoint의 중복 view도 포함된 기술 통계이며 독립 표본 수로 쓰지 않는다. 이 지표를 보고 재정렬·재보정하거나 선택 규칙을 바꾸지 않았다.

## 방법론 주장에 남은 과제

이번 단위는 공식 XY cell을 고정 과제에 옮긴 직접 비교를 채웠다. 원 논문의 전 설정 재현과 모든 robust PEFT 대비 우위는 이번 실험이 확인한 범위 밖이다. 방법론 논문에 모든 데이터·상태에서의 승리를 요구하는 것은 아니다. 현재 남은 핵심은 C3 지속성 요소의 추가 가치와 신규성, 이미 반복 사용한 E 밖에서의 확인이다. 실제 확인된 좁은 이득은 남기되 논문 전체 목표를 완료로 표시하지 않는다. 추가 실험은 자동 시작하지 않았다.
'''
    (OUT/'COMPARISON_REVIEW_KO.md').write_text(text)
    files=['BOOTSTRAP_SUPPORT.csv','MATCHED_REPEAT_RESOURCES.csv','DESCRIPTIVE_SEED_CONTRASTS.csv','tradeoffs.png','tradeoffs.pdf','COMPARISON_REVIEW_KO.md']
    audit=dict(status='VERIFIED',new_optimizer_updates=0,new_inference=0,effect_intervals_recomputed=checked,source_receipt_count=len(receipt_hashes),source_receipt_hashes=receipt_hashes,conditional_on_two_seeds=True,script_sha256=sha(Path(__file__)),input_hashes={n:sha(OUT/n) for n in ['ORIGIN_SCORES.csv.gz','EFFECTS.csv','RAW_SCORES.csv','PREDICTIONS.json','PUBLICATION_AUDIT.json','QUANTILE_CROSSING.csv']},output_hashes={n:sha(OUT/n) for n in files})
    (OUT/'COMPARISON_AUDIT.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print('COMPARISON_AUDIT_VERIFIED',checked,len(receipt_hashes))
if __name__=='__main__':main()
