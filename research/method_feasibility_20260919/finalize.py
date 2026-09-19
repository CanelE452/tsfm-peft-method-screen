"""Independent ledger/math checks, plots, and Korean report; no model calls."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/block_gradient_feasibility_20260919'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(p.read_text())


def main():
    seal=read(OUT/'SEAL.json')
    for p,h in seal['hashes'].items(): assert sha(ROOT/p)==h,p
    ledger=[json.loads(s) for s in (OUT/'GRADIENT_LEDGER.jsonl').read_text().splitlines()]
    assert len(ledger)==96
    for k in range(1,49):
        r=[v for v in ledger if v['call']==k]
        assert [v['status'] for v in r]==['INTENT','COMPLETE']
        assert r[1]['at']>=r[0]['at']>seal['at']
        assert r[1]['optimizer_updates']==0
    done=read(OUT/'GRADIENTS_COMPLETE.json')
    assert done['gradient_calls']==48 and done['optimizer_updates']==0
    f=pd.read_csv(OUT/'LAYER_RESULTS.csv')
    a=pd.read_csv(OUT/'AGGREGATE_RESULTS.csv')
    assert len(f)==360 and len(a)==10
    assert f.groupby(['source','arm']).size().eq(36).all()
    assert f.holdout_direction_cosine.isna().sum()==60
    for row in a.itertuples():
        z=f[(f.source==row.source)&(f.arm==row.arm)]
        ip=math.fsum(z.holdout_direction_inner)
        us=math.fsum(z.update_squared_norm)
        hs=math.fsum(z.holdout_gradient_squared_norm)
        np.testing.assert_allclose([ip, math.sqrt(us),ip/math.sqrt(us*hs)],
                                  [row.holdout_direction_inner,row.update_norm,row.holdout_direction_cosine],rtol=1e-11,atol=1e-14)
    independent=[]
    for rec in done['models']:
        assert rec['model_hash_before']==rec['model_hash_after'] and rec['native_forward_equal']
        assert sha(ROOT/rec['path'])==rec['sha256']
        g=np.load(ROOT/rec['path'],mmap_mode='r')
        zero=[n for j,n in enumerate(rec['layers']) if np.count_nonzero(g[:,j])==0]
        assert len(zero)==6 and all(n.endswith('SelfAttention.q') and n.startswith('decoder.') for n in zero)
        # Recompute selected Gram entries by scalar sums over different blocks,
        # independently of (K*M^T*M - Q)/(K-1).
        for j in [0,11,12,23,35]:
            train=np.asarray(g[:12,j],dtype=float)
            hold=np.asarray(g[12:,j],dtype=float).mean(0)
            mean=train.mean(0)
            exact=math.fsum(float(x)*float(y) for x,y in zip(mean.ravel(),hold.ravel()))
            rr=f[(f.source==rec['source'])&(f.layer==rec['layers'][j])&(f.arm=='FULL_GRADIENT')].iloc[0]
            np.testing.assert_allclose(exact,rr.holdout_direction_inner,rtol=1e-11,atol=1e-14)
            for p,q in [(0,0),(0,511),(31,79)]:
                explicit=sum(float(train[b,:,p]@train[c,:,q]) for b in range(12) for c in range(12) if b!=c)/132
                algebra=(12*float(mean[:,p]@mean[:,q])-sum(float(v[:,p]@v[:,q]) for v in train)/12)/11
                np.testing.assert_allclose(explicit,algebra,rtol=1e-10,atol=1e-16)
        independent.append(dict(source=rec['source'],zero_gradient_layers=zero,
                                scalar_dot_layers=5,cross_block_scalar_entries=15))
    gpu=[json.loads(s) for s in (OUT/'gpu_gradient_probe.jsonl').read_text().splitlines()]
    unapproved=sum(any(not p['own'] and not p.get('allowed_desktop',False) for p in v['apps']) for v in gpu)
    assert unapproved==0 and not any(v['busy'] for v in gpu)
    pair=read(HERE/'PAIR_AUDIT.json')
    for p,h in pair['source_hashes'].items(): assert sha(ROOT/p)==h,p
    assert pair['prediction_views']==96 and not pair['affects_TRP_primary_criterion']
    for r in read(HERE/'PRIOR_CODE_RECEIPTS.json'): assert sha(ROOT/r['local'])==r['sha256']
    plt.rcParams.update({'font.size':10,'savefig.bbox':'tight'})
    arms=['RANDOM_ORTHO','MEAN_SVD','SECOND_MOMENT','CROSS_BLOCK']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,source in zip(axes,['electricity','ettm1']):
        z=a[a.source==source].set_index('arm').loc[arms]
        ax.bar(np.arange(4),z.holdout_direction_cosine,color=['#888888','#2076a6','#a59976','#bf604f'])
        ax.set_xticks(np.arange(4),['Random','Mean SVD','Second moment','Cross-block'],rotation=15)
        ax.set_ylim(0,.135);ax.set_title(source);ax.set_ylabel('Later-TRAIN gradient cosine')
        for i,v in enumerate(z.holdout_direction_cosine): ax.text(i,v+.002,f'{v:.4f}',ha='center',fontsize=9)
    fig.suptitle('Zero-update direction diagnostic — not forecast accuracy')
    fig.tight_layout();fig.savefig(OUT/'gradient_direction.png',dpi=180);fig.savefig(OUT/'gradient_direction.pdf');plt.close(fig)
    p=pd.read_csv(HERE/'PAIRED_RISK_MEAN.csv')
    p=p[(p.panel=='electricity_transfer')&(p.stage=='selected')].set_index('arm')
    names=['B0','PLAIN','C3','MAG_ONLY','ANCHOR','SHUFFLE','IDEAL','TRP'];p=p.loc[names]
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(names,p.oracle_pair_floor_nmae,label='Identical-input lower bound',color='#a8bac6')
    ax.bar(names,p.avoidable_excess_nmae,bottom=p.oracle_pair_floor_nmae,label='Excess above bound',color='#bf604f')
    ax.set_ylabel('Equal-weight paired nMAE');ax.set_title('PULSE / persistent SHIFT pair: same history, different future')
    ax.legend(loc='lower right');ax.set_ylim(0,4.2)
    for i,v in enumerate(p.oracle_joint_improvement_upper_bound_pct):ax.text(i,float(p.pair_nmae.iloc[i])+.04,f'{v:.2f}% room',ha='center',fontsize=8)
    fig.tight_layout();fig.savefig(HERE/'paired_risk.png',dpi=180);fig.savefig(HERE/'paired_risk.pdf');plt.close(fig)
    table='\n'.join(f"| {r.source} | {r.arm} | {r.holdout_direction_inner:.9g} | {r.holdout_direction_cosine:.6f} | {r.update_norm:.6f} |" for r in a.itertuples())
    resources='\n'.join(f"| {r['source']} | 24 | {r['wall_seconds']:.2f} | {r['peak_allocated_bytes']/2**20:.2f} |" for r in done['models'])
    report=f'''# 일반 예측 PEFT 초기 방향의 실모델 전제 검사

**실행 완료, 새 방법의 추가 근거 미확보.** 두 원천에서 CROSS_BLOCK은 단순 MEAN_SVD보다 후반 TRAIN gradient와의 정렬이 낮았다. 새 방법론 논문의 목표는 미달이며 이번 검사를 논문 PASS나 실제 예측 성능 실패로 표시하지 않는다.

## 실제로 수행한 것

원본 Chronos-Bolt-small F0에서 Electricity/ETTm1의 기존 raw TRAIN만 사용했다. 학습된 B0/C3/TRP 가중치나 합성 상태를 사용하지 않았다. 첫96일과 마지막96일 사이에 Electricity3270 / ETTm1 9875 time slots의 input+target embargo를 확인했다. 두 집합 안에는 target overlap이 남아 있으므로12블록을 독립 반복이라고 부르지 않는다. Electricity는 timestamp가 없는 index-day다.

GPU gradient 계산48회, 추가 native 출력 동일성 forward2회, **새 학습0 fits·optimizer update0회·V/E 평가0회**다. CPU 수학 검사에는 작은 LoRA의3회 autograd가 별도로 포함된다. 원본 출력 동일성과 전체 모델 state hash 보존을 두 원천 모두 확인했다. 각 층의 rank는8로 고정했다. 36개 층 모두를 집계했으며 불리한 층을 빼지 않았다.

## 원점수

| 원천 | 방식 | 후반 TRAIN과 방향 내적 | 전체 cosine | update 방향 norm |
|---|---|---:|---:|---:|
{table}

내적이 양수라는 것은 해당 방향의 음의 미소 update가 후반 TRAIN 손실을 줄일 것이라는 **1차 근사**다. 실제 Adam 학습·장기 경로·예측 오차의 개선율은 아니다. CROSS가 MEAN보다 낮다는 것은 이번 초기화 가설의 직접적인 초기 근거가 없다는 뜻이며, 모든 후속 비선형 학습의 실패를 증명하지 않는다. FULL_GRADIENT는 rank 제한 없는 참고값이므로 동등 예산 PEFT 대조가 아니다.

평균 gradient의 방향을 쓰는 알려진 대조 자체는 random 직교 초기 방향보다 두 원천에서 훨씬 정렬이 높았다. 이를 우리의 새 기여로 세지 않는다. CROSS의 covariance subtraction은 이 고정 자료에서 후반 방향 정렬을 더 높이지 못했다. 진단을 본 뒤 rank·날짜 블록·감산 계수·seed를 바꾸지 않았다.

## 0-gradient 처리와 비용

각 원천 decoder self-attention q6개는24개 블록에서 모두 gradient0이다. 이 계산 경로에서는 decoder query가 하나여서 자기 attention 가중치가1이 되는 구조와 부합한다. 초기 CPU 집계의0분모 assertion이 중단됐으며, 원래 봉인 코드를 보존하고 별도 [집계 보정](ANALYSIS_AMENDMENT_01.json)으로 모든0행을 남겼다. 360행 중60행의 cosine은 미정의다. GPU 재계산·추가 update는0이다. 이 알려진 퇴화 구조 제거를 새 방법이라고 주장하지 않는다.

| 원천 | gradient 호출 | source 처리 초 | peak allocated MiB |
|---|---:|---:|---:|
{resources}

전체 GPU guard 구간44.39초 중 안전 대기30.27초. 최소 여유8239MiB, 비승인 외부 compute 표본0. generic wall 기록의 external_compute_samples는 허용된 RustDesk를 포함하므로 비승인 간섭 횟수가 아니다. CPU eigendecomposition/파일 검산 시간은 위 GPU source 처리 시간과 별개다. gradients의 dense 저장은 진단 비용이며 새 PEFT의 메모리 절감으로 보고하지 않는다.

## 선행과 주장 한계

LoRA-GA 공식 코드는 여러 batch의 gradient를 평균하고 SVD로 A/B 양쪽 방향을 초기화하며 scale와 base offset 보상을 적용한다. 이번 MEAN_SVD는 A의 한쪽 직교 부분공간 비교로, LoRA-GA 전체 재현이 아니다. CROSS의 일반적인 off-diagonal moment 수식은 알려진 통계 구성이다. 시간 블록을 적용한 구현 차이만으로 충분한 신규성이 생기지 않는다. [실제 코드 출처와 문헌 검토](../../research/method_feasibility_20260919/NOVELTY_AND_NEXT_REQUIREMENTS_KO.md).

현재 근거로 CROSS 본학습을 추가 투자 대상으로 올리지 않는다. 이 판단은 수치 오류·자료 부재 때문이 아니며, 새 예측 방법의 실제 성능을 검증 완료했다는 뜻도 아니다. 가까운 초기화 선행과의 구체적 차별성, 같은 계산 예산의 본학습 이득, 다른 기간/원천의 확인이 모두 남아 있다. 새 후보나 추가 학습을 자동 연결하지 않았다.

## 재현과 검산

[봉인 계약](../../research/method_feasibility_20260919/PROTOCOL.md), [블록](BLOCKS.json), [원점 감사](ORIGIN_AUDIT.csv), [360개 층별 점수](LAYER_RESULTS.csv), [독립 검산](PUBLICATION_AUDIT.json), [그림](gradient_direction.png). GPU 단계는 기존 complete marker를 확인해 중복 실행하지 않는다. 최초 분석 entry point는0방향 처리 오류를 그대로 보존했다. 현재 집계 명령은 `OPENBLAS_NUM_THREADS=4 .venv/bin/python research/method_feasibility_20260919/analyse_gradients.py`다.

gradient 배열·사전학습 가중치·raw TRAIN은 ignored 로컬 cache이고 GitHub에는 hash/원점수/코드가 있다. GitHub만으로 수치 replay가 즉시 가능한 것은 아니다. 자료는 기존 개발 연구에서 사용한 원천이므로 새 독립 시험이라고 부르지 않는다.
'''
    (OUT/'REPORT.md').write_text(report)
    (OUT/'FINAL_DECISION.md').write_text('''# 최종 결정

상태: **NO_INITIAL_DIRECTION_SUPPORT**. 실행 완료, 신규성 미확보, 방법론 논문 목표 미달.

- 유지: TRAIN-only 원본 모델 gradient 추출, 날짜 격리·원본 출력·동결 hash 검사, 동일 rank 방향 대조와 전체 결과.
- 현재 투자 대상에서 제외: CROSS_BLOCK 초기화 후보. 두 원천에서 단순 MEAN_SVD보다 초기 방향 정렬이 낮다. 초기 미소 update 검사이며 실제 학습 성능 실패와 다르다.
- 새로운 기여로 주장하지 않을 것: gradient-aware initialization 자체, off-diagonal U-statistic, decoder 단일 query의0-gradient 제거.
- 미실행: 본학습, validation 선택, E 평가, 독립 source, 정식 선행 전체 재현. 각각0회이며 완료했다고 쓰지 않는다.
- 전체 목표: 새 방법론 근거 확보는 **미달**이다. 기존 분석 원고가 있다는 사실로 대체 완료하지 않는다. 이번 단위에서 새 후보·추가 학습은 자동 시작하지 않는다.
''')
    b0=p.loc['B0']
    note=f'''# 동일 과거의 다른 미래 — 방법 설계 전 정량 확인

이는 새 발견이 아니다. 기존 shape generator의 explicit equality assertion과 2026-09-17 보고서가 이미 PULSE/PAIRED_SHIFT의 구별 불가능성을 설명했다. 이번에는 최신8군×3패널×2seed×2checkpoint정책의 **96개 저장 예측 view**에서 입력·예측의 bitwise 동일성과192개 기존 점수를 직접 검산했다. 새 학습·추론0회다.

같은 입력 x의 두 정답 a,b에 동일 예측 p를 내는 경우, 두 상태를 같은 비중으로 평가한 absolute loss는 `|a-b|/2 + distance(p,[min(a,b),max(a,b)])`다. lower bound는 정답을 본 사후 oracle 계산이며 모델 입력·교정 규칙이 아니다. 구간 안에 예측이 있으면 한쪽 오차 개선은 다른 쪽 손해와 상쇄된다. 구간 밖에서는 양쪽을 함께 개선할 여지가 있으므로 모든 개선이 불가능하다고 결론내리지 않는다.

전력16계열 selected B0의 paired nMAE는 **{b0.pair_nmae:.6f}**, 하한은 **{b0.oracle_pair_floor_nmae:.6f}**, 초과분은 **{b0.avoidable_excess_nmae:.6f}**다. 하한 비중{b0.floor_fraction_pct:.3f}%, label-aware oracle를 허용해도 가능한 joint 개선 상한은{b0.oracle_joint_improvement_upper_bound_pct:.3f}%다. 이것은 이 균형 pair의 범위이며, 실제 발생확률이나 전체 예측 오차의 비중을 추정한 값이 아니다.

**TRP의 주 실패 이유로 사용하지 않는다.** 봉인된 TRP primary family4개는 standard SHIFT8이며 PULSE pair가 없다. 따라서 이번 결과로 TRP 판정을 뒤집거나 단순 대조군보다 나쁜 것을 불가능한 과제 탓으로 돌릴 수 없다. C3/IDEAL의 지속 변화 이득도 그 자체로 보존한다.

다음 방법을 구성한다면 구별 불가능한 상태를 모두 맞히는 탐지 주장을 목표에서 제외하고, 관측 가능한 입력에서의 기대 예측 손실·비용·가까운 선행 대비 추가 가치를 직접 보여야 한다. 이는 새로운 방법의 성과가 아니라 과제 정의의 한계다. [전체 원점수](PAIRED_RISK.csv), [검산](PAIR_AUDIT.json), [그림](paired_risk.png).
'''
    (HERE/'PAIR_REPORT_KO.md').write_text(note)
    changed=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
    assert set(changed)<= {'docs/RESULTS_INDEX.md'},changed
    artifact_paths=[p for folder in [OUT,HERE] for p in folder.iterdir() if p.is_file() and p.name!='PUBLICATION_AUDIT.json']
    audit=dict(status='VERIFIED',gradient_calls=48,optimizer_updates=0,new_fits=0,
               ledger_entries=96,layer_rows=360,all_layers_retained=True,
               undefined_cosine_rows=60,aggregate_replays=10,independent_gradient_checks=independent,
               GPU_samples=len(gpu),unapproved_external_compute_samples=unapproved,
               historical_tracked_files_unchanged=True,
               source_seal_hashes_verified=len(seal['hashes']),
               paired_prediction_views_verified=96,goal_achieved=False,paper_pass=False,
               artifact_hashes={str(p.relative_to(ROOT)):sha(p) for p in artifact_paths})
    (OUT/'PUBLICATION_AUDIT.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print('VERIFIED: 48 TRAIN gradient calls; 0 optimizer updates; 96 saved pair views',flush=True)


if __name__=='__main__': main()
