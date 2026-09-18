"""Publish descriptive analyses only, after sealed compute and independent audit."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT, OUT, sha, read, save, check_seal


def table(frame, columns, digits=6):
    labels={'source':'원천','panel':'패널','kind':'구분','condition':'조건','arm':'방법','matched_nmae':'원래 nMAE','swapped_nmae':'교환 nMAE','penalty_pct':'교환 손해(%)','interval_nmae':'차이 구간','B0_gradient_cos_min':'B0 cosine 최소','B0_gradient_cos_max':'B0 cosine 최대','output_J_ratio_mean':'출력/J norm 비','gate_cos_min':'gate cosine 최소','gate_cos_max':'gate cosine 최대','first_batch_common':'공통 표본/32','order_gradient_cos_min':'배치 cosine 최소','order_gradient_cos_max':'배치 cosine 최대','Adam_m_cos_min':'moment cosine 최소','Adam_m_cos_max':'moment cosine 최대','down_relative_drift':'Down 상대 이동','residual_relative_norm':'보정/표현 norm','tanh_saturation_fraction':'tanh 포화 비율'}
    lines=['| '+' | '.join(labels.get(c,c) for c in columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame[columns].itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.{digits}f}' if isinstance(v,(float,np.floating)) else {'electricity':'전력','electricity_transfer':'전력 전이','ettm1':'ETTm1'}.get(str(v),str(v)) for v in row)+' |')
    return '\n'.join(lines)


def report():
    check_seal()
    audit=read(OUT/'INDEPENDENT_AUDIT.json'); assert audit['status']=='VERIFIED'
    dec=pd.read_csv(OUT/'INITIAL_GRADIENT_DECOMPOSITION.csv')
    gate=pd.read_csv(OUT/'INITIAL_GATE_GRADIENTS.csv')
    first=pd.read_csv(OUT/'ORDER_FIRST_BATCH.csv')
    moments=pd.read_csv(OUT/'ORDER_FINAL_MOMENTS.csv')
    tr=pd.read_csv(OUT/'TRAJECTORY.csv'); pairs=pd.read_csv(OUT/'TRAJECTORY_GRADIENT_PAIRS.csv')
    eff=pd.read_csv(OUT/'PAIRING_EFFECTS.csv'); raw=pd.read_csv(OUT/'RAW_SCORES.csv')
    dec['output_to_J_norm_ratio']=dec.output_signal_component_norm/dec.Jacobian_component_norm
    signals=pd.read_csv(OUT/'OUTPUT_SIGNAL_CHECKS.csv').groupby('source').mean(numeric_only=True)
    geom=pd.read_csv(OUT/'FEATURE_GEOMETRY.csv')
    rules=pd.read_csv(OUT/'CHAIN_RULE_CHECKS.csv')
    sources=['electricity','ettm1']; panels=['electricity','electricity_transfer','ettm1']; arms=['C3','MAG_ONLY']
    ratio=dec.groupby('source').output_to_J_norm_ratio.agg(['min','max','mean'])
    summary=[]
    for src in sources:
        d=dec[dec.source==src];g=gate[gate.source==src];f=first[first.source==src];m=moments[moments.source==src]
        summary.append(dict(source=src,B0_gradient_cos_min=d.native_cosine.min(),B0_gradient_cos_max=d.native_cosine.max(),output_J_ratio_mean=d.output_to_J_norm_ratio.mean(),gate_cos_min=g.C3_MAG_gradient_cosine.min(),gate_cos_max=g.C3_MAG_gradient_cosine.max(),first_batch_common=int(f.first_batch_overlap.iloc[0]),order_gradient_cos_min=f.gradient_cosine.min(),order_gradient_cos_max=f.gradient_cosine.max(),Adam_m_cos_min=m.exp_avg_cosine.min(),Adam_m_cos_max=m.exp_avg_cosine.max()))
    summary=pd.DataFrame(summary); summary.to_csv(OUT/'MECHANISM_SUMMARY.csv',index=False)
    dec.to_csv(OUT/'INITIAL_COMPONENT_SUMMARY.csv',index=False)
    gpu=[json.loads(x) for x in (OUT/'gpu_mechanism.jsonl').read_text().splitlines()]
    pred=read(OUT/'PREDICTIONS.json'); new=[r for r in pred.values() if not r['reused_prediction']]
    resource=dict(new_fits=0,optimizer_updates=0,autograd_calls=1196,diagnostic_forward_calls=812,new_E_views=96,reused_E_views=96,small_E_check_forwards=288,checkpoint_probes=160,wall_seconds_to_last_GPU_sample=gpu[-1]['at']-read(OUT/'gpu_budget.json')['started_at'],new_E_inference_seconds=sum(r['inference_seconds'] for r in new),E_peak_allocated_mib=max(r['peak_allocated'] for r in new)/2**20,GPU_min_free_mib=min(r['free_mib'] for r in gpu),GPU_max_used_mib=max(r['used_mib'] for r in gpu),external_training_samples=audit['external_training_samples'])
    save(OUT/'RESOURCE_SUMMARY.json',resource)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42,'svg.fonttype':'none'})
    def savefig(fig,name):
        for suffix in ['png','pdf','svg']:
            target=figdir/f'{name}.{suffix}';fig.savefig(target,dpi=180)
            if suffix=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for k,src in enumerate(sources):
        d=dec[dec.source==src]
        for shift,col,label in [(-.17,'Jacobian_component_norm','Downstream Jacobian'),(.17,'output_signal_component_norm','Output-loss signal')]:
            vals=d[col].to_numpy();axs[0].bar(k+shift,vals.mean(),.3,label=label if k==0 else None);axs[0].scatter(np.full(len(vals),k+shift),vals,s=12,c='black',zorder=3)
        for shift,vals,label,color in [(-.17,d.native_cosine,'Change B0','#3b78a1'),(.17,gate[gate.source==src].C3_MAG_gradient_cosine,'Change gate','#ce8848')]:
            axs[1].scatter(np.full(len(vals),k+shift),vals,label=label if k==0 else None,c=color,s=25)
    for ax in axs:ax.set_xticks([0,1],['Electricity','ETTm1']);ax.legend(fontsize=8)
    axs[0].set_ylabel('Component L2 norm');axs[0].set_title('Initial full-epoch gradient decomposition')
    axs[1].set_ylabel('Gradient cosine');axs[1].set_ylim(0,1.05);axs[1].set_title('Fixed initialization, paired contrasts')
    savefig(fig,'initial_gradient')
    fig,axs=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for src,color in zip(sources,['#3b78a1','#ce8848']):
        for arm,style in zip(arms,['-','--']):
            q=tr[(tr.source==src)&(tr.arm==arm)].groupby('step').mean(numeric_only=True)
            axs[0].plot(q.index,q.down_relative_drift,style,c=color,label=f'{src}: {arm}')
            axs[1].plot(q.index,100*q.residual_relative_norm,style,c=color)
        q=pairs[pairs.source==src].groupby('step').C3_MAG_gradient_cosine.agg(['mean','min','max'])
        axs[2].plot(q.index,q['mean'],c=color,label=src);axs[2].fill_between(q.index,q['min'],q['max'],color=color,alpha=.2)
    axs[0].set_ylabel('Relative Down-parameter drift');axs[0].legend(fontsize=7)
    axs[1].set_ylabel('Residual / patch-representation norm (%)')
    axs[2].set_ylabel('C3 / MAG gradient cosine');axs[2].legend(fontsize=8)
    for ax in axs:ax.set_xlabel('Stored training update');ax.set_xticks([0,512,1024])
    savefig(fig,'trajectory')
    fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for pi,panel in enumerate(panels):
        for ki,kind in enumerate(['standard','shape']):
            ax=axs[ki,pi];conditions=sorted(eff[(eff.panel==panel)&(eff.kind==kind)].condition.unique())
            for arm,offset,color in [('C3',-.15,'#3b78a1'),('MAG_ONLY',.15,'#ce8848')]:
                q=eff[(eff.panel==panel)&(eff.kind==kind)&(eff.arm==arm)].set_index('condition').loc[conditions]
                yy=np.arange(len(q))+offset;xx=q.penalty_pct.to_numpy();lo=100*q.bonferroni2_low/q.matched_nmae;hi=100*q.bonferroni2_high/q.matched_nmae
                ax.hlines(yy,lo,hi,color=color);ax.scatter(xx,yy,s=16,c=color,label=arm)
            ax.axvline(0,color='black',lw=.7);ax.set_yticks(np.arange(len(conditions)),conditions,fontsize=8);ax.invert_yaxis();ax.set_title(f'{panel} / {kind}',fontsize=10)
            ax.set_xlabel('Swap penalty (% of matched nMAE)');ax.legend(fontsize=7)
    savefig(fig,'B0_pairing')
    main=eff[(eff.kind=='standard')&eff.condition.isin(['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT'])].copy()
    main['interval_nmae']=main.apply(lambda r:f'[{r.bonferroni2_low:+.6f}, {r.bonferroni2_high:+.6f}]',axis=1)
    counts=[]
    for (panel,arm),q in eff.groupby(['panel','arm']):
        counts.append(dict(panel=panel,arm=arm,positive_interval=int((q.bonferroni2_low>0).sum()),negative_interval=int((q.bonferroni2_high<0).sum()),includes_zero=int(((q.bonferroni2_low<=0)&(q.bonferroni2_high>=0)).sum()),conditions=len(q)))
    count=pd.DataFrame(counts);count.to_csv(OUT/'PAIRING_DIRECTION_SUMMARY.csv',index=False)
    full=eff.copy();full['interval_nmae']=full.apply(lambda r:f'[{r.bonferroni2_low:+.6f}, {r.bonferroni2_high:+.6f}]',axis=1)
    (OUT/'ALL_CONDITIONS.md').write_text('# B0 교환 — 전체 조건\n\n양수는 교환 후 손해, 음수는 교환 후 개선이다. 조건별 C3/MAG 두 대조에만 Bonferroni를 적용했으며 전체 120개 대조에 대한 동시 유의성은 아니다. 개별 점추정과 구간을 보존한다.\n\n'+table(full,['panel','kind','condition','arm','matched_nmae','swapped_nmae','penalty_pct','interval_nmae'])+'\n')
    selected=eff[(eff.panel=='electricity_transfer')&(eff.condition=='SHIFT8')&(eff.kind=='standard')].set_index('arm')
    pairing_sentence='; '.join(f'{a} {selected.loc[a,"penalty_pct"]:+.4f}% (nMAE 차이 {selected.loc[a,"pairing_penalty"]:+.6f}, 구간 [{selected.loc[a,"bonferroni2_low"]:+.6f}, {selected.loc[a,"bonferroni2_high"]:+.6f}])' for a in arms)
    # Descriptive outcome, without a post-hoc selection/gate or automatic next experiment.
    npos=int((eff.bonferroni2_low>0).sum());nneg=int((eff.bonferroni2_high<0).sum());nzero=len(eff)-npos-nneg
    final=tr[tr.step==1024].groupby(['source','arm'])[['down_relative_drift','residual_relative_norm','tanh_saturation_fraction']].mean().reset_index()
    text=f'''# C3 내부 gradient와 B0 조합 진단 — 한국어 보고서

## 결론과 식별 범위

성능이 달라지는 내부 경로의 일부를 확인했다. 같은 입력 patch 표현에서도, B0에 따른 예측의 loss 신호와 downstream Jacobian, 어댑터 초기 특징이 결합하여 초기 gradient가 달라진다. Electricity의 공통 TRAIN epoch0에서는 출력 loss 신호 항의 norm이 Jacobian 항보다 평균 {ratio.loc['electricity','mean']:.2f}배 컸다. ETTm1에서는 {ratio.loc['ettm1','mean']:.2f}배로 비슷했다. 이는 초기 gradient 차이의 국소 진단이며 최종 성능 변화의 완전한 인과 설명이나 기여율은 아니다.

학습 후 B0 교환은 전력 전이 SHIFT8에서 {pairing_sentence}. C3와 MAG_ONLY 모두에서 학습한 B0와의 조합 의존성이 나타났다. B0는 추가 어댑터 학습 중 동결됐으므로 양쪽 가중치가 동시에 적응했다는 뜻은 아니다. 이 결과는 C3 지속성 규칙만의 설명보다 B0에 맞춰 학습된 보정 함수라는 공통 설명을 지지한다. 전체 조건에서 원래 짝이 항상 유리하다는 주장은 하지 않으며 부호가 다른 조건도 보존한다.

## 실행·보존·검증

신규 학습 0 fits, optimizer update 0회. 기존 32경로 × 5 checkpoint = 160개 공통 TRAIN probe, 초기 전체 TRAIN 2,048문맥, autograd 1,196회(한도1,280), 진단 forward812회, 새 E 예측96개·기존96개 재사용, 소규모 E 복원/동등성 forward288회를 수행했다. 최적화 객체를 만들거나 학습 설정을 바꾸지 않았다. 최종 Adam 상태32개도 파일에서 읽었다.

사전 봉인 commit `f0767d4`, 비교 기준 `c41c96e`. 기존 코드·입력·가중치 {len(read(OUT/'SEAL.json')['hashes'])}개 hash를 보존했다. 초기 chain rule140건, gradient 분해8건, probe gradient160건, pairing120건, 저장 예측192개 hash를 독립 검산했다. 기존 원점수 재현과 스칼라 nMAE/pinball 검산은 SCORE_VERIFICATION.json에 있다. 전체 E 예측 저장을 확인한 뒤 채점했다. 복원 및 residual-off B0 일치는 새 view마다 검사했다.

마지막 GPU 표본까지 경과 {resource['wall_seconds_to_last_GPU_sample']/60:.1f}분, GPU 최소 여유 {resource['GPU_min_free_mib']:,}MiB, 새 E 추론 peak allocated {resource['E_peak_allocated_mib']:.1f}MiB. 외부 학습 감지 {resource['external_training_samples']}건이며 사전 승인된 RustDesk만 예외였다. 계산 시간과 fit 수는 구분한다. 이 진단의 추가 학습 파라미터·updates는 0이다.

## 1. 같은 시작 예측이 다른 학습 방향으로 갈 수 있는 이유

추가 어댑터는 z=GELU(Dh+d), u=h+cap·g·tanh(Uz+b)이며 처음 U=b=0이다. 따라서 같은 B0에서 초기화와 gate가 달라도 시작 예측은 정확히 같다. 그러나 r=∂L/∂u에 대해 초기 ∇U=Σ(cap·g·r)zᵀ, ∇b=Σ(cap·g·r)이다. 초기 ∇D=∇d=0이어도 서로 다른 초기 z는 Up의 학습 방향에 들어간다. 140개 실제 autograd 검산의 최대 상대 L2={rules.relative_l2.max():.3e}, 최대 절대차={rules.max_abs.max():.3e}로 봉인된 허용오차 안이었다.

두 B0의 어댑터 이전 patch 표현 h는 각 source 1,024 TRAIN 문맥에서 bitwise 동일했다. 이 구조의 B0 LoRA는 그 뒤 attention q/v에 있으므로 h가 같아도 downstream 함수는 다르다. 모든 층의 표현이 같다는 뜻은 아니다. 초기 z의 두 초기값 간 linear CKA는 Electricity {geom.set_index('source').loc['electricity','initial_feature_linear_CKA']:.4f}, ETTm1 {geom.set_index('source').loc['ettm1','initial_feature_linear_CKA']:.4f}였다. CKA를 성능 원인 비율로 해석하지 않는다.

## 2. B0는 출력 loss 신호와 역전파 전달 경로를 함께 바꾼다

예측 P의 loss 미분 v=∂L/∂P와 downstream Jacobian J를 교차하여 가상 미분 Jᵀv를 계산했다. B0 저/고 수준의 조합 A=G(J₀,v₀), B=G(J₀,v₁), C=G(J₁,v₀), D=G(J₁,v₁)에서 J항=((C−A)+(D−B))/2, v항=((B−A)+(D−C))/2이며 D−A=J항+v항이다. 두 변경 순서의 평균 분해이며 모델 학습·타깃 교체가 아니다. 가상 cotangent가 반드시 어떤 일관된 새 loss에서 유래한다는 주장도 하지 않는다.

Electricity v항/J항 norm 비는 {ratio.loc['electricity','min']:.3f}–{ratio.loc['electricity','max']:.3f}, ETTm1은 {ratio.loc['ettm1','min']:.3f}–{ratio.loc['ettm1','max']:.3f}였다. 두 항이 일부 상쇄하므로 norm을 더해 백분율 기여도로 바꾸면 안 된다. 이번 normalized pinball loss의 신호는 예측 quantile과 정답의 상하관계·동률에 영향을 받는다. 두 B0의 v 원소 불일치 비율은 Electricity {100*signals.loc['electricity','v_disagreement_fraction']:.3f}%, ETTm1 {100*signals.loc['ettm1','v_disagreement_fraction']:.3f}%였다. 단순히 오차 크기가 더 커서 gradient가 커졌다는 설명과 구분한다.

{table(summary,['source','B0_gradient_cos_min','B0_gradient_cos_max','output_J_ratio_mean','gate_cos_min','gate_cos_max'])}

동일 초기값에서 B0 변경의 gradient cosine은 Electricity 약0.39–0.41, ETTm1 약0.73–0.74였다. C3/MAG gate 변경의 초기 cosine은 모든 경우0.990 이상이었다. 이는 특정 고정 대조의 방향 유사성이며 gate 효과가 0이라는 증명이나 모든 요인의 공정한 중요도 순위가 아니다. 초기 전체 epoch0(주로 amplitude4, SHIFT4)의 측정이고, 실제 첫 optimizer batch 및 이후 모든 업데이트와 구분한다.

![초기 gradient의 국소 분해와 방향](figures/initial_gradient.png)

점은 고정한 gate·초기값 조합이다. 별도의 seed 모집단 표본이나 신뢰구간이 아니다.

## 3. 배치 구성과 후속 학습 경로

이전 실험에서 'ORDER/순서' 요인은 배치 구성과 순서를 함께 변경했다. 실제 첫32개 표본 중 두 수준이 공유한 표본은 Electricity1개, ETTm1 2개였다. 따라서 동일 배치 묶음의 나열 순서만 바꾼 인과 실험으로 서술하지 않는다.

{table(summary,['source','first_batch_common','order_gradient_cos_min','order_gradient_cos_max','Adam_m_cos_min','Adam_m_cos_max'])}

저장된 최종 Adam 1차 moment도 달랐다. 그러나 weight 경로와 moment가 함께 변하므로 Adam moment만을 최종 차이의 독립 원인으로 확정할 수 없다. optimizer state 교환이나 새 학습은 하지 않았다.

아래는 같은128 TRAIN probe에 대한 source·gate별8경로 평균이다. REFERENCE/POINT/BURST/SHIFT32개씩, SHIFT4/8·길이24/48을 포함한다. 각 checkpoint에서 실제 학습에 사용한 그 순간의 gradient를 복원한 것은 아니다.

{table(final,['source','arm','down_relative_drift','residual_relative_norm','tanh_saturation_fraction'])}

Down 파라미터는 크게 이동했으므로 초기 random feature가 학습 내내 고정됐다는 설명은 성립하지 않는다. 검사한160개 checkpoint-probe에서 |tanh|>0.95 비율의 최대값은 {100*tr.tanh_saturation_fraction.max():.6f}%였다. 광범위한 tanh 포화가 지배적이라는 근거는 없지만 모든 입력·좌표의 포화를 배제한 것은 아니다. residual_relative_norm은 patch 표현에 대한 보정 norm 비율이며 예측 오차 감소율이 아니다.

![고정 TRAIN probe에서 본 학습 경로](figures/trajectory.png)

선은8경로 평균이다. 오른쪽 음영은8개 C3/MAG 쌍의 최소–최대 범위이며 신뢰구간이 아니다.

## 4. 고정 가중치의 B0 교환 — 최종 예측 확인

추가 어댑터를 학습한 B0와 실제 수신 B0를 2×2로 교차했다. 고정1,024 updates의 gate·초기값·배치 순서·어댑터는 유지했다. 같은 수신 B0에서 자체 어댑터와 다른 B0에서 온 어댑터의 오차를 비교하고 두 수신 B0를 균형 평균한다. 양의 pairing penalty는 교환 손해, 음수는 교환 개선이다. 균형 잡힌 2×2 대조는 B0 및 어댑터의 단순 주효과를 상쇄하고 학습 B0·수신 B0의 조합 상호작용을 측정한다. 그 상호작용을 특정 attention head나 초기 gradient 성분 하나에 귀속시키지는 않는다.

{table(main,['panel','condition','arm','matched_nmae','swapped_nmae','penalty_pct','interval_nmae'])}

구간은 고정 모델에 조건부인2,000회7 index-day paired bootstrap이며 패널·조건별 C3/MAG 두 대조에 Bonferroni를 적용했다. 120대조 전체의 동시 구간이 아니며 조건별 유의성 수를 독립 성공 횟수로 세지 않는다. 전체적으로 양수 구간 {npos}개, 음수 구간 {nneg}개,0 포함 {nzero}개였다. 전체20조건×3패널×2gate를 ALL_CONDITIONS.md·PAIRING_EFFECTS.csv에 보존했다. 표의 FAULT는 표준 POINT/BURST 조건 평균이다. 교환이 유리했던 예외는 원래 전력 패널의 PULSE8_D32와 전력 전이 패널의 PULSE8_D32·STEP6_D63·STEP12_D63이며 두 gate에서 모두 나타났다. 원래 전력의 STEP12_D63 두 gate는 구간에0을 포함했다. 따라서 조합 의존성을 모든 변화 형태에서 같은 방향의 이득으로 서술하지 않는다.

![모든 조건의 B0 교환 효과](figures/B0_pairing.png)

## 5. 논문에서 남길 주장과 남기지 않을 주장

확인된 것은 '초기 예측 동등성 아래에서도 학습의 미분 경로가 달라진다'는 이 구현의 국소 메커니즘과, 이전 B0×초기화/배치 구성 상호작용의 구체적 가능한 경로다. 기존 factorial에서 고정 update의 B0×초기값 항이 가장 컸다는 결과와 이번 초기 gradient 분해는 서로 다른 estimand다. v항 norm이 약3배라는 사실을 최종 성능 차이의75% 설명으로 변환하지 않는다. 초기 gradient → 전체 학습 trajectory → 최종 E 차이의 완전한 매개 분석, 특정 attention head의 최대 영향은 미실행·미식별이다.

현재의 합성 입력, Electricity/ETTm1, 재사용 개발 E, 두 B0·두 초기값·두 배치 수준에 조건부다. 전력 transfer는 같은 자료의 추가 계열이며 독립 source가 아니다. 새로운 독립 test, 실제 센서 사건 레이블, 범용 robust PEFT 우위, C3 지속성 규칙 고유의 일반적 우위, 방법 신규성을 이 결과로 선언하지 않는다. 기존 C3의 좁은 전력 이득과 ETT·FAULT 손해를 모두 보존한다.

## 재현과 자료 제공 범위

PROTOCOL.md/SEAL.json이 사전 계획과 고정 입력을 기록하고, CPU_CHECKS.json·INDEPENDENT_AUDIT.json·SCORE_VERIFICATION.json이 검산을 기록한다. RAW_SCORES.csv에는 모든 학습 B0×수신 B0×초기화×순서·gate 원점수가, ORIGIN_SCORES.csv.gz에는 원점 단위값이 있다. 첫 배치와 gradient 원벡터, 가중치·raw input·전체 예측은 로컬 ignored cache에 있고 hash manifest만 공개한다. GitHub만으로 raw numerical replay를 완결할 수 있다고 주장하지 않는다. 캐시가 있는 환경에서 `python -m experiments.c3_internal_mechanism_20260918.audit`와 `.report`는 CPU 검산·문서 생성만 한다. 완료 runner를 다시 실행하여 진단을 중복하지 않는다. 자동 후속 학습은 없다.
'''
    (OUT/'REPORT.md').write_text(text)
    claim=f'''# 논문 본문에 반영할 내부 기전 보충

## 삽입할 결과 문단

고정된 입력과 예측 초기 상태만으로 학습 경로의 동등성이 보장되지 않았다. 본 구현의 zero-output adapter에서는 초기 Down gradient가0이지만, 초기 hidden feature는 Up gradient에 직접 곱해진다. 두 B0의 adapter 이전 patch embedding이 동일함을 확인한 후, 출력 loss 신호와 downstream Jacobian의 교차 미분으로 초기 gradient 차이를 분해했다. Electricity의 공통 TRAIN epoch0에서 출력 신호 항 norm은 Jacobian 항보다 평균 {ratio.loc['electricity','mean']:.2f}배 컸고, ETTm1에서는 {ratio.loc['ettm1','mean']:.2f}배였다. 이 결과는 학습된 B0와 adapter 초기값의 상호작용에 대한 국소 설명을 제공하지만, 이후 optimizer 경로와 최종 예측 차이의 완전한 인과 매개 분석은 아니다.

고정1,024 update 어댑터의 B0 교환에서 전력 전이 SHIFT8의 균형 평균 penalty는 {pairing_sentence}. 두 gate 모두에서 B0와 학습된 보정 함수의 조합 의존성이 나타났으므로 이를 C3 지속성 규칙의 고유 효과로 돌리지 않는다. B0는 추가 학습 중 동결됐으며 동시 공동 학습을 뜻하지 않는다. 모든 reference·fault·shift·형태 대조를 함께 공개한다. pulse 및 일부 step 형태의 반대 방향도 남기고, 단일 조건의 pairing 부호를 학습 경로 전체의 인과 설명으로 사용하지 않는다.

## 원고의 용어 수정

- 기존 '학습 순서'는 '표본의 배치 구성 및 순서'로 정확히 쓴다. 첫 배치의 공통 표본은32개 중 Electricity1개, ETTm1 2개였다.
- '초기 특징이 결과를 결정한다'는 단정 대신 '초기 특징이 첫 gradient 경로에 영향을 주며, 이후 Down 가중치도 변화했다'고 쓴다.
- '약3배'는 두 gradient 성분의 norm 비다. 최종 오차 기여율·설명분산·원인 확률이 아니다.
- '지속성 규칙이 최대 요인' 또는 'B0 조합에 항상 특화됐다'는 주장은 전체 대조의 부호와 범위 없이 쓰지 않는다.
- 이전 고정-update factorial의 B0×INIT 최대 점추정과 현재 국소미분 결과는 구분한다. 독립 source·seed 모집단의 최대 요인은 아직 식별하지 않았다.

## 근거와 한계

추가 학습0회, 기존32경로의160 checkpoint probe, 1,196 autograd 호출, 신규96개·재사용96개 E view, 독립 검산 완료. 모든 결과는 재사용 개발자료의 사후 기전 분석이다. 합성 변화에 대한 제한된 실증 및 학습 조건 민감성의 증거로 사용하며, 알려진 chain rule을 새 PEFT 방법으로 주장하지 않는다. 기존 원고의 전력 개선·ETT 손해·단순 대안 대조와 함께 제시한다. 현 증거만으로 방법론 논문 PASS나 투고 준비 완료를 선언하지 않는다.
'''
    (OUT/'PAPER_CLAIM_UPDATE_KO.md').write_text(claim)
    decision=f'''# 최종 판단

실행 상태: COMPLETE_VERIFIED. 신규 fits0, optimizer updates0, autograd1,196,160 checkpoint probe, 새96/재사용96 E view를 완료했다. 요구 범위의 미실행 항목은 없다. 새 원인 분리 학습이나 특정 head 매개 분석은 이번 범위 밖이므로 실행하지 않았다.

판정: LOCAL_GRADIENT_MECHANISM_IDENTIFIED / FULL_PERFORMANCE_MEDIATION_NOT_IDENTIFIED.

1. 남길 근거: zero-output 초기화가 시작 예측을 같게 해도 초기 feature가 Up gradient를 바꾼다. B0는 adapter 전 patch h 대신 downstream loss 신호와 Jacobian을 바꾼다. Electricity 초기 공통 TRAIN에서 loss 신호 항이 더 컸고 ETTm1에서는 비슷했다.
2. 정확히 고칠 설명: ORDER는 배치 구성도 바꿨다. 초기 Down은 gradient0이나 학습 후 크게 이동했다. 최종 moment 차이는 관측했으나 moment만의 인과 효과는 분리하지 않았다. 광범위한 tanh 포화가 주요 원인이라는 근거는 없었다.
3. 최종 함수 대조: C3/MAG 공통의 B0 조합 의존성은 확인됐고 C3만의 지속성 규칙 설명과 구분한다. B0는 추가 학습 중 동결됐다.  전력 전이 SHIFT8의 교환 penalty는 {pairing_sentence}. 전체120조건별 대조에는 양수 구간{npos}개·음수 구간{nneg}개·0포함{nzero}개가 있다. 조건별 두 대조 보정이며120개 전체 동시 추론은 아니다. 원래 B0와의 pairing 이득은 모든 조건에 일반화하지 않는다.
4. 중단할 주장: 지속성 규칙이 일관되게 핵심이고 범용 PEFT를 능가한다는 주장, 초기 gradient 성분의 norm을 최종 오차 기여율로 읽는 주장, 실제 사건 해결·독립 test·새 방법 신규성 선언.
5. 논문에 남길 결과: 기존 전력의 좁은 양성 결과, 단순 대안의 경쟁력, 학습 요인 상호작용, 이번 국소 기전과 해석 한계를 함께 쓴다. 좋은 부분을 지우거나 새 방법 PASS로 올리지 않는다.

현재 구현과 전 결과를 보존한다. 새 후보·새 dataset·추가 학습·튜닝을 자동 시작하지 않는다. 후속 실행 후보를 새로 선정하지 않았다.
'''
    (OUT/'FINAL_DECISION.md').write_text(decision)
    save(OUT/'REPORT_VERIFICATION.json',dict(status='VERIFIED',source_files={str(p.relative_to(ROOT)):sha(p) for p in OUT.glob('*.csv')},pairing_rows=120,summary_rows=2,figure_count=3,descriptive_only=True,new_training=0))
    print('REPORT_COMPLETE',flush=True)

if __name__=='__main__':report()
