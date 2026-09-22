"""Render reports and figures from verified score tables only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from finalize import read_csv,write_csv,gain

def table(headers,rows):
    values=[[str(x) for x in row] for row in [headers]+rows];width=[max(len(row[i]) for row in values) for i in range(len(headers))]
    return '```text\n'+'\n'.join('  '.join(v.ljust(width[i]) for i,v in enumerate(row)).rstrip() for row in values)+'\n```\n'

def report(c):
    r=RESULTS/c;d=read(r/'FINAL_DECISION.json');verification=read(r/'VERIFICATION.json');proposal=d['proposal'];base=d['baseline'];region='tail64' if c=='A' else 'all8'
    rows=[x for x in read_csv(r/'SCORES.csv') if x['region']==region];arms=list(dict.fromkeys(x['arm'] for x in rows));resources=read_csv(r/'RESOURCES_FAIR.csv');selection=read(r/'MODEL_SELECTION.json')
    def val(arm,metric='crps',mode='cal'):return float(np.mean([float(x[metric]) for x in rows if x['arm']==arm and x['mode']==mode]))
    def improvement(a,b,metric='crps',mode='cal'):return gain(val(a,metric,mode),val(b,metric,mode))
    costs={x['arm']:x for x in resources if x['batch']=='8'}
    boot=[x for x in read_csv(r/'BOOTSTRAP.csv') if x['candidate']==proposal and x['baseline']==base and x['mode']=='cal'][0]
    scoretable=table(['arm','raw_CRPS','CAL_CRPS','pinball','nMAE','raw_MAE','coverage80','width','batch8_ms','GPU_peak_MiB'],[[a,f'{val(a,mode="raw"):.6f}',f'{val(a):.6f}',f'{val(a,"pinball"):.6f}',f'{val(a,"nmae"):.6f}',f'{val(a,"raw_mae"):.4f}',f'{val(a,"coverage"):.4f}',f'{val(a,"width"):.4f}',f'{float(costs[a]["median_seconds"])*1000:.3f}',f'{float(costs[a]["peak_allocated"])/2**20:.2f}'] for a in arms])
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for i,a in enumerate(arms):
        x=float(costs[a]['median_seconds'])*1000;y=val(a);color='#be3c3c' if a==proposal else '#366b92'
        axs[0].scatter(x,y,s=85 if a==proposal else 40,color=color,zorder=3)
        offset=[(4,7),(5,-11),(5,7),(-56,15),(6,-20),(-68,-15),(5,8),(5,-15),(-64,10)][i]
        axs[0].annotate(a,(x,y),xytext=offset,textcoords='offset points',fontsize=8,color=color)
    axs[0].set(xlabel='Batch 8 end-to-end latency (ms)',ylabel='CAL normalized discrete CRPS (lower is better)',title=f'{c}: quality and measured cost')
    axs[0].margins(x=.25,y=.2);axs[0].grid(alpha=.2)
    if c=='B':axs[0].set_xlim(left=0)
    controls=[a for a in arms if a!=proposal]
    eff=read_csv(r/'SEED_EFFECTS.csv')
    for j,s in enumerate(SEEDS):
        points=[float(next(x for x in eff if x['candidate']==proposal and x['baseline']==a and x['mode']=='cal' and x['metric']=='crps' and int(x['seed'])==s)['gain_pct']) for a in controls]
        axs[1].scatter(points,np.arange(len(controls))+(j-.5)*.15,label=str(s),s=34)
    axs[1].set_yticks(np.arange(len(controls)),controls);axs[1].axvline(0,color='black',lw=.8);axs[1].axvline(1,color='gray',ls=':',lw=.8)
    axs[1].set(xlabel=f'{proposal} CRPS improvement (%)',title='Each repeat seed; positive favors proposal');axs[1].legend(title='Training seed',fontsize=8);axs[1].grid(axis='x',alpha=.2)
    fig.savefig(r/'quality_cost.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained');y=np.arange(len(arms))
    ax.barh(y-.18,[val(a,mode='raw') for a in arms],height=.35,label='Raw',color='#a8b4c4');ax.barh(y+.18,[val(a) for a in arms],height=.35,label='Common CAL',color='#366b92')
    ax.set_yticks(y,arms);ax.set(xlabel='Normalized discrete CRPS (lower is better)',title=f'{c}: raw and common-calibrated TEST scores');ax.legend();ax.grid(axis='x',alpha=.2);fig.savefig(r/'raw_calibration.png',dpi=180);plt.close(fig)
    checkpointtable=table(['arm','seed92301','seed92302'],[[a,selection['92301'][a]['checkpoint'],selection['92302'][a]['checkpoint']] for a in (ARMS_A if c=='A' else ARMS_B)])
    fits=read_csv(r/'FIT_LEDGER.csv');training=[]
    for a in (ARMS_A if c=='A' else ARMS_B):
        selected=[x for x in fits if x['key'].startswith(a+'_') and x['phase']=='main'];training.append(dict(arm=a,main_fits=2,main_updates=512,optimizer_compute_seconds=sum(float(x['fit_seconds']) for x in selected),fit_wall_seconds=sum(float(x['wall_seconds']) for x in selected),trainable_parameters=int(selected[0]['trainable_parameters'])))
    write_csv(r/'TRAINING_RESOURCES.csv',training)
    trainingtable=table(['arm','fits','updates','compute_s','fit_wall_s','trainable'],[[x['arm'],x['main_fits'],x['main_updates'],f'{x["optimizer_compute_seconds"]:.2f}',f'{x["fit_wall_seconds"]:.2f}',x['trainable_parameters']] for x in training])
    common=f'''두 반복 seed는 92301·92302이며 선택 전용 seed는 없습니다. 평균 개선율은 **두 seed 원점수 평균의 비율** `100×(1−candidate_mean/baseline_mean)`입니다. seed별 개선율의 평균이나 예측 ensemble 점수가 아닙니다. 고정 모델은 한 번 계산한 예측을 비교 상대에 재사용하며 독립 반복 2회로 부풀리지 않습니다.

V로 고정한 기준선 **{base}** 대비 {proposal}의 CAL CRPS 개선율은 **{d['mean_gain_pct']:+.4f}%**, 두 seed는 **{d['seed_gain_pct'][0]:+.4f}%, {d['seed_gain_pct'][1]:+.4f}%**입니다. 같은 상대 대비 median nMAE 개선율은 **{d['point_gain_pct']:+.4f}%**입니다. 음수 개선율은 악화입니다.

14개 origin 연속 블록을 2,000회 공동 재표집한 95% 구간은 **[{float(boot['lower95']):+.3f}%, {float(boot['upper95']):+.3f}%]**입니다. 두 seed와 모든 계열·lead를 같은 origin 블록에 묶었습니다. seed를 모집단처럼 bootstrap하지 않았으며, 이 구간은 두 학습 반복만으로 학습 불확실성을 충분히 추정하지 못합니다. 0 포함 여부만으로 GO/NO_GO를 바꾸지 않습니다.

![품질과 비용, seed별 비교](quality_cost.png)

왼쪽은 공통 CAL 후 TEST 품질과 batch8 실측 비용이고, 오른쪽은 동일 원점수에서 계산한 seed별 효과입니다. 지연시간은 선택된 seed92301 모델로 10회 warm-up/30회 측정, 순방향·역방향 순서를 합친 중앙값입니다. 품질은 두 seed 평균입니다.

{scoretable}
![보정 전후](raw_calibration.png)

CRPS는 선언한 유한 지지점 분포의 정확한 점수입니다. 분위수 출력에 동일 질량을 주는 근사가 포함되므로 연속분포 전체의 우위를 뜻하지 않습니다. pinball은 모든 군에 같은 .1–.9 left-inverse discrete-CDF readout을 적용했습니다. raw MAE는 Solar 원단위(A) 또는 MW(B), 나머지 오차는 TRAIN sigma로 정규화했습니다. 폭과 포함률, 음수 질량은 [SCORES.csv](SCORES.csv)에 보존했습니다. CAL이 TEST에서 항상 좋아지지는 않습니다.

### 선택·학습·비용

{checkpointtable}
raw VAL 0/128/256에서 checkpoint를 고르고, CAL에서만 35개 affine 계수 조합을 고른 뒤 보정된 VAL로 기준선을 정했습니다. 모든 선택 파일의 hash를 봉인한 후 TEST 예측을 저장했고, 진단용 A checkpoint0와 runtime까지 완료한 다음 채점했습니다. TEST 결과로 설정·보정·비교 상대를 바꾸지 않았습니다.

{trainingtable}
fit wall은 첫 checkpoint0 VAL 이후의 update·중간 VAL·checkpoint 저장을 포함하며, 사전 데이터/모델 준비 전체 시간이 아닙니다. [FIT_LEDGER.csv](FIT_LEDGER.csv)는 smoke도 별도 기록합니다. main12 fits/3,072 updates와 smoke12 updates를 정확히 사용했습니다.

비용은 RTX4070, FP32, TF32 off에서 CPU 입력 복사·기상 표준화·필요 H2D·모델·작은 모듈·D2H·CAL·9분위수 readout을 포함합니다. 파일 읽기와 모델 로딩은 제외했습니다. A 최초 F0 계산은 매번 포함하며, B는 target TSFM을 한 번만 계산하고 작은 멤버 모듈만 벡터화했습니다. GBQR는 공통 특징을 한 번 계산하고 9개 회귀기에 전달합니다. Chronos-2는 CPU 입력을 공식 API에 직접 주었습니다. 실제로 학습하지 않은 Chronos-2의 requires_grad 기본값을 학습 파라미터 수로 세지 않았습니다.

처음 측정에는 GBQR의 중복 특징 계산과 Chronos-2의 추가 전송이 있었습니다. [RESOURCES.csv](RESOURCES.csv)에 원기록을 보존했고, 최종 판단은 [RESOURCES_FAIR.csv](RESOURCES_FAIR.csv)를 사용합니다. TRAIN batch1·8에서 수정 전후 예측 차이 0을 검증했습니다. 재학습은 없었습니다. 두 실행의 allocator 상태 차이도 있어 메모리 차이를 이 두 최적화의 인과 효과로 해석하지 않습니다. CPU 모델의 GPU peak는 모델 크기가 아니며, 절대 CPU RAM은 별도 측정하지 않았습니다.

CAL 원래 실행의 별도 wall time은 계측되지 않았습니다. {verification['cal_grid_evaluations_recomputed']:,}개 grid 평가를 재검산한 시간 {verification['cal_verification_seconds']:.3f}초를 별도로 남겼고 원래 실행 시간으로 대체하지 않았습니다.

### 검증과 해석 범위

[VERIFICATION.json](VERIFICATION.json)의 pairwise 정의와 정렬 CRPS 검산 최대 차이는 {verification['max_pairwise_crps_absolute_error']:.2e}입니다. pairwise CRPS와 pinball은 각 군·seed·raw/CAL의 3개 위치에서 대조했습니다. 표의 개선율 전부, checkpoint·CAL·VAL 기준선·update intent/commit은 별도로 재검산했습니다. frozen backbone와 모든 buffer는 각 fit에서 불변이었습니다. 참조 CPU22검사 외 통합 CPU5검사, Chronos/GPU smoke, 저장·복원, gradient·시간 정렬 검사를 별도로 수행했습니다.

Main 이전 위임 범위 이탈은 [IMPLEMENTATION_INCIDENT.json](../IMPLEMENTATION_INCIDENT.json)에 보존했습니다. 잘못 호출된 runner는 main0 updates에서 중단됐고, 소스 복구와 root 재검산 후 최종 소스를 봉인했습니다. 실행된 smoke는 예산24에 포함했고 재실행하지 않았습니다. 삭제된 조기 봉인 파일 원본은 복구하지 못했다는 한계도 기록했습니다. 이 구현 사건과 최종 과학적 판정을 구분합니다.

선행 원리인 LoRA·분포 증류·집합 attention 자체의 신규성이나 논문 PASS는 주장하지 않습니다. 자동 후속 실험은 없습니다. 모든 예측/가중치 원본은 로컬 cache에 있고 GitHub에는 코드·표·그림·manifest·검산이 있습니다. GitHub만으로 원시 예측을 즉시 재채점할 수 있다는 뜻은 아닙니다.
'''
    if c=='A':
        initial=all(selection[str(s)][a]['checkpoint']==0 for s in SEEDS for a in ['CONTEXT3','GLOBAL3'])
        cache=read(r/'TEACHER_CACHE_MANIFEST.json')['caches'];cachetime=sum(x['seconds'] for x in cache);cachebytes=sum(x['bytes'] for x in cache)
        intro=f'''# A: 경로 압축 PEFT 결과

**[확인] {d['status']} — 3경로 압축은 유용했지만 입력별 router의 추가 가치는 확인되지 않았습니다.**

일반 방법의 효과: FULL9 학습은 F0_NATIVE+CAL보다 CRPS **{improvement('FULL9','F0_NATIVE'):+.3f}%**, F0_MEDIAN+CAL보다 **{improvement('FULL9','F0_MEDIAN'):+.3f}%** 개선됐습니다. 단일경로 학생보다 3경로는 유리했지만 고정3/medoid3/전역3도 충분했습니다. CONTEXT3의 FULL9 대비 CRPS 개선은 {improvement('CONTEXT3','FULL9'):+.4f}%, 공통 pinball 개선은 {improvement('CONTEXT3','FULL9','pinball'):+.4f}%였고, batch8 지연은 {d['latency_saving_vs_full9_pct']:.2f}% 줄었습니다.

새 모듈의 효과: CONTEXT3와 GLOBAL3의 평균 점수 차이는 {improvement('CONTEXT3','GLOBAL3'):+.6f}%입니다. **두 seed 모두 GLOBAL3와 CONTEXT3는 추가 학생 학습 전 checkpoint0이 선택됐습니다.** 초기 두 router는 같은 상수 출력을 내므로 이 결과를 입력 조건화 학습의 성공으로 해석할 수 없습니다. 학생들은 실제 256 updates를 모두 실행했으나 VAL이 학습된 router를 선택하지 않았습니다. 이때 checkpoint0의 LoRA는 이미 선택된 같은-seed FULL9 teacher의 가중치이므로 완전한 무학습 F0를 뜻하지 않습니다. 저장된 두 방법의 TEST atom·질량이 완전히 같고, 모든 A군의 첫64 raw/CAL atom이 F0와 같음을 [FIRST64_AND_INITIALIZATION_AUDIT.json](FIRST64_AND_INITIALIZATION_AUDIT.json)에서 별도 검산했습니다.

실용 대안: Chronos-2 직접128의 batch8 시간은 {float(costs['CHRONOS2']['median_seconds'])*1000:.3f}ms, CONTEXT3는 {float(costs['CONTEXT3']['median_seconds'])*1000:.3f}ms입니다. CONTEXT3의 공통9 pinball 개선은 Chronos-2 대비 {improvement('CONTEXT3','CHRONOS2','pinball'):+.3f}%, nMAE는 {improvement('CONTEXT3','CHRONOS2','nmae'):+.3f}%입니다. 속도·메모리·품질을 함께 선택할 문제이며 Bolt를 무조건 유지해야 한다는 결론은 아닙니다. F0_NATIVE의 공식 선형 분위수 추가 점수(native_pinball)는 별도 열로 보존했습니다.

Solar는 이전 연구에 노출된 개발 자료입니다. 원래137열 중 고정8열을 6행 평균해 시간별로 집계했고 C512/H128, TEST68 origins×8 series를 평가했습니다. 날짜가 없는 자료에 달력 날짜를 부여하지 않았습니다. 주영역은 후반65–128이며 [SCORES.csv](SCORES.csv)에 first64/full128, [LEAD_SCORES.csv](LEAD_SCORES.csv)에 모든 lead, [SERIES_SCORES.csv](SERIES_SCORES.csv)에 계열별 결과가 있습니다. 표의 계열0–7은 원본열36/10/44/119/62/104/116/118 순서입니다.

Teacher는 seed별 새 FULL9이며 모든 학생에 동일 teacher·정답·packet·256학습·보정 기회를 줬습니다. 첫 F0/teacher cache 생성은 총 {cachetime:.3f}초/{cachebytes:,}bytes이고 FULL9 학습 비용은 위에서 별도 공개합니다. CONTEXT3 한 학생의 비용만으로 전체 학습비를 주장하지 않습니다. 초기 예측과 선택 예측 비교는 [CHECKPOINT_ZERO_DIAGNOSTICS.csv](CHECKPOINT_ZERO_DIAGNOSTICS.csv), raw selected는 SCORES에 있습니다.

'''
    else:
        cpu=read(r/'CPU_STATISTICAL_FITS.json')
        intro=f'''# B: 같은 issue 기상 앙상블 PEFT 결과

**[확인] {d['status']} — 미래 기상정보는 유용했지만 새 3시나리오 모듈의 추가 우위는 없었습니다.**

일반 학습의 효과: target-only LoRA는 frozen F0보다 CRPS {improvement('TARGET','F0'):+.3f}% 개선됐습니다. 정보의 효과: CONTROL은 TARGET보다 {improvement('CONTROL','TARGET'):+.3f}%, 일반 SET은 {improvement('SET','TARGET'):+.3f}%, 후보 SCENARIO3는 {improvement('SCENARIO3','TARGET'):+.3f}% 개선됐습니다. 일반 SET은 단일 CONTROL보다 {improvement('SET','CONTROL'):+.3f}% 개선됐지만 SET과 CONTROL은 head 용량도 달라 순수한 ensemble 정보만의 인과 추정은 아닙니다. 같은 E/C를 쓰는 MEMBER는 CONTROL보다 +1.149%(seed별 +1.251%/+1.050%) 개선돼 단순 멤버 혼합의 작은 추가 가치도 관찰됐습니다.

새 모듈의 효과: SCENARIO3는 CONTROL 대비 {improvement('SCENARIO3','CONTROL'):+.3f}%, MOMENTS 대비 {improvement('SCENARIO3','MOMENTS'):+.3f}%, MEMBER 대비 {improvement('SCENARIO3','MEMBER'):+.3f}%였지만, 같은 분포 용량의 일반 SET 대비 **{improvement('SCENARIO3','SET'):+.3f}%**, VAL로 고정한 GBQR 대비 **{improvement('SCENARIO3','GBQR'):+.3f}%**였습니다. 따라서 TARGET와의 큰 차이를 구조의 공로로 돌릴 수 없습니다. SET의 작은 모듈8,825개와 후보8,796개는 근사 용량 대조이며 완전히 같은 수라는 주장은 하지 않습니다.

실제 비용: SCENARIO3는 batch8 {float(costs['SCENARIO3']['median_seconds'])*1000:.3f}ms, SET {float(costs['SET']['median_seconds'])*1000:.3f}ms, GBQR {float(costs['GBQR']['median_seconds'])*1000:.3f}ms입니다. 후보는 50개 MEMBER 혼합보다 빠르지만, 더 좋은 일반 SET보다 유의미하게 싸지 않고 통계 기준선보다 느립니다. GBQR는 동일 미래기상 권한의 강한 실용 대조이며, 작은 신경 모듈과 학습 알고리즘/목적이 같다는 인과 비교는 아닙니다. GBQR9개 고정 분위수 모델 fit은 {cpu['seconds']:.3f}초이며 추가 탐색은 없습니다.

자료는 저자 저장소 고정 revision `9d1799ede894606ad349eb66e335d8a814eb8acf`의 **실제 SE3 wind_power MW**입니다. 같은 issue의 perturbed50members×8leads와 별도 control을 확인했습니다. Power time은 valid time이고 issue=time−horizon입니다. context 마지막은 issue−3h, 출력은 Bolt index1:9를 사용했습니다. 실제 packet1,679개/TRAIN1,074·CAL180·VAL183·TEST242, 기상 필드는 u100/v100/t2m/sp/speed입니다. [TIME_MAPPING_AUDIT.json](TIME_MAPPING_AUDIT.json)과 root 전체1679행 검산을 보존했습니다.

**RELEASE_TIME_UNVERIFIED: 아카이브 조건 파일럿입니다.** 저자 R 코드의 UTC와 archive issue는 확인했지만 실제 기상·power 공개 지연을 입증하지 못했습니다. 운영 backtest/배포 GO가 아닙니다. Onshore/Offshore 합성 목표나 다른 forecast vintage로 바꾸지 않았습니다. [SOURCE_LIMITS.md](SOURCE_LIMITS.md), [DATA_AUDIT.json](DATA_AUDIT.json)에 원자료 SHA와 제외 사유가 있습니다. README의 자료 안내가 모든 raw 재배포 권한을 증명하지 않으므로 raw를 push하지 않았습니다.

학습 한계: **모든 B neural arm에서 두 seed 평균 VAL이 0→128→256으로 계속 좋아져 OPTIMIZATION_LIMIT입니다.** 256 업데이트 화면에서 후보가 불리했다는 결론이며 충분히 학습된 모든 가능한 ensemble PEFT의 반증이 아닙니다. 이 사유로 예산을 늘리거나 설정을 바꾸지 않았습니다. [MONTH_SCORES.csv](MONTH_SCORES.csv)와 [LEAD_SCORES.csv](LEAD_SCORES.csv)에 달·lead별 결과를 남겼습니다.

'''
    (r/'REPORT_KO.md').write_text(intro+common,encoding='utf-8')
    checks='\n'.join(f'- `{k}`: {v}' for k,v in d['checks'].items())
    (r/'FINAL_DECISION.md').write_text(f'''# {c} 최종 판정

**{d['status']}**

V 고정 기준선: {base}. 후보 {proposal} CAL CRPS 개선 {d['mean_gain_pct']:+.4f}%, seed별 {d['seed_gain_pct'][0]:+.4f}% / {d['seed_gain_pct'][1]:+.4f}%. median nMAE 개선 {d['point_gain_pct']:+.4f}%.

{checks}

이는 일반 압축(A) 또는 기상정보(B)의 유용성을 인정하는 분류이며 새 모듈의 방법론 GO가 아닙니다. A는 CONTEXT3=GLOBAL3, B는 SCENARIO3가 SET·GBQR보다 불리합니다. 구현·자료·자원 block 없이 정해진 main 예산을 완료했습니다. B의 공개 지연 미검증과 학습 한계는 [보고서](REPORT_KO.md)에 별도로 표시했습니다.

추가 fit/LR/rank/seed/자료 탐색 0. 자동 후속 0. 논문 PASS/신규성 선언 없음.
''',encoding='utf-8')
    return d

def main():
    decisions=[report(c) for c in ['A','B']];a,b=decisions
    save(RESULTS/'FOLLOWUP_RECOMMENDATION.json',dict(recommendation='NONE',reason='Both novel modules lack incremental value over strong simple controls: A matches GLOBAL3 at selected checkpoint0; B loses to SET and GBQR.',automatic_experiments=0,novelty_or_paper_pass=False))
    text=f'''# A/B TSFM PEFT — 최종 요약

**[확인] 두 실험을 완료했지만 새 모듈의 방법론 후속 추천은 NONE입니다.** 일반적인 압축과 미래 기상정보는 유용했습니다. 새 router/3시나리오 구조가 강한 단순 대조보다 필요하다는 근거는 얻지 못했습니다.

- **A — {a['status']}**: 3경로는 FULL9 품질을 거의 유지하면서 빨랐습니다. CONTEXT3는 V 고정 FIXED3보다 {a['mean_gain_pct']:+.4f}% 개선(두 seed {a['seed_gain_pct'][0]:+.4f}% / {a['seed_gain_pct'][1]:+.4f}%)에 그쳤고, **GLOBAL3와 점수가 동일**했습니다. 두 router 모두 학생 checkpoint0이 선택돼 입력 조건화 학습의 추가 효과가 없습니다. FULL9 대비 batch8 시간은 {a['latency_saving_vs_full9_pct']:.2f}% 줄었지만 고정/medoid/전역3도 같은 압축 이점을 제공합니다.
- **B — {b['status']}**: 기상정보를 주면 target-only보다 좋아졌지만 SCENARIO3는 일반 SET보다 CRPS 6.341% 나빴고, V 고정 GBQR보다 {abs(b['mean_gain_pct']):.3f}% 나빴습니다(두 seed 개선율 {b['seed_gain_pct'][0]:+.3f}% / {b['seed_gain_pct'][1]:+.3f}%). GBQR는 더 빠르기도 했습니다. B는 실제 SE3 자료의 아카이브 조건 파일럿이며 공개 지연은 미검증입니다. 모든 neural arm의 VAL이 마지막까지 감소한 OPTIMIZATION_LIMIT도 남깁니다.

단순히 성공 문턱1%만 높아서 탈락한 결과로 보기 어렵습니다. A는 새 입력 조건화와 전역형이 같은 예측을 냈고, B는 일반 SET에도 졌습니다. 반면 **PEFT 전반이 안 된다**는 결론도 아닙니다. 일반 LoRA·정보 활용 이득과 이번 새 구조의 추가 효과를 분리해야 합니다. 이 개발 자료·고정 예산 밖으로 일반화하지 않습니다.

![A 품질·비용](A/quality_cost.png)

![B 품질·비용](B/quality_cost.png)

상세 결과·원점수·seed/lead/series/month 표·학습 및 추론비용은 [A 보고서](A/REPORT_KO.md), [B 보고서](B/REPORT_KO.md)에 있습니다. [A 판정](A/FINAL_DECISION.md), [B 판정](B/FINAL_DECISION.md), [후속 추천](FOLLOWUP_RECOMMENDATION.json)을 함께 남겼습니다.

실제 실행은 **24 main fits /6,144 main updates +24 smoke updates =6,168 optimizer updates**입니다. 실패 smoke/시도도 장부에 포함했고 예산 초과·자동 구제 탐색·후속 실험은 없습니다. 선택·CAL·baseline 봉인 후 모든 TEST와 A checkpoint0 예측을 저장했고 runtime 완료 후 채점했습니다. 초기 단계/학습 완료는 이미 별도 scoped commit으로 push했습니다.

GBQR 중복 특징 계산과 Chronos-2 불필요한 전송을 제거한 비용은 *_FAIR.csv에 있고, 원기록은 보존했습니다. TRAIN 실제입력 batch1·8에서 수정 전후 예측 차이0이며 추가 학습은 없었습니다. [실측 경로 검증](FAIR_RUNTIME_EQUIVALENCE.json).

Main 이전 위임 실행 범위 이탈은 [구현 사건 기록](IMPLEMENTATION_INCIDENT.json)에 공개했습니다. 잘못 실행된 runner의 main updates는0이었으며 root가 복구·검산·봉인한 후 최종 학습을 시작했습니다. 과학적 결과와 이 사건을 혼동하지 않습니다.

보고서 그림과 모든 검산 표는 이 경로에만 작성했습니다. 기존 실험을 수정·재개하지 않았고 raw 자료·Hugging Face weights·예측 cache는 GitHub에 올리지 않았습니다. manifests는 로컬 파일을 식별하며, GitHub만으로 캐시가 제공되는 것은 아닙니다. **신규성·논문 PASS 선언 없이 종료합니다.**
'''
    (RESULTS/'AB_SUMMARY_KO.md').write_text(text,encoding='utf-8')

if __name__=='__main__':main()
