"""Descriptive reports only: no selection, training, or model changes."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/c3_temporal_mpl')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,ScalarFormatter
from .common import *
LABELS={'electricity':'Electricity (4)','electricity_transfer':'Electricity transfer (16)','ettm1':'ETTm1','neso_2025':'NESO 2025 H2',NEW:'NESO 2026 H1'}

def table(f,cols=None):
    if cols is not None:f=f[cols]
    lines=['| '+' | '.join(map(str,f.columns))+' |','| '+' | '.join(['---']*len(f.columns))+' |']
    for row in f.itertuples(index=False,name=None):lines.append('| '+' | '.join('—' if pd.isna(x) else f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)

def build():
    assert read(OUT/'SCORE_VERIFICATION.json')['status']=='VERIFIED';assert read(OUT/'DIAGNOSTIC_VERIFICATION.json')['status']=='VERIFIED';check_seal()
    e=pd.read_csv(OUT/'EFFECTS.csv');r=pd.read_csv(OUT/'RAW_SCORES.csv');s=pd.read_csv(OUT/'SEED_EFFECTS.csv');c=pd.read_csv(OUT/'CONDITIONAL_EFFECTS.csv');cost=read(OUT/'COST.json');audit=pd.read_csv(OUT/'ORIGIN_AUDIT.csv').iloc[0]
    primary=e[(e.kind=='standard')&(e.condition=='SHIFT8')&(e.new=='C3')&e.baseline.isin(['C0','C2','M_RECENCY'])];trade=e[(e.kind=='standard')&e.condition.isin(['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT'])&(e.new=='C3')&e.baseline.isin(['C0','C2','M_RECENCY','F0'])]
    maincols=['condition','baseline','new_nmae','baseline_nmae','gain_pct','ci_low_pct','ci_high_pct'];condcols=['panel','stratum','contexts','fraction','gain_pct','ci_low_pct','ci_high_pct','gate','weights'];cs=c[(c.kind=='standard')&(c.condition=='SHIFT8')]
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def finish(name):
        plt.tight_layout()
        for extn in ['png','pdf','svg']:plt.savefig(figdir/(name+'.'+extn),dpi=180,bbox_inches='tight')
        p=figdir/(name+'.svg');p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');plt.close()
    conditions=['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT'];fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,base in zip(axes,['C0','C2','M_RECENCY']):
        f=trade[trade.baseline==base].set_index('condition').loc[conditions]
        for i,(_,z) in enumerate(f.iterrows()):
            ax.hlines(i,z.ci_low_pct,z.ci_high_pct,color='black');ax.scatter(z.gain_pct,i,color='tab:blue');ss=s[(s.kind=='standard')&(s.condition==conditions[i])&(s.new=='C3')&(s.baseline==base)];ax.scatter(ss.gain_pct,np.full(3,i+.12),marker='x',s=18,color='tab:orange')
        ax.axvline(0,color='grey',lw=.7);ax.set_yticks(range(5),conditions);ax.invert_yaxis();ax.set_title('C3 vs '+base);ax.set_xlabel('nMAE reduction (%)')
    fig.suptitle('Frozen-model temporal replication: NESO 2026 H1 / no new training');finish('F10_temporal_effects')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));a=cs[cs.stratum=='DIFFERENT'].set_index('panel').reindex(DIAG_PANELS)
    for i,p in enumerate(DIAG_PANELS):
        z=a.loc[p]
        if not z.contexts:
            for ax in axes:ax.text(0,i,'empty: identical masks',fontsize=8)
            continue
        axes[0].hlines(i,z.ci_low_pct,z.ci_high_pct,color='black');axes[0].scatter(z.gain_pct,i,color='tab:blue')
        for col,dy,color in [('gate',-.12,'tab:orange'),('weights',.12,'tab:green')]:axes[1].barh(i+dy,z[col],height=.22,color=color,label=col if i==0 else None);axes[1].hlines(i+dy,z[col+'_low'],z[col+'_high'],color='black',lw=.8)
    for ax in axes:ax.axvline(0,color='grey',lw=.7);ax.set_yticks(range(5),[LABELS[p] for p in DIAG_PANELS]);ax.set_ylim(4.45,-.5)
    axes[0].set_title('C3 / RECENCY when masks differ');axes[0].set_xlabel('nMAE reduction (%)');axes[1].set_title('Fixed learned-function decomposition');axes[1].set_xlabel('absolute nMAE benefit');axes[1].legend(loc='upper left');axes[1].xaxis.set_major_locator(MaxNLocator(4));fmt=ScalarFormatter(useMathText=True);fmt.set_powerlimits((-3,-3));axes[1].xaxis.set_major_formatter(fmt);fig.suptitle('SHIFT8 input-only strata: explanatory diagnosis, not a new selection criterion');finish('F11_conditional_mechanism')
    fig,axes=plt.subplots(1,2,figsize=(12,5));prev=pd.read_csv(ext.OUT/'EFFECTS.csv');names=SHAPES+['PAIRED_SHIFT8_D32']
    for ax,base in zip(axes,['C2','M_RECENCY']):
        old=prev[(prev.panel=='neso_2025')&(prev.kind=='shape')&(prev.new=='C3')&(prev.baseline==base)&(prev.ci_type=='time')].set_index('condition');new=e[(e.kind=='shape')&(e.new=='C3')&(e.baseline==base)].set_index('condition')
        for j,(f,color,label,dy) in enumerate([(old,'tab:grey','2025 H2',-.12),(new,'tab:blue','2026 H1',.12)]):
            f=f.loc[names];ax.scatter(f.gain_pct,np.arange(9)+dy,color=color,label=label);ax.hlines(np.arange(9)+dy,f.ci_low_pct,f.ci_high_pct,color=color,lw=.8)
        ax.axvline(0,color='grey',lw=.7);ax.set_yticks(range(9),names);ax.invert_yaxis();ax.set_title('C3 vs '+base);ax.set_xlabel('nMAE reduction (%)');ax.legend()
    fig.suptitle('All registered shapes retained; no shape promoted to the primary endpoint');finish('F12_temporal_shapes')
    pseed=s[(s.kind=='standard')&(s.condition=='SHIFT8')&(s.new=='C3')&s.baseline.isin(['C0','C2','M_RECENCY'])]
    newmask=cs[cs.panel==NEW];neg=trade[(trade.condition=='REFERENCE')&(trade.baseline=='F0')].iloc[0];gain=primary.set_index('baseline').gain_pct
    text=f'''# C3 약점 보완: 입력 조건 분해와 고정 모델의 새 기간 검증

## 실행과 범위

기준 d213b15, 결과 전 규약 커밋 0863f50. 새 학습 **0 fits / 0 optimizer updates**. NESO 2026년 상반기 **60/60 예측 views(56 GPU, 4 CPU)**, 3개 기존 Electricity seed와 frozen F0 및 단순 점예측을 평가했다. 모두 저장·hash 고정 후 정답을 채점했다. 주 비교나 LR/checkpoint/보정/입력식을 결과에 맞춰 변경하지 않았다. 기존 4개 panel과 새 기간의 C3/RECENCY 설명 비교는 저장 예측을 재사용했다. 이전 결과를 지우거나 수정하지 않았다.

추론 벽시계 {cost['wall_seconds']/60:.2f}분, peak allocated {cost['peak_allocated_bytes']/2**20:.1f} MiB. 준비/CPU 분석/문서 시간과 다르다. GPU 안전 감시는 RustDesk만 예외로 허용했다. 실제 전력 소모/탄소 배출을 측정한 수치는 아니다.

## 무엇이 아직 약했으며 이번에 무엇을 보완했나

1. **새 규칙의 추가 가치**: 단순 RECENCY보다 전체 이득이 작고 종종 mask 자체가 동일했다. 관측 입력만으로 SAME/DIFFERENT를 나눈 후 두 draw·세 seed를 유지해 고정 가중치×추론 gate의 효과를 분해했다. 집단 비중을 곱한 nMAE 차의 합이 전체 차와 일치함을 확인했다. 이 분해는 학습 gradient의 인과 식별이 아니다.
2. **반복 사용한 개발 평가**: ETT/Electricity와 NESO2025를 또 평가해 새 test라고 부르는 대신, 이전 프로젝트 기록에서 평가 흔적이 없던 NESO2026 H1을 사전 고정했다. 모델 선택은 기존 source V에서 끝난 상태다. 같은 국가 집계계열의 새 시기이므로 새로운 독립 source라고 부르지 않는다. 과거 성과를 보고 NESO를 골랐다는 연구 선택 이력과 2025H2/2026H1 계절 차이는 남는다.
3. **원자료/오류 조건의 손해**: 새 기간에서도 REFERENCE, 개별 6 fault 및 그 평균, SHIFT4/8/POINT를 모두 남겼다. 확인된 손해를 소거하는 튜닝은 하지 않았다.
4. **신규성·정식 선행·실제 사건**: 새로운 정식 선행 재현이나 실제 오류 label 검증을 수행한 것은 아니다. 이번 분석으로 그 미완료 범위가 해결됐다고 주장하지 않는다.

## 자료 및 정보 권한

[NESO 공식 자료](https://www.neso.energy/data-portal/historic-demand-data)의 ND를 DST settlement-period 검사 후 UTC hourly mean으로 집계했다. 미래64시간이 UTC 2026-01-01~2026-07-01 안에 있는 **128 distinct days / {audit['week_blocks']} week blocks**, 24 phase 각각 5~6회, 기간 10구간당12~13일이다. target-overlap 최대 {audit['max_overlap']}회이며 bootstrap은 7일 block으로 묶었다. 입력512시간의 일부는 2025년 말의 관측된 과거일 수 있다. sigma는 기존 2025H1 값과 byte-level 숫자 동일이다. 예측 입력에는 state 이름, 원본 clean x, fault mask, true delta, 미래 y가 들어가지 않는다. 생성기의 합성 offset은 채점 때만 정답에 적용한다.

기존 tracked text 4540개에서 NESO2026 사용 흔적 없음. 이는 프로젝트 기록의 부재이지 전세계/미기록 노출 부재 증명이 아니다. 실제 사용한 foundation weight의 2024 공개 byte 동일성은 앞선 봉인 감사를 유지했다. 제공자는 21일 지연 및 사후 수정과 자료 품질 문제를 공지한다. 실제 시점별 vintage도, 실제 regime-change/센서오류 정답 자료도 아니다. ND 한 계열을 128 독립 데이터셋처럼 세지 않는다.

## 새 기간 주 비교: SHIFT8

양수는 C3 nMAE 감소. 95% CI는 고정된 세 seed에 조건부인 paired 7일 block bootstrap 2000회다. 세 주 비교의 다중성 보정 구간도 원표에 있다. CI가 0을 포함하지 않는 것과 실용적으로 큰 효과 또는 논문 성공은 서로 다르다.

{table(primary,maincols+['bonferroni3_low_pct','bonferroni3_high_pct'])}

세 seed별 주 비교:

{table(pseed,['baseline','seed','new_nmae','baseline_nmae','gain_pct'])}

## 전체 주요 상태의 절충

{table(trade,maincols)}

C3/F0 REFERENCE의 상대 이득은 {neg['gain_pct']:+.4f}%. 원자료에 대한 손익은 synthetic shift 이득과 별도로 판단해야 한다. F0/PERSISTENCE/SEASONAL은 seed0 한 경로씩이며 세 seed처럼 복제하지 않았다. 모든 군의 seed별 nMAE/MAE/nRMSE/pinball/crossing은 RAW_SCORES.csv, individual fault와9개 shape도 EFFECTS.csv에 그대로 있다.

## 관측 mask가 같은 경우와 다른 경우

{table(cs,condcols)}

NESO2025와 새 NESO2026 H1의 SHIFT8 모두256개 입력에서 C3/RECENCY mask가 완전히 같아 DIFFERENT 집단이 0개였다. 따라서 새 기간 SHIFT8 역시 추론 규칙의 우위를 식별하는 조건은 아니며, 2026년의 작은 차이도 가중치 항에서 나왔다. 빈 집단은 0점 효과라고 채우지 않았다. 전력16계열 DIFFERENT에서 전체 상대 이득은 약0.0854%지만 gate 절대 기여는 음수였으며 학습된 가중치 항이 이를 상쇄했다. 같은 mask인 경우도 학습된 가중치가 달라 최종 예측은 다를 수 있다. ‘같은 mask이므로 gate가 학습 중 전혀 필요 없다’ 또는 ‘가중치 항이 있으므로 학습 gate의 인과 효과가 증명됐다’는 해석은 모두 하지 않는다.

A=C3weights/C3gate, B=C3weights/RECgate, C=RECweights/C3gate, D=RECweights/RECgate. total=D−A, gate=((B−A)+(D−C))/2, weights=((C−A)+(D−B))/2. 동일 mask에서는 같은 가중치의 교환 예측 오차가 일치함을 검사했다. 이 input-only 진단은 이전 자료를 보고 정한 설명 분석이며 유리한 하위집단만 새 핵심 실험으로 삼지 않는다. 원점별 합/빈도와 seed별 결과도 공개했다.

## 검산·실행 한계

- CPU 손계산 factorial, 동일/상이 mask 예제, nMAE/pinball scalar, 가중 집계, block bootstrap 검사 통과.
- 모든 새 모델의 기존 checkpoint forward, 전 가중치 동결/hash 보존 및 같은 계산 조건 복원 검사 통과. 예측60개 hash와 전체 저장 완료 시각 후 채점을 확인했다.
- 모든 조건·draw에서 첫/마지막 원점/채널 scalar 검산, 집단 가중 재구성, 원점수에서 모든 효과 재계산을 수행했다. 자세한 숫자는 VERIFICATION.json 참조.
- ETTm2의 RECENCY 가중치는 기존에 없어 조건부 진단을 수행하지 않았으며 이를 위해 재학습하지 않았다.
- 새 fits/후속 후보/추가 seed/LR/threshold는 실행하지 않았다. 공식 선행과의 동일 정보·예산 직접 재현, 실제 사건 label, 다른 독립 원천, train-gate의 완전한 인과 분리, 더 많은 독립 학습 반복은 미완료다.
- 소스·규약·원점수·표·검산은 GitHub로 공개하지만 원자료/weights/대용량 predictions는 로컬 ignored cache다. GitHub만으로 수치 재생이 완결된다고 주장하지 않는다.

## 그림

- [새 기간 효과 및 seed](figures/F10_temporal_effects.png)
- [상이한 mask의 조건부 분해](figures/F11_conditional_mechanism.png)
- [모든 변화 형태의 시간적 비교](figures/F12_temporal_shapes.png)

실행 완료와 논문 성공은 별도다. 새 기간 C3/B0={gain['C0']:+.4f}%, C3/C2={gain['C2']:+.4f}%, C3/RECENCY={gain['M_RECENCY']:+.4f}%를 모두 함께 해석한다. 최종 주장 범위는 FINAL_DECISION.md에 남긴다.
'''
    (OUT/'REPORT.md').write_text(text)
    decision=f'''# 최종 판단: C3 약점 보완

보완 실행은 완료했다. 새 학습0회, 고정 모델의 새 기간 예측60개 및 기존 저장 예측의 입력 조건 분해다. 논문 PASS 또는 범용 PEFT 우위 선언은 하지 않는다.

NESO2026 H1 SHIFT8 C3/B0 {gain['C0']:+.6f}%, C3/C2 {gain['C2']:+.6f}%, C3/RECENCY {gain['M_RECENCY']:+.6f}%. 모든 seed와 손해 조건을 REPORT/원표에 남겼다. 주요 세 대조의 다중성 보정 구간까지 확인해야 하며, 좋은 평균만으로 지속성 규칙 자체의 필요성을 단정하지 않는다.

새 기간 C3/C2 이득은 세 seed 모두 양수였지만, C3/RECENCY는 두 seed 양수·한 seed 음수이고 주요 구간이0을 포함했다. NESO2026 SHIFT8에서도256개 입력의 mask가 모두 동일하여 추론 gate의 고유 기여를 이 조건으로 입증할 수 없었다. 이 발견을 별도 한계로 명시한다.

**남길 구현과 근거:** 고정 B0 위 추가 adapter와 관측 기반 gate, matched-size C2 및 MEAN/ROTATE/RECENCY 대조, 동일 학습량의 부모 비교, 이번 봉인 시간 전이와 모든 raw scores. 좁은 효과가 관찰된 것은 보존한다.

**중단할 주장:** ‘모든 입력/데이터에서 우수’, ‘실제 센서 오류를 구분’, ‘현재 mask의 시간적 위치가 모든 이득의 원인’, ‘간단한 RECENCY를 충분히 크게 일관되게 능가’, ‘공식 선행보다 우수’. 기존 전력16계열 DIFFERENT 집단에서도 inference gate 기여가 음수라는 반례를 숨기지 않는다.

현재 가능한 논문의 중심은 **잘 적응된 예측기 위 추가 PEFT의 조건부 이득과 원자료 보존의 절충, 그리고 관측 gate 설명의 식별 한계**다. 구체적 구현을 제안하고 대조하는 PEFT 연구로 기술할 수 있으나 새로운 규칙의 필요성과 정식 선행 대비 독창성이 충분하다는 결론은 아직 아니다. 채택 가능성은 투고처·전체 원고·선행 심사에 달려 있다.

후속 투자 후보: 자동 선정/학습하지 않음. 새 구조를 추가하거나 RECENCY에 맞춰 규칙을 튜닝하지 않고 이번 범위에서 종료한다. 정식 동일 권한 선행 비교와 실제 사건 자료 부족은 논문의 제한 및 미완료 비교로 남긴다.
'''
    (OUT/'FINAL_DECISION.md').write_text(decision)
    save(OUT/'FIGURE_MANIFEST.json',dict(figures=[p.name for p in figdir.glob('*.png')],formats=['png','pdf','svg'],tables=['RAW_SCORES.csv','EFFECTS.csv','SEED_EFFECTS.csv','CONDITIONAL_EFFECTS.csv','CONDITIONAL_SEEDS.csv'],new_training=0))
    print('REPORT AND 3 FIGURES WRITTEN',flush=True)
if __name__=='__main__':build()
