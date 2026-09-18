"""Descriptive tables and figures; never select a method from diagnostic E."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/c3_magnitude_diagnostic_plots')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
from experiments.c3_weakness_controls_20260918.report import table

def report():
    assert read(OUT/'INDEPENDENT_AUDIT.json')['status']=='VERIFIED'
    f=pd.read_csv(OUT/'DECOMPOSITION.csv');allf=f[(f.family=='ALL')&(f.status=='SCORED')];main=allf[(allf.kind=='standard')&(allf.condition=='SHIFT8')];raw=pd.read_csv(OUT/'RAW_SCORES.csv');seeds=pd.read_csv(OUT/'SEED_DECOMPOSITION.csv');gs=pd.read_csv(OUT/'GATE_SUMMARY.csv');ch=pd.read_csv(OUT/'CHANNEL_CONTRIBUTIONS.csv');sel=pd.read_csv(OUT/'CHECKPOINT_CONTEXT.csv');ft=OUT/'figures';ft.mkdir(exist_ok=True)
    names={'electricity':'Electricity (4)','electricity_transfer':'Electricity transfer (16)','ettm1':'ETTm1'}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def finish(name):
        plt.tight_layout()
        for extn in ['png','pdf','svg']:plt.savefig(ft/(name+'.'+extn),dpi=180,bbox_inches='tight')
        p=ft/(name+'.svg');p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n');plt.close()
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,panel in zip(axes,PANELS):
        r=main[main.panel==panel].iloc[0]
        for j,c in enumerate(['total','gate','weights']):
            ax.hlines(j,100*r[c+'_low']/r.A,100*r[c+'_high']/r.A,color='black');ax.scatter(100*r[c]/r.A,j)
        ax.axvline(0,color='grey',lw=.7);ax.set_yticks(range(3),['total','gate switch','weights + selection']);ax.invert_yaxis();ax.set_title(names[panel]);ax.set_xlabel('Error benefit / original C3 error (%)')
    fig.suptitle('SHIFT8 frozen-function decomposition: positive favors MAG; descriptive 95% CI');finish('01_gate_weights_decomposition')
    fig,axes=plt.subplots(1,3,figsize=(12,4));states=['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT']
    for ax,panel in zip(axes,PANELS):
        s=allf[(allf.panel==panel)&(allf.kind=='standard')].set_index('condition').loc[states]
        for col,label in [('total','total'),('gate','gate switch'),('weights','weights + selection')]:ax.plot(100*s[col]/s.A,range(5),'o-',label=label)
        ax.axvline(0,color='grey',lw=.7);ax.set_yticks(range(5),states);ax.invert_yaxis();ax.set_title(names[panel]);ax.set_xlabel('Benefit / original C3 error (%)');ax.legend(fontsize=7)
    fig.suptitle('All main states retained; components sum to total');finish('02_state_tradeoffs')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,panel in zip(axes,PANELS):
        s=ch[(ch.panel==panel)&(ch.kind=='standard')&(ch.condition=='SHIFT8')];ax.bar(range(len(s)),s.total*1e3);ax.axhline(0,color='grey',lw=.7);ax.set_xticks(range(len(s)),s.channel.astype(str),rotation=60);ax.set_title(names[panel]);ax.set_ylabel('C3 error - MAG error (nMAE x 1000)');ax.set_xlabel('All registered series IDs')
    fig.suptitle('SHIFT8 per-series contributions, no unfavorable series excluded');finish('03_channel_effects')
    fig,axes=plt.subplots(1,3,figsize=(13,6))
    for ax,panel in zip(axes,PANELS):
        s=allf[(allf.panel==panel)&(allf.kind=='shape')];vals=np.stack([100*s[c]/s.A for c in ['total','gate','weights']],-1);lim=max(1,np.abs(vals).max());ax.imshow(vals,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(3),['total','gate','weights'],rotation=30);ax.set_yticks(range(len(s)),s.condition);ax.set_title(names[panel])
        for i in range(len(s)):
            for j in range(3):ax.text(j,i,f'{vals[i,j]:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(vals[i,j])>.6*lim else 'black')
    fig.suptitle('All registered shapes: positive error benefit favors MAG; each panel own color scale');finish('04_shapes_decomposition')
    fields=['panel','A','B','C','D','MAG_gain_pct','total','gate','weights','interaction','gate_low','gate_high','weights_low','weights_high']
    st=seeds[(seeds.family=='ALL')&(seeds.kind=='standard')&(seeds.condition=='SHIFT8')]
    safety=allf[(allf.kind=='standard')&allf.condition.isin(states)]
    first=gs[(gs.kind=='standard')&gs.condition.isin(['REFERENCE','SHIFT4','SHIFT8','SHIFT_POINT'])]
    text=f'''# C3와 MAG_ONLY의 차이는 어디서 생기는가

이번 작업은 **새 학습 0회·optimizer update 0회**, 기존 예측36개 재사용과 고정 가중치 교차 예측36개로 수행한 사후 진단이다. 기존 C3/MAG_ONLY의 가중치·checkpoint·입력·원점·조건을 바꾸지 않았다. 재사용 개발 E이므로 새로운 독립 시험이 아니다.

[실제 결과에 따른 해석](INTERPRETATION_KO.md)을 먼저 참고한다. 교차 규칙은 원인 분해용 함수 개입이며 새로운 배포 후보나 재튜닝 결과가 아니다.

## 무엇을 따로 바꿨나

| 기호 | 학습된 가중치 | 평가 가중 규칙 | 실행 |
|---|---|---|---|
| A | C3 | C3 | 기존 예측 재사용 |
| B | C3 | MAG_ONLY | 새 교차 예측 |
| C | MAG_ONLY | C3 | 새 교차 예측 |
| D | MAG_ONLY | MAG_ONLY | 기존 예측 재사용 |

동일 입력에서 규칙만 교체한 차이와 가중치만 교체한 차이를 모두 측정했다. total=A−D, gate=0.5[(A−B)+(C−D)], weights=0.5[(A−C)+(B−D)]이며 total=gate+weights다. interaction=A−B−C+D도 남긴다. **양수는 MAG 방향의 오차 감소**다. weights에는 학습 경로와 V checkpoint 선택의 차이가 함께 들어 있으므로 학습 gate만의 인과효과라고 부르지 않는다.

## 코드와 전수 입력 검사

C3의 p는 엄밀한 연속 길이 자체가 아니라 최근8개 관측의 같은 부호 extreme 비율이다. I=1(|d|>3), p≤I이므로 C3 gate는 MAG gate 이상이다. 같은 어댑터 출력에 곱하는 계수 관점에서 C3는 MAG보다 보정을 덜 억제한다. 이것이 반드시 최종 예측 오차 증가를 뜻하지는 않는다.

관측 입력 **89,088개**에서 두 gate의 차이는 같은 부호 extreme run의 첫7개 관측 안에만 존재했다. 길이8 이상으로 이어진 부분에서는 p=1이라 차이가 없다. 원래 관측에도 여러 extreme run이 있으므로 이를 합성 사건의 진짜 시작 위치를 알아냈다고 해석하지 않는다. gate 차이의 위치는 전체 입력에서 확인했으나, 최근128/32 구간의 오차 기여를 따로 제거하는 추가 개입은 하지 않았다.

{table(first,['panel','condition','contexts','gate_equal_fraction','mean_gate_gap','mean_longest_run','recent128_mass_fraction','recent32_mass_fraction'])}

## SHIFT8의 2×2 분해

{table(main,fields)}

단위는 nMAE다. 7일 날짜 block bootstrap 2,000회로 고정 세 seed 평균에 조건부인 사후95% 구간을 계산했다. 이 구간은 다중 비교나 이전 연구 선택을 보정하지 않는다. 전체E128 날짜 블록을 사용했고 같은 날짜의 채널·draw를 독립 날짜로 세지 않았다.

## seed 및 checkpoint 선택

{table(st,['panel','seed','A','B','C','D','total','gate','weights','interaction'])}

{table(sel,['source','arm','seed','lr','step','objective'])}

최종 오차만으로 checkpoint를 다시 고르지 않았다. step0 선택은 학습 미실행이 아니며, 해당 가중치의 추가 어댑터가 0이므로 gate 교체가 효과 없을 수 있다.

## 원자료·오류·다른 변화

{table(safety,['panel','condition','MAG_gain_pct','total','gate','weights','interaction'])}

개별 오류6종과 shape9종을 포함한 RAW_SCORES.csv에는 A/B/C/D의 모든 seed별 nMAE·원단위MAE·normalized MSE·pinball·crossing을 남겼다. 모든 조건의 분해·구간은 DECOMPOSITION.csv, 원점별 오차는 ORIGIN_ERRORS.csv.gz에 있다. 서로 다른 상태의 점수를 합쳐 우승자를 만들지 않았다.

## 관측 특성과 계열의 영향

gate 동일/상이, robust 크기≤3·3–6·6–12·>12, 같은 부호 extreme 최대연속0·1–7·8–31·≥32, 마지막extreme 없음·오래된구간·최근128로 층화했다. 조건과 입력값만으로 정한 완전 분할이며 비어 있는 그룹도 EMPTY로 기록한다. 이 변수들은 서로 연관되고 source·조건의 영향도 섞이므로 독립적 인과 중요도로 순위화하지 않는다.

각 층의 표본 수와 가중 기여가 전체 결과를 정확히 복원하는지 검사했다. CHANNEL_CONTRIBUTIONS.csv에 등록된 모든 계열과 하나씩 제외했을 때의 전체 효과를 기록했다. 이는 민감도 진단이며 계열 제외나 데이터 변경의 허가가 아니다.

## 검산과 한계

GPU 진단 실행 시간은 {read(OUT/"diagnostic_wall.json")["seconds"]/60:.2f}분(시작 안전 대기 포함), 최소 GPU 여유는 {read(OUT/"INDEPENDENT_AUDIT.json")["minimum_gpu_free_mib"]}MiB였다. 승인되지 않은 외부 compute는0건이며 새 optimizer update는0회다. CPU 집계·문서 시간은 이 GPU 진단 시간과 별도다.

[VERIFICATION.json](VERIFICATION.json)과 [INDEPENDENT_AUDIT.json](INDEPENDENT_AUDIT.json)에 실제 모델의 기존 대각 예측 재현, off=B0, 동결 가중치와 buffer 보존, 복원, 예측 hash, 같은 gate/같은 가중치의 오차 일치, scalar metric 검사, 분해·층화·채널 재집계를 남겼다. CPU 특성 추출에서 처음 발생한 연산 경로 오류와 수정은 [IMPLEMENTATION_CORRECTION.md](IMPLEMENTATION_CORRECTION.md)에 보존했다. 모델이나 성능 기준을 바꾸지 않았다.

과거 완료 파일은 불변이며, 새로운 학습이나 성능 개선을 이 진단의 성과로 주장하지 않는다. 고정 함수에서 gate 교체의 효과는 측정했지만, 학습 내내 gate가 미친 영향과 checkpoint 선택 효과의 완전 분리는 추가 통제가 필요하다. 새 학습·threshold·window·rank·자료는 자동 추가하지 않았다.

[그림1: 규칙과 가중치](figures/01_gate_weights_decomposition.png) · [그림2: 상태별 절충](figures/02_state_tradeoffs.png) · [그림3: 계열별 차이](figures/03_channel_effects.png) · [그림4: 모든 형태](figures/04_shapes_decomposition.png)
'''
    (OUT/'REPORT.md').write_text(text);print('REPORT_READY',flush=True)
if __name__=='__main__':report()
