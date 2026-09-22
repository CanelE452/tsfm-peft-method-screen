import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table(df):return '\n```text\n'+df.to_string(index=False,float_format=lambda x:f'{x:.6f}')+'\n```\n'
def main():
    assert read(RESULTS/'VERIFICATION.json')['status']=='PASS'
    summaries=[]
    for track in ['A','B']:
        out=RESULTS/track;decision=read(out/'DECISION.json');selection=read(out/'MODEL_SELECTION.json');bs=read(out/'BASELINE_SELECTION.json')
        raw=pd.read_csv(out/'RAW_SCORES.csv');scores=pd.read_csv(out/'SEED_SCORES.csv');effects=pd.read_csv(out/'EFFECTS.csv');se=pd.read_csv(out/'SEED_EFFECTS.csv');cost=pd.read_csv(out/'AMORTIZATION_COSTS.csv');resources=pd.read_csv(out/'RESOURCES.csv');budgets=pd.read_csv(out/'ADAPTATION_BUDGET_CURVES.csv');prop=track+'_TIME'
        def e(a,b):return effects[(effects.candidate==a)&(effects.control==b)].iloc[0]
        mean=scores.groupby('arm',sort=False).mean(numeric_only=True).drop(columns='seed');baseline=decision['baseline'];primary=e(prop,baseline);direct=e(prop,track+'_SET');feedback=e(prop,track+('_NOERROR' if track=='A' else '_FULLGEN'));general=e('STATIC','G0')
        seeds=se[(se.candidate==prop)&(se.control==baseline)].improvement_pct.to_list();special=se[(se.candidate==prop)&(se.control==track+'_SET')].improvement_pct.to_list()
        figs=out/'figures';figs.mkdir(exist_ok=True)
        fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
        arms=list(mean.index)
        for i,a in enumerate(arms):
            color='#c54338' if a==prop else '#41698c';vals=scores[scores.arm==a].pinball.to_numpy();axes[0].scatter(vals,[i-.1,i+.1],color=color,s=25);axes[0].scatter(vals.mean(),i,color=color,marker='D',s=45)
        axes[0].set(yticks=range(len(arms)),yticklabels=arms,xlabel='Past-context scaled twice-pinball (lower is better)',title='Two seeds and mean')
        pe=effects[effects.candidate==prop]
        for i,r in enumerate(pe.itertuples()):axes[1].plot([r.ci_low,r.ci_high],[i,i],color='#c54338',linewidth=2);axes[1].scatter(r.improvement_pct,i,color='#c54338')
        axes[1].axvline(0,color='black',linewidth=.8);axes[1].set(yticks=range(len(pe)),yticklabels=pe.control.tolist(),xlabel='TIME improvement (%)',title='Paired bootstrap 95% intervals')
        for a in axes:a.grid(axis='x',alpha=.2)
        fig.suptitle(f'{track}: independently selected, lawful feedback comparisons');fig.savefig(figs/'quality_and_effects.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
        if track=='A':
            for arm in ['A_LOCAL','A_COEFF']:
                b=budgets[(budgets.role=='TEST')&(budgets.arm==arm)];avg=b.groupby('updates_per_episode').pinball.mean();axes[0].plot(avg.index,avg.values,'-o',label=arm)
            axes[0].axhline(mean.loc[prop,'pinball'],color='#c54338',label='A_TIME');axes[0].set(xlabel='Per-episode optimizer updates',ylabel='TEST pinball',title='Fixed adaptation budgets; DEV selects primary k')
        else:
            for arm in [prop,'B_SET','B_FULLGEN','B_MASKED','B_WAIT','STATIC']:
                b=raw[raw.arm==arm];byissue=b.groupby('issue').pinball.mean();axes[0].plot(range(len(byissue)),byissue.values,label=arm,alpha=.8)
            axes[0].set(xlabel='Issue tick (8 h spacing)',ylabel='TEST pinball',title='Issued forecasts; no retrospective replacement')
        axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
        b=pd.read_csv(out/'LEAD_SCORES.csv')
        for arm in [prop,track+'_SET',track+('_NOERROR' if track=='A' else '_FULLGEN'),baseline]:
            if arm in [l.get_label() for l in axes[1].lines]:continue
            v=b[b.arm==arm].groupby('lead').pinball.mean();axes[1].plot(v.index,v.values,label=arm)
        axes[1].set(xlabel='Forecast lead (hours)',ylabel='TEST pinball',title='All 64 leads retained');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
        fig.savefig(figs/'budget_or_stream_and_leads.png',dpi=170);plt.close(fig)
        first=pd.DataFrame([dict(candidate=prop,strong_baseline=baseline,direct_control=track+'_SET',raw_score=primary.candidate_score,baseline_score=primary.control_score,effect_pct=primary.improvement_pct,seed1_pct=seeds[0],seed2_pct=seeds[1],prep_est_seconds=cost[cost.arm==prop].standalone_prep_estimate_seconds.mean(),field_cold_seconds=cost[cost.arm==prop].field_cold_seconds.mean(),decision=decision['label'],novelty='UNVERIFIED')])
        parts=[f'''# {track}_{'AMORTIZED' if track=='A' else 'PARTIAL'}: residual feedback PEFT

[확인] **{decision['label']}**. 아래 숫자는 이번 Electricity 개발 screen의 결과이며 논문 PASS나 신규성 확정이 아니다.
''',table(first),f'''
## 결과가 답하는 질문

- 일반 donor META 적응: STATIC은 G0 대비 **{general.improvement_pct:+.4f}%**. 둘 다 같은 신규 G0에서 출발했고 STATIC과 생성기는 동일 donor query와 정답·512updates를 받았다. 이 비교는 추가 META 적응의 효과이며, pretrained F0 대비 G0 학습의 절대 효과를 별도로 측정한 것은 아니다.
- 피드백의 추가 효과: {prop}은 {feedback.control} 대비 **{feedback.improvement_pct:+.4f}%**. {'명시적 residual 특징을 제거한 TIME 대조이며, raw context에 남은 관측정보까지 지운 비교는 아니다.' if track=='A' else '완전히 실현된 task만 residual로 주는 생성기와 비교한다. 실제 age/maturity metadata 및 동일 최신 context는 유지했다.'}
- 시간순 구조의 추가 효과: {prop}은 SET 대비 **{direct.improvement_pct:+.4f}%**, 두 seed는 {special[0]:+.4f}% / {special[1]:+.4f}%다. SET도 같은 예측·오차·age/maturity를 받는 거의 동일 파라미터 수 대조다.
- 강한 합법 대조: DEV에서 고정한 **{baseline}** 대비 **{primary.improvement_pct:+.4f}%**, 두 seed {seeds[0]:+.4f}% / {seeds[1]:+.4f}%다. TEST를 보고 baseline을 바꾸지 않았다. paired 95% 구간은 [{primary.ci_low:+.4f}%, {primary.ci_high:+.4f}%]다.

![전체 비교와 효과](figures/quality_and_effects.png)

원점수는 native .1–.9 quantile의 mean twice-pinball을 현재256시간 context의 표준편차 floor로 나눈 값이다. 계열별 평균 후 equal-series 평균이며 exact CRPS가 아니다. 이전 TRAIN-sigma 정규화 실험과 점수 크기를 직접 비교하지 않는다. 개선율은 두 seed 원점수 평균의 비율이다.
''',table(mean.reset_index()),'''
## 선택·정보 시계·평가 범위

선택은 DEV8고객에서 수행했다. TEST8고객은 generator/bank offline 학습에 사용하지 않았다. 원본26304시간×321고객의 첫30% 품질만 보고 SHA 순서로 DONOR32/DEV8/TEST8을 고정했다. 원본 행 index를 시간으로 사용하며 날짜·weekday를 만들지 않았다. BASE0–30%, META30–60%, DEV60–80%, TEST80–100%다. 과거 실험·사전학습 비노출을 보증하는 독립 확증 자료는 아니다.

현재 t에서 관측 가능한 정답은 index<t뿐이다. 현재 query는[t,t+64)이고, input은[t−256,t)다. G0 reference는 고정 진단 예측기이며 후보 자신의 최근 오차로 바뀌지 않는다. 각 logical support issue의 과거 input hash와 forecast hash를 기록했다. Source와 선택 manifest를 봉인하고 A/B의 TEST 예측을 모두 저장한 뒤 채점했다.
''']
        if track=='A':parts.append('''A는 고객당12개의 사전고정 episode에서 support4개가 모두 실현된 과거 오차를 읽는다. 매 episode bank/c/optimizer를 초기화하고 이전 episode의 적응 상태를 넘기지 않는다. LOCAL/COEFF의0/1/4/8 curve는 한 trajectory의 prefix이며 DEV에서 고른k만 주 비교다. AFFINE도 support만으로 gain/bias를 고른다. LONG512는 같은 전체정보 범위를 긴 문맥으로 읽는 대조다.
'''+table(pd.DataFrame([dict(arm=a,k=k) for a,k in bs['kselected'].items()])))
        else:parts.append('''[확인] 이번 B에서는 STATIC과 세 생성기 모두 두 seed에서 checkpoint0이 DEV 최선이었다. 따라서 주 비교의 세 생성기는 c=1로 G0와 동일한 예측을 냈다. 이것은 학습 실행이나 gradient 연결 실패가 아니라, 정해진 학습이 검증 성능을 개선하지 못해 선택되지 않은 결과다. 학습된512 모델의 TEST 결과로 주장을 바꾸지 않았다. 이 결과로 부분 피드백이 모든 조건에서 무용하다고 결론내리지 않는다. 배포 시 선택된 영출력 생성기를 제거해 G0로 단순화할 수 있으므로, 아래 B 생성기 실행비용은 불필요한 호출을 포함한 실제 비교 경로의 비용이지 최적화된 G0 배포비용은 아니다.

B는48ticks×8시간 간격의 제한된16일 stream이다. support issue=t−64,t−32,t−16,t−8이고 visible counts는64/32/16/8이다. 생성기 weights는 동결하고 c만 재생성한다. WAIT/MASKED/COEFF/COSA는 계열별 독립 optimizer를 유지하며 매tick2updates한다. 미공개 NaN은 먼저 선택에서 제외한 뒤 loss/residual을 계산한다. 과거에 발행한 예측을 덮어쓰거나 소급해 재채점하지 않았다. TEST 전체를 한 번도 열지 않았다는 주장은 하지 않는다. 당시 아직 알 수 없는 future를 학습·조건 생성에 사용하지 않는 as-of 접근이 검증 대상이다.

기존 Maturity-PEFT는 미공개 예측 보존 규제였다. 이번 B는 donor의 이후 실제 query loss로 수정 생성기를 학습하며 해당 규제의 재실행이 아니다. 부분 정답 문제를 최초로 다뤘다는 주장도 하지 않는다. COSA_CELL은 공식 linear core를 quantile9채널·이번 clock/loss에 이식한 대조로, PAAS/CALR 등 원논문 전체를 재현하지 않았다.
''')
        parts.append(table(pd.DataFrame([dict(arm=a,seed=s,selected_step=k) for a,v in selection['checkpoints'].items() for s,k in v.items()])))
        parts.append('''![적응/stream 및 lead](figures/budget_or_stream_and_leads.png)

## 실제 비용과 회수 가능성

G0/STATIC 공유 학습은 배치에서 각각 두 seed로 한 번만 수행했다. 각 후보의 독립 배포 추정에는 G0와 해당 생성기 학습, 필요한 reference forecast 계산을 각각 청구한다. reference cache를 공유해 절약한 연구 실행비용과 단독 준비 추정을 구분한다. 아래 cold 현장 시간은 실제 TEST 실행의 feature 구성+generator+query 평균에5회 읽기전용 benchmark의 G0 support4 중앙값을 더한 추정치다. 별도5회 generator/query benchmark는 준비된 record를 사용하므로 feature 구성 비용을 무료 처리하는 현장값으로 사용하지 않는다.

**비용 해석 제한:** optimizer trajectory 시간에는 update intent/commit 장부 I/O가 들어가고 읽기전용 forward에는 그 I/O가 없다. 준비비용 추정도 실제 개별 fit wall time과 reference 계산을 조합한 값이다. 따라서 이 값만으로20% 알고리즘 가속 성공을 선언하지 않는다. 손익분기는 동일 품질을 먼저 충족해야 하며 아래 수치 자체가 비용 회수 보증은 아니다. 음수·0인 시간 절약은 NEVER_RECOVERED로 남긴다. 동일 장치에서 seed2의 방법순서를 뒤집었으며5회 중앙값·최소·최대는 FORWARD_BENCHMARK에 있다.

offline fit의 GPU peak는 각 fit에서 reset해 측정했다. 현장/online resource 행의 peak는 마지막 offline reset 이후 process high-water이므로 방법별 독립 peak-memory 순위로 사용하지 않는다. 이 배치로 메모리 우위를 주장하지 않는다.
''')
        parts.append(table(cost[['arm','seed','baseline','standalone_prep_estimate_seconds','baseline_prep_estimate_seconds','field_cold_seconds','baseline_field_seconds','field_saving_pct','break_even_episodes','status']]))
        if track=='A':parts.append(f"\n동일 품질(오차1% 이내) 조건: {decision['efficiency']['same_quality']}. 현장 실측상20%절약 조건: {decision['efficiency']['operational_measured_saving20']}. 장부 비용을 분리한 엄밀한 속도 성공은 미확인이다.\n")
        parts.append(f'''
## 판정의 한계와 검산

특화 신호 조건 충족: **{decision['specific_signal']}**. {'A는 SET/NOERROR 각각0.5% 이상과 두seed 양성, DEV에서 고른 단순대조보다 낮은 오차를 구분해서 확인한다. LOCAL/COEFF와의 동일품질1% 조건은 효율 화면에서 따로 적용한다.' if track=='A' else 'B는 DEV-selected baseline/SET/FULLGEN 각각0.5% 이상과 두seed 양성을 확인한다.'} 모든 offline 선택이 끝checkpoint인 상태: **{decision['all_selected_final_checkpoint']}**. 끝checkpoint만 선택되는 경우 고정예산 내 수렴 불확실성을 보존하며 학습을 연장하지 않는다. 0.49%를 효과0이라고 부르지 않고, 0.5% 이상도 신규성 증거로 쓰지 않는다.

{'A 구간은 8고객을 재표집하고 고객별3연속episode block을 재표집한 paired bootstrap2,000회다.' if track=='A' else 'B는8tick 연속block을 모든계열·method·lead·seed에 같이 적용한 paired bootstrap2,000회와 별도의8고객 resampling 구간을 함께 공개한다.'} 같은 calendar의 상관과 적은 두seed의 한계는 남는다. 진단용 순서 역전은 고정 첫TEST episode에서 추가학습0회로 수행했으며 인과적인 시간 효과를 증명하지 않는다.

main 전 cuDNN eval-GRU backward 제약을 발견해 backend를 비활성화하고 native/gradient/복원 검사를 통과했다. 실패 전2회를 포함한 smoke누적14회이며 이 구현 수정은 성능 개선으로 세지 않는다. 본학습 결과와 구현·자료·자원 문제를 구분한다.

[공유 hypernetwork](https://aclanthology.org/2021.acl-long.47/)와 [PROCEED](https://lifan-zhao.github.io/publication/proceed/) 등 가까운 선행이 있다. 생성기나 부분 피드백 자체를 최초성으로 주장하지 않는다. [COSA 공식 core](https://github.com/bigbases/COSA_ICLR2026/blob/527c0feb9e997dd85af485ee027616b446e4ae77/tta/cosa.py)의 라이선스·고정 blob·parity를 보존했다.

[검산](VERIFICATION.json), [raw episode 점수](RAW_SCORES.csv), [seed 효과](SEED_EFFECTS.csv), [고객별 효과](SERIES_EFFECTS.csv), [lead 점수](LEAD_SCORES.csv), [적응 예산](ADAPTATION_BUDGET_CURVES.csv), [자원](RESOURCES.csv), [총비용·손익분기](AMORTIZATION_COSTS.csv), [발행 장부](ISSUED_FORECASTS.jsonl), [선택 봉인](SELECTION_SEAL.json), [예측 manifest](PREDICTION_MANIFEST.json).

raw data/HF weights/checkpoints/예측 npz는 로컬 ignored cache에 남기고 GitHub에는 코드·버전·hash·작은 집계표·그림을 남겼다. GitHub만으로 모든 개별 예측을 재채점할 수 있다는 뜻은 아니다. 기존 실험은 수정·재개하지 않았고 자동 후속 실험 없이 종료한다.
''')
        (out/'REPORT_KO.md').write_text('\n'.join(parts),encoding='utf-8')
        final=f"# 최종 판단\n\n**{decision['label']}**\n\n[확인] {prop} vs DEV-selected {baseline}: {primary.improvement_pct:+.4f}%, 두seed {seeds[0]:+.4f}% / {seeds[1]:+.4f}%. SET 대비 {direct.improvement_pct:+.4f}%, {feedback.control} 대비 {feedback.improvement_pct:+.4f}%.\n\n시간 구조 특화 조건 충족: {decision['specific_signal']}. 모든 선택이 끝checkpoint: {decision['all_selected_final_checkpoint']}. 엄밀한 비용 우위는 장부 I/O를 포함한 운영 시간과 읽기전용 시간이 달라 별도 한계를 둔다. 논문 PASS·신규성 확정·PEFT 전체 반증으로 해석하지 않는다.\n\n[그림·전체 보고서](REPORT_KO.md). 자동 후속 학습 없이 종료.\n"
        (out/'FINAL_DECISION.md').write_text(final,encoding='utf-8')
        summaries.append(dict(track=track,label=decision['label'],baseline=baseline,general_static_gain=general.improvement_pct,feedback_gain=feedback.improvement_pct,time_vs_set_gain=direct.improvement_pct,primary_gain=primary.improvement_pct,seed1=seeds[0],seed2=seeds[1]))
    counts=read(RESULTS/'VERIFICATION.json')['counts']
    text=f'''# Residual feedback PEFT 두 후보 통합 결과

[확인] A와 B를 독립적으로 완료했다. 신규 offline16경로 **{counts['offline']}updates**, A 현장 **{counts['A_local']}updates**, B 온라인 **{counts['B_online']}updates**, smoke **{counts['smoke']}updates**다. A 결과로 B를 취소하거나 두 모델을 합치지 않았다. 16경로는 offline 학습만 센 값이다. A 현장 비교는 DEV/TEST를 합쳐768개의 독립 episode trajectory, B는128개의 독립 고객 stream이며 비용을 별도로 포함했다.
'''+table(pd.DataFrame(summaries))+'''
각 gain은 서로 다른 질문이다. general_static_gain은 G0 이후 donor META 적응, feedback_gain은 NOERROR/FULLGEN 대비, time_vs_set_gain은 같은 정보의 일반 생성기 대비, primary_gain은 DEV에서 고정한 최강 합법 대조 대비다. 서로 더하거나 가장 좋은 항목으로 주장을 바꾸지 않는다.

- [A_AMORTIZED 한국어 보고서·그림](A/REPORT_KO.md) / [판정](A/FINAL_DECISION.md)
- [B_PARTIAL 한국어 보고서·그림](B/REPORT_KO.md) / [판정](B/FINAL_DECISION.md)
- [공통 검산](VERIFICATION.json), [독립 데이터 재검산](DATA_POSTRUN_AUDIT.json), [체크포인트·수치 재검산](CHECKPOINT_AND_METRIC_AUDIT.json), [실모델 사전검사](PREFLIGHT.json), [학습 장부](UPDATE_LEDGER.jsonl), [환경](ENVIRONMENT.json), [모델](MODEL_MAP.json), [원자료](DATA_SOURCE.json), [계열분리](SERIES_SPLIT.json), [시간분리](TIME_SPLIT.json), [reference manifest](REFERENCE_FORECAST_MANIFEST.json).

실행 정상 여부와 연구 효과를 구분했다. 사후 검산기의 잘못된 content-hash 유일성 조건을 수정했다. 같은 초기 모델의 서로 다른 방법이 동일 예측을 저장할 수 있기 때문이다. 발행 key/path 유일성과 파일 hash 일치는 계속 검사했으며 [초기 실패와 수정 근거](AUDIT_CORRECTION.json)를 보존했다. 학습·예측은 변경하지 않았다. 기존 부분정답 Maturity 규제의 FAIL도 보존한다. 새 후보는 기존 STOP을 해제하거나 추가설정을 탐색하지 않았다. 비용에는 공유 준비와 단독 사용 추정을 구분하며 optimizer 장부 I/O에 의한 시간 비교 한계를 공개했다. 신규성·논문 PASS를 선언하지 않으며 자동 후속 실험 없이 종료한다.
'''
    (RESULTS/'RESIDUAL_FEEDBACK_SUMMARY_KO.md').write_text(text,encoding='utf-8');print(table(pd.DataFrame(summaries)))

if __name__=='__main__':main()
