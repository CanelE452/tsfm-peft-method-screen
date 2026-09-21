from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/mag_standalone_ablation_v1_20260919'
def table(f):
 return '| '+' | '.join(map(str,f.columns))+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
audit=json.loads((OUT/'AUDIT.json').read_text());assert audit['status']=='VERIFIED' and audit['main_updates']==16384
raw=pd.read_csv(OUT/'RAW_SCORES.csv');eff=pd.read_csv(OUT/'EFFECTS.csv');inter=pd.read_csv(OUT/'INTERACTION_EFFECTS.csv');answers=pd.read_csv(OUT/'FIVE_QUESTIONS.csv');sel=pd.read_csv(OUT/'CHECKPOINT_SELECTION.csv');res=pd.read_csv(OUT/'RESOURCES.csv')
panels=['electricity','ettm1','electricity_transfer','neso_2026_jul_aug'];labels=['Electricity','ETTm1','전력 전이 16계열','NESO 7–8월']
q=answers[answers.condition=='SHIFT8'].set_index('panel').loc[panels].reset_index();q['자료']=labels
brief=q[['자료','F0_PLAIN_vs_F0_pct','F0_MAG_vs_F0_pct','F0_MAG_vs_PLAIN_pct','mag_gain_F0','mag_gain_B0']].rename(columns={'F0_PLAIN_vs_F0_pct':'Q1 PLAIN→F0 개선%','F0_MAG_vs_F0_pct':'Q2 MAG→F0 개선%','F0_MAG_vs_PLAIN_pct':'Q3 MAG→PLAIN 개선%','mag_gain_F0':'Q4 F0 절대감소','mag_gain_B0':'Q4 B0 절대감소'})
main=eff[(eff.stage=='selected')&(eff.kind=='standard')&(eff.condition=='SHIFT8')]
contrast=main[((main.proposed=='F0_MAG')&(main.baseline=='F0_PLAIN'))|((main.proposed=='B0_MAG')&(main.baseline=='B0_PLAIN'))][['panel','proposed','baseline','gain_pct','ci_low','ci_high','seed_gains']]
neso=eff[(eff.stage=='selected')&(eff.kind=='standard')&(eff.panel=='neso_2026_jul_aug')&(eff.proposed=='F0_MAG')&(eff.baseline=='F0_PLAIN')&eff.condition.isin(['SHIFT4','SHIFT_POINT'])][['condition','gain_pct','ci_low','ci_high','seed_gains']]
summary=raw[(raw.stage=='selected')&(raw.kind=='standard')&raw.condition.isin(['REFERENCE','FAULT','SHIFT4','SHIFT8','SHIFT_POINT'])].groupby(['panel','condition','arm'],as_index=False).nmae.mean().pivot(index=['panel','condition'],columns='arm',values='nmae').reset_index()
seedtable=raw[(raw.stage=='selected')&(raw.kind=='standard')&(raw.condition=='SHIFT8')].pivot(index=['panel','seed'],columns='arm',values='nmae').reset_index()
r=res.groupby('arm',as_index=False)[['trainable_parameters','optimizer_seconds','validation_seconds','peak_allocated']].mean();r['peak_allocated_MiB']=r.pop('peak_allocated')/2**20
# Presentation of existing contrasts, no scoring, model inference, fitting, or bootstrap.
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,terms,title in zip(axes,[['mag_gain_F0','mag_gain_B0'],['mag_specific_F0','mag_specific_B0']],['Total adapter benefit vs own base','Magnitude-rule benefit vs PLAIN']):
 for j,term in enumerate(terms):
  z=inter[(inter.stage=='selected')&(inter.kind=='standard')&(inter.condition=='SHIFT8')&(inter.term==term)].set_index('panel').loc[panels]
  ax.bar(np.arange(4)+(j-.5)*.34,z.nmae_difference,width=.32,label='F0 base' if j==0 else 'B0 base',color='#346fa0' if j==0 else '#c67834')
 ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(4),['Electricity','ETTm1','Elec. transfer','NESO'],rotation=20);ax.set_title(title);ax.set_ylabel('nMAE reduction (positive = benefit)');ax.legend()
fig.suptitle('SHIFT8 / selected / two fixed seeds; unequal cumulative adaptation budgets');fig.tight_layout();fig.savefig(OUT/'base_vs_rule.png',dpi=180);fig.savefig(OUT/'base_vs_rule.pdf');plt.close(fig)
if not (OUT/'RUNNER_REPORT.md').exists():
 (OUT/'RUNNER_REPORT.md').write_bytes((OUT/'REPORT.md').read_bytes());(OUT/'RUNNER_FINAL_DECISION.md').write_bytes((OUT/'FINAL_DECISION.md').read_bytes())
text=f'''# LoRA 선행 여부 직접 검증 — 최종 보고서

**16/16 fits, 본학습 16,384회와 smoke 8회, 지정 평가 및 독립 검산을 완료했다.** 기존 B0/PLAIN/MAG 재학습 0회, joint 학습 0회, 승인 범위 밖 추가 학습 0회다. 아래는 방법의 효과에 대한 개발 근거이며 논문 PASS 선언이 아니다.

## 먼저 답하는 다섯 질문

1. **F0+PLAIN은 F0보다 좋아졌는가?** SHIFT8에서는 네 패널 모두 개선했다. Electricity **27.06%**, ETTm1 **36.41%**, 전력 전이 **25.14%**, NESO **38.29%**다. 다만 REFERENCE/FAULT까지 모두 좋아진 것은 아니다.
2. **F0+MAG는 F0보다 좋아졌는가?** SHIFT8에서는 각각 **23.31%, 33.64%, 22.42%, 35.82%** 개선했다. LoRA 없이 어댑터를 학습해 예측을 개선하는 것은 가능했다. 그러나 이것만으로 MAG 규칙의 기여를 입증하지는 못한다.
3. **LoRA 없이도 MAG가 PLAIN보다 좋아졌는가?** SHIFT8에서는 각각 **5.15%, 4.36%, 3.63%, 4.02% 악화**했고 두 seed의 방향이 같았다. NESO의 날짜 구간은 0을 포함한다. 반면 NESO SHIFT_POINT는 **3.17% 개선**, 95% 날짜 구간 **[0.75%, 5.22%]**로 좁은 양성 근거가 있다. NESO SHIFT4도 **1.69% 개선**했으나 구간 **[−2.35%, 5.37%]**이어서 불확실하다. 양성·음성 조건을 모두 보존한다.
4. **B0+MAG와 F0+MAG 중 추가 이득이 더 큰 쪽은?** 자기 출발 모델 대비 **전체 어댑터의 절대 오차 감소**는 SHIFT8 네 패널 모두 F0 쪽이 컸다(아래 Q4). F0의 시작 오차가 더 크기 때문이다. 하지만 **PLAIN 대비 MAG 규칙 자체의 추가 가치**는 B0 쪽이 더 컸고, Electricity·전력 전이·NESO에서는 부호가 음수에서 양수로 바뀌었다. ETTm1의 B0+MAG는 step0 fallback으로 추가 개선이 없었다. 큰 감소량과 좋은 최종 정확도는 같은 뜻이 아니다.
5. **standalone PEFT인가, second-stage PEFT인가?** 구현상 둘 다 가능하지만, **현재 MAG 규칙의 주된 양성 근거는 선행 적응 B0 위의 제한된 second-stage 적용**이다. 범용 standalone MAG 우위는 확보하지 못했다. NESO SHIFT_POINT의 좁은 standalone 이득은 남긴다. “LoRA가 반드시 필요하다”, “LoRA 없이 MAG는 전혀 작동하지 않는다”, “LoRA가 필요 없다” 어느 쪽도 일반화할 수 없다.

다음 표는 selected·두 seed 평균이다. 상대 개선률은 `100×(1−방법오차/비교군오차)`이며 양수가 좋다. Q4는 상대%가 아닌 **절대 nMAE 감소**다. 표의 패널들을 합친 우승 점수는 만들지 않았다.

{table(brief)}

## 전체 어댑터 이득과 MAG 규칙의 추가 가치

![출발 모델 대비 전체 이득과 PLAIN 대비 규칙의 추가 가치](base_vs_rule.png)

[벡터 PDF](base_vs_rule.pdf). 이 그림은 기존 검산 CSV를 그린 것이며 새 학습·채점·bootstrap을 추가하지 않았다. B0는 과거 LoRA 학습을 거친 출발점이므로 두 경로의 누적 예산은 다르다. 차분을 완전한 인과효과로 해석하지 않는다.

SHIFT8에서 MAG와 PLAIN을 직접 비교한 결과:

{table(contrast)}

위 구간은 고정된 두 seed와 채널에 조건부인 **2,000회 paired index 7일 block bootstrap의 95% 구간**이다. 날짜가 아닌 seed/조건/draw/채널을 독립 날짜로 세지 않았다. optimizer-seed 모집단 불확실성이나 과거 전체 후보 탐색의 다중검정을 해결한 구간은 아니다. 계약에 없는 새 전역 PASS 문턱이나 보정 family는 만들지 않았다.

NESO의 좁은 standalone 양성 조건:

{table(neso)}

`CONDITION_DECISIONS.json`의 `STANDALONE_MAG_SUPPORTED`는 두 seed의 조건별 양성 방향을 나타내며, 그 자체가 신뢰구간 유의성·독립 검증·논문 성공 판정은 아니다. 특히 SHIFT4는 구간이 0을 포함한다. [모든 조건의 다섯 질문 답변](FIVE_QUESTIONS.csv), [seed별 효과](SEED_EFFECTS.csv), [절대 감소 및 interaction](INTERACTION_EFFECTS.csv)을 함께 읽어야 한다.

## 원점수와 손해

아래는 주요 다섯 조건의 selected·두 seed 평균 nMAE다. 낮을수록 좋다. FAULT는 기존 여섯 POINT/BURST 조건의 동등 가중 평균이다.

{table(summary)}

F0_MAG는 FAULT에서 F0보다 Electricity **0.95%**, ETTm1 **4.03%**, 전력 전이 **4.72%**, NESO **4.09%** 나빴다. PLAIN 대비 FAULT도 네 패널 모두 나빴다. 변화 조건의 개선을 오류 강건성 전반의 성공으로 옮기지 않는다. REFERENCE 손해도 표에 남겼다.

SHIFT8의 seed별 원점수:

{table(seedtable)}

전체 standard 10조건과 FAULT 집계, 9개 변화 형태, selected/fixed1024, 원점·채널별 기록은 [RAW_SCORES.csv](RAW_SCORES.csv), [EFFECTS.csv](EFFECTS.csv), [ORIGIN_SCORES.csv.gz](ORIGIN_SCORES.csv.gz), [ORIGIN_CHANNEL_SCORES.csv.gz](ORIGIN_CHANNEL_SCORES.csv.gz)에 있다. raw MAE, 채널 평균 nRMSE, normalized twice-pinball, 분위수 교차율도 RAW_SCORES.csv에 포함했다. 분위수 정렬로 출력을 수정하지 않았다. [조건별 이득 그림](tradeoffs.png) · [여섯 비교군 원점수 그림](comparison.png).

## 선택과 구현

{table(sel[sel.stage=='selected'])}

F0는 pinned pretrained Chronos-Bolt-small, F0_PLAIN/F0_MAG는 LoRA 없이 residual 512→8→512만 학습한다. 기존 `MAG_ONLY`는 no-LoRA가 아니며 이번 표의 B0_MAG에 해당한다. `F0_PLAIN`과 `F0_MAG`의 동일 source/seed 초기 어댑터는 bitwise 동일하고, up weight/bias는 0이다. MAG는 기존 raw observation median/MAD·floor 0.1sigma·threshold 3·patch16 식을 변경하지 않았다.

전체 foundation parameter/buffer를 동결했고 새 trainable은 정확히 8,712개, 모두 `adapter.*`에만 있다. LoRA parameter/module은 0개다. 관측 입력과 TRAIN sigma만 forward에 들어가며 미래 y, clean x0, 합성 state/mask/delta는 전달하지 않는다. raw 입력은 바꾸지 않는다. [실제 모델 검사](NO_LORA_AUDIT.json) · [기존 B0 q/v rank8 검사](BASELINE_STRUCTURE_CHECK.json).

모든 학습률은 고정 후보 1e-4/3e-4 중 V로 선택했으며 새 두 군은 모두 3e-4였다. seed81550으로 선택, 81551/81552로 반복했다. 32epochs×32updates, batch32, FP32/TF32off, dropout0, AdamW(.9,.999)/eps1e-8/wd0, gradclip1, scheduler 없음, normalized twice-pinball과 기존 V 다섯 조건 평균 nMAE를 유지했다. 기존 TRAIN/V/E draws·labels·sigma·origins·순서 계약을 hash 재사용했다.

step0 선택은 출발 모델로의 fallback이며 추가 PEFT의 성공으로 세지 않는다. F0의 seed/stage 반복 표기는 같은 고정 모델을 정렬한 것으로 독립 반복 실험이 아니다. B0 fixed1024는 기존 selected B0 위 추가 어댑터의1024updates 비교이며 B0 자체를 다른 checkpoint로 바꾼 것이 아니다.

## 파라미터와 누적 비용

| 경로 | 해당 적응 stage 학습 파라미터 | 배포되는 적응 파라미터 | 누적 적응 stage | 반복 경로당 실제 학습 비용 |
| --- | ---: | ---: | ---: | --- |
| F0 | 0 | 0 | 0 | 0 |
| F0_PLAIN / F0_MAG | 8,712 | 8,712 | 1 | 이번 1,024 updates |
| B0 | 과거 294,912 | 294,912 | 1 | 과거 LoRA 1,024-update fit 재사용 |
| B0_PLAIN / B0_MAG | 과거 추가 8,712 | 303,624 | 2 | 과거 LoRA 1,024 + adapter 1,024-update fit 재사용 |

위 단일 반복 경로 비용 외에 LR 선택용 학습도 존재한다. 이번 실제 총량은 **16 fits / 16,384 main + 8 smoke**이며 선택용 경로를 제외해 비용을 작게 표시하지 않았다. 과거 B0 선택 step과 fit 비용은 [HISTORICAL_COST.json](HISTORICAL_COST.json)에 있다. 기존 LoRA를 이번 새 학습으로 다시 계산하지 않지만 누적 비용에서 지우지도 않는다. compute-matched 2,048-update standalone 또는 joint LoRA+MAG는 실행하지 않았다.

아래 시간은 현재 runner에서 fit별로 측정한 평균이다. `optimizer_seconds`에는 optimizer intent 저장 등이 포함되며 GPU 경계 검사 등 전체 실행시간과는 다르다. 중단 경로의 해당 시간은 resume 상태에서 누적 복원했다. 과거 runner와의 시간비를 순수 계산 속도 우위로 주장하지 않는다.

{table(r)}

## 실행·검산과 중단 이력

- **완료 16/16 fits, unique main updates 16,384, smoke 8.** 미실행 승인 학습 0, 기존 B0/PLAIN/MAG 재학습 0.
- 실제 no-LoRA 검사, native F0/step0/adapter-off 예측 동일성, finite gradient와 parameter 변경, foundation/buffer 보존, 배치 순서 정합, 복원 동일성을 확인했다.
- 80개 checkpoint hash, LR·checkpoint 선택 및 selected/fixed1024 연결을 검증했다. 새 모델 복원 예측 검사는 각 실제 새 prediction 파일의 앞32행을 재추론해 정확 일치시켰다. 전체 E를 두 번 재추론했다는 뜻은 아니다.
- 선택 봉인 뒤 **192 prediction views를 모두 저장한 후** 채점했다. 중복 참조를 제외한 로컬 prediction 파일은124개다. F0·selected/fixed 동일 체크포인트 등의 alias를 새 학습이나 독립 표본으로 세지 않았다.
- 첫/마지막 origin·첫/마지막 channel의 독립 scalar metric 12,768개, 전체 저장 예측의 독립 metric **793,560개**, 기존 점수 **960행**, 원점수→효과·interaction·bootstrap을 검산했다.
- 사용자 요청으로 두 차례 멈춘 기록을 보존했다. 최종 모든 경로에서 journal step1..1024, epoch 순서, AdamW state step1024, resume와 최종 checkpoint 일치를 확인했다. 중복 update 0. [재개·상한 검산](RESUME_AND_BUDGET_AUDIT.json).
- 긴 사용자 일시정지 기간만 실행시간 계수에서 제외했다. 실제 실행시간이나 GPU 대기 예산을 초기화하지 않았고,8시간 guard 및 학습 상한은 유지했다. [시간 계수 기록](USER_PAUSE_TIME_ACCOUNTING.json).
- RustDesk만 외부 GPU 예외였고, 미승인 외부 compute 표본은0개였다. 예전 ERROR/EXECUTION_ERROR의 `REQUESTED_STOP_EPOCH_BOUNDARY`는 사용자 일시정지 이력이며 최종 상태는 `COMPLETE_VERIFIED`다.

[최종 검산](VERIFICATION.json) · [감사 요약](AUDIT.json) · [예측 manifest](PREDICTIONS_MANIFEST.json) · [단일 실행 계약](../../experiments/mag_standalone_ablation_v1_20260919/CONTRACT.txt). 원자료·가중치·예측은 로컬 캐시이며 GitHub에는 hash·manifest·집계/검산을 게시한다. GitHub만으로 모든 수치가 재현된다고 주장하지 않는다.

## 논문 해석과 종료

현재 결과는 **기반 모델의 선행 적응 여부와 조건에 따라 고정 MAG 규칙의 추가 가치가 달라진다**는 개발 근거다. 큰 전체 어댑터 이득을 MAG 규칙의 기여로 바꾸어 말하지 않는다. B0가 최종 정확도에서 더 좋더라도 누적 학습량과 파라미터가 더 많으므로 LoRA의 필수성을 증명한 것은 아니다.

네 패널 모두 이미 사용한 개발 평가다. NESO는 같은 provider의 기존 노출 기간이며 독립 source가 아니다. 실제 오류·실제 변화 사건 레이블은 없고 합성 조건에서의 비교다. 정식 선행 전체 우위, 독립 source 일반화, 충분한 신규성 및 논문 게재 가능성은 이번 실험만으로 확정하지 않는다. 추가 seed/LR/rank/gate/threshold/data, joint, 새 후보 및 후속 학습은 시작하지 않고 이 범위에서 종료한다.
'''
(OUT/'REPORT.md').write_text(text)
(OUT/'FINAL_DECISION.md').write_text('''# 최종 판단

실행·평가·독립 검산은 완료했다. 논문 PASS를 선언하지 않는다.

**현재 MAG 규칙의 주된 근거는 선행 적응 B0 위의 제한된 second-stage PEFT다. 범용 standalone MAG 우위는 확보하지 못했다.**

- F0에 PLAIN 또는 MAG 어댑터만 학습해도 SHIFT8 오차는 크게 줄었다. 따라서 LoRA 없는 어댑터 학습 자체는 가능하다.
- 하지만 SHIFT8의 F0_MAG는 F0_PLAIN보다 Electricity5.15%, ETTm1 4.36%, 전력 전이3.63%, NESO4.02% 나빴다. 네 패널 모두 두 seed의 방향이 같았고 NESO 날짜 구간은0을 포함했다.
- B0_MAG의 PLAIN 대비 SHIFT8 이득은 Electricity1.46%, 전력 전이3.79%, NESO1.94%였다. ETTm1은 step0 fallback이므로 MAG의 추가 개선이 없었다.
- 예외를 지우지 않는다. NESO SHIFT_POINT에서는 standalone MAG가 PLAIN보다3.17% 좋았고 날짜95%구간[0.75%,5.22%]였다. SHIFT4의1.69% 양성은 구간이0을 포함해 불확실하다.
- F0_MAG의 FAULT는 F0보다 네 패널 모두 나빴다. 변화 조건의 이득을 오류 강건성 전반의 성공으로 주장하지 않는다.

전체 어댑터의 자기 출발점 대비 절대 감소량은 SHIFT8에서 F0 쪽이 더 크지만, MAG 규칙의 PLAIN 대비 추가 가치는 B0 쪽이 더 컸다. 누적 예산·출발 오차가 달라 이 차이를 LoRA의 필요/불필요에 대한 보편적 인과 결론으로 해석하지 않는다.

구현은 F0_PLAIN/F0_MAG와 기존 B0 경로 모두 재현용으로 보존한다. 기존 좁은 B0 양성 결과와 standalone 예외도 남긴다. 독립 source·실제 사건·정식 선행 전체 비교·신규성은 해결됐다고 주장하지 않는다.

16fits/16384main+8smoke로 승인 범위를 종료한다. 추가 학습·joint·자동 successor·새 후보는 실행하지 않는다. 상세 원점수·seed·불확실성·비용·검산은 [REPORT.md](REPORT.md)를 따른다.
''')
files=['AUDIT.json','VERIFICATION.json','RAW_SCORES.csv','EFFECTS.csv','SEED_EFFECTS.csv','INTERACTION_EFFECTS.csv','FIVE_QUESTIONS.csv','CHECKPOINT_SELECTION.csv','RESOURCES.csv']
save(OUT/'REPORT_PRESENTATION_MANIFEST.json',dict(source_hashes={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in files},scope='Presentation of already independently verified results; no fitting/inference/scoring/bootstrap',scientific_settings_changed=False,main_updates=16384,smoke_updates=8))
print('POLISHED_REPORT_WRITTEN')
