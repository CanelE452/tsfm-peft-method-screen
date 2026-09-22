import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import pandas as pd

assert read(RESULTS/'POSTRUN_AUDIT.json')['status']=='PASS'
decision=read(RESULTS/'DECISION.json')
scores=pd.read_csv(RESULTS/'SCORES.csv')
effects=pd.read_csv(RESULTS/'EFFECTS.csv')
seed=pd.read_csv(RESULTS/'SEED_EFFECTS.csv')
resources=pd.read_csv(RESULTS/'RESOURCES.csv')
targets=pd.read_csv(RESULTS/'TIME_TO_TARGET.csv')
def effect(source,candidate,control,variant):
    return effects.query('source==@source and candidate==@candidate and control==@control and variant==@variant').iloc[0]
def block(df):return '\n```text\n'+df.to_string(index=False,float_format=lambda x:f'{x:.6f}')+'\n```\n'

parts=[f'''# 시간 의존성 LoRA 초기화 개발 screen

[확인] 16 fits / 8,192 main optimizer updates + 8 smoke updates를 완료했다. 최종 판정은 **{decision['overall']}**이다. 실행·자료·GPU 오류가 아니라 사전 고정 대조에서 얻은 과학적 screen 결과이다. 학습을 추가하거나 LR/rank/seed/lag를 바꿔 결과를 구제하지 않았다.

이번 후보는 **TSFM의 대부분을 동결하고 작은 LoRA를 학습하되, 시간차 공분산 방향으로 초기화하는 PEFT 학습 방법**이다. 방법 범주에 속한다는 것과 유용한 새 방법임을 입증했다는 것은 별개다. 이 실험은 후자를 검토한다.

## 무엇을 고정해서 비교했나

Chronos-Bolt-small의 같은 pretrained revision에서 q/v rank8 LoRA 294,912개 파라미터를 학습했다. RANDOM은 일반 LoRA, PCA는 입력 표현의 분산 방향, TEMP는 인접 16시간 patch의 시간차 공분산 방향, SHUFFLE는 시간 짝을 섞은 대조다. 초기화 후 A와 B 모두 학습하며, encoder q/v A만 구조화한다. Decoder와 cross-attention은 같은 seed의 일반 LoRA 초기화를 유지한다.

모든 B는 0에서 시작한다. 실제 native prediction parity, 네 군 초기 예측의 완전 일치, gradient, frozen backbone/buffer, 저장·복원을 GPU에서 검사했다. 초기화에는 TRAIN의 입력 256개만 사용했다. 비중첩 input context, window 중심화, 같은 row norm, 같은 seed별 minibatch 순서를 적용했다. 시간 짝만 변경하는 SHUFFLE와 학습 기회가 같다.

Electricity 8계열과 ETTh1 7계열, context256/horizon64, hourly native 예측이다. 모든 fit을 512updates까지 실행했으며 seed92341/92342는 모두 반복 평가에 포함했다. TRAIN/CAL/VAL/TEST 60/10/10/20%, 평가 target은 64시간 간격으로 비중첩이다. 원자료 SHA, 기존 캐시와 원자료의 실제 값, 시간축·결측, 분할과 TRAIN scale을 재검사했다. Electricity에 원자료에 없는 달력 timestamp를 만들지 않았다.

Primary는 **고정128 updates의 raw TEST normalized twice-pinball**이다. 0/32/128/256/512 중 VAL 최소 checkpoint와 CAL의 공통 위치·폭 보정은 별도 보조 비교다. RANDOM/PCA 중 primary baseline도 두 seed 평균 VAL128에서 TEST 전에 선택했다. 점수는 낮을수록 좋고 개선율은 양수일수록 좋다. 정확한 CRPS로 부르지 않는다.

## 자료별 결과
''']
for source in SOURCES:
    d=decision['datasets'][source];base=d['baseline']
    e=effect(source,'TEMP',base,'raw128');sh=effect(source,'TEMP','SHUFFLE','raw128');r=effect(source,'TEMP','RANDOM','raw128');general=effect(source,'RANDOM','F0','selected_cal')
    se=seed.query("source==@source and candidate=='TEMP' and control==@base and variant=='raw128'")
    parts.append(f'''### {source}

사전 VAL 선택 baseline은 **{base}**다. TEMP의 고정128 TEST 개선율은 baseline 대비 **{e.improvement_pct:+.4f}%**, SHUFFLE 대비 **{sh.improvement_pct:+.4f}%**, RANDOM 대비 **{r.improvement_pct:+.4f}%**다. baseline 대비 seed별 개선은 {', '.join(f'{x:+.4f}%' for x in se.improvement_pct)}다. 시간블록 bootstrap 95% 구간은 [{e.ci_low:+.4f}%, {e.ci_high:+.4f}%]다.

일반 RANDOM LoRA의 VAL 선택+CAL 성능은 같은 CAL 기회를 받은 F0 대비 **{general.improvement_pct:+.4f}%**다. 이것은 일반 적응 효과이며 TEMP의 추가 효과와 구분한다. 자료별 판정: **{d['status']}**.
''')
    means=scores.query('source==@source').groupby(['variant','arm'],sort=False)[['pinball','nmae','raw_mae','coverage','width','crossing']].mean().reset_index()
    parts.append(block(means))
parts.append('''## 초기화 비용과 속도

![검증 학습곡선](figures/learning_curves.png)

왼쪽은 update 수, 오른쪽은 초기화 포함 적응 계산시간이다. 선은 두 seed 평균, 음영은 두 seed 범위다. Activation 수집을 공유했어도 PCA/TEMP/SHUFFLE의 개별 사용 비용에는 전체 수집 비용을 각각 청구했다. 모델 로딩·자료 읽기·VAL·checkpoint/장부 I/O는 적응 계산시간에서 분리하고 실제 wall time도 아래에 공개한다. GPU peak는 training만이 아니라 VAL/CAL batch32를 포함한다.

시각 측정은 동일 GPU에서 순차 실행한 실측값이며 격리된 반복 성능 벤치마크가 아니다. 일부 fit에서 동일 update 수의 시간이 크게 달라졌다. 따라서 높은 시간 절약률 하나만으로 구조적 속도 우위를 주장하지 않는다. 사전 품질 조건을 먼저 만족해야 한다.
''')
parts.append(block(resources[['source','arm','seed','selected','prep_seconds','adaptation_seconds512','fit_wall_seconds','peak_allocated']]))
parts.append('''목표는 같은 seed의 RANDOM512 VAL score ×1.01이다. 고정 checkpoint에서 첫 도달만 기록한다. F0에서 이미 도달하면 NO_ADAPTATION_HEADROOM, 미도달은 CENSORED이며 사후에 목표를 완화하지 않는다.
'''+block(targets))
for source in SOURCES:
    d=decision['datasets'][source]
    value=f"{d['time_saving_pct']:+.4f}%" if d['time_comparison_valid'] else '계산 불가(미도달 또는 초기 headroom 없음)'
    parts.append(f"\n{source}: TEMP의 초기화 포함 평균 time-to-target 절약률은 **{value}**다.\n")
parts.append('''## 해석과 중단 판단

이번 결과를 단순히 “아무것도 안 됐다”고 표현하지 않는다. TEMP는 일반 RANDOM LoRA보다 두 자료에서 작게 개선했고, 사전 선택 baseline 및 SHUFFLE 대비도 점추정은 두 seed 모두 양성이다. 다만 시간 짝을 이용한 추가 효과가 Electricity0.0642%, ETTh1 0.0071%로 매우 작고 두 구간 모두0을 포함한다. **작은 양성 관찰은 있지만, 내일의 주력 방법 후보로 확정할 실용적·기전적 근거는 부족하여 HOLD**다.

ETTh1에서는 primary baseline을 VAL에서 RANDOM으로 고정했지만 TEST의 PCA는 TEMP보다 더 좋았다(TEMP의 PCA 대비 개선율 −0.0894%). 이 사실도 숨기지 않으며, TEST를 보고 baseline을 다시 선택하지 않는다. 시간 의존성을 사용하지 않는 초기화도 충분할 수 있다는 반례다.

두 자료 모두 TEMP와 RANDOM의 목표 도달 update가 seed별로 정확히 같았다(Electricity128/128, ETTh1 256/512). ETTh1의 측정 시간 절약27.74%는 더 적은 update로 수렴한 증거가 아니다. RANDOM 첫 seed가 느리게 실행된 영향이 크고 원인은 확인되지 않았다. Electricity는 준비 비용 포함5.88% 더 느렸다. **현재 측정으로 알고리즘 자체의 적응 가속을 입증하지 못했다.**

![고정128 TEST 효과](figures/test_effects.png)

그림의 빨간 점은 두 seed 원점수 평균에서 계산한 개선율, 검은 점은 개별 seed다. 구간은 계열과 두 seed를 함께 유지한 7개 연속 origin block bootstrap 2,000회다. 64시간 간격의 7origin이므로 block 길이는 448시간이다. 두 seed로 seed 모집단의 불확실성을 충분히 추정했다는 뜻이 아니다.

사전 기준은 두 자료 각각 baseline 및 SHUFFLE 대비 평균0.5% 이상 개선, 두 seed 양성이다. 이 품질 조건과 time-to-target20% 절약, 선택+CAL에서 baseline보다1% 넘게 악화하지 않는 조건을 모두 충족해야 GO다. 0.5%는 보편적인 논문 통과 기준이 아니라 작은 개발 투자의 선별값이다. 결과에서 음수인 대조는 기준을 낮춰도 개선 근거가 되지 않는다.

PCA 대비 이득이 없으면 일반적인 데이터 기반 초기화로 충분할 수 있다. SHUFFLE 대비 이득이 없으면 실제 시간 짝이 추가 가치의 원인이라는 근거가 없다. 일반 LoRA가 F0를 개선하는 것만으로 TEMP 가설이 확인되는 것은 아니다. 현재 판정은 고정된 이 구현·자료·예산에 한정하며 모든 시간 기반 PEFT나 충분히 학습한 모델의 성능에 대한 반증은 아니다.

## 한계·선행과 재현 범위

- 이 자료는 기존 후보 개발에 이미 사용됐다. 독립 holdout 확증이나 배포 검증이 아니다.
- Encoder 표현은 양방향 attention 이후이므로 인접 patch covariance를 순수한 인과적 시간 의존성으로 해석할 수 없다. 단일 16시간 patch lag만 시험했으며 계절성 전체를 대표하지 않는다.
- PCA는 같은 rank·용량의 분산 초기화 대조다. [EVA](https://arxiv.org/abs/2410.07170)의 rank 재배분을 포함한 전체 재현이 아니다. [CorDA](https://arxiv.org/abs/2406.05223)처럼 covariance를 활용한 선행도 있어 초기화 아이디어 자체의 신규성을 선언하지 않는다.
- 512updates의 제한된 적응 screen이다. 일부 VAL 곡선이 계속 개선할 수 있어 수렴 성능을 주장하지 않는다. 그 이유로 예산을 늘리거나 재탐색하지 않았다.
- 원자료·HF 가중치·checkpoint·예측 npz는 로컬 ignored cache에 보존했다. GitHub에는 코드, 정확한 버전, upstream·SHA manifest, 점수표·origin 집계·그림·검산을 공개한다. GitHub 파일만으로 저장 예측을 완전히 재채점할 수 있다는 뜻은 아니다.

검산 파일: [GPU 사전검사](PREFLIGHT.json), [본검산](VERIFICATION.json), [CPU 사후검산](POSTRUN_AUDIT.json), [환경·모델](MODEL_AND_ENVIRONMENT.json), [version lock](requirements-lock.txt), [선택 봉인](SELECTION_SEAL.json), [예측 manifest](PREDICTIONS_MANIFEST.json), [fit 장부](FIT_LEDGER.csv), [seed 효과](SEED_EFFECTS.csv), [lead 점수](LEAD_SCORES.csv), [계열 점수](SERIES_SCORES.csv), [origin 점수](ORIGIN_SCORES.csv), [자원표](RESOURCES.csv).

초기 학습 코드는 main 전에 봉인하고, 결과 분석 코드는 별도 analysis 경로에 두었다. 모든 TEST 예측 저장 후 채점했다. 사후검산은 새 모델 추론·추가 optimizer update 없이 저장된 checkpoint·optimizer state·원자료와 독립 pinball 수식으로 수행했다. 자동 후속 실험 없이 종료한다.
''')
(RESULTS/'REPORT_KO.md').write_text('\n'.join(parts),encoding='utf-8')
summary=[f"# 최종 판단\n\n**{decision['overall']}**\n\n[확인] 16 fits / 8,192 main + 8 smoke updates 완료. 자료·구현·자원 오류가 아닌 고정 조건의 과학적 screen 결과다.\n"]
for source in SOURCES:
    d=decision['datasets'][source];e=effect(source,'TEMP',d['baseline'],'raw128');sh=effect(source,'TEMP','SHUFFLE','raw128')
    summary.append(f"- {source}: TEMP vs {d['baseline']} {e.improvement_pct:+.4f}%, vs SHUFFLE {sh.improvement_pct:+.4f}%. {d['status']}.")
summary.append('\nTEMP는 일반 LoRA보다 작게 개선했으나, 시간 짝을 섞은 대조의 성능과 거의 같고 목표 도달 update 수가 동일하다. 작은 양성 관찰은 보존하되 실용적인 시간 관계의 추가 가치를 입증했다고 하지 않는다. ETTh1의 시간 절약 실측은 실행시간 변동에 민감하다.\n\n시간 기반 LoRA 초기화는 TSFM PEFT 방법론 후보에 해당한다. 그러나 방법 범주·일반 LoRA 적응 효과와 새 초기화의 추가 가치는 구분해야 한다. 판정은 이 초기화 구현에 한정하며 PEFT 전체 반증이나 논문 PASS가 아니다. [그림과 전체 보고서](REPORT_KO.md). 자동 후속 학습 없이 종료한다.\n')
(RESULTS/'FINAL_DECISION.md').write_text('\n'.join(summary),encoding='utf-8')
print('\n'.join(summary))
