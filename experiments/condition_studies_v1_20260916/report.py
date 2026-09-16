"""Korean evidence reports; execution and novelty are separate from measured gains."""
import csv,time
import pandas as pd
from .common import *

def table(df,columns=None,limit=40):
 if columns:df=df[columns]
 if len(df)>limit:df=df.head(limit)
 if not len(df):return '측정된 값 없음.'
 def fmt(x):return (f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)).replace('|','/')
 return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))

def literature():
 sources={
 'N01':('t-PatchGNN','https://proceedings.mlr.press/v235/zhang24bw.html','공식 초록·서지','불규칙 관측의 패치와 시간 적응 그래프로 비동기 채널 관계를 모델링한다. 이번 ridge/age 입력 보정은 해당 구조의 재현이 아니다.'),
 'N02':('RAFT','https://proceedings.mlr.press/v267/han25d.html','공식 초록·서지','유사한 학습 과거와 그 이후 관측을 검색해 입력에 활용한다. 검색 자체와 continuation 사용은 알려져 있으며 이번8계수 보정의 독자성은 미확인이다.'),
 'N03':('FlowState / t-PatchGNN','https://research.ibm.com/publications/flowstate-sampling-rate-equivariant-time-series-forecasting','기관 초록·ICML2026 게재 정보','FlowState는 SSM 인코더와 함수 기저 디코더로 관측률 적응을 다룬다. 이번 가림 모사의 단순 kernel 보간과 학습4계수는 그 아키텍처 재현이 아니다.'),
 'R04':('Dynamic loss weighting for forecast stability','https://arxiv.org/html/2409.18267v2','저자원문v2의 초록·안정성 정의','정확도와 수정 안정성의 다중목적 및 동적 가중은 알려져 있다. 2024공개/2025v2로 기재하며 최종저널연도는 미검증이다. 이번 혁신량 역수 가중과 제한된 LoRA 비교의 결과로 논문 전체를 이겼다고 하지 않는다.'),
 'R05':('기존 PATH/LATEST 비교','../../forecast_path_structure_v1_20260916/REPORT.md','로컬 코드·저장 예측·완료 감사','저장된 선택모델의 convex quantile vector 정책 재분석이다. 학습법 신규성은 평가하지 않으며 확률분포 mixture의 quantile이 아니다.'),
 'N06':('TACTiS-2 / correlated sample paths','https://proceedings.iclr.cc/paper_files/paper/2024/hash/63796148c99205adb0fcac069cc714d4-Abstract-Conference.html','공식초록과 Amazon기관 워크숍 초록','copula와 rank 재배열은 알려진 대조다. Amazon correlated sample paths는 NeurIPS2025 TSFM Workshop이며 메인학회 논문으로 쓰지 않는다. 정식 구현 재현은 수행하지 않는다.'),
 'N07':('Fredformer / MSFT','https://arxiv.org/html/2406.09009v2','저자원문의 초록·주파수 편향 설명; MSFT저자원문','Fredformer는 주파수대별 처리·정규화로 에너지 편향을 다룬다. MSFT는 다중척도 finetuning이다. 고정 Fourier 가중 loss는 이 구조들의 재현이나 최초성 증거가 아니다.'),
 'R08':('LIFT','https://proceedings.iclr.cc/paper_files/paper/2024/hash/b52b07a239a7afa155ca25cf17a55074-Abstract-Conference.html','공식초록·서지','선행 채널 정보와 지연 정렬은 알려진 접근이다. 이번 TRAIN 고정시차 ridge 및8계수 가용성 보정은 동적 선행모델 LIFT의 전체 재현이 아니다.'),
 'R09':('Temporal Disaggregation','https://journal.r-project.org/articles/RJ-2013-028/','공식문서의 문제·서지','시간 집계와 상세 복원은 Denton/Chow–Lin 계열의 오래된 문제다. NULL 투영은 알려진 수학이며 IMPUTE와 계수1 NULL의 동일성을 직접 검산한다.')}
 for t,(name,url,scope,desc) in sources.items():
  (OUT/t/'LITERATURE_BOUNDARY.md').write_text(f'# 선행과 주장 경계\n\n[확인] [{name}]({url}) — 읽은 범위: {scope}.\n\n{desc}\n\n[미검증] 정식 선행 구현과 동등 예산 직접 비교는 수행하지 않았다. 이번 단순 대조 우위는 해당 논문 우위 또는 NEW_METHOD를 뜻하지 않는다. 알려진 구성요소에 대한 제한된 조건 비교다.\n')
  (OUT/t/'LIMITATIONS.md').write_text('# 한계\n\n단일 원천/동일 프로그램에서 노출된 기간의 제한된 개발 비교다. 두 반복seed는 새 도메인이 아니다. 2,000 paired 7일 bootstrap은 관측한 원천·채널·seed에 조건부이며 다중 주제 선택/사전학습 중복의 불확실성을 해결하지 않는다. 두 학습률과512update 한도이며 모든 LoRA recipe를 대표하지 않는다. 추가계수·보조입력·CPU fitting은 비용 차이다. 정식 선행 미재현은 LITERATURE_BOUNDARY.md 참조.\n')

def plot_track(t):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=OUT/t;df=pd.read_csv(out/'contrasts.csv')
 if t in SOURCES:
  d=df[(df.policy=='selected')];labels=[f'{r.baseline}/{r.seed}/{r.condition}' for r in d.itertuples()];vals=d.gain_percent.to_numpy()
 else:
  d=df[(df.scope=='ALL')&(df.variant=='RAW')];labels=[f'{r.method}:{r.baseline}/{r.condition}/{r.metric}' for r in d.itertuples()];vals=d.gain_percent.to_numpy()
 fig,ax=plt.subplots(figsize=(10,max(4,min(24,len(d)*.18))));ax.barh(range(len(vals)),vals,color=np.where(vals>=0,'#26786b','#b95158'));ax.set_yticks(range(len(vals)),labels,fontsize=6);ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Gain vs direct control (%)');ax.set_title(t+' paired gains (all reported conditions)');fig.tight_layout();fig.savefig(out/'paired_gains.png',dpi=140);plt.close(fig)
 if t in SOURCES:
  d=pd.read_csv(out/'scores_summary.csv');d=d[(d.policy=='selected')&(d.condition=='PRIMARY')];fig,ax=plt.subplots(figsize=(8,4));arms=list(d.arm.unique())
  for seed in sorted(d.seed.unique()):
   g=d[d.seed==seed];ax.scatter([arms.index(a) for a in g.arm],g.score,label=str(seed))
  ax.set_xticks(range(len(arms)),arms,rotation=30);ax.set_ylabel('Primary normalized RMSE');ax.legend()
 else:
  d=pd.read_csv(out/'raw_scores.csv');metric='normalized_2pinball' if t=='R05' else 'sum_CRPS_normalized';d=d[(d.metric==metric)&(d.variant=='RAW')];g=d.groupby(['arm','target']).score.mean().unstack();fig,ax=plt.subplots(figsize=(8,4));g.plot.bar(ax=ax);ax.set_ylabel(metric)
 ax.set_title(t+' target / repeat trade-offs');fig.tight_layout();fig.savefig(out/'condition_tradeoffs.png',dpi=140);plt.close(fig)

def report_track(t):
 out=OUT/t;st=read(out/'STATUS.json');spec=read(out/'PROTOCOL.json');state=st['EXECUTION'];text=f'# {t} {NAMES[t]} 결과\n\n실행: **{state}** / 근거: **{st["EVIDENCE"]}** / 신규성: **{st["NOVELTY"]}**.\n\n## ① 문제와 정보\n\n{QUESTIONS[t]}. [계약과 방법](TOPIC_ONEPAGE.md), [정보 권한](permissions.json), [원천·원점](data_receipt.json). 원점 이후의 정답은 입력으로 허용하지 않는다. R09 숨긴 상세값의 권한은 절대시간 블록에 적용한다.\n\n## ② 선행 연결\n\n[LITERATURE_BOUNDARY.md](LITERATURE_BOUNDARY.md). 알려진 구성요소와 이번 제한된 변형을 구분하며 정식 선행의 전체 재현을 주장하지 않는다.\n\n## ③ 비교 조건과 비용\n\n군: '+', '.join(spec['arms'])+'. '+('rank8 LoRA 1,179,648개 + 명시 보조계수, FP32 point MSE, TRAIN64,512updates, 선택seed73100의2LR 후 반복73101/73102. tau=.5 슬롯을 점예측으로 학습하므로 확률 보정 개선을 주장하지 않는다.' if t in SOURCES else '신규 neural fit=0. 기존 확률 예측 해시를 검증하고 CPU 정책/의존 구조만 비교한다. E는 REANALYSIS_REUSED_E다.')+'\n\n## ④ 실제 실행과 미실행\n\n'
 if state!='COMPLETE':
  fits=read(out/'fits.json') if (out/'fits.json').exists() else [];text+=f'완료경로 {sum(f["status"]=="COMPLETE" for f in fits)} / 상한 {spec["main_fit_cap"]}; 기록된 본학습 업데이트 {sum(f["updates"] for f in fits)}. 상태 사유: `{st.get("error",st.get("reason","아직 실행 순서에 도달하지 않음"))}`. 미측정은 성능실패나0점이 아니다.\n\n## ⑤ 원점수\n\n평가 완료를 주장하지 않는다. 보존된 중간 파일은 학습 완료의 대체 근거가 아니다.\n\n## ⑥ 단순 대안 / ⑦ 후속 근거\n\n직접 대비가 완결되지 않아 NOT_COMPARABLE. 후속 자동실행 없음.\n';(out/'REPORT.md').write_text(text);return
 verify=read(out/'verification.json');fits=read(out/'fits.json') if t in SOURCES else [];resource=pd.read_csv(out/'resources.csv');text+=f'완료 본학습 {st.get("neural_fits",0)}경로, 본학습 {st.get("main_updates",0)}updates. '+(f'폐기 smoke {sum(r["updates"] for r in read(out/"smoke.json"))}updates. 선택·복원·원점수 검산 완료. ' if t in SOURCES else f'CPU 작업: {st.get("CPU_policy_grid_evaluations",st.get("CPU_dependence_fits",0))}건. ')+'[자원](resources.csv), [검산](verification.json), [선택](selections.json). 큰 prediction/weight는 로컬 ignored cache에 있고 GitHub에는 해시와 수치가 있다.\n\n'
 if fits:text+=f'optimizer 실측 합계 {sum(f["optimizer_seconds"] for f in fits)/60:.2f}분; 최대 allocated {max(f["peak_allocated_bytes"] for f in fits)/2**20:.1f}MiB. INIT 선택 {st.get("selected_INIT",0)}/{st.get("repeated_models",0)}. 학습 예산 미사용은 alias 또는 차단으로 구분한다.\n\n'
 text+='## ⑤ 원점수·효과·seed·조건 손익\n\n'
 if t in SOURCES:
  df=pd.read_csv(out/'scores_summary.csv');text+=table(df[(df.policy=='selected')&(df.condition=='PRIMARY')],['arm','seed','score'],80)+'\n\n';c=pd.read_csv(out/'contrasts.csv');main=c[(c.policy=='selected')&(c.seed.astype(str)=='MEAN')&(c.condition.isin(['PRIMARY','REVISION']))];text+=table(main,['method','baseline','condition','gain_percent','ci_low','ci_high'],40)+'\n\n';text+='[전체 원단위 RMSE/MAE](raw_scores.csv), [모든 반복·조건](scores_summary.csv), [512고정점 포함 대비](contrasts.csv), [부가지표](secondary_scores.csv). RMSE는 원점·horizon 제곱오차 평균 후 제곱근, 채널과 지정 조건을 동일 가중한다.\n\n'
  if t=='R04':text+='[정확도–수정 frontier](accuracy_revision_frontier.csv)의 E_accuracy_protected를 확인한다. 수정량 감소만으로 목표 달성이 아니다.\n\n'
 else:
  df=pd.read_csv(out/'raw_scores.csv');metric='normalized_2pinball' if t=='R05' else 'sum_CRPS_normalized';g=df[(df.metric==metric)&(df.variant=='RAW')];text+=table(g[['target','seed','arm','condition','score']],limit=100)+'\n\n[모든 원점·지표](scores_by_origin.csv), [주변분포 또는 조건 포함 원점수](raw_scores.csv), [직접 대비와 CI](contrasts.csv).\n\n'
 plot_track(t);text+='![직접대비](paired_gains.png)\n\n![조건별 손익](condition_tradeoffs.png)\n\n## ⑥ 단순 대안과 남은 정식 비교\n\n'+('N06의 coupling은 주변분포를 완전히 동일하게 유지한다. 합계 위험 개선은 시점별 분포 개선이나 새 PEFT의 증거가 아니다. AR1로 충분한지 SHRUNK/EMPIRICAL의 추가 효과를 별도로 읽는다.' if t=='N06' else '직접 단순 대조의 충분성과 제안 구성요소의 추가 가치는 contrasts.csv에서 별도로 판단한다. 0근처 CI는 동등성 입증이 아니다. 알려진 단순 방법의 개선을 새 방법론 PASS로 바꾸지 않는다.')+'\n\n## ⑦ 다음 방법을 정의할 근거\n\n현재 증거 상태: '+st['EVIDENCE']+'. 단일 개발 원천과 제한된 recipe의 결과이며 정식 선행 대비·독립 확증이 남는다. 연구프로그램의 불가능성 판정은 아니다. 최종 문제 선택은 전체 MASTER_REPORT/FINAL_DECISION에서 최대2개로 제한한다. 자동 후속 학습 없음.\n'
 if t=='R09':text+='\nIMPUTE의 목적은 계수1 NULL과 동일하며 이번 계수.1 NULL과의 차이는 보존 강도 차이일 수 있다. I0는16개상세512반복, 나머지는64개×8회라 정보량까지 다르다.\n'
 if t=='R05':text+='\nE4에서 alpha0=0이면 S0는 E0와 동일하다. 중간 alpha는 두 모델 추론이며 단일 PEFT 비용이 아니다. latest 모델도 이전4case평균 V로 선택됐다는 한계가 있다.\n'
 (out/'REPORT.md').write_text(text)

def report_all():
 rows=[];ledger=[]
 for t in ORDER:
  st=read(OUT/t/'STATUS.json');fits=read(OUT/t/'fits.json') if (OUT/t/'fits.json').exists() else [];smoke=read(OUT/t/'smoke.json') if (OUT/t/'smoke.json').exists() else [];rows.append(dict(ID=t,주제=NAMES[t],실행=st['EXECUTION'],근거=st['EVIDENCE'],신규성=st['NOVELTY'],완료경로=sum(f['status']=='COMPLETE' for f in fits),업데이트=sum(f['updates'] for f in fits),smoke=sum(f['updates'] for f in smoke)))
  ledger.append(dict(track=t,fit_cap=4*len(ARMS[t]) if t in SOURCES else 0,attempted=len(fits),completed=sum(f['status']=='COMPLETE' for f in fits),main_updates=sum(f['updates'] for f in fits),smoke_updates=sum(f['updates'] for f in smoke),optimizer_seconds=sum(f.get('optimizer_seconds',0) for f in fits)))
 csvwrite(OUT/'BUDGET_LEDGER.csv',ledger);state=read(OUT/'controller_state.json') if (OUT/'controller_state.json').exists() else {};done=all(r['실행']!='PREPARED' for r in rows)
 text='# 아홉 조건 PEFT 비교 — 전체 결과\n\n단일 MASTER_PROTOCOL 계약으로 N01 → N02 → R05 → N06 → N03 → R04 → N07 → R08 → R09 순서로 실행한다. 서로 다른 목적을 합산한 순위는 만들지 않는다.\n\n'+table(pd.DataFrame(rows),limit=20)+'\n\n'+f'현재 controller 상태: `{state.get("status","PREPARED")}`. 새 본학습 업데이트 {state.get("main_updates",0)}/59,392, smoke {state.get("smoke_updates",0)}/96, native forwards {sum(state.get("forwards",{}).values())}/200,000. [세부 예산](BUDGET_LEDGER.csv).\n\n'
 for t in ORDER:text+=f'- [{t} {NAMES[t]} 한국어 보고서]({t}/REPORT.md)\n'
 text+='\n[원천·노출 장부](SOURCE_AND_EXPOSURE_LEDGER.md). 학습 실험의 E는 기존 프로그램에서 노출된 개발 평가이고 R05/N06는 기존 평가 점수의 재분석이다. 기존 결과는 덮어쓰지 않는다. 전체 적응 성공과 신규성은 별도 축이다. 단순방법으로 충분하면 새학습법을 만들어 살리지 않는다.\n\n미실행 부분은 QUEUE_STATUS와 후보 STATUS에 명시한다. 공통 안전/예산 중단 시 `scripts/with_cuda.sh .venv/bin/python scripts/run_condition_studies.py resume-all`을 쓰며 해시와 완전한epoch상태가 일치해야 한다. partial epoch/모호한update는 자동 재실행하지 않는다.\n'
 (OUT/'MASTER_REPORT.md').write_text(text)
 if not (OUT/'FINAL_DECISION.md').exists():(OUT/'FINAL_DECISION.md').write_text('# 최종 판단 대기\n\n현재 후속 확정 문제 **0개**. 지정된 아홉 비교가 완료되어 근거가 검토되기 전에는 연구 주제를 확정하지 않는다. 이는 미실행 후보의 성능실패 판정이 아니다. 새후보·데이터·추가학습 자동실행 없음.\n')
