"""Korean final artifacts, keeping execution, evidence and novelty separate."""
import pandas as pd
from .common import *
def report():
 status=read(OUT/'status.json');ds=read(OUT/'data_seal.json');fits=read(OUT/'fits.json') if (OUT/'fits.json').exists() else []
 if not (OUT/'independent_verification.json').exists():
  (OUT/'REPORT.md').write_text(f'# 예보 경로 연결 LoRA — 부분 실행\n\n실행 상태 `{status["status"]}`. 성능 실패로 바꾸어 기록하지 않는다. 완료 {sum(f["status"]=="COMPLETE" for f in fits)}/{ds["planned_fits"]} fits, 본학습 업데이트 {status["main_updates"]}, smoke {status["smoke_updates"]}.\n\n오류 원기록은 [status.json](status.json), 완료·미완료 경로는 [fit_manifest.csv](fit_manifest.csv), 실행 계약은 [PROTOCOL.md](PROTOCOL.md)에 보존했다. 봉인 평가·독립 검산이 완료되지 않았으므로 원점수에 근거한 최종 비교는 미실행이다. 허용오차나 레시피를 바꾸어 자동 재실행하지 않는다.\n\nPATH는 알려진 학습 규칙, POINT는 통제용 합성 입력이다. SIMULATED_ASOF이며 실제 발행 로그의 재현이나 새 PEFT 방법 성립을 주장하지 않는다.\n')
  (OUT/'FINAL_DECISION.md').write_text('# 최종 결정\n\n추가근거미확보. 실행 미완료와 성능 실패를 구분한다. 완료 결과·중단 위치를 보존하며 새 구조·다른 데이터셋·후속 학습은 시작하지 않는다.\n');return
 agg=pd.read_csv(OUT/'scores_by_target_seed_case.csv');eff=pd.read_csv(OUT/'contrasts.csv');se=pd.read_csv(OUT/'contrasts_by_seed.csv');ver=read(OUT/'independent_verification.json');res=pd.read_csv(OUT/'resource_usage.csv');valid=read(OUT/'scoring_status.json');wall=read(OUT/'controller_wall.json');sel=read(OUT/'selection_seal.json')['selections'];intensity=pd.read_csv(OUT/'train_intervention.csv');scales=read(OUT/'train_scaling.json');cal=read(OUT/'calibration_parameters.json');tele=[json.loads(l) for l in (OUT/'gpu_controller.jsonl').read_text().splitlines()]
 main=eff[(eff.scope=='ALL')&(eff.policy=='SELECTED')&(eff.case=='MEAN')]
 component=main[(main.method=='PATH')&main.baseline.isin(['POINT','DROP'])]
 support=len(component)==4 and bool((component.ci_low>0).all())
 decision='이 조건에서 다음 방법을 정의할 근거 있음' if support else '추가 근거 미확보'
 lines=['# 예보 입력의 시간적 연결을 보존하는 LoRA — 최종 보고', '',f'**실행: EXECUTION_COMPLETE. 최종 추천: {decision}. 신규 PEFT 방법 성립: 부여하지 않음.**', '',
 '문제는 다음 날 부하를 예측할 때 입력되는 미래 날씨가 정답이 아니라 갱신되는 예보라는 점이다. 기존 한 타깃의 작은 개발 비교에서 과거 버전 증강 신호가 있었지만, epoch 간 순서를 바꾼 대조는 24시간 경로 안의 연결 가치를 검사하지 않았다. 이번에는 추가 타깃 둘, 새 TRAIN64와 뒤 달력 구간, 같은 시점별 예보값 노출을 맞춘 PATH/POINT를 직접 비교했다.', '',
 '## 정보·분할 계약', '',
 'OpenSTEF Liander2024 고정 revision의 부하와 versioned forecast만 사용했다. **SIMULATED_ASOF**: 실제 공개시각·물리 예보 issue/run을 복원한 것이 아니다. k는 시점별 가용 버전 순위다. S0도 정답 날씨가 아니며 S1–S3은 미래 날씨 입력만 이전 순위로 바꾼 통제 조건이다. POINT는 시간별로 서로 다른 순위의 값을 연결한 합성 통제 입력이며 배포 후보가 아니다.', '',
 'TRAIN 3–6월 64원점, V_SELECT 7월 16원점, V_CALIBRATE 8월 16원점, 9월은 이번 실행의 학습·선택·보정·채점에서 제외했다. TEST는 10–12월의 가용 입력과 유효 정답을 갖춘 모든 원점을 공통 사용했다. UTC08시, context336h/horizon24h이며 15분 W 값 네 개의 평균이지 에너지 합계가 아니다. 원점 끝의 24시간이 역할 종료 경계 밖인 날은 경계 규칙으로 제외했다. 결측 보간이나 성능에 따른 날짜 선택은 없었다.', '',
 'TRAIN 외부 통계는 중복을 제거한 과거 기상 선택 행만, 부하 sigma는 TRAIN 입력·정답의 unique hourly timestamps만 사용했다. 모델 부하 입력의 native normalization은 유지했다. TEST 값은 선택·보정에 사용하지 않았고 모든 예측 해시 후 scorer가 정답을 추출했다. [노출 이력](exposure_ledger.md), [데이터 수령](data_receipt.json), [원점 manifest](origin_manifest.csv), [as-of 선택 행](selected_weather_rows.json).', '',
 '| 타깃 | 이름 | TRAIN/V/CAL | 유효 TEST | TRAIN sigma |', '|---|---|---:|---:|---:|']
 for t in ds['targets']:lines.append(f'| {t["target"]} | {t["name"]} | 64/16/16 | {valid["valid_by_target"][t["target"]]} | {scales[t["target"]]["sigma_y"]:.6g} |')
 lines += ['', '추가 타깃은 NFC(group,name)의 SHA256 순서로 TRAIN 조건을 만족한 첫 두 개다. 같은 배전망의 세 타깃이며 독립 도메인 세 개가 아니다. T0의 과거 개발 노출과 이번 후속 시간 구간을 구별하고 Chronos-2 사전학습 corpus 중복 부재는 입증하지 못했다. [선정 순서](candidate_order.json), [선정 결과](target_manifest.csv).', '',
 '## 실제 실행과 검증', '',f'최대 {ds["planned_fits"]} 경로 중 **{len(fits)} fits ×512 updates={ver["main_updates"]:,}**를 완료했다. 기존 학습 경로 재사용 0, smoke {ver["smoke_updates"]} updates, 별도 resource optimizer0, 총 신규 optimizer {ver["new_optimizer_updates"]:,} updates다. 세 타깃×네 군×두 LR×두 seed를 전부 실행했으며 불리한 결과로 중단하거나 교체한 경로는 없다. 미실행 범위: 추가 타깃·추가 seed·다른 구조·후속 학습은 계약 밖이므로 실행하지 않았다.', '',
 '공통 Chronos-2 고정 revision, 96 projection rank1/alpha2 LoRA 147,456개 파라미터(192텐서), FP32·TF32 off·dropout0·native quantile loss다. 모든 군은 같은 초기값·원점 순서·정답 노출8회와 같은 두 LR을 받았다. 체크포인트는 0/256/512만 선택하고 LR는 타깃/군별 두 seed 검증 최소값 평균으로 고정했다. 선택에서 INIT가 이겼다면 무적응 선택 결과로 그대로 남겼다.', '',
 f'독립 scalar 검산 {ver["scalar_metrics_checked"]:,}개, V checkpoint {ver["validation_checkpoint_checks"]}개, 선택 {ver["selection_checks"]}개, 보정 scalar {ver["calibration_scalar_checks"]}개를 고정 rtol/atol1e-10으로 통과했다. 기존 파일 {ver["historical_files_preserved"]:,}개 hash를 보존했다. 모든 frozen 가중치·buffer 불변, 실제 LoRA update 및 fresh-model 복원을 검사했다. [검산 원기록](independent_verification.json), [smoke](smoke.json), [복원](replay.json), [학습 경로](fit_manifest.csv), [선택 봉인](selection_seal.json), [평가 봉인](evaluation_seal.json).', '',
 f'GPU controller 벽시계 {wall["seconds"]/60:.2f}분(안전 대기 {wall["wait_seconds"]:.2f}초 포함), 실제 optimizer 합계 {res.optimizer_seconds.sum()/60:.2f}분. 최저 GPU 여유 {wall["minimum_free_mib"]}MiB. 승인된 RustDesk를 제외한 외부 compute 표본 {sum(any(not a["own"] and not a.get("allowed_desktop",False) for a in r["apps"]) for r in tele)}개, 업데이트 오염 {ver["contaminated_updates"]}개. GUI 부하는 제거하지 않았으므로 시간 측정은 이 호스트 조건의 값이다. 네 입력 상태의 평가 호출을 단일 배포 추론의 4배 ensemble로 해석하지 않는다.', '',
 '| 군 | 512-step optimizer 평균초 | peak allocated 최대 MiB |', '|---|---:|---:|']
 for arm in ARMS:
  rr=res[res.arm==arm];lines.append(f'| {arm} | {rr.optimizer_seconds.mean():.3f} | {rr.peak_allocated_bytes.max()/2**20:.2f} |')
 lines += ['',f'선택된 step512 경로는 {sum(s["selected"]["step"]==512 for s in sel)}/{len(sel)}로 BUDGET_LIMITED 가능성을 명시한다. 유한 예산에서의 비교이며 충분한 수렴이나 더 오래 학습해도 개선 불가능함을 입증하지 않는다. [자원](resource_usage.csv), [학습 곡선](train_curves.csv).', '',
 '## 원점수: 네 입력 상태 동일가중 primary', '',
 '각 원점의 21분위수×24시간 mean 2-pinball을 TRAIN sigma로 나눈 뒤 날짜→seed→타깃 동일가중으로 집계했다. 아래는 seed 평균이며 작은 값이 좋다. raw가 기본 primary이고 동일 보정 후 결과도 반드시 함께 남긴다. raw RMSE/MAE, 80%포함률·폭, 각 입력 상태와 각 seed 원점수는 링크 표에 모두 있다. 규모가 다른 타깃의 raw 오차를 그대로 합치지 않는다.', '',
 '| 타깃 | 군 | raw | 동일 보정 | FIXED512 raw |', '|---|---|---:|---:|---:|']
 for t in ds['targets']:
  for arm in ARMS+['FROZEN_WEATHER','FROZEN_HISTORY']:
   ss=agg[(agg.target==t['target'])&(agg.arm==arm)];raw=ss[(ss.policy=='SELECTED')&(ss.variant=='RAW')].primary.mean();ca=ss[(ss.policy=='SELECTED')&(ss.variant=='CALIBRATED')].primary.mean();fi=ss[ss.policy=='FIXED512'].primary.mean();fix='—' if pd.isna(fi) else f'{fi:.6f}'
   lines.append(f'| {t["target"]} | {arm} | {raw:.6f} | {ca:.6f} | {fix} |')
 lines += ['', '[타깃·seed·case 전체 원점수](scores_by_target_seed_case.csv), [raw/동일 보정](raw_and_calibrated.csv), [원점별](scores_by_origin.csv), [월별](monthly_scores.csv), [lead별](per_lead_hour.csv), [LAST_DAY 점예측](last_day_scores.csv). FROZEN_HISTORY는 부하만 받는 참고선이고 1변수 대4변수 forward 차이를 새 구조 이득으로 부르지 않는다.', '',
 '## 경로 연결, dropout, 보정의 추가 가치', '',f'실제 TRAIN 타깃·seed·origin·4epoch 블록 {len(intensity)}개에서 PATH/POINT multiset exact 검사를 통과했고, 그 중 {int(intensity.distinct.sum())}개는 입력 배열이 달랐다. 각 시간의 세 기상변수는 같은 rank에서 가져왔다. 한 경로 안의 시간 연결을 바꾸면서 연속성·국소 smoothness·비현실성도 함께 바뀌므로 한 가지 물리 인과원인을 증명하지 않는다. [개입 강도](train_intervention.csv), [CPU 검사](cpu_checks.json), [증강 봉인](augmentation_schedule.json).', '',
 '개선율=100×(대조 점수−PATH 점수)/대조 점수다. 기술적95%구간은 같은 7일 날짜 블록을 군·seed·case·동시기 타깃에 공통 적용한 2,000회 bootstrap이다. 관측한 세 타깃·두 seed에 조건부이며 모집단 불확실성·다중검정 보정이 아니다.', '',
 '| 정책 | 보정 | 대조 | PATH 개선율% | 95% 기술적 구간 |', '|---|---|---|---:|---:|']
 for r in eff[(eff.scope=='ALL')&(eff.case=='MEAN')&(eff.method=='PATH')].itertuples():lines.append(f'| {r.policy} | {r.variant} | {r.baseline} | {r.gain_percent:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] |')
 lines += ['', 'PATH 대 LATEST는 과거 버전 정보와 다양화가 함께 바뀐다. PATH 대 POINT만 시점별 정보 노출을 맞춘 연결 대조다. PATH 대 DROP은 같은 정보량 대조가 아니라 알려진 단순 학습 규칙과의 실용 비교다. 21상수 보정은 모든 선택 모델에 똑같이 주었으며 case·시간별 보정기나 CAL 기반 checkpoint 선택은 없었다. FIXED512는 고정 예산 진단이지 사후 주 정책 교체가 아니다.', '',
 '## 불리한 타깃·seed·입력 상태 포함', '', '| 타깃 | seed | PATH 대 POINT 평균 개선% | PATH 대 DROP 평균 개선% | PATH 대 LATEST S0 개선% | PATH 대 LATEST S3 개선% |', '|---|---:|---:|---:|---:|---:|']
 for t in ds['targets']:
  for seed in SEEDS:
   sub=se[(se.target==t['target'])&(se.seed==seed)&(se.policy=='SELECTED')&(se.variant=='RAW')&(se.method=='PATH')]
   def gain(b,c):return float(sub[(sub.baseline==b)&(sub.case==c)].gain_percent.iloc[0])
   lines.append(f'| {t["target"]} | {seed} | {gain("POINT","MEAN"):+.3f} | {gain("DROP","MEAN"):+.3f} | {gain("LATEST","S0"):+.3f} | {gain("LATEST","S3"):+.3f} |')
 lines += ['', '최신/이전 상태의 손익은 위 표와 전체 S0–S3 표로 분리했다. worst-scenario는 타깃별 case 평균 중 최대이며 per-origin maximum이 아니다. 모든 타깃과 추가 타깃 둘만의 구간도 [전체 대비](contrasts.csv)에 남겼다. [seed별 모든 효과](contrasts_by_seed.csv), [불확실성](uncertainty.csv).', '',
 '## 신규성과 최종 해석', '',
 'PATH는 알려진 예보 버전 증강 학습 규칙이고 POINT는 정보 통제다. LoRA나 공변량 주입 자체도 새 제안이 아니다. 좋은 결과가 나와도 이 실행에서는 NEW_PEFT_METHOD_ESTABLISHED를 부여하지 않는다. 이번 계약의 차이는 시점별 값 노출을 맞춘 24시간 연결 검사이며, 새 신경망 모듈의 필요성·효율성까지 입증하지 않는다. 실제 발행 로그·독립 도메인·사전학습 비중복을 확보하지 못했다. [정확히 읽은 선행 범위](literature_boundary.md).', '',
 f'최종 추천은 **{decision}** 하나다. raw와 동일보정에서 PATH 대 POINT/DROP의 방향·불확실성을 함께 보며, 평균이 양수라는 이유만으로 성공을 강제하지 않는다. 구간이 0을 포함하는 경우 동등성이나 효과 부재가 입증됐다고도 쓰지 않는다. [FINAL_DECISION.md](FINAL_DECISION.md).', '',
 '코드·작은 원점수·검증 manifest는 GitHub에 보존한다. 원자료·가중치·원시 예측·augmentation npz는 ignored 로컬 cache이며 GitHub만으로 수치 재현에 필요한 모든 파일이 포함된다고 주장하지 않는다. 새 구조·다른 데이터셋·추가 학습은 자동 실행하지 않는다.', '',
 '![타깃별 네 입력 상태](figures/case_scores.png)', '', '![PATH 대비 효과](figures/paired_gains.png)', '', '![raw와 동일보정](figures/calibration.png)', '']
 (OUT/'REPORT.md').write_text('\n'.join(lines))
 (OUT/'FINAL_DECISION.md').write_text(f'# 최종 결정\n\n**{decision}.**\n\n실행·평가·독립 검산은 완료했다. 예측 이득, 같은 값 노출에서의 시간적 연결 추가 가치, 신규성은 각각 [REPORT.md](REPORT.md)의 원점수·대비·한계로 판단한다. 단순 양성 평균을 PASS로 바꾸거나 구간이 넓은 결과를 효과 부재로 확정하지 않는다.\n\nPATH는 알려진 학습 규칙, POINT는 통제용 합성 경로다. 새 PEFT 구조는 제안·검증하지 않았으므로 NEW_PEFT_METHOD_ESTABLISHED를 부여하지 않는다. 모든 군·타깃·seed·입력 상태와 코드를 보존한다. 새 구조·다른 데이터셋·후속 학습은 시작하지 않는다.\n')
 plots(agg,eff)

def plots(agg,eff):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':9,'figure.dpi':130});folder=OUT/'figures';folder.mkdir(exist_ok=True);targets=sorted(agg.target.unique())
 fig,axes=plt.subplots(1,len(targets),figsize=(13,3.8),squeeze=False)
 for ax,t in zip(axes[0],targets):
  for arm in ARMS+['FROZEN_WEATHER']:
   s=agg[(agg.target==t)&(agg.arm==arm)&(agg.policy=='SELECTED')&(agg.variant=='RAW')].groupby('case').primary.mean();ax.plot(s.index,s.values,marker='o',label=arm)
  ax.set_title(t);ax.set_ylabel('Normalized mean 2-pinball');ax.grid(alpha=.2)
 axes[0][-1].legend(fontsize=7);fig.tight_layout();fig.savefig(folder/'case_scores.png');plt.close(fig)
 ss=eff[(eff.scope=='ALL')&(eff.policy=='SELECTED')&(eff.variant=='RAW')&(eff.case=='MEAN')&(eff.method=='PATH')];fig,ax=plt.subplots(figsize=(7,3.5));ax.errorbar(ss.gain_percent,np.arange(len(ss)),xerr=np.array([ss.gain_percent-ss.ci_low,ss.ci_high-ss.gain_percent]),fmt='o',capsize=4);ax.set_yticks(np.arange(len(ss)),['PATH vs '+b for b in ss.baseline]);ax.axvline(0,color='gray',ls='--');ax.set_xlabel('Gain %, paired 7-day block descriptive 95% interval');fig.tight_layout();fig.savefig(folder/'paired_gains.png');plt.close(fig)
 fig,axes=plt.subplots(1,len(targets),figsize=(13,3.8),squeeze=False)
 for ax,t in zip(axes[0],targets):
  for i,v in enumerate(['RAW','CALIBRATED']):
   vals=[agg[(agg.target==t)&(agg.arm==a)&(agg.policy=='SELECTED')&(agg.variant==v)].primary.mean() for a in ARMS];ax.bar(np.arange(4)+(i-.5)*.36,vals,width=.36,label=v)
  ax.set_xticks(range(4),ARMS);ax.set_title(t);ax.set_ylabel('Normalized mean 2-pinball');ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(folder/'calibration.png');plt.close(fig)
