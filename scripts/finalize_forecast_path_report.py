"""Add reviewed interpretation and audit provenance without changing numerical results."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from experiments.forecast_path_structure_v1_20260916.common import *
def main():
 p=OUT/'REPORT.md';base=p.read_text().split('\n## 최종 대조 해석과 완료 감사')[0]
 eff=pd.read_csv(OUT/'contrasts.csv');monthly=pd.read_csv(OUT/'monthly_contrasts.csv');audit=read(OUT/'completion_audit.json');replay=read(OUT/'replay.json');smoke=read(OUT/'smoke.json')
 lines=['','## 최종 대조 해석과 완료 감사','','**이번 결과는 모든 방법의 실패가 아니다. PATH의 DROP 대비 이득은 관측됐지만, 핵심인 시간 연결의 추가 가치는 타깃 전반에서 확보되지 않았다.**', '',
 'PATH 대 DROP의 raw 평균 이득은1.826%(기술적95%구간0.508~3.034), 동일 보정 뒤1.657%(0.206~2.934)다. 반면 동일 시점별 값 노출을 맞춘 POINT 대비 raw 이득은0.033%(−1.001~0.955), 보정 뒤0.668%(−0.111~1.463)로 불확실하다. 보정 후 이득이 사라진 사례라고 설명할 수는 없으며, 이번에는 보정이 상대 차이를 키웠지만 전체 근거가 충분해지지 않았다.', '',
 'T0와T1은 PATH가 POINT보다 좋았고, 보정 뒤 두 타깃의 기술적 구간도 양수였다. T2는 두 seed 모두 방향이 반대였다. 따라서 T2를 제외하거나 T0/T1만 사후 주 결과로 삼지 않는다. 추가 타깃 둘만 모으면 raw −0.472%, 보정 뒤 −0.008%다. 이 반전을 설명하는 원인은 이번 실험으로 확정하지 못했다.', '',
 '| 범위 | PATH 대 POINT raw 개선% [95%] | 동일 보정 개선% [95%] |', '|---|---:|---:|']
 for scope in ['T0','T1','T2','ADDITIONAL','ALL']:
  sub=eff[(eff.scope==scope)&(eff.policy=='SELECTED')&(eff.case=='MEAN')&(eff.method=='PATH')&(eff.baseline=='POINT')]
  def val(variant):
   r=sub[sub.variant==variant].iloc[0];return f'{r.gain_percent:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]'
  lines.append(f'| {scope} | {val("RAW")} | {val("CALIBRATED")} |')
 lines += ['',
 'T0에서 PATH는 LATEST보다 최신S0에 두 seed 모두 불리했고 이전S3에는 두 seed 모두 유리했다. T1의 LATEST 대비 이득은 주로 이전 입력에서 커졌고, T2는 최신·이전 양쪽에서 악화했다. 네 case 평균이 모든 입력 상태의 개선을 뜻하지 않는다. T2의 동결 모델이 적응 모델들보다 좋았다는 사실도 포함해 전부 보존한다.', '',
 'FIXED512는 **선택된 LR를 그대로 두고 step512만 고정한 대조**다. 이 경우 PATH 대 POINT는 −0.265%(−1.560~0.987)로 역시 추가 가치를 뒷받침하지 못했다. 두 seed를 LR 선택에도 사용했으므로 새로운 tuning-free seed의 독립 반복이라고 부르지 않는다. 실제 선택은256-step11개/512-step13개이며 INIT 선택0개다.', '',
 '네 군은 동일 LoRA 구조와 같은 수의 학습 파라미터를 사용했다. 최대 allocated memory는 모두589.29MiB였고 512-step optimizer 평균시간은 약55초였다. 이번에 PATH만의 자원 절감은 관측되지 않았다. 시간 연결의 비교를 새 LoRA 행렬 구조나 메모리 혁신으로 해석하지 않는다.', '',
 '## 월별 차이와 불확실성', '',
 '아래는 전체 세 타깃에서 PATH 대 POINT의 월별 결과다. 기존2,000개의 공통 달력 resample을 그대로 재사용하고 각 월만 집계했다. 어떤 resample이 해당 월의 유효 날짜를 한 개도 포함하지 않으면 월별 통계는 정의되지 않는다. 그 개수를 명시했으며 관측 날짜·불리한 타깃·seed를 제거한 것이 아니다. 주 대비의2,000회 구간은 변경하지 않았다.', '',
 '| 월 | 원점/타깃 | raw 개선% [95%] | 동일 보정 개선% [95%] | 정의된 월별 resample |', '|---|---:|---:|---:|---:|']
 for month in [10,11,12]:
  sub=monthly[(monthly.month==month)&(monthly.scope=='ALL')&(monthly.policy=='SELECTED')&(monthly.case=='MEAN')&(monthly.baseline=='POINT')]
  vals=[]
  for variant in ['RAW','CALIBRATED']:
   r=sub[sub.variant==variant].iloc[0];vals.append(f'{r.gain_percent:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]')
  lines.append(f'| {month} | {r.origins_per_target} | {vals[0]} | {vals[1]} | {r.defined_month_resamples}/2000 |')
 lines += ['', '[월별 타깃·입력 상태별 전체 대비](monthly_contrasts.csv), [월별 불확실성 정의](monthly_uncertainty_manifest.json). 12월 raw에서는 방향이 반대였으며, 모든 월의 전체 타깃 구간이0을 포함했다. 월별 가장 좋은 값으로 주 결과를 교체하지 않았다.', '',
 '## 검증 범위·오류 이력·재현', '',
 f'완전한 Adam/RNG/stream 포함 재개 상태48개와 최종 checkpoint의 파라미터가 모두 exact 일치했다. TRAIN 통계의 실제 dtype 계산도 독립 재현했고 전체 대비630개의 집계 개선율 오차는 최대{audit["max_contrast_gain_error"]:.3g}%p였다. smoke 복원12개와 선택 모델 복원24개 모두 bitwise 일치했다. [완료 감사](completion_audit.json).', '',
 '각 타깃의 입력76개에서2024-10-26 정답 결측1개를 모든 군·case에 공통 제외해75개를 채점했다. [정답 결측 manifest](test_label_manifest.json), [달력 경계 제외](calendar_boundary_exclusions.csv), [구체적 데이터 정의](data_implementation_notes.md). 원자료의 과거 접근을 포함한 노출 이력은 exposure_ledger에 구분했다. T0는 이미 개발에 사용한 대상의 후속 시간 구간 반복이며, T1/T2도 같은 데이터셋의 타깃이다. 자료 자체가 새 독립 데이터이거나 사전학습 비중복임을 주장하지 않는다.', '',
 '원시 phase counter는 일부 smoke 추론33회를 train 태그로 분류했다. 실제 호출 구성은 본학습24,576+smoke학습24+저장예측19,892+검증72=44,564 native forwards다. 24,600개의 optimizer update와 혼동하지 않는다. 추가 감사의 float32/float64 가정 오류, 미사용 재개 경계 보완 및 실행 소스 commit은 [감사·유지보수 기록](AUDIT_CORRECTIONS.md)에 모두 남겼다. 허용오차·데이터·학습률·평가 기준은 변경하지 않았고 새 학습을 실행하지 않았다.', '',
 '최종 추천은 **추가 근거 미확보** 하나다. PATH/DROP의 실용 대조 신호와 특정 타깃의 양성 결과는 보존하지만, 이를 근거로 시간 연결 특화 PEFT 구조를 바로 추가할 만큼 일관된 근거는 얻지 못했다. 새 구조·다른 데이터셋·후속 학습은 시작하지 않는다.', '']
 p.write_text(base+'\n'.join(lines))
 (OUT/'FINAL_DECISION.md').write_text('''# 최종 결정 — 추가 근거 미확보

**실행은 완료됐지만, 시간적 연결에 특화된 다음 PEFT 방법을 정의할 충분한 근거는 확보하지 못했다.** 새 구조·다른 데이터셋·후속 학습은 시작하지 않는다.

- 실행: 본학습48/48fits·24,576updates, smoke24updates, 봉인 평가와 독립 검산 완료. 미실행 계약 내 경로0.
- 예측 근거: PATH는 DROP 대비 raw1.826%, 동일 보정 뒤1.657% 개선했고 기술적 구간 하한도 양수였다. 모든 방법이 실패했다고 묶지 않는다.
- 시간 연결의 추가 가치: 동일 시점별 값 노출 POINT 대비 raw0.033%[−1.001,0.955], 동일 보정 뒤0.668%[−0.111,1.463]로 전체 근거는 불확실하다. T0/T1의 양성 결과와 T2의 두 seed 모두 반대 방향을 함께 보존한다. 추가 타깃 둘만의 보정 뒤 이득은−0.008%다.
- 선택·입력 상태: FIXED512에서도 POINT 대비−0.265%로 우위가 확인되지 않았다. T0에서는 최신S0 손해와 이전S3 이득이 함께 있었다. 좋은 타깃·seed·case만 고르지 않는다.
- 자원: 같은 LoRA 구조·147,456파라미터·같은 예산이며 네 군 최대 allocated memory589.29MiB로 같다. PATH 고유의 자원 이득은 없다.
- 신규성: PATH는 알려진 학습 규칙이고 POINT는 합성 통제다. NEW_PEFT_METHOD_ESTABLISHED를 부여하지 않는다. 실제 발행 로그, 독립 도메인, 사전학습 corpus 비중복도 확보하지 못했다.

불확실한 차이는 동등성이나 효과 부재의 증명이 아니다. 전체 원점수·월별·seed별 결과와 감사 범위는 [REPORT.md](REPORT.md)에 남겼다. 기존 구현·원시 예측·가중치·부정 결과를 보존하고 자동 연구 확장은 종료한다.
''')
if __name__=='__main__':main()
