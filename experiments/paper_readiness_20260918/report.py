"""Korean paper-preparation evidence report, without new method selection."""
import pandas as pd
from .common import *
from .figures import PAPER,LABELS

def table(f):
    def fmt(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v).replace('|','/')
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in r)+' |' for r in f.itertuples(index=False,name=None))

def report():
    check_seal();v=read(OUT/'VERIFICATION.json');assert v['status']=='VERIFIED';status=read(OUT/'status.json');e=pd.read_csv(OUT/'EFFECTS.csv');fac=pd.read_csv(OUT/'FACTORIAL.csv');raw=pd.read_csv(OUT/'RAW_SCORES.csv');seeds=pd.read_csv(OUT/'SEED_EFFECTS.csv')
    std=e[e.kind=='standard'];shift=std[std.condition=='SHIFT8'];f=fac[(fac.kind=='standard')&(fac.condition=='SHIFT8')]
    core=shift[(shift.new=='C3')&shift.baseline.isin(['C0','C2','M_RECENCY'])];base=std[std.new.isin(['C0','C3'])&std.baseline.isin(['F0','PERSISTENCE','SEASONAL'])&std.condition.isin(['REFERENCE','FAULT','SHIFT8'])]
    summary=raw[(raw.kind=='standard')&raw.condition.isin(['REFERENCE','FAULT','SHIFT8'])&raw.arm.isin(['F0','PERSISTENCE','SEASONAL','C0','C2','C3','M_RECENCY'])].groupby(['panel','arm','condition']).nmae.mean().unstack().reset_index()
    manifest=read(OUT/'PREDICTIONS.json');checks=read(OUT/'MODEL_CHECKS.json');logs=[json.loads(l) for l in open(OUT/'gpu_evaluation.jsonl')];unapproved=sum(any(not a['own'] and not a.get('allowed_desktop') for a in r['apps']) for r in logs);assert unapproved==0
    cost=dict(new_fits=0,optimizer_updates=0,logical_views=len(manifest),reused_views=sum(r['reused'] for r in manifest.values()),new_views=sum(not r['reused'] for r in manifest.values()),new_point_baseline_views=sum(r['arm'] in ['PERSISTENCE','SEASONAL'] for r in manifest.values()),new_GPU_views=sum(not r['reused'] and r['arm'] not in ['PERSISTENCE','SEASONAL'] for r in manifest.values()),minimum_free_gpu_mib=min(r['free_mib'] for r in logs),unapproved_compute_samples=unapproved,wall=read(OUT/'evaluation_wall.json'),recorded_seconds=sum(r['seconds'] for r in manifest.values()));save(OUT/'COST.json',cost)
    # Interpretations stay data-driven and include detrimental conditions.
    paragraphs=[]
    for panel in ['electricity','electricity_transfer','neso_2025']:
        r=f[(f.panel==panel)&(f.comparison=='C3_vs_C2_fixed1024')].iloc[0]
        paragraphs.append(f"{LABELS[panel]}의 고정 예산 SHIFT8에서 총 nMAE 이득은 {r.total:+.6f}, gate 항은 {r.gate:+.6f}, 가중치 항은 {r.weights:+.6f}다. gate 항의 조건부 95% 구간은 [{r.gate_low:+.6f}, {r.gate_high:+.6f}]다.")
    n=f[(f.panel=='neso_2025')&(f.comparison=='C3_vs_RECENCY_selected')].iloc[0]
    paragraphs.append(f"NESO의 selected C3/RECENCY 분해는 총 {n.total:+.6f}, gate {n.gate:+.6f}, 가중치 {n.weights:+.6f}다. 이는 이미 본 NESO 평가의 후속 진단이다.")
    point=base[(base.new=='C3')&(base.condition=='SHIFT8')]
    baseline_sentences=[]
    for panel in PANELS:
        z=point[point.panel==panel].set_index('baseline')
        baseline_sentences.append(f"{LABELS[panel]}: C3의 F0 대비 {z.loc['F0','gain_pct']:+.3f}%, 마지막 값 유지 대비 {z.loc['PERSISTENCE','gain_pct']:+.3f}%, 계절 반복 대비 {z.loc['SEASONAL','gain_pct']:+.3f}%.")
    text=f'''# 논문 준비용 보강 실험과 시각화 결과

이번 작업은 원고 작성만 진행한 것이 아니라, 기존 발견을 설명하는 데 빠졌던 직접 비교를 실행하고 논문용 자료를 만든 작업이다. **{len(manifest)}개 논리적 비교를 완료했다: 기존 정확 재사용 {cost['reused_views']}개, 새 평가 {cost['new_views']}개(GPU {cost['new_GPU_views']}개·CPU 단순 예측 {cost['new_point_baseline_views']}개). 새 학습과 optimizer update는 0회다.** 기존 가중치로 필요한 비교를 구성할 수 있어 중복 학습하지 않았다.

## 1. 채운 근거와 그대로 남은 한계

기존 selected 결과만으로는 C2/C3의 checkpoint 선택 효과와 추론 gate 직접 효과를 구분하기 어려웠다. 이번에는 Electricity source의 두 방법 모두 LR=0.0003, 1,024 updates인 저장 가중치를 사용해 학습 규칙 × 추론 규칙의 2×2 네 구성을 평가했다. 초기값·B0·학습 기회·학습 순서가 일치하는 실제 receipt를 재사용했다. ETTm1의 C2/C3는 선택 LR가 달라 이 matched 비교를 억지로 동일 조건이라고 부르지 않았다. ETTm1/2의 기존 불리한 결과는 전체 표와 그림에 계속 포함한다.

동시에 F0와 단순 점예측을 더해 B0의 출발 강도를 확인했다. NESO에는 현지 B0 학습을 추가하지 않았으며 Electricity에서 학습한 모델의 원천 전이다. F0와 단순 예측은 각각 deterministic 한 번이며 세 seed로 부풀리지 않았다.

정식 COSA/TATO/SOLID/Time-PEFT 전체 재현은 이번에도 하지 않았다. 실제 비교한 C1은 기존 LoRA의 추가 학습이고 C2는 동일 크기의 일반 residual adapter다. 현재 근거로 모든 PEFT 선행보다 우수하다는 주장은 불가하다. 이 미완료 범위와 비교에 필요한 정보 권한 차이는 [선행 대조](../../papers/persistence_adaptation/RELATED_WORK_AND_SCOPE.md)에 적었다.

## 2. 기존 주 결과를 유지한 종합 표

다음 표는 기존 selected 주 결과다. 이번 실험에 유리한 fixed1024 값으로 교체하지 않는다. 구간은 이번 명세의 2,000회 paired 7일 block bootstrap이며 세 모델 seed와 관측기간에 조건부다. 탐색적 진단의 구간을 다중 검정 보정된 확증 결과로 부르지 않는다.

{table(core[['panel','baseline','new_nmae','baseline_nmae','gain_pct','ci_low_pct','ci_high_pct']])}

## 3. 일반 adapter 대비 학습·추론 분해

A=C3 학습/C3 추론 gate, B=C3 학습/all-one gate, C=C2 학습/C3 gate, D=C2 학습/all-one gate다. gate=((B−A)+(D−C))/2, weights=((C−A)+(D−B))/2이며 합은 D−A다. 단위는 절대 nMAE이며 양수가 C3 쪽에 유리하다. 직접 개입으로 분해한 것은 저장된 두 함수의 오차다. 어떤 gradient나 표현이 학습 이득을 만들었는지의 보편적 인과 기전을 규명한 것은 아니다.

{chr(10).join(paragraphs)}

{table(f[['panel','comparison','total','gate','weights','interaction','gate_low','gate_high']])}

이 결과는 기존 C3/RECENCY 비교를 C3/C2 비교로 대체하지 않는다. 두 대조가 묻는 질문이 다르다. 4구성 중 평가 점수가 더 좋다는 이유로 교차 구성을 새 방법으로 선택하지 않았다. 전 조건의 원점수와 seed별 효과는 CSV에 남겼다.

### 3.1 추론 mask 자체가 구별되는가

교차 평가에서 NESO의 C3/RECENCY gate 항이 0인 이유를 확인하기 위해, 채점 이후 입력만 사용하는 CPU 감사를 추가했다. 새 예측이나 업데이트는 없고 모든 95개 panel/condition을 포함했다. NESO standard SHIFT8의 128원점×2draw=256입력 모두에서 C3와 RECENCY mask가 정확히 같았다. REFERENCE도 두 mask가 같고 모두 all-one이었다. 따라서 이 조건의 selected C3/RECENCY 차이는 동일한 추론 mask 아래 서로 다른 학습 가중치의 차이다. 이를 ‘모델이 gate를 무시한다’고 해석하면 안 된다. 서로 다른 규칙이 해당 입력에서 같은 mask를 만든 것이다. 형태별·오류별 일치율은 `GATE_IDENTIFIABILITY.csv`에 있다.

## 4. 미적응 모델과 단순 기준선

{chr(10).join(baseline_sentences)}

SHIFT8의 양성 이득과 함께 원자료 손해가 확인됐다. NESO REFERENCE에서 B0는 F0보다 5.805400%, C3는 6.205869% 악화했다. 전력16계열 REFERENCE에서도 C3는 F0보다 4.839067% 악화했다. 따라서 ‘잘 학습된 B0’는 source의 정해진 validation objective로 적응한 모델이라는 뜻으로 한정해야 하며 모든 대상·조건에서 미적응 모델보다 강한 기준선이라는 뜻으로 쓰지 않는다.

위 문장은 SHIFT8만 요약한다. 원자료와 오류 조건까지 포함한 nMAE는 다음과 같다. 단순 지속 예측이 잘하는 합성 지속 변화와 실제 사건 해결은 같지 않다. 단순 예측의 9개 저장 slot은 동일한 점예측이므로 확률 보정 우위를 주장하는 근거로 쓰지 않는다.

{table(summary)}

## 5. 논문에 사용할 시각화

[그림·표 목록](../../papers/persistence_adaptation/FIGURE_GUIDE.md)의9개 그림은 모두 PNG/PDF/SVG로 제공한다. 첫 원점 예시는 성능을 보고 고르지 않았다. 학습곡선은 저장된 V checkpoint 값이고 추가 학습이 아니다. 운영 혼합 그림은 가상 SHIFT8 비중 w와 나머지 중 FAULT 비중 q를 변화시킨 산술 민감도다. 실제 현장 빈도 추정이나 유리한 조건만 선택하는 통과 기준이 아니다.

## 6. 검산·자원·공개 범위

{v['new_origin_rows']:,}개 원점 점수 행, 독립 scalar {v['scalar_rows']:,}행×2지표, 직접 효과 {v['effect_rows']}행과 분해 {v['factorial_rows']}행을 검산했다. 실제 새 모델 추론에는 동결 가중치 전후 hash와 복원 검사를 적용했다. 기존 예측은 hash와 부모 검산 기록으로 재사용했다. 최소 GPU 여유 {cost['minimum_free_gpu_mib']}MiB, 허용되지 않은 외부 compute 표본 {unapproved}개다. 기존 승인된 RustDesk만 예외다.

기존 두 연구의 결과·가중치·입력 hash를 보존했다. 원시 데이터·모델·예측 cache는 로컬이며 공개 저장소에 모두 포함된 것이 아니다. 코드·명세·원점 점수·검산·논문용 표와 그림은 commit/push한다. 전체 학습 비용은 부모 연구 비용과 이번 0-update 평가 비용을 분리한다.

## 7. 논문 준비의 현재 상태

현재 자료로 가능한 중심은 **잘 적응된 예측기 위 추가 적응의 조건부 이득, 단순 대안의 설명력, 지속 변화와 원자료 보존 사이의 절충을 조사하는 통제 실증 연구**다. 관측 규칙을 구체적인 방법으로 설명할 수 있으나 identity/gate/adapter 자체의 최초성이나 범용 최우수 성능을 주장하지 않는다.

실제 사건 레이블, 다른 backbone, 모든 공식 선행 비교는 미완료다. 이들은 이번 고정 질문을 답하기 위해 기존 모델을 다시 학습할 이유가 되지는 않지만 강한 범용 방법론 주장에는 한계다. 채택 가능성을 보장하거나 실행 완료를 논문 PASS로 바꾸지 않는다. 후속 seed/LR/새 후보/자료를 자동 추가하지 않았다.
'''
    (OUT/'REPORT.md').write_text(text)
    (PAPER/'EVIDENCE_REPORT.md').write_text(text.replace('../../papers/persistence_adaptation/',''))
    # The copy in PAPER has no result-relative links; local links above point to paper assets only.
    (PAPER/'FIGURE_GUIDE.md').write_text('''# 논문 그림과 표 사용 안내

모든 그림은 같은 이름의 PNG,PDF,SVG로 제공한다. PDF/SVG는 벡터 편집용이며 PNG는 빠른 검토용이다. 표의 원자료는 `tables/`에 있다. 양수 gain/benefit은 C3의 오차 감소다.

1. **F1_main_seed_effects** — selected SHIFT8의 C3/B0,C2,RECENCY 비교. 검은 구간은 탐색적 날짜 조건부95%구간,주황×는개별seed. ETTm2 RECENCY는 미실행 표시. 서로 다른 데이터의 gain을 합산하지 않는다.
2. **F2_foundation_simple_baselines** — 원자료·FAULT·SHIFT8에서 F0,마지막값유지,계절반복,B0,C2,C3,RECENCY의 nMAE. 로그축이며 빠진 비교군은0이 아니다. 단순 점예측을 학습 seed로 복제하지 않는다.
3. **F3_matched_factorial** — 동일 LR·1,024 updates의 C2/C3 가중치와 추론 gate 교차 결과. 막대는 대칭분해의 gate/weights 항,×는합. 서로 다른 항의 크기는 순수한 기전 중요도 순위가 아니다.
4. **F4_all_shapes** — 등록된8형태와 pairedSHIFT8 전체. D17 양성과D63/pulse 음성을함께제시. standardSHIFT8과pairedSHIFT8의draw/표본차이를설명해야한다.
5. **F5_operating_mix_sensitivity** — 가상운영혼합에대한평균오차차이. w는SHIFT8비중,q는나머지중FAULT비중. q=0,.1,.5,1을모두표시하며 실제빈도·운영최적정책 추정이아니다.
6. **F6_validation_trajectories** — 기존selected LR에서0/256/512/768/1024 V점수와선택시점. 가중치를다시학습한곡선이아니다. ETTm1의C2와C3 LR는다르므로matched-training그림으로해석하지않는다.
7. **F7_cost_accuracy** — 기존RTX3080 FP32 128입력profile의seed별중앙값평균과원자료/SHIFT8점수. source B0학습비용은별도이며 latency비교를GPU메모리절감률로바꾸지않는다.
8. **F8_fixed_examples** — Electricity전이와NESO의첫E원점·첫채널·첫draw·seed81551,REFERENCE/POINT8/SHIFT8. 유리한예측으로선별한대표사례가아니다. 검은선은합성정답이며실제사건label이아니다.

9. **F9_gate_identifiability** — 평가 후 CPU 감사. C3/RECENCY mask의 context 단위 완전 일치율과 C3 all-one 비율을 전체 standard 조건에서 표시. NESO SHIFT8에서 두 규칙이 동일 mask를 만드는 사실은 학습 효과와 추론 위치 효과를 구분하는 데 필요하다.

본문 추천 배치는 F1,F3,F4,F9이며 F2,F5,F6,F7,F8은 주장과 분량에 따라 본문 또는 부록에 둔다. 원자료·오류 손해와 seed 역전은 본문에서 언급하고 부록에만 숨기지 않는다.
''')
    (PAPER/'README.md').write_text('''# 지속 변화와 추가 PEFT — 논문 준비 근거 묶음

사용자 요청에 따라 필요한 직접 비교와 시각화를 수행한 자료다. 원고만 작성한 작업이 아니며, 투고·게재 완료를 뜻하지 않는다.

- [한국어 검토 PDF](EVIDENCE_BRIEF_KO.pdf) · [편집용 DOCX](EVIDENCE_BRIEF_KO.docx)
- [방법·실험 설정](METHODS_AND_PROTOCOL.md)
- [보강 실험 결과와 논문에 쓸 수 있는 주장](EVIDENCE_REPORT.md)
- [그림9종과 표 사용 안내](FIGURE_GUIDE.md)
- [선행 대조와 신규성의 경계](RELATED_WORK_AND_SCOPE.md)
- [재현 절차와 자료 제공 범위](REPRODUCIBILITY.md)
- [주장–근거 대응표](CLAIM_EVIDENCE.md)
- [남은 제출 단계](SUBMISSION_READINESS.md)

새 학습이나 모델 변경 없이 기존 가중치로 빠진 기전 대조와 기준선 평가를 완료했다. 전체 수치는 `tables/`,편집 가능한 벡터 그림은 `figures/`에 있다. 검산 결과는 [이번 검증 기록](../../results/paper_readiness_20260918/VERIFICATION.json)을 참조한다.
''')
    print('REPORT_COMPLETE',flush=True)
if __name__=='__main__':report()
