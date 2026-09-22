from common import *
import csv
import numpy as np
import pandas as pd

SEEDS=[92201,92202]
ARMS={'Q':['Q_FP','Q_STD','Q_LOFTQ','Q_QERA','Q_IO16','Q_FORECAST'],
      'T':['A0','T_A1','N0','T_RECENT','T_KD','T_BLEND','T_DELTA'],
      'F':['F_LOCAL','F_SHARED','F_AFFINE','F_HEAD','F_PERIODIC']}
QUESTIONS={'Q':'실제 4bit 저장 제약에서 forecast sensitivity rank 배치가 좋은 초기화와 단순 mixed precision보다 미래 예측을 개선하는가',
           'T':'OLD 자료 없이 최근 BRIDGE를 사용할 때 옛 적응의 예측 delta가 최근 자료 재학습·일반 증류보다 유용한가',
           'F':'원자료를 서버에 보내지 않는 시뮬레이션에서 주기 개인화가 독립 LoRA·공유 적응·단순 개인화보다 유용한가'}
NOVELTY={'Q':'QLoRA·LoftQ·QERA·TQS/rank allocation과 인접',
         'T':'Trans-LoRA·distillation·task-vector/model merging과 인접',
         'F':'FFA-LoRA·개인화 FL·Fourier 보정과 인접'}


def md_table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)


def scores_for(summaries,arm):
    return [summaries[f'{arm}_{s}']['macro'] for s in ([0] if arm in ['A0','N0'] else SEEDS)]


def main():
    final=read(RESULTS/'FINAL_CLASSIFICATION.json')
    all_scores=read(RESULTS/'TEST_SCORES.json');summaries=all_scores['summaries']
    effects=read(RESULTS/'EFFECTS.json');selections=read(RESULTS/'SELECTIONS.json')
    resource_rows=[]
    for path in sorted((RESULTS/'resources').glob('*.json')):
        item=read(path)
        if 'measurements' not in item:continue
        fitpath=RESULTS/'fits'/f"{item['arm']}_{item['seed']}"/'FIT.json'
        fit=read(fitpath) if fitpath.exists() else {}
        for measurement in item['measurements']:
            resource_rows.append({'arm':item['arm'],'seed':item['seed'],**measurement,
                'base_artifact_bytes':item['base_artifact_bytes'],'adapter_artifact_bytes':item['adapter_artifact_bytes'],
                'total_artifact_bytes':item['total_artifact_bytes'],'small_bf16_ratio':item['storage_ratio'],
                'load_seconds':item['load_seconds'],'fit_seconds':fit.get('seconds',0),
                'uploaded_bytes':fit.get('uploaded_bytes',0),'downloaded_bytes':fit.get('downloaded_bytes',0),
                'scope':item['artifact_scope'],'inference_client':item['inference_client'],
                'fresh_process_pid':item['fresh_process_pid']})
    pd.DataFrame(resource_rows).to_csv(RESULTS/'resource_table.csv',index=False)
    teacher=[read(p) for p in (RESULTS/'teacher').glob('*.json')]
    summary_rows=[]
    for category in ['Q','T','F']:
        candidate=ARMS[category][-1]
        selected=[selections[category][str(s)]['baseline'] for s in SEEDS]
        rows=[r for r in effects['comparisons'] if r['category']==category and r['primary_V_selected']]
        point=next(r for r in rows if r['seed']=='score_mean')
        seed_gains=[next(r for r in rows if r['seed']==s)['gain_pct'] for s in SEEDS]
        ci=point['ci_gain_pct'];evidence=effects['decision_evidence'][category]
        label=final[category]['decision']
        rr=read(RESULTS/'resources'/f'{candidate}_92201.json')
        cost=f"{rr['storage_ratio']*100:.2f}% BF16 bytes" if category=='Q' else ('teacher-free inference' if category=='T' else '7 private params/client')
        summary_rows.append([category,candidate,' / '.join(selected),f"{point['gain_pct']:+.3f}%",'/'.join('+' if v>0 else '-' if v<0 else '0' for v in seed_gains),cost,label,'unresolved'])
        folder=RESULTS/category;folder.mkdir(exist_ok=True)
        introduction=(f"이번 후보는 ‘{QUESTIONS[category]}’를 Electricity 개발 화면에서 시험했다. "
            f"V로 선택한 직접 기준선은 seed92201 `{selected[0]}`, seed92202 `{selected[1]}`이며 후보의 TEST 개선율은 각각 {seed_gains[0]:+.3f}%, {seed_gains[1]:+.3f}%, 평균 점수 기준 {point['gain_pct']:+.3f}%이다. "
            f"비용 조건은 {cost}이며 median nMAE 변화는 기준선보다 {evidence['nmae_harm_pct']:+.3f}%이다(양수는 손해). "
            f"{final[category]['standard_ko']} "
            f"판정은 **{label}**이며, {final[category]['reason_ko']} 신규성·논문 PASS는 판정하지 않았다.\n")
        metric_rows=[]
        for arm in ARMS[category]:
            values=scores_for(summaries,arm)
            metrics={k:float(np.mean([v[k] for v in values])) for k in values[0]}
            metric_rows.append([arm,*[f"{metrics[k]:.6f}" for k in ['pinball','nmae','nrmse','coverage80','width80','raw_crossing','raw_mae']]])
        text=f'# {category} REPORT_KO\n\n'+introduction
        text+=f'\n![{category} 결과와 비용](../figures/{category}_results.png)\n\n'
        text+='## 전체 비교 결과\n\n점수는 두 seed 점수의 평균이다. 예측 ensemble이 아니다. A0/N0는 고정 pretrained baseline이다.\n\n'
        text+=md_table(['Method','Pinball','nMAE','nRMSE','Coverage80','Width80','RawCrossing','RawMAE'],metric_rows)
        text+=f"\n주 비교의 paired 14일 block bootstrap 95% CI는 **[{ci[0]:+.3f}%, {ci[1]:+.3f}%]**이다. 6시간 간격 TEST 원점 866개를 56개씩 묶어 2,000회 재표집하고 모든 계열·두 seed를 함께 보존했다. overlapping target을 독립 표본으로 세지 않았다. CI는 고정된 두 seed와 계열에 조건부이며 optimizer population과 새 자료 일반화를 보장하지 않는다.\n\n"
        text+='[모든 seed 점수](../scores.csv) · [계열 점수](../series_scores.csv) · [seed별/평균 효과와 CI](../seed_effects.csv) · [자원 표](../resource_table.csv)\n'
        text+='\n## 선택과 구현\n\n'
        if category=='Q':
            a=read(RESULTS/'Q_ALLOCATION.json')
            text+=f"Chronos-Bolt-small의 eligible linear 102개 전체를 사용했다. NF4는 실제90개 linear를 packed uint8로 저장하고 backend 기본 고정밀 예외12개를 유지했다. IO16은 경계6개를 추가 BF16으로 유지하므로84개를 양자화했다. uniform rank4는 {a['uniform_budget']:,}개, Q_FORECAST는 {a['candidate_parameters']:,}개 trainable parameter로 미사용 예산 {a['unused_fraction']*100:.3f}%다. rank map은 TEST 전에 봉인했다.\n\n"
            text+='LoftQ는 HF 공식 one-step replacement이며 전체 iterative LoftQ가 아니다. QERA-diag는 공식 residual×RMS-scale SVD 수식과 동일 packed matrix에서 수치 대조했다. factor product 최대 차이는 '+f"{a['qera_formula_check']['official_function_product_max_abs']:.3g}"+'였지만, 전체 QERA 논문 재현이라고 부르지 않는다. activation RMS는 동일 TRAIN-only32 contexts로 계산했다. sensitivity는 미래 y를 읽지 않고 최대288 example forwards를 사용했다.\n\n'
            text+='저장량은 직렬화된 packed weights·scales/metadata·고정밀 예외·config·선택 adapter를 합산했다. 기준 분모는 BF16 pretrained base이고 60% cap은 프로젝트의 운영 제약이다. 새 프로세스에서 실제 packed artifact를 읽어 batch1/8을 측정했으며 배포 파일의 공식/native 예측 parity도 검사했다. 4bit가 자동으로 빠르다고 가정하지 않았다.\n\n'
            text+='원래 각 fit 초기화 타이밍이 별도로 계측되지 않아 같은 고정 초기화의 runtime-only 재생 결과를 resources/*_initialization.json에 별도로 저장했다. optimizer와 sensitivity probe를 추가하지 않았으며 원래 학습의 직접 계측값으로 표기하지 않는다.\n'
        elif category=='T':
            text+='사용 조건은 `OLD_DATA_UNAVAILABLE + RECENT_BRIDGE_AVAILABLE`이다. 원래 50% OLD_TRAIN으로 small 교사 두 개를 학습하고 OLD_VAL로 선택했다. base 학생은 별도 프로세스에서 BRIDGE-only 패킷만 읽었으며 전체 원자료 디렉터리 접근을 차단했다. 학생의 실제 context와 target은 BRIDGE 내부이고, sigma metadata는 최초50%에서 고정한 값을 허용했다.\n\n'
            text+='RECENT도 실제 최근 y로 학습했다. KD·BLEND·DELTA는 같은 true loss에 lambda0.5의 pseudo L1을 추가했다. 같은 seed의 A1을 사용했고 교사 ensemble은 없다. 학생 checkpoint에는 base의 q/v LoRA만 있으며 추론에 A0/A1을 호출하지 않는다. label-free/data-free 방법이라고 부르지 않는다.\n\n'
            text+=f"교사 cache 생성은 총 {sum(t['example_queries'] for t in teacher):,} example queries / {sum(t['batch_calls'] for t in teacher):,} batch calls, {sum(t['seconds'] for t in teacher):.2f}초, {sum(t['cache_bytes'] for t in teacher)/2**20:.2f} MiB였다. 입력·모델 revision·matched teacher checkpoint hash를 묶었다. 실제 true/pseudo loss와 각각의 gradient norm은 모든 update의 training.jsonl에 남겼다.\n\n"
            text+='A1 대 A0와 N0 대 A1 비교는 seed_effects.csv의 진단 행에 모두 공개했다. 교사가 새 모델보다 약한 것만으로 delta 이전의 논리적 불가능성을 주장하지 않는다.\n'
        else:
            text+='선택한16개 열의 처음4개(col148/41/176/135)를 client로 사용했다. 각 client는 자기 값·scale로만 학습한다. 공유 FFA는 seed별 같은 A를 고정하고 B만 학습·equal-client 평균했다. LOCAL은 양쪽 factor를 학습하므로 trainable count가 다르며 INITIAL_AUDIT에 공개했다.\n\n'
            text+='16 rounds ×4 clients ×4 local updates, client당64 updates다. 공유 optimizer는 round마다 reset, private optimizer state는 client에 유지했다. 실제 업로드 key는 B뿐이고, mean(B)A=mean(BA)를 NumPy 산술과 함께 검산했다. 서버 V 선택에는 네 scalar만 반환했다. public series의 소프트웨어 시뮬레이션이며 실제4회사·DP·규제 준수 실험이 아니다.\n\n'
            text+='AFFINE2개, HEAD128개, PERIODIC7개의 private coefficient를 같은 lr1e-3으로 학습했다. 주기24/168은 hourly slot 가설이며 weekday label이 아니다. [모든 client의 LOCAL/SHARED 대비 손익](../F_client_effects.csv)과 [validation-fixed tail](../F_validation_fixed_tail.csv)을 공개했다. TEST worst-client를 선택 과정에 사용하지 않았다.\n\n'
            text+=f"계약의 보수적 client 위험 검사에서 확인한 최대 LOCAL 대비 손해는 {evidence['max_F_client_harm_pct_vs_LOCAL']:.3f}%다. 추론 latency/VRAM은 client0 대표 입력이고, 모든 client의 학습 wall/peak·저장·통신은 FIT에 별도 기록했다. 자원표의 adapter bytes는 네 client 전체 workflow 상태이며 한 client의 배포 크기라고 해석하지 않는다.\n\n"
            text+='F smoke 최초 AFFINE1 update 뒤 동결 hash 검사가 실패했다. PEFT disable_adapter 복원이 고정 A의 requires_grad를 켠 것이 원인이었고, optimizer0 재현으로 tensor 값 변화 없이 mask/hash가 바뀜을 확인했다. mask 복원 후 남은5 smoke updates 안에서 검증을 마쳤다. 실패1회도24회 상한에 포함하며 F main 이전 문제였다. 수정 전 source와 장부를 보존했고 main 재학습은 없었다.\n'
        text+='\n## 판정 범위와 검산\n\n'
        text+=f"수치 판정은 `{evidence['mechanical_category']}`, 최종 해석은 `{label}`이다. 검증이 마지막 checkpoint까지 계속 개선하는 경우의 flag는 `{evidence['optimization_flag']}`이며 자동 연장하지 않았다. {NOVELTY[category]}하므로 한 데이터·두 seed의 결과를 최초 방법론이나 논문 PASS로 바꾸지 않는다.\n\n"
        text+='학습의 frozen hash는 가중치와 persistent buffer를 포함한다. nonpersistent quantiles metadata의 학습 전후 hash는 수집하지 않았으며 배포 roundtrip의 buffer/공식 예측 검사를 별도 수행했다. [검산 범위](../VERIFICATION.json), [실행 무결성 설명](../../../experiments/'+NAME+'/DECISION_DETAILS.md), [source·cache manifest](../MANIFEST.json)를 함께 확인해야 한다.\n\n'
        text+='GitHub에는 코드·표·그림·hash를 남겼고 raw data, HF weights, checkpoint, 전체 예측 cache는 제외했다. 수치 재생에는 로컬 cache 또는 동일 계약의 재실행이 필요하다. 추가 seed/LR/rank/bit-width/dataset/자동 v2는 실행하지 않는다.\n'
        (folder/'REPORT_KO.md').write_text(text,encoding='utf-8')
        decision=f"# {category}: {label}\n\n{introduction}\n"
        decision+=f"- Mean gain: {point['gain_pct']:+.6f}%\n- Seed gains: {seed_gains[0]:+.6f}%, {seed_gains[1]:+.6f}%\n- 95% block CI: [{ci[0]:+.6f}%, {ci[1]:+.6f}%]\n- Mechanical category: {evidence['mechanical_category']}\n- Recommendation: {'RECOMMEND_FOR_NEXT_CONFIRMATION' if final.get('recommendation')==category else 'NO_AUTOMATIC_FOLLOWUP'}\n\n[전체 보고서](REPORT_KO.md) · [판정 근거](../EFFECTS.json)\n"
        (folder/'FINAL_DECISION.md').write_text(decision,encoding='utf-8')
    text='# TSFM PEFT 세 후보 Go / Hold / No-go\n\n'
    text+=final['summary_ko']+'\n\n'
    text+='![세 후보의 추가 효과와 CI](figures/triage_effects.png)\n\n'
    text+=md_table(['Candidate','Method','V-selected baseline (seed order)','Mean gain','Seeds','Cost condition','Decision','Novelty'],summary_rows)
    text+='\n'+f"**다음 확인 추천: {final.get('recommendation') or 'NONE'}**. 추천은 최대 하나이며 새 실험을 자동 실행하지 않는다.\n\n"
    text+='[Q 보고서](Q/REPORT_KO.md) · [T 보고서](T/REPORT_KO.md) · [F 보고서](F/REPORT_KO.md)\n\n'
    text+='모든 비교는 Electricity16개 계열(F4 client), past512/native64, seed92201/92202에서 수행했다. 선택용 seed는 없으며 두 seed 점수를 평균했다. Q3072 + T2560 + F2560 = **main8192**, smoke는 실패1회 포함 **24**, 전체8216 optimizer 호출이다. F_LOCAL의 두 workflow는 각4개 독립 client 모델이므로 중앙 fit10개와 동일한 단위라고 부르지 않는다.\n\n'
    text+='Q/T/F는 독립 후보이며 기존 MAG/HIER/rollout/FR을 재개하거나 기존 학습 checkpoint를 초기값으로 사용하지 않았다. 각 후보 source를 main 전에 봉인하고 V로 checkpoint와 baseline을 선택했다. 34개 TEST prediction을 모두 저장한 다음에만 채점했고, raw scalar metric과 gain을 독립 재계산했다.\n\n'
    text+='F preflight의 PEFT requires_grad 복원 문제는 main 전에 수정했다. 실패 기록을 성공 기록으로 덮지 않았고 실패1 update도 장부에 포함했다. 검산은 동결 persistent state와 저장/복원 및 source/prediction hash 범위이며, nonpersistent metadata의 학습 전후 hash와 원래 fit별 Q 초기화 타이밍은 수집하지 않은 제한을 공개했다.\n\n'
    text+='## 결과와 검산 자료\n\n'
    text+='- [Raw/seed scores](scores.csv), [series scores](series_scores.csv), [seed effects / CI](seed_effects.csv)\n'
    text+='- [Resource table](resource_table.csv), [F client effects](F_client_effects.csv), [F validation-fixed tail](F_validation_fixed_tail.csv)\n'
    text+='- [Selection seal](SELECTIONS.json), [prediction seal](TEST_PREDICTIONS_SEAL.json), [optimizer ledger](OPTIMIZER_LEDGER.json)\n'
    text+='- [Verification](VERIFICATION.json), [manifest](MANIFEST.json), [failure/repair](F_SMOKE_FAILURE.json)\n'
    text+='- [실행 계약](../../experiments/'+NAME+'/contract/MASTER_PLAN.txt), [고정 세부 규칙](../../experiments/'+NAME+'/PROTOCOL.md), [선행 경계](../../experiments/'+NAME+'/SOURCE_NOTES.md)\n\n'
    text+='이 결과는 공개 개발 자료 하나, 작은 반복 수, 한 모델 계열에 한정된다. GO는 후속 예산 배분 신호일 뿐 신규성·논문 PASS가 아니며, NO_GO는 현재 작은 구현의 판단으로 PEFT 분야 전체를 반증하지 않는다. raw/weights/checkpoints/전체 예측은 로컬 cache에만 있으므로 GitHub 자료만으로 수치 재생이 완결된다고 주장하지 않는다.\n'
    (RESULTS/'TRIAGE_SUMMARY_KO.md').write_text(text,encoding='utf-8')
    (RESULTS/'README.md').write_text('# Three-candidate PEFT screen\n\n[한국어 통합 보고서와 그림](TRIAGE_SUMMARY_KO.md)\n',encoding='utf-8')


if __name__=='__main__':main()
