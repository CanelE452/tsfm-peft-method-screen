"""Record the observed data block; never imports a model or starts training."""
from pathlib import Path
import csv, hashlib, io, json, zipfile, datetime, subprocess
ROOT=Path(__file__).resolve().parents[2]
NAME='mag_real_levelshift_confirmation_v1_20260921'
OUT=ROOT/'results'/NAME; CACHE=ROOT/'.cache'/NAME

def write(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def main():
 schema=json.loads((OUT/'NYISO_SCHEMA_AUDIT.json').read_text())
 receipts=json.loads((OUT/'SOURCE_RECEIPT.json').read_text())
 # Independent standard-library replay, separate from pandas schema audit.
 names=set();rows=0;csvs=0
 for receipt in receipts:
  raw=(CACHE/receipt['url'].split('/')[-1]).read_bytes()
  assert hashlib.sha256(raw).hexdigest()==receipt['sha256']
  with zipfile.ZipFile(io.BytesIO(raw)) as z:
   for f in receipt['csvs']:
    b=z.read(f['name']);assert len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256']
    reader=csv.DictReader(io.StringIO(b.decode()));assert reader.fieldnames==schema['schema'][0]
    for row in reader: names.add(row['Name']);rows+=1
    csvs+=1
 assert sorted(names)==schema['entities'] and rows==schema['rows'] and csvs==243
 access=json.loads((OUT/'ISONE_ACCESS_AUDIT.json').read_text())
 assert access[0]['status']==200 and access[0]['captcha_mentioned'] and access[1]['status']==403
 for item in access: assert hashlib.sha256((ROOT/item['local']).read_bytes()).hexdigest()==item['sha256']
 parent=json.loads((OUT/'PARENT_EVIDENCE.json').read_text())
 for p,h in parent['files'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 snapshot=json.loads((ROOT/'results/outlier_signal_peft_v1_20260917/download_receipts.json').read_text())['amazon/chronos-bolt-small']
 for p,h in snapshot['files'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
 status='BLOCKED_NO_INDEPENDENT_REAL_LOAD_SOURCE'
 dec=json.loads((OUT/'SOURCE_DECISION.json').read_text());dec.update(status=status,fallback_status='BLOCKED_DOWNLOAD_ACCESS',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fallback_detail='Official historical search requires CAPTCHA; requested official hourlysystemdemand CSV returned HTTP 403. No challenge bypass attempted.',performance_viewed=False,selected_source=None)
 write('SOURCE_DECISION.json',dec)
 write('DATA_MANIFEST.json',{'status':status,'accepted_target':None,'NYISO':schema,'ISO_NE':{'status':'BLOCKED_DOWNLOAD_ACCESS','receipt':'ISONE_ACCESS_AUDIT.json'},'raw_cache':str(CACHE.relative_to(ROOT)),'raw_files_published':False,'interpolation':False,'zone_aggregation':False})
 budget={'status':'NOT_STARTED_DATA_BLOCK','historical_lr_grid':[1e-4,3e-4],'selection_seed':92100,'repeat_seeds':[92101,92102],'families':['B0','B0_PLAIN','B0_MAG','OUTPUT_CONTEXT'],'fits_per_family':4,'fit_cap':16,'updates_per_fit':1024,'main_cap':16384,'smoke_cap':8,'smoke_allocation':'at most two per family, contingent on a valid source and implementation checks','actual_fits':0,'actual_main_updates':0,'actual_smoke_updates':0,'checkpoint_steps':[0,256,512,768,1024],'V_objective':'equal-condition nMAE: REFERENCE, POINT8, BURST8, SHIFT4, SHIFT8','tie_break':'smaller LR then earlier checkpoint','training_permitted_now':False}
 write('TRAINING_BUDGET.json',budget)
 for n in ['LR_SELECTION.json','MODEL_SELECTION.json','PREDICTIONS_MANIFEST.json']:
  write(n,{'status':'NOT_RUN_DATA_BLOCK','items':[],'reason':status})
 (OUT/'UPDATE_LEDGER.jsonl').write_text('')
 headers={'ORIGIN_AUDIT.csv':'source,split,origin_utc,eligible,reason','SHIFT_STRATA.csv':'source,origin_utc,shift_score,bin','RAW_SCORES.csv':'source,method,seed,stratum,nMAE,MAE,nRMSE,twice_pinball,quantile_crossing','PRIMARY_EFFECTS.csv':'contrast,stratum,gain_pct,ci_low,ci_high','STRATIFIED_EFFECTS.csv':'contrast,stratum,gain_pct,origins,unique_days','SEED_EFFECTS.csv':'contrast,seed,stratum,gain_pct','RESOURCES.csv':'method,actual_fits,main_updates,smoke_updates,gpu_training_seconds'}
 for name,header in headers.items():(OUT/name).write_text(header+'\n')
 write('ARTIFACT_STATUS.json',{'status':status,'empty_csv_meaning':'NOT_RUN, not zero error or zero effect','unexecuted':['source preprocessing and origins','past-only S strata and coverage','actual model parity/gradient/freeze/restore checks','all 16 fits and selection','all E predictions and scoring','bootstrap and scalar score replay'],'completed':['parent and repository audit','NYISO original archive receipt and full schema/time/quality audit','ISO-NE official access check','independent raw archive schema/hash replay'],'successor_count':0})
 audit=json.loads((OUT/'PRIOR_EXPOSURE_AUDIT.json').read_text());audit.update(local_filename_scan='rg --files --hidden --no-ignore .cache results research; provider/report/file-name regex; only current experiment matched',results_research_text_scan='NYISO|P-58B|iso-ne\\.com|Real.Time Actual Load in JSON/MD/CSV; only current experiment matched',prior_performance_exposure_found=False,limitation='Repository-visible and filename/text checks only; not proof of absence in unrecorded external work.')
 write('PRIOR_EXPOSURE_AUDIT.json',audit)
 (OUT/'PROTOCOL.md').write_text('''# 실행 계약과 중단 범위

유일한 실행 계약은 [CONTRACT.txt](../../experiments/mag_real_levelshift_confirmation_v1_20260921/CONTRACT.txt)다. 고정 MAG를 수정하지 않았으며 과거 지시문을 병합하지 않았다.

TRAIN/V/E 경계, 실제 raw E 512→64, past-only S>=3, 14일/50원점 coverage, 두 repeat seed, 7일 UTC calendar block 2000회 및 방법별 비교는 계약 그대로다. 기존 학습은 batch32/FP32/AdamW, 1024updates, 두 LR, checkpoint0/256/512/768/1024, 다섯 V 조건의 동일 가중 nMAE다. B0 shuffle key83100과 second-stage84100을 구분한다. MAG·PLAIN8712개, LoRA294912개, OUTPUT_CONTEXT4673개는 기존 코드의 값이며 이번 source의 실제 모델 검산을 했다는 뜻은 아니다.

NYISO 전체 243 CSV에 canonical system-total이 없어 source 계약이 성립하지 않는다. 임의 zone 합산은 금지되어 생성하지 않았다. 예측과 채점0회인 상태에서 승인된 ISO-NE fallback을 확인했으나 공식 CSV 접근이403이었다. 따라서 학습·모델 smoke·평가 runner는 실행하지 않았다. 데이터가 유효하지 않은 상태에서 예산을 채우기 위한 실행을 하지 않는다.

재개 조건: 계약의 ISO-NE 공식 Hourly Real-Time System Demand 2026년1~8월 원본을 정당한 공식 다운로드 절차로 확보하고 원본 receipt와 시간·schema·품질·노출 감사를 통과해야 한다. 다른 원천/threshold/기간 변경은 이 실행 범위가 아니다. 유효 입력 확보 뒤 실제 runner 구현·검사와 실행 봉인이 아직 필요하다. 자동 재개/추가 학습은 없다.
''')
 (OUT/'REPORT.md').write_text('''# 고정 MAG 실제 전력 부하 확인 — 데이터 단계 차단

## 1. 질문
실제 관측의 past-defined high-shift 구간에서 MAG가 일반 second-stage adapter보다 유용한가?

## 2. 데이터 독립성과 확보 상태
기준 local/remote main은 `40a80f6ddf0d69978a51051bc95aaa6e8a473554`다. 저장소의 기존 tracked text와 로컬 results/research 및 cache 파일명 검색에서 NYISO/ISO-NE의 이전 성능 사용 흔적을 찾지 못했다. 이는 기록 밖의 노출까지 없었다는 증명은 아니다. 부모 보고서·학습 코드·모델 코드 hash 및 pinned Chronos-Bolt-small snapshot hash를 확인했다.

[NYISO 공식 archive](https://mis.nyiso.com/public/P-58Blist.htm)의 완료된2026년1~8월 ZIP8개와 일별 CSV243개, 총785,334행을 검사했다. 컬럼은 Time Stamp / Time Zone / Name / PTID / Load다. 11개 지역만 있고 공식 system-total 행이 없어 `BLOCKED_SCHEMA_NO_SYSTEM_TOTAL`이다. N.Y.C.는 뉴욕시 지역이며 시스템 전체로 대체하지 않았다. timestamp/entity중복0, 누락·비유한값0, EST/EDT를 반영한 UTC↔America/New_York roundtrip 불일치0이다. 개별 sampling interval 빈도도 schema audit에 기록했다. 이 품질 검사는 모델 성능 평가가 아니다.

NYISO 예측·target score0회에서 [ISO-NE 공식 Hourly Real-Time System Demand](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/dmnd-rt-hourly-sys)를 확인했다. 페이지200응답에는 과거 다운로드 CAPTCHA가 있고, 동일 보고서의2026-01-01 공식 CSV 요청은403이었다. fallback은 `BLOCKED_DOWNLOAD_ACCESS`다. 인증/접근 제어를 우회하지 않았으며 다른 데이터 제품이나 provider로 대체하지 않았다.

## 3. Primary 결과
**미실행.** S>=3의 MAG 대 PLAIN 원점수, seed92101/92102 이득, block interval은 모두 N/A다. 적합한 시스템 시계열을 확보하지 못했으므로 S>=3 원점 수/날짜 수 자체도 계산하지 않았다. 이는 효과0이나 coverage 부족을 관측했다는 뜻이 아니다.

## 4. 전체 raw trade-off
F0/B0/B0_PLAIN/B0_MAG/OUTPUT_CONTEXT의 실제 E 예측과 점수 모두 N/A다. synthetic 자료로 대신 평가하지 않았다. 실제 raw 관측에서의 MAG 추가 가치는 이번 실행으로 확인되지 않았다.

## 5. 조건별 결과
S<1, 1≤S<2, 2≤S<3, S≥3의 네 bin 모두 미실행이다. 빈 CSV는 점수0이 아니라 NOT_RUN이다.

## 6. 비용과 미실행 범위
새 학습 **0/16 fits**, main **0/16,384 updates**, smoke **0/8 updates**, 모델 예측0회, GPU 학습 시간0이다. source 전처리·origin 선정·모델 검산·학습·선택·raw E 평가·통계가 남았으며, 차단 종료 때문에 실행하지 않았다.

| 방법 | first-stage params | second-stage params | deployed adaptation params | 적응 단계 | 실제 새 updates |
|---|---:|---:|---:|---:|---:|
| F0 |0|0|0|0|0|
| B0 |294912|0|294912|1|0|
| B0_PLAIN |294912|8712|303624|2|0|
| B0_MAG |294912|8712|303624|2|0|
| OUTPUT_CONTEXT |294912|4673|299585|2|0|

파라미터 표는 고정된 기존 구현의 구성값이고 이번 원천에서 모델을 만들고 측정한 결과가 아니다. 다운로드 archive와 응답 원문은 로컬 ignored cache에 있으며 공개 receipt/hash만으로 원자료가 GitHub에 포함됐다고 주장하지 않는다.

## 7. 판단
**BLOCKED_NO_INDEPENDENT_REAL_LOAD_SOURCE**

NYISO는 계약의 전체 시스템 합계가 없고 ISO-NE는 공식 과거 CSV 접근이 차단됐다. 따라서 MAG의 실제 성능 실패나 계속 개발 근거를 판단할 수 없다. 기존 synthetic 양성 결과는 development stress-test evidence로 보존한다. MAG 구조·threshold·rank·seed·source를 바꿔 rescue하지 않는다. 새 후보 및 자동 후속 학습은0이다.

## 논문 주장 경계
현재 주장 범위는 already-adapted forecaster의 작은 second-stage residual에 대한 magnitude-aware restriction의 조건부 추가 가치다. 이번 데이터 차단은 이를 강화하거나 반증하지 않는다. Time-PEFT의 joint LoRA/frequency/channel, MSFT의 multi-scale, AdaPTS의 multivariate 설정 전체와 직접 비교하지 않았다. δ-Adapter/COSA의 frozen predictor 보강은 관련 큰 설정이지만 정보 권한과 개입 위치가 다르고 OUTPUT_CONTEXT는 공식 전체 재현이 아니다. 독립 실제 자료 성능과 정식 선행 우위는 여전히 미확인이다.
''')
 (OUT/'FINAL_DECISION.md').write_text('''# 최종 판단

**BLOCKED_NO_INDEPENDENT_REAL_LOAD_SOURCE**

NYISO243개 CSV 전체에 시스템 총부하 행이 없으며, 지정된 ISO-NE 과거 자료의 공식 요청은403으로 차단됐다. 성능을 보기 전에 차단되어 학습0fits/main0/smoke0이고 MAG의 성공·실패 판정은 하지 않는다. 기존 synthetic 결과는 보존하지만 실제 raw 부하 근거로 승격하지 않는다. 정당하게 확보한 같은 ISO-NE 공식 원본 없이는 남은 비교를 실행할 수 없다. 이번 실행은 여기서 종료하며 MAG 변경·추가 후보·자동 후속 학습을 시작하지 않는다.
''')
 write('VERIFICATION.json',{'status':'VERIFIED_DATA_BLOCK_NOT_EXPERIMENT_COMPLETION','independent_csv_replay_rows':rows,'independent_csv_files':csvs,'monthly_archive_hashes_verified':8,'all_csv_hashes_verified':True,'parent_hashes_verified':True,'snapshot_hashes_verified':True,'isone_response_hashes_verified':True,'actual_fits':0,'main_updates':0,'smoke_updates':0,'predictions':0,'scores':0,'all_model_and_score_checks':'NOT_RUN_DATA_BLOCK','automatic_successors':0,'final_decision':status})
 print(status,rows,csvs)
if __name__=='__main__':main()
