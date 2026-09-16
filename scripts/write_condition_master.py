"""Render final Korean master from completed, independently audited records only."""
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.condition_studies_v1_20260916.common import *
from experiments.condition_studies_v1_20260916.report import table

def run():
 pub=read(OUT/'publication_audit.json');assert pub['passed'] and pub['all_nine_complete'];assert read(OUT/'verification.json')['passed'];assert read(OUT/'gradient_contribution_verification.json')['passed'];assert (OUT/'OUTCOME_INTERPRETATION.csv').exists();assert '최종 판단 대기' not in (OUT/'FINAL_DECISION.md').read_text()
 rows=[];resources=[]
 for t in ORDER:
  s=read(OUT/t/'STATUS.json');assert s['EXECUTION']=='COMPLETE';fits=read(OUT/t/'fits.json') if t in SOURCES else [];rows.append(dict(후보=t,실행=s['EXECUTION'],본학습=len(fits),업데이트=sum(x['updates'] for x in fits),근거=s['EVIDENCE'],신규성=s['NOVELTY']))
  if fits:
   d=pd.read_csv(OUT/t/'optimization_diagnostics.csv');pc=pd.read_csv(OUT/t/'prediction_costs.csv').drop_duplicates('path');resources.append(dict(후보=t,본학습_optimizer_분=sum(f['optimizer_seconds'] for f in fits)/60,본학습_선택포함_분=sum(f['seconds'] for f in fits)/60,E_실측예측_분=pc.seconds.sum()/60,최대_allocated_MiB=max(f['peak_allocated_bytes'] for f in fits)/2**20,오염updates=int(d.contaminated_updates.sum())))
 csvwrite(OUT/'MASTER_RESOURCE_SUMMARY.csv',resources)
 text='# 아홉 조건 PEFT 비교 — 최종 결과\n\n실제 실행·검산: 2026-09-16~17 KST. 버전 식별자는20260916으로 유지한다.\n\n'
 text+='**아홉 비교의 실행·평가·독립 검산을 완료했다.** 새 본학습116/116경로·59,392updates, 폐기smoke58updates를 수행했다. smoke 상한96 중 남은38updates는 필요한 검사 완료 후 사용하지 않은 여유 예산이며 필수 본학습 누락이 아니다. R05/N06는 기존 저장 예측을 사용했고 새 LoRA 학습·신경망 추론은0회다. 실행 완료는 새 방법의 성능·논문 통과를 뜻하지 않는다. 서로 다른 목표의 점수를 합산한 우승자는 만들지 않았다.\n\n'
 text+='## 계약·기존 결과·실제 실행\n\n'
 text+='단일 [MASTER_PROTOCOL](MASTER_PROTOCOL.md)을 사용했고 N01 → N02 → R05 → N06 → N03 → R04 → N07 → R08 → R09 순서를 지켰다. 전체 명세·정보 권한·원점·비교군·예산을 처음에 봉인했다. 기준485b15b의 기존48forecast경로는 정확히 같은 가중치·예측 해시를 검증해 R05/N06에서 참조했고 재학습하지 않았다. 이전 실험을 함께 재개하지 않았다.\n\n'
 text+=table(pd.DataFrame(rows),limit=20)+'\n\n'
 text+='후속 집중 문제는 **0개**로 결정했다. 알려진 단순 대안의 유용성과 새로운 PEFT 구성요소의 필요성을 구분한 판단이며 분야 전체의 불가능 결론은 아니다.\n\n'
 text+='필수 비교의 미실행 경로는0개이며 추가 후보·추가 seed·추가 학습률·추가 데이터셋은 실행하지 않았다. 정식 선행 전체 재현과 새 독립 기간 확증은 이번 계약에서 실행한 비교가 아니며 여전히 남은 한계다. CPU 정책 선택225설정, 의존구조 fitting9건을 신경망 fit과 구분한다. 학습 전 TRAIN 통계/간단한 ridge fitting·검색은 각 후보 정보 장부에 별도 기록했다.\n\n'
 text+='## 실제 효과·단순 대안·신규성\n\n'
 text+=table(pd.read_csv(OUT/'OUTCOME_INTERPRETATION.csv'),limit=20)+'\n\n'
 text+='gain은 양수일수록 비교군보다 낮은 손실이다. 표의 서로 다른 지표를 크기순으로 정렬하지 않는다. 모든 원점수·seed·조건·고정512체크포인트 결과는 각 보고서와 CSV에 남겼다. 불리한 타깃이나 seed를 제외하지 않았다. [보고 태그의 의미](REPORTING_SCOPE.md), [최종 문제 선택](FINAL_DECISION.md).\n\n'
 text+='## 실측 자원과 계산 한도\n\n'+table(pd.DataFrame(resources),limit=20)+'\n\n'
 text+=f'본학습+smoke optimizer는 {pub["main_updates"]+pub["smoke_updates"]:,}/59,488회, nativeforward는 {sum(pub["native_forwards"].values()):,}/200,000회다. controller 분류별 계수는 {pub["native_forwards"]}이다. 본학습/smoke/검증/E/동결예측/복원/gradient 검사의 독립 분리는 [FORWARD_LEDGER](FORWARD_LEDGER.csv)에 있으며 전체 native 호출 수와 일치한다. 주 controller의 실제 경과는 {pub["main_controller_seconds"]/60:.2f}분, 최종 로컬 cache는 {pub["cache_bytes"]/2**30:.3f}GiB/100GiB다. 원자료 다운로드0바이트. 외부 compute 비용 오염은 {pub["contaminated_updates"]}updates다. GPU는 한 worker와 사전 허용된 RustDesk 예외만 사용했다.\n\n'
 text+=f'저장 예측 CPU 비교의 실측 루프 시간은 R05 {pd.read_csv(OUT/"R05/resources.csv").seconds.iloc[0]:.3f}초, N06 {pd.read_csv(OUT/"N06/resources.csv").seconds.iloc[0]:.3f}초다. 기존 모델을 만드는 과거 학습 비용이나 추가 독립 검산 시간은 이 수치에 포함되지 않는다.\n\n'
 text+='CPU 검산·보고 작업을 학습과 병행했으므로 작은 시간 차이는 엄격히 격리된 속도 벤치마크의 우위로 해석하지 않는다. optimizer 시간은 forward/backward/update의 실측이고, 선택 포함 경로 시간에는 검증·체크포인트 처리가 추가된다. E 예측 시간은 guard/Python 비용을 포함하며 공유 경로는 한 번만 합산했다. CPU 전처리·검산·보고 시간까지 합친 end-to-end 비용과 같지 않다. N02 검색 비용은 별도 독립 재측정이며 원래 전처리 전체 실측이라고 주장하지 않는다. R04는 update당2 nativeforward이고 나머지는1이다. 파라미터 기본값은 rank8 LoRA1,179,648개; 보조계수·입력 행·CPU 통계·고유 정답 수 차이는 각 정보 예산 표에 있다.\n\n'
 text+='## 독립 검산과 보존\n\n'
 text+=f'[전체 봉인·예산 검산](verification.json), [게시 감사](publication_audit.json)에서 기존 파일 {pub["historical_files_preserved"]:,}개 hash 보존과 고정 설치 모델·소스의 일치를 확인했다. [선택 재계산](independent_selection_verification.json), [116경로의 원점 순서·최종 resume/Adam/RNG](training_record_verification.json), [CPU 단순 대조 선택](independent_CPU_control_verification.json), [추가 gradient 분해](gradient_contribution_verification.json)를 분리했다. 추가 gradient 검사는 사전 지정 TRAIN 사례의 선택 가중치에서64 nativeforward·0optimizer로 수행했고 가중치를 바꾸지 않았다.\n\n'
 text+='봉인 후 발견된 수식/권한·보고 수정은 [AUDIT_CORRECTIONS](AUDIT_CORRECTIONS.md)와 hash 연쇄에 남겼다. R09 TRAIN 경계의 미완료 하루 통계는 해당 후보의 첫 학습 전에 교정했고 원래 통계도 보존했다. N06 variogram의 원단위/정규화 표기도 해당 CPU 실행 전에 분리했다. 희소 원점 bootstrap의 빈 재표집은 버리거나 다시 추첨하지 않고 undefined로 기록했다. 학습률·원점·seed·목표·허용오차의 성능 맞춤 변경은 없다.\n\n'
 text+='## 해석의 범위\n\n'
 text+='E는 기존 프로그램에서 사용한 공개 원천의 개발 평가다. R05/N06는 이미 공개된 E의 재분석이다. Chronos 사전학습과의 비중복도 확인하지 못했다. 두 seed·네 채널은 새 도메인이 아니다. 원점64개가 곧 독립64일이라는 뜻도 아니다. 특히 N03의 E 원점 범위는23.75시간, V는11.5시간에 불과하다. 다른 다수 track도 평가 원점이5–6개 index날짜/3–4개 관측 주간 블록에 몰려 있다. [원점 시간 분산](origin_dispersion.csv)을 참고한다.\n\n'
 text+='2,000회 paired7일 bootstrap은 관측 자료·채널·seed에 조건부이며 아홉 주제 탐색의 다중선택이나 새 원천 불확실성을 해결하지 않는다. 빈 재표집이 있는 후보는 계산 가능한 표본에 조건부인 CI다. 0을 포함하는 구간을 동등성 증거로 쓰지 않았다. 각 후보의 알려진 구성요소와 정식 선행 미재현 범위는 LITERATURE_BOUNDARY와 [추가 확인 기록](LITERATURE_RECHECK.md)에 남겼다.\n\n'
 text+='## 후보별 한국어 보고서\n\n'
 for t in ORDER:text+=f'- [{t} {NAMES[t]}]({t}/REPORT.md)\n'
 text+='\n코드·보고서·원점수·해시·검산 기록은 GitHub에 게시한다. 큰 원자료·예측 배열·가중치는 ignored 로컬 cache에 있다. GitHub만으로 전체 수치 재생이 가능하다고 주장하지 않는다. [최종 결정](FINAL_DECISION.md) 이후 자동 후속 실행은 없다.\n'
 (OUT/'MASTER_REPORT.md').write_text(text);print('FINAL_MASTER_WRITTEN')
if __name__=='__main__':run()
