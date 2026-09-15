"""Post-hoc interpretation only: no predictions, selection, or threshold changes."""
import csv
from common import *
def main():
 with open(OUT/'monthly.csv') as f:rows=list(csv.DictReader(f))
 m={(r['method'],int(r['month'])):float(r['primary']) for r in rows};rng=np.random.default_rng(61701);idx=rng.integers(0,4,size=(2000,4));pairs=[]
 for c,b in [('LATEST','F0'),('MIXTURE','LATEST'),('MIXTURE','MEAN_INPUT'),('MIXTURE_CAL','MIXTURE')]:
  x=np.array([m[c,k] for k in [6,7,8,9]]);y=np.array([m[b,k] for k in [6,7,8,9]]);bs=100*(y[idx].mean(1)-x[idx].mean(1))/y[idx].mean(1)
  pairs.append(dict(candidate=c,baseline=b,candidate_primary=float(x.mean()),baseline_primary=float(y.mean()),gain_percent=float(100*(y.mean()-x.mean())/y.mean()),positive_months=int(np.sum(x<y)),ci_low=float(np.quantile(bs,.025)),ci_high=float(np.quantile(bs,.975)),monthly_gain_percent=(100*(y-x)/y).tolist()))
 v=read(OUT/'verification.json');t=v['timing'];ratio=t['PATHS']['median_seconds']/t['LATEST']['median_seconds'];save(OUT/'posthoc_pairs.json',dict(scope='DESCRIPTIVE_NO_RESELECTION_NO_NEW_FORECASTS',pairs=pairs,batched_four_vs_single_latency_ratio=ratio,prior_decision_unchanged=read(OUT/'decision.json')))
 text=['# 추가 해석 — 정보 이득과 혼합 이득을 분리','',
 '[확인] 아래는 이미 저장된 예측의 사후 설명이다. MIXTURE_CAL을 주 비교로 한 사전 판정과 원점수는 그대로다. 새로운 학습·예측·보정 설정 선택·성공 문턱 변경은 하지 않았다.','',
 '| 비교 | Primary 개선율 % | 개선한 달 | 월 bootstrap95% % |','|---|---:|---:|---:|']
 for r in pairs:text.append(f"| {r['candidate']} / {r['baseline']} | {r['gain_percent']:.4f} | {r['positive_months']}/4 | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] |")
 text += ['',
 '미래 기상 입력의 가치가 없었다는 결론은 틀리다. 최신 미래 경로를 넣는 것 자체의 이득이 혼합의 추가 이득보다 컸다. 네 경로의 원본 혼합도 양의 평균 이득이 있었다. 이를 사전 주 비교 대신 골라 PASS로 바꾸지는 않지만 양의 관찰을 실패라는 단어로 지우지도 않는다.','',
 'C에서 선택한 MIXTURE 보정은 D에서 원본보다 나빴다. 이는 C-only 보정의 전달 실패라는 관찰이며, 원본 CDF 혼합의 정확도 향상을 부정하지 않는다. 기상 vintage 분포 자체의 올바른 보정 여부, 계절 이동, 추정 표본수는 아직 분리하지 못했다. 현재 C/D로 새 보정 규칙을 반복 선택하면 독립 근거가 되지 않는다.','',
 f"네 경로를 함께 배치한 추론의 median 시간은 단일 경로의 {ratio:.4f}배였다. 약4배라고 가정하는 학생 속도 주장은 성립하지 않는다. 이번 Python CDF 혼합의 평균 CPU 시간은 {v['mixture_cpu_seconds']/28*1000:.3f}ms/원점이지만, 이는 현재 구현의 비용이다. 이를 더 효율적인 CDF 처리 없이 학습 방식의 필연적인 비용 우위로 쓰면 안 된다. GPU pipeline 시간에는 작은 원장 저장 비용도 포함되어 있어 작은 차이를 정밀한 서비스 지연 차이라고 주장하지 않는다.",'',
 '따라서 다음 방법론 투자에 남은 문제는 두 개다. 첫째, 동일 정보의 강한 적응/보정 대조보다 참조 예측에 충분한 추가 가치가 있는가. 둘째, 작은 추가 가치에 비해 단일 실행 학생의 고유 이득이 무엇인가. 이번 고정 경로 혼합만으로 이 두 조건을 충족했다고 볼 수 없어 학생 학습은 진행하지 않았다. 이는 모든 외생변수 PEFT 방향의 실패 판정이 아니다.','',
 '가용성 감사 뒤 추가로 읽은 Open-Meteo 공식 문서는 Previous Runs API가 valid time 기준의 고정 lead offset을 제공하며, 특정 초기화 run 전체를 원하면 Single Runs API를 사용하라고 설명한다. 이는 fixed-offset 경로와 발행 run을 구분해야 한다는 근거다. Liander 파일이 그 API의 특정 호출로 생성됐다는 연결까지 확인한 것은 아니다. [Open-Meteo 공식 설명](https://open-meteo.com/en/docs/previous-runs-api).','']
 (OUT/'INTERPRETATION.md').write_text('\n'.join(text));print(json.dumps(pairs,indent=2))
if __name__=='__main__':main()
