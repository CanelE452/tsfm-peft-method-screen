"""Build a portable Korean evidence brief, not a submitted manuscript."""
from pathlib import Path
import subprocess,json
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'papers/persistence_adaptation'
figures=[('F1_main_seed_effects','주 결과와 seed별 변동','selected SHIFT8의 평균 효과와 조건부 날짜 구간. 주황 ×는 개별 seed다. ETTm2의 RECENCY는 미실행이다.'),('F2_foundation_simple_baselines','미적응 모델과 단순 기준선','원자료·오류·지속 변화의 손익을 분리한다. 로그축이며, 단순 점예측을 확률 예측 성능의 대체물로 해석하지 않는다.'),('F3_matched_factorial','학습된 가중치와 추론 gate의 분해','Electricity source의 동일 학습률·1,024 업데이트 모델을 교차 평가했다. 구간은 탐색적이며, 서로 다른 source의 효과를 합산하지 않는다.'),('F4_all_shapes','변화 형태에 따른 추가 가치','등록한 모든 형태와 불리한 조건을 함께 보인다. 색과 숫자는 C3의 상대 오차 감소율이다. N/A는 미실행이며 0점이 아니다.'),('F5_operating_mix_sensitivity','가상 변화 빈도에 따른 절충','w는 SHIFT8 비중, q는 나머지 중 FAULT 비중이다. 실제 현장 빈도 추정이나 배포 정책의 최적화가 아니다.'),('F6_validation_trajectories','기존 validation 학습곡선','새 학습 없이 기존 checkpoint의 V 점수를 그렸다. ETTm1의 C2/C3는 선택 학습률이 다르므로 동일 학습 조건의 비교라고 부르지 않는다.'),('F7_cost_accuracy','자원 비용과 정확도','기존 RTX 3080 FP32 측정값을 사용했다. 128입력 기준이며 B0의 선행 학습 비용은 별도다. 시간축과 오차축 모두 낮을수록 좋다.'),('F8_fixed_examples','사전에 고정한 첫 원점 예시','첫 E 원점·첫 채널·첫 draw·seed81551이다. 성능이 좋은 사례로 선택하지 않았다. 합성 변화로 음수 수요 등 비현실적 값이 나타날 수 있다.'),('F9_gate_identifiability','두 gate 규칙을 구분할 수 있는가','평가 후 CPU 진단이다. NESO SHIFT8의 모든 256개 입력에서 C3와 RECENCY가 같은 mask를 만든다. 두 모델의 차이를 이 조건의 추론 위치 선택 능력으로 설명할 수 없다.')]
front='''# 지속 변화와 추가 PEFT: 논문 준비 근거 묶음

한국어 검토본 · 2026-09-18 · 기존 결과 보존 · 투고 원고가 아닌 실험·시각화 요약

## 무엇을 실제로 했는가

같은 학습률과 1,024회 업데이트를 사용한 C2/C3의 가중치와 추론 gate를 교차 평가했다. 미적응 Chronos-Bolt(F0), 마지막 값 유지, 계절 반복을 같은 입력과 정답으로 비교했다. NESO의 C3/RECENCY 교차 평가도 추가했다.

114개 비교를 완료했다. 이 중 24개는 검증된 기존 예측을 재사용했고, 90개는 새로 평가했다(GPU 70개, CPU 단순 예측 20개). 새 학습과 optimizer update는 0회다. 원점 점수 640,320행, 독립 scalar 3,534행×2지표, 직접 효과 742행과 분해 56행을 검산했다.

## 논문에 남길 발견

- 기존 selected SHIFT8에서 C3는 전력16계열의 B0보다 7.112%, 일반 어댑터 C2보다 2.399% 개선됐다. NESO에서는 각각 5.459%, 1.312% 개선됐다.
- 전력16계열의 동일 예산 C2/C3 분해는 총 nMAE 이득 +0.011603 중 gate 항 +0.002701, 가중치 항 +0.008903을 보였다. 학습된 함수의 차이와 추론 규칙 양쪽을 구분할 근거다.
- NESO의 같은 분해는 gate 항 −0.000045, 가중치 항 +0.005175였다. gate 구간은 0을 포함한다. 전력16계열의 설명을 모든 원천에 그대로 확장할 수 없다.
- SHIFT8에서 C3는 이번 다섯 패널 모두 F0와 두 단순 점예측보다 낮은 평균 오차를 보였다. 그러나 NESO 원자료에서는 F0보다 6.206% 악화했다.

## 반드시 함께 적을 한계

C3는 NESO의 RECENCY보다 평균 0.109% 나빴고 세 번째 seed의 C2 대비 이득도 역전했다. NESO SHIFT8의 C3/RECENCY mask는 입력 256개에서 모두 동일했다. 따라서 이 조건은 두 규칙의 추론 위치 선택 능력을 구분하지 못한다. 모든 mask가 유용하지 않다는 뜻도 아니다.

ETTm1/2의 불리한 결과, 긴 변화와 pulse의 손해, 원자료와 오류 조건의 손해를 남긴다. 합성 변화는 실제 센서 사건 레이블이 아니다. 이번 추가 평가는 이미 결과를 본 자료의 후속 진단이며 새 독립 시험이 아니다. 정식 COSA/TATO/SOLID/Time-PEFT 전체 비교는 실행하지 않았다.

현재 근거는 **추가 PEFT의 조건부 이득과 보존의 절충을 다루는 통제 실증 연구**에 맞는다. 범용 새 방법의 우위나 논문 채택을 보장하지 않는다. 전체 표·코드·자료 계보는 같은 폴더의 README, EVIDENCE_REPORT, CLAIM_EVIDENCE, REPRODUCIBILITY에 있다.
'''
body=front
for i,(name,title,caption) in enumerate(figures,1):
 body+='\n<div class="figure-page">\n\n'+f'## 그림 {i}. {title}\n\n![{title}](figures/{name}.png)\n\n{caption}\n\n</div>\n'
(P/'EVIDENCE_BRIEF_KO.md').write_text(body)
(P/'brief.css').write_text('''@page { size:A4; margin:18mm; } body{font-family:"Noto Sans CJK KR",sans-serif;font-size:10pt;line-height:1.65;color:#14212c;max-width:100%;}h1{font-size:21pt;line-height:1.4;}h2{font-size:14pt;color:#153b57;break-after:avoid;}p,li{orphans:3;widows:3;}img{max-width:100%;max-height:200mm;}figure{margin:4mm 0;}figcaption{display:none;}.figure-page{break-before:page;break-inside:avoid;}a{color:#225b83;}''')
pandoc='/home/minjae/anaconda3/bin/pandoc'
subprocess.run([pandoc,'EVIDENCE_BRIEF_KO.md','--standalone','--embed-resources','--css=brief.css','--metadata=lang:ko','--metadata=pagetitle:추가 PEFT 논문 준비 근거','-o','EVIDENCE_BRIEF_KO.html'],cwd=P,check=True)
subprocess.run([pandoc,'EVIDENCE_BRIEF_KO.md','--resource-path=.','--metadata=lang:ko','-o','EVIDENCE_BRIEF_KO.docx'],cwd=P,check=True)
p=P/'EVIDENCE_BRIEF_KO.html';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
print('BRIEF_HTML_DOCX_READY')
