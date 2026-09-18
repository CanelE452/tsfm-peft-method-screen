from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
s=pd.read_csv(P/'LEAVE_ONE_GROUP_OUT.csv');v=pd.read_csv(P/'VARIANCE_COMPONENTS.csv');p=json.loads((P/'TRAINING_PROVENANCE.json').read_text());q=s[(s.panel=='electricity_transfer')&(s.condition=='SHIFT8')];r=v[(v.panel=='electricity_transfer')&(v.condition=='SHIFT8')]
fig,ax=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
for i,k in enumerate(['seed','channel','index_week']):
 z=q[q.factor==k];ax[0].plot([z.gain_pct.min(),z.gain_pct.max()],[i,i],lw=4);ax[0].scatter(z.gain_pct,[i]*len(z),s=15,alpha=.6)
ax[0].axvline(.2508879168645637,color='k',ls='--',label='Full mean');ax[0].axvline(0,color='grey',lw=.7);ax[0].set_yticks(range(3),['Leave one seed','Leave one channel','Leave one index-week']);ax[0].set_xlabel('MAG gain vs C3 (%)');ax[0].legend();ax[0].set_title('Mean sensitivity (different group sizes)')
ax[1].barh(r.component,r.share_pct,color='#346889');ax[1].set_xlabel('Share of local paired-error variance (%)');ax[1].set_title('Descriptive variation, not causal attribution')
fig.suptitle('Electricity transfer / SHIFT8 / 3 seeds / 128 origins / 16 channels')
for ext in ['png','pdf','svg']:fig.savefig(P/f'INFLUENCE.{ext}',dpi=180)
svg=P/'INFLUENCE.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
rows=[]
for factor in ['seed','channel','index_week']:
 z=q[q.factor==factor];rows.append(f'| {factor} | {z.gain_pct.min():.4f}–{z.gain_pct.max():.4f}% | {z.change_pp.abs().max():.4f}%p |')
vr='\n'.join(f'| {x.component} | {x.share_pct:.2f}% |' for x in r.itertuples())
text='''# 어떤 요인의 영향이 가장 큰가 — 기존 기록의 식별 범위

**현재 확인되는 핵심은 “전력 전이 SHIFT8의 평균 차이가 seed별로 달라지는 학습 결과에 민감하다”는 것이다. B0·초기 어댑터·학습 순서 중 가장 큰 원인은 아직 식별되지 않았다.** 기존 E를 사용한 사후 분석이며 새 학습·추론·bootstrap은 모두 0회다. C3와 MAG_ONLY의 3패널·표준10조건·형태9조건 및 FAULT 평균까지 60개 조합을 빠짐없이 분석했다.

## 1. 어떤 ‘영향’을 묻는가

세 질문을 구분해야 한다. (a) 완성된 모델의 규칙과 가중치를 바꿨을 때 평균 오차가 얼마나 달라지는가, (b) 어떤 실행·표본이 평균 차이를 지탱하는가, (c) 어떤 원인이 좋은 가중치를 만들었는가다. (a)와 (b)를 측정했다고 (c)가 밝혀지지는 않는다. 아래 수치를 서로 더하거나 하나의 중요도 순위로 합치지 않는다.

## 2. 완성된 함수에서는 가중치 변경 항이 더 컸다

기존 2×2 교차 진단의 전력 전이 SHIFT8에서 C3 원오차를 분모로 한 가중치 항은 MAG 방향 +0.6436%, 규칙 항은 −0.3928%, 순차이는 +0.2509%였다. 완성된 가중치 변경의 항이 더 크고 두 항이 상쇄한다. 하지만 가중치는 학습의 결과이지 그 결과를 만든 원인 이름이 아니다. 이 분해는 두 변경 순서의 평균이며 학습 기전의 인과 분해가 아니다.

## 3. 평균 차이는 seed별 실행에 가장 민감했다

| 하나씩 제외하는 집단 | 남은 자료에서 MAG 이득 | 전체 대비 최대 변화 |
|---|---|---|
'''+ '\n'.join(rows)+'''

81552의 signed 기여는 평균 차이의85.4%다. 해당 실행을 빼면 +0.0557%로 작아지지만, 공식 결과에서는 제외하지 않는다. 계열은13/16에서 MAG 방향이었다. 따라서 특정 한 계열이나 한 주간만의 현상보다는 seed별 학습 결과에 민감한 평균이라는 설명이 맞는다. 단, seed 하나는 전체의1/3, 계열 하나는1/16, 주간은 그보다 작은 부분을 제거한다. 이 서로 다른 삭제 규모로 보편적 요인 중요도를 순위화하지 않는다. index-week는 원점 index를 자료 주기(Electricity24, ETT96)×7로 나눈 블록이며 달력상의 월요일 시작 주간이 아니다.

## 4. 개별 오차 차이의 이질성은 결합 요인이 컸다

동일 seed×원점×계열의 nMAE(C3)−nMAE(MAG)를 만들고 균형 배열의 중심화 제곱합을 직교 분해했다. 이 값은 평균 이득을 설명하는 비율이 아니며 seed 간 일반화 분산 또는 인과 중요도도 아니다.

| 성분 | 국소 오차 차이 분산의 비중 |
|---|---|
'''+vr+'''

가장 큰 성분은 seed×원점×계열(41.17%), 그다음 원점×계열(33.96%)이다. 즉 같은 실행의 차이도 어느 계열의 어느 날짜를 보는지에 따라 달라진다. 삼원 성분에는 따로 분리하지 않은 모든 잔여 결합 효과가 포함된다. 이를 “삼원 상호작용이 실제 원인의41%”라고 말할 수 없다. 원점 주효과가 날짜 인과 효과이거나 seed 주효과5.71%가 초기화 영향이라는 해석도 잘못이다. 날짜의 의존성은 남아 있으며 별도 유의성 검정을 하지 않았다.

![기존 평균과 국소 오차 차이의 민감도](INFLUENCE.png)

## 5. seed를 바꾸면 무엇이 함께 변했나

저장 checkpoint를 CPU로 읽고 실제 tensor와 SHA256을 확인했다. 각 source에서 세 seed의 B0 checkpoint, 초기 어댑터 tensor, 32epoch 전체 학습 순서가 서로 다르다. 같은 source·seed 안의 C3/MAG는 B0·초기 어댑터 tensor·전체 순서·동결 가중치 hash·LR·microbatch가 같다. TRAIN x/y/σ 파일은 같은 실제 파일로 연결되어 있으며 hash를 기록했다. 합성 draws를 다시 생성하지 않는다.

Electricity의 B0 선택 단계는 seed81551/81552/81553에서 각각768/512/1024다. 추가 어댑터의 선택 단계는 두 방법 모두768/768/1024다. 따라서 방법 간 학습 길이 차이는 없지만 seed 간에는 B0 자체와 초기화·순서가 묶여 있다. ETTm1 seed81552는 C3가1024, MAG가0을 선택했으므로 선택 효과도 추가로 섞인다. provenance 전체는 TRAINING_PROVENANCE.json에 보존했다.

**결론: ‘랜덤 초기화가85.4%의 원인’이라는 주장은 불가능하다. 정확한 표현은 ‘세 실행 중 하나가 관찰된 평균 차이의85.4%를 기여했으며, 그 실행의 B0·초기화·순서가 함께 달랐다’다.**

## 6. 원인 규명에 남은 통제

최대 원인을 확인하려면 동일 B0에서 초기 어댑터만 바꾸는 비교, 동일 B0·초기값에서 batch 순서만 바꾸는 비교, 동일 초기값·순서에서 B0만 바꾸는 비교가 필요하다. 하나씩 바꾸는 비교만으로는 상호작용을 놓칠 수 있으므로 가능한 경우 세 요인을 교차해야 한다. 양쪽 gate는 항상 같은 각 조건에 짝지어야 하며 TRAIN·LR·학습 예산·선택 규칙을 먼저 고정해야 한다. 기존 평가를 보고 가장 좋은 조합을 새 방법으로 선택하면 안 된다.

이번에는 이 통제 학습을 실행하지 않았다. 기존 seed 기록에는 위 요인들을 독립적으로 바꾼 결과가 없으므로 통계 처리나 문장 수정으로 원인을 복원할 수 없다. 새 후보도 만들지 않았다. 현 논문에는 좁은 이득·교차 함수 진단·seed 묶음과 조건의 민감도까지 쓸 수 있고, 지속성 규칙이 좋은 가중치를 만드는 고유 원리라는 주장은 보류한다.

## 7. 검산과 재현

`.venv/bin/python research/c3_influence_audit_20260918/analyze.py` 및 `write_report.py`로 재생성한다. 원점별 저장 점수와 로컬 checkpoint·TRAIN 캐시가 필요하다. 공개 저장소만으로 tensor hash 재검증까지 할 수 있다는 뜻은 아니다. 60조건의 전체·seed·원점 집계가 기존 교차 진단의 대각선과 일치하며 합성 순수 주효과/상호작용 검사와 제곱합 항등식을 통과했다. AUDIT.json의 입력 hash를 작업 후 다시 확인했다.

첫 실행의 CSV chunk 읽기에서 숫자·문자 계열 ID의 dtype 추론이 달라 대응 검사가 중단됐다. channel을 처음부터 문자열로 읽어 수정했으며 데이터·방법·metric·포함 조건은 바꾸지 않았다. 검사 없이 누락된 행으로 분석을 계속하지 않았다.
'''
(P/'REPORT.md').write_text(text)
print('Report and 3 figure formats written')
