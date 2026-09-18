"""Describe observed comparisons without inventing a paper acceptance threshold."""
import pandas as pd
from .common import *

def build():
    assert read(OUT/'VERIFICATION.json')['status']=='VERIFIED'
    e=pd.read_csv(OUT/'MATCHED_CONTRASTS.csv');f=e[(e.kind=='standard')&(e.condition=='SHIFT8')&(e.new=='C3')&e.baseline.isin(ARMS)&e.ci_type.eq('time')];seeds=pd.read_csv(OUT/'SEED_EFFECTS.csv')
    rows=[]
    for r in f.to_dict('records'):
        interpretation='C3 방향의 조건부 근거' if r['bonferroni3_low_pct']>0 else '단순 대조 방향의 조건부 근거' if r['bonferroni3_high_pct']<0 else '불확실; 동등성 입증 아님'
        ss=seeds[(seeds.panel==r['panel'])&(seeds.kind=='standard')&(seeds.condition=='SHIFT8')&(seeds.new=='C3')&(seeds.baseline==r['baseline'])]
        rows.append(dict(panel=r['panel'],control=r['baseline'],C3_gain_pct=r['gain_pct'],bonferroni3_low=r['bonferroni3_low_pct'],bonferroni3_high=r['bonferroni3_high_pct'],seed_gains=ss.sort_values('seed').gain_pct.tolist(),interpretation=interpretation))
    save(OUT/'SCIENTIFIC_INTERPRETATION.json',dict(contrasts=rows,execution='COMPLETE',paper_pass=False,formal_online_baseline_reproduced=False,new_method_proposed=False,independent_test=False,automatic_successor=False))
    lines=['# 최종 판단 — C3 약점 해결 대조','', '**핵심 결론:** 전력의 C3/C2·위치 대조·출력 보정 대비 이득은 남지만, 연속성 없는 MAG_ONLY를 넘지 못했다. 연속성의 고유 추가 가치가 강화되지 않았다. [수치에 근거한 해석](INTERPRETATION_KO.md)에 자료별 양성·음성·비용 절충을 구분했다.', '', '**실행 완료:** 30fits/30,720 main+12 smoke updates,54개 새 prediction views 및 검산. 과학적 이득·단순 대안·신규성은 실행 완료와 별개다.','', '아래 구간은 고정 seed에 조건부인 날짜 bootstrap이며 각 패널 안의 세 사전 대조에 대한 Bonferroni 구간이다. 전체 패널·조건의 다중성을 보정한 것은 아니다. 과거 개발 선택을 제거하거나 실사용 중요성을 결정하지 않는다. 계열+시간 구간과 개별 seed는 MATCHED_CONTRASTS.csv/SEED_EFFECTS.csv에 함께 있다.','', '| panel | 대조 | C3 이득(%) | 보정 구간 | 해석 |','|---|---|---:|---|---|']
    for r in rows:lines.append(f"| {r['panel']} | {r['control']} | {r['C3_gain_pct']:+.6f} | [{r['bonferroni3_low']:+.6f}, {r['bonferroni3_high']:+.6f}] | {r['interpretation']} |")
    lines += ['','**POS_ONLY 해석:** 입력값을 보지 않는 학습 위치만으로 얻는 효과와 비교한다. C3가 더 좋으면 관측별 가중의 제한된 근거이지만, 지속성 연속 길이 자체의 원인 증명은 아니다.','', '**MAG_ONLY 해석:** 큰 값 기반만으로 가능한 효과와 비교한다. C3의 추가 이득이 약하면 연속성의 고유 가치를 강하게 주장하지 않는다. 같은 평가 mask에서 규칙 차이가 식별되지 않을 수 있다.','', '**OUTPUT_CONTEXT 해석:** 알려진 출력 보정 원리를 현재와 같은 오프라인 정보 권한으로 적용한 대조다. 이것이 유리한 조건에서는 내부 patch 수정의 필수성을 낮춘다. COSA/TAFAS 전체 성능이나 온라인 방법 우위를 판단한 것은 아니다.','', '**유지할 것:** C3와 모든 대조 구현·원점수·양성 및 음성 결과, 일반 추가 어댑터 위의 조건부 이득과 원자료 보존 절충. 단순 대조가 유리하다고 기존에 실제 관찰된 C3/C2 이득을 삭제하지 않는다.','', '**주장하지 않을 것:** 기존 PEFT가 모든 시간 위치를 동일하게 수정한다는 설명, 일반 어댑터가 지속 변화를 항상 망가뜨린다는 설명, RECENCY 유사성이 지속성 정보 전체의 무의미함을 뜻한다는 설명, 완전한 인과 기전/실제 사건 해결/범용 새 PEFT 우위/논문 PASS.','', 'REFERENCE/FAULT/SHIFT4/SHIFT_POINT 및모든shape의손해를REPORT에같이남겼다. CI0포함은동등성검정이아니며효과크기와seed편차를함께본다. 어떤control이유리하다는이유로새핵심후보로바꾸거나이를추가튜닝하지않는다.','', '**남은 범위:** 정식 온라인 선행의 정보 공개·갱신 절차 비교, 독립 원천, 실제 사건 label. 문서의 LCL 준비 단계는 별도 합의 범위여서 이번에 수행하지 않았다. 현 자료를 새 독립 시험이라고 부르지 않는다.','', '**후속 실행:** 0개. 새 구조·추가 seed/LR·다른 dataset을 자동 시작하지 않는다.']
    (OUT/'FINAL_DECISION.md').write_text('\n'.join(lines)+'\n')
    print('DECISION_WRITTEN',flush=True)
if __name__=='__main__':build()
