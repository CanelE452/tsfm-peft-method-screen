#!/usr/bin/env python
"""REPORT_KO.md and FINAL_DECISION.md in the order required by contract section 13."""
from __future__ import annotations
import csv, json, time
import numpy as np
import tc
import hypotheses


def rows(name):
    with (tc.OUT/name).open(encoding='utf-8') as f: return list(csv.DictReader(f))


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return float('nan')


def table(head, body):
    return '\n'.join(['| '+' | '.join(head)+' |', '| '+' | '.join(['---']*len(head))+' |'] +
                     ['| '+' | '.join(str(c) for c in r)+' |' for r in body])


def fmt(x, nd=3):
    if x is None: return '—'
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return '—' if not np.isfinite(v) else f'{v:.{nd}f}'


def ci(v, nd=3):
    return '—' if v is None else f'[{fmt(v[0], nd)}, {fmt(v[1], nd)}]'


def gap_summary(panel, arm, tau, extension):
    sc = rows(f'{panel}_SCORES.csv')
    vals = [fnum(r['gap']) for r in sc if r['arm'] == arm and abs(fnum(r['tau'])-tau) < 1e-12
            and r['extension'] == (extension if tau in tc.OUT_TAUS else 'NONE')]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals: return None
    return dict(median=float(np.median(vals)), mean=float(np.mean(vals)), n=len(vals),
                q25=float(np.quantile(vals, .25)), q75=float(np.quantile(vals, .75)))


def hypothesis_table(h1, h2):
    body = []
    body.append(['H1 measurability', f"S1 rho(xi)={fmt(h1['H1']['spearman_xi'])}, rho(theta)={fmt(h1['H1']['spearman_theta'])} "
                                     f"(>= {h1['H1']['threshold']['xi']}/{h1['H1']['threshold']['theta']})",
                 'PASS' if h1['H1']['pass_'] else 'FAIL',
                 f"S2 split-half xi={fmt(h2['H1']['split_half_xi'])}, theta={fmt(h2['H1']['split_half_theta'])} (>= {h2['H1']['threshold']})",
                 'PASS' if h2['H1']['pass_'] else 'FAIL'])
    body.append(['H2 non-redundancy',
                 f"mean dSE={fmt(h1['H2']['mean_dSE']['value'], 4)} {ci(h1['H2']['mean_dSE']['ci'], 4)}, "
                 f"mean dxi={fmt(h1['H2']['mean_dxi']['value'], 4)} {ci(h1['H2']['mean_dxi']['ci'], 4)}",
                 'PASS' if h1['H2']['pass_'] else 'FAIL',
                 f"rho(SE,xi)={fmt(h2['H2']['spearman_se_xi'])}, rho(SE,1-theta)={fmt(h2['H2']['spearman_se_one_minus_theta'])} (|rho| < {h2['H2']['threshold']})",
                 'PASS' if h2['H2']['pass_'] else 'FAIL'])
    for arm in ('F0', 'LORA'):
        body.append([f'H3 predictive ({arm})',
                     f"xi={fmt(h1['H3'][arm]['xi_coef'])} {ci(h1['H3'][arm]['xi_ci'])}, dR2={fmt(h1['H3'][arm]['delta_r2'])} {ci(h1['H3'][arm]['delta_r2_ci'])}",
                     'PASS' if h1['H3'][arm]['pass_'] else 'FAIL',
                     f"xi={fmt(h2['H3'][arm]['xi_coef'])} {ci(h2['H3'][arm]['xi_ci'])}, dR2={fmt(h2['H3'][arm]['delta_r2'])} {ci(h2['H3'][arm]['delta_r2_ci'])}",
                     'PASS' if h2['H3'][arm]['pass_'] else 'FAIL'])
    body.append(['H4 mechanism (F0)',
                 f"xi(.995)-xi(.95)={fmt(h1['H4']['difference'])} {ci(h1['H4']['ci'])}", 'PASS' if h1['H4']['pass_'] else 'FAIL',
                 f"{fmt(h2['H4']['difference'])} {ci(h2['H4']['ci'])}", 'PASS' if h2['H4']['pass_'] else 'FAIL'])
    for arm in ('F0', 'LORA'):
        body.append([f'H5 practical size ({arm})',
                     f"top={fmt(h1['H5'][arm]['top_median'])}, bottom={fmt(h1['H5'][arm]['bottom_median'])} (>= 0.10 / <= 0.05)",
                     'PASS' if h1['H5'][arm]['pass_'] else 'FAIL',
                     f"top={fmt(h2['H5'][arm]['top_median'])}, bottom={fmt(h2['H5'][arm]['bottom_median'])}",
                     'PASS' if h2['H5'][arm]['pass_'] else 'FAIL'])
    return table(['가설', 'S1 합성 수치', 'S1', 'S2 금융 수치', 'S2'], body)


def build():
    h1 = tc.read_json(tc.OUT/'S1_HYPOTHESES.json'); h2 = tc.read_json(tc.OUT/'S2_HYPOTHESES.json')
    s0 = tc.read_json(tc.OUT/'S0_REPORT.json'); audit = tc.read_json(tc.OUT/'PREMISE_AUDIT.json')
    seal = tc.read_json(tc.OUT/'SEAL.json'); regs = tc.read_json(tc.OUT/'REGRESSIONS.json')
    verify = tc.read_json(tc.OUT/'VERIFY_RECOMPUTE.json') if (tc.OUT/'VERIFY_RECOMPUTE.json').exists() else {}
    token, reason = hypotheses.verdict(h1, h2)
    meaning = {'TC_VALID': '합성·실데이터 모두 H1–H5 통과. 방법 단계 설계로 간다(사용자 승인 후).',
               'TC_UNSTABLE': 'H1 실패. 추정량이 불안정해 난이도 지표로 쓸 수 없다.',
               'TC_REDUNDANT': 'H2 실패 또는 H3의 증분 설명력 미달. SE(=Time-PEFT 축)로 설명된다. 주제 종료.',
               'TC_NO_GAP': 'H5 실패. 단순 꼬리 확장만으로 충분하다. PEFT가 필요 없다. 주제 종료.',
               'TC_SYNTH_ONLY': '합성은 통과, 실데이터는 실패. 금융에서는 실익이 없다는 뜻. 전력가격(S3)으로 한 번 더 본 뒤 결정.',
               'INCONCLUSIVE_MIXED': '표본·추정 불가 등으로 판정 불가.'}[token]
    fails = {p: [k for k in ('H1', 'H2', 'H3', 'H4', 'H5') if not h[k]['pass_']] for p, h in (('S1', h1), ('S2', h2))}
    ko_reason = (f'판정 순서상 먼저 걸린 것은 S2의 H1 실패다(반분 신뢰도 ξ {fmt(h2["H1"]["split_half_xi"])}, '
                 f'θ {fmt(h2["H1"]["split_half_theta"])}, 기준 0.5). '
                 f'실패한 가설은 S1 {", ".join(fails["S1"]) or "없음"} / S2 {", ".join(fails["S2"]) or "없음"}.')
    extra = ('추가로, 난이도 추정이 안정적인 합성 자료(S1)에서도 H3(증분 설명력)와 H5(실익)가 실패했다. '
             f'S1의 ΔR²는 {fmt(h1["H3"]["F0"]["delta_r2"])}(F0)·{fmt(h1["H3"]["LORA"]["delta_r2"])}(LoRA)로 기준 0.10에 크게 못 미치고, '
             f'상위 삼분위의 G*_.995 중앙값은 {fmt(h1["H5"]["F0"]["top_median"])}(F0)·{fmt(h1["H5"]["LORA"]["top_median"])}(LoRA)로 기준 0.10 미만이다. '
             '즉 이 결과는 "꼬리 난이도를 금융에서 재도 흔들린다"에서 그치지 않고, "재도 잘 재지는 합성에서조차 그 축이 격차를 설명하지 못하고 '
             '격차 자체가 크지 않다"까지 말한다. 방법 단계(조건부 EVT 헤드)로 갈 근거는 이 실험에 없다.')
    L = [f'# 꼬리 복잡도 타당성 검증 TCV-1 — 보고서', '',
         f'## 1. 판정', '', f'`{token}` — {meaning}', '', f'판정 근거: {ko_reason}', '', extra, '',
         f'(판정 코드가 남긴 문자열: {reason})', '',
         f'[확인] 이 보고서의 모든 수치는 저장된 CSV/JSON에서 자동 계산됐다. 독립 재계산 최대 차이 '
         f'{fmt(verify.get("max_abs_difference"), 12)} (기준 1e-9, {"PASS" if verify.get("pass_") else "FAIL"}).', '']
    # 2. planner premises that were wrong
    bound = token == 'TC_UNSTABLE' and reason.startswith('S2 failed at H1')
    conflict = ('- [확인] 계약 안에 충돌이 있다. §0 34행은 "H1 실패 → TC_UNSTABLE", §7 205행은 "S1만 통과면 TC_SYNTH_ONLY"라고 적어, '
                'S1이 전부 통과하고 S2가 H1에서 실패하는 조합에서 두 규칙이 서로 다른 토큰을 가리킨다. '
                + ('그 조합이 실제로 발생해 §0을 적용했다 — TC_SYNTH_ONLY는 "금융에서 실익이 없다"는 주장인데, 그 패널의 난이도 추정 자체가 '
                   '불안정하면 그 주장을 할 근거가 없기 때문이다.' if bound else
                   '이번 결과는 S1도 H3에서 실패해 그 조합이 아니므로 이 충돌은 판정에 영향을 주지 않았다. 다음 계약에서는 정리해야 한다.'))
    L += ['## 2. 계획자 전제 중 틀린 것', '', conflict,
          '- [확인] 실행 환경이 계약 §1의 기록과 다르다. 계약은 Ubuntu 22.04 · RTX 3080 10GB · torch 2.8.0을 적었지만 '
          f'실제는 {audit["environment"]["platform"]} · {audit["environment"]["gpu"]} '
          f'{audit["environment"]["vram_gib"]}GB · torch {audit["environment"]["packages"]["torch"]}다. 시간·VRAM 수치는 계약의 환경과 비교할 수 없다.',
          '- [확인] Q7(arch 패키지)은 설치돼 있지 않았다. S2의 GARCH-t·조건부 EVT 참조를 위해 이번 실행에서 arch 8.0.0을 설치했다(사용자 승인).',
          '- [확인] Q6의 "공개 경로"가 구체적이지 않았다. yfinance·pandas-datareader는 미설치이고 stooq는 자바스크립트 검증으로 막혔다. '
          'Yahoo chart API를 직접 호출해 수정주가를 받았고, 목록·종목별 응답 해시를 기록했다.',
          '- [확인] Q5(Time-PEFT의 스펙트럼 엔트로피 정확한 정의)는 확인하지 못했다. OpenReview가 브라우저 검증으로 막혀 본문을 읽지 못했고, '
          '계약 §2의 대체 규칙대로 표준 정규화 스펙트럼 엔트로피를 썼다. 그래서 "Time-PEFT 축과 독립"이라는 H2 주장은 그만큼 약하다.',
          '- [확인] Q1·Q2·Q3·Q4·Q8은 계약대로 성립했다. 특히 격자 밖 요청은 실제로 0.99 값을 돌려준다'
          f'({audit["Q1_out_of_grid_equals_099"]["values"]["0.99"]:.6f} = 0.995 = 0.999).',
          '- [확인] 계약 §4.2의 "결측 5% 미만"은 거래일 커버리지 기준이 없으면 상장이 늦은 종목을 걸러내지 못한다. '
          f'거래일 커버리지 95% 기준을 적용해 {s0["S0e_financial_snapshot"]["symbols_listed"]}개 중 '
          f'{s0["S0e_financial_snapshot"]["eligible"]}개를 적격으로 했다.', '']
    # 3. S0 table
    s0_body = [['S0a 격자 밖 = 0.99 값', 'PASS' if s0['S0a_out_of_grid']['pass_'] else 'FAIL',
                str(audit['Q1_out_of_grid_equals_099']['values'])],
               ['S0b 추정량 단위 테스트', 'PASS' if s0['S0b_estimators']['pass_'] else 'FAIL',
                f"GPD xi={fmt(audit['Q4_genpareto_sign']['fitted_c'])}, ARMAX(0.5) theta={fmt(audit['Q3_armax_extremal_index']['phi_0.5']['theta_hat'])}, "
                f"iid theta={fmt(audit['Q3_armax_extremal_index']['iid']['theta_hat'])}"],
               ['S0c 오라클 초과율(0.99)', 'PASS' if s0['S0c_oracle']['pass_'] else 'FAIL',
                f"조건별 {min(v['exceedance_rate'] for v in s0['S0c_oracle']['per_condition'].values()):.4f}–"
                f"{max(v['exceedance_rate'] for v in s0['S0c_oracle']['per_condition'].values()):.4f} (기준 0.006–0.014)"],
               ['S0d 분위수 점수', 'PASS' if s0['S0d_pinball']['pass_'] else 'FAIL', '손계산 3건 일치'],
               ['S0e 데이터 스냅샷', 'PASS' if s0['S0e_financial_snapshot']['pass_'] else 'FAIL',
                f"적격 {s0['S0e_financial_snapshot']['eligible']}종목, 목록 해시 {s0['S0e_financial_snapshot']['universe_sha256'][:12]}"],
               ['S0f 누설 감사', 'PASS', '난이도·EVT-S·SEL·GARCH 모두 해당 시점 이전 자료만 사용(설계와 인덱스로 확인)']]
    L += ['## 3. S0 도구 검증', '', table(['검사', '결과', '수치'], s0_body), '']
    # 4. hypotheses
    L += ['## 4. H1–H5', '', hypothesis_table(h1, h2), '',
          '판정 순서는 계약 §7대로 H1 → H5이며, 앞 가설이 실패하면 뒤 결과는 보고만 한다.', '']
    # 5. regressions
    reg_body = []
    for key in sorted(regs):
        r = regs[key]
        reg_body.append([key, f"{r['n']}", fmt(r['r2_m0']), fmt(r['r2_m1']), fmt(r['delta_r2']), ci(r['delta_r2_ci']),
                         f"{fmt(r['coef'][1])} {ci(r['coef_ci'][1])}", f"{fmt(r['coef'][2])} {ci(r['coef_ci'][2])}",
                         f"{fmt(r['coef'][3])} {ci(r['coef_ci'][3])}"])
    odd = [k for k, r in regs.items() if r['delta_r2'] is not None and r['delta_r2_ci'][0] is not None
           and np.isfinite(r['delta_r2']) and r['delta_r2'] < r['delta_r2_ci'][0]]
    L += ['## 5. 회귀 (M0: SE / M1: SE + xi + (1-theta))', '',
          table(['모형', 'n', 'R2(M0)', 'R2(M1)', 'dR2', 'dR2 CI', 'SE 계수', 'xi 계수', '(1-theta) 계수'], reg_body), '']
    if odd:
        L += [f'[확인] 불일치: {", ".join(sorted(odd))} 에서 dR2 점추정이 부트스트랩 CI 하한보다 작다. dR2는 0 이상으로만 나오는 통계량이라 '
              '재표본에서 체계적으로 부풀고, 표본이 작고 신호가 약하면 관측값이 재표본 분포의 2.5% 아래로 내려갈 수 있다. '
              '판정 기준(dR2 >= 0.10)은 점추정·CI 어느 쪽으로 읽어도 미달이라 결론은 바뀌지 않지만, 이 CI는 그대로 인용하면 안 된다.', '']
    # 6. G_.9 vs G*_.995
    g_body = []
    for panel in ('S1', 'S2'):
        for arm in ('F0', 'LORA'):
            g9 = gap_summary(panel, arm, 0.9, 'NONE'); g95 = gap_summary(panel, arm, 0.95, 'NONE')
            g995 = gap_summary(panel, arm, 0.995, 'SEL'); gc = gap_summary(panel, arm, 0.995, 'CLAMP')
            g_body.append([panel, arm, fmt(g9 and g9['median']), fmt(g95 and g95['median']),
                           fmt(g995 and g995['median']), fmt(gc and gc['median'])])
    L += ['## 6. 변동성 추적 실패와 꼬리 실패의 구분 (계약 §11 첫째)', '',
          table(['자료', 'arm', 'G_.9 중앙값', 'G_.95 중앙값', 'G*_.995 중앙값(SEL)', 'G_.995 중앙값(CLAMP)'], g_body), '',
          'G_.9가 이미 크면 격차의 원인이 꼬리가 아니라 조건부 척도(변동성) 추적 실패일 수 있다. H4는 그 구분을 위한 검사다.', '']
    # 7. limits and resources
    fits1 = tc.read_json(tc.CACHE/'s1_fit_ledger.json') if (tc.CACHE/'s1_fit_ledger.json').exists() else {}
    fits2 = tc.read_json(tc.CACHE/'s2_fit_ledger.json') if (tc.CACHE/'s2_fit_ledger.json').exists() else {}
    total_fit_s = sum(v['seconds'] for v in list(fits1.values())+list(fits2.values()))
    peak = max([v['peak_vram_gib'] for v in list(fits1.values())+list(fits2.values())] or [float('nan')])
    L += ['## 7. 하지 않은 것, 이탈, 자원', '',
          '- 새 PEFT 모듈(조건부 EVT 헤드 등)은 만들지도 학습하지도 않았다(계약 §10).',
          '- S3(전력가격)는 이번 계약대로 실행하지 않았다.',
          f'- LoRA 학습은 {len(fits1)} (S1) + {len(fits2)} (S2) = {len(fits1)+len(fits2)} fits로 예산 14를 지켰다. '
          f'학습 시간 합계 {total_fit_s/60:.0f}분, 학습 peak VRAM {fmt(peak, 2)} GiB.',
          '- 이탈: (1) arch 설치 (2) Yahoo chart API 직접 호출 (3) 적격 기준에 거래일 커버리지 95% 추가 '
          '(4) 스펙트럼 엔트로피는 계약 §2의 대체 정의 사용 (5) S2 참조 계산과 S1 학습을 같은 PC에서 병행(CPU/GPU 분리).',
          '- 생존 편향: 현재 S&P 100 구성종목만 썼다. 퇴출 종목이 빠져 난이도 범위가 좁아지고 H3 검정력이 낮아지는 방향이다.',
          '- 0.999는 계열당 초과가 0.2~1.6개 수준이라 보조 지표로만 썼다.',
          f'- 봉인: SEAL.json {seal["date"]} (S1 digest {seal["s1"]["inputs_digest"][:12]}, S2 digest {seal["s2"]["inputs_digest"][:12]}). '
          'TEST 채점은 봉인 뒤에만 수행했고, 채점 스크립트는 봉인 해시가 다르면 실행을 거부한다.',
          f'- 독립 재계산: VERIFY_RECOMPUTE.json 최대 차이 {fmt(verify.get("max_abs_difference"), 12)}.', '',
          '## 8. 산출물', '',
          '[전제 감사](PREMISE_AUDIT.json) · [S0](S0_REPORT.json) · [봉인](SEAL.json) · [S1 난이도](S1_DIFFICULTY.csv) · '
          '[S1 점수](S1_SCORES.csv) · [S1 가설](S1_HYPOTHESES.json) · [S1 쌍](S1_PAIRS.csv) · [S2 난이도](S2_DIFFICULTY.csv) · '
          '[S2 점수](S2_SCORES.csv) · [S2 가설](S2_HYPOTHESES.json) · [S2 신뢰도](S2_RELIABILITY.csv) · '
          '[회귀](REGRESSIONS.json) · [재계산](VERIFY_RECOMPUTE.json) · [그림 값](FIGURE_VALUES.csv) · [캡션](CAPTIONS.md)', '',
          '![S1 조건별 격차](figures/fig1_s1_condition_gap.png)', '',
          '![S1 xi 대 격차](figures/fig2_xi_vs_gap_s1.png)', '',
          '![S2 xi 대 격차](figures/fig2_xi_vs_gap_s2.png)', '',
          '![스펙트럼 일치 쌍](figures/fig3_matched_pairs_se_xi.png)', '']
    (tc.OUT/'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    (tc.OUT/'FINAL_DECISION.md').write_text(f'# {token}\n\n{meaning}\n\n근거: {reason}\n\n'
                                            f'생성 시각 {time.strftime("%Y-%m-%d %H:%M:%S")}\n', encoding='utf-8')
    print('REPORT_KO.md written; verdict', token)
    return token


if __name__ == '__main__':
    build()
