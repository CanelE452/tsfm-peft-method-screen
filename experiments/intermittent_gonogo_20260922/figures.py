"""REPORT.md 임베드용 그림 4장. 수치는 전부 results/ 의 산출물에서 읽는다 (재계산·재측정 없음).

python experiments/intermittent_gonogo_20260922/figures.py
"""
import os, sys, json
from datetime import datetime
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/intermittent_gonogo_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ASSETS = Path.home() / '.claude/agents/viz-expert/assets'
plt.style.use(ASSETS / 'analysis.mplstyle')
sys.path.insert(0, str(ASSETS))
from palette import color_for, colors_for, SEMANTIC, LINESTYLE  # noqa: E402

ROOT = Path('/home/minjae/Documents/github/tsfm-peft-method-screen')
OUT = ROOT / 'results/intermittent_gonogo_20260922'
SRC = 'results/intermittent_gonogo_20260922/'
STAMP = datetime.now()

curves = np.load(OUT / 'curves.npz')
gonogo = json.load(open(OUT / 'gonogo.json'))
ci = json.load(open(OUT / 'ci.json'))
pre_a = json.load(open(OUT / 'precheck_a.json'))
g2b = {k: json.load(open(OUT / f'{k}.json')) for k in ('g2b', 'g2b_full', 'g2b_full12')}
N_SERIES = gonogo['config']['n']


def footer(fig, files):
    fig.text(0.005, 0.004, f'src: {SRC}{{{files}}}', ha='left', fontsize=8, color='gray')
    fig.text(0.995, 0.004, STAMP.strftime('%Y-%m-%d %H:%M'), ha='right', fontsize=8, color='gray')


def save(fig, name, files):
    eng = fig.get_layout_engine()
    if eng is not None:
        eng.set(rect=(0.004, 0.028, 0.992, 0.968))   # 푸터 자리 확보
    footer(fig, files)
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


# ── Figure 1 — service / inventory frontier ─────────────────────────────────
ARMS = ['ZS-QMEAN', 'ZS-Q80', 'ZS-Q90', 'NZ-QMEAN', 'SBA', 'TSB']
DESC = {
    'ZS-QMEAN': 'ZS-QMEAN (zero-shot, quantile mean)',
    'ZS-Q80':   'ZS-Q80 (zero-shot, q0.80)',
    'ZS-Q90':   'ZS-Q90 (zero-shot, q0.90)',
    'NZ-QMEAN': 'NZ-QMEAN (non-zero-conditional norm.)',
    'SBA':      'SBA (classical intermittent)',
    'TSB':      'TSB (classical intermittent)',
}
CLASSICAL = {'SBA', 'TSB'}
CARM = dict(zip(ARMS, colors_for(ARMS)))
alpha = curves['alpha']
i_a1 = int(np.argmin(np.abs(alpha - 1.0)))

fig, ax = plt.subplots(figsize=(10.5, 6.8))
ax.axhline(0.90, color=SEMANTIC['baseline'], ls=LINESTYLE['baseline'], lw=1.2,
           label=r'fill target $\tau$ = 0.90 (defines Stock@90)', zorder=1)

for arm in ARMS:
    fill, inv = curves[f'{arm}_fill'], curves[f'{arm}_inv']
    res = gonogo['res'][arm]
    lw = 2.8 if arm == 'NZ-QMEAN' else 1.6
    ls = '--' if arm in CLASSICAL else '-'
    if res['stock90'] is None:
        lab = f"{DESC[arm]} — 0.90 not reached (max fill {res['max_fill']:.3f})"
    else:
        lab = f"{DESC[arm]} — Stock@90 = {res['stock90']:.2f}"
    ax.plot(inv, fill, color=CARM[arm], lw=lw, ls=ls, label=lab, zorder=3)
    ax.plot(inv[i_a1], fill[i_a1], marker='s', ms=5, mfc='white',
            mec=CARM[arm], mew=1.2, ls='none', zorder=4)
    if res['stock90'] is not None:
        ax.plot(res['stock90'], 0.90, marker='o', ms=10, mfc=CARM[arm],
                mec='black', mew=1.0, ls='none', zorder=6)
    else:
        ax.plot(inv[-1], fill[-1], marker='X', ms=11, mfc=CARM[arm],
                mec='black', mew=1.0, ls='none', zorder=6)

ax.plot([], [], marker='o', ms=9, mfc='0.85', mec='black', ls='none',
        label='Stock@90 (inventory at fill 0.90)')
ax.plot([], [], marker='X', ms=10, mfc='0.85', mec='black', ls='none',
        label=r'end of $\alpha$ grid without reaching 0.90')
ax.plot([], [], marker='s', ms=5, mfc='white', mec='0.4', ls='none',
        label=r'$\alpha$ = 1.00 (no scalar correction)')

zs_best, nz = gonogo['res']['ZS-Q90']['stock90'], gonogo['res']['NZ-QMEAN']['stock90']
g2a = gonogo['g2_a']
ax.annotate('', xy=(nz, 0.876), xytext=(zs_best, 0.876),
            arrowprops=dict(arrowstyle='<->', color=SEMANTIC['annotation'], lw=1.3))
ax.text(np.sqrt(nz * zs_best), 0.870,
        f"{g2a['delta_pct']:.2f}% less inventory\nCI [{g2a['ci'][0]:.2f}, {g2a['ci'][1]:.2f}]",
        ha='center', va='top', fontsize=9, color=SEMANTIC['annotation'], zorder=7,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85))

ax.text(4.25, 0.594,
        r'Reading: scalar $\alpha$ only moves an arm along its own curve. '
        r'No $\alpha$ in 0.50-4.00 lifts a zero-shot arm onto the NZ-QMEAN curve.',
        ha='left', va='center', fontsize=10, color=SEMANTIC['annotation'], zorder=7)

q80 = gonogo['res']['ZS-Q80']
q80_inv = curves['ZS-Q80_inv']
ax.annotate(f"ZS-Q80 saturates at fill {q80['max_fill']:.3f}\n"
            rf"even at $\alpha$ = 4.00 (inventory {q80_inv[-1]:.1f})",
            xy=(q80_inv[-1], q80['max_fill']), xytext=(52, 0.775),
            fontsize=9, color=SEMANTIC['annotation'], ha='center',
            arrowprops=dict(arrowstyle='->', color=SEMANTIC['annotation'], lw=1.0))

cfg = gonogo['config']
ax.text(0.015, 0.975,
        f"n series   {N_SERIES:,}\n"
        f"horizon    28 (d_1914-d_1941)\n"
        f"L = {cfg['L']}  R = {cfg['R']}  ctx = {cfg['context']}\n"
        rf"$\alpha$ grid    {alpha[0]:.2f}-{alpha[-1]:.2f} ({len(alpha)} pts)",
        transform=ax.transAxes, ha='left', va='top', fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85), zorder=7)

ax.set_xscale('log')
ax.set_xticks([5, 10, 20, 50, 100, 200])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlim(4.0, 210)
ax.set_ylim(0.575, 1.005)               # 아래 빈 띠는 읽는 법 한 줄을 놓을 자리
ax.set_yticks(np.arange(0.65, 1.001, 0.05))
ax.set_xlabel('Mean on-hand inventory (units, log scale)')
ax.set_ylabel('Mean fill-rate proxy')
ax.set_title('Service-inventory frontier, M5 daily, all eligible series\n'
             rf"each curve traces $\alpha$ = {alpha[0]:.2f}-{alpha[-1]:.2f}; "
             f"Stock@90 {nz:.2f} (NZ-QMEAN) vs {zs_best:.2f} (best zero-shot arm ZS-Q90)")
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.10), fontsize=8.5, ncol=1)
save(fig, 'service_inventory_frontier.png', 'curves.npz, gonogo.json')


# ── Figure 2 — point-extraction bias ────────────────────────────────────────
# "Quantile mean, 9 levels" 는 선행 기록(REPORT.md 표)의 추정기라 CI 산출물이 없다.
ROWS = [
    ('Median (q0.50)',                          ci['MED']['prb'],            ci['MED']['ci'],           False),
    ('Quantile mean, 9 levels q0.10-q0.90',     -0.1292,                     None,                      False),
    ('sNaive (weekly)',                         ci['SNAIVE_week']['prb'],    ci['SNAIVE_week']['ci'],   False),
    ('QMEAN trapezoid, 21 train levels',        ci['C_train_const']['prb'],  ci['C_train_const']['ci'], False),
    ('QMEAN trapezoid, constant tail',          ci['A_spec_const']['prb'],   ci['A_spec_const']['ci'],  True),
    ('QMEAN trapezoid, linear tail',            ci['B_spec_linear']['prb'],  ci['B_spec_linear']['ci'], False),
]

fig, ax = plt.subplots(figsize=(11, 5.6))
c_used, c_other = color_for('ours'), color_for('baseline')
TEXT_X = 3.0
for i, (lab, prb, cint, used) in enumerate(ROWS):
    c = c_used if used else c_other
    if used:
        ax.axhspan(i - 0.42, i + 0.42, color='0.85', alpha=0.45, zorder=0)
    if cint is not None:
        ax.plot([cint[0] * 100, cint[1] * 100], [i, i], color=c, lw=2.0, zorder=3)
        for b in cint:
            ax.plot([b * 100, b * 100], [i - 0.12, i + 0.12], color=c, lw=2.0, zorder=3)
        txt = f'{prb * 100:7.2f}   [{cint[0] * 100:6.2f}, {cint[1] * 100:6.2f}]'
    else:
        txt = f'{prb * 100:7.2f}   no CI (prior record)'
    ax.plot(prb * 100, i, marker='o', ms=11 if used else 8,
            mfc=c if cint is not None else 'white',
            mec='black' if used else c, mew=1.0, ls='none', zorder=4)
    ax.text(TEXT_X, i, txt, va='center', ha='left', fontsize=9, family='monospace',
            color='black' if used else '0.25')

ax.axvline(0, color=SEMANTIC['zero'], ls=LINESTYLE['zero'], lw=0.8, zorder=1)
ax.text(0.5, 5.8, 'unbiased', fontsize=9, color=SEMANTIC['zero'], ha='left', va='center')
ax.annotate('under-forecast', xy=(-7.0, 5.8), xytext=(-0.6, 5.8), fontsize=9,
            color=SEMANTIC['zero'], ha='right', va='center',
            arrowprops=dict(arrowstyle='->', color=SEMANTIC['zero'], lw=0.9))
ax.text(TEXT_X, -0.8, 'PRB (%)   95% CI', fontsize=9, family='monospace',
        color='0.25', ha='left', va='center')
ax.annotate('definition used for the gate-1 verdict',
            xy=(ROWS[4][1] * 100, 4), xytext=(-14, 4.75), fontsize=9, color=SEMANTIC['annotation'],
            ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color=SEMANTIC['annotation'], lw=1.0))

ax.set_yticks(range(len(ROWS)), [r[0] for r in ROWS])
ax.set_ylim(len(ROWS) + 0.2, -1.05)
ax.set_xlim(-34, 17)
ax.set_xticks([-30, -25, -20, -15, -10, -5, 0])
ax.set_xlabel('Pooled relative bias (%), negative = under-forecast')
ax.grid(axis='y', visible=False)
ax.set_title('Point extraction sets the measured under-forecast: '
             f"{ROWS[0][1] * 100:.2f}% (median) to {ROWS[5][1] * 100:.2f}% (QMEAN, linear tail)\n"
             f'same Chronos-2 zero-shot quantiles, {N_SERIES:,} series, '
             'series bootstrap 2,000 draws')
save(fig, 'point_extraction_bias.png', 'ci.json, precheck_a.json')


# ── Figure 3 — gate 2-B headroom ────────────────────────────────────────────
G2B = [
    ('No covariates, 4 train origins',   g2b['g2b']),
    ('With covariates, 4 train origins', g2b['g2b_full']),
    ('With covariates, 12 train origins', g2b['g2b_full12']),
]
THRESH = 1.0

fig, ax = plt.subplots(figsize=(10.5, 5.2))
c_bar = color_for('ref-lgb')
for i, (_lab, d) in enumerate(G2B):
    ax.barh(i, d['delta_pct'], height=0.42, color=c_bar, zorder=3)
    ax.plot([d['ci'][0], d['ci'][1]], [i, i], color='black', lw=1.4, zorder=4)
    for b in d['ci']:
        ax.plot([b, b], [i - 0.08, i + 0.08], color='black', lw=1.4, zorder=4)
    ax.text(d['ci'][1] + 0.03, i,
            f"{d['delta_pct']:+.2f}%  CI [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]",
            va='center', ha='left', fontsize=9, family='monospace')

ax.axvline(0, color=SEMANTIC['zero'], ls=LINESTYLE['zero'], lw=0.8, zorder=2)
ax.axvline(THRESH, color=SEMANTIC['threshold'], ls=LINESTYLE['threshold'], lw=1.6, zorder=5)
ax.text(THRESH + 0.02, -0.78, f'pass threshold +{THRESH:.1f}%', color=SEMANTIC['threshold'],
        fontsize=9.5, ha='left', va='center')
best = G2B[-1][1]['delta_pct']
ax.annotate('', xy=(THRESH, 2.45), xytext=(best, 2.45),
            arrowprops=dict(arrowstyle='<->', color=SEMANTIC['annotation'], lw=1.3))
ax.text((best + THRESH) / 2, 2.58, f'{THRESH - best:.2f} pp short of the threshold',
        ha='center', va='top', fontsize=9.5, color=SEMANTIC['annotation'])

ax.set_yticks(range(len(G2B)),
              [f"{lab}\nRMSSE {d['rmsse_zs']:.4f} -> {d['rmsse_lgb']:.4f}" for lab, d in G2B])
ax.set_ylim(2.85, -1.05)
ax.set_xlim(-0.18, 1.42)
ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
ax.set_xlabel('RMSSE improvement of target-trained LightGBM over zero-shot Chronos-2 (%)')
ax.grid(axis='y', visible=False)
ax.set_title('Gate 2-B adaptation headroom: every setting falls short of the +1.0% pass threshold\n'
             f'best setting +{best:.2f}%, both arms scaled by their own '
             rf"$\alpha^*$; {N_SERIES:,} series, series bootstrap 2,000 draws")
save(fig, 'gate2b_headroom.png', 'g2b.json, g2b_full.json, g2b_full12.json')


# ── Figure 4 — PRB by SBC regime ────────────────────────────────────────────
A = pre_a['prb']['A_spec_const']
REG = ['smooth', 'erratic', 'intermittent', 'lumpy']
vals = [A[f'regime_{r}'] * 100 for r in REG]
ns = [A[f'n_{r}'] for r in REG]
pooled = A['pooled_all'] * 100
CREG = dict(zip(REG, colors_for(REG)))

fig, ax = plt.subplots(figsize=(9.5, 6.0))
ax.bar(range(len(REG)), vals, width=0.6, color=[CREG[r] for r in REG], zorder=3)
for i, (v, n) in enumerate(zip(vals, ns)):
    off = 0.12 if v >= 0 else -0.12
    ax.text(i, v + off, f'{v:+.2f}%', ha='center',
            va='bottom' if v >= 0 else 'top', fontsize=10)

ax.axhline(0, color=SEMANTIC['zero'], ls=LINESTYLE['zero'], lw=0.8, zorder=2)
ax.axhline(pooled, color=SEMANTIC['mean'], ls=LINESTYLE['median'], lw=1.2, zorder=2,
           label=f'pooled over all series {pooled:+.2f}% (n = {N_SERIES:,})')
il = A['pooled_intermittent_lumpy'] * 100
ax.axhline(il, color=SEMANTIC['baseline'], ls=LINESTYLE['baseline'], lw=1.2, zorder=2,
           label=f'pooled intermittent + lumpy {il:+.2f}% '
                 f"(n = {A['n_intermittent'] + A['n_lumpy']:,})")

G1III_CI_HI = -0.0108   # FINAL_DECISION.md 의 G1(iii) 부트스트랩 CI 상한 (ci.json 미수록)
ax.text(0.985, 0.03,
        'gate 1 (iii)\n'
        f"(intermittent + lumpy) - smooth = {(A['pooled_intermittent_lumpy'] - A['regime_smooth']):+.4f}\n"
        f'CI upper {G1III_CI_HI:+.4f}  (FINAL_DECISION.md)',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85), zorder=6)

ax.set_xticks(range(len(REG)), [f'{r}\nn = {n:,}' for r, n in zip(REG, ns)])
ax.set_ylabel('Pooled relative bias (%), negative = under-forecast')
ax.set_ylim(-4.6, 1.5)
ax.grid(axis='x', visible=False)
ax.set_title('Under-forecast concentrates in intermittent demand, smooth series are over-forecast\n'
             'QMEAN trapezoid with constant tail (the definition used for the verdict), '
             'SBC regimes from d_1-d_1885')
ax.legend(loc='lower left', fontsize=9)
save(fig, 'prb_by_regime.png', 'precheck_a.json, gonogo.json')
