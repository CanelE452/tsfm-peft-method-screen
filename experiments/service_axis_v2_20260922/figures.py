"""REPORT_KO.md 임베드용 그림 3장. 수치는 전부 results/ 의 산출물에서 읽는다 (재계산 없음).

python experiments/service_axis_v2_20260922/figures.py
"""
import os, sys, csv, json
from datetime import datetime
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/service_axis_v2_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ASSETS = Path.home() / '.claude/agents/viz-expert/assets'
plt.style.use(ASSETS / 'analysis.mplstyle')
sys.path.insert(0, str(ASSETS))
from palette import color_for, colors_for, SEMANTIC, LINESTYLE, ALPHA  # noqa: E402

ROOT = Path('/home/minjae/Documents/github/tsfm-peft-method-screen')
OUT = ROOT / 'results/service_axis_v2_20260922'
SRC = 'results/service_axis_v2_20260922/'
STAMP = datetime.now()

DEC = json.load(open(OUT / 'DECISION.json'))
LAD = json.load(open(OUT / 'LADDER_V2.json'))
PRB = json.load(open(OUT / 'PRB_BY_WINDOW.json'))
TH = json.load(open(OUT / 'THRESHOLDS.json'))
SEAL = json.load(open(OUT / 'SEAL.json'))
T_STOCK = TH['T_stock']


def win_labels(csv_name):
    rows = list(csv.DictReader(open(OUT / csv_name)))
    return [r['window'].replace('_', '') for r in rows if r['window'].startswith('d_')]


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


MONO = {'family': 'monospace', 'fontsize': 9}
BOX = dict(boxstyle='round,pad=0.45', facecolor='white', edgecolor='0.7', alpha=0.92)


# ── Figure 1 — per-window decision deltas (gate V3 and gate V1) ──────────────
labels = win_labels('DECISION_V1.csv')
v3, v1 = DEC['V3'], DEC['V1']
d3 = np.array(v3['delta'], float)
d1 = np.array(v1['gap'], float)
c3, c1 = color_for('V3'), color_for('V1')

fig, ax = plt.subplots()
x = np.arange(len(labels))
w = 0.38
b3 = ax.bar(x - w / 2, d3, w, color=c3, edgecolor='white', linewidth=0.6,
            label=f"Gate V3  trained REF-LGB-U  vs  best training-free {v3['baseline']}")
b1 = ax.bar(x + w / 2, d1, w, color=c1, edgecolor='white', linewidth=0.6,
            label=f"Gate V1  zero-shot {v3['baseline']}  vs  best classical {v1['baseline']}")
for bars, vals in ((b3, d3), (b1, d1)):
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v - 0.35, f'{v:+.2f}',
                ha='center', va='top', fontsize=8, color='0.25')

ax.axhline(0, color=SEMANTIC['zero'], ls=LINESTYLE['zero'], lw=1.0, zorder=1)
ax.axhline(T_STOCK, color=SEMANTIC['threshold'], ls=LINESTYLE['threshold'], lw=1.4,
           label=f'pass threshold  T_stock = +{T_STOCK:.1f}%')
ax.axhspan(T_STOCK, 8.0, color=SEMANTIC['threshold'], alpha=0.07, zorder=0)
ax.text(len(labels) - 0.55, (T_STOCK + 7.4) / 2, 'pass region (no window reaches it)',
        ha='right', va='center', fontsize=9, color=SEMANTIC['threshold'])
ax.axhline(v3['mean'], color=c3, ls='--', lw=1.1, alpha=0.9)
ax.axhline(v1['mean'], color=c1, ls='--', lw=1.1, alpha=0.9)
ax.text(len(labels) - 0.35, v3['mean'] + 0.22, f"V3 mean {v3['mean']:+.2f}", color=c3,
        fontsize=9, va='bottom', ha='right')
ax.text(len(labels) - 0.35, v1['mean'] - 0.22, f"V1 mean {v1['mean']:+.2f}", color=c1,
        fontsize=9, va='top', ha='right')

ax.set_xticks(x)
ax.set_xticklabels([l.replace('-', '–') for l in labels], rotation=18, ha='right')
ax.set_xlim(-0.75, len(labels) - 0.25)
ax.set_ylim(-17.2, 8.0)
ax.set_yticks([-15, -10, -5, 0, 5])
ax.set_xlabel('EVAL window (sealed before scoring, 28 days each)')
ax.set_ylabel('Δ in stock held at 90% unit fill rate  (%)')
ax.set_title('Per-window decision deltas on the 8 EVAL windows — '
             f"0/8 positive in both gates  (V3 mean {v3['mean']:+.2f}%, V1 mean {v1['mean']:+.2f}%)")
ax.legend(loc='upper left', fontsize=9)

sign = ('Sign convention differs between the two gates\n'
        f"  V3  Δ = 100 · (I@90[{v3['baseline']}] − I@90[REF-LGB-U]) / I@90[{v3['baseline']}]\n"
        '        Δ > 0  the trained model reaches 90% fill with LESS stock\n'
        '        Δ < 0  the trained model needs MORE stock  ← all 8 windows\n'
        f"  V1  Δ = 100 · (I@90[{v3['baseline']}] − I@90[{v1['baseline']}]) / I@90[{v1['baseline']}]\n"
        '        Δ > 0  zero-shot TSFM needs MORE stock  = headroom PEFT could close\n'
        '        Δ < 0  zero-shot TSFM already needs LESS stock  ← all 8 windows')
ax.text(0.505, 0.455, sign, transform=ax.transAxes, va='top', ha='left',
        fontsize=8.6, color='0.15', bbox=BOX)

stats = ('gate   mean      sd      t_low   pos/8   verdict\n'
         f"V3    {v3['mean']:+6.2f}  {v3['sd']:6.2f}  {v3['t_low']:+7.2f}    {v3['n_pos']}/{v3['n']}    "
         f"{'PASS' if v3['pass'] else 'FAIL'}\n"
         f"V1    {v1['mean']:+6.2f}  {v1['sd']:6.2f}  {v1['t_low']:+7.2f}    {v1['n_pos']}/{v1['n']}    "
         f"{'PASS' if v1['pass'] else 'FAIL'}\n"
         f'pass rule: mean ≥ +{T_STOCK:.1f}%  and  t_low > 0  and  ≥ 7/8 windows positive')
ax.text(0.505, 0.175, stats, transform=ax.transAxes, va='top', ha='left',
        color='0.15', bbox=BOX, **MONO)

ax.text(0.004, 0.015,
        'Reading: the two gates disagree in what a negative bar means, yet both are negative in every window — '
        'the trained model never buys stock back, and the zero-shot TSFM was never behind the classical baseline to begin with.',
        transform=ax.transAxes, fontsize=9, color='0.25')
save(fig, 'decision_windows.png', 'DECISION.json, DECISION_V1.csv, DECISION_V3.csv, THRESHOLDS.json')


# ── Figure 2 — where the v1 +43.78% gap went (ladder L0 -> L6) ───────────────
rows = LAD['rows']
steps = [r['step'] for r in rows]
gaps = np.array([r['gap_pct'] for r in rows], float)
NOTE_EN = {
    'L0': 'v1 as-is\n(ZS-best = ZS-Q90)',
    'L1': '+ unit-weighted\nfill and stock',
    'L2': '+ sigma clip\n(no negative quantiles)',
    'L3': '+ QMEAN21\npoint forecast',
    'L4': '+ alpha by\nSBC class',
    'L5': 'ERP policy\n(L=7, R=1, no alpha)',
    'L6': 'EVAL 8-window\nmean',
}
drop = gaps[0] - gaps[1]
tail = gaps[1:]
span = tail.max() - tail.min()

fig, ax = plt.subplots()
cols = [color_for('before')] + [color_for('after')] * (len(steps) - 1)
bars = ax.bar(np.arange(len(steps)), gaps, 0.6, color=cols, edgecolor='white', linewidth=0.8)
for k, (r, v) in enumerate(zip(bars, gaps)):
    off = 1.1 if v > 0 else -1.1
    ax.text(r.get_x() + r.get_width() / 2, v + off, f'{v:+.2f}%',
            ha='center', va='bottom' if v > 0 else 'top', fontsize=10, color='0.15')

ax.axhline(0, color=SEMANTIC['zero'], ls=LINESTYLE['zero'], lw=1.0, zorder=1)
ax.axhspan(tail.min(), tail.max(), xmin=0.10, color=color_for('after'), alpha=ALPHA['range'], zorder=0)

ax.annotate('', xy=(0.55, gaps[0]), xytext=(0.55, gaps[1]),
            arrowprops=dict(arrowstyle='<->', color=SEMANTIC['annotation'], lw=1.6))
ax.text(0.68, (gaps[0] + gaps[1]) / 2,
        f'{-drop:+.2f} %p in a single rung\nequal-weighted average of per-series stock\n'
        '→ unit-weighted (total served / total demand, total stock)',
        va='center', ha='left', fontsize=10, color='0.15', bbox=BOX)
ax.annotate(f'the five later corrections change almost nothing:\n'
            f'L1–L6 all inside a {span:.2f} %p band  ({tail.min():+.2f}% to {tail.max():+.2f}%)',
            xy=(5.55, tail.min()), xytext=(3.55, 9.0), fontsize=10, color='0.15', va='center',
            arrowprops=dict(arrowstyle='->', color=SEMANTIC['annotation'], lw=1.3,
                            connectionstyle='arc3,rad=-0.2'), bbox=BOX)

ax.set_xticks(np.arange(len(steps)))
ax.set_xticklabels([f'{s}\n{NOTE_EN[s]}' for s in steps], fontsize=9)
ax.set_xlim(-0.6, len(steps) - 0.4)
ax.set_ylim(-9.5, 50)
ax.set_yticks([0, 10, 20, 30, 40])
ax.set_xlabel('ladder rung — one factor changed per rung, cumulative from left to right')
ax.set_ylabel('Gap = 100 · (stock[ZS] − stock[NZ]) / stock[ZS]   (%)\n'
              '+ = zero-shot needs more stock')
ax.set_title('Where the +43.78% gap of v1 went — it is spent at the first rung, '
             f'{-drop:+.2f} %p from L0 to L1')
ax.text(0.004, 0.015,
        f'Reading: the gap that motivated the PEFT headroom claim is an artefact of the v1 metric weighting; '
        f'once the stock metric is unit-weighted it is gone ({gaps[1]:+.2f}%), and the five later corrections '
        f'move it by at most {span:.2f} %p in total.',
        transform=ax.transAxes, fontsize=9, color='0.25')
save(fig, 'ladder_v2.png', 'LADDER_V2.json, LADDER_V2.csv')


# ── Figure 3 — per-window point-forecast relative bias ───────────────────────
wins = PRB['windows']
n_calb = len(SEAL['windows']['CAL_B'])
summ = PRB['summary']
arms = sorted(PRB['prb'], key=lambda a: summ[a]['mean'])          # most negative on top
FAMILY = {'ZS-MED': 'zero-shot TSFM', 'ZS-Q': 'zero-shot TSFM', 'NZ-Q': 'non-zero normalised',
          'SNAIVE': 'classical', 'SES': 'classical', 'SBA': 'classical', 'TSB': 'classical',
          'REF-LGB-U': 'trained reference'}
fams = ['zero-shot TSFM', 'non-zero normalised', 'classical', 'trained reference']
CFAM = dict(zip(fams, colors_for(fams)))
DESC = {'ZS-MED': 'median extraction', 'ZS-Q': 'QMEAN21 extraction', 'NZ-Q': 'QMEAN21 extraction',
        'SNAIVE': '', 'SES': '', 'SBA': '', 'TSB': '', 'REF-LGB-U': 'LightGBM, trained'}

XLO, XHI, YLO = -0.375, 0.575, -2.55   # 오른쪽은 값 표, 아래는 범례·주석 몫으로 비워 둔다
X_SEP = 0.262          # 데이터 영역과 값 표 영역의 경계
COL = {'mean': 0.360, 'neg': 0.452, 'flag': 0.532}

fig, ax = plt.subplots(figsize=(14, 7.4))
for i, arm in enumerate(arms):
    y = len(arms) - 1 - i
    v = np.array(PRB['prb'][arm], float)
    c = CFAM[FAMILY[arm]]
    m = summ[arm]['mean']
    if summ[arm]['systematic']:
        ax.axhspan(y - 0.46, y + 0.46, xmax=(X_SEP - XLO) / (XHI - XLO),
                   color=c, alpha=0.12, zorder=0)
    ax.plot([0, m], [y, y], color=c, lw=3.0, alpha=0.45, solid_capstyle='butt', zorder=2)
    dy = (np.arange(len(v)) / (len(v) - 1) - 0.5) * 0.52
    ax.scatter(v[:n_calb], y + dy[:n_calb], s=36, facecolors='white', edgecolors=c,
               linewidths=1.3, zorder=3)
    ax.scatter(v[n_calb:], y + dy[n_calb:], s=36, color=c, zorder=3)
    ax.plot([m, m], [y - 0.36, y + 0.36], color='0.15', lw=1.6, zorder=4)
    ax.plot([m], [y], marker='D', ms=6.5, color='0.15', zorder=5)
    flag = 'YES' if summ[arm]['systematic'] else 'no'
    ax.text(COL['mean'], y, f'{m:+.4f}', ha='right', va='center', color='0.15', **MONO)
    ax.text(COL['neg'], y, f"{summ[arm]['n_neg']:2d} / 12", ha='right', va='center', color='0.15', **MONO)
    ax.text(COL['flag'], y, flag, ha='center', va='center',
            color=SEMANTIC['threshold'] if flag == 'YES' else '0.45', **MONO)

hy = len(arms) - 0.38
ax.text(COL['mean'], hy, 'PRB mean', ha='right', va='bottom', color='0.35', **MONO)
ax.text(COL['neg'], hy, 'neg / 12', ha='right', va='bottom', color='0.35', **MONO)
ax.text(COL['flag'], hy, 'systematic', ha='center', va='bottom', color='0.35', **MONO)
ax.axvline(X_SEP, color='0.8', lw=1.0, ymin=0.14, zorder=1)

ax.axvline(0, color=SEMANTIC['zero'], ls=LINESTYLE['zero'], lw=1.2, zorder=1)
ax.text(0.006, hy, 'unbiased', fontsize=9, color='0.35', va='bottom')
ax.set_yticks(np.arange(len(arms)))
ax.set_yticklabels([f'{a}\n{DESC[a]}' if DESC[a] else a for a in arms[::-1]], fontsize=9.5)
ax.set_ylim(YLO, len(arms) - 0.05)
ax.set_xlim(XLO, XHI)
ax.set_xticks([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20])
ax.set_xticks([-0.35, -0.25, -0.15, -0.05, 0.05, 0.15], minor=True)
ax.set_xlabel('PRB  =  Σ (forecast − actual) / Σ actual        negative = under-forecast')
ax.set_title(f'Point-forecast relative bias, one dot per window '
             f'(CAL-B {n_calb} + EVAL {len(wins) - n_calb}, origins o={wins[0]}–{wins[-1]})')

h = [plt.Line2D([], [], ls='', marker='o', mfc='white', mec='0.35', ms=6.5, label=f'CAL-B window ({n_calb})'),
     plt.Line2D([], [], ls='', marker='o', color='0.35', ms=6.5, label=f'EVAL window ({len(wins) - n_calb})'),
     plt.Line2D([], [], ls='', marker='D', color='0.15', ms=6.5, label='arm mean (bar = mean vs 0)')]
h += [plt.Line2D([], [], ls='', marker='s', color=CFAM[f], ms=7, label=f) for f in fams]
ax.legend(handles=h, loc='lower left', bbox_to_anchor=(0.0, 0.095), fontsize=9, ncol=4)

ax.text(0.004, 0.060,
        f"Reading: systematic under-forecast belongs to the median extraction alone — ZS-MED {summ['ZS-MED']['mean']:+.4f}, "
        f"{summ['ZS-MED']['n_neg']}/12 windows negative.",
        transform=ax.transAxes, fontsize=9.5, color='0.25')
ax.text(0.004, 0.035,
        f"The same model read with the quantile-mean point forecast is ZS-Q {summ['ZS-Q']['mean']:+.4f} "
        f"({summ['ZS-Q']['n_neg']}/12) — the bias is an extraction artefact, not a model property.",
        transform=ax.transAxes, fontsize=9.5, color='0.25')
ax.text(0.004, 0.008,
        'Flag rule: ≥ 10 of 12 windows negative and upper t-bound < 0.   '
        'Dot height inside a row encodes window order, oldest at the bottom.',
        transform=ax.transAxes, fontsize=9, color='0.45')
save(fig, 'prb_by_window.png', 'PRB_BY_WINDOW.json, PRB_BY_WINDOW.csv, SEAL.json')
