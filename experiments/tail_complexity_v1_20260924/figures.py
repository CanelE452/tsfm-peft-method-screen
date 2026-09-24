#!/usr/bin/env python
"""Figures for the tail-complexity validity study (contract section 9).

Visualization only: this script reads the saved result CSVs and never imports
experiment, data or model code. Every plotted coordinate and every caption
number is computed from those CSVs and exported to FIGURE_VALUES.csv, so a
reader can recompute any mark in a figure without rerunning the study.
One figure is one file; no subplots are used.

    python figures.py [--results DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

# --------------------------------------------------------------------- style
# rcParams copied from
# hierarchical-tsfm-peft/results/internal_adaptation_gap_v1_20260922/paper_figures/generate_figures.py
# so that both studies produce the same looking pages; only svg.hashsalt differs.
# mathtext.fontset is the single addition: the axis labels carry Greek symbols,
# and the Matplotlib default would typeset them in a sans-serif face.
FONT = next(n for n in ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
            if n in {f.name for f in font_manager.fontManager.ttflist})
mpl.rcParams.update({'font.family': FONT, 'font.size': 8, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': .55, 'xtick.major.width': .5, 'ytick.major.width': .5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'svg.hashsalt': 'tail-complexity-v1',
    'mathtext.fontset': 'stix'})

# Colours follow the viz-expert single source of truth
# (~/.claude/agents/viz-expert/assets/palette.py, Okabe-Ito). They are copied as
# literals because this script may only import matplotlib, numpy and pandas.
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7']
ARM_COLOR = {'F0': '#0072B2',     # palette REGISTRY['baseline'] - the untouched arm
             'LORA': '#D55E00'}   # palette REGISTRY['ours']     - the adapted arm
ARM_MARKER = {'F0': 'o', 'LORA': '^'}
ARM_LINE = {'F0': '-', 'LORA': '--'}
ARM_LABEL = {'F0': 'F0 (zero-shot)', 'LORA': 'LoRA'}
ZERO_COLOR = '#444444'
GRID_COLOR = '#E6E6E6'
MEAN_COLOR = '#222222'

# ---------------------------------------------------------------- vocabulary
ARMS = ['F0', 'LORA']
TAU_MAIN = 0.995
EXT_MAIN = 'SEL'
EPS_RANK = {'gauss': 0, 't6': 1, 't3.5': 2}
EPS_LABEL = {'gauss': 'Gaussian', 't6': 't(6)', 't3.5': 't(3.5)'}
GARCH_LABEL = {'off': 'GARCH off', 'on': 'GARCH on'}
GAP_LABEL = r'$G^{*}_{.995}$   ($\mathrm{QS}_{0.995}/\mathrm{QS}_{\mathrm{ref}} - 1$)'
XI_LABEL = r'Estimated tail index  $\hat{\xi}$  (GPD shape, TRAIN)'
DSE_LABEL = r'$\Delta$ spectral entropy  (matched $-$ baseline)'
DXI_LABEL = r'$\Delta\,\hat{\xi}$  (matched $-$ baseline)'
UNIT_GAP = 'gap ratio (dimensionless)'
UNIT_XI = 'GPD shape (dimensionless)'
UNIT_SE = 'normalized spectral entropy (0-1)'

FIG1 = 'fig1_s1_condition_gap'
FIG2A = 'fig2_xi_vs_gap_s1'
FIG2B = 'fig2_xi_vs_gap_s2'
FIG3 = 'fig3_matched_pairs_se_xi'

REQUIRED = {
    'S1_DIFFICULTY.csv': ['condition', 'series', 'phi', 'eps', 'garch', 'xi', 'theta', 'se'],
    'S1_SCORES.csv': ['arm', 'condition', 'series', 'tau', 'extension', 'gap'],
    'S1_PAIRS.csv': ['phi', 'cond_a', 'cond_b', 'series', 'dSE', 'dxi', 'se_a', 'se_b', 'xi_a', 'xi_b'],
    'S2_DIFFICULTY.csv': ['symbol', 'xi', 'theta', 'se'],
    'S2_SCORES.csv': ['arm', 'symbol', 'tau', 'extension', 'gap', 'gap_vs_garch_t'],
}
BOOT_DRAWS = 2000
BOOT_SEED = 20260924


class DataError(Exception):
    """Raised when the saved results cannot support the contracted figures."""


def fail(message):
    raise DataError(message)


# ------------------------------------------------------------------ loading
def load_inputs(results: Path) -> dict:
    """Read and check every input before a single output file is written."""
    if not results.is_dir():
        fail(f'results directory not found: {results}')
    missing = [name for name in REQUIRED if not (results/name).is_file()]
    if missing:
        fail('missing result files in {}: {}  (run the S1/S2 scoring steps first)'
             .format(results, ', '.join(sorted(missing))))
    frames = {}
    for name, columns in REQUIRED.items():
        frame = pd.read_csv(results/name, float_precision='round_trip')
        absent = [c for c in columns if c not in frame.columns]
        if absent:
            fail(f'{name} is missing required columns: {", ".join(absent)}')
        if frame.empty:
            fail(f'{name} has a header but no rows')
        frames[name] = frame

    s1d = frames['S1_DIFFICULTY.csv']
    meta = s1d[['condition', 'phi', 'eps', 'garch']].drop_duplicates()
    if meta.condition.duplicated().any():
        fail('S1_DIFFICULTY.csv maps one condition to several phi/eps/garch settings')
    meta = meta.set_index('condition')
    order = sorted(meta.index, key=lambda c: (meta.at[c, 'phi'],
                                              EPS_RANK.get(meta.at[c, 'eps'], 99),
                                              meta.at[c, 'garch'] == 'on', c))

    s1 = main_gap(frames['S1_SCORES.csv'], ['arm', 'condition', 'series'], 'S1_SCORES.csv')
    s1 = s1.merge(s1d[['condition', 'series', 'phi', 'eps', 'garch', 'xi', 'theta', 'se']],
                  on=['condition', 'series'], how='inner')
    if s1.empty:
        fail('no S1 series matched between S1_SCORES.csv and S1_DIFFICULTY.csv on (condition, series)')
    unknown = sorted(set(s1.condition) - set(order))
    if unknown:
        fail(f'S1_SCORES.csv refers to conditions absent from S1_DIFFICULTY.csv: {", ".join(unknown)}')

    s2 = main_gap(frames['S2_SCORES.csv'], ['arm', 'symbol'], 'S2_SCORES.csv')
    s2 = s2.merge(frames['S2_DIFFICULTY.csv'][['symbol', 'xi', 'theta', 'se']],
                  on='symbol', how='inner')
    if s2.empty:
        fail('no S2 symbol matched between S2_SCORES.csv and S2_DIFFICULTY.csv on symbol')

    pairs = frames['S1_PAIRS.csv'].copy()
    for side in ['cond_a', 'cond_b']:
        unknown = sorted(set(pairs[side]) - set(meta.index))
        if unknown:
            fail(f'S1_PAIRS.csv {side} refers to unknown conditions: {", ".join(unknown)}')
    pairs['contrast'] = [f'{meta.at[c, "eps"]}_{meta.at[c, "garch"]}' for c in pairs.cond_b]
    pairs['baseline'] = [f'{meta.at[c, "eps"]}_{meta.at[c, "garch"]}' for c in pairs.cond_a]
    if pairs.baseline.nunique() != 1:
        fail('S1_PAIRS.csv uses more than one baseline condition per matched pair; '
             'the delta figure assumes a single baseline (contract section 7, H2)')

    dropped = {}
    s1, dropped['S1'] = drop_incomplete(s1, ['xi', 'gap'], 'S1')
    s2, dropped['S2'] = drop_incomplete(s2, ['xi', 'gap'], 'S2')
    pairs, dropped['S1_PAIRS'] = drop_incomplete(pairs, ['dSE', 'dxi'], 'S1_PAIRS')
    return {'s1': s1, 's2': s2, 'pairs': pairs, 'meta': meta, 'order': order,
            'dropped': dropped, 'results': results}


def main_gap(scores: pd.DataFrame, keys: list, source: str) -> pd.DataFrame:
    """The primary metric of the contract: G*_.995 = gap at tau=0.995 with the SEL extension."""
    sub = scores[np.isclose(scores.tau.astype(float), TAU_MAIN) & (scores.extension == EXT_MAIN)]
    if sub.empty:
        fail(f'{source} has no rows with tau={TAU_MAIN} and extension={EXT_MAIN!r}; '
             'the primary metric G*_.995 cannot be formed')
    absent = [a for a in ARMS if a not in set(sub.arm)]
    if absent:
        fail(f'{source} has no tau={TAU_MAIN} / {EXT_MAIN} rows for arm(s): {", ".join(absent)}')
    sub = sub[sub.arm.isin(ARMS)]
    if sub.duplicated(keys).any():
        dup = sub[sub.duplicated(keys, keep=False)].head(3)[keys].to_dict('records')
        fail(f'{source} has duplicate rows for {keys} at tau={TAU_MAIN}/{EXT_MAIN}, e.g. {dup}')
    return sub.copy()


def drop_incomplete(frame: pd.DataFrame, columns: list, label: str):
    keep = frame[columns].notna().all(axis=1)
    dropped = int((~keep).sum())
    frame = frame[keep].copy()
    if len(frame) < 3:
        fail(f'{label} keeps only {len(frame)} rows with finite {columns}; nothing to plot')
    return frame, dropped


# --------------------------------------------------------------- statistics
def spearman(x, y) -> float:
    return pearson(pd.Series(np.asarray(x, float)).rank().to_numpy(),
                   pd.Series(np.asarray(y, float)).rank().to_numpy())


def pearson(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def ols_line(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def boot_mean_ci(values):
    values = np.asarray(values, float)
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(values), size=(BOOT_DRAWS, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def fmt(value, digits=3) -> str:
    return 'n/a' if not np.isfinite(value) else f'{value:.{digits}f}'


# ------------------------------------------------------------------ helpers
VALUES = []


def record(figure, panel, series, x, y, unit, source_file):
    VALUES.append({'figure': figure, 'panel': panel, 'series': series,
                   'x': float(x), 'y': float(y), 'unit': unit, 'source_file': source_file})


def condition_label(meta, condition, with_phi=True):
    row = meta.loc[condition]
    head = rf'$\varphi$ = {row.phi:g}, ' if with_phi else ''
    return (head + EPS_LABEL.get(row.eps, str(row.eps)) + ', '
            + GARCH_LABEL.get(row.garch, str(row.garch)))


def axes_style(ax, grid_axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=.4)
    ax.yaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))


def export(fig, outdir: Path, name: str):
    fig.canvas.draw()
    for ext in ['pdf', 'svg', 'png']:
        meta = ({'Description': name} if ext == 'png' else
                {'Title': name, 'Creator': 'Matplotlib; saved-result visualization only'})
        if ext == 'pdf':
            meta.update(CreationDate=None, ModDate=None)
        elif ext == 'svg':
            meta['Date'] = None
        fig.savefig(outdir/f'{name}.{ext}', dpi=600, metadata=meta)
    path = outdir/f'{name}.svg'
    path.write_text('\n'.join(line.rstrip() for line in
                              path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)


# ------------------------------------------------------------------ figure 1
def figure1(data, outdir):
    s1, meta, order = data['s1'], data['meta'], data['order']
    fig, ax = plt.subplots(figsize=(7, 2.95))
    fig.subplots_adjust(left=.085, right=.985, bottom=.265, top=.87)
    width = .38
    stats = {}
    for k, arm in enumerate(ARMS):
        xs, med, low, high = [], [], [], []
        for i, condition in enumerate(order):
            values = s1[(s1.arm == arm) & (s1.condition == condition)].gap.to_numpy()
            if values.size == 0:
                fail(f'no G*_.995 rows for arm {arm} in condition {condition}')
            q25, q50, q75 = (float(q) for q in np.quantile(values, [.25, .5, .75]))
            x = i + (k - .5)*width
            xs.append(x)
            med.append(q50)
            low.append(max(q50 - q25, 0.))
            high.append(max(q75 - q50, 0.))
            for tag, value in [('median', q50), ('q25', q25), ('q75', q75), ('n_series', len(values))]:
                unit = ('x=condition index; y=series count' if tag == 'n_series'
                        else f'x=condition index; y={UNIT_GAP}')
                record(FIG1, 'main', f'{arm}|{condition}|{tag}', x, value, unit,
                       'S1_SCORES.csv + S1_DIFFICULTY.csv')
            stats[(arm, condition)] = (q25, q50, q75, len(values))
        ax.bar(xs, med, width=width*.92, color=ARM_COLOR[arm], edgecolor='white',
               linewidth=.3, zorder=2)
        ax.errorbar(xs, med, yerr=[low, high], fmt='none', ecolor='#333333',
                    elinewidth=.6, capsize=1.6, capthick=.6, zorder=3)
    ax.axhline(0, color=ZERO_COLOR, lw=.7, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([EPS_LABEL.get(meta.at[c, 'eps'], meta.at[c, 'eps']) + '\n'
                        + GARCH_LABEL.get(meta.at[c, 'garch'], meta.at[c, 'garch']) for c in order])
    ax.tick_params(axis='x', labelsize=6.8, length=0)
    ax.set_xlim(-.6, len(order) - .4)
    ax.set_ylabel(GAP_LABEL)
    ax.set_xlabel('Synthetic condition: innovation distribution and volatility switch', labelpad=4)
    axes_style(ax)
    blended = ax.get_xaxis_transform()
    for group, (phi, indices) in enumerate(group_by_phi(meta, order)):
        if group:
            ax.axvline(indices[0] - .5, color='#BBBBBB', lw=.6, ls=(0, (2, 2)), zorder=1)
        ax.text(float(np.mean(indices)), 1.015, rf'$\varphi$ = {phi:g}', transform=blended,
                ha='center', va='bottom', fontsize=8)
    handles = [Line2D([], [], color=ARM_COLOR[a], lw=5, label=ARM_LABEL[a]) for a in ARMS]
    handles.append(Line2D([], [], color='#333333', lw=.8, marker='_', ms=4,
                          label='Interquartile range over series (bar: median)'))
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.53, .012))
    export(fig, outdir, FIG1)
    return stats


def group_by_phi(meta, order):
    groups = []
    for i, condition in enumerate(order):
        phi = float(meta.at[condition, 'phi'])
        if groups and groups[-1][0] == phi:
            groups[-1][1].append(i)
        else:
            groups.append((phi, [i]))
    return groups


# ------------------------------------------------------------------ figure 2
def scatter_fit(ax, figure, frame, arm, colours, source, line_colour=MEAN_COLOR, size=2.6):
    """Draw one arm of a xi-hat vs G*_.995 scatter plus its least-squares line."""
    sub = frame[frame.arm == arm]
    if sub.empty:
        fail(f'{source}: arm {arm} has no rows at tau={TAU_MAIN}/{EXT_MAIN}')
    filled = arm == 'F0'
    for row in draw_order(sub).itertuples():
        colour = colours(row)
        ax.plot(row.xi, row.gap, ARM_MARKER[arm], ms=size, mew=.45,
                mec=colour, mfc=colour if filled else 'none', alpha=.85, zorder=2)
    for row in sub.itertuples():
        record(figure, arm, f'{arm}|{key_of(row)}', row.xi, row.gap,
               f'x={UNIT_XI}; y={UNIT_GAP}', source)
    slope, intercept = ols_line(sub.xi, sub.gap)
    xs = np.array([sub.xi.min(), sub.xi.max()], float)
    ax.plot(xs, slope*xs + intercept, ls=ARM_LINE[arm], color=line_colour, lw=.9, zorder=5)
    for x, y in zip(xs, slope*xs + intercept):
        record(figure, arm, f'{arm}|ols_fit_endpoint', x, y, f'x={UNIT_XI}; y={UNIT_GAP}', source)
    return {'arm': arm, 'n': int(len(sub)), 'slope': slope, 'intercept': intercept,
            'spearman': spearman(sub.xi, sub.gap), 'pearson': pearson(sub.xi, sub.gap),
            'median_gap': float(sub.gap.median())}


def draw_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed shuffle so that no group is systematically drawn on top of another."""
    index = np.random.default_rng(BOOT_SEED).permutation(len(frame))
    return frame.iloc[index]


def key_of(row):
    return f'{row.condition}|series{row.series}' if hasattr(row, 'condition') else str(row.symbol)


def figure2_s1(data, outdir):
    s1, meta, order = data['s1'], data['meta'], data['order']
    palette = condition_palette(meta, order)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.subplots_adjust(left=.085, right=.70, bottom=.17, top=.96)
    ax.axhline(0, color=ZERO_COLOR, lw=.7, zorder=1)
    fits = [scatter_fit(ax, FIG2A, s1, arm, lambda row: palette[row.condition],
                        'S1_SCORES.csv + S1_DIFFICULTY.csv') for arm in ARMS]
    ax.set_xlabel(XI_LABEL)
    ax.set_ylabel(GAP_LABEL)
    axes_style(ax, grid_axis='both')
    handles = [Line2D([], [], marker='o', ls='', ms=3.2, color=palette[c],
                      label=condition_label(meta, c)) for c in order]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.715, .985), frameon=False,
               fontsize=6.6, labelspacing=.32, handletextpad=.5, borderaxespad=0,
               title='Condition', title_fontsize=7)
    arm_handles = [Line2D([], [], marker=ARM_MARKER[a], ls=ARM_LINE[a], color=MEAN_COLOR,
                          mec=MEAN_COLOR, mfc=MEAN_COLOR if a == 'F0' else 'none', ms=3.4, lw=.9,
                          label=f'{ARM_LABEL[a]} + least squares') for a in ARMS]
    fig.legend(handles=arm_handles, loc='lower left', bbox_to_anchor=(.715, .045), frameon=False,
               fontsize=6.6, labelspacing=.32, handletextpad=.5, borderaxespad=0)
    export(fig, outdir, FIG2A)
    return fits


def figure2_s2(data, outdir):
    s2 = data['s2']
    # Canvas is narrower than figure 2a, but the axes box is the same size, so the
    # two scatters can be compared directly.
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.subplots_adjust(left=.119, right=.98, bottom=.17, top=.96)
    ax.axhline(0, color=ZERO_COLOR, lw=.7, zorder=1)
    fits = [scatter_fit(ax, FIG2B, s2, arm, lambda row, a=arm: ARM_COLOR[a],
                        'S2_SCORES.csv + S2_DIFFICULTY.csv',
                        line_colour=ARM_COLOR[arm], size=3.0) for arm in ARMS]
    ax.set_xlabel(XI_LABEL)
    ax.set_ylabel(GAP_LABEL)
    axes_style(ax, grid_axis='both')
    handles = [Line2D([], [], marker=ARM_MARKER[a], ls=ARM_LINE[a], color=ARM_COLOR[a],
                      mfc=ARM_COLOR[a] if a == 'F0' else 'none', ms=3.4, lw=.9,
                      label=f'{ARM_LABEL[a]} + least squares') for a in ARMS]
    ax.legend(handles=handles, loc='best', frameon=False, fontsize=6.6, handletextpad=.5)
    fits_aux = [{'arm': arm,
                 'spearman': spearman(s2[s2.arm == arm].xi, s2[s2.arm == arm].gap_vs_garch_t)}
                for arm in ARMS]
    export(fig, outdir, FIG2B)
    return fits, fits_aux


def condition_palette(meta, order):
    """Two hue families (one per phi) with six steps ordered by tail severity."""
    groups = group_by_phi(meta, order)
    maps = ['Blues', 'Oranges', 'Greens', 'Purples']
    palette = {}
    for g, (_phi, indices) in enumerate(groups):
        cmap = plt.get_cmap(maps[g % len(maps)])
        shades = cmap(np.linspace(.50, .98, max(len(indices), 2)))
        for j, index in enumerate(indices):
            palette[order[index]] = to_hex(shades[j])
    return palette


# ------------------------------------------------------------------ figure 3
def figure3(data, outdir):
    pairs, meta = data['pairs'], data['meta']
    contrasts = sorted(pairs.contrast.unique(),
                       key=lambda c: (EPS_RANK.get(c.rsplit('_', 1)[0], 99), c.rsplit('_', 1)[1] == 'on'))
    colours = {c: OKABE_ITO[i % len(OKABE_ITO)] for i, c in enumerate(contrasts)}
    phis = sorted(pairs.phi.unique())
    markers = ['o', '^', 's', 'D']
    fig, ax = plt.subplots(figsize=(7, 3.3))
    fig.subplots_adjust(left=.085, right=.695, bottom=.165, top=.96)
    ax.axhline(0, color=ZERO_COLOR, lw=.7, zorder=1)
    ax.axvline(0, color=ZERO_COLOR, lw=.7, zorder=1)
    for row in draw_order(pairs).itertuples():
        marker = markers[phis.index(row.phi) % len(markers)]
        colour = colours[row.contrast]
        ax.plot(row.dSE, row.dxi, marker, ms=2.8, mew=.45, mec=colour,
                mfc=colour if phis.index(row.phi) == 0 else 'none', alpha=.8, zorder=2)
    for row in pairs.itertuples():
        record(FIG3, 'main', f'phi{row.phi:g}|{row.cond_a}->{row.cond_b}|series{row.series}',
               row.dSE, row.dxi, f'x=delta {UNIT_SE}; y=delta {UNIT_XI}', 'S1_PAIRS.csv')
    mean_se, mean_xi = float(pairs.dSE.mean()), float(pairs.dxi.mean())
    ci_se, ci_xi = boot_mean_ci(pairs.dSE), boot_mean_ci(pairs.dxi)
    ax.errorbar(mean_se, mean_xi, xerr=[[mean_se - ci_se[0]], [ci_se[1] - mean_se]],
                yerr=[[mean_xi - ci_xi[0]], [ci_xi[1] - mean_xi]], fmt='X', ms=7,
                mec='white', mew=.9, color=MEAN_COLOR, ecolor=MEAN_COLOR, elinewidth=1.1,
                capsize=2.4, capthick=1.1, zorder=6)
    for tag, x, y in [('mean', mean_se, mean_xi), ('ci_low', ci_se[0], ci_xi[0]),
                      ('ci_high', ci_se[1], ci_xi[1])]:
        record(FIG3, 'main', f'all_pairs|{tag}', x, y,
               f'x=delta {UNIT_SE}; y=delta {UNIT_XI}', 'S1_PAIRS.csv')
    for contrast in contrasts:
        sub = pairs[pairs.contrast == contrast]
        record(FIG3, 'summary', f'{contrast}|mean', float(sub.dSE.mean()), float(sub.dxi.mean()),
               f'x=delta {UNIT_SE}; y=delta {UNIT_XI}', 'S1_PAIRS.csv')
    ax.set_xlabel(DSE_LABEL)
    ax.set_ylabel(DXI_LABEL)
    axes_style(ax, grid_axis='both')
    handles = [Line2D([], [], marker='o', ls='', ms=3.2, color=colours[c],
                      label='vs ' + contrast_label(c)) for c in contrasts]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.71, .955), frameon=False,
               fontsize=6.6, labelspacing=.32, handletextpad=.5, borderaxespad=0,
               title=f'Matched contrast\n(baseline: {contrast_label(pairs.baseline.iloc[0])})',
               title_fontsize=6.8, alignment='left')
    extra = [Line2D([], [], marker=markers[i % len(markers)], ls='', ms=3.2, color=MEAN_COLOR,
                    mfc=MEAN_COLOR if i == 0 else 'none', label=rf'$\varphi$ = {phi:g}')
             for i, phi in enumerate(phis)]
    extra.append(Line2D([], [], marker='X', ls='', ms=4.5, color=MEAN_COLOR,
                        label='Mean of all pairs, 95% CI'))
    fig.legend(handles=extra, loc='lower left', bbox_to_anchor=(.71, .05), frameon=False,
               fontsize=6.6, labelspacing=.32, handletextpad=.5, borderaxespad=0)
    export(fig, outdir, FIG3)
    return {'n': int(len(pairs)), 'mean_dse': mean_se, 'ci_dse': ci_se, 'mean_dxi': mean_xi,
            'ci_dxi': ci_xi, 'share_dxi_positive': float((pairs.dxi > 0).mean()),
            'max_abs_dse': float(pairs.dSE.abs().max()),
            'baseline': contrast_label(pairs.baseline.iloc[0]),
            'contrasts': [contrast_label(c) for c in contrasts],
            'se_range': (float(pairs[['se_a', 'se_b']].to_numpy().min()),
                         float(pairs[['se_a', 'se_b']].to_numpy().max()))}


def contrast_label(key):
    eps, garch = key.rsplit('_', 1)
    return f'{EPS_LABEL.get(eps, eps)}, {GARCH_LABEL.get(garch, garch)}'


# ----------------------------------------------------------------- captions
def captions(data, bars, fits1, fits2, fits2_aux, pair_stats, outdir: Path, results: Path):
    s1, s2, order, meta = data['s1'], data['s2'], data['order'], data['meta']
    n_series = sorted({n for (_a, _c), (_q25, _q50, _q75, n) in bars.items()})
    per_condition = (f'{n_series[0]} series per condition' if len(n_series) == 1
                     else f'{n_series[0]}-{n_series[-1]} series per condition')
    medians = {arm: {c: bars[(arm, c)][1] for c in order} for arm in ARMS}
    lines = ['# Figure captions — tail-complexity validity study (TCV-1)', '',
             'English captions for the contracted figures. Every number below is computed by '
             '`figures.py` from the CSVs in this directory; the plotted coordinates are exported '
             'to `FIGURE_VALUES.csv`. The primary metric is '
             'G*_.995 = QS_0.995(arm with the SEL extension) / QS_0.995(reference) - 1, so 0 means '
             'the arm matches its reference and positive values mean it is worse. Hypothesis '
             'decisions are not recomputed here; they live in the S1/S2 hypothesis files.', '']

    worst = {arm: max(order, key=lambda c: medians[arm][c]) for arm in ARMS}
    best = {arm: min(order, key=lambda c: medians[arm][c]) for arm in ARMS}
    lines += [f'## Figure 1 — `{FIG1}`', '',
              'Quantile-score gap of the two forecasting arms at the out-of-grid level tau = 0.995, '
              'by synthetic condition. Bars are medians over the series of each condition '
              f'({per_condition}); whiskers span the interquartile range; the horizontal line at 0 '
              'marks parity with the oracle reference of that condition. Conditions are grouped by '
              'the autoregressive coefficient and ordered by innovation tail weight and by the '
              'volatility switch. ' +
              ' '.join(f'For {ARM_LABEL[arm]}, the condition medians run from '
                       f'{fmt(medians[arm][best[arm]])} ({condition_label(meta, best[arm], False)}) to '
                       f'{fmt(medians[arm][worst[arm]])} ({condition_label(meta, worst[arm], False)}).'
                       for arm in ARMS) +
              ' The comparison is within condition: each arm is scored against the reference of its '
              'own condition, so bar heights measure relative loss, not absolute difficulty.', '']

    for tag, fits, name, unit_note in [
            ('Figure 2a', fits1, FIG2A,
             f'Each point is one synthetic series ({len(s1[s1.arm == ARMS[0]])} per arm); colour '
             'encodes the generating condition.'),
            ('Figure 2b', fits2, FIG2B,
             f'Each point is one symbol ({len(s2[s2.arm == ARMS[0]])} per arm); the reference is the '
             'conditional-EVT forecast.')]:
        stats = '; '.join(
            f'{ARM_LABEL[f["arm"]]}: Spearman rho = {fmt(f["spearman"])}, Pearson r = {fmt(f["pearson"])}, '
            f'least-squares slope = {fmt(f["slope"])} per unit of xi-hat, n = {f["n"]}' for f in fits)
        lines += [f'## {tag} — `{name}`', '',
                  'Estimated tail index of the training window against the out-of-grid gap '
                  'G*_.995. ' + unit_note + ' Filled circles are the frozen arm and open triangles '
                  'the LoRA arm, drawn on the same axes; the two straight lines are ordinary '
                  'least-squares fits of the gap on xi-hat (solid: frozen, dashed: LoRA). '
                  f'{stats}. The horizontal line at 0 marks parity with the reference. These are '
                  'unconditional associations on the raw gap; the contracted regressions model '
                  'log(1 + G*_.995) with spectral entropy and the extremal index included and are '
                  'reported separately.', '']
    aux = '; '.join(f'{ARM_LABEL[f["arm"]]}: Spearman rho = {fmt(f["spearman"])}' for f in fits2_aux)
    lines[-2] += (' Against the auxiliary GARCH-t reference, the same association is ' + aux +
                  '; that alternative reference is not plotted, because it would put two different '
                  'reference definitions on one axis.')

    lines += [f'## Figure 3 — `{FIG3}`', '',
              'Spectrum-matched pairs of synthetic series: within one autoregressive coefficient, '
              f'each series generated under the baseline condition ({pair_stats["baseline"]}) is '
              'paired with the same series index under a heavier-tail or volatility-switched '
              f'condition. The axes are the within-pair differences of spectral entropy (x) and of '
              f'the estimated tail index (y), n = {pair_stats["n"]} pairs. Colour encodes the matched '
              'contrast and marker fill the autoregressive coefficient; the cross marks the mean '
              'over all pairs with a percentile-bootstrap 95% interval '
              f'({BOOT_DRAWS} draws, seed {BOOT_SEED}; pairs that share a series index are not '
              'independent). Mean difference in spectral entropy is '
              f'{fmt(pair_stats["mean_dse"], 4)} (95% CI {fmt(pair_stats["ci_dse"][0], 4)} to '
              f'{fmt(pair_stats["ci_dse"][1], 4)}), while the mean difference in the tail index is '
              f'{fmt(pair_stats["mean_dxi"], 4)} (95% CI {fmt(pair_stats["ci_dxi"][0], 4)} to '
              f'{fmt(pair_stats["ci_dxi"][1], 4)}), with '
              f'{100*pair_stats["share_dxi_positive"]:.1f}% of pairs above the horizontal zero line. '
              'For scale, the spectral entropy of '
              f'these series itself runs from {fmt(pair_stats["se_range"][0])} to '
              f'{fmt(pair_stats["se_range"][1])}, while the largest absolute within-pair difference '
              f'is {fmt(pair_stats["max_abs_dse"], 4)}. '
              'Points therefore concentrate on a vertical band around zero on the x axis and sit '
              'above zero on the y axis: the second-order spectrum is held while the tail shape '
              'moves, which is the non-redundancy the design requires.', '']

    dropped = {k: v for k, v in data['dropped'].items() if v}
    if dropped:
        lines += ['## Excluded rows', '',
                  'Rows without a finite estimate were excluded before plotting: '
                  + ', '.join(f'{k}: {v}' for k, v in dropped.items()) + '.', '']
    lines += ['---', '',
              f'Generated by `figures.py` from `{results}`. Fonts: {FONT}; '
              'PNG at 600 dpi plus vector PDF and SVG.']
    (results/'CAPTIONS.md').write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')


# --------------------------------------------------------------------- main
def main(argv=None) -> int:
    default = Path(__file__).resolve().parents[2]/'results'/'tail_complexity_v1_20260924'
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results', type=Path, default=default,
                        help=f'directory holding the S1/S2 result CSVs (default: {default})')
    args = parser.parse_args(argv)
    results = args.results.expanduser().resolve()
    try:
        data = load_inputs(results)
    except DataError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    outdir = results/'figures'
    outdir.mkdir(parents=True, exist_ok=True)
    bars = figure1(data, outdir)
    fits1 = figure2_s1(data, outdir)
    fits2, fits2_aux = figure2_s2(data, outdir)
    pair_stats = figure3(data, outdir)
    pd.DataFrame(VALUES, columns=['figure', 'panel', 'series', 'x', 'y', 'unit', 'source_file']) \
        .to_csv(results/'FIGURE_VALUES.csv', index=False)
    captions(data, bars, fits1, fits2, fits2_aux, pair_stats, outdir, results)
    for name in [FIG1, FIG2A, FIG2B, FIG3]:
        print('wrote', ', '.join(str(outdir/f'{name}.{e}') for e in ['png', 'svg', 'pdf']))
    print('wrote', results/'FIGURE_VALUES.csv', f'({len(VALUES)} rows)')
    print('wrote', results/'CAPTIONS.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
