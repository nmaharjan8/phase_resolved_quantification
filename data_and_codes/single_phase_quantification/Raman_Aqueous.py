
# -*- coding: utf-8 -*-
"""Individual aqueous Raman figures and peak-height analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# ============================================================
# USER SETTINGS
# ============================================================
DATA_DIR = r'path/to/folder/data_and_codes/single_phase_quantification/single_phase_data/'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

PRE_GROUPS = {
    'Deionized water': (6, 10),
    '250 mM HNO$_3$': (56, 60),
    '500 mM HNO$_3$': (62, 66),
    '750 mM HNO$_3$': (68, 72),
    '1 M HNO$_3$': (74, 78),
    '1.5 M HNO$_3$': (80, 84),
    '2.5 M HNO$_3$': (86, 90),
    '3.5 M HNO$_3$': (92, 96),
    '5 M HNO$_3$': (98, 102),
}

# Only one post-contact sample per concentration is confirmed here.
# Replace single-number ranges with actual replicate ranges to calculate SD.
POST_GROUPS = {
    '500 mM HNO$_3$': (105, 109),
    '1 M HNO$_3$': (112, 116),
    '2.5 M HNO$_3$': (118, 122),
    '3.5 M HNO$_3$': (124, 128),
    '5 M HNO$_3$': (130, 134),
}

GROUP_COLORS = ['black', 'red', 'blue', 'green', 'brown', 'purple', 'magenta', 'orange', 'cyan']
B_PRE = {
    'Sample (63)': '500 mM HNO$_3$',
    'Sample (75)': '1 M HNO$_3$',
    'Sample (86)': '2.5 M HNO$_3$',
    'Sample (94)': '3.5 M HNO$_3$',
    'Sample (99)': '5 M HNO$_3$',
}
B_POST = {
    'Sample (106)': '500 mM HNO$_3$',
    'Sample (113)': '1 M HNO$_3$',
    'Sample (119)': '2.5 M HNO$_3$',
    'Sample (127)': '3.5 M HNO$_3$',
    'Sample (133)': '5 M HNO$_3$',
}
B_COLORS = ['blue', 'brown', 'magenta', 'orange', 'cyan']

# ============================================================
# FIGURE STYLE
# ============================================================
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 18,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 14,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.8,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'savefig.dpi': 300,
})


def read_csv(folder, sample):
    if isinstance(sample, (int, np.integer)):
        sample = f'Sample ({sample})'
    path = os.path.join(folder, f'{sample}.csv')
    if not os.path.isfile(path):
        print(f'[WARN] Missing file: {path}')
        return None
    data = pd.read_csv(path, skiprows=1)
    x = pd.to_numeric(data.iloc[:, 0], errors='coerce').to_numpy()
    y = pd.to_numeric(data.iloc[:, 1], errors='coerce').to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid][np.argsort(x[valid])], y[valid][np.argsort(x[valid])]


def baseline_correct(wn, intensity, x1=1025, x2=1070, target=1048):
    sel = (wn >= x1) & (wn <= x2)
    x, y = wn[sel], intensity[sel]
    if len(x) < 2:
        raise ValueError('Insufficient data in baseline window')
    baseline = y[0] + (y[-1] - y[0]) * (x - x[0]) / (x[-1] - x[0])
    corr = y - baseline
    ix = np.argmin(np.abs(x - target))
    return x, corr, x[ix], corr[ix]


def representative_sample(folder, sample_range):
    available = [n for n in range(sample_range[0], sample_range[1] + 1)
                 if os.path.isfile(os.path.join(folder, f'Sample ({n}).csv'))]
    if not available:
        return None
    return available[len(available) // 2]


def format_ax(ax, xlim=None, ylim=None, ylabel='Intensity (a.u.)'):
    ax.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(direction='out', pad=5, labelsize=18)


os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(fig, filename):
    fig.savefig(os.path.join(OUTPUT_DIR, filename + '.jpg'),
                dpi=300, facecolor='white')
    print('Saved:', filename + '.jpg')


# --------- PANEL (a) ---------
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
# ---------------- (ax) Full-range aqueous Raman spectra ----------------
raw_reps = {}
for (label, rng), color in zip(PRE_GROUPS.items(), GROUP_COLORS):
    rep = representative_sample(DATA_DIR, rng)
    if rep is None:
        continue
    xy = read_csv(DATA_DIR, rep)
    if xy is None:
        continue
    wn, inten = xy
    raw_reps[label] = (wn, inten, color)
    ax.plot(wn, inten, color=color, lw=1.8, label=label)
format_ax(ax, xlim=(0, 3500))
ax.legend(loc='upper left', fontsize=14, frameon=True, ncol=1,
         borderpad=0.3, labelspacing=0.3, handlelength=1.5)
# Sparse major ticks for the full-range spectrum.
ax.set_xticks(np.arange(0, 3501, 500))
ax.set_yticks(np.arange(0, 18001, 4000))
ax.set_ylim(-500, 17000)

# The zoom is exported as a separate figure.
fig.tight_layout()
save_plot(fig, 'raman_panel_a_full_range')
plt.show()

# --------- STANDALONE ZOOM FOR PANEL (a) ---------
fig_zoom, ax_zoom = plt.subplots(figsize=(10, 10), dpi=300)
for wn, inten, color in raw_reps.values():
    ax_zoom.plot(wn, inten, color=color, lw=2.2)
ax_zoom.set_xlim(1025, 1070)
ax_zoom.set_xticks([1030, 1050, 1070])
ax_zoom.set_yticks([0, 8000, 16000])
ax_zoom.tick_params(axis='both', labelsize=24, pad=7)
fig_zoom.tight_layout()
save_plot(fig_zoom, 'raman_panel_a_zoom_separate')
plt.show()

# --------- PANEL (b) ---------
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
# ---------------- (ax) Pre/post aqueous comparison ----------------
for i, ((pre_file, label), (post_file, _)) in enumerate(zip(B_PRE.items(), B_POST.items())):
    for fname, linestyle in [(pre_file, '-'), (post_file, '--')]:
        xy = read_csv(DATA_DIR, fname)
        if xy is not None:
            ax.plot(*xy, color=B_COLORS[i], linestyle=linestyle, lw=1.8)
format_ax(ax, xlim=(1025, 1070))
ax.set_xticks([1030, 1040, 1050, 1060, 1070])
ax.yaxis.set_major_locator(MultipleLocator(4000))
color_handles = [Line2D([0], [0], color=color, lw=1.6, label=label)
                 for color, label in zip(B_COLORS, B_PRE.values())]
style_handles = [Line2D([0], [0], color='black', lw=1.5, ls='-', label='Pre-contact'),
                 Line2D([0], [0], color='black', lw=1.5, ls='--', label='Post-contact')]
leg_b1 = ax.legend(handles=color_handles, loc='upper right', fontsize=14,
                  frameon=True, borderpad=0.25, labelspacing=0.25)
ax.add_artist(leg_b1)
ax.legend(handles=style_handles, loc='upper left', fontsize=14,
         frameon=True, borderpad=0.25, labelspacing=0.25)

fig.tight_layout()
save_plot(fig, 'raman_panel_b')
plt.show()

# --------- BASELINE-CORRECTED PRE- AND POST-CONTACT FIGURES ---------
# Generate two independent figures, and calculate statistics from all group files.
for contact, groups, colors, line_style in [
    ('pre', PRE_GROUPS, GROUP_COLORS, '-'),
    ('post', POST_GROUPS, B_COLORS, ':'),
]:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    for (label, rng), color in zip(groups.items(), colors):
        rep = representative_sample(DATA_DIR, rng)
        if rep is None:
            continue
        xy = read_csv(DATA_DIR, rep)
        if xy is None:
            continue
        try:
            wn, corr, wn_peak, peak = baseline_correct(*xy)
        except ValueError as exc:
            print(f'[WARN] {label}: {exc}')
            continue
        ax.plot(wn, corr, color=color, ls=line_style, lw=1.8, label=label)
        ax.plot(wn_peak, peak, marker='*', ms=9, color=color, ls='None')

    ax.axvline(1048, ls='--', color='gray', lw=0.8)
    format_ax(ax, xlim=(1025, 1070),
              ylabel='Baseline Corrected Intensity (a.u.)')
    ax.set_xticks([1030, 1040, 1050, 1060, 1070])
    ax.yaxis.set_major_locator(MultipleLocator(4000))
    ax.legend(loc='upper right', fontsize=14, frameon=True,
              borderpad=0.3, labelspacing=0.25, handlelength=1.5)
    fig.tight_layout()
    save_plot(fig, f'raman_panel_c_{contact}_contact')
    plt.show()

    # Peak heights use every spectrum in each group, not only the representative.
    peak_rows = []
    summary_rows = []
    for label, (start, end) in groups.items():
        heights = []
        for n in range(start, end + 1):
            xy = read_csv(DATA_DIR, n)
            if xy is None:
                continue
            try:
                _, _, wn_peak, height = baseline_correct(*xy)
            except ValueError as exc:
                print(f'[WARN] Sample ({n}): {exc}')
                continue
            heights.append(height)
            peak_rows.append({
                'Contact': contact, 'Group': label, 'Sample': n,
                'Peak wavenumber (cm-1)': wn_peak,
                'Peak height (a.u.)': height,
            })
        summary_rows.append({
            'Contact': contact, 'Group': label, 'n': len(heights),
            'Mean peak height (a.u.)': np.mean(heights) if heights else np.nan,
            'SD peak height (a.u.)': np.std(heights, ddof=1)
            if len(heights) > 1 else np.nan,
        })

    per_spectrum = pd.DataFrame(peak_rows, columns=[
        'Contact', 'Group', 'Sample', 'Peak wavenumber (cm-1)', 'Peak height (a.u.)'])
    summary = pd.DataFrame(summary_rows)
    per_spectrum.to_csv(
        os.path.join(OUTPUT_DIR, f'raman_peak_heights_{contact}_contact.csv'),
        index=False)
    summary.to_csv(
        os.path.join(OUTPUT_DIR, f'raman_peak_statistics_{contact}_contact.csv'),
        index=False)
    print(f'\n{contact.capitalize()}-contact peak heights at 1048 cm-1:')
    print(summary.to_string(index=False))

print('This code generates plots used to make Figures 1(a)-(c). This code also plots figure similar to Figure 1(c) for post slug flow extraction for aqueous phase.')
