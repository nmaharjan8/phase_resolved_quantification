
# -*- coding: utf-8 -*-
"""Organic-phase Raman spectra: separate panels (e) and (f).
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

FOLDER = r'path/to/folder/data_and_codes/single_phase_quantification/single_phase_data/'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLES = [183, 187, 195, 203, 209, 217]
COLORS = ['black', 'blue', 'brown', 'magenta', 'orange', 'cyan']

# Panel (e): initial aqueous HNO3 concentration used for contact.
LABELS_E = [
    '30% (v/v) TBP in n-dodecane (OP)',
    'OP after extraction with 500 mM HNO$_3$',
    'OP after extraction with 1 M HNO$_3$',
    'OP after extraction with 2.5 M HNO$_3$',
    'OP after extraction with 3.5 M HNO$_3$',
    'OP after extraction with 5 M HNO$_3$',
]

# Panel (f): measured post-contact HNO3 concentration in organic phase.
LABELS_F = [
    '0 mM HNO$_3$ in OP',
    '76 mM HNO$_3$ in OP',
    '218 mM HNO$_3$ in OP',
    '463 mM HNO$_3$ in OP',
    '588 mM HNO$_3$ in OP',
    '784 mM HNO$_3$ in OP',
]

# Consistent with the corrected aqueous figures.
FIGSIZE = (10, 8)
AXIS_FONT = 22
TICK_FONT = 18
LEGEND_FONT = 14

mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': TICK_FONT,
    'axes.labelsize': AXIS_FONT,
    'xtick.labelsize': TICK_FONT,
    'ytick.labelsize': TICK_FONT,
    'legend.fontsize': LEGEND_FONT,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})


def read_spectrum(n):
    path = os.path.join(FOLDER, f'Sample ({n}).csv')
    if not os.path.isfile(path):
        print(f'[WARN] File not found: {path}')
        return None
    df = pd.read_csv(path, skiprows=1)
    wn = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy()
    intensity = pd.to_numeric(df.iloc[:, 1], errors='coerce').to_numpy()
    good = np.isfinite(wn) & np.isfinite(intensity)
    wn, intensity = wn[good], intensity[good]
    order = np.argsort(wn)
    return wn[order], intensity[order]


def plot_panel(panel, labels, xlim, xticks, ylim=None, yticks=None):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=300)
    for n, color, label in zip(SAMPLES, COLORS, labels):
        spectrum = read_spectrum(n)
        if spectrum is not None:
            ax.plot(*spectrum, color=color, lw=2, label=label)

    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    if yticks is not None:
        ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=AXIS_FONT)
    ax.set_ylabel('Intensity (a.u.)', fontsize=AXIS_FONT)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT)
    ax.legend(loc='upper left', fontsize=LEGEND_FONT, frameon=True,
              borderpad=0.4, labelspacing=0.35, handlelength=1.8)

    # Identical axes positions for both panels; don't use bbox_inches='tight'.
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.96)

    base = os.path.join(OUTPUT_DIR, f'raman_panel_{panel}')
    fig.savefig(base + '.jpg', dpi=300, facecolor='white')
    print('Saved:', base + '.jpg')
    return fig


# (e) Full-range organic spectra, no inset or arrows.
plot_panel('e', LABELS_E, xlim=(0, 3500),
           xticks=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500],
           ylim=(-400, 8200), yticks=[0, 2000, 4000, 6000, 8000])

# (f) Zoomed organic spectra, same samples with post-contact concentrations.
plot_panel('f', LABELS_F, xlim=(550, 1350),
           xticks=[600, 800, 1000, 1200],
           ylim=(-250, 5000), yticks=[0, 1000, 2000, 3000, 4000, 5000])

plt.show()

print('This code generates plot Figure 1(e,f).')
