
# -*- coding: utf-8 -*-
"""This code plots Figure 4."""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------- SETTINGS ----------------
FOLDER = r'path/to/folder/data_and_codes/CLS_analysis/biphasic_spectra_set_4/'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

mpl.rcParams['font.family'] = 'Arial'

# ---------------- LOAD FILES ----------------
# Sort numerically by sample number.
csv_files = sorted(
    (f for f in os.listdir(FOLDER) if f.lower().endswith('.csv')),
    key=lambda f: int(re.search(r'Sample \((\d+)\)', f).group(1))
)

print(f'Found {len(csv_files)} CSV files.')

# Green gradient, ordered by sample number
colors = plt.cm.Greens(np.linspace(0.35, 0.95, len(csv_files)))

# ---------------- PLOT ----------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

n_plotted = 0

for i, filename in enumerate(csv_files):
    try:
        data = pd.read_csv(os.path.join(FOLDER, filename), skiprows=1)

        if data.shape[1] < 2:
            print(f'Skipped {filename}: fewer than 2 columns.')
            continue

        data = data.iloc[:, :2].apply(
            pd.to_numeric, errors='coerce'
        ).dropna().sort_values(by=data.columns[0])

        if data.empty:
            print(f'Skipped {filename}: no numeric data.')
            continue

        ax.plot(
            data.iloc[:, 0],
            data.iloc[:, 1],
            color=colors[i],
            linewidth=0.9,
            alpha=0.9
        )

        n_plotted += 1

    except Exception as e:
        print(f'Could not plot {filename}: {e}')

# ---------------- FORMATTING ----------------
ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=16)
ax.set_ylabel('Intensity', fontsize=16)
ax.tick_params(axis='both', labelsize=13)

fig.tight_layout()

# ---------------- SAVE ----------------
output = os.path.join(OUTPUT_DIR, 'Raman_spectra_biphasic_spectra_stacked_plot.jpg')
fig.savefig(output, dpi=300, facecolor='white')

print(f'Successfully plotted {n_plotted} spectra.')
print(f'Saved: {output}')
print('This code plots Figure 4.')

plt.show()
