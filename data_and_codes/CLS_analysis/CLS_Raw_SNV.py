# -*- coding: utf-8 -*-
"""Non-negative CLS analysis of mixed Raman spectra (raw and SNV)."""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.metrics import r2_score, mean_squared_error

# ---------------- SETTINGS ----------------
DATA_DIR = Path(r'path/to/folder/data_and_codes/CLS_analysis/biphasic_spectra_set_4/') #switch to different folders for example biphasic_spectra_set_1, biphasic_spectra_set_2, ... to get different sets of CLS fits
REFERENCE_DIR = DATA_DIR / 'reference_spectra'
OUTPUT_DIR = DATA_DIR / 'CLS_results'

RAW_DIR = OUTPUT_DIR / 'Raw_CLS_Fits'
SNV_DIR = OUTPUT_DIR / 'SNV_CLS_Fits'
COMPARE_DIR = OUTPUT_DIR / 'Raw_vs_SNV_Parity'
for folder in (RAW_DIR, SNV_DIR, COMPARE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.family': 'Arial', 'figure.dpi': 300})


def csv_files(folder):
    # Natural sorting: Sample (9), Sample (10), Sample (100).
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.csv'),
                  key=lambda p: [int(s) if s.isdigit() else s.lower()
                                 for s in re.split(r'(\d+)', p.name)])


def read_spectrum(path):
    data = pd.read_csv(path)
    return (data.iloc[:, 0].to_numpy(dtype=float),
            data.iloc[:, 1].to_numpy(dtype=float))


def snv(y):
    sd = np.std(y, ddof=1)
    if sd == 0:
        raise ValueError('Cannot SNV-normalize a constant spectrum.')
    return (y - np.mean(y)) / sd


def cls_fit(y, aq, org):
    refs = np.column_stack((aq, org))
    weights, _ = nnls(refs, y)
    fit = refs @ weights
    norm = np.linalg.norm(y)
    residual = 100 * np.linalg.norm(y - fit) / norm if norm else np.nan
    return weights, fit, residual


def plot_fit(wn, measured, fitted, aq_part, org_part, weights, residual,
             name, is_snv, output):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(wn, measured, color='black', lw=1.5,
            label='SNV experimental spectrum' if is_snv else 'Experimental spectrum')
    ax.plot(wn, fitted, '--', lw=1.5,
            label='SNV CLS fit' if is_snv else 'CLS fit')
    ax.plot(wn, aq_part, '-.', lw=1.2, label='Aqueous contribution')
    ax.plot(wn, org_part, ':', lw=1.2, label='Organic contribution')
    ax.set_xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=15)
    ax.set_ylabel('SNV-normalized intensity' if is_snv else 'Intensity (a.u.)', fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    label = 'SNV CLS' if is_snv else 'Raw CLS'
    ax.set_title(f'{name} | {label}\n'
                 f'$f_{{aq}}$ = {weights[0]:.4f} | '
                 f'$f_{{org}}$ = {weights[1]:.4f} | '
                 f'Residual = {residual:.2f}%', fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)


def parity_plot(x, y, phase, output):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    r2 = r2_score(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.05 * (hi - lo) if hi != lo else 0.05
    lo, hi = max(0, lo - pad), hi + pad

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    ax.scatter(x, y, s=70, edgecolor='black', alpha=0.85)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='1:1')
    ax.set(xlim=(lo, hi), ylim=(lo, hi))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'Raw CLS $f_{{{phase}}}$', fontsize=15)
    ax.set_ylabel(f'Corrected SNV-CLS $f_{{{phase}}}$', fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    ax.text(0.05, 0.95, f'$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}',
            transform=ax.transAxes, va='top', fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------- REFERENCES ----------------
reference_files = csv_files(REFERENCE_DIR)
if len(reference_files) != 2:
    raise ValueError('Reference folder must contain exactly two CSV files.')

# First naturally sorted reference is aqueous; second is organic.
wn, aq_ref = read_spectrum(reference_files[0])
wn_org, org_ref = read_spectrum(reference_files[1])
if wn.shape != wn_org.shape or not np.allclose(wn, wn_org):
    raise ValueError('Reference wavenumber grids do not match.')

print('Aqueous reference:', reference_files[0].name)
print('Organic reference:', reference_files[1].name)

aq_snv, org_snv = snv(aq_ref), snv(org_ref)
aq_std, org_std = np.std(aq_ref, ddof=1), np.std(org_ref, ddof=1)

# ---------------- FIT MIXED SPECTRA ----------------
files = csv_files(DATA_DIR)
if not files:
    raise ValueError('No mixed Raman CSV files found.')

rows = []
for i, path in enumerate(files, 1):
    wn_mix, mixture = read_spectrum(path)
    if wn_mix.shape != wn.shape or not np.allclose(wn_mix, wn):
        print('Skipping (wavenumber mismatch):', path.name)
        continue

    raw_w, raw_fit, raw_res = cls_fit(mixture, aq_ref, org_ref)
    mixture_snv = snv(mixture)
    snv_w, snv_fit, snv_res = cls_fit(mixture_snv, aq_snv, org_snv)

    # Convert SNV CLS coefficients to the original intensity scale.
    corrected_w = snv_w * np.std(mixture, ddof=1) / np.array([aq_std, org_std])

    plot_fit(wn, mixture, raw_fit, raw_w[0] * aq_ref, raw_w[1] * org_ref,
             raw_w, raw_res, path.stem, False,
             RAW_DIR / f'{path.stem}_Raw_CLS_fit.jpg')
    plot_fit(wn, mixture_snv, snv_fit, snv_w[0] * aq_snv, snv_w[1] * org_snv,
             corrected_w, snv_res, path.stem, True,
             SNV_DIR / f'{path.stem}_SNV_CLS_fit.jpg')

    rows.append({
        'Filename': path.stem,
        'f_aq_Raw_CLS': raw_w[0],
        'f_org_Raw_CLS': raw_w[1],
        'f_aq_SNV_direct': snv_w[0],
        'f_org_SNV_direct': snv_w[1],
        'f_aq_SNV_corrected': corrected_w[0],
        'f_org_SNV_corrected': corrected_w[1],
        'Raw_f_sum': raw_w.sum(),
        'Corrected_SNV_f_sum': corrected_w.sum(),
        'Raw_CLS_residual_percent': raw_res,
        'SNV_CLS_residual_percent': snv_res,
    })
    print(f'{i}/{len(files)}  {path.stem}  '
          f'Raw: ({raw_w[0]:.5f}, {raw_w[1]:.5f})  '
          f'Corrected SNV: ({corrected_w[0]:.5f}, {corrected_w[1]:.5f})')

# ---------------- EXPORT ----------------
results = pd.DataFrame(rows)
if results.empty:
    raise ValueError('No spectra matched the reference wavenumber grid.')

results.to_excel(OUTPUT_DIR / 'CLS_Phase_Contribution_Coefficients.xlsx', index=False)
for phase in ('aq', 'org'):
    parity_plot(results[f'f_{phase}_Raw_CLS'],
                results[f'f_{phase}_SNV_corrected'], phase,
                COMPARE_DIR / f'f_{phase}_Raw_vs_Corrected_SNV.jpg')

print(f'Processed {len(results)} spectra. Results saved in: {OUTPUT_DIR}')
