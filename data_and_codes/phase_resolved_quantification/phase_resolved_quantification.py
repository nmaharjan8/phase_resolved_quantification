# -*- coding: utf-8 -*-
"""Phase-resolved HNO3 quantification from mixed Raman spectra.

Fits raw and SNV PLSR models using precomputed CLS responses.
Response columns: f_aq, f_org, q_aq, q_org, c_aq (M), c_org (M),
ordered to match the numbered Raman CSV files.
"""

from pathlib import Path
import re

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# Input and output paths

DATA_DIR = Path(r'path/to/folder/phase_resolved_quantification_data')
RAW_RESPONSES = DATA_DIR / 'response_vectors' / 'raw_response_vectors.xlsx'
SNV_RESPONSES = DATA_DIR / 'response_vectors' / 'SNV_non_corrected_response_vectors.xlsx'
OUTPUT_DIR = DATA_DIR / 'PLSR_Raw_SNV_Final_Workflow'
RAW_DIR = OUTPUT_DIR / 'Raw_SI'
SNV_DIR = OUTPUT_DIR / 'SNV_Main_Text'
COMPARE_DIR = OUTPUT_DIR / 'Raw_vs_SNV_Comparison'

TEST_SIZE = 0.30
SEED = 69
FOLDS = 10
MAX_COMPONENTS = 20
CUTOFFS = (0.05, 0.10, 0.15, 0.20, 0.25)
Q_SCALE = 1000.0  # q = 1000*f*c; c in M
RESPONSES = ('f_aq', 'f_org', 'q_aq', 'q_org')
COLORS = {'Raw': '#D17B32', 'SNV': '#167C91'}

mpl.rcParams.update({
    'font.family': 'Arial', 'axes.titlesize': 14, 'axes.labelsize': 15,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 10,
    'figure.dpi': 300,
})


def snv(X):
    """SNV for each spectrum, using sample SD."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def load_data():
    files = sorted(DATA_DIR.glob('*.csv'),
                   key=lambda p: tuple(map(int, re.findall(r'\d+', p.name))))
    if not files:
        raise FileNotFoundError(f'No CSV spectra in {DATA_DIR}')
    spectra = [pd.read_csv(p).iloc[:, :2].to_numpy(dtype=float) for p in files]
    if any(a.shape != spectra[0].shape or
           not np.allclose(a[:, 0], spectra[0][:, 0]) for a in spectra):
        raise ValueError('Raman wavenumber grids do not match.')
    X_raw = np.stack([s[:, 1] for s in spectra])
    raw = pd.read_excel(RAW_RESPONSES).iloc[:, :6].to_numpy(dtype=float)
    normalized = pd.read_excel(SNV_RESPONSES).iloc[:, :6].to_numpy(dtype=float)
    if raw.shape != (len(files), 6) or normalized.shape != raw.shape:
        raise ValueError('Response rows must match the number of sorted CSV files.')
    if not np.allclose(raw[:, 4:], normalized[:, 4:], equal_nan=True):
        raise ValueError('Raw/SNV reference concentration rows do not agree.')
    # Check consistency of CLS coefficients and concentrations.
    for label, data in [('Raw', raw), ('SNV', normalized)]:
        errors = np.max(np.abs(data[:, 2:4] - Q_SCALE * data[:, :2] * data[:, 4:6]), axis=0)
        print(f'{label} q=1000*f*c maximum differences: {errors}')
    return files, X_raw, raw, normalized


def cv_predict(X, Y, strata, components):
    """Stratified out-of-fold predictions."""
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    predictions = np.empty_like(Y)
    fold_ids = np.empty(len(Y), dtype=int)
    for fold, (tr, val) in enumerate(splitter.split(X, strata)):
        model = PLSRegression(n_components=components, scale=False)
        predictions[val] = model.fit(X[tr], Y[tr]).predict(X[val])
        fold_ids[val] = fold
    return predictions, fold_ids


def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2, axis=0))


def recover(predicted):
    """Calculate concentration and relative phase contributions from PLSR outputs."""
    f, q = predicted[:, :2], predicted[:, 2:4]
    c = np.full_like(f, np.nan)
    np.divide(q, Q_SCALE * f, out=c, where=np.abs(f) > 1e-12)
    r = np.full_like(f, np.nan)
    total = f.sum(axis=1)
    valid = np.isfinite(f).all(axis=1) & (f >= 0).all(axis=1) & (total > 0)
    r[valid] = f[valid] / total[valid, None]
    return c, r


def select_cutoff(reference, cv_pred, fold_ids):
    """Select the smallest cutoff within one SEM of the minimum CV error."""
    c, r = recover(cv_pred)
    ranges = np.ptp(reference, axis=0)
    rows = []
    for cutoff in CUTOFFS:
        accepted = (r >= cutoff) & np.isfinite(c)
        scores = []
        for fold in range(FOLDS):
            errors = []
            for phase in range(2):
                keep = (fold_ids == fold) & accepted[:, phase]
                if keep.any() and ranges[phase] > 0:
                    errors.append(100 * rmse(reference[keep, phase],
                                             c[keep, phase]) / ranges[phase])
            if errors:
                scores.append(np.mean(errors))
        avg = np.mean(scores) if scores else np.nan
        sem = np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else np.nan
        overall = []
        for phase in range(2):
            keep = accepted[:, phase]
            overall.append(100 * rmse(reference[keep, phase], c[keep, phase]) /
                           ranges[phase] if keep.any() and ranges[phase] > 0 else np.nan)
        rows.append([cutoff, avg, sem, *overall, *accepted.mean(axis=0) * 100])
    table = pd.DataFrame(rows, columns=['cutoff', 'mean_cv_error', 'sem',
                                         'aq_nrmsecv', 'org_nrmsecv',
                                         'aq_retained_pct', 'org_retained_pct'])
    if table.mean_cv_error.isna().all():
        raise ValueError('No valid concentrations for any candidate cutoff.')
    best = table.loc[table.mean_cv_error.idxmin()]
    limit = best.mean_cv_error + (best['sem'] if np.isfinite(best['sem']) else 0)
    selected = float(table.loc[table.mean_cv_error <= limit, 'cutoff'].min())
    return selected, table


def scores(actual, predicted, mask):
    mask = mask & np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return {'n': 0, 'R2': np.nan, 'RMSE': np.nan}
    a, p = actual[mask], predicted[mask]
    r2 = r2_score(a, p) if len(a) > 1 and np.ptp(a) > 0 else np.nan
    return {'n': int(mask.sum()), 'R2': r2, 'RMSE': float(rmse(a, p))}


def axis_limits(*arrays):
    values = np.concatenate([np.asarray(a)[np.isfinite(a)] for a in arrays])
    if not len(values):
        return 0., 1.
    delta = np.ptp(values) or 1.0
    return min(0.0, values.min() - 0.05 * delta), values.max() + 0.06 * delta


def save_show(fig, folder, filename):
    fig.savefig(folder / filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def response_labels(method):
    suffix = r'^{\mathrm{SNV}}' if method == 'SNV' else ''
    symbols = [rf'$f_{{\mathrm{{aq}}}}{suffix}$', rf'$f_{{\mathrm{{org}}}}{suffix}$',
               rf'$q_{{\mathrm{{aq}}}}{suffix}$', rf'$q_{{\mathrm{{org}}}}{suffix}$']
    titles = [f'Aqueous CLS coefficient ({symbols[0]})',
              f'Organic CLS coefficient ({symbols[1]})',
              f'Aqueous phase-weighted concentration term ({symbols[2]})',
              f'Organic phase-weighted concentration term ({symbols[3]})']
    if method == 'SNV':
        titles[0] = f'Aqueous SNV-CLS coefficient ({symbols[0]})'
        titles[1] = f'Organic SNV-CLS coefficient ({symbols[1]})'
    return titles, symbols


def plot_components(table, selected, method, folder):
    _, symbols = response_labels(method)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, label in enumerate(symbols):
        ax.plot(table.components, table[RESPONSES[i]], marker='o', label=label)
    ax.plot(table.components, table['mean'], marker='s', ls='--', lw=2,
            label='Mean of four responses')
    ax.axvline(selected, ls=':', lw=1.7, label=f'Selected = {selected}')
    ax.set(xlabel='Number of PLS components', ylabel='NRMSECV (%)')
    ax.set_xticks(np.arange(1, MAX_COMPONENTS + 1))
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    save_show(fig, folder, f'{method}_Component_Selection.png')


def plot_responses(method, y_train, y_test, p_train, p_test, cv_nrmse,
                   folder):
    titles, symbols = response_labels(method)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2), dpi=300)
    axes = axes.ravel()
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    for j, ax in enumerate(axes):
        a, b, c, d = y_train[:, j], p_train[:, j], y_test[:, j], p_test[:, j]
        low, high = axis_limits(a, b, c, d)
        line = np.linspace(low, high, 200)
        ax.plot(line, line, color='black', lw=1.35, zorder=1)
        ax.scatter(a, b, marker='^', s=54, edgecolor='black', linewidth=0.40,
                   alpha=0.72, zorder=3, label='Train')
        ax.scatter(c, d, marker='o', s=54, edgecolor='black', linewidth=0.45,
                   alpha=0.86, zorder=4, label='Test')
        ax.set_title(titles[j], pad=5, fontsize=12.5)
        ax.set_xlabel(f'Reference {symbols[j]}')
        ax.set_ylabel(f'Predicted {symbols[j]}')
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect('equal', adjustable='box')
        ax.text(0.03, 0.97, panel_labels[j], transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='top', ha='left')
        ax.text(0.03, 0.85,
                f'NRMSECV = {cv_nrmse[j]:.2f}%\n'
                f'Test $R^2$ = {r2_score(c, d):.3f}\n'
                f'Test RMSE = {rmse(c, d):.3g}',
                transform=ax.transAxes, fontsize=10.8, va='top', ha='left')
    axes[3].legend(loc='lower right', frameon=True, fontsize=9.8)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.075, top=0.965,
                        wspace=0.20, hspace=0.34)
    save_show(fig, folder, f'{method}_Four_Response_Parity_2x2.png')


def plot_thresholds(table, cutoff, method, folder):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), dpi=300)
    x = table.cutoff * 100

    axes[0].plot(x, table['aq_nrmsecv'], marker='o', label=r'$c_{aq}$')
    axes[0].plot(x, table['org_nrmsecv'], marker='o', label=r'$c_{org}$')
    axes[0].axvline(cutoff * 100, linestyle=':', linewidth=1.6,
                    label=f'Selected = {cutoff:.0%}')
    axes[0].set_xlabel('Minimum relative phase contribution (%)')
    axes[0].set_ylabel('Concentration NRMSECV (%)')
    axes[0].set_title('Cross-validation error')
    axes[0].legend(frameon=True, fontsize=9)

    axes[1].plot(x, table['aq_retained_pct'], marker='o', label='Aqueous')
    axes[1].plot(x, table['org_retained_pct'], marker='o', label='Organic')
    axes[1].axvline(cutoff * 100, linestyle=':', linewidth=1.6,
                    label=f'Selected = {cutoff:.0%}')
    axes[1].set_xlabel('Minimum relative phase contribution (%)')
    axes[1].set_ylabel('Out-of-fold training predictions retained (%)')
    axes[1].set_ylim(0, 105)
    axes[1].set_title('Dataset coverage')
    axes[1].legend(frameon=True, fontsize=9)

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.16, top=0.91,
                        wspace=0.22)
    save_show(fig, folder, f'{method}_Phase_Contribution_Threshold_Sensitivity.png')


def plot_concentrations(method, train, test, cutoff, folder):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.35), dpi=300)
    plot_info = [
        ('aq', axes[0], r'Aqueous HNO$_3$ concentration', '(a)'),
        ('org', axes[1], r'Organic HNO$_3$ concentration', '(b)')
    ]

    for j, (phase, ax, title, panel) in enumerate(plot_info):
        true_tr, pred_tr, keep_tr = (train[x][:, j] for x in ('c', 'pred', 'keep'))
        true_te, pred_te, keep_te = (test[x][:, j] for x in ('c', 'pred', 'keep'))
        a, b = true_tr[keep_tr], pred_tr[keep_tr]
        c, d = true_te[keep_te], pred_te[keep_te]
        low, high = axis_limits(a, b, c, d)
        line = np.linspace(low, high, 200)
        ax.plot(line, line, color='black', linestyle='--', linewidth=1.35, zorder=1)
        ax.scatter(a, b, marker='^', s=50, edgecolor='black', linewidth=0.40,
                   alpha=0.68, zorder=3, label='Train')
        ax.scatter(c, d, marker='o', s=52, edgecolor='black', linewidth=0.45,
                   alpha=0.86, zorder=4, label='Test')

        unique_c = np.sort(np.unique(c))
        if len(unique_c) > 1:
            spacing = np.min(np.diff(unique_c))
            box_width = max(0.02, 0.10 * spacing)
        else:
            box_width = 0.05
        if len(unique_c):
            groups = [d[np.isclose(c, ref)] for ref in unique_c]
            ax.boxplot(groups, positions=unique_c, widths=box_width, patch_artist=False,
                       showfliers=False, showmeans=True, manage_ticks=False,
                       meanprops={'marker': 'D', 'markerfacecolor': 'white',
                                  'markeredgecolor': 'black', 'markersize': 4.0},
                       medianprops={'color': 'black', 'linewidth': 1.0},
                       whiskerprops={'color': 'black', 'linewidth': 0.8},
                       capprops={'color': 'black', 'linewidth': 0.8},
                       boxprops={'color': 'black', 'linewidth': 0.8},
                       zorder=2)

        symbol = rf'$c_{{\mathrm{{{phase}}}}}$'
        result = scores(true_te, pred_te, keep_te)
        ax.set_title(title, pad=5, fontsize=14)
        ax.set_xlabel(f'Reference {symbol} (M)')
        ax.set_ylabel(f'Predicted {symbol} (M)')
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect('equal', adjustable='box')
        ax.text(0.03, 0.97, panel, transform=ax.transAxes, fontsize=15,
                fontweight='bold', va='top', ha='left')
        ax.text(0.03, 0.84,
                f'Cutoff = {cutoff:.0%}\nTest $R^2$ = {result["R2"]:.3f}\n'
                f'Test RMSE = {result["RMSE"]:.3f} M',
                transform=ax.transAxes, fontsize=10.8, va='top', ha='left')

    axes[1].legend(loc='lower right', frameon=True, fontsize=9.8)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.14, top=0.92, wspace=0.15)
    save_show(fig, folder, f'{method}_Concentration_Parity_1x2.png')


def plot_nrmsecv_comparison(component_tables):
    raw = component_tables['Raw']
    norm = component_tables['SNV']
    r = raw.loc[raw['selected'], RESPONSES].iloc[0].to_numpy(dtype=float)
    s = norm.loc[norm['selected'], RESPONSES].iloc[0].to_numpy(dtype=float)
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for vals, off, label, color in [(r, -.17, 'Non-normalized', COLORS['Raw']),
                                    (s, .17, 'SNV-normalized', COLORS['SNV'])]:
        bars = ax.bar(x + off, vals, width=.34, color=color,
                      edgecolor='black', lw=.6, label=label)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, value, f'{value:.2f}%',
                    ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x, [rf'${z}_{{\mathrm{{{p}}}}}$' for z, p in
                      (('f', 'aq'), ('f', 'org'), ('q', 'aq'), ('q', 'org'))])
    ax.set(xlabel='Response variable', ylabel='NRMSECV (%)')
    ax.legend()
    fig.tight_layout()
    save_show(fig, COMPARE_DIR, 'Raw_vs_SNV_NRMSECV_Comparison.png')


def plot_percentage_error_by_concentration(raw_test, snv_test):
    """Compare held-out errors using samples accepted by both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, j, phase in zip(axes, (0, 1), ('Aqueous', 'Organic')):
        both = raw_test['keep'][:, j] & snv_test['keep'][:, j]
        y_ref = raw_test['c'][:, j]
        levels = np.sort(np.unique(y_ref[both & (y_ref > 0)]))
        x = np.arange(len(levels))
        for i, (label, data) in enumerate((('Raw', raw_test), ('SNV', snv_test))):
            means, stdevs = [], []
            for level in levels:
                selected = both & np.isclose(y_ref, level)
                deviations = 100 * np.abs((data['pred'][selected, j] - level)/level)
                means.append(deviations.mean() if len(deviations) else np.nan)
                stdevs.append(deviations.std(ddof=1) if len(deviations) > 1 else np.nan)
            ax.bar(x + (i - .5)*.35, means, width=.35, color=COLORS[label],
                   yerr=np.nan_to_num(stdevs, nan=0), capsize=4,
                   error_kw={'elinewidth': 1.2, 'capthick': 1.2},
                   label={'Raw': 'Non-normalized', 'SNV': 'SNV-normalized'}[label], zorder=3)
        ax.set(title=f'{phase} HNO$_3$', xlabel='Reference concentration (M)',
               ylabel='Mean absolute percentage deviation (%)')
        ax.set_xticks(x, [f'{v:g}' for v in levels])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -.02),
               ncol=2, frameon=False, fontsize=12)
    fig.tight_layout(rect=[0, .08, 1, 1])
    save_show(fig, COMPARE_DIR, 'Raw_vs_SNV_Percentage_Error_By_Concentration.png')


def train_model(method, X, responses, train_idx, test_idx, strata, files, folder):
    Y, c = responses[:, :4], responses[:, 4:6]
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    ranges = np.ptp(Y_train, axis=0)
    # Select latent variables using minimum mean NRMSECV.
    comp_rows = []
    for n in range(1, MAX_COMPONENTS + 1):
        pred, _ = cv_predict(X_train, Y_train, strata[train_idx], n)
        errors = np.divide(100 * rmse(Y_train, pred), ranges,
                           out=np.full(4, np.nan), where=ranges != 0)
        comp_rows.append([n, *errors, np.nanmean(errors)])
    comps = pd.DataFrame(comp_rows, columns=['components', *RESPONSES, 'mean'])
    n = int(comps.loc[comps['mean'].idxmin(), 'components'])
    comps['selected'] = comps.components == n
    cv_pred, folds = cv_predict(X_train, Y_train, strata[train_idx], n)
    # Select the phase cutoff from out-of-fold predictions.
    cutoff, cutoffs = select_cutoff(c[train_idx], cv_pred, folds)
    cutoffs['selected'] = np.isclose(cutoffs.cutoff, cutoff)

    model = PLSRegression(n_components=n, scale=False).fit(X_train, Y_train)
    train_responses, test_responses = model.predict(X_train), model.predict(X_test)
    train_c, train_r = recover(train_responses)
    test_c, test_r = recover(test_responses)
    train = {'c': c[train_idx], 'pred': train_c,
             'keep': (train_r >= cutoff) & np.isfinite(train_c)}
    test = {'c': c[test_idx], 'pred': test_c,
            'keep': (test_r >= cutoff) & np.isfinite(test_c)}


    prediction = pd.DataFrame({'filename': [files[i].name for i in test_idx]})
    for j, phase in enumerate(('aq', 'org')):
        prediction[f'c_{phase}_reference_M'] = test['c'][:, j]
        prediction[f'c_{phase}_predicted_M'] = test['pred'][:, j]
        prediction[f'c_{phase}_accepted'] = test['keep'][:, j]
        prediction[f'r_{phase}_predicted'] = test_r[:, j]
    prediction.to_csv(folder / 'test_predictions.csv', index=False)
    comps.to_csv(folder / 'component_selection.csv', index=False)
    cutoffs.to_csv(folder / 'cutoff_selection.csv', index=False)

    cv_nrmse = comps.loc[comps.selected, RESPONSES].iloc[0].to_numpy(dtype=float)
    results = []
    for j, key in enumerate(RESPONSES):
        results.append({'response': key, 'n': len(test_idx),
                        'RMSE': float(rmse(Y_test[:, j], test_responses[:, j])),
                        'R2': float(r2_score(Y_test[:, j], test_responses[:, j]))})
    for j, phase in enumerate(('Aqueous', 'Organic')):
        stats = scores(test['c'][:, j], test['pred'][:, j], test['keep'][:, j])
        results.append({'response': f'{phase} HNO3 (M)', **stats})
    pd.DataFrame(results).to_csv(folder / 'test_performance.csv', index=False)
    joblib.dump({'model': model, 'cutoff': cutoff, 'use_snv': method == 'SNV'},
                folder / f'{method.lower()}_plsr.joblib')

    plot_components(comps, n, method, folder)
    plot_responses(method, Y_train, Y_test, train_responses, test_responses,
                   cv_nrmse, folder)
    plot_thresholds(cutoffs, cutoff, method, folder)
    plot_concentrations(method, train, test, cutoff, folder)

    print(f'{method}: {n} PLS components, cutoff {cutoff:.0%}')
    for row in results[-2:]:
        print(f'  {row["response"]}: RMSE={row["RMSE"]:.4f} M; '
              f'R2={row["R2"]:.4f}; n={row["n"]}')
    return comps, test


def main():
    for path in (RAW_DIR, SNV_DIR, COMPARE_DIR):
        path.mkdir(parents=True, exist_ok=True)
    files, X_raw, raw, normalized = load_data()
    # Keep the same train/test split for raw and SNV models.
    labels = np.array([f'{a:.8f}|{b:.8f}' for a, b in raw[:, 4:6]])
    train_idx, test_idx = train_test_split(np.arange(len(files)),
                                          test_size=TEST_SIZE, random_state=SEED,
                                          shuffle=True, stratify=labels)
    tables, test_predictions = {}, {}
    for method, X, responses, folder in [
        ('Raw', X_raw, raw, RAW_DIR),
        ('SNV', snv(X_raw), normalized, SNV_DIR),
    ]:
        tables[method], test_predictions[method] = train_model(
            method, X, responses, train_idx, test_idx, labels, files, folder)

    plot_nrmsecv_comparison(tables)
    plot_percentage_error_by_concentration(test_predictions['Raw'],
                                           test_predictions['SNV'])

    rows = []
    for j, phase in enumerate(('Aqueous', 'Organic')):
        raw_test, snv_test = test_predictions['Raw'], test_predictions['SNV']
        common = raw_test['keep'][:, j] & snv_test['keep'][:, j]
        for method, data in [('Raw', raw_test), ('SNV', snv_test)]:
            stat = scores(data['c'][:, j], data['pred'][:, j], common)
            rows.append({'method': method, 'phase': phase, **stat})
    pd.DataFrame(rows).to_csv(COMPARE_DIR / 'matched_test_comparison.csv', index=False)
    print('Plots generated: Figures 6 and 7; Supporting Information Figures S6-S11; '
          'non-normalized versions of Figures S6 and S8.')


if __name__ == '__main__':
    main()
