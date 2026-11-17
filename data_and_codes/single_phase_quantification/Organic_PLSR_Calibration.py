
# -*- coding: utf-8 -*-
"""SNV-normalized Raman PLSR calibration of HNO3 in the organic phase."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_predict

# ---------------- SETTINGS ----------------

DATA_DIR = Path('path/to/folder/data_and_codes/single_phase_quantification/single_phase_data/')

EXCEL_FILE = (r'path/to/folder/data_and_codes/single_phase_quantification/organic_phase_calibration/Organic_phase_calibration.xlsx')

OUTPUT_DIR = Path(__file__).resolve().parent

wn_min, wn_max = None, None  # Use full spectrum
TEST_SIZE = 0.30
RANDOM_STATE = 69
CV_FOLDS = 5
MAX_COMPONENTS = 20

plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'Arial',
    'axes.labelsize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'axes.titlesize': 16,
})

# ---------------- LOAD DATA ----------------
# Excel: column A = original sample number; column C = HNO3 concentration (M).
responses = pd.read_excel(EXCEL_FILE, skiprows=1).dropna()
sample_nums = responses.iloc[:, 0].astype(int).to_numpy()
Y = responses.iloc[:, 2].astype(float).to_numpy()

X = []
wavenumber = None
for n in sample_nums:
    filename = DATA_DIR / f'Sample ({n}).csv'
    spectrum = pd.read_csv(filename)
    wn = spectrum.iloc[:, 0].to_numpy(dtype=float)
    intensity = spectrum.iloc[:, 1].to_numpy(dtype=float)
    if wavenumber is None:
        wavenumber = wn
    elif wn.shape != wavenumber.shape or not np.allclose(wn, wavenumber):
        raise ValueError(f'Wavenumber grid mismatch: {filename}')
    X.append(intensity)

X = np.asarray(X)
if wn_min is not None and wn_max is not None:
    mask = (wavenumber >= wn_min) & (wavenumber <= wn_max)
    X = X[:, mask]
    wavenumber = wavenumber[mask]

# ---------------- SNV AND TRAIN/TEST SPLIT ----------------
# Standard normal variate is applied independently to each spectrum.
std = X.std(axis=1, ddof=1, keepdims=True)
std[std == 0] = 1.0
X = (X - X.mean(axis=1, keepdims=True)) / std

indices = np.arange(len(Y))
idx_train, idx_test = train_test_split(
    indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
idx_train, idx_test = np.sort(idx_train), np.sort(idx_test)
X_train, X_test = X[idx_train], X[idx_test]
Y_train, Y_test = Y[idx_train], Y[idx_test]

# ---------------- CROSS-VALIDATION ----------------
cv = KFold(n_splits=min(CV_FOLDS, len(Y_train)),
           shuffle=True, random_state=RANDOM_STATE)
smallest_fold = min(len(train) for train, _ in cv.split(X_train))
n_max = min(MAX_COMPONENTS, X_train.shape[1], smallest_fold - 1)
if n_max < 1:
    raise ValueError('Not enough data to fit PLSR.')

components = np.arange(1, n_max + 1)
rmsecv = []
for n in components:
    model = PLSRegression(n_components=int(n), scale=False)
    predicted = cross_val_predict(model, X_train, Y_train, cv=cv).ravel()
    predicted = np.maximum(predicted, 0)
    rmsecv.append(np.sqrt(mean_squared_error(Y_train, predicted)))

rmsecv = np.asarray(rmsecv)
best_n = int(components[np.argmin(rmsecv)])

# ---------------- FINAL MODEL ----------------
model = PLSRegression(n_components=best_n, scale=False)
model.fit(X_train, Y_train)
Yhat_train = np.maximum(model.predict(X_train).ravel(), 0)
Yhat_test = np.maximum(model.predict(X_test).ravel(), 0)

r2_train, r2_test = r2_score(Y_train, Yhat_train), r2_score(Y_test, Yhat_test)
rmse_train = np.sqrt(mean_squared_error(Y_train, Yhat_train))
rmse_test = np.sqrt(mean_squared_error(Y_test, Yhat_test))

print(f'Samples: {len(Y)} (train: {len(Y_train)}, test: {len(Y_test)})')
print(f'Optimal PLS components: {best_n}; RMSECV: {rmsecv.min():.4f} M')
print(f'Train: R² = {r2_train:.4f}, RMSE = {rmse_train:.4f} M')
print(f'Test:  R² = {r2_test:.4f}, RMSE = {rmse_test:.4f} M')

# ---------------- PLOTS ----------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(components, rmsecv, marker='o', lw=1.5)
ax.scatter(best_n, rmsecv.min(), s=100, edgecolor='black', zorder=3,
           label=f'Selected = {best_n}')
ax.set_xlabel('Number of PLS components')
ax.set_ylabel('RMSECV (M)')
ax.set_xticks(components)
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'organic_plsr_rmsecv.jpg',
            dpi=300, facecolor='white')

fig, ax = plt.subplots(figsize=(8, 8))
m = 1.10 * max(Y_train.max(), Y_test.max(), Yhat_train.max(), Yhat_test.max())
ax.plot([0, m], [0, m], 'k-', lw=1.5, label='Ideal', zorder=1)
ax.scatter(Y_train, Yhat_train, marker='^', color='blue', s=110,
           edgecolor='k', linewidth=0.6, alpha=0.85, zorder=3,
           label=f'Train (R²={r2_train:.3f}, RMSE={rmse_train:.3g})')
ax.scatter(Y_test, Yhat_test, marker='s', color='red', s=70,
           edgecolor='k', linewidth=0.7, alpha=0.95, zorder=4,
           label=f'Test (R²={r2_test:.3f}, RMSE={rmse_test:.3g})')
ax.set_xlabel('Known HNO$_3$ (M) in organic phase')
ax.set_ylabel('Predicted HNO$_3$ (M) in organic phase')
ax.set_xlim(-0.05, m)
ax.set_ylim(-0.05, m)
ax.set_title('Preprocessing: SNV Normalization')
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'organic_plsr_parity.jpg',
            dpi=300, facecolor='white')

plt.show()

print('This code generates plot Figure S3(b).')
