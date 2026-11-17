
# -*- coding: utf-8 -*-
"""HNO3 Raman calibration: pre-contact and post-contact.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- STYLE: consistent with corrected aqueous plots ----------
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 18,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 14,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.1,
    'ytick.major.width': 1.1,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
})

# ---------- ORIGINAL DATA ----------
cal_conc = np.array([0, 0.25, 0.5, 0.75, 1., 1.5, 2.5, 3.5, 5.])
cal_int = np.array([2.02, 1169.90, 2220.27, 3335.00, 4566.61,
                    6529.76, 9924.36, 12385.10, 14679.86])
cal_std = np.array([3.85, 4.78, 4.53, 24.07, 11.99,
                    9.92, 15.25, 55.03, 35.38])
post_conc = np.array([0.5, 1., 2.5, 3.5, 5.])
post_int = np.array([2018.65, 3611.49, 8392.39, 10983.43, 13712.04])
post_std = np.array([19.71, 21.21, 79.99, 112.19, 32.18])

for name, xx, yy, ss in [('Pre-contact',cal_conc,cal_int,cal_std),
                          ('Post-contact',post_conc,post_int,post_std)]:
    if not (len(xx)==len(yy)==len(ss)) or not np.all(np.isfinite(np.r_[xx,yy,ss])) or np.any(ss<0):
        raise ValueError(f'Invalid {name} data')

# ---------- FIT y = a*x^2 + b*x, NO INTERCEPT ----------
# np.linalg.lstsq uses same unweighted least-squares objective as the original sklearn model.
b, a = np.linalg.lstsq(np.column_stack([cal_conc, cal_conc**2]), cal_int, rcond=None)[0]
cal_pred = a*cal_conc**2 + b*cal_conc
r2 = 1 - np.sum((cal_int-cal_pred)**2)/np.sum((cal_int-cal_int.mean())**2)
print(f'Calibration: y = {a:.5f} x² + {b:.5f} x; R² = {r2:.6f}')

check_x = np.linspace(0,5,1000)
if np.any(np.diff(a*check_x**2+b*check_x)<=0):
    raise ValueError('Calibration not strictly monotonic on [0,5] M')

def invert_intensities(ys):
    out = []
    for yy in ys:
        roots = np.roots([a,b,-yy])
        valid = roots[np.isclose(roots.imag,0,atol=1e-8)].real
        valid = valid[(valid>=-1e-8)&(valid<=5+1e-8)]
        out.append(float(valid[0]) if len(valid)==1 else np.nan)
    return np.array(out)

post_pred_conc = invert_intensities(post_int)
for initial, predicted in zip(post_conc, post_pred_conc):
    print(f'Initial {initial:.3f} M | Post-contact {predicted:.4f} M | Delta {initial-predicted:.4f} M')

# ---------- PLOT ----------
fig, ax = plt.subplots(figsize=(10,8), dpi=300)
x_fit = np.linspace(0,5.5,500)
y_fit = a*x_fit**2+b*x_fit
ax.plot(x_fit,y_fit,'--',color='black',lw=1.6,zorder=1)

# Light guides only at 0.25, 0.50, 0.75 M, as in original figure.
special_heights = {0.25:7500,0.5:8500,0.75:9500}
for x,height in special_heights.items():
    ax.vlines(x,0,height,colors='blue',linestyles=':',lw=2.5,alpha=0.60,zorder=0)

# Pre-contact and post-contact points
ax.errorbar(
    cal_conc, cal_int, yerr=cal_std,
    fmt='D', color='blue', ecolor='blue',
    elinewidth=1.2, capsize=3, markersize=8,
    label='Pre-contact', zorder=3
)
ax.errorbar(
    post_pred_conc, post_int, yerr=post_std,
    fmt='*', color='red', ecolor='red',
    elinewidth=1.2, capsize=3, markersize=14,
    label='Post-contact', zorder=4
)

# ---------- ANNOTATIONS ----------
ANNOTATION_FONT = 15
for x, height in special_heights.items():
    ax.annotate(
        f'{x:g} M', xy=(x, height), xytext=(0, 5),
        textcoords='offset points', fontsize=ANNOTATION_FONT,
        color='blue', ha='center', va='bottom', zorder=6
    )
for x, y in zip(post_pred_conc, post_int):
    if np.isfinite(x):
        ax.annotate(
            f'{x:.4f} M', xy=(x, y), xytext=(0, 12),
            textcoords='offset points', fontsize=ANNOTATION_FONT,
            color='red', ha='center', va='bottom', zorder=7
        )

ax.set_xlabel('HNO$_3$ Concentration (M)')
ax.set_ylabel('Baseline Corrected Intensity (a.u.)')
ax.set_xlim(-0.15,5.6)
ax.set_ylim(-350,max(np.max(cal_int+cal_std),np.max(post_int+post_std),np.max(y_fit))*1.08)
ax.set_xticks([0,1,2,3,4,5])
ax.set_yticks([0,4000,8000,12000,16000])
ax.legend(loc='upper left',frameon=True,fontsize=14)
fig.subplots_adjust(left=0.15,right=0.97,bottom=0.15,top=0.96)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_STEM = os.path.join(OUTPUT_DIR, 'raman_calibration_balanced')

fig.savefig(f'{OUTPUT_STEM}.jpg', dpi=300, facecolor='white')

print(f'Saved {OUTPUT_STEM}.jpg')
print('This code generates the plot used for Figure 1(d) and data used in this code were obtained from baseline subtraction statistics from Raman_Aqueous.py')
plt.show()
