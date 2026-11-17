Single-Phase Raman Analysis and Organic-Phase HNO3 Calibration
=============================================================

Python scripts and Raman data used to characterize the aqueous and organic
phases before and after nitric acid extraction under PUREX-relevant conditions.

DATA AND SAMPLE NUMBERING
-------------------------
The original acquisition identifiers, Sample (N).csv, are preserved. The
released data are drawn from these portions of the experimental sequence:

  Samples 6-10:     Deionized water
  Samples 55-134:   Aqueous-phase measurements
  Samples 179-219:  Organic-phase measurements

Not every sample number in these ranges is present, and not every included
spectrum is used in the figures or calibration models. Some measurements were
collected during flow-cell cleaning, after contamination was observed, or while
re-establishing consistent, uncontaminated spectra. A small number of spectra
between concentration sets were excluded because the measurements had not yet
stabilized. These files are retained to document the experimental sequence.
Gaps in sample numbering are intentional.

AQUEOUS SPECTRA USED FOR PEAK-HEIGHT ANALYSIS
---------------------------------------------
The aqueous Raman script groups the following original sample IDs for
baseline-corrected spectra and peak-height statistics at approximately
1048 cm-1. Ranges are inclusive.

Pre-contact:
  Deionized water:  Samples 6-10
  250 mM HNO3:     Samples 56-60
  500 mM HNO3:     Samples 62-66
  750 mM HNO3:     Samples 68-72
  1 M HNO3:       Samples 74-78
  1.5 M HNO3:     Samples 80-84
  2.5 M HNO3:     Samples 86-90
  3.5 M HNO3:     Samples 92-96
  5 M HNO3:       Samples 98-102

Post-contact:
  500 mM HNO3:     Samples 105-109
  1 M HNO3:       Samples 112-116
  2.5 M HNO3:     Samples 118-122
  3.5 M HNO3:     Samples 124-128
  5 M HNO3:       Samples 130-134

These sample assignments follow PRE_GROUPS and POST_GROUPS in
Raman_Aqueous.py. The post-contact labels refer to the initial aqueous
HNO3 concentrations used for extraction, not the final concentrations
measured after extraction. Before interpreting post-contact group means
and standard deviations as replicate statistics, confirm that every
listed sample is a valid measurement of its assigned condition.

PYTHON SCRIPTS
--------------
Raman_Aqueous.py
  Plots aqueous Raman spectra, pre-/post-contact comparisons, and separate
  baseline-corrected spectra for each contact condition. Uses a straight-line
  baseline from 1025 to 1070 cm-1 and evaluates the intensity nearest
  1048 cm-1. Exports individual peak heights and group mean/SD statistics.

Raman_Calibration.py
  Fits a quadratic aqueous HNO3 calibration with no intercept and estimates
  post-contact aqueous HNO3 concentrations from measured intensities.
  The calibration and post-contact intensity statistics are entered directly
  in this script, rather than read automatically from the exported CSV files.

Raman_Organic.py
  Plots full-range and expanded-region organic-phase Raman spectra using
  selected original sample numbers.

Organic_PLSR_Calibration.py
  Calibrates HNO3 concentration in the organic phase from Raman spectra using
  standard normal variate (SNV) normalization and partial least squares
  regression (PLSR). Uses a 70:30 training/test split (random_state=69),
  5-fold shuffled cross-validation on the training set, and selection of the
  number of PLS components by minimum RMSECV. Displays and exports RMSECV
  and measured-versus-predicted parity plots.

ORGANIC-PHASE CALIBRATION DATA
------------------------------
PUREX_Raman_study_org_Batch_9_slugflow_organic_phase.xlsx

The workbook associates each selected organic-phase Raman spectrum with an
HNO3 reference concentration established using the aqueous-phase Raman
calibration and extraction measurements. These are reference values, not
concentrations predicted by the organic-phase PLSR model.

  Column A: original sample number
  Column B: HNO3 concentration (mM)
  Column C: HNO3 concentration (M)

The workbook lists 35 spectra:

  Samples 181-184:  0 mM
  Samples 185-191:  76.2 mM
  Samples 193-197:  218.4 mM
  Samples 199-205:  463.3 mM
  Samples 207-213:  588.2 mM
  Samples 215-219:  784.5 mM

Only CSV files whose sample numbers appear in the workbook are used to fit
and evaluate the organic-phase PLSR model. Other CSV files in the data folder
are not loaded by this script.



RUNNING THE SCRIPTS
-------------------
1. Run each script separately. Generated figures and, where
   applicable, peak-statistics CSV files are saved beside the Python scripts.

