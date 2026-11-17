PHASE-RESOLVED HNO3 QUANTIFICATION FROM MIXED RAMAN SPECTRA
=========================================================

Script: phase_resolved_quantification.py

PURPOSE
-------
Trains and evaluates separate non-normalized (Raw) and standard normal
variate (SNV) PLSR models for estimating HNO3 concentrations in the aqueous
and organic phases from mixed biphasic Raman spectra.

This script uses CLS-derived response vectors prepared separately. It does
not perform the initial CLS decomposition.

REQUIREMENTS
------------
Python packages: numpy, pandas, matplotlib, scikit-learn, joblib, openpyxl

Install, if needed:
    pip install numpy pandas matplotlib scikit-learn joblib openpyxl

INPUT FILES
-----------
1. Raman CSV spectra directly inside DATA_DIR:
   - First column: Raman shift
   - Second column: Raman intensity
   - One header row (read using pandas' default header)
   - Identical Raman-shift grids for all files
   - Filenames containing numbers, sorted in numeric order

2. Excel response files inside DATA_DIR/response_vectors/:
   - raw_response_vectors.xlsx
   - SNV_non_corrected_response_vectors.xlsx

   The first six columns, in the same order as the sorted CSV files, must be:
       f_aq, f_org, q_aq, q_org, c_aq (M), c_org (M)

   Here q = 1000 * f * c; c is in mol/L (M). The phase-contribution
   coefficients describe Raman spectral contributions, not physical
   phase-volume fractions.

   The reference concentrations c_aq and c_org were determined from
   single-phase Raman measurements of the aqueous and organic phases
   collected after slug-flow extraction and downstream phase separation,
   using the respective phase-specific calibration models. These values
   are used for PLSR model development and evaluation.

HOW TO RUN
----------
1. Set DATA_DIR near the top of the script to the folder containing your
   Raman CSV files. Update the two Excel paths if necessary.
2. Run:
       phase_resolved_quantification.py
3. The script creates output folders automatically. If figure windows open,
   close them to let the script continue.

METHOD SUMMARY
--------------
- Uses the same stratified 70:30 train/test split for Raw and SNV models.
- Tests 1-20 PLS latent variables using stratified 10-fold cross-validation.
- Selects the number with the lowest mean NRMSECV across four responses.
- Selects a minimum relative phase-contribution cutoff using the
  one-standard-error rule on cross-validated concentration errors.
- Fits final four-response PLSR models and evaluates held-out test spectra.
- Estimates each phase's concentration from predicted q/(1000*f).

OUTPUTS
-------
PLSR_Raw_SNV_Final_Workflow/
    Raw_SI/
    SNV_Main_Text/
    Raw_vs_SNV_Comparison/

The Raw and SNV folders contain model-selection plots, four-response parity
plots, cutoff-sensitivity plots, concentration parity plots, and CSV tables
for component selection, cutoff selection, test predictions, and test
performance. Each also contains a saved PLSR model (*.joblib), including
its selected cutoff and SNV setting.

The comparison folder contains Raw-versus-SNV error plots and a matched-test
performance CSV.

MANUSCRIPT FIGURES
------------------
This script is used to generate Figures 6 and 7; Supporting Information
Figures S6-S11; and non-normalized versions of Figures S6 and S8.

