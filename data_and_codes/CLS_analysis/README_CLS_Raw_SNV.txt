NON-NEGATIVE CLASSICAL LEAST SQUARES (CLS) ANALYSIS
==================================================

Script: CLS_Raw_SNV.py

This script performs non-negative classical least squares (CLS) decomposition
of mixed biphasic Raman spectra using separate aqueous- and organic-phase
reference spectra. Non-negative CLS means that the fitted coefficients for
both spectral contributions are constrained to be greater than or equal to
zero. The resulting f_aq and f_org values represent relative spectral
contributions, not physical volume fractions; their sum is not constrained
to equal one.

Two analyses are performed on each mixed Raman spectrum:
- Raw CLS: non-negative CLS fitting of the original, non-normalized spectra.
- SNV CLS: standard normal variate (SNV) normalization of each spectrum,
  followed by non-negative CLS fitting. The SNV-derived coefficients are also
  converted back to the original intensity scale using the ratio of the
  mixture and reference spectral standard deviations.

EXPERIMENTAL DATASETS
---------------------
The folders contain mixed biphasic Raman spectra at the following measured
HNO3 concentrations. All concentrations are in mol/L (M).

Folder                    Aqueous HNO3 (M)   Organic HNO3 (M)
biphasic_spectra_set_1         0.0000              0.0000
biphasic_spectra_set_2         0.4238              0.0762
biphasic_spectra_set_3         0.7816              0.2184
biphasic_spectra_set_4         2.0367              0.4633
biphasic_spectra_set_5         2.9118              0.5882
biphasic_spectra_set_6         4.2155              0.7845

INPUTS
------
- Set DATA_DIR in the script to the biphasic_spectra_set_N folder to analyze.
  The default setting refers to biphasic_spectra_set_4. Change this single
  path to run the same analysis on sets 1 through 6.
- The selected folder contains the mixed Raman spectra named Sample (N).csv.
- Its reference_spectra subfolder must contain exactly two CSV files. In
  natural filename order, the first must be the aqueous-phase reference and
  the second the organic-phase reference.
- Each CSV is read using the first column as wavenumber and the second as
  intensity. The mixed spectra must share the reference wavenumber grid.

OUTPUTS
-------
The script automatically generates these three folders inside the selected
biphasic_spectra_set_N/CLS_results directory:

1. Raw_CLS_Fits
   Individual fits of the non-normalized mixed Raman spectra, displaying
   measured spectra, CLS reconstructions, and aqueous/organic contributions.

2. SNV_CLS_Fits
   Corresponding fits of the SNV-normalized mixed Raman spectra. The titles
   report the coefficients corrected to the original intensity scale.

3. Raw_vs_SNV_Parity
   Parity plots comparing the raw CLS coefficients with the corrected SNV-CLS
   coefficients, separately for the aqueous and organic phases.

The script also creates CLS_results/CLS_Phase_Contribution_Coefficients.xlsx,
which lists the raw CLS, direct SNV-CLS, and corrected SNV-CLS coefficients,
coefficient sums, and relative fitting residuals for each processed spectrum.
Plots are saved as 300 dpi JPG files.

FIGURES USED IN THE PUBLICATION
-------------------------------
The individual CLS fit plots used in Figure 5 and Supporting Information
Figure S5 are generated for Sample (4) in biphasic_spectra_set_4:

biphasic_spectra_set_4/CLS_results/Raw_CLS_Fits/
    Sample (4)_Raw_CLS_fit.jpg

biphasic_spectra_set_4/CLS_results/SNV_CLS_Fits/
    Sample (4)_SNV_CLS_fit.jpg

The raw CLS versus corrected SNV-CLS parity plots presented in Supporting
Information Figure S5 are generated in:

biphasic_spectra_set_4/CLS_results/Raw_vs_SNV_Parity/
    f_aq_Raw_vs_Corrected_SNV.jpg
    f_org_Raw_vs_Corrected_SNV.jpg

The same output structure is generated separately for any of the other five
experimental sets when DATA_DIR is changed accordingly.

The script requires Python with numpy, pandas, matplotlib, scipy,
scikit-learn, and openpyxl (for Excel export).
