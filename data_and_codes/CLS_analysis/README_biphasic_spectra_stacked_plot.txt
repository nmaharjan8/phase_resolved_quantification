BIPHASIC RAMAN SPECTRA: STACKED PLOT
====================================

Script: biphasic_spectra_stacked_plot.py

This script plots the mixed biphasic Raman spectra from a selected experimental
set. It reads all Sample (N).csv files directly in the selected folder, sorts
them numerically by sample number, and displays the spectra using a green color
gradient. The spectra are plotted without normalization or baseline subtraction.
The resulting plot is used for Figure 4.

EXPERIMENTAL DATASETS
---------------------
The six folders contain mixed biphasic Raman spectra measured at the following
HNO3 concentrations. All concentrations are in mol/L (M).

Folder                    Aqueous HNO3 (M)   Organic HNO3 (M)
biphasic_spectra_set_1         0.0000              0.0000
biphasic_spectra_set_2         0.4238              0.0762
biphasic_spectra_set_3         0.7816              0.2184
biphasic_spectra_set_4         2.0367              0.4633
biphasic_spectra_set_5         2.9118              0.5882
biphasic_spectra_set_6         4.2155              0.7845

INPUT AND OUTPUT
----------------
- Set FOLDER to the path of the desired biphasic_spectra_set_N folder.
- Input spectra are named Sample (N).csv. Their first two spectral data columns
  contain wavenumber and Raman intensity, respectively. The script skips the
  first row when reading the exported CSV files.
- The default setting uses biphasic_spectra_set_4 for Figure 4.
- Output: Raman_spectra_biphasic_spectra_stacked_plot.jpg (300 dpi).
- The figure is saved in the same folder as the Python script (OUTPUT_DIR),
  not inside the selected dataset folder unless the script is stored there.

The script requires Python with numpy, pandas, and matplotlib.
