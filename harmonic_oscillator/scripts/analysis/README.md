# Analysis Scripts

This directory contains Python analysis programs for the quantum harmonic
oscillator PIMC data produced by `bin/qho_pimc`.

The scripts are divided into final production analyses, spectrum analyses,
Monte Carlo consistency checks, and shared readers. Most programs read either
raw measurement files from `data/raw/` or processed block files from
`data/processed/`, then write tables used by the plotting scripts and by the
final figures.

Runtime data are not distributed with the repository. These programs assume
that the required simulations or preceding analyses have been run locally;
their generated outputs below `data/` remain ignored by Git. The retained PNG
report figures are available below `plots/`.

## Final production analyses

`analyze_thermo_beta5.py` analyzes the clean beta=5 thermodynamic production
runs. It blocks saved measurements, writes long and wide thermodynamic tables,
and performs weighted continuum fits in `eta^2` for `<y^2>` and `H_ren`.

`analyze_temperature_bare_scan.py` analyzes the temperature scan of the bare
thermodynamic estimator and compares the measured behavior with the exact
harmonic-oscillator thermal result.

`analyze_qho_position_distribution.py` is the final position-distribution
analysis. It reads block histograms, rebins adjacent Monte Carlo blocks,
estimates bin and moment uncertainties from block dispersion, and compares
integrated-bin probabilities with the exact Gaussian distribution.

## Spectrum analyses

`analyze_spectrum_blocks.py` analyzes block-resolved spectrum correlators with
a leave-one-block jackknife. Connected correlators are reconstructed inside
each jackknife sample before extracting gap estimates.

## Monte Carlo consistency checks

`analyze_autocorr.py` estimates normalized autocorrelation functions and
integrated autocorrelation times for saved measurement series. It also prints
conservative suggestions for measurement stride, blocking size, and discarded
thermalization length.

`analyze_sampling_efficiency.py` compares local update choices using
autocorrelation estimates, slowest-observable diagnostics, and rough CPU-cost
projections. It is a consistency comparison, not a final production analysis.

`analyze_equilibration.py` compares starts and update settings for the
heatbath-overrelaxation chain. It is used to choose stable Monte Carlo
parameters for later production runs.

## Shared readers

`qho_binary_io.py` reads the fixed-size binary measurement format written by
the C executable. It is used by plotting and analysis scripts that need access
to binary production data.

Use each script only after its required local runtime inputs have been
generated. Command-line options and expected schemas are documented by the
individual scripts.
