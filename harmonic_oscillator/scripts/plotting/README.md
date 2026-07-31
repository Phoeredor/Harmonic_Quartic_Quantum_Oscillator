# Plotting Scripts

`plot_thermodynamics.py` generates final report figures from existing QHO_PIMC
data files. It uses only Python, numpy, and matplotlib, and it never runs
simulations. Missing inputs are skipped with a warning.

Run:

```sh
make plot-thermodynamics
```

The required path snapshots and processed thermodynamic summaries are runtime
data and are not distributed. Generate them locally with the simulation and
analysis workflows before rerunning the plotter. Generated data below `data/`
remain ignored by Git.

Generated figures:

- `plots/euclidean_path/fig_path_snapshots_beta5.png`
- `plots/thermodynamics/fig_position_variance_continuum_beta5_eta2.png`
- `plots/thermodynamics/fig_renormalized_energy_continuum_beta5_eta2.png`
- `plots/thermodynamics/fig_continuum_fit_window_stability_beta5.png`
- `plots/thermodynamics/fig_position_variance_pull_beta5_eta.png`
- `plots/thermodynamics/fig_virial_estimator_check_beta5_eta.png`
- `plots/thermodynamics/fig_divergent_kinetic_term_beta5_eta.png`


`plot_position_distribution.py` is the canonical plotter for the
future schema-v1 processed tables. It prepares panel distributions,
`<y^2>` versus beta, and `beta<y^2>` versus beta; it does not run simulations.
The `position-distribution-plots` Make target uses `--variance` and writes only:

- `plots/distribution/fig_position_distribution_beta_scan.png`
- `plots/distribution/fig_position_variance_crossover.png`

The beta=5 thermodynamic report plots use only processed production summaries
and selected fit-window information generated locally. The retained report PNGs
are distributed under `plots/`.
