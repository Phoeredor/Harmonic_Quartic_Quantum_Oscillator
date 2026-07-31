# Documentation

This directory contains short reproducibility notes for the Path-Integral Monte Carlo implementation of the one-dimensional quantum harmonic oscillator.

The main project overview, build instructions, workflow commands, figures, and references are provided in the repository-level README.md.

## Model

The code uses dimensionless variables with hbar omega = 1. The Euclidean lattice spacing is eta = beta / Nt.

The lattice action is

S[y] = sum_j [ (y_{j+1} - y_j)^2 / (2 eta) + eta y_j^2 / 2 ],

with periodic boundary conditions y_Nt = y_0.

Periodic boundary conditions are used because the simulations estimate the thermal trace Z(beta) = Tr exp(-beta H).

## Monte Carlo updates

The public executable supports:

- local Metropolis updates;
- exact local heatbath updates for the harmonic oscillator;
- heatbath plus microcanonical overrelaxation sweeps.

The final thermodynamic production uses the heatbath-plus-overrelaxation update, which reduces autocorrelations while preserving the target distribution.

## Main observables

The analysis focuses on:

- continuum extrapolation of <y^2>;
- renormalized thermodynamic energy estimators;
- Euclidean position distributions P(y);
- imaginary-time correlators and harmonic-oscillator energy gaps.

Generated raw and processed data are intentionally excluded from the public repository. The retained report PNG figures are available under `../plots/`.
