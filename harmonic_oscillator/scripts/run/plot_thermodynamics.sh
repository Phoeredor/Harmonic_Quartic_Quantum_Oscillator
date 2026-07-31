#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/run/plot_thermodynamics.sh
# Purpose: Generate final thermodynamic plots from processed data.
# The figures compare blocked PIMC observables and continuum extrapolations with
# exact harmonic-oscillator expectations and representative Euclidean paths.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: workflow configuration
# -----------------------------------------------------------------------------

set -euo pipefail

# Render continuum thermodynamics and representative Euclidean paths.
mkdir -p plots/euclidean_path plots/thermodynamics
python3 scripts/plotting/plot_thermodynamics.py

# List the generated physical summaries in their final output locations.
printf '\nFinal plot files:\n'
find plots/euclidean_path plots/thermodynamics -maxdepth 1 -type f -name 'fig_*.png' -print | sort
