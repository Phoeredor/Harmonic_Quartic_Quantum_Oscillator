#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/run/run_path_snapshots.sh
# Purpose: Generate representative Euclidean path snapshots.
# Several Nt values resolve the same beta interval, illustrating the approach
# from a coarse periodic lattice to a finely sampled continuum path.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: workflow configuration
# -----------------------------------------------------------------------------

set -euo pipefail

PATH_NT32="data/processed/qho_path_beta3_nt32.dat"
PATH_NT256="data/processed/qho_path_beta3_nt256.dat"
PATH_NT2048="data/processed/qho_path_beta3_nt2048.dat"
PATH_NT4096="data/processed/qho_path_beta3_nt4096.dat"

mkdir -p data/raw data/processed

# Build once for the fixed-beta sequence of lattice resolutions.
make clean
make

rm -f "$PATH_NT32" "$PATH_NT256" "$PATH_NT2048" "$PATH_NT4096" "$PLOT_FILE" \
      data/raw/qho_path_beta3_nt32_raw.dat \
      data/raw/qho_path_beta3_nt256_raw.dat \
      data/raw/qho_path_beta3_nt2048_raw.dat \
      data/raw/qho_path_beta3_nt4096_raw.dat

# Sample one equilibrated periodic path for a specified Nt and output location.
run_snapshot() {
  local nt="$1"
  local therm="$2"
  local sweeps="$3"
  local seed="$4"
  local stream="$5"
  local raw_out="$6"
  local path_out="$7"
  local label="$8"

  printf '\nRunning path snapshot %s: Nt=%s beta=3 seed=%s stream=%s\n' "$label" "$nt" "$seed" "$stream"
  ./bin/qho_pimc \
    --nt "$nt" \
    --beta 3.0 \
    --therm "$therm" \
    --sweeps "$sweeps" \
    --stride 100 \
    --seed "$seed" \
    --stream "$stream" \
    --init zero \
    --update hb-over \
    --n-over 5 \
    --delta 1.0 \
    --out "$raw_out" \
    --path-out "$path_out"

  printf '\nFirst 12 lines of %s:\n' "$path_out"
  head -n 12 "$path_out"
}

# Generate coarse-to-fine views of the same Euclidean interval beta=3.
run_snapshot 32 5000 5000 11111 61 \
  data/raw/qho_path_beta3_nt32_raw.dat "$PATH_NT32" "Nt=32, eta=3/32"

run_snapshot 256 5000 5000 22222 62 \
  data/raw/qho_path_beta3_nt256_raw.dat "$PATH_NT256" "Nt=256, eta=3/256"

run_snapshot 2048 2000 2000 33333 63 \
  data/raw/qho_path_beta3_nt2048_raw.dat "$PATH_NT2048" "Nt=2048, eta=3/2048"

run_snapshot 4096 2000 2000 44444 64 \
  data/raw/qho_path_beta3_nt4096_raw.dat "$PATH_NT4096" "Nt=4096, eta=3/4096"
