#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/run/run_thermo_beta5_production.sh
# Purpose: Run the beta-equals-five thermodynamic production workflow.
# A broad Nt scan at fixed beta measures <y^2> and H_ren for blocked eta^2
# continuum extrapolations and exact finite-temperature comparisons.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: workflow configuration
# -----------------------------------------------------------------------------

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"

cd "${project_root}"

THERMO_BETA="5"
THERMO_THERM="${THERMO_THERM:-200000}"
THERMO_SWEEPS="${THERMO_SWEEPS:-1000000}"
THERMO_STRIDE="${THERMO_STRIDE:-10}"
THERMO_BLOCK_SAVED="${THERMO_BLOCK_SAVED:-2000}"
THERMO_N_OVER="${THERMO_N_OVER:-5}"

RAW_DIR="data/raw/production/thermo_beta5"
PROCESSED_DIR="data/processed/production"
MANIFEST_FILE="${PROCESSED_DIR}/qho_thermo_beta5_manifest.dat"
NT_VALUES=(5 10 16 20 25 28 32 36 40 50 64 80 100 128 160 200 256 320 400 512)

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
    printf 'error: %s must be a positive integer, got %s\n' "$name" "$value" >&2
    exit 1
  fi
}

require_positive_int "THERMO_THERM" "$THERMO_THERM"
require_positive_int "THERMO_SWEEPS" "$THERMO_SWEEPS"
require_positive_int "THERMO_STRIDE" "$THERMO_STRIDE"
require_positive_int "THERMO_BLOCK_SAVED" "$THERMO_BLOCK_SAVED"
require_positive_int "THERMO_N_OVER" "$THERMO_N_OVER"

# Build the sampler for the fixed-beta lattice-spacing scan.
make

mkdir -p "$RAW_DIR" "$PROCESSED_DIR"
rm -f \
  "${PROCESSED_DIR}"/qho_thermo_beta5_*.dat \

printf '# QHO beta=5 thermodynamic production manifest\n' > "$MANIFEST_FILE"
printf '# beta eta Nt update init seed stream n_therm n_sweeps meas_stride n_over block_size_saved raw_file runtime_seconds seconds_per_sweep seconds_per_site_sweep\n' >> "$MANIFEST_FILE"

eta_from_nt() {
  python3 - "$THERMO_BETA" "$1" <<'PY'
import sys

beta = float(sys.argv[1])
nt = int(sys.argv[2])
print(f"{beta / nt:.17g}")
PY
}

format_seconds_division() {
  awk -v numerator="$1" -v denominator="$2" 'BEGIN { printf "%.12g", numerator / denominator }'
}

add_production_metadata() {
  local raw_file="$1"
  local run_label="$2"
  local run_date="$3"
  local nt="$4"
  local staging_file="${raw_file}.staging"

  {
    printf '# run_label %s\n' "$run_label"
    printf '# production_date_utc %s\n' "$run_date"
    printf '# Nt %s\n' "$nt"
    printf '# n_therm %s\n' "$THERMO_THERM"
    printf '# n_sweeps %s\n' "$THERMO_SWEEPS"
    printf '# meas_stride %s\n' "$THERMO_STRIDE"
    printf '# block_size_saved %s\n' "$THERMO_BLOCK_SAVED"
    cat "$raw_file"
  } > "$staging_file"
  mv "$staging_file" "$raw_file"
}

# Increase Nt at beta=5 so eta decreases toward the continuum limit.
case_index=0
for nt in "${NT_VALUES[@]}"; do
  eta="$(eta_from_nt "$nt")"
  seed="$((5100001 + case_index * 104729 + nt))"
  stream="$((72001 + case_index * 130363))"
  raw_file="$(printf '%s/qho_thermo_beta5_ntherm%s_nt%03d.dat' "$RAW_DIR" "$THERMO_THERM" "$nt")"
  run_label="$(printf 'thermo_beta5_nt%03d_hbover' "$nt")"

  printf 'Running beta=5 thermo production Nt=%s eta=%s seed=%s stream=%s therm=%s sweeps=%s stride=%s block_saved=%s\n' \
    "$nt" "$eta" "$seed" "$stream" "$THERMO_THERM" "$THERMO_SWEEPS" "$THERMO_STRIDE" "$THERMO_BLOCK_SAVED"

  start_time="$(date +%s.%N)"
  ./bin/qho_pimc \
    --nt "$nt" \
    --beta "$THERMO_BETA" \
    --eta "$eta" \
    --therm "$THERMO_THERM" \
    --sweeps "$THERMO_SWEEPS" \
    --stride "$THERMO_STRIDE" \
    --delta 1.0 \
    --seed "$seed" \
    --stream "$stream" \
    --init zero \
    --update hb-over \
    --n-over "$THERMO_N_OVER" \
    --out "$raw_file"
  end_time="$(date +%s.%N)"

  runtime_seconds="$(awk -v start="$start_time" -v end="$end_time" 'BEGIN { printf "%.9f", end - start }')"
  seconds_per_sweep="$(format_seconds_division "$runtime_seconds" "$THERMO_SWEEPS")"
  seconds_per_site_sweep="$(format_seconds_division "$runtime_seconds" "$((THERMO_SWEEPS * nt))")"
  run_date="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  add_production_metadata "$raw_file" "$run_label" "$run_date" "$nt"

  printf '%s %.17g %d hb-over zero %s %s %s %s %s %s %s %s %.9f %.12g %.12g\n' \
    "$THERMO_BETA" "$eta" "$nt" "$seed" "$stream" "$THERMO_THERM" "$THERMO_SWEEPS" \
    "$THERMO_STRIDE" "$THERMO_N_OVER" "$THERMO_BLOCK_SAVED" "$raw_file" \
    "$runtime_seconds" "$seconds_per_sweep" "$seconds_per_site_sweep" >> "$MANIFEST_FILE"

  case_index="$((case_index + 1))"
done

# Block each chain and fit <y^2> and H_ren versus eta^2.
python3 scripts/analysis/analyze_thermo_beta5.py --manifest "$MANIFEST_FILE"
