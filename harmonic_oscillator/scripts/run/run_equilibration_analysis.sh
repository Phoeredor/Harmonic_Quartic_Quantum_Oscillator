#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/run/run_equilibration_analysis.sh
# Purpose: Run heatbath-plus-overrelaxation consistency checks.
# Ensembles from zero and random initial paths test equilibration and blocking
# stability over several beta and Nt values.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: workflow configuration
# -----------------------------------------------------------------------------

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"

cd "${project_root}"

DIAG_SWEEPS="${DIAG_SWEEPS:-50000}"
DIAG_STRIDE="${DIAG_STRIDE:-1}"
DIAG_N_OVER="${DIAG_N_OVER:-5}"
DIAG_BURN_FRACTION="${DIAG_BURN_FRACTION:-0.2}"

RAW_DIR="data/raw/checks/hbover"
PROCESSED_DIR="data/processed/checks"
MANIFEST_FILE="${PROCESSED_DIR}/qho_hbover_diag_manifest.dat"

BETA5_NT_VALUES=(100 200 400 512)
BETA8_NT_VALUES=(128 256 512)
INITS=(zero uniform)

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
    printf 'error: %s must be a positive integer, got %s\n' "$name" "$value" >&2
    exit 1
  fi
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    printf 'error: %s must be a non-negative integer, got %s\n' "$name" "$value" >&2
    exit 1
  fi
}

require_positive_int "DIAG_SWEEPS" "$DIAG_SWEEPS"
require_positive_int "DIAG_STRIDE" "$DIAG_STRIDE"
require_nonnegative_int "DIAG_N_OVER" "$DIAG_N_OVER"

# Build one sampler for all beta, Nt, and initial-path ensembles.
make clean && make

mkdir -p "$RAW_DIR" "$PROCESSED_DIR"
rm -f \
  "${RAW_DIR}"/qho_hbover_diag_*.dat \
  "${PROCESSED_DIR}"/qho_hbover_diag_*.dat \

printf '# QHO hb-over check manifest\n' > "$MANIFEST_FILE"
printf '# beta eta Nt update init seed stream sweeps stride n_over raw_file runtime_seconds seconds_per_sweep seconds_per_site_sweep\n' >> "$MANIFEST_FILE"

eta_from_beta_nt() {
  python3 - "$1" "$2" <<'PY'
import sys

beta = float(sys.argv[1])
nt = int(sys.argv[2])
print(f"{beta / nt:.17g}")
PY
}

format_seconds_division() {
  awk -v numerator="$1" -v denominator="$2" 'BEGIN { printf "%.12g", numerator / denominator }'
}

add_diag_metadata() {
  local raw_file="$1"
  local run_label="$2"
  local diag_date="$3"
  local nt="$4"
  local staging_file="${raw_file}.staging"

  {
    printf '# run_label %s\n' "$run_label"
    printf '# check_date_utc %s\n' "$diag_date"
    printf '# Nt %s\n' "$nt"
    cat "$raw_file"
  } > "$staging_file"
  mv "$staging_file" "$raw_file"
}

run_chain() {
  local beta="$1"
  local nt="$2"
  local init="$3"
  local case_index="$4"
  local eta
  local beta_tag
  local seed
  local stream
  local raw_file
  local run_label
  local start_time
  local end_time
  local runtime_seconds
  local seconds_per_sweep
  local seconds_per_site_sweep
  local diag_date

  eta="$(eta_from_beta_nt "$beta" "$nt")"
  beta_tag="${beta//./p}"
  seed="$((1200001 + case_index * 104729 + nt + ${beta_tag//p/} * 1000))"
  stream="$((33001 + case_index * 130363))"
  raw_file="${RAW_DIR}/qho_hbover_diag_beta${beta_tag}_nt${nt}_init_${init}.dat"
  run_label="hbover_diag_beta${beta_tag}_nt${nt}_init_${init}"

  printf 'Running hb-over check: beta=%s eta=%s Nt=%s init=%s seed=%s stream=%s sweeps=%s stride=%s n_over=%s\n' \
    "$beta" "$eta" "$nt" "$init" "$seed" "$stream" "$DIAG_SWEEPS" "$DIAG_STRIDE" "$DIAG_N_OVER"

  start_time="$(date +%s.%N)"
  ./bin/qho_pimc \
    --nt "$nt" \
    --beta "$beta" \
    --eta "$eta" \
    --therm 0 \
    --sweeps "$DIAG_SWEEPS" \
    --stride "$DIAG_STRIDE" \
    --delta 1.0 \
    --seed "$seed" \
    --stream "$stream" \
    --init "$init" \
    --update hb-over \
    --n-over "$DIAG_N_OVER" \
    --out "$raw_file"
  end_time="$(date +%s.%N)"

  runtime_seconds="$(awk -v start="$start_time" -v end="$end_time" 'BEGIN { printf "%.9f", end - start }')"
  seconds_per_sweep="$(format_seconds_division "$runtime_seconds" "$DIAG_SWEEPS")"
  seconds_per_site_sweep="$(format_seconds_division "$runtime_seconds" "$((DIAG_SWEEPS * nt))")"
  diag_date="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  add_diag_metadata "$raw_file" "$run_label" "$diag_date" "$nt"

  printf '%s %.17g %d hb-over %s %s %s %s %s %s %s %.9f %.12g %.12g\n' \
    "$beta" "$eta" "$nt" "$init" "$seed" "$stream" "$DIAG_SWEEPS" "$DIAG_STRIDE" \
    "$DIAG_N_OVER" "$raw_file" "$runtime_seconds" "$seconds_per_sweep" "$seconds_per_site_sweep" \
    >> "$MANIFEST_FILE"
}

# Compare convergence and correlations at beta=5 over several lattice spacings.
case_index=0
for nt in "${BETA5_NT_VALUES[@]}"; do
  for init in "${INITS[@]}"; do
    run_chain "5" "$nt" "$init" "$case_index"
    case_index="$((case_index + 1))"
  done
done

# Repeat at lower temperature to probe longer Euclidean-time correlations.
for nt in "${BETA8_NT_VALUES[@]}"; do
  for init in "${INITS[@]}"; do
    run_chain "8" "$nt" "$init" "$case_index"
    case_index="$((case_index + 1))"
  done
done

# Convert the chains into thermalization, autocorrelation, and block summaries.
python3 scripts/analysis/analyze_equilibration.py \
  --manifest "$MANIFEST_FILE" \
  --burn-fraction "$DIAG_BURN_FRACTION"
