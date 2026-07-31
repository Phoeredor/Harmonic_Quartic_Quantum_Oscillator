#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/run/run_sampling_efficiency.sh
# Purpose: Run update-algorithm comparison chains.
# Metropolis, heatbath, and heatbath-plus-overrelaxation sample the same
# exp(-S_E) measure; autocorrelation and cost distinguish their efficiency.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: workflow configuration
# -----------------------------------------------------------------------------

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"

cd "${project_root}"

COMPARISON_SWEEPS="${COMPARISON_SWEEPS:-30000}"
COMPARISON_STRIDE="${COMPARISON_STRIDE:-1}"
COMPARISON_BETA="${COMPARISON_BETA:-5}"
COMPARISON_N_OVER="${COMPARISON_N_OVER:-5}"

RAW_DIR="data/raw/checks/algorithm_comparison"
PROCESSED_DIR="data/processed/checks"
MANIFEST_FILE="${PROCESSED_DIR}/qho_algorithm_comparison_manifest.dat"

ETA_VALUES=(0.2 0.1 0.05 0.025 0.0125)
UPDATES=(metro heatbath hb-over)
CONVERGENCE_NT_VALUES=(100 200 400)
CONVERGENCE_INITS=(zero uniform)

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

require_positive_int "COMPARISON_SWEEPS" "$COMPARISON_SWEEPS"
require_positive_int "COMPARISON_STRIDE" "$COMPARISON_STRIDE"
require_nonnegative_int "COMPARISON_N_OVER" "$COMPARISON_N_OVER"

# Build the common sampler before comparing Markov kernels at identical physics.
make clean && make

mkdir -p "$RAW_DIR" "$PROCESSED_DIR"
rm -f \
  "${RAW_DIR}"/qho_algorithm_comparison_*.dat \
  "${PROCESSED_DIR}"/qho_algorithm_comparison_*.dat \

printf '# QHO algorithm comparison manifest\n' > "$MANIFEST_FILE"
printf '# beta eta Nt update init seed stream sweeps stride n_over raw_file runtime_seconds seconds_per_sweep seconds_per_site_sweep\n' >> "$MANIFEST_FILE"

nt_from_eta() {
  python3 - "$COMPARISON_BETA" "$1" <<'PY'
import math
import sys

beta = float(sys.argv[1])
eta = float(sys.argv[2])
nt_float = beta / eta
nt = int(round(nt_float))
if nt <= 1 or not math.isclose(nt_float, nt, rel_tol=0.0, abs_tol=1.0e-10):
    raise SystemExit(f"beta/eta is not an integer: beta={beta:.17g} eta={eta:.17g}")
print(nt)
PY
}

eta_from_nt() {
  python3 - "$COMPARISON_BETA" "$1" <<'PY'
import sys

beta = float(sys.argv[1])
nt = int(sys.argv[2])
print(f"{beta / nt:.17g}")
PY
}

format_seconds_division() {
  awk -v numerator="$1" -v denominator="$2" 'BEGIN { printf "%.12g", numerator / denominator }'
}

add_comparison_metadata() {
  local raw_file="$1"
  local run_label="$2"
  local comparison_date="$3"
  local staging_file="${raw_file}.staging"

  {
    printf '# run_label %s\n' "$run_label"
    printf '# comparison_date_utc %s\n' "$comparison_date"
    cat "$raw_file"
  } > "$staging_file"
  mv "$staging_file" "$raw_file"
}

run_chain() {
  local chain_group="$1"
  local update="$2"
  local init="$3"
  local nt="$4"
  local eta="$5"
  local seed="$6"
  local stream="$7"
  local update_tag="${update//-/_}"
  local run_label="algorithm_comparison_${chain_group}_${update_tag}_nt${nt}_init_${init}"
  local raw_file="${RAW_DIR}/qho_algorithm_comparison_${chain_group}_${update_tag}_nt${nt}_init_${init}.dat"
  local start_time
  local end_time
  local runtime_seconds
  local seconds_per_sweep
  local seconds_per_site_sweep
  local comparison_date

  printf 'Running %s: beta=%s eta=%s Nt=%s update=%s init=%s seed=%s stream=%s sweeps=%s stride=%s n_over=%s\n' \
    "$chain_group" "$COMPARISON_BETA" "$eta" "$nt" "$update" "$init" "$seed" "$stream" "$COMPARISON_SWEEPS" "$COMPARISON_STRIDE" "$COMPARISON_N_OVER"

  start_time="$(date +%s.%N)"
  ./bin/qho_pimc \
    --nt "$nt" \
    --beta "$COMPARISON_BETA" \
    --eta "$eta" \
    --therm 0 \
    --sweeps "$COMPARISON_SWEEPS" \
    --stride "$COMPARISON_STRIDE" \
    --delta 1.0 \
    --seed "$seed" \
    --stream "$stream" \
    --init "$init" \
    --update "$update" \
    --n-over "$COMPARISON_N_OVER" \
    --out "$raw_file"
  end_time="$(date +%s.%N)"

  runtime_seconds="$(awk -v start="$start_time" -v end="$end_time" 'BEGIN { printf "%.9f", end - start }')"
  seconds_per_sweep="$(format_seconds_division "$runtime_seconds" "$COMPARISON_SWEEPS")"
  seconds_per_site_sweep="$(format_seconds_division "$runtime_seconds" "$((COMPARISON_SWEEPS * nt))")"
  comparison_date="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  add_comparison_metadata "$raw_file" "$run_label" "$comparison_date"

  printf '%s %.17g %d %s %s %s %s %s %s %s %s %.9f %.12g %.12g\n' \
    "$COMPARISON_BETA" "$eta" "$nt" "$update" "$init" "$seed" "$stream" \
    "$COMPARISON_SWEEPS" "$COMPARISON_STRIDE" "$COMPARISON_N_OVER" "$raw_file" \
    "$runtime_seconds" "$seconds_per_sweep" "$seconds_per_site_sweep" >> "$MANIFEST_FILE"
}

# Scan eta at fixed beta to compare algorithmic autocorrelations across resolutions.
case_index=0
for eta in "${ETA_VALUES[@]}"; do
  nt="$(nt_from_eta "$eta")"
  for update in "${UPDATES[@]}"; do
    seed="$((810001 + case_index * 7919 + nt))"
    stream="$((12001 + case_index * 104729))"
    run_chain "grid" "$update" "zero" "$nt" "$eta" "$seed" "$stream"
    case_index="$((case_index + 1))"
  done
done

# Contrast initial paths to assess convergence toward the same equilibrium measure.
for nt in "${CONVERGENCE_NT_VALUES[@]}"; do
  eta="$(eta_from_nt "$nt")"
  for init in "${CONVERGENCE_INITS[@]}"; do
    seed="$((910001 + case_index * 7919 + nt))"
    stream="$((22001 + case_index * 104729))"
    run_chain "convergence" "hb-over" "$init" "$nt" "$eta" "$seed" "$stream"
    case_index="$((case_index + 1))"
  done
done

# Summarize autocorrelation times, blocking behavior, and cost per lattice sweep.
python3 scripts/analysis/analyze_sampling_efficiency.py \
  --manifest "$MANIFEST_FILE"
