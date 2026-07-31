#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/run/run_temperature_bare_scan.sh
# Purpose: Run the fixed-spacing temperature scan.
# Beta is varied through Nt at constant eta, isolating thermal dependence from
# changes in the ultraviolet lattice cutoff of the bare energy estimator.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: workflow configuration
# -----------------------------------------------------------------------------

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"

cd "${project_root}"

TEMP_ETA="${TEMP_ETA:-0.05}"
TEMP_BETA_LIST="${TEMP_BETA_LIST:-1 1.2 1.4 1.6 2 2.5 3.2 4 5 6.4 8 10 12.8 16 20}"
TEMP_THERM="${TEMP_THERM:-200000}"
TEMP_SWEEPS="${TEMP_SWEEPS:-1000000}"
TEMP_STRIDE="${TEMP_STRIDE:-10}"
TEMP_BLOCK_SAVED="${TEMP_BLOCK_SAVED:-2000}"
TEMP_N_OVER="${TEMP_N_OVER:-5}"

RAW_DIR="data/raw/production/temperature_bare"
PROCESSED_DIR="data/processed/production"
MANIFEST_FILE="${PROCESSED_DIR}/qho_temperature_bare_manifest.dat"
TEMP_ETA_TAG=""
ETA_MANIFEST_FILE=""

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
    printf 'error: %s must be a positive integer, got %s\n' "$name" "$value" >&2
    exit 1
  fi
}

require_positive_float() {
  local name="$1"
  local value="$2"
  python3 - "$name" "$value" <<'PY'
import math
import sys

name = sys.argv[1]
try:
    value = float(sys.argv[2])
except ValueError:
    print(f"error: {name} must be a positive float, got {sys.argv[2]}", file=sys.stderr)
    raise SystemExit(1)
if not math.isfinite(value) or value <= 0.0:
    print(f"error: {name} must be a positive finite float, got {sys.argv[2]}", file=sys.stderr)
    raise SystemExit(1)
PY
}

require_positive_float "TEMP_ETA" "$TEMP_ETA"
require_positive_int "TEMP_THERM" "$TEMP_THERM"
require_positive_int "TEMP_SWEEPS" "$TEMP_SWEEPS"
require_positive_int "TEMP_STRIDE" "$TEMP_STRIDE"
require_positive_int "TEMP_BLOCK_SAVED" "$TEMP_BLOCK_SAVED"
require_positive_int "TEMP_N_OVER" "$TEMP_N_OVER"

# Build the sampler used for every temperature at the common lattice spacing.
make

mkdir -p "$RAW_DIR" "$PROCESSED_DIR" "plots/thermodynamics"

eta_tag() {
  python3 - "$1" <<'PY'
import sys

eta = float(sys.argv[1])
print("eta" + f"{eta:.12g}".replace("-", "m").replace(".", "p"))
PY
}

TEMP_ETA_TAG="$(eta_tag "$TEMP_ETA")"
ETA_MANIFEST_FILE="${PROCESSED_DIR}/qho_temperature_bare_manifest_${TEMP_ETA_TAG}.dat"

write_manifest_header() {
  local path="$1"
  printf '# QHO temperature scan with non-renormalized internal-energy estimator\n' > "$path"
  printf '# beta x eta Nt update init seed stream n_therm n_sweeps meas_stride n_over block_size_saved raw_file runtime_seconds seconds_per_sweep seconds_per_site_sweep\n' >> "$path"
}

write_manifest_header "$MANIFEST_FILE"
write_manifest_header "$ETA_MANIFEST_FILE"

derive_lattice() {
  python3 - "$1" "$TEMP_ETA" <<'PY'
import math
import sys

beta = float(sys.argv[1])
eta = float(sys.argv[2])
nt = int(round(beta / eta))
if nt <= 0:
    print("error: derived Nt is not positive", file=sys.stderr)
    raise SystemExit(1)
actual_beta = nt * eta
tol = 1.0e-10 * max(1.0, abs(beta))
if abs(actual_beta - beta) > tol:
    print(
        f"error: beta={beta:.17g} is not compatible with eta={eta:.17g}; "
        f"round(beta/eta)={nt} gives Nt*eta={actual_beta:.17g}",
        file=sys.stderr,
    )
    raise SystemExit(1)
x = 1.0 / actual_beta
print(f"{nt} {actual_beta:.17g} {x:.17g}")
PY
}

format_seconds_division() {
  awk -v numerator="$1" -v denominator="$2" 'BEGIN { printf "%.12g", numerator / denominator }'
}

add_temperature_metadata() {
  local raw_file="$1"
  local run_label="$2"
  local run_date="$3"
  local beta="$4"
  local x="$5"
  local nt="$6"
  local staging_file="${raw_file}.staging"

  {
    printf '# run_label %s\n' "$run_label"
    printf '# production_date_utc %s\n' "$run_date"
    printf '# temperature_scan 1\n'
    printf '# estimator U_b = 0.5*y2_mean - dy2_mean/(2*eta^2)\n'
    printf '# beta_target %s\n' "$beta"
    printf '# x %.17g\n' "$x"
    printf '# Nt %s\n' "$nt"
    printf '# n_therm %s\n' "$TEMP_THERM"
    printf '# n_sweeps %s\n' "$TEMP_SWEEPS"
    printf '# meas_stride %s\n' "$TEMP_STRIDE"
    printf '# block_size_saved %s\n' "$TEMP_BLOCK_SAVED"
    cat "$raw_file"
  } > "$staging_file"
  mv "$staging_file" "$raw_file"
}

# Vary beta through Nt while keeping eta, and therefore the cutoff, unchanged.
case_index=0
for beta in $TEMP_BETA_LIST; do
  read -r nt actual_beta x < <(derive_lattice "$beta")
  seed="$((6100001 + case_index * 104729 + nt))"
  stream="$((83001 + case_index * 130363))"
  raw_file="$(printf '%s/qho_temperature_bare_ntherm%s_beta%s_%s_nt%04d.dat' "$RAW_DIR" "$TEMP_THERM" "${beta//./p}" "$TEMP_ETA_TAG" "$nt")"
  run_label="$(printf 'temperature_bare_ntherm%s_beta%s_%s_nt%04d_hbover' "$TEMP_THERM" "${beta//./p}" "$TEMP_ETA_TAG" "$nt")"

  if [[ -e "$raw_file" ]]; then
    printf 'error: raw file already exists, refusing to overwrite: %s\n' "$raw_file" >&2
    exit 1
  fi

  printf 'Running temperature bare scan beta=%s Nt=%s eta=%s x=%s seed=%s stream=%s therm=%s sweeps=%s stride=%s block_saved=%s\n' \
    "$actual_beta" "$nt" "$TEMP_ETA" "$x" "$seed" "$stream" "$TEMP_THERM" "$TEMP_SWEEPS" "$TEMP_STRIDE" "$TEMP_BLOCK_SAVED"

  start_time="$(date +%s.%N)"
  ./bin/qho_pimc \
    --nt "$nt" \
    --beta "$actual_beta" \
    --eta "$TEMP_ETA" \
    --therm "$TEMP_THERM" \
    --sweeps "$TEMP_SWEEPS" \
    --stride "$TEMP_STRIDE" \
    --delta 1.0 \
    --seed "$seed" \
    --stream "$stream" \
    --init zero \
    --update hb-over \
    --n-over "$TEMP_N_OVER" \
    --out "$raw_file"
  end_time="$(date +%s.%N)"

  runtime_seconds="$(awk -v start="$start_time" -v end="$end_time" 'BEGIN { printf "%.9f", end - start }')"
  seconds_per_sweep="$(format_seconds_division "$runtime_seconds" "$TEMP_SWEEPS")"
  seconds_per_site_sweep="$(format_seconds_division "$runtime_seconds" "$((TEMP_SWEEPS * nt))")"
  run_date="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  add_temperature_metadata "$raw_file" "$run_label" "$run_date" "$actual_beta" "$x" "$nt"

  printf '%.17g %.17g %.17g %d hb-over zero %s %s %s %s %s %s %s %s %.9f %.12g %.12g\n' \
    "$actual_beta" "$x" "$TEMP_ETA" "$nt" "$seed" "$stream" "$TEMP_THERM" "$TEMP_SWEEPS" \
    "$TEMP_STRIDE" "$TEMP_N_OVER" "$TEMP_BLOCK_SAVED" "$raw_file" \
    "$runtime_seconds" "$seconds_per_sweep" "$seconds_per_site_sweep" >> "$MANIFEST_FILE"
  printf '%.17g %.17g %.17g %d hb-over zero %s %s %s %s %s %s %s %s %.9f %.12g %.12g\n' \
    "$actual_beta" "$x" "$TEMP_ETA" "$nt" "$seed" "$stream" "$TEMP_THERM" "$TEMP_SWEEPS" \
    "$TEMP_STRIDE" "$TEMP_N_OVER" "$TEMP_BLOCK_SAVED" "$raw_file" \
    "$runtime_seconds" "$seconds_per_sweep" "$seconds_per_site_sweep" >> "$ETA_MANIFEST_FILE"

  case_index="$((case_index + 1))"
done

# Block the bare estimator and subtract the largest-beta reference contribution.
python3 scripts/analysis/analyze_temperature_bare_scan.py --manifest "$ETA_MANIFEST_FILE"
