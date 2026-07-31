#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/run/run_qho_position_distribution_production.sh
# Purpose: Run the production workflow for position probability densities.
# Block-resolved histograms span low- and high-temperature ensembles and retain
# sufficient information for reblocking, moments, and exact Gaussian comparisons.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: workflow configuration
# -----------------------------------------------------------------------------

set -euo pipefail

block_dir="data/raw/production/position_distribution"
log_dir="data/raw/production/position_distribution/logs"
manifest="${block_dir}/qho_position_distribution_production_manifest.dat"

# Record the physical parameters and block-histogram location for every ensemble.
mkdir -p "$block_dir" "$log_dir"
if [[ ! -e "$manifest" ]]; then
  printf '# beta eta Nt seed stream seed_source therm sweeps stride block_size bins y_min y_max bin_width block_file runtime_seconds\n' > "$manifest"
fi

# Reuse an ensemble only when beta, eta, Nt, and its block count all match.
compatible() {
  local file="$1" beta="$2" eta="$3" nt="$4" blocks="$5"
  [[ -f "$file" ]] || return 1
  python3 - "$file" "$beta" "$eta" "$nt" "$blocks" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts/analysis")
from analyze_qho_position_distribution import load_block_histogram
h = load_block_histogram(Path(sys.argv[1]))
ok = (abs(float(h.metadata["beta"]) - float(sys.argv[2])) < 1e-13
      and abs(float(h.metadata["eta"]) - float(sys.argv[3])) < 1e-13
      and int(h.metadata["nt"]) == int(sys.argv[4])
      and h.n_blocks == int(sys.argv[5]))
raise SystemExit(0 if ok else 1)
PY
}

# Choose a histogram range from the exact finite-lattice Gaussian width and sample it.
run_point() {
  local beta="$1" eta="$2" nt="$3" therm="$4" sweeps="$5" block_size="$6"
  local bin_width="$7" seed="$8" stream="$9" seed_source="${10}"
  local beta_tag="${beta//./p}" eta_tag="${eta//./p}"
  local stem="qho_position_distribution_beta${beta_tag}_eta${eta_tag}_nt${nt}_production"
  local block_file="${block_dir}/${stem}_blocks.dat"
  local stdout_log="${log_dir}/${stem}.stdout.log"
  local stderr_log="${log_dir}/${stem}.stderr.log"
  local command_log="${log_dir}/${stem}.command.txt"
  local runtime_file="${log_dir}/${stem}.runtime.txt"
  local geometry y_abs bins y_min y_max start end runtime

  geometry="$(python3 - "$beta" "$eta" "$nt" "$bin_width" <<'PY'
import math, sys
import numpy as np
beta, eta, nt, width = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
p = np.arange(nt, dtype=float)
variance = float(np.mean(eta / (eta*eta + 4.0*np.sin(np.pi*p/nt)**2)))
y_abs = math.ceil((6.0*math.sqrt(variance))/0.5) * 0.5
bins_real = 2.0*y_abs/width
bins = round(bins_real)
if abs(bins_real-bins) > 1e-10:
    raise SystemExit("range is incompatible with bin width")
print(f"{y_abs:.17g} {bins}")
PY
)"
  read -r y_abs bins <<< "$geometry"
  y_min="-${y_abs}"
  y_max="$y_abs"

  if compatible "$block_file" "$beta" "$eta" "$nt" 100; then
    printf 'Reusing compatible %s\n' "$block_file"
    return 0
  fi
  for path in "$block_file" "$stdout_log" "$stderr_log" "$command_log" "$runtime_file"; do
    if [[ -e "$path" ]]; then
      printf 'error: refusing to overwrite %s\n' "$path" >&2
      exit 1
    fi
  done

  cmd=(./bin/qho_pimc --nt "$nt" --beta "$beta" --eta "$eta"
       --therm "$therm" --sweeps "$sweeps" --stride 10
       --seed "$seed" --stream "$stream" --init zero
       --update hb-over --n-over 5 --format none
       --hist-min "$y_min" --hist-max "$y_max" --hist-bin-width "$bin_width"
       --hist-block-out "$block_file" --hist-block-size-saved "$block_size")
  printf '%q ' "${cmd[@]}" > "$command_log"
  printf '\n' >> "$command_log"
  printf 'Running beta=%s eta=%s Nt=%s bins=%s range=[%s,%s]\n' "$beta" "$eta" "$nt" "$bins" "$y_min" "$y_max"
  start="$(date +%s)"
  "${cmd[@]}" >"$stdout_log" 2>"$stderr_log"
  end="$(date +%s)"
  runtime="$((end-start))"
  printf '%s\n' "$runtime" > "$runtime_file"
  printf '# runtime_seconds %s\n# seed_source %s\n' "$runtime" "$seed_source" >> "$block_file"
  printf '%s %s %s %s %s %s %s %s 10 %s %s %s %s %s %s %s\n' \
    "$beta" "$eta" "$nt" "$seed" "$stream" "$seed_source" "$therm" "$sweeps" \
    "$block_size" "$bins" "$y_min" "$y_max" "$bin_width" "$block_file" "$runtime" >> "$manifest"
}

# Cover the ground-state regime through the thermally broadened high-temperature regime.
run_point 40 0.05 800 100000 100000 100 0.05 \
  11116231314413106520 16771360361412893095 spectrum_beta40_highstat_nt800_manifest
run_point 5 0.05 100 100000 200000 200 0.05 \
  6356849 1636357 thermo_beta5_nt100_manifest
run_point 1 0.05 20 100000 500000 500 0.05 \
  6100021 83001 temperature_bare_beta1_nt20_header
run_point 0.5 0.05 10 100000 500000 500 0.10 \
  6500001 500001 new_fixed_position_production
run_point 0.25 0.05 5 100000 1000000 1000 0.10 \
  6250001 250001 new_fixed_position_production
