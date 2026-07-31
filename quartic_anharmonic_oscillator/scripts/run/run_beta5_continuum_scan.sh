#!/usr/bin/env bash
# Generate beta=5 quartic-oscillator ensembles over lambda and Nt.  Holding beta
# fixed while increasing Nt decreases eta=beta/Nt, providing moments and position
# histograms for blocked eta^2 continuum extrapolations.
set -euo pipefail

# Common physical and sampling parameters for the continuum ensemble grid.
BETA=5
C_DELTA="${C_DELTA:-2.4}"
N_THERM=400000
N_SWEEPS=1000000
MEAS_STRIDE=10
UPDATE=metro-over
N_OVER=5
INIT=gaussian
HIST_YMIN=-3.5
HIST_YMAX=3.5
HIST_BINS=160

# Lambda scans the quartic interaction; Nt scans the Euclidean lattice spacing.
LAMBDAS=(0.0125 0.025 0.05 0.075 0.10 0.15 0.20 0.25 0.35 0.50 0.75 1.00)
NTS=(5 10 16 20 25 28 32 36 40 50 64 80 100 128 160 200 256 320 400 512)

RAW_DIR="data/raw/production/beta5_continuum_v2"
HIST_DIR="${RAW_DIR}/histograms"

# Convert physical parameters to stable filename components.
tag_float() {
    local value="$1"
    value="${value%0}"
    value="${value%0}"
    value="${value%.}"
    echo "${value/./p}"
}

if [[ "${RUN_PRODUCTION_V2:-0}" == "1" ]]; then
    mkdir -p "$RAW_DIR" "$HIST_DIR"
fi

# Construct one reproducible Markov chain for every (lambda,Nt) pair.
planned=0
lambda_i=0
for lambda in "${LAMBDAS[@]}"; do
    for nt in "${NTS[@]}"; do
        # Scale the proposal with sqrt(eta) as the lattice is refined.
        eta="$(awk -v beta="$BETA" -v nt="$nt" 'BEGIN { printf "%.17g", beta / nt }')"
        delta="$(awk -v c="$C_DELTA" -v beta="$BETA" -v nt="$nt" 'BEGIN { printf "%.17g", c * sqrt(beta / nt) }')"
        lambda_tag="$(tag_float "$lambda")"
        c_tag="$(tag_float "$C_DELTA")"
        seed=$((7300000 + 100000 * lambda_i + nt))
        base="anharmonic_beta5_v2_lambda${lambda_tag}_nt${nt}_c${c_tag}"
        raw_file="${RAW_DIR}/${base}_measurements.dat"
        cmd=(./bin/anharmonic_pimc
            --beta "$BETA"
            --nt "$nt"
            --lambda "$lambda"
            --n-therm "$N_THERM"
            --n-sweeps "$N_SWEEPS"
            --meas-stride "$MEAS_STRIDE"
            --seed "$seed"
            --delta "$delta"
            --update "$UPDATE"
            --n-over "$N_OVER"
            --init "$INIT"
            --hist-ymin "$HIST_YMIN"
            --hist-ymax "$HIST_YMAX"
            --hist-bins "$HIST_BINS"
            --out "$raw_file")

        # Save the finest-lattice position density for the lambda comparison.
        if [[ "$nt" == "512" ]]; then
            cmd+=(--hist-out "${HIST_DIR}/${base}_histogram.dat")
        fi

        planned=$((planned + 1))
        # Sampling occurs only when explicitly enabled; otherwise the full grid is listed.
        if [[ "${RUN_PRODUCTION_V2:-0}" != "1" ]]; then
            printf "PLAN lambda=%s Nt=%s eta=%s delta=%s seed=%s out=%s\n" "$lambda" "$nt" "$eta" "$delta" "$seed" "$raw_file"
            continue
        fi
        if [[ -f "$raw_file" ]]; then
            echo "SKIP existing ${raw_file}"
            continue
        fi
        "${cmd[@]}"
    done
    lambda_i=$((lambda_i + 1))
done

if [[ "${RUN_PRODUCTION_V2:-0}" == "1" ]]; then
    echo "completed or skipped ${planned} planned production runs"
else
    echo "planned ${planned} production runs; set RUN_PRODUCTION_V2=1 to execute them"
fi
