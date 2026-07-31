#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT_DIR/bin/anharmonic_pimc"
SCAN_DIR="$ROOT_DIR/data/raw/spectrum/scan_quick"
ETA_DIR="$ROOT_DIR/data/raw/spectrum/eta_check_lambda0p25"

LAMBDAS=(0.025 0.05 0.10 0.25 0.50 1.00)
SCAN_SEEDS=(2026071301 2026071302 2026071303)
ETA_SEEDS=(2026072301 2026072302)
ETA_NTS=(200 400 800)

beta=40
n_therm=100000
n_sweeps=300000
meas_stride=20
corr_block_size=100
n_over=5
update=metro-over

delta_for() {
    awk -v beta="$1" -v nt="$2" 'BEGIN { printf "%.17g", 2.4 * sqrt(beta / nt) }'
}

lambda_tag() {
    awk -v x="$1" 'BEGIN {
        s = sprintf("%.3f", x)
        sub(/0+$/, "", s)
        sub(/\.$/, "", s)
        gsub(/\./, "p", s)
        print s
    }'
}

show_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

run_one() {
    local raw_dir="$1" lam="$2" nt="$3" seed="$4" max_dt="$5"
    local tag delta out corr start end elapsed blocks
    tag="beta40_lambda$(lambda_tag "$lam")_nt${nt}_seed${seed}"
    delta="$(delta_for "$beta" "$nt")"
    out="$raw_dir/anharmonic_spectrum_${tag}_measurements.dat"
    corr="$raw_dir/anharmonic_spectrum_${tag}_corr.dat"
    CMD=(
        "$BIN" --beta "$beta" --nt "$nt" --lambda "$lam"
        --n-therm "$n_therm" --n-sweeps "$n_sweeps" --meas-stride "$meas_stride"
        --seed "$seed" --delta "$delta" --update "$update" --n-over "$n_over"
        --init gaussian --hist-ymin -5 --hist-ymax 5 --hist-bins 200
        --out "$out" --corr-out "$corr" --corr-block-size "$corr_block_size"
        --corr-max-dt "$max_dt"
    )
    if [[ "${DRY_ONLY:-0}" == "1" ]]; then
        show_command "${CMD[@]}"
        return
    fi
    if [[ -e "$out" || -e "$corr" ]]; then
        echo "SKIP existing $corr"
        return
    fi
    echo "[RUN] lambda=$lam Nt=$nt seed=$seed corr_max_dt=$max_dt"
    start="$(date +%s)"
    "${CMD[@]}"
    end="$(date +%s)"
    elapsed=$((end - start))
    blocks="$(awk '!/^#/ && $7 == 0 {n++} END {print n+0}' "$corr")"
    printf 'lambda %s Nt %s seed %s seconds %d blocks %d\n' \
        "$lam" "$nt" "$seed" "$elapsed" "$blocks" >> "$raw_dir/timing.log"
    echo "[PASS] lambda=$lam Nt=$nt seed=$seed: ${elapsed}s, $blocks blocks"
}

if [[ "${RUN_SPECTRUM_SCAN_QUICK:-0}" != "1" ]]; then
    echo "DRY RUN: set RUN_SPECTRUM_SCAN_QUICK=1 to execute."
    echo
    echo "Main scan commands:"
    DRY_ONLY=1
    for lam in "${LAMBDAS[@]}"; do
        for seed in "${SCAN_SEEDS[@]}"; do
            run_one "$SCAN_DIR" "$lam" 400 "$seed" 80
        done
    done
    echo
    echo "Eta-check commands:"
    for nt in "${ETA_NTS[@]}"; do
        eta="$(awk -v beta="$beta" -v nt="$nt" 'BEGIN { printf "%.17g", beta / nt }')"
        max_dt="$(awk -v eta="$eta" 'BEGIN { printf "%d", int(4.0 / eta + 0.999999) }')"
        for seed in "${ETA_SEEDS[@]}"; do
            run_one "$ETA_DIR" 0.25 "$nt" "$seed" "$max_dt"
        done
    done
    echo
    echo "Estimated cost from previous pilot: comfortably below 1 hour on this machine."
    exit 0
fi

if [[ ! -x "$BIN" ]]; then
    echo "[ERROR] missing executable $BIN; run make first" >&2
    exit 1
fi

mkdir -p "$SCAN_DIR" "$ETA_DIR"

for lam in "${LAMBDAS[@]}"; do
    for seed in "${SCAN_SEEDS[@]}"; do
        run_one "$SCAN_DIR" "$lam" 400 "$seed" 80
    done
done

for nt in "${ETA_NTS[@]}"; do
    eta="$(awk -v beta="$beta" -v nt="$nt" 'BEGIN { printf "%.17g", beta / nt }')"
    max_dt="$(awk -v eta="$eta" 'BEGIN { printf "%d", int(4.0 / eta + 0.999999) }')"
    for seed in "${ETA_SEEDS[@]}"; do
        run_one "$ETA_DIR" 0.25 "$nt" "$seed" "$max_dt"
    done
done
