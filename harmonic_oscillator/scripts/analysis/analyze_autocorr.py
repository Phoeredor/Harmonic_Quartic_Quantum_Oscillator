#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/analysis/analyze_autocorr.py
# Purpose: Estimate autocorrelation times for selected Monte Carlo time series.
# Normalized autocorrelations and self-consistent integrated times quantify the
# effective number of independent measurements for each path observable.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: imports and configuration
# -----------------------------------------------------------------------------

"""Autocorrelation checks for QHO_PIMC ASCII output."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

COLUMNS = (
    "sweep",
    "y_mean",
    "y2_mean",
    "dy2_mean",
    "potential",
    "kinetic_ren",
    "energy_ren",
    "acc_rate",
)
OBSERVABLES = COLUMNS[1:]
META_KEYS = {"beta", "eta", "nt", "therm", "sweeps", "stride", "seed", "stream", "delta", "init"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate integrated autocorrelation time from QHO_PIMC output."
    )
    parser.add_argument("input", type=Path, help="Input .dat file produced by bin/qho_pimc")
    parser.add_argument(
        "--observable",
        choices=OBSERVABLES,
        required=True,
        help="Observable column to analyze",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=None,
        help="Maximum lag in saved-measurement units",
    )
    parser.add_argument(
        "--window-c",
        type=float,
        default=6.0,
        help="Self-consistent window parameter",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output file: lag autocorrelation tau_int_running",
    )
    return parser.parse_args()


def parse_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}

    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if not line.startswith("#"):
                continue
            fields = line[1:].strip().split(maxsplit=1)
            if len(fields) == 2 and fields[0] in META_KEYS:
                metadata[fields[0]] = fields[1]

    return metadata


def load_data(path: Path) -> np.ndarray:
    try:
        data = np.loadtxt(path, comments="#")
    except ValueError as exc:
        raise SystemExit(f"error: failed to load numeric rows from {path}: {exc}") from exc

    if data.size == 0:
        raise SystemExit(f"error: no numeric measurement rows found in {path}")

    data = np.atleast_2d(data)
    if data.shape[1] != len(COLUMNS):
        raise SystemExit(
            f"error: expected {len(COLUMNS)} columns, found {data.shape[1]} in {path}"
        )

    return data


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def normalized_autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = x.size

    variance = float(np.dot(x, x) / n)
    if variance <= 0.0 or not math.isfinite(variance):
        raise SystemExit("error: selected observable has zero or invalid variance")

    n_fft = next_power_of_two(2 * n)
    spectrum = np.fft.rfft(x, n=n_fft)
    raw = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft)[: max_lag + 1]
    normalization = np.arange(n, n - max_lag - 1, -1, dtype=float)
    autocov = raw / normalization
    rho = autocov / autocov[0]
    return rho


def running_tau_int(rho: np.ndarray) -> np.ndarray:
    tau = np.empty_like(rho, dtype=float)
    tau[0] = 0.5
    if rho.size > 1:
        tau[1:] = 0.5 + np.cumsum(rho[1:])
    return tau


def choose_window(tau: np.ndarray, window_c: float) -> tuple[int, bool]:
    for w in range(1, tau.size):
        if tau[w] > 0.0 and w >= window_c * tau[w]:
            return w, True
    return tau.size - 1, False


def parse_stride(metadata: dict[str, str]) -> int:
    if "stride" not in metadata:
        print("warning: stride metadata not found; using stride = 1", file=sys.stderr)
        return 1

    stride = int(metadata["stride"])
    if stride <= 0:
        raise SystemExit("error: invalid non-positive stride metadata")
    return stride


def format_float(value: float) -> str:
    return f"{value:.12g}"


def write_autocorr(path: Path, rho: np.ndarray, tau: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# lag autocorrelation tau_int_running\n")
        for lag, (rho_lag, tau_lag) in enumerate(zip(rho, tau)):
            handle.write(f"{lag} {rho_lag:.17g} {tau_lag:.17g}\n")


def main() -> int:
    args = parse_args()
    path = args.input

    if not path.is_file():
        print(f"error: input file does not exist: {path}", file=sys.stderr)
        return 1
    if args.window_c <= 0.0:
        print("error: --window-c must be positive", file=sys.stderr)
        return 1

    metadata = parse_metadata(path)
    stride = parse_stride(metadata)
    data = load_data(path)
    n_measurements = data.shape[0]

    if n_measurements < 2:
        print("error: at least two measurements are required", file=sys.stderr)
        return 1

    if args.max_lag is None:
        max_lag = min(n_measurements // 4, 5000)
    else:
        if args.max_lag < 1:
            print("error: --max-lag must be positive", file=sys.stderr)
            return 1
        max_lag = args.max_lag

    max_lag = min(max_lag, n_measurements - 1)
    column = COLUMNS.index(args.observable)
    rho = normalized_autocorrelation(data[:, column], max_lag)
    tau = running_tau_int(rho)
    window, found_window = choose_window(tau, args.window_c)

    if not found_window:
        print(
            "warning: self-consistent window not found; using the largest available lag",
            file=sys.stderr,
        )

    tau_saved = float(tau[window])
    tau_sweeps = tau_saved * float(stride)
    recommended_stride = max(1, math.ceil(2.0 * tau_sweeps))
    recommended_block = max(1, math.ceil(10.0 * tau_saved))
    recommended_therm = max(1, math.ceil(20.0 * tau_sweeps))

    if args.out is not None:
        write_autocorr(args.out, rho, tau)

    print("Autocorrelation analysis")
    print(f"Input file: {path}")
    print(f"Observable: {args.observable}")
    print(f"Measurements: {n_measurements}")
    print(f"Measurement stride: {stride}")
    print(f"Max lag: {max_lag}")
    print(f"Window c: {format_float(args.window_c)}")
    print()
    print(f"tau_int(saved units) = {format_float(tau_saved)}")
    print(f"tau_int(sweeps)      = {format_float(tau_sweeps)}")
    print(f"window W             = {window}")
    print()
    print("Recommendations:")
    print(f"  meas_stride >= {recommended_stride}")
    print(f"  block_size  >= {recommended_block}")
    print(f"  n_therm     >= {recommended_therm}")

    if args.out is not None:
        print(f"Autocorrelation data: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
