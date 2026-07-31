#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/analysis/analyze_spectrum_blocks.py
# Purpose: Build jackknife samples for spectrum observables from block data.
# Connected correlators and effective gaps are reconstructed within each
# leave-one-block sample so nonlinear subtractions retain their covariance.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: imports and configuration
# -----------------------------------------------------------------------------

"""Jackknife analysis for QHO block-level spectrum correlators.

The input file is produced by ``bin/qho_pimc --spectrum-block-out``.  Each
row contains block means of the one-point operators and raw two-point
correlators.  Connected correlators are reconstructed inside each
leave-one-block sample, not by jackknifing already-connected block values.

For leave-one-block estimates theta_i, the jackknife variance used here is

    sigma^2 = (N_blocks - 1) / N_blocks * sum_i (theta_i - mean(theta_i))^2.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


OPERATORS = {
    "y": {"exact": 1.0, "label": "C_y"},
    "y2": {"exact": 2.0, "label": "C_y2_conn"},
    "y3": {"exact": 1.0, "label": "C_y3_conn"},
    "a3": {"exact": 3.0, "label": "C_A_conn"},
}


@dataclass(frozen=True)
class BlockData:
    metadata: dict[str, str]
    has_y3: bool
    block_ids: np.ndarray
    weights: np.ndarray
    lag: np.ndarray
    tau: np.ndarray
    mean_y: np.ndarray
    mean_y2: np.ndarray
    mean_y3: np.ndarray
    mean_a: np.ndarray
    raw_y: np.ndarray
    raw_y2: np.ndarray
    raw_y3: np.ndarray
    raw_a: np.ndarray


@dataclass(frozen=True)
class Estimate:
    mean_y: float
    mean_y2: float
    mean_y3: float
    mean_a: float
    raw_y: np.ndarray
    raw_y2: np.ndarray
    raw_y3: np.ndarray
    raw_a: np.ndarray
    conn_y: np.ndarray
    conn_y2: np.ndarray
    conn_y3: np.ndarray
    conn_a: np.ndarray


def parse_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if not line.startswith("#"):
                continue
            fields = line[1:].strip().split(maxsplit=1)
            if len(fields) == 2:
                metadata[fields[0]] = fields[1]
    return metadata


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.12g}"


def load_blocks(path: Path) -> BlockData:
    metadata = parse_metadata(path)
    data = np.loadtxt(path, comments="#")
    if data.size == 0:
        raise ValueError(f"{path}: no numeric rows found")
    data = np.atleast_2d(data)
    if data.shape[1] not in (10, 12):
        raise ValueError(f"{path}: expected 10 schema-0 columns or 12 y3-format columns, found {data.shape[1]}")
    has_y3 = data.shape[1] == 12

    block_col = data[:, 0].astype(int)
    meas_col = data[:, 1].astype(float)
    lag_col = data[:, 2].astype(int)
    tau_col = data[:, 3]
    block_ids = np.unique(block_col)
    lags = np.unique(lag_col)
    if block_ids.size < 2:
        raise ValueError("at least two completed blocks are required for jackknife errors")
    if not np.array_equal(lags, np.arange(int(lags[0]), int(lags[-1]) + 1)):
        raise ValueError("lag values must be contiguous")
    if int(lags[0]) != 0:
        raise ValueError("lag values must start at 0")

    n_blocks = block_ids.size
    n_lag = lags.size
    weights = np.empty(n_blocks, dtype=float)
    tau = np.empty(n_lag, dtype=float)
    mean_y = np.empty(n_blocks, dtype=float)
    mean_y2 = np.empty(n_blocks, dtype=float)
    mean_y3 = np.full(n_blocks, np.nan, dtype=float)
    mean_a = np.empty(n_blocks, dtype=float)
    raw_y = np.empty((n_blocks, n_lag), dtype=float)
    raw_y2 = np.empty((n_blocks, n_lag), dtype=float)
    raw_y3 = np.full((n_blocks, n_lag), np.nan, dtype=float)
    raw_a = np.empty((n_blocks, n_lag), dtype=float)

    mean_a_col = 7 if has_y3 else 6
    raw_y_col = 8 if has_y3 else 7
    raw_y2_col = 9 if has_y3 else 8
    raw_y3_col = 10
    raw_a_col = 11 if has_y3 else 9

    for b_index, block_id in enumerate(block_ids):
        rows = data[block_col == block_id]
        order = np.argsort(rows[:, 2])
        rows = rows[order]
        if rows.shape[0] != n_lag or not np.array_equal(rows[:, 2].astype(int), lags):
            raise ValueError(f"block {block_id} does not contain the complete lag set")
        if not np.all(rows[:, 1] == rows[0, 1]):
            raise ValueError(f"block {block_id} has inconsistent block_measurements")
        one_point_consistent = (
            np.allclose(rows[:, 4], rows[0, 4])
            and np.allclose(rows[:, 5], rows[0, 5])
            and np.allclose(rows[:, mean_a_col], rows[0, mean_a_col])
        )
        if has_y3:
            one_point_consistent = one_point_consistent and np.allclose(rows[:, 6], rows[0, 6])
        if not one_point_consistent:
            raise ValueError(f"block {block_id} has inconsistent one-point means across lags")
        weights[b_index] = rows[0, 1]
        mean_y[b_index] = rows[0, 4]
        mean_y2[b_index] = rows[0, 5]
        if has_y3:
            mean_y3[b_index] = rows[0, 6]
        mean_a[b_index] = rows[0, mean_a_col]
        raw_y[b_index, :] = rows[:, raw_y_col]
        raw_y2[b_index, :] = rows[:, raw_y2_col]
        if has_y3:
            raw_y3[b_index, :] = rows[:, raw_y3_col]
        raw_a[b_index, :] = rows[:, raw_a_col]
        if b_index == 0:
            tau[:] = rows[:, 3]
        elif not np.allclose(tau, rows[:, 3]):
            raise ValueError(f"block {block_id} has inconsistent tau values")

    if np.any(weights <= 0.0):
        raise ValueError("all block weights must be positive")

    return BlockData(
        metadata=metadata,
        has_y3=has_y3,
        block_ids=block_ids,
        weights=weights,
        lag=lags.astype(int),
        tau=tau,
        mean_y=mean_y,
        mean_y2=mean_y2,
        mean_y3=mean_y3,
        mean_a=mean_a,
        raw_y=raw_y,
        raw_y2=raw_y2,
        raw_y3=raw_y3,
        raw_a=raw_a,
    )


def weighted_scalar(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(weights, values) / np.sum(weights))


def weighted_rows(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(weights @ values / np.sum(weights), dtype=float)


def estimate_from_blocks(data: BlockData, omit: int | None = None) -> Estimate:
    mask = np.ones(data.weights.size, dtype=bool)
    if omit is not None:
        mask[omit] = False
    weights = data.weights[mask]
    if weights.size == 0 or np.sum(weights) <= 0.0:
        raise ValueError("empty jackknife sample")

    mean_y = weighted_scalar(data.mean_y[mask], weights)
    mean_y2 = weighted_scalar(data.mean_y2[mask], weights)
    mean_y3 = weighted_scalar(data.mean_y3[mask], weights) if data.has_y3 else math.nan
    mean_a = weighted_scalar(data.mean_a[mask], weights)
    raw_y = weighted_rows(data.raw_y[mask, :], weights)
    raw_y2 = weighted_rows(data.raw_y2[mask, :], weights)
    raw_y3 = weighted_rows(data.raw_y3[mask, :], weights) if data.has_y3 else np.full_like(raw_y, np.nan)
    raw_a = weighted_rows(data.raw_a[mask, :], weights)
    conn_y3 = raw_y3 - mean_y3 * mean_y3 if data.has_y3 else np.full_like(raw_y, np.nan)

    return Estimate(
        mean_y=mean_y,
        mean_y2=mean_y2,
        mean_y3=mean_y3,
        mean_a=mean_a,
        raw_y=raw_y,
        raw_y2=raw_y2,
        raw_y3=raw_y3,
        raw_a=raw_a,
        conn_y=raw_y - mean_y * mean_y,
        conn_y2=raw_y2 - mean_y2 * mean_y2,
        conn_y3=conn_y3,
        conn_a=raw_a - mean_a * mean_a,
    )


def jackknife_error(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=float)
    n_blocks = arr.shape[0]
    if n_blocks < 2:
        return np.full(arr.shape[1:], np.nan, dtype=float)
    finite = np.all(np.isfinite(arr), axis=0)
    mean = np.zeros(arr.shape[1:], dtype=float)
    err = np.full(arr.shape[1:], np.nan, dtype=float)
    if np.any(finite):
        mean[finite] = np.mean(arr[:, finite], axis=0)
        var = (n_blocks - 1.0) / n_blocks * np.sum((arr[:, finite] - mean[finite]) ** 2, axis=0)
        err[finite] = np.sqrt(var)
    return np.where(finite, err, np.nan)


def effective_mass(conn: np.ndarray, eta: float) -> np.ndarray:
    out = np.full(conn.shape, np.nan, dtype=float)
    positive = (conn[:-1] > 0.0) & (conn[1:] > 0.0)
    out[:-1] = np.where(positive, np.log(conn[:-1] / conn[1:]) / eta, np.nan)
    return out


def collect_jackknife(data: BlockData) -> tuple[Estimate, list[Estimate]]:
    central = estimate_from_blocks(data)
    samples = [estimate_from_blocks(data, omit=i) for i in range(data.weights.size)]
    return central, samples


def write_correlators(prefix: Path, data: BlockData, central: Estimate, samples: list[Estimate]) -> Path:
    path = prefix.with_name(prefix.name + "_correlators_jackknife.dat")
    sample_raw_y = np.asarray([s.raw_y for s in samples])
    sample_raw_y2 = np.asarray([s.raw_y2 for s in samples])
    sample_raw_a = np.asarray([s.raw_a for s in samples])
    sample_conn_y = np.asarray([s.conn_y for s in samples])
    sample_conn_y2 = np.asarray([s.conn_y2 for s in samples])
    sample_conn_a = np.asarray([s.conn_a for s in samples])
    if data.has_y3:
        sample_raw_y3 = np.asarray([s.raw_y3 for s in samples])
        sample_conn_y3 = np.asarray([s.conn_y3 for s in samples])

    raw_y_err = jackknife_error(sample_raw_y)
    raw_y2_err = jackknife_error(sample_raw_y2)
    raw_a_err = jackknife_error(sample_raw_a)
    conn_y_err = jackknife_error(sample_conn_y)
    conn_y2_err = jackknife_error(sample_conn_y2)
    conn_a_err = jackknife_error(sample_conn_a)
    if data.has_y3:
        raw_y3_err = jackknife_error(sample_raw_y3)
        conn_y3_err = jackknife_error(sample_conn_y3)

    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO_PIMC spectrum block jackknife correlators\n")
        for key in ("beta", "eta", "nt", "therm", "sweeps", "stride", "seed", "stream", "init", "update", "n_over", "spectrum_max_lag", "spectrum_block_size_saved"):
            if key in data.metadata:
                handle.write(f"# {key} {data.metadata[key]}\n")
        handle.write(f"# n_blocks {data.weights.size}\n")
        handle.write(f"# total_saved_measurements {format_float(float(np.sum(data.weights)))}\n")
        handle.write("# jackknife_variance (N-1)/N * sum_i (theta_i - mean(theta_i))^2\n")
        if data.has_y3:
            handle.write("# y3_definition y^3\n")
            handle.write("# A_definition y^3 - 1.5*y\n")
            handle.write("# columns lag tau mean_y mean_y2 mean_y3 mean_a raw_y raw_y_err C_y C_y_err raw_y2 raw_y2_err C_y2_conn C_y2_conn_err raw_y3 raw_y3_err C_y3_conn C_y3_conn_err raw_A raw_A_err C_A_conn C_A_conn_err\n")
        else:
            handle.write("# columns lag tau mean_y mean_y2 mean_a raw_y raw_y_err C_y C_y_err raw_y2 raw_y2_err C_y2_conn C_y2_conn_err raw_A raw_A_err C_A_conn C_A_conn_err\n")
        for i, lag in enumerate(data.lag):
            if data.has_y3:
                handle.write(
                    f"{int(lag)} {data.tau[i]:.17g} "
                    f"{central.mean_y:.17g} {central.mean_y2:.17g} {central.mean_y3:.17g} {central.mean_a:.17g} "
                    f"{central.raw_y[i]:.17g} {raw_y_err[i]:.17g} {central.conn_y[i]:.17g} {conn_y_err[i]:.17g} "
                    f"{central.raw_y2[i]:.17g} {raw_y2_err[i]:.17g} {central.conn_y2[i]:.17g} {conn_y2_err[i]:.17g} "
                    f"{central.raw_y3[i]:.17g} {raw_y3_err[i]:.17g} {central.conn_y3[i]:.17g} {conn_y3_err[i]:.17g} "
                    f"{central.raw_a[i]:.17g} {raw_a_err[i]:.17g} {central.conn_a[i]:.17g} {conn_a_err[i]:.17g}\n"
                )
            else:
                handle.write(
                    f"{int(lag)} {data.tau[i]:.17g} "
                    f"{central.mean_y:.17g} {central.mean_y2:.17g} {central.mean_a:.17g} "
                    f"{central.raw_y[i]:.17g} {raw_y_err[i]:.17g} {central.conn_y[i]:.17g} {conn_y_err[i]:.17g} "
                    f"{central.raw_y2[i]:.17g} {raw_y2_err[i]:.17g} {central.conn_y2[i]:.17g} {conn_y2_err[i]:.17g} "
                    f"{central.raw_a[i]:.17g} {raw_a_err[i]:.17g} {central.conn_a[i]:.17g} {conn_a_err[i]:.17g}\n"
                )
    return path


def write_meff(prefix: Path, data: BlockData, central: Estimate, samples: list[Estimate], eta: float) -> tuple[Path, dict[str, np.ndarray]]:
    path = prefix.with_name(prefix.name + "_meff_jackknife.dat")
    central_meff = {
        "y": effective_mass(central.conn_y, eta),
        "y2": effective_mass(central.conn_y2, eta),
        "a3": effective_mass(central.conn_a, eta),
    }
    sample_meff = {
        "y": np.asarray([effective_mass(s.conn_y, eta) for s in samples]),
        "y2": np.asarray([effective_mass(s.conn_y2, eta) for s in samples]),
        "a3": np.asarray([effective_mass(s.conn_a, eta) for s in samples]),
    }
    if data.has_y3:
        central_meff["y3"] = effective_mass(central.conn_y3, eta)
        sample_meff["y3"] = np.asarray([effective_mass(s.conn_y3, eta) for s in samples])
    meff_err = {name: jackknife_error(values) for name, values in sample_meff.items()}

    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO_PIMC spectrum block jackknife effective masses\n")
        handle.write("# effective_mass log(C(lag) / C(lag+1)) / eta\n")
        handle.write("# effective masses are computed inside each leave-one-block sample\n")
        if data.has_y3:
            handle.write("# y3 effective mass is a contamination check and should be compared to Delta E_1 = 1\n")
            handle.write("# columns lag tau meff_y meff_y_err meff_y2 meff_y2_err meff_y3 meff_y3_err meff_A meff_A_err\n")
        else:
            handle.write("# columns lag tau meff_y meff_y_err meff_y2 meff_y2_err meff_A meff_A_err\n")
        for i, lag in enumerate(data.lag):
            if data.has_y3:
                handle.write(
                    f"{int(lag)} {data.tau[i]:.17g} "
                    f"{central_meff['y'][i]:.17g} {meff_err['y'][i]:.17g} "
                    f"{central_meff['y2'][i]:.17g} {meff_err['y2'][i]:.17g} "
                    f"{central_meff['y3'][i]:.17g} {meff_err['y3'][i]:.17g} "
                    f"{central_meff['a3'][i]:.17g} {meff_err['a3'][i]:.17g}\n"
                )
            else:
                handle.write(
                    f"{int(lag)} {data.tau[i]:.17g} "
                    f"{central_meff['y'][i]:.17g} {meff_err['y'][i]:.17g} "
                    f"{central_meff['y2'][i]:.17g} {meff_err['y2'][i]:.17g} "
                    f"{central_meff['a3'][i]:.17g} {meff_err['a3'][i]:.17g}\n"
                )
    return path, central_meff


def parse_window(values: list[int] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    lo, hi = values
    if lo > hi:
        raise argparse.ArgumentTypeError("plateau window minimum must be <= maximum")
    return lo, hi


def plateau_from_meff(meff: np.ndarray, lag: np.ndarray, window: tuple[int, int]) -> float:
    lo, hi = window
    mask = (lag >= lo) & (lag <= hi) & np.isfinite(meff)
    values = meff[mask]
    if values.size == 0:
        return math.nan
    return float(np.mean(values))


def write_plateaus(
    prefix: Path,
    data: BlockData,
    central: Estimate,
    samples: list[Estimate],
    eta: float,
    windows: dict[str, tuple[int, int]],
    warnings: list[str],
) -> Path | None:
    if not windows:
        return None

    path = prefix.with_name(prefix.name + "_plateau_estimates_jackknife.dat")
    central_conn = {"y": central.conn_y, "y2": central.conn_y2, "a3": central.conn_a}
    sample_conn = {
        "y": [s.conn_y for s in samples],
        "y2": [s.conn_y2 for s in samples],
        "a3": [s.conn_a for s in samples],
    }
    if data.has_y3:
        central_conn["y3"] = central.conn_y3
        sample_conn["y3"] = [s.conn_y3 for s in samples]

    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO_PIMC check plateau estimates from block jackknife\n")
        handle.write("# windows are user-provided checks, not final hard-coded fit choices\n")
        handle.write("# y3 windows are contamination checks and compare to Delta E_1, not Delta E_3\n")
        handle.write("# columns operator exact_gap lag_min lag_max n_points estimate error diff\n")
        for name, window in windows.items():
            if name == "y3" and not data.has_y3:
                warnings.append("plateau window for y3 requested, but the input block file has no y3 columns")
                continue
            lo, hi = window
            conn = central_conn[name]
            selected = (data.lag >= lo) & (data.lag <= hi)
            meff_mask = selected & (np.arange(data.lag.size) < data.lag.size - 1)
            positive = np.zeros(data.lag.size, dtype=bool)
            positive[:-1] = (conn[:-1] > 0.0) & (conn[1:] > 0.0)
            n_points = int(np.count_nonzero(meff_mask & positive))
            if n_points < 2:
                warnings.append(f"plateau window for {name} has fewer than two positive effective-mass points")
            if np.any(meff_mask & ~positive):
                warnings.append(f"plateau window for {name} includes non-positive correlator pairs")

            central_value = plateau_from_meff(effective_mass(conn, eta), data.lag, window)
            jk_values = np.asarray([
                plateau_from_meff(effective_mass(sample, eta), data.lag, window)
                for sample in sample_conn[name]
            ], dtype=float)
            err = float(jackknife_error(jk_values.reshape(jk_values.size, 1))[0]) if jk_values.size else math.nan
            exact = float(OPERATORS[name]["exact"])
            diff = central_value - exact if math.isfinite(central_value) else math.nan
            handle.write(
                f"{name} {exact:.17g} {lo} {hi} {n_points} "
                f"{central_value:.17g} {err:.17g} {diff:.17g}\n"
            )
    return path


def write_summary(
    prefix: Path,
    input_path: Path,
    data: BlockData,
    correlator_path: Path,
    meff_path: Path,
    plateau_path: Path | None,
    warnings: list[str],
) -> Path:
    path = prefix.with_name(prefix.name + "_analysis_summary.md")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# QHO spectrum block jackknife analysis summary\n\n")
        handle.write(f"Input: `{input_path}`\n\n")
        handle.write("## Metadata\n\n")
        for key in ("beta", "eta", "nt", "therm", "sweeps", "stride", "seed", "stream", "init", "update", "n_over", "spectrum_max_lag", "spectrum_block_size_saved"):
            if key in data.metadata:
                handle.write(f"- `{key}`: `{data.metadata[key]}`\n")
        handle.write(f"- `n_blocks`: `{data.weights.size}`\n")
        handle.write(f"- `total_saved_measurements`: `{format_float(float(np.sum(data.weights)))}`\n\n")
        handle.write("## Statistical Method\n\n")
        handle.write("Blocks are combined with weights equal to the number of saved measurements in each block. ")
        handle.write("For every leave-one-block sample, the script reconstructs raw one-point means and raw two-point correlators, then forms connected correlators as `C = <OO> - <O>^2`. ")
        handle.write("Effective masses are secondary observables and are computed inside each jackknife sample.\n\n")
        handle.write("Channels analyzed: `y -> Delta E_1`, connected `y2 -> Delta E_2`, `A = y^3 - 1.5*y -> Delta E_3`. ")
        if data.has_y3:
            handle.write("The connected `y3` channel is also analyzed as a contamination check and is compared to `Delta E_1`, not `Delta E_3`.\n\n")
        else:
            handle.write("The input uses the 10-column block schema without `y3`, so no `y3` channel is written.\n\n")
        handle.write("Jackknife variance formula:\n\n")
        handle.write("```text\n")
        handle.write("sigma^2 = (N_blocks - 1) / N_blocks * sum_i (theta_i - mean(theta_i))^2\n")
        handle.write("```\n\n")
        handle.write("## Outputs\n\n")
        handle.write(f"- Correlators: `{correlator_path}`\n")
        handle.write(f"- Effective masses: `{meff_path}`\n")
        if plateau_path is not None:
            handle.write(f"- Check plateau estimates: `{plateau_path}`\n")
        else:
            handle.write("- Check plateau estimates: not written because no plateau windows were provided.\n")
        handle.write("\n## Warnings\n\n")
        if warnings:
            for warning in warnings:
                handle.write(f"- {warning}\n")
        else:
            handle.write("- None.\n")
        handle.write("\n## Limitations\n\n")
        handle.write("- Plateau estimates, when requested, are check averages over effective masses, not final correlated constant fits.\n")
        handle.write("- No plateau window is selected automatically as final.\n")
        handle.write("- Cosh/PBC fits are not implemented in this script; use them as a separate optional cross-check after block-level errors are available.\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze QHO block-level spectrum correlators with jackknife errors.")
    parser.add_argument("input", type=Path, help="Block-level spectrum .dat file from --spectrum-block-out")
    parser.add_argument("--out-prefix", required=True, type=Path, help="Output prefix for jackknife analysis files")
    parser.add_argument("--plateau-y", nargs=2, type=int, metavar=("LAG_MIN", "LAG_MAX"), help="Optional check plateau window for y")
    parser.add_argument("--plateau-y2", nargs=2, type=int, metavar=("LAG_MIN", "LAG_MAX"), help="Optional check plateau window for connected y^2")
    parser.add_argument("--plateau-y3", nargs=2, type=int, metavar=("LAG_MIN", "LAG_MAX"), help="Optional check plateau window for connected y^3 contamination channel")
    parser.add_argument("--plateau-a", nargs=2, type=int, metavar=("LAG_MIN", "LAG_MAX"), help="Optional check plateau window for A=y^3-1.5*y")
    args = parser.parse_args()

    data = load_blocks(args.input)
    if "eta" not in data.metadata:
        raise SystemExit("error: missing eta metadata")
    eta = float(data.metadata["eta"])
    if eta <= 0.0 or not math.isfinite(eta):
        raise SystemExit("error: eta metadata must be positive and finite")

    windows: dict[str, tuple[int, int]] = {}
    parsed_windows = {
        "y": parse_window(args.plateau_y),
        "y2": parse_window(args.plateau_y2),
        "y3": parse_window(args.plateau_y3),
        "a3": parse_window(args.plateau_a),
    }
    for name, window in parsed_windows.items():
        if window is not None:
            windows[name] = window

    warnings: list[str] = []
    if not data.has_y3:
        warnings.append("input block file uses the 10-column schema without y3 columns; y3 correlators and effective masses were not produced")
    central, samples = collect_jackknife(data)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    correlator_path = write_correlators(args.out_prefix, data, central, samples)
    meff_path, _ = write_meff(args.out_prefix, data, central, samples, eta)
    plateau_path = write_plateaus(args.out_prefix, data, central, samples, eta, windows, warnings)
    summary_path = write_summary(args.out_prefix, args.input, data, correlator_path, meff_path, plateau_path, warnings)

    print("Spectrum block jackknife analysis")
    print(f"Input: {args.input}")
    print(f"Blocks: {data.weights.size}")
    print(f"y3 columns: {'yes' if data.has_y3 else 'no'}")
    print(f"Total saved measurements: {format_float(float(np.sum(data.weights)))}")
    print(f"Correlators: {correlator_path}")
    print(f"Effective masses: {meff_path}")
    if plateau_path is not None:
        print(f"Check plateaus: {plateau_path}")
    print(f"Summary: {summary_path}")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
