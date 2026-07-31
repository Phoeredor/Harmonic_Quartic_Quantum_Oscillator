#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/analysis/analyze_equilibration.py
# Purpose: Analyze heatbath-plus-overrelaxation consistency checks.
# It examines equilibration, integrated autocorrelations, and blocking stability
# of primary Euclidean-path observables across initial states and resolutions.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: imports and configuration
# -----------------------------------------------------------------------------

"""Analyze hb-over thermalization, autocorrelation, and blocking checks."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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
OBSERVABLES = ("y_mean", "y2_mean", "dy2_mean", "energy_ren")
CURVE_KEYS = {(5.0, 512), (8.0, 512)}
MANIFEST_COLUMNS = (
    "beta",
    "eta",
    "Nt",
    "update",
    "init",
    "seed",
    "stream",
    "sweeps",
    "stride",
    "n_over",
    "raw_file",
    "runtime_seconds",
    "seconds_per_sweep",
    "seconds_per_site_sweep",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze hb-over check chains for thermalization and blocking."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/checks/qho_hbover_diag_manifest.dat"),
        help="Manifest written by scripts/run/run_equilibration_analysis.sh",
    )
    parser.add_argument(
        "--burn-fraction",
        type=float,
        default=0.2,
        help="Initial fraction of each chain to discard before autocorrelation analysis",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=None,
        help="Maximum autocorrelation lag; default min(N_meas//4, 10000)",
    )
    parser.add_argument(
        "--window-c",
        type=float,
        default=6.0,
        help="Self-consistent autocorrelation window parameter",
    )
    return parser.parse_args()


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.12g}"


def parse_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != len(MANIFEST_COLUMNS):
                raise ValueError(
                    f"{path}:{line_number}: expected {len(MANIFEST_COLUMNS)} columns, "
                    f"found {len(fields)}"
                )
            row = dict(zip(MANIFEST_COLUMNS, fields))
            row["beta"] = float(row["beta"])
            row["eta"] = float(row["eta"])
            row["Nt"] = int(row["Nt"])
            row["seed"] = int(row["seed"])
            row["stream"] = int(row["stream"])
            row["sweeps"] = int(row["sweeps"])
            row["stride"] = int(row["stride"])
            row["n_over"] = int(row["n_over"])
            row["runtime_seconds"] = float(row["runtime_seconds"])
            row["seconds_per_sweep"] = float(row["seconds_per_sweep"])
            row["seconds_per_site_sweep"] = float(row["seconds_per_site_sweep"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no manifest rows found")
    return rows


def project_root_from_manifest(path: Path) -> Path:
    resolved = path.resolve()
    if (
        len(resolved.parents) >= 4
        and resolved.parents[0].name == "checks"
        and resolved.parents[1].name == "processed"
        and resolved.parents[2].name == "data"
    ):
        return resolved.parents[3]
    return Path.cwd()


def resolve_raw_path(raw_file: str, manifest: Path) -> Path:
    raw_path = Path(raw_file)
    if raw_path.is_absolute():
        return raw_path
    if raw_path.is_file():
        return raw_path
    return project_root_from_manifest(manifest) / raw_path


def load_measurements(path: Path) -> np.ndarray:
    try:
        data = np.loadtxt(path, comments="#")
    except ValueError as exc:
        raise ValueError(f"{path}: failed to load numeric rows: {exc}") from exc
    if data.size == 0:
        raise ValueError(f"{path}: no measurement rows found")
    data = np.atleast_2d(data)
    if data.shape[1] < len(COLUMNS):
        raise ValueError(f"{path}: expected at least {len(COLUMNS)} columns, found {data.shape[1]}")
    return data[:, : len(COLUMNS)]


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def normalized_autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray | None:
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = int(x.size)
    variance = float(np.dot(x, x) / float(n))
    if variance <= 0.0 or not math.isfinite(variance):
        return None

    n_fft = next_power_of_two(2 * n)
    spectrum = np.fft.rfft(x, n=n_fft)
    raw = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft)[: max_lag + 1]
    normalization = np.arange(n, n - max_lag - 1, -1, dtype=float)
    autocov = raw / normalization
    if autocov[0] <= 0.0 or not math.isfinite(float(autocov[0])):
        return None
    return autocov / autocov[0]


def running_tau_int(rho: np.ndarray) -> np.ndarray:
    tau = np.empty_like(rho, dtype=float)
    tau[0] = 0.5
    if rho.size > 1:
        tau[1:] = 0.5 + np.cumsum(rho[1:])
    return tau


def choose_window(tau: np.ndarray, window_c: float) -> tuple[int, int]:
    for window in range(1, int(tau.size)):
        tau_value = float(tau[window])
        if math.isfinite(tau_value) and tau_value > 0.0 and window >= window_c * tau_value:
            return window, 1
    return int(tau.size) - 1, 0


def analyze_series(
    series: np.ndarray,
    max_lag_override: int | None,
    window_c: float,
) -> dict[str, Any]:
    n = int(series.size)
    mean = float(series.mean()) if n else math.nan
    if n < 2:
        return {
            "mean": mean,
            "naive_error": math.nan,
            "tau_int": math.nan,
            "tau_error": math.nan,
            "W_opt": 0,
            "saturated": 0,
            "sigma_true": math.nan,
            "rho": None,
            "tau_curve": None,
        }

    sample_var = float(series.var(ddof=1))
    naive_error = math.sqrt(sample_var / float(n)) if sample_var >= 0.0 else math.nan
    variance = float(series.var(ddof=0))
    if variance <= 0.0 or not math.isfinite(variance):
        return {
            "mean": mean,
            "naive_error": naive_error,
            "tau_int": math.nan,
            "tau_error": math.nan,
            "W_opt": 0,
            "saturated": 0,
            "sigma_true": math.nan,
            "rho": None,
            "tau_curve": None,
        }

    if max_lag_override is None:
        max_lag = min(n // 4, 10000)
    else:
        max_lag = max_lag_override
    max_lag = max(0, min(max_lag, n - 1))

    rho = normalized_autocorrelation(series, max_lag)
    if rho is None:
        return {
            "mean": mean,
            "naive_error": naive_error,
            "tau_int": math.nan,
            "tau_error": math.nan,
            "W_opt": 0,
            "saturated": 0,
            "sigma_true": math.nan,
            "rho": None,
            "tau_curve": None,
        }

    tau_curve = running_tau_int(rho)
    window, saturated = choose_window(tau_curve, window_c)
    tau_int = float(tau_curve[window])
    tau_error = (
        tau_int * math.sqrt(2.0 * float(2 * window + 1) / float(n))
        if math.isfinite(tau_int)
        else math.nan
    )
    sigma_true = (
        math.sqrt(sample_var / float(n) * 2.0 * tau_int)
        if math.isfinite(tau_int) and tau_int > 0.0
        else math.nan
    )

    return {
        "mean": mean,
        "naive_error": naive_error,
        "tau_int": tau_int,
        "tau_error": tau_error,
        "W_opt": window,
        "saturated": saturated,
        "sigma_true": sigma_true,
        "rho": rho,
        "tau_curve": tau_curve,
    }


def post_burn_data(data: np.ndarray, burn_fraction: float) -> np.ndarray:
    n = int(data.shape[0])
    burn = int(math.floor(float(n) * burn_fraction))
    if burn >= n:
        burn = n - 1
    return data[burn:, :]


def running_mean(series: np.ndarray) -> np.ndarray:
    return np.cumsum(series, dtype=float) / np.arange(1, series.size + 1, dtype=float)


def blocking_rows_for_series(
    manifest_row: dict[str, Any],
    observable: str,
    series: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(series.size)
    max_block = n // 16
    block_size = 1
    while block_size <= max_block and block_size >= 1:
        n_blocks = n // block_size
        if n_blocks < 1:
            break
        trimmed = series[: n_blocks * block_size]
        block_means = trimmed.reshape(n_blocks, block_size).mean(axis=1)
        blocked_mean = float(block_means.mean())
        if n_blocks >= 2:
            blocked_error = float(block_means.std(ddof=1) / math.sqrt(float(n_blocks)))
        else:
            blocked_error = math.nan
        rows.append(
            {
                **manifest_row,
                "observable": observable,
                "block_size": block_size,
                "n_blocks": n_blocks,
                "blocked_mean": blocked_mean,
                "blocked_error": blocked_error,
            }
        )
        block_size *= 2
    return rows


def thermalization_for_pair(
    zero_record: dict[str, Any],
    uniform_record: dict[str, Any],
) -> dict[str, Any]:
    beta = float(zero_record["manifest"]["beta"])
    eta = float(zero_record["manifest"]["eta"])
    nt = int(zero_record["manifest"]["Nt"])
    tau_values = [
        float(zero_record["slowest"]["tau_slow"]),
        float(uniform_record["slowest"]["tau_slow"]),
    ]
    finite_tau = [value for value in tau_values if math.isfinite(value)]
    tau_slow_pair = max(finite_tau) if finite_tau else math.nan
    block_size_recommended = math.ceil(10.0 * tau_slow_pair) if math.isfinite(tau_slow_pair) else math.nan

    n = min(int(zero_record["data"].shape[0]), int(uniform_record["data"].shape[0]))
    status = "unresolved"
    n_therm_0 = math.nan

    if n >= 1:
        zero_y2 = running_mean(zero_record["data"][:n, COLUMNS.index("y2_mean")])
        uniform_y2 = running_mean(uniform_record["data"][:n, COLUMNS.index("y2_mean")])
        zero_energy = running_mean(zero_record["data"][:n, COLUMNS.index("energy_ren")])
        uniform_energy = running_mean(uniform_record["data"][:n, COLUMNS.index("energy_ren")])

        sigma_y2 = math.sqrt(
            zero_record["summary_by_obs"]["y2_mean"]["sigma_true"] ** 2
            + uniform_record["summary_by_obs"]["y2_mean"]["sigma_true"] ** 2
        )
        sigma_energy = math.sqrt(
            zero_record["summary_by_obs"]["energy_ren"]["sigma_true"] ** 2
            + uniform_record["summary_by_obs"]["energy_ren"]["sigma_true"] ** 2
        )

        if (
            math.isfinite(sigma_y2)
            and sigma_y2 > 0.0
            and math.isfinite(sigma_energy)
            and sigma_energy > 0.0
        ):
            compatible_y2 = np.abs(zero_y2 - uniform_y2) <= 2.0 * sigma_y2
            compatible_energy = np.abs(zero_energy - uniform_energy) <= 2.0 * sigma_energy
            compatible = compatible_y2 & compatible_energy
            suffix_compatible = np.logical_and.accumulate(compatible[::-1])[::-1]
            indices = np.nonzero(suffix_compatible)[0]
            if indices.size > 0:
                idx = int(indices[0])
                n_therm_0 = float(zero_record["data"][idx, COLUMNS.index("sweep")])
                status = "resolved"

    if math.isfinite(tau_slow_pair):
        tau_term = math.ceil(20.0 * tau_slow_pair)
        if math.isfinite(n_therm_0):
            n_therm_recommended = max(math.ceil(2.0 * n_therm_0), tau_term)
        else:
            n_therm_recommended = tau_term
    else:
        n_therm_recommended = math.nan

    return {
        "beta": beta,
        "eta": eta,
        "Nt": nt,
        "n_therm_0": n_therm_0,
        "thermalization_status": status,
        "tau_slow_pair": tau_slow_pair,
        "n_therm_recommended": n_therm_recommended,
        "block_size_recommended": block_size_recommended,
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO hb-over check autocorrelation summary\n")
        handle.write(
            "# beta eta Nt init observable mean naive_error tau_int tau_error W_opt "
            "saturated sigma_true n_meas_post_burn runtime_seconds seconds_per_sweep "
            "seconds_per_site_sweep\n"
        )
        for row in rows:
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} {row['Nt']} "
                f"{row['init']} {row['observable']} {format_float(row['mean'])} "
                f"{format_float(row['naive_error'])} {format_float(row['tau_int'])} "
                f"{format_float(row['tau_error'])} {row['W_opt']} {row['saturated']} "
                f"{format_float(row['sigma_true'])} {row['n_meas_post_burn']} "
                f"{format_float(row['runtime_seconds'])} {format_float(row['seconds_per_sweep'])} "
                f"{format_float(row['seconds_per_site_sweep'])}\n"
            )


def write_slowest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO hb-over check slowest observable per chain\n")
        handle.write("# beta eta Nt init tau_slow tau_error_slow slowest_observable W_opt_slow saturated\n")
        for row in rows:
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} {row['Nt']} "
                f"{row['init']} {format_float(row['tau_slow'])} "
                f"{format_float(row['tau_error_slow'])} {row['slowest_observable']} "
                f"{row['W_opt_slow']} {row['saturated']}\n"
            )


def write_thermalization(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO hb-over check zero/uniform thermalization estimates\n")
        handle.write(
            "# beta eta Nt n_therm_0 thermalization_status tau_slow_pair "
            "n_therm_recommended block_size_recommended\n"
        )
        for row in rows:
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} {row['Nt']} "
                f"{format_float(row['n_therm_0'])} {row['thermalization_status']} "
                f"{format_float(row['tau_slow_pair'])} "
                f"{format_float(float(row['n_therm_recommended']))} "
                f"{format_float(float(row['block_size_recommended']))}\n"
            )


def write_autocorr(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO hb-over check autocorrelation curves\n")
        handle.write("# beta eta Nt init observable lag C_lag\n")
        for row in rows:
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} {row['Nt']} "
                f"{row['init']} {row['observable']} {row['lag']} {format_float(row['C_lag'])}\n"
            )


def write_tau_window(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO hb-over check running tau_int windows\n")
        handle.write("# beta eta Nt init observable W tau_int_W\n")
        for row in rows:
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} {row['Nt']} "
                f"{row['init']} {row['observable']} {row['W']} {format_float(row['tau_int_W'])}\n"
            )


def write_blocking(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO hb-over check blocking errors\n")
        handle.write("# beta eta Nt init observable block_size n_blocks blocked_mean blocked_error\n")
        for row in rows:
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} {row['Nt']} "
                f"{row['init']} {row['observable']} {row['block_size']} {row['n_blocks']} "
                f"{format_float(row['blocked_mean'])} {format_float(row['blocked_error'])}\n"
            )


def write_convergence(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO hb-over check running means\n")
        handle.write("# beta eta Nt init sweep running_y2_mean running_energy_ren\n")
        for row in rows:
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} {row['Nt']} "
                f"{row['init']} {row['sweep']} {format_float(row['running_y2_mean'])} "
                f"{format_float(row['running_energy_ren'])}\n"
            )


def write_recommendation(
    path: Path,
    slowest_rows: list[dict[str, Any]],
    thermalization_rows: list[dict[str, Any]],
) -> None:
    finite_tau_rows = [row for row in slowest_rows if math.isfinite(row["tau_slow"])]
    tau_max = max((row["tau_slow"] for row in finite_tau_rows), default=math.nan)
    slow_counts = Counter(row["slowest_observable"] for row in finite_tau_rows)
    slowest_common = slow_counts.most_common(1)[0] if slow_counts else ("none", 0)

    def saturation_for(beta: float, nt: int) -> str:
        selected = [
            row
            for row in slowest_rows
            if abs(row["beta"] - beta) < 1.0e-12 and row["Nt"] == nt
        ]
        if not selected:
            return "missing"
        if all(row["saturated"] == 1 for row in selected):
            return "yes"
        if any(row["saturated"] == 1 for row in selected):
            return "partial"
        return "no"

    finite_therm = [
        float(row["n_therm_recommended"])
        for row in thermalization_rows
        if math.isfinite(float(row["n_therm_recommended"]))
    ]
    finite_block = [
        float(row["block_size_recommended"])
        for row in thermalization_rows
        if math.isfinite(float(row["block_size_recommended"]))
    ]
    global_therm = math.ceil(max(finite_therm)) if finite_therm else math.nan
    global_block = math.ceil(max(finite_block)) if finite_block else math.nan
    decorrelated_stride = math.ceil(2.0 * tau_max) if math.isfinite(tau_max) else math.nan

    lines = [
        "# QHO Hb-Over Check Recommendation",
        "",
        "This file summarizes hb-over-only checks for choosing later production Monte Carlo parameters. It is not a final production choice.",
        "",
        f"- Maximum tau_slow across all check chains: {format_float(tau_max)}.",
        f"- The slowest observable most often is `{slowest_common[0]}` ({slowest_common[1]} chains).",
        f"- beta=5 Nt=512 saturated: {saturation_for(5.0, 512)}.",
        f"- beta=8 Nt=512 saturated: {saturation_for(8.0, 512)}.",
        f"- Recommended global n_therm in sweeps: {format_float(float(global_therm))}.",
        f"- Recommended global block_size in sweeps: {format_float(float(global_block))}.",
        f"- Suggested production policy: save every sweep and use blocking with blocks of at least {format_float(float(global_block))} sweeps, or use a decorrelated stride of about ceil(2*tau_slow_max) = {format_float(float(decorrelated_stride))}.",
        "",
        "Final production parameters should be chosen only after inspecting these checks and, if needed, rerunning this hb-over check with longer chains.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def main() -> int:
    args = parse_args()
    if not (0.0 <= args.burn_fraction < 1.0):
        print("error: --burn-fraction must satisfy 0 <= burn < 1", file=sys.stderr)
        return 1
    if args.max_lag is not None and args.max_lag < 0:
        print("error: --max-lag must be non-negative", file=sys.stderr)
        return 1
    if args.window_c <= 0.0:
        print("error: --window-c must be positive", file=sys.stderr)
        return 1
    if not args.manifest.is_file():
        print(f"error: manifest does not exist: {args.manifest}", file=sys.stderr)
        return 1

    try:
        manifest_rows = parse_manifest(args.manifest)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_dir = args.manifest.parent
    summary_rows: list[dict[str, Any]] = []
    slowest_rows: list[dict[str, Any]] = []
    autocorr_rows: list[dict[str, Any]] = []
    tau_window_rows: list[dict[str, Any]] = []
    blocking_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    records: dict[tuple[float, int, str], dict[str, Any]] = {}

    for manifest_row in manifest_rows:
        if manifest_row["update"] != "hb-over":
            print(f"warning: skipping non-hb-over row: {manifest_row['raw_file']}", file=sys.stderr)
            continue

        raw_path = resolve_raw_path(str(manifest_row["raw_file"]), args.manifest)
        try:
            data = load_measurements(raw_path)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        burned = post_burn_data(data, args.burn_fraction)
        summary_by_obs: dict[str, dict[str, Any]] = {}
        selected_for_curves = (
            (round(float(manifest_row["beta"]), 12), int(manifest_row["Nt"])) in CURVE_KEYS
        )

        for observable in OBSERVABLES:
            column = COLUMNS.index(observable)
            result = analyze_series(burned[:, column], args.max_lag, args.window_c)
            summary_by_obs[observable] = result
            summary_rows.append(
                {
                    **manifest_row,
                    "observable": observable,
                    "mean": result["mean"],
                    "naive_error": result["naive_error"],
                    "tau_int": result["tau_int"],
                    "tau_error": result["tau_error"],
                    "W_opt": result["W_opt"],
                    "saturated": result["saturated"],
                    "sigma_true": result["sigma_true"],
                    "n_meas_post_burn": int(burned.shape[0]),
                }
            )

            blocking_rows.extend(
                blocking_rows_for_series(manifest_row, observable, burned[:, column])
            )

            if selected_for_curves and result["rho"] is not None:
                for lag, c_lag in enumerate(result["rho"]):
                    autocorr_rows.append(
                        {
                            **manifest_row,
                            "observable": observable,
                            "lag": lag,
                            "C_lag": float(c_lag),
                        }
                    )
            if selected_for_curves and result["tau_curve"] is not None:
                for window, tau_value in enumerate(result["tau_curve"]):
                    tau_window_rows.append(
                        {
                            **manifest_row,
                            "observable": observable,
                            "W": window,
                            "tau_int_W": float(tau_value),
                        }
                    )

        finite = [
            (observable, result)
            for observable, result in summary_by_obs.items()
            if math.isfinite(float(result["tau_int"]))
        ]
        if finite:
            slowest_observable, slowest_result = max(
                finite, key=lambda item: float(item[1]["tau_int"])
            )
            slowest = {
                **manifest_row,
                "tau_slow": float(slowest_result["tau_int"]),
                "tau_error_slow": float(slowest_result["tau_error"]),
                "slowest_observable": slowest_observable,
                "W_opt_slow": int(slowest_result["W_opt"]),
                "saturated": int(slowest_result["saturated"]),
            }
        else:
            slowest = {
                **manifest_row,
                "tau_slow": math.nan,
                "tau_error_slow": math.nan,
                "slowest_observable": "none",
                "W_opt_slow": 0,
                "saturated": 0,
            }
        slowest_rows.append(slowest)

        y2_running = running_mean(data[:, COLUMNS.index("y2_mean")])
        energy_running = running_mean(data[:, COLUMNS.index("energy_ren")])
        for sweep, y2_value, energy_value in zip(
            data[:, COLUMNS.index("sweep")], y2_running, energy_running
        ):
            convergence_rows.append(
                {
                    **manifest_row,
                    "sweep": int(round(float(sweep))),
                    "running_y2_mean": float(y2_value),
                    "running_energy_ren": float(energy_value),
                }
            )

        records[(float(manifest_row["beta"]), int(manifest_row["Nt"]), manifest_row["init"])] = {
            "manifest": manifest_row,
            "data": data,
            "summary_by_obs": summary_by_obs,
            "slowest": slowest,
        }

    thermalization_rows: list[dict[str, Any]] = []
    pair_keys = sorted({(beta, nt) for beta, nt, _ in records})
    for beta, nt in pair_keys:
        zero_record = records.get((beta, nt, "zero"))
        uniform_record = records.get((beta, nt, "uniform"))
        if zero_record is None or uniform_record is None:
            print(f"warning: missing zero/uniform pair for beta={beta:g} Nt={nt}", file=sys.stderr)
            continue
        thermalization_rows.append(thermalization_for_pair(zero_record, uniform_record))

    write_summary(output_dir / "qho_hbover_diag_summary.dat", summary_rows)
    write_slowest(output_dir / "qho_hbover_diag_slowest.dat", slowest_rows)
    write_thermalization(output_dir / "qho_hbover_diag_thermalization.dat", thermalization_rows)
    write_autocorr(output_dir / "qho_hbover_diag_autocorr.dat", autocorr_rows)
    write_tau_window(output_dir / "qho_hbover_diag_tau_window.dat", tau_window_rows)
    write_blocking(output_dir / "qho_hbover_diag_blocking.dat", blocking_rows)
    write_convergence(output_dir / "qho_hbover_diag_convergence.dat", convergence_rows)
    write_recommendation(
        output_dir / "qho_hbover_diag_recommendation.md",
        slowest_rows,
        thermalization_rows,
    )

    print(f"Wrote {output_dir / 'qho_hbover_diag_summary.dat'}")
    print(f"Wrote {output_dir / 'qho_hbover_diag_slowest.dat'}")
    print(f"Wrote {output_dir / 'qho_hbover_diag_thermalization.dat'}")
    print(f"Wrote {output_dir / 'qho_hbover_diag_autocorr.dat'}")
    print(f"Wrote {output_dir / 'qho_hbover_diag_tau_window.dat'}")
    print(f"Wrote {output_dir / 'qho_hbover_diag_blocking.dat'}")
    print(f"Wrote {output_dir / 'qho_hbover_diag_convergence.dat'}")
    print(f"Wrote {output_dir / 'qho_hbover_diag_recommendation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
