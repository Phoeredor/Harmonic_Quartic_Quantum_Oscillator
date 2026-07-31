#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/analysis/analyze_sampling_efficiency.py
# Purpose: Analyze the update-algorithm comparison at fixed physical parameters.
# It compares autocorrelation times and sampling cost for Metropolis, heatbath,
# and heatbath-plus-overrelaxation chains at common beta and lattice spacing.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: imports and configuration
# -----------------------------------------------------------------------------

"""Analyze QHO update-algorithm comparison chains."""

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
UPDATES = ("metro", "heatbath", "hb-over")
CURVE_NT = {100, 200, 400}
CANDIDATE_NT = (256, 400, 512, 640)
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
        description="Analyze the QHO algorithm-comparison check chains."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/checks/qho_algorithm_comparison_manifest.dat"),
        help="Manifest written by scripts/run/run_sampling_efficiency.sh",
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
        help="Maximum autocorrelation lag; default min(N_meas//4, 5000)",
    )
    parser.add_argument(
        "--window-c",
        type=float,
        default=6.0,
        help="Self-consistent window parameter",
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
    mean = float(series.mean())
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
        max_lag = min(n // 4, 5000)
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


def chain_kind(row: dict[str, Any]) -> str:
    return "convergence" if "convergence" in Path(str(row["raw_file"])).name else "grid"


def post_burn_data(data: np.ndarray, burn_fraction: float) -> np.ndarray:
    n = int(data.shape[0])
    burn = int(math.floor(float(n) * burn_fraction))
    if burn >= n:
        burn = n - 1
    return data[burn:, :]


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO algorithm comparison autocorrelation summary\n")
        handle.write(
            "# beta eta Nt update init observable mean naive_error tau_int tau_error "
            "W_opt saturated sigma_true n_meas_post_burn runtime_seconds "
            "seconds_per_sweep seconds_per_site_sweep\n"
        )
        for row in rows:
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} {row['Nt']} "
                f"{row['update']} {row['init']} {row['observable']} "
                f"{format_float(row['mean'])} {format_float(row['naive_error'])} "
                f"{format_float(row['tau_int'])} {format_float(row['tau_error'])} "
                f"{row['W_opt']} {row['saturated']} {format_float(row['sigma_true'])} "
                f"{row['n_meas_post_burn']} {format_float(row['runtime_seconds'])} "
                f"{format_float(row['seconds_per_sweep'])} "
                f"{format_float(row['seconds_per_site_sweep'])}\n"
            )


def write_slowest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO algorithm comparison slowest observable per chain\n")
        handle.write(
            "# beta eta Nt update init tau_slow tau_error_slow slowest_observable "
            "W_opt_slow saturated runtime_seconds seconds_per_sweep seconds_per_site_sweep\n"
        )
        for row in rows:
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} {row['Nt']} "
                f"{row['update']} {row['init']} {format_float(row['tau_slow'])} "
                f"{format_float(row['tau_error_slow'])} {row['slowest_observable']} "
                f"{row['W_opt_slow']} {row['saturated']} "
                f"{format_float(row['runtime_seconds'])} {format_float(row['seconds_per_sweep'])} "
                f"{format_float(row['seconds_per_site_sweep'])}\n"
            )


def write_autocorr(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO algorithm comparison autocorrelation curves\n")
        handle.write("# beta eta Nt update init observable lag C_lag\n")
        for row in rows:
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} {row['Nt']} "
                f"{row['update']} {row['init']} {row['observable']} "
                f"{row['lag']} {format_float(row['C_lag'])}\n"
            )


def write_tau_window(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO algorithm comparison running tau_int windows\n")
        handle.write("# beta eta Nt update init observable W tau_int_W\n")
        for row in rows:
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} {row['Nt']} "
                f"{row['update']} {row['init']} {row['observable']} "
                f"{row['W']} {format_float(row['tau_int_W'])}\n"
            )


def write_convergence(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO algorithm comparison hb-over zero/uniform convergence running means\n")
        handle.write("# beta eta Nt update init sweep running_y2_mean running_energy_ren\n")
        for row in rows:
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} {row['Nt']} "
                f"{row['update']} {row['init']} {row['sweep']} "
                f"{format_float(row['running_y2_mean'])} "
                f"{format_float(row['running_energy_ren'])}\n"
            )


def fit_tau_scaling(rows: list[dict[str, Any]]) -> tuple[float, float, int]:
    selected = [
        row
        for row in rows
        if row["init"] == "zero"
        and row["kind"] == "grid"
        and row["saturated"] == 1
        and math.isfinite(row["tau_slow"])
        and row["tau_slow"] > 0.0
        and row["eta"] > 0.0
    ]
    if len(selected) < 3:
        return math.nan, math.nan, len(selected)

    x = np.asarray([math.log(row["eta"]) for row in selected], dtype=float)
    y = np.asarray([math.log(row["tau_slow"]) for row in selected], dtype=float)
    design = np.column_stack((np.ones_like(x), x))
    coeff, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    amplitude = float(math.exp(float(coeff[0])))
    z_eff = float(-coeff[1])
    return amplitude, z_eff, len(selected)


def median_site_cost(manifest_rows: list[dict[str, Any]], update: str) -> float:
    values = [
        float(row["seconds_per_site_sweep"])
        for row in manifest_rows
        if row["update"] == update
        and chain_kind(row) == "grid"
        and math.isfinite(float(row["seconds_per_site_sweep"]))
        and float(row["seconds_per_site_sweep"]) > 0.0
    ]
    if not values:
        values = [
            float(row["seconds_per_site_sweep"])
            for row in manifest_rows
            if row["update"] == update
            and math.isfinite(float(row["seconds_per_site_sweep"]))
            and float(row["seconds_per_site_sweep"]) > 0.0
        ]
    return float(np.median(np.asarray(values, dtype=float))) if values else math.nan


def write_cost_projection(
    path: Path,
    slowest_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    beta: float,
) -> dict[str, dict[str, float]]:
    fit_by_update: dict[str, dict[str, float]] = {}
    cost_rows: list[dict[str, Any]] = []

    for update in UPDATES:
        update_rows = [row for row in slowest_rows if row["update"] == update]
        amplitude, z_eff, n_fit_points = fit_tau_scaling(update_rows)
        site_cost = median_site_cost(manifest_rows, update)
        fit_by_update[update] = {
            "A": amplitude,
            "z_eff": z_eff,
            "n_fit_points": float(n_fit_points),
            "median_seconds_per_site_sweep": site_cost,
        }
        if n_fit_points < 3:
            print(
                f"warning: {update}: fewer than three finite saturated tau_slow points; "
                "writing NaN scaling projection",
                file=sys.stderr,
            )

        for nt in CANDIDATE_NT:
            eta_candidate = beta / float(nt)
            if (
                math.isfinite(amplitude)
                and math.isfinite(z_eff)
                and math.isfinite(site_cost)
                and site_cost > 0.0
            ):
                tau_pred = amplitude * eta_candidate ** (-z_eff)
                seconds_per_sweep_pred = float(nt) * site_cost
                seconds_for_neff500 = 2.0 * tau_pred * 500.0 * seconds_per_sweep_pred
                seconds_for_neff1000 = 2.0 * tau_pred * 1000.0 * seconds_per_sweep_pred
            else:
                tau_pred = math.nan
                seconds_per_sweep_pred = math.nan
                seconds_for_neff500 = math.nan
                seconds_for_neff1000 = math.nan
            cost_rows.append(
                {
                    "update": update,
                    "z_eff": z_eff,
                    "A": amplitude,
                    "candidate_Nt": nt,
                    "eta_candidate": eta_candidate,
                    "tau_pred": tau_pred,
                    "seconds_per_sweep_pred": seconds_per_sweep_pred,
                    "seconds_for_Neff500": seconds_for_neff500,
                    "seconds_for_Neff1000": seconds_for_neff1000,
                    "n_fit_points": n_fit_points,
                    "median_seconds_per_site_sweep": site_cost,
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO algorithm comparison rough cost projection\n")
        handle.write("# tau_slow(eta) ~= A * eta^(-z_eff), using finite saturated init=zero grid points\n")
        handle.write("# seconds_per_sweep_pred ~= Nt * median(seconds_per_site_sweep for update)\n")
        handle.write(
            "# update z_eff A candidate_Nt eta_candidate tau_pred seconds_per_sweep_pred "
            "seconds_for_Neff500 seconds_for_Neff1000 n_fit_points "
            "median_seconds_per_site_sweep\n"
        )
        for row in cost_rows:
            handle.write(
                f"{row['update']} {format_float(row['z_eff'])} {format_float(row['A'])} "
                f"{row['candidate_Nt']} {format_float(row['eta_candidate'])} "
                f"{format_float(row['tau_pred'])} {format_float(row['seconds_per_sweep_pred'])} "
                f"{format_float(row['seconds_for_Neff500'])} "
                f"{format_float(row['seconds_for_Neff1000'])} {row['n_fit_points']} "
                f"{format_float(row['median_seconds_per_site_sweep'])}\n"
            )

    return fit_by_update


def classify_tau_growth(rows: list[dict[str, Any]], update: str) -> str:
    selected = [
        row
        for row in rows
        if row["update"] == update
        and row["init"] == "zero"
        and row["kind"] == "grid"
        and math.isfinite(row["tau_slow"])
    ]
    if len(selected) < 2:
        return "insufficient finite data"

    largest_eta_row = max(selected, key=lambda row: row["eta"])
    smallest_eta_row = min(selected, key=lambda row: row["eta"])
    if smallest_eta_row["tau_slow"] > largest_eta_row["tau_slow"]:
        return (
            f"yes, from {format_float(largest_eta_row['tau_slow'])} at eta="
            f"{format_float(largest_eta_row['eta'])} to "
            f"{format_float(smallest_eta_row['tau_slow'])} at eta="
            f"{format_float(smallest_eta_row['eta'])}"
        )
    return (
        f"not clearly, from {format_float(largest_eta_row['tau_slow'])} at eta="
        f"{format_float(largest_eta_row['eta'])} to "
        f"{format_float(smallest_eta_row['tau_slow'])} at eta="
        f"{format_float(smallest_eta_row['eta'])}"
    )


def write_recommendation(
    path: Path,
    slowest_rows: list[dict[str, Any]],
    fit_by_update: dict[str, dict[str, float]],
) -> None:
    grid_zero = [
        row
        for row in slowest_rows
        if row["kind"] == "grid" and row["init"] == "zero" and math.isfinite(row["tau_slow"])
    ]
    lines: list[str] = []
    lines.append("# QHO Algorithm Comparison Recommendation")
    lines.append("")
    lines.append(
        "This file summarizes a preliminary check comparison. It is not a final "
        "production parameter choice."
    )
    lines.append("")

    if grid_zero:
        smallest_eta = min(row["eta"] for row in grid_zero)
        at_smallest_eta = [row for row in grid_zero if row["eta"] == smallest_eta]
        best = min(at_smallest_eta, key=lambda row: row["tau_slow"])
        lines.append(
            f"- At the smallest measured eta ({format_float(smallest_eta)}), "
            f"`{best['update']}` has the smallest finite tau_slow: "
            f"{format_float(best['tau_slow'])} from `{best['slowest_observable']}`."
        )
    else:
        lines.append("- No finite init=zero grid tau_slow values were available.")

    observable_counts = Counter(
        row["slowest_observable"]
        for row in grid_zero
        if row["slowest_observable"] != "none"
    )
    if observable_counts:
        observable, count = observable_counts.most_common(1)[0]
        lines.append(f"- The slowest observable most often is `{observable}` ({count} grid chains).")
    else:
        lines.append("- The slowest observable could not be identified from finite grid chains.")

    for update in UPDATES:
        lines.append(f"- Tau growth as eta decreases for `{update}`: {classify_tau_growth(grid_zero, update)}.")

    for update in UPDATES:
        fit = fit_by_update.get(update, {})
        z_eff = float(fit.get("z_eff", math.nan))
        n_fit_points = int(fit.get("n_fit_points", 0.0))
        lines.append(
            f"- Rough z_eff for `{update}`: {format_float(z_eff)} "
            f"from {n_fit_points} saturated point(s)."
        )

    nt512_candidates: list[tuple[str, float]] = []
    projection_path = path.with_name("qho_algorithm_comparison_cost_projection.dat")
    if projection_path.is_file():
        with projection_path.open("r", encoding="ascii") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                fields = stripped.split()
                if len(fields) >= 9 and fields[3] == "512":
                    seconds = float(fields[8]) if fields[8] != "nan" else math.nan
                    if math.isfinite(seconds):
                        nt512_candidates.append((fields[0], seconds))

    if nt512_candidates:
        fastest_update, fastest_seconds = min(nt512_candidates, key=lambda item: item[1])
        feasible = "yes" if fastest_seconds < 86400.0 else "no"
        lines.append(
            "- Under a rough one-day-per-chain reference for Neff=1000, "
            f"Nt=512 looks feasible: {feasible}. Fastest projection is "
            f"`{fastest_update}` at {format_float(fastest_seconds)} seconds."
        )
    else:
        lines.append("- Nt=512 feasibility cannot be judged because the projection is NaN.")

    lines.append("")
    lines.append(
        "The projection is deliberately rough: it uses comparison tau_slow fits and a "
        "median measured seconds-per-site-sweep. It does not set final therm, "
        "stride, block_size, or sweeps."
    )

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
    convergence_rows: list[dict[str, Any]] = []

    for manifest_row in manifest_rows:
        raw_path = resolve_raw_path(str(manifest_row["raw_file"]), args.manifest)
        try:
            data = load_measurements(raw_path)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        burned = post_burn_data(data, args.burn_fraction)
        observable_results: dict[str, dict[str, Any]] = {}
        selected_for_curves = (
            chain_kind(manifest_row) == "grid"
            and manifest_row["init"] == "zero"
            and manifest_row["Nt"] in CURVE_NT
            and manifest_row["update"] in UPDATES
        )

        for observable in OBSERVABLES:
            column = COLUMNS.index(observable)
            result = analyze_series(burned[:, column], args.max_lag, args.window_c)
            observable_results[observable] = result
            summary_rows.append(
                {
                    **manifest_row,
                    "kind": chain_kind(manifest_row),
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

            if selected_for_curves and result["rho"] is not None:
                rho = result["rho"]
                for lag, c_lag in enumerate(rho):
                    autocorr_rows.append(
                        {
                            **manifest_row,
                            "observable": observable,
                            "lag": lag,
                            "C_lag": float(c_lag),
                        }
                    )
            if selected_for_curves and result["tau_curve"] is not None:
                tau_curve = result["tau_curve"]
                for window, tau_value in enumerate(tau_curve):
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
            for observable, result in observable_results.items()
            if math.isfinite(float(result["tau_int"]))
        ]
        if finite:
            slowest_observable, slowest_result = max(
                finite, key=lambda item: float(item[1]["tau_int"])
            )
            tau_slow = float(slowest_result["tau_int"])
            tau_error_slow = float(slowest_result["tau_error"])
            w_opt_slow = int(slowest_result["W_opt"])
            saturated = int(slowest_result["saturated"])
        else:
            slowest_observable = "none"
            tau_slow = math.nan
            tau_error_slow = math.nan
            w_opt_slow = 0
            saturated = 0

        slowest_rows.append(
            {
                **manifest_row,
                "kind": chain_kind(manifest_row),
                "tau_slow": tau_slow,
                "tau_error_slow": tau_error_slow,
                "slowest_observable": slowest_observable,
                "W_opt_slow": w_opt_slow,
                "saturated": saturated,
            }
        )

        if (
            chain_kind(manifest_row) == "convergence"
            and manifest_row["update"] == "hb-over"
            and manifest_row["Nt"] in CURVE_NT
            and manifest_row["init"] in {"zero", "uniform"}
        ):
            sweeps = data[:, COLUMNS.index("sweep")]
            y2_running = np.cumsum(data[:, COLUMNS.index("y2_mean")]) / np.arange(
                1, data.shape[0] + 1, dtype=float
            )
            energy_running = np.cumsum(data[:, COLUMNS.index("energy_ren")]) / np.arange(
                1, data.shape[0] + 1, dtype=float
            )
            for sweep, y2_value, energy_value in zip(sweeps, y2_running, energy_running):
                convergence_rows.append(
                    {
                        **manifest_row,
                        "sweep": int(round(float(sweep))),
                        "running_y2_mean": float(y2_value),
                        "running_energy_ren": float(energy_value),
                    }
                )

    write_summary(output_dir / "qho_algorithm_comparison_summary.dat", summary_rows)
    write_slowest(output_dir / "qho_algorithm_comparison_slowest.dat", slowest_rows)
    write_autocorr(output_dir / "qho_algorithm_comparison_autocorr.dat", autocorr_rows)
    write_tau_window(output_dir / "qho_algorithm_comparison_tau_window.dat", tau_window_rows)
    write_convergence(output_dir / "qho_algorithm_comparison_convergence.dat", convergence_rows)

    beta_values = sorted({float(row["beta"]) for row in manifest_rows})
    projection_beta = 5.0 if 5.0 in beta_values else beta_values[0]
    fit_by_update = write_cost_projection(
        output_dir / "qho_algorithm_comparison_cost_projection.dat",
        slowest_rows,
        manifest_rows,
        projection_beta,
    )
    write_recommendation(
        output_dir / "qho_algorithm_comparison_recommendation.md",
        slowest_rows,
        fit_by_update,
    )

    print(f"Wrote {output_dir / 'qho_algorithm_comparison_summary.dat'}")
    print(f"Wrote {output_dir / 'qho_algorithm_comparison_slowest.dat'}")
    print(f"Wrote {output_dir / 'qho_algorithm_comparison_autocorr.dat'}")
    print(f"Wrote {output_dir / 'qho_algorithm_comparison_tau_window.dat'}")
    print(f"Wrote {output_dir / 'qho_algorithm_comparison_convergence.dat'}")
    print(f"Wrote {output_dir / 'qho_algorithm_comparison_cost_projection.dat'}")
    print(f"Wrote {output_dir / 'qho_algorithm_comparison_recommendation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
