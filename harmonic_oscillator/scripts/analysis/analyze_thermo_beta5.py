#!/usr/bin/env python3
# Analyze beta=5 thermodynamic ensembles across lattice spacings, using blocked
# Monte Carlo errors and eta^2 fits for continuum limits of <y^2> and H_ren.
"""
Analyze beta=5 thermodynamic production runs for the QHO PIMC project.

The script reads the production manifest written by
`scripts/run/run_thermo_beta5_production.sh`, loads the raw measurement time
series for each lattice spacing, builds block averages, and writes the tables
used by the final thermodynamic plots and numerical summary.

The main continuum model is

    O(eta) = A + B eta^2,

applied to `<y^2>` and to the renormalized energy estimator `H_ren`. Reported
fit errors use blocked Monte Carlo errors only; they do not include an
additional systematic uncertainty from the choice of fit window.

For the harmonic oscillator in units hbar omega = 1,

    <y^2>_beta = <H>_beta = 0.5 coth(beta / 2),

so the exact beta=5 benchmark is used for both `<y^2>` and `energy_ren`.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


# Column order written by the C executable for ASCII thermodynamic outputs.
# Keeping this tuple explicit avoids relying on column names inside the data file.
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

# Observables averaged directly from saved measurements. Only the two entries
# in FIT_OBSERVABLES are used for continuum extrapolations.
OBSERVABLES = ("y_mean", "y2_mean", "dy2_mean", "energy_ren", "acc_rate")
FIT_OBSERVABLES = ("y2_mean", "energy_ren")
ETA_CUTS = (0.2, 0.15, 0.1, 0.075, 0.05)

# Exact finite-temperature benchmark for the harmonic oscillator.
BETA_FINAL = 5.0
EXACT_BETA5 = 0.5 / math.tanh(0.5 * BETA_FINAL)
# Continuum-window scans are restricted to the small-eta region used in the
# final beta=5 analysis.
SCAN_MAX_ETA = 0.2
SCAN_MIN_POINTS = 6
SCAN_TARGET_CHI2_RED = 0.9
# Production settings expected from the run script. They are used only to label
# shortened runs in the recommendation file.
DEFAULT_THERM = 400000
DEFAULT_SWEEPS = 1000000
DEFAULT_STRIDE = 10
DEFAULT_BLOCK_SAVED = 2000
VIRIAL_ETA_CUT = 0.2
Y2_FIT_NT_MIN = 25
Y2_FIT_NT_MAX = 200
ENERGY_FIT_NT_MIN = 25
ENERGY_FIT_NT_MAX = 80
# Final y2 continuum-fit parameters used in the z-score table. These constants
# keep the check synchronized with the displayed beta=5 result.
Y2_CONT_A = 0.50596354359
Y2_CONT_A_ERR = 0.000801253769023
Y2_CONT_B = -0.0234823662359
Y2_CONT_B_ERR = 0.0340175848446
Y2_FIT_ETA_MIN = 0.025
Y2_FIT_ETA_MAX = 0.2
# Markers used to update only the generated sections of the recommendation file.
VIRIAL_SUMMARY_BEGIN = "<!-- beta5_virial_check_summary:start -->"
VIRIAL_SUMMARY_END = "<!-- beta5_virial_check_summary:end -->"
Y2_ZSCORE_SUMMARY_BEGIN = "<!-- beta5_y2_zscore_summary:start -->"
Y2_ZSCORE_SUMMARY_END = "<!-- beta5_y2_zscore_summary:end -->"
# Manifest schema written by scripts/run/run_thermo_beta5_production.sh.
MANIFEST_COLUMNS = (
    "beta",
    "eta",
    "Nt",
    "update",
    "init",
    "seed",
    "stream",
    "n_therm",
    "n_sweeps",
    "meas_stride",
    "n_over",
    "block_size_saved",
    "raw_file",
    "runtime_seconds",
    "seconds_per_sweep",
    "seconds_per_site_sweep",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the beta=5 production analysis."""
    parser = argparse.ArgumentParser(description="Analyze beta=5 thermodynamic production data.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/production/qho_thermo_beta5_manifest.dat"),
        help="Manifest written by scripts/run/run_thermo_beta5_production.sh",
    )
    parser.add_argument(
        "--virial-only",
        action="store_true",
        help="Only write the beta=5 virial table, leaving existing fit outputs untouched.",
    )
    parser.add_argument(
        "--update-recommendation",
        action="store_true",
        help="When used with --virial-only, also update the recommendation markdown checks.",
    )
    return parser.parse_args()


def format_float(value: float) -> str:
    """Format table values compactly while preserving enough significant digits."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.12g}"


def format_exact(value: float) -> str:
    """Format exact reference values with higher precision than noisy estimates."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.17g}"


def parse_manifest(path: Path) -> list[dict[str, Any]]:
    """Read the production manifest and convert numeric fields to Python types."""
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

            # Convert early so that downstream code can treat rows as typed records.
            row["beta"] = float(row["beta"])
            row["eta"] = float(row["eta"])
            row["Nt"] = int(row["Nt"])
            row["seed"] = int(row["seed"])
            row["stream"] = int(row["stream"])
            row["n_therm"] = int(row["n_therm"])
            row["n_sweeps"] = int(row["n_sweeps"])
            row["meas_stride"] = int(row["meas_stride"])
            row["n_over"] = int(row["n_over"])
            row["block_size_saved"] = int(row["block_size_saved"])
            row["runtime_seconds"] = float(row["runtime_seconds"])
            row["seconds_per_sweep"] = float(row["seconds_per_sweep"])
            row["seconds_per_site_sweep"] = float(row["seconds_per_site_sweep"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no manifest rows found")
    return rows


def project_root_from_manifest(path: Path) -> Path:
    """Infer the repository root from the standard processed/production path."""
    resolved = path.resolve()
    if (
        len(resolved.parents) >= 4
        and resolved.parents[0].name == "production"
        and resolved.parents[1].name == "processed"
        and resolved.parents[2].name == "data"
    ):
        return resolved.parents[3]
    return Path.cwd()


def resolve_raw_path(raw_file: str, manifest: Path) -> Path:
    """Resolve a raw-data path stored in the manifest."""
    raw_path = Path(raw_file)
    if raw_path.is_absolute():
        return raw_path
    if raw_path.is_file():
        return raw_path
    return project_root_from_manifest(manifest) / raw_path


def load_measurements(path: Path) -> np.ndarray:
    """Load one ASCII measurement time series produced by bin/qho_pimc."""
    try:
        data = np.loadtxt(path, comments="#")
    except ValueError as exc:
        raise ValueError(f"{path}: failed to load numeric rows: {exc}") from exc
    if data.size == 0:
        raise ValueError(f"{path}: no measurement rows found")
    data = np.atleast_2d(data)
    if data.shape[1] < len(COLUMNS):
        raise ValueError(f"{path}: expected at least {len(COLUMNS)} columns, found {data.shape[1]}")

    # Ignore any extra columns so the analysis remains compatible with extended outputs.
    return data[:, : len(COLUMNS)]


def block_means(series: np.ndarray, block_size_saved: int) -> np.ndarray:
    """Return means of consecutive saved-measurement blocks."""
    n_measurements = int(series.size)
    if n_measurements <= 0:
        return np.asarray([], dtype=float)
    actual_block_size = min(block_size_saved, n_measurements)
    n_blocks = n_measurements // actual_block_size

    # Drop a final incomplete block; all reported errors use equal-size blocks.
    trimmed = series[: n_blocks * actual_block_size]
    return trimmed.reshape(n_blocks, actual_block_size).mean(axis=1)


def mean_error_from_blocks(block_values: np.ndarray) -> tuple[float, float, int]:
    """Return mean, standard error of the mean, and number of blocks."""
    n_blocks = int(block_values.size)
    if n_blocks <= 0:
        return math.nan, math.nan, 0
    mean = float(block_values.mean())
    if n_blocks >= 2:
        error = float(block_values.std(ddof=1) / math.sqrt(float(n_blocks)))
    else:
        error = math.nan
    return mean, error, n_blocks


def blocked_mean_error(series: np.ndarray, block_size_saved: int) -> tuple[float, float, int]:
    """Convenience wrapper for block means followed by a standard-error estimate."""
    return mean_error_from_blocks(block_means(series, block_size_saved))


def virial_estimates(data: np.ndarray, eta: float, block_size_saved: int) -> dict[str, float | int]:
    """
    Build block estimates for the virial check at one lattice spacing.

    The diagnostic is D = K_ren - V. For the harmonic oscillator, D should be
    compatible with zero up to statistical errors and finite-sampling noise.
    """
    y2_blocks = block_means(data[:, COLUMNS.index("y2_mean")], block_size_saved)
    dy2_blocks = block_means(data[:, COLUMNS.index("dy2_mean")], block_size_saved)
    energy_blocks = block_means(data[:, COLUMNS.index("energy_ren")], block_size_saved)
    if not (y2_blocks.size == dy2_blocks.size == energy_blocks.size):
        raise ValueError("inconsistent block counts while building beta=5 virial estimates")

    # Work block-by-block so the error on D includes correlations between K and V.
    v_blocks = 0.5 * y2_blocks
    kdiv_blocks = -(0.5 / (eta * eta)) * dy2_blocks
    k_blocks = kdiv_blocks + 0.5 / eta
    d_blocks = k_blocks - v_blocks
    v_mean, v_err, n_blocks = mean_error_from_blocks(v_blocks)
    k_mean, k_err, _ = mean_error_from_blocks(k_blocks)
    kdiv_mean, kdiv_err, _ = mean_error_from_blocks(kdiv_blocks)
    d_mean, d_err, _ = mean_error_from_blocks(d_blocks)
    energy_mean, energy_err, _ = mean_error_from_blocks(energy_blocks)
    z_virial = d_mean / d_err if math.isfinite(d_err) and d_err > 0.0 else math.nan
    abs_z_virial = abs(z_virial) if math.isfinite(z_virial) else math.nan
    return {
        "V_mean": v_mean,
        "V_err": v_err,
        "Kren_mean": k_mean,
        "Kren_err": k_err,
        "Kdiv_mean": kdiv_mean,
        "Kdiv_err": kdiv_err,
        "D_mean": d_mean,
        "D_err": d_err,
        "z_virial": z_virial,
        "abs_z_virial": abs_z_virial,
        "pass_1sigma": 1 if math.isfinite(abs_z_virial) and abs_z_virial <= 1.0 else 0,
        "pass_2sigma": 1 if math.isfinite(abs_z_virial) and abs_z_virial <= 2.0 else 0,
        "energy_ren": energy_mean,
        "energy_ren_err": energy_err,
        "V_plus_Kren_diff": v_mean + k_mean - energy_mean,
        "n_blocks": n_blocks,
    }


def analyze_chain(row: dict[str, Any], manifest: Path) -> dict[str, Any]:
    """Analyze one Nt entry from the production manifest."""
    raw_path = resolve_raw_path(str(row["raw_file"]), manifest)
    data = load_measurements(raw_path)
    block_size_saved = int(row["block_size_saved"])
    block_size_sweeps = block_size_saved * int(row["meas_stride"])
    estimates: dict[str, tuple[float, float, int]] = {}
    n_blocks_common = 0
    for observable in OBSERVABLES:
        # The same block definition is used for all observables at this Nt.
        mean, error, n_blocks = blocked_mean_error(data[:, COLUMNS.index(observable)], block_size_saved)
        estimates[observable] = (mean, error, n_blocks)
        n_blocks_common = n_blocks
    return {
        **row,
        "eta2": float(row["eta"]) * float(row["eta"]),
        "n_measurements": int(data.shape[0]),
        "n_blocks": n_blocks_common,
        "block_size_sweeps": block_size_sweeps,
        "estimates": estimates,
        "virial": virial_estimates(data, float(row["eta"]), block_size_saved),
    }


def weighted_fit_selected(selected: list[dict[str, Any]], observable: str) -> dict[str, Any]:
    """Fit one observable to A + B eta^2 using blocked Monte Carlo errors."""
    n_points = len(selected)
    if n_points < 3:
        return {
            "observable": observable,
            "n_points": n_points,
            "A": math.nan,
            "A_error": math.nan,
            "B": math.nan,
            "B_error": math.nan,
            "chi2": math.nan,
            "dof": n_points - 2,
            "chi2_red": math.nan,
            "exact": EXACT_BETA5,
            "A_minus_exact": math.nan,
            "weighted": 0,
        }

    x = np.asarray([float(row["eta2"]) for row in selected], dtype=float)
    y = np.asarray([float(row["estimates"][observable][0]) for row in selected], dtype=float)
    sigma = np.asarray([float(row["estimates"][observable][1]) for row in selected], dtype=float)
    design = np.column_stack((np.ones_like(x), x))

    # Weighted least squares with diagonal covariance from block errors.
    weights = 1.0 / (sigma * sigma)
    normal = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * y)
    try:
        coeff = np.linalg.solve(normal, rhs)
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return {
            "observable": observable,
            "n_points": n_points,
            "A": math.nan,
            "A_error": math.nan,
            "B": math.nan,
            "B_error": math.nan,
            "chi2": math.nan,
            "dof": n_points - 2,
            "chi2_red": math.nan,
            "exact": EXACT_BETA5,
            "A_minus_exact": math.nan,
            "weighted": 0,
        }

    residual = y - design @ coeff
    chi2 = float(np.sum((residual / sigma) ** 2))
    dof = n_points - 2
    chi2_red = chi2 / float(dof) if dof > 0 else math.nan
    return {
        "observable": observable,
        "n_points": n_points,
        "A": float(coeff[0]),
        "A_error": float(math.sqrt(max(float(covariance[0, 0]), 0.0))),
        "B": float(coeff[1]),
        "B_error": float(math.sqrt(max(float(covariance[1, 1]), 0.0))),
        "chi2": chi2,
        "dof": dof,
        "chi2_red": chi2_red,
        "exact": EXACT_BETA5,
        "A_minus_exact": float(coeff[0]) - EXACT_BETA5,
        "weighted": 1,
    }


def weighted_fit_eta2(rows: list[dict[str, Any]], observable: str, eta_cut: float) -> dict[str, Any]:
    """Fit all points with eta <= eta_cut for one observable."""
    selected = [
        row
        for row in rows
        if float(row["eta"]) <= eta_cut
        and math.isfinite(float(row["estimates"][observable][1]))
        and float(row["estimates"][observable][1]) > 0.0
    ]
    fit = weighted_fit_selected(selected, observable)
    fit["eta_cut"] = eta_cut
    return fit


def scan_contiguous_windows(rows: list[dict[str, Any]], observable: str) -> list[dict[str, Any]]:
    """Scan all contiguous eta windows in the accepted small-eta region."""
    available = [
        row
        for row in sorted(rows, key=lambda item: float(item["eta"]))
        if 0.0 <= float(row["eta"]) <= SCAN_MAX_ETA
        and math.isfinite(float(row["estimates"][observable][1]))
        and float(row["estimates"][observable][1]) > 0.0
    ]
    scans: list[dict[str, Any]] = []
    for start in range(len(available)):
        for stop in range(start + SCAN_MIN_POINTS, len(available) + 1):
            selected = available[start:stop]
            fit = weighted_fit_selected(selected, observable)
            eta_values = [float(row["eta"]) for row in selected]
            chi2_red = float(fit["chi2_red"])
            fit.update(
                {
                    "eta_min": min(eta_values),
                    "eta_max": max(eta_values),
                    "target_distance": (
                        abs(chi2_red - SCAN_TARGET_CHI2_RED)
                        if math.isfinite(chi2_red)
                        else math.nan
                    ),
                }
            )
            scans.append(fit)
    return scans


def choose_recommended_scan(scans: list[dict[str, Any]], observable: str) -> dict[str, Any] | None:
    """Pick a representative contiguous-window fit for the recommendation file."""
    valid = [
        scan
        for scan in scans
        if scan["observable"] == observable
        and scan["weighted"] == 1
        and math.isfinite(float(scan["chi2_red"]))
        and int(scan["n_points"]) >= SCAN_MIN_POINTS
    ]
    if not valid:
        return None
    preferred = [
        scan
        for scan in valid
        if 0.8 <= float(scan["chi2_red"]) <= 1.0
    ]
    if preferred:
        return min(
            preferred,
            key=lambda scan: (
                -int(scan["n_points"]),
                abs(float(scan["A_minus_exact"])),
                abs(float(scan["chi2_red"]) - SCAN_TARGET_CHI2_RED),
            ),
        )
    return min(valid, key=lambda scan: abs(float(scan["chi2_red"]) - SCAN_TARGET_CHI2_RED))


def fit_window_metadata(selected: list[dict[str, Any]]) -> dict[str, Any]:
    """Return Nt and eta bounds for a selected fit window."""
    eta_values = [float(row["eta"]) for row in selected]
    nt_values = [int(row["Nt"]) for row in selected]
    return {
        "Nt_min": min(nt_values),
        "Nt_max": max(nt_values),
        "eta_min": min(eta_values),
        "eta_max": max(eta_values),
    }


def add_sigma_flags(fit: dict[str, Any]) -> None:
    """Annotate a fit with one- and two-sigma agreement flags."""
    diff = abs(float(fit["A_minus_exact"]))
    err = float(fit["A_error"])
    if math.isfinite(diff) and math.isfinite(err) and err > 0.0:
        fit["within_1sigma"] = 1 if diff <= err else 0
        fit["within_2sigma"] = 1 if diff <= 2.0 * err else 0
    else:
        fit["within_1sigma"] = 0
        fit["within_2sigma"] = 0


def scan_targeted_windows(rows: list[dict[str, Any]], observable: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the two targeted scans used to choose the displayed continuum fits."""
    available = [
        row
        for row in sorted(rows, key=lambda item: float(item["eta"]))
        if float(row["eta"]) <= SCAN_MAX_ETA
        and math.isfinite(float(row["estimates"][observable][1]))
        and float(row["estimates"][observable][1]) > 0.0
    ]
    scan_a: list[dict[str, Any]] = []
    scan_b: list[dict[str, Any]] = []
    if len(available) < SCAN_MIN_POINTS:
        return scan_a, scan_b

    # Scan A removes the finest points first while keeping eta_max fixed.
    for removed in range(0, len(available) - SCAN_MIN_POINTS + 1):
        selected = available[removed:]
        fit = weighted_fit_selected(selected, observable)
        fit.update(fit_window_metadata(selected))
        fit["removed_largest_Nt_count"] = removed
        add_sigma_flags(fit)
        scan_a.append(fit)

    # Scan B fixes the finest point and progressively adds coarser points.
    for stop in range(SCAN_MIN_POINTS, len(available) + 1):
        selected = available[:stop]
        fit = weighted_fit_selected(selected, observable)
        fit.update(fit_window_metadata(selected))
        add_sigma_flags(fit)
        scan_b.append(fit)

    return scan_a, scan_b


def choose_targeted_candidate(rows: list[dict[str, Any]], observable: str) -> dict[str, Any] | None:
    """Select one targeted-window candidate for one observable."""
    valid = [
        row
        for row in rows
        if row["observable"] == observable
        and row["weighted"] == 1
        and math.isfinite(float(row["chi2_red"]))
        and int(row["n_points"]) >= SCAN_MIN_POINTS
        and float(row["chi2_red"]) < 1.0
    ]
    if not valid:
        return None
    within_two_sigma = [row for row in valid if int(row["within_2sigma"]) == 1]
    pool = within_two_sigma if within_two_sigma else valid
    return min(
        pool,
        key=lambda row: (
            abs(float(row["chi2_red"]) - SCAN_TARGET_CHI2_RED),
            -int(row["n_points"]),
            abs(float(row["A_minus_exact"])),
        ),
    )


def write_points(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one long table with one row per Nt and observable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO beta=5 thermodynamic production points\n")
        handle.write("# beta eta eta2 Nt observable mean error n_blocks block_size_saved block_size_sweeps\n")
        for row in rows:
            for observable in OBSERVABLES:
                mean, error, n_blocks = row["estimates"][observable]
                handle.write(
                    f"{format_float(row['beta'])} {format_float(row['eta'])} "
                    f"{format_float(row['eta2'])} {row['Nt']} {observable} "
                    f"{format_float(mean)} {format_float(error)} {n_blocks} "
                    f"{row['block_size_saved']} {row['block_size_sweeps']}\n"
                )


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one wide table with the main thermodynamic estimates at each Nt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO beta=5 thermodynamic production summary\n")
        handle.write(
            "# beta eta eta2 Nt y_mean y_mean_err y2_mean y2_mean_err "
            "dy2_mean dy2_mean_err energy_ren energy_ren_err acc_rate acc_rate_err "
            "n_blocks block_size_saved block_size_sweeps\n"
        )
        for row in rows:
            est = row["estimates"]
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} {format_float(row['eta2'])} {row['Nt']} "
                f"{format_float(est['y_mean'][0])} {format_float(est['y_mean'][1])} "
                f"{format_float(est['y2_mean'][0])} {format_float(est['y2_mean'][1])} "
                f"{format_float(est['dy2_mean'][0])} {format_float(est['dy2_mean'][1])} "
                f"{format_float(est['energy_ren'][0])} {format_float(est['energy_ren'][1])} "
                f"{format_float(est['acc_rate'][0])} {format_float(est['acc_rate'][1])} "
                f"{row['n_blocks']} {row['block_size_saved']} {row['block_size_sweeps']}\n"
            )


def y2_zscore_flags(row: dict[str, Any]) -> dict[str, int]:
    """Label y2 points by their role in the final continuum window."""
    eta = float(row["eta"])
    nt = int(row["Nt"])
    return {
        "selected_y2_fit": 1 if Y2_FIT_NT_MIN <= nt <= Y2_FIT_NT_MAX else 0,
        "coarse_eta_point": 1 if eta > Y2_FIT_ETA_MAX + 1.0e-12 else 0,
        "fine_excluded_point": 1 if eta < Y2_FIT_ETA_MIN - 1.0e-12 else 0,
    }


def y2_zscore_values(row: dict[str, Any]) -> dict[str, float | int]:
    """Compute z-scores relative to the exact value and the final fit curve."""
    y2_mean, y2_err, _ = row["estimates"]["y2_mean"]
    eta2 = float(row["eta2"])
    if math.isfinite(float(y2_err)) and float(y2_err) > 0.0:
        z_cont = (float(y2_mean) - EXACT_BETA5) / float(y2_err)
        z_fit = (float(y2_mean) - (Y2_CONT_A + Y2_CONT_B * eta2)) / float(y2_err)
    else:
        z_cont = math.nan
        z_fit = math.nan
    return {
        "y2_mean": float(y2_mean),
        "y2_err": float(y2_err),
        "z_cont": z_cont,
        "abs_z_cont": abs(z_cont) if math.isfinite(z_cont) else math.nan,
        "z_fit": z_fit,
        "abs_z_fit": abs(z_fit) if math.isfinite(z_fit) else math.nan,
    }


def write_y2_zscore(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the y2 z-score table used to visualize coarse-lattice breakdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO beta=5 y2 continuum-scaling z-score check\n")
        handle.write("# z_cont = (y2_mean - exact_y2) / y2_err\n")
        handle.write("# z_fit = (y2_mean - (A_cont + B_fit*eta2)) / y2_err\n")
        handle.write("# y2_err is the blocked Monte Carlo error from the beta=5 production analysis.\n")
        handle.write(
            f"# selected y2 fit window: Nt={Y2_FIT_NT_MIN}..{Y2_FIT_NT_MAX}, "
            f"eta={format_float(Y2_FIT_ETA_MIN)}..{format_float(Y2_FIT_ETA_MAX)}\n"
        )
        handle.write(
            "# beta eta eta2 Nt y2_mean y2_err exact_y2 A_cont A_err B_fit B_err "
            "z_cont abs_z_cont z_fit abs_z_fit selected_y2_fit coarse_eta_point "
            "fine_excluded_point\n"
        )
        for row in rows:
            values = y2_zscore_values(row)
            flags = y2_zscore_flags(row)
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} "
                f"{format_float(row['eta2'])} {row['Nt']} "
                f"{format_float(values['y2_mean'])} {format_float(values['y2_err'])} "
                f"{format_exact(EXACT_BETA5)} "
                f"{format_float(Y2_CONT_A)} {format_float(Y2_CONT_A_ERR)} "
                f"{format_float(Y2_CONT_B)} {format_float(Y2_CONT_B_ERR)} "
                f"{format_float(values['z_cont'])} {format_float(values['abs_z_cont'])} "
                f"{format_float(values['z_fit'])} {format_float(values['abs_z_fit'])} "
                f"{flags['selected_y2_fit']} {flags['coarse_eta_point']} "
                f"{flags['fine_excluded_point']}\n"
            )


def write_virial(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write V, K_ren, and the divergent kinetic contribution for each Nt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO beta=5 virial-theorem production table\n")
        handle.write("# V = 0.5 * <y^2>\n")
        handle.write("# Kren = -(1/(2*eta^2)) * <Delta y^2> + 1/(2*eta)\n")
        handle.write("# Kdiv = -(1/(2*eta^2)) * <Delta y^2>\n")
        handle.write("# Errors are standard errors of saved-block means built from the raw production measurements.\n")
        handle.write(
            "# beta eta eta2 Nt V_mean V_err Kren_mean Kren_err Kdiv_mean Kdiv_err "
            "energy_ren energy_ren_err V_plus_Kren_diff n_blocks block_size_saved block_size_sweeps\n"
        )
        for row in rows:
            virial = row["virial"]
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} "
                f"{format_float(row['eta2'])} {row['Nt']} "
                f"{format_float(virial['V_mean'])} {format_float(virial['V_err'])} "
                f"{format_float(virial['Kren_mean'])} {format_float(virial['Kren_err'])} "
                f"{format_float(virial['Kdiv_mean'])} {format_float(virial['Kdiv_err'])} "
                f"{format_float(virial['energy_ren'])} {format_float(virial['energy_ren_err'])} "
                f"{format_float(virial['V_plus_Kren_diff'])} {virial['n_blocks']} "
                f"{row['block_size_saved']} {row['block_size_sweeps']}\n"
            )


def write_kdiv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the divergent kinetic term separately for plotting its eta scaling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO beta=5 divergent kinetic contribution\n")
        handle.write("# Kdiv = -<Delta y^2> / (2*eta^2)\n")
        handle.write("# Errors are standard errors of saved-block Kdiv means.\n")
        handle.write("# beta eta eta2 Nt Kdiv_mean Kdiv_err n_blocks block_size_saved block_size_sweeps\n")
        for row in rows:
            virial = row["virial"]
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} "
                f"{format_float(row['eta2'])} {row['Nt']} "
                f"{format_float(virial['Kdiv_mean'])} {format_float(virial['Kdiv_err'])} "
                f"{virial['n_blocks']} {row['block_size_saved']} {row['block_size_sweeps']}\n"
            )


def virial_fit_flags(row: dict[str, Any]) -> dict[str, int]:
    """Flag whether a point belongs to the selected final fit windows."""
    eta = float(row["eta"])
    nt = int(row["Nt"])
    coarse_eta_point = 1 if eta > VIRIAL_ETA_CUT + 1.0e-12 else 0
    selected_y2_fit = 1 if Y2_FIT_NT_MIN <= nt <= Y2_FIT_NT_MAX else 0
    selected_energy_fit = 1 if ENERGY_FIT_NT_MIN <= nt <= ENERGY_FIT_NT_MAX else 0
    excluded_y2_fit = 1 if coarse_eta_point == 0 and selected_y2_fit == 0 else 0
    excluded_energy_fit = 1 if coarse_eta_point == 0 and selected_energy_fit == 0 else 0
    return {
        "selected_y2_fit": selected_y2_fit,
        "selected_energy_fit": selected_energy_fit,
        "excluded_y2_fit": excluded_y2_fit,
        "excluded_energy_fit": excluded_energy_fit,
        "coarse_eta_point": coarse_eta_point,
    }


def write_virial_check(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the block-level virial consistency diagnostic D = K_ren - V."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO beta=5 virial-consistency check\n")
        handle.write("# D = Kren - V; the continuum virial theorem predicts D = 0.\n")
        handle.write("# V_block = 0.5 * y2_block_mean\n")
        handle.write("# Kren_block = -dy2_block_mean/(2*eta^2) + 1/(2*eta)\n")
        handle.write("# Kdiv_block = -dy2_block_mean/(2*eta^2)\n")
        handle.write("# D_err is the standard error over D_block, not a quadrature of final errors.\n")
        handle.write(
            f"# selected_y2_fit: 1 for {Y2_FIT_NT_MIN} <= Nt <= {Y2_FIT_NT_MAX}.\n"
        )
        handle.write(
            f"# selected_energy_fit: 1 for {ENERGY_FIT_NT_MIN} <= Nt <= {ENERGY_FIT_NT_MAX}.\n"
        )
        handle.write("# coarse_eta_point: 1 for eta > 0.2, outside the small-eta continuum-fit candidate region.\n")
        handle.write(
            "# beta eta eta2 Nt V_mean V_err Kren_mean Kren_err Kdiv_mean Kdiv_err D_mean D_err "
            "z_virial abs_z_virial pass_1sigma pass_2sigma selected_y2_fit "
            "selected_energy_fit excluded_y2_fit excluded_energy_fit coarse_eta_point "
            "n_blocks block_size_saved block_size_sweeps\n"
        )
        for row in rows:
            virial = row["virial"]
            flags = virial_fit_flags(row)
            handle.write(
                f"{format_float(row['beta'])} {format_float(row['eta'])} "
                f"{format_float(row['eta2'])} {row['Nt']} "
                f"{format_float(virial['V_mean'])} {format_float(virial['V_err'])} "
                f"{format_float(virial['Kren_mean'])} {format_float(virial['Kren_err'])} "
                f"{format_float(virial['Kdiv_mean'])} {format_float(virial['Kdiv_err'])} "
                f"{format_float(virial['D_mean'])} {format_float(virial['D_err'])} "
                f"{format_float(virial['z_virial'])} {format_float(virial['abs_z_virial'])} "
                f"{virial['pass_1sigma']} {virial['pass_2sigma']} "
                f"{flags['selected_y2_fit']} {flags['selected_energy_fit']} "
                f"{flags['excluded_y2_fit']} {flags['excluded_energy_fit']} "
                f"{flags['coarse_eta_point']} {virial['n_blocks']} "
                f"{row['block_size_saved']} {row['block_size_sweeps']}\n"
            )


def virial_small_eta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the eta <= VIRIAL_ETA_CUT subset used in the virial summary."""
    return [row for row in rows if float(row["eta"]) <= VIRIAL_ETA_CUT + 1.0e-12]


def count_virial_failures(rows: list[dict[str, Any]], threshold: float) -> int:
    """Count points whose virial z-score is larger than the chosen threshold."""
    return sum(
        1
        for row in rows
        if math.isfinite(float(row["virial"]["abs_z_virial"]))
        and float(row["virial"]["abs_z_virial"]) > threshold
    )


def virial_summary_section(rows: list[dict[str, Any]]) -> str:
    """Build the Markdown section summarizing the beta=5 virial check."""
    small_rows = virial_small_eta_rows(rows)
    y2_excluded = [row for row in small_rows if virial_fit_flags(row)["excluded_y2_fit"] == 1]
    energy_excluded = [
        row for row in small_rows if virial_fit_flags(row)["excluded_energy_fit"] == 1
    ]
    fail1 = [
        row
        for row in small_rows
        if math.isfinite(float(row["virial"]["abs_z_virial"]))
        and float(row["virial"]["abs_z_virial"]) > 1.0
    ]
    fail2 = [
        row
        for row in small_rows
        if math.isfinite(float(row["virial"]["abs_z_virial"]))
        and float(row["virial"]["abs_z_virial"]) > 2.0
    ]
    fail1_y2 = count_virial_failures(y2_excluded, 1.0)
    fail2_y2 = count_virial_failures(y2_excluded, 2.0)
    fail1_energy = count_virial_failures(energy_excluded, 1.0)
    fail2_energy = count_virial_failures(energy_excluded, 2.0)
    fail1_total = len(fail1)
    fail2_total = len(fail2)
    if fail1_total == 0:
        concentration = "No eta <= 0.2 point fails at 1 sigma."
    elif fail1_y2 == fail1_total or fail1_energy == fail1_total:
        concentration = (
            "The 1-sigma virial failures are concentrated among finest-eta points "
            "excluded from at least one selected final window."
        )
    else:
        concentration = (
            "The 1-sigma virial failures are not exclusively concentrated among the "
            "finest-eta points excluded from the selected final windows."
        )

    lines = [
        VIRIAL_SUMMARY_BEGIN,
        "## Beta=5 Virial Consistency Check",
        "",
        f"- eta <= {format_float(VIRIAL_ETA_CUT)} points: {len(small_rows)}",
        f"- points failing virial at 1 sigma: {fail1_total}",
        f"- points failing virial at 2 sigma: {fail2_total}",
        (
            "- excluded from final y2_mean window: "
            f"{len(y2_excluded)} points; failing at 1 sigma: {fail1_y2}; "
            f"failing at 2 sigma: {fail2_y2}"
        ),
        (
            "- excluded from final energy_ren window: "
            f"{len(energy_excluded)} points; failing at 1 sigma: {fail1_energy}; "
            f"failing at 2 sigma: {fail2_energy}"
        ),
        f"- concentration check: {concentration}",
        (
            "- interpretation: the final continuum-fit windows were selected from "
            "chi2_red scans, not from the virial test. This virial check is an "
            "independent check of the same production data and highlights the "
            "statistical noise of Kren at fine eta."
        ),
        VIRIAL_SUMMARY_END,
        "",
    ]
    return "\n".join(lines)


def update_virial_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """Insert or replace the virial section in the recommendation Markdown file."""
    section = virial_summary_section(rows)
    if path.exists():
        text = path.read_text(encoding="ascii")
    else:
        text = "# QHO Beta=5 Thermodynamic Production Recommendation\n"
    if VIRIAL_SUMMARY_BEGIN in text and VIRIAL_SUMMARY_END in text:
        start = text.index(VIRIAL_SUMMARY_BEGIN)
        end = text.index(VIRIAL_SUMMARY_END, start) + len(VIRIAL_SUMMARY_END)
        text = text[:start].rstrip() + "\n\n" + section + text[end:].lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + section
    path.write_text(text, encoding="ascii")


def y2_zscore_summary_section(rows: list[dict[str, Any]]) -> str:
    """Build the Markdown section summarizing y2 scaling z-scores."""
    coarse_rows = [row for row in rows if y2_zscore_flags(row)["coarse_eta_point"] == 1]
    coarse_breakdown = [
        row
        for row in coarse_rows
        if float(y2_zscore_values(row)["abs_z_cont"]) > 3.0
    ]
    max_abs_z_coarse = (
        max(float(y2_zscore_values(row)["abs_z_cont"]) for row in coarse_rows)
        if coarse_rows
        else math.nan
    )
    breakdown_items = (
        ", ".join(
            f"Nt={row['Nt']} eta={format_float(row['eta'])} "
            f"|z|={format_float(float(y2_zscore_values(row)['abs_z_cont']))}"
            for row in sorted(coarse_breakdown, key=lambda item: float(item["eta"]), reverse=True)
        )
        if coarse_breakdown
        else "none"
    )
    excludes_coarse = all(y2_zscore_flags(row)["selected_y2_fit"] == 0 for row in coarse_rows)
    lines = [
        Y2_ZSCORE_SUMMARY_BEGIN,
        "## Beta=5 y2 Continuum-Scaling Z-Score",
        "",
        f"- definition: z_cont = (y2_mean - exact_y2) / y2_mean_err with exact_y2={format_exact(EXACT_BETA5)}.",
        "- error convention: y2_mean_err is the blocked Monte Carlo error from the beta=5 production summary.",
        f"- max abs_z_cont for coarse_eta_point=1: {format_float(max_abs_z_coarse)}",
        f"- coarse points with abs_z_cont > 3: {breakdown_items}",
        (
            "- final selected y2 fit window excludes the coarse breakdown region: "
            f"{'yes' if excludes_coarse else 'no'}."
        ),
        (
            "- interpretation: the final y2 fit window was selected by chi2_red scans; "
            "the z-score plot is an independent visualization of coarse-lattice breakdown."
        ),
        Y2_ZSCORE_SUMMARY_END,
        "",
    ]
    return "\n".join(lines)


def update_y2_zscore_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """Insert or replace the y2 z-score section in the recommendation file."""
    section = y2_zscore_summary_section(rows)
    if path.exists():
        text = path.read_text(encoding="ascii")
    else:
        text = "# QHO Beta=5 Thermodynamic Production Recommendation\n"
    if Y2_ZSCORE_SUMMARY_BEGIN in text and Y2_ZSCORE_SUMMARY_END in text:
        start = text.index(Y2_ZSCORE_SUMMARY_BEGIN)
        end = text.index(Y2_ZSCORE_SUMMARY_END, start) + len(Y2_ZSCORE_SUMMARY_END)
        text = text[:start].rstrip() + "\n\n" + section + text[end:].lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + section
    path.write_text(text, encoding="ascii")


def print_virial_checks(rows: list[dict[str, Any]]) -> None:
    """Print compact terminal checks for the virial and kinetic estimators."""
    ordered = sorted(rows, key=lambda row: float(row["eta"]))
    energy_diff = np.asarray([float(row["virial"]["V_plus_Kren_diff"]) for row in rows], dtype=float)
    err_ratio = np.asarray(
        [float(row["virial"]["Kren_err"]) / float(row["virial"]["V_err"]) for row in rows],
        dtype=float,
    )
    abs_z = np.asarray([float(row["virial"]["abs_z_virial"]) for row in rows], dtype=float)
    kdiv = np.asarray([float(row["virial"]["Kdiv_mean"]) for row in ordered], dtype=float)
    eta = np.asarray([float(row["eta"]) for row in ordered], dtype=float)
    pass_1sigma = int(np.count_nonzero(abs_z <= 1.0))
    pass_2sigma = int(np.count_nonzero(abs_z <= 2.0))
    print("Virial check uses D = Kren_mean - V_mean; V_mean + Kren_mean is only used for energy closure.")
    print(f"energy closure max |V + Kren - energy_ren| = {np.max(np.abs(energy_diff)):.16e}")
    print(f"Kren_err / V_err range = {np.min(err_ratio):.6g} .. {np.max(err_ratio):.6g}")
    print(f"|Kren - V| <= 1 sigma: {pass_1sigma}/{len(rows)}; <= 2 sigma: {pass_2sigma}/{len(rows)}")
    print(
        "Kdiv trend: "
        f"Kdiv(eta_min={eta[0]:.12g})={kdiv[0]:.12g}, "
        f"Kdiv(eta_max={eta[-1]:.12g})={kdiv[-1]:.12g}, "
        f"all_negative={int(np.all(kdiv < 0.0))}, "
        f"small_eta_more_negative={int(kdiv[0] < kdiv[-1])}"
    )


def write_fits(
    path: Path,
    fits: list[dict[str, Any]],
    scans: list[dict[str, Any]],
    recommended_scans: dict[str, dict[str, Any]],
    scan_a_rows: list[dict[str, Any]],
    scan_b_rows: list[dict[str, Any]],
    final_selected: dict[str, dict[str, Any]],
) -> None:
    """Write eta-cut fits, scan tables, and the final selected fit metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO beta=5 thermodynamic continuum fits\n")
        handle.write("# O(eta) = A + B eta^2; weights use blocked MC errors only\n")
        handle.write("# --- eta_cut fits ---\n")
        handle.write(
            "# observable eta_cut n_points A A_error B B_error chi2 dof chi2_red "
            "exact A_minus_exact weighted\n"
        )
        for fit in fits:
            handle.write(
                f"{fit['observable']} {format_float(fit['eta_cut'])} {fit['n_points']} "
                f"{format_float(fit['A'])} {format_float(fit['A_error'])} "
                f"{format_float(fit['B'])} {format_float(fit['B_error'])} "
                f"{format_float(fit['chi2'])} {fit['dof']} {format_float(fit['chi2_red'])} "
                f"{format_exact(fit['exact'])} {format_float(fit['A_minus_exact'])} {fit['weighted']}\n"
            )
        handle.write("# --- contiguous eta-window scan ---\n")
        handle.write(
            "# scan observable eta_min eta_max n_points A A_error B B_error chi2 dof "
            "chi2_red exact A_minus_exact weighted target_distance\n"
        )
        for scan in scans:
            handle.write(
                f"scan {scan['observable']} {format_float(scan['eta_min'])} "
                f"{format_float(scan['eta_max'])} {scan['n_points']} "
                f"{format_float(scan['A'])} {format_float(scan['A_error'])} "
                f"{format_float(scan['B'])} {format_float(scan['B_error'])} "
                f"{format_float(scan['chi2'])} {scan['dof']} {format_float(scan['chi2_red'])} "
                f"{format_exact(scan['exact'])} {format_float(scan['A_minus_exact'])} "
                f"{scan['weighted']} {format_float(scan['target_distance'])}\n"
            )
        handle.write("# --- recommended contiguous eta-window scan fits ---\n")
        handle.write(
            "# recommended_scan observable eta_min eta_max n_points A A_error B B_error "
            "chi2 dof chi2_red exact A_minus_exact weighted target_distance\n"
        )
        for observable in FIT_OBSERVABLES:
            scan = recommended_scans.get(observable)
            if scan is None:
                continue
            handle.write(
                f"# recommended_scan {scan['observable']} {format_float(scan['eta_min'])} "
                f"{format_float(scan['eta_max'])} {scan['n_points']} "
                f"{format_float(scan['A'])} {format_float(scan['A_error'])} "
                f"{format_float(scan['B'])} {format_float(scan['B_error'])} "
                f"{format_float(scan['chi2'])} {scan['dof']} {format_float(scan['chi2_red'])} "
                f"{format_exact(scan['exact'])} {format_float(scan['A_minus_exact'])} "
                f"{scan['weighted']} {format_float(scan['target_distance'])}\n"
            )
        handle.write(
            "# --- targeted scan A: fixed eta_max=0.2, progressively remove largest Nt / smallest eta ---\n"
        )
        handle.write(
            "# scanA observable n_points Nt_min Nt_max eta_min eta_max removed_largest_Nt_count "
            "A A_error B B_error chi2 dof chi2_red exact A_minus_exact within_1sigma "
            "within_2sigma weighted\n"
        )
        for scan in scan_a_rows:
            handle.write(
                f"scanA {scan['observable']} {scan['n_points']} {scan['Nt_min']} {scan['Nt_max']} "
                f"{format_float(scan['eta_min'])} {format_float(scan['eta_max'])} "
                f"{scan['removed_largest_Nt_count']} "
                f"{format_float(scan['A'])} {format_float(scan['A_error'])} "
                f"{format_float(scan['B'])} {format_float(scan['B_error'])} "
                f"{format_float(scan['chi2'])} {scan['dof']} {format_float(scan['chi2_red'])} "
                f"{format_exact(scan['exact'])} {format_float(scan['A_minus_exact'])} "
                f"{scan['within_1sigma']} {scan['within_2sigma']} {scan['weighted']}\n"
            )
        handle.write(
            "# --- targeted scan B: fixed eta_min at finest point, progressively move eta_max ---\n"
        )
        handle.write(
            "# scanB observable n_points Nt_min Nt_max eta_min eta_max A A_error B B_error "
            "chi2 dof chi2_red exact A_minus_exact within_1sigma within_2sigma weighted\n"
        )
        for scan in scan_b_rows:
            handle.write(
                f"scanB {scan['observable']} {scan['n_points']} {scan['Nt_min']} {scan['Nt_max']} "
                f"{format_float(scan['eta_min'])} {format_float(scan['eta_max'])} "
                f"{format_float(scan['A'])} {format_float(scan['A_error'])} "
                f"{format_float(scan['B'])} {format_float(scan['B_error'])} "
                f"{format_float(scan['chi2'])} {scan['dof']} {format_float(scan['chi2_red'])} "
                f"{format_exact(scan['exact'])} {format_float(scan['A_minus_exact'])} "
                f"{scan['within_1sigma']} {scan['within_2sigma']} {scan['weighted']}\n"
            )
        handle.write("# --- final selected continuum fits for plots and recommendation ---\n")
        handle.write(
            "# final_selected observable scan Nt_min Nt_max eta_min eta_max n_points "
            "A A_error B B_error chi2_red exact A_minus_exact\n"
        )
        for observable in FIT_OBSERVABLES:
            scan = final_selected.get(observable)
            if scan is None:
                continue
            handle.write(
                f"# final_selected {scan['observable']} scanA {scan['Nt_min']} {scan['Nt_max']} "
                f"{format_float(scan['eta_min'])} {format_float(scan['eta_max'])} "
                f"{scan['n_points']} {format_float(scan['A'])} {format_float(scan['A_error'])} "
                f"{format_float(scan['B'])} {format_float(scan['B_error'])} "
                f"{format_float(scan['chi2_red'])} {format_exact(scan['exact'])} "
                f"{format_float(scan['A_minus_exact'])}\n"
            )


def agreement_text(fit: dict[str, Any]) -> str:
    """Return a short text flag for one-sigma agreement with the exact value."""
    diff = float(fit["A_minus_exact"])
    err = float(fit["A_error"])
    if not math.isfinite(diff) or not math.isfinite(err) or err <= 0.0:
        return "unresolved"
    return "yes" if abs(diff) <= err else "no"


def sigma_agreement_text(fit: dict[str, Any], nsigma: float) -> str:
    """Return a short text flag for nsigma agreement with the exact value."""
    diff = float(fit["A_minus_exact"])
    err = float(fit["A_error"])
    if not math.isfinite(diff) or not math.isfinite(err) or err <= 0.0:
        return "unresolved"
    return "yes" if abs(diff) <= nsigma * err else "no"


def choose_preferred_fit(fits: list[dict[str, Any]]) -> float:
    """Choose the eta_cut fit summarized in the recommendation file."""
    preferred = [
        fit for fit in fits if fit["weighted"] == 1 and math.isclose(float(fit["eta_cut"]), 0.2)
    ]
    if preferred:
        return 0.2
    valid = [fit for fit in fits if fit["weighted"] == 1 and math.isfinite(float(fit["chi2_red"]))]
    if not valid:
        return math.nan
    by_cut: dict[float, list[float]] = {}
    for fit in valid:
        by_cut.setdefault(float(fit["eta_cut"]), []).append(abs(float(fit["chi2_red"]) - 1.0))
    return min(by_cut, key=lambda eta_cut: max(by_cut[eta_cut]))


def targeted_fit_line(label: str, observable: str, candidate: dict[str, Any] | None) -> str:
    """Format one targeted-scan candidate as a Markdown bullet."""
    if candidate is None:
        return f"- {label} candidate for {observable} with chi2_red < 1: unresolved."
    suffix = "" if int(candidate["within_2sigma"]) == 1 else " (not within 2 sigma)."
    return (
        f"- {label} candidate for {observable}: "
        f"Nt={candidate['Nt_min']}..{candidate['Nt_max']}, "
        f"eta={format_float(candidate['eta_min'])}..{format_float(candidate['eta_max'])}, "
        f"n_points={candidate['n_points']}, "
        f"chi2_red={format_float(candidate['chi2_red'])}, "
        f"A={format_float(candidate['A'])}, "
        f"A_error={format_float(candidate['A_error'])}, "
        f"A-exact={format_float(candidate['A_minus_exact'])}, "
        f"within_1sigma={candidate['within_1sigma']}, "
        f"within_2sigma={candidate['within_2sigma']}{suffix}"
    )


def final_selected_fit_line(observable: str, candidate: dict[str, Any] | None) -> str:
    """Format the final displayed continuum fit as a Markdown bullet."""
    if candidate is None:
        return f"- final selected fit for {observable}: unresolved."
    return (
        f"- final selected fit for {observable}: Scan A, "
        f"Nt={candidate['Nt_min']}..{candidate['Nt_max']}, "
        f"eta={format_float(candidate['eta_min'])}..{format_float(candidate['eta_max'])}, "
        f"n_points={candidate['n_points']}, "
        f"chi2_red={format_float(candidate['chi2_red'])}, "
        f"A={format_float(candidate['A'])}, "
        f"A_error={format_float(candidate['A_error'])}."
    )


def choose_overall_targeted_candidate(
    candidates: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Choose the best candidate among Scan A and Scan B outputs."""
    valid = [candidate for candidate in candidates if candidate is not None]
    if not valid:
        return None
    return min(
        valid,
        key=lambda row: (
            abs(float(row["chi2_red"]) - SCAN_TARGET_CHI2_RED),
            -int(row["n_points"]),
            abs(float(row["A_minus_exact"])),
        ),
    )


def common_window_text(
    y2_candidate: dict[str, Any] | None,
    energy_candidate: dict[str, Any] | None,
) -> str:
    """Describe whether y2 and energy prefer overlapping continuum windows."""
    if y2_candidate is None or energy_candidate is None:
        return (
            "- combined targeted suggestion: unresolved because at least one observable "
            "has no chi2_red < 1 targeted candidate."
        )
    nt_min = max(int(y2_candidate["Nt_min"]), int(energy_candidate["Nt_min"]))
    nt_max = min(int(y2_candidate["Nt_max"]), int(energy_candidate["Nt_max"]))
    eta_min = max(float(y2_candidate["eta_min"]), float(energy_candidate["eta_min"]))
    eta_max = min(float(y2_candidate["eta_max"]), float(energy_candidate["eta_max"]))
    if nt_min <= nt_max and eta_min <= eta_max:
        return (
            "- combined targeted suggestion: the preferred observable windows overlap; "
            f"a common candidate range is Nt={nt_min}..{nt_max} "
            f"(eta={format_float(eta_min)}..{format_float(eta_max)}). "
            "Final Nt selection should still be made manually from the scan tables."
        )
    return (
        "- combined targeted suggestion: y2_mean and energy_ren prefer different "
        "targeted windows, so no common range is forced; final Nt selection should "
        "be made manually."
    )


def write_recommendation(
    path: Path,
    rows: list[dict[str, Any]],
    fits: list[dict[str, Any]],
    recommended_scans: dict[str, dict[str, Any]],
    scan_a_best: dict[str, dict[str, Any]],
    scan_b_best: dict[str, dict[str, Any]],
    final_selected: dict[str, dict[str, Any]],
) -> None:
    """Write a human-readable Markdown summary of the beta=5 analysis choices."""
    beta = float(rows[0]["beta"])
    nt_grid = [int(row["Nt"]) for row in rows]
    eta_values = [float(row["eta"]) for row in rows]
    n_blocks = sorted({int(row["n_blocks"]) for row in rows})
    therm_values = sorted({int(row["n_therm"]) for row in rows})
    sweep_values = sorted({int(row["n_sweeps"]) for row in rows})
    stride_values = sorted({int(row["meas_stride"]) for row in rows})
    block_saved_values = sorted({int(row["block_size_saved"]) for row in rows})
    block_sweeps_values = sorted({int(row["block_size_sweeps"]) for row in rows})
    shortened = any(
        int(row["n_therm"]) != DEFAULT_THERM
        or int(row["n_sweeps"]) != DEFAULT_SWEEPS
        or int(row["meas_stride"]) != DEFAULT_STRIDE
        or int(row["block_size_saved"]) != DEFAULT_BLOCK_SAVED
        for row in rows
    )

    preferred_cut = choose_preferred_fit(fits)
    preferred_fits = {
        fit["observable"]: fit
        for fit in fits
        if math.isfinite(preferred_cut)
        and fit["weighted"] == 1
        and math.isclose(float(fit["eta_cut"]), preferred_cut)
    }
    y2_agree = agreement_text(preferred_fits["y2_mean"]) if "y2_mean" in preferred_fits else "unresolved"
    energy_agree = (
        agreement_text(preferred_fits["energy_ren"]) if "energy_ren" in preferred_fits else "unresolved"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# QHO Beta=5 Thermodynamic Production Recommendation\n\n")
        handle.write(f"- beta: {format_float(beta)}\n")
        handle.write("- Nt grid: " + " ".join(str(nt) for nt in nt_grid) + "\n")
        handle.write(
            f"- eta range: {format_float(min(eta_values))} to {format_float(max(eta_values))}\n"
        )
        handle.write(
            f"- MC parameters: n_therm={therm_values}, n_sweeps={sweep_values}, "
            f"meas_stride={stride_values}, update=hb-over, n_over=5, init=zero\n"
        )
        handle.write(
            f"- block_size_saved={block_saved_values}, block_size_sweeps={block_sweeps_values}, "
            f"n_blocks={n_blocks}\n"
        )
        handle.write(f"- exact continuum value at beta=5: {format_exact(EXACT_BETA5)}\n")
        handle.write("- final displayed continuum fits: selected targeted Scan A windows.\n")
        for observable in FIT_OBSERVABLES:
            handle.write(final_selected_fit_line(observable, final_selected.get(observable)) + "\n")
        handle.write(f"- eta_cut fit window candidate: eta_cut={format_float(preferred_cut)}\n")
        handle.write(f"- y2_mean agrees with exact within errors in that window: {y2_agree}\n")
        handle.write(
            f"- energy_ren agrees with exact within errors in that window: {energy_agree}\n"
        )
        handle.write(
            "- contiguous-window scan: systematic scan of all eta windows with "
            f"eta_max <= {format_float(SCAN_MAX_ETA)} and n_points >= {SCAN_MIN_POINTS}.\n"
        )
        handle.write("- scan errors: blocked MC errors; errors are not rescaled.\n")
        for observable in FIT_OBSERVABLES:
            scan = recommended_scans.get(observable)
            if scan is None:
                handle.write(f"- recommended scan window for {observable}: unresolved.\n")
                continue
            handle.write(
                f"- recommended scan window for {observable}: "
                f"eta_min={format_float(scan['eta_min'])}, "
                f"eta_max={format_float(scan['eta_max'])}, "
                f"n_points={scan['n_points']}, "
                f"chi2_red={format_float(scan['chi2_red'])}, "
                f"A={format_float(scan['A'])}, "
                f"A_error={format_float(scan['A_error'])}, "
                f"A-exact={format_float(scan['A_minus_exact'])}, "
                f"within_1sigma={sigma_agreement_text(scan, 1.0)}, "
                f"within_2sigma={sigma_agreement_text(scan, 2.0)}.\n"
            )
        handle.write(
            "- targeted scan A: eta_max is fixed to 0.2 while the finest points "
            "are removed one by one.\n"
        )
        handle.write(
            "- targeted scan B: eta_min is fixed to the finest point while coarser "
            "points are added one by one.\n"
        )
        handle.write(
            "- targeted scan selection: require chi2_red < 1, prefer chi2_red closest "
            "to 0.9, then larger n_points, then smaller |A-exact|; within_2sigma "
            "is required when available.\n"
        )
        for observable in FIT_OBSERVABLES:
            handle.write(targeted_fit_line("best Scan A", observable, scan_a_best.get(observable)) + "\n")
        for observable in FIT_OBSERVABLES:
            handle.write(targeted_fit_line("best Scan B", observable, scan_b_best.get(observable)) + "\n")
        y2_overall = choose_overall_targeted_candidate(
            [scan_a_best.get("y2_mean"), scan_b_best.get("y2_mean")]
        )
        energy_overall = choose_overall_targeted_candidate(
            [scan_a_best.get("energy_ren"), scan_b_best.get("energy_ren")]
        )
        handle.write(common_window_text(y2_overall, energy_overall) + "\n")
        if shortened:
            handle.write(
                "- warning: this is a shortened check run, not final production data.\n"
            )
        else:
            handle.write("- run mode: full configured beta=5 thermodynamic production.\n")


def main() -> int:
    """Run the beta=5 thermodynamic analysis pipeline."""
    args = parse_args()
    if not args.manifest.is_file():
        print(f"error: manifest does not exist: {args.manifest}", file=sys.stderr)
        return 1
    try:
        manifest_rows = parse_manifest(args.manifest)
        rows = [analyze_chain(row, args.manifest) for row in manifest_rows]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rows.sort(key=lambda row: int(row["Nt"]))

    # This script is intentionally specialized to the beta=5 final dataset.
    if any(abs(float(row["beta"]) - BETA_FINAL) > 1.0e-12 for row in rows):
        print("error: this analysis script expects beta=5 only", file=sys.stderr)
        return 1

    output_dir = args.manifest.parent

    # These checks are useful both in full analysis mode and in --virial-only mode.
    write_y2_zscore(output_dir / "qho_thermo_beta5_y2_zscore.dat", rows)
    write_virial(output_dir / "qho_thermo_beta5_virial.dat", rows)
    write_virial_check(output_dir / "qho_thermo_beta5_virial_check.dat", rows)
    write_kdiv(output_dir / "qho_thermo_beta5_kdiv.dat", rows)
    if args.virial_only:
        if args.update_recommendation:
            update_virial_summary(output_dir / "qho_thermo_beta5_recommendation.md", rows)
            print(f"Updated {output_dir / 'qho_thermo_beta5_recommendation.md'}")
        print(f"Wrote {output_dir / 'qho_thermo_beta5_virial.dat'}")
        print(f"Wrote {output_dir / 'qho_thermo_beta5_virial_check.dat'}")
        print(f"Wrote {output_dir / 'qho_thermo_beta5_kdiv.dat'}")
        print(f"Wrote {output_dir / 'qho_thermo_beta5_y2_zscore.dat'}")
        print_virial_checks(rows)
        return 0

    fits: list[dict[str, Any]] = []

    # Standard eta-cut fits: simple reference windows for quick comparison.
    for observable in FIT_OBSERVABLES:
        for eta_cut in ETA_CUTS:
            fits.append(weighted_fit_eta2(rows, observable, eta_cut))
    scans: list[dict[str, Any]] = []

    # Exhaustive contiguous-window scan inside the accepted eta range.
    for observable in FIT_OBSERVABLES:
        scans.extend(scan_contiguous_windows(rows, observable))
    recommended_scans = {
        observable: scan
        for observable in FIT_OBSERVABLES
        if (scan := choose_recommended_scan(scans, observable)) is not None
    }
    scan_a_rows: list[dict[str, Any]] = []
    scan_b_rows: list[dict[str, Any]] = []

    # Targeted scans reproduce the windows used in the final displayed plots.
    for observable in FIT_OBSERVABLES:
        observable_scan_a, observable_scan_b = scan_targeted_windows(rows, observable)
        scan_a_rows.extend(observable_scan_a)
        scan_b_rows.extend(observable_scan_b)
    scan_a_best = {
        observable: scan
        for observable in FIT_OBSERVABLES
        if (scan := choose_targeted_candidate(scan_a_rows, observable)) is not None
    }
    scan_b_best = {
        observable: scan
        for observable in FIT_OBSERVABLES
        if (scan := choose_targeted_candidate(scan_b_rows, observable)) is not None
    }
    final_selected = scan_a_best

    # The plotting scripts and final collector read these processed tables.
    write_points(output_dir / "qho_thermo_beta5_points.dat", rows)
    write_summary(output_dir / "qho_thermo_beta5_summary.dat", rows)
    write_fits(
        output_dir / "qho_thermo_beta5_fits.dat",
        fits,
        scans,
        recommended_scans,
        scan_a_rows,
        scan_b_rows,
        final_selected,
    )
    write_recommendation(
        output_dir / "qho_thermo_beta5_recommendation.md",
        rows,
        fits,
        recommended_scans,
        scan_a_best,
        scan_b_best,
        final_selected,
    )
    update_virial_summary(output_dir / "qho_thermo_beta5_recommendation.md", rows)
    update_y2_zscore_summary(output_dir / "qho_thermo_beta5_recommendation.md", rows)

    print(f"Wrote {output_dir / 'qho_thermo_beta5_points.dat'}")
    print(f"Wrote {output_dir / 'qho_thermo_beta5_summary.dat'}")
    print(f"Wrote {output_dir / 'qho_thermo_beta5_fits.dat'}")
    print(f"Wrote {output_dir / 'qho_thermo_beta5_recommendation.md'}")
    print(f"Wrote {output_dir / 'qho_thermo_beta5_virial.dat'}")
    print(f"Wrote {output_dir / 'qho_thermo_beta5_virial_check.dat'}")
    print(f"Wrote {output_dir / 'qho_thermo_beta5_kdiv.dat'}")
    print(f"Wrote {output_dir / 'qho_thermo_beta5_y2_zscore.dat'}")
    print_virial_checks(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
