#!/usr/bin/env python3
# Analyze the temperature dependence of the bare energy estimator at fixed
# lattice spacing, using blocked errors and a zero-temperature subtraction to
# isolate the Bose occupation contribution.
"""
Analyze the fixed-eta temperature dependence of the bare QHO energy estimator.

The production script `scripts/run/run_temperature_bare_scan.sh` runs the same
Euclidean lattice spacing `eta` at several inverse temperatures beta. Since
beta = Nt * eta, changing beta at fixed eta means changing the number of time
slices rather than changing the lattice cutoff.

For each raw measurement file this script builds block estimates of

    U_b = 0.5 * <y^2> - <(Delta y)^2> / (2 eta^2),

where the averages are path averages stored by the C executable. This is the
non-renormalized estimator used in the temperature scan. The analysis then
subtracts the largest-beta point,

    Delta U_b(beta) = U_b(beta) - U_b(beta0),

so that the dominant eta-dependent offset cancels. The remaining temperature
dependence is compared with

    n_B(beta) - n_B(beta0),       n_B(beta) = 1 / (exp(beta) - 1),

through a one-parameter weighted fit.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    # Seaborn only improves plot styling; the analysis itself does not depend on it.
    sns = None


# Column order written by the ASCII measurement output of `bin/qho_pimc`.
# Only `y2_mean` and `dy2_mean` enter U_b, but keeping the full schema here
# makes malformed input files fail early.
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
# Manifest schema written by the temperature-scan run script.
# The analysis keeps timing fields so the generated manifest remains a useful
# record of the production run.
MANIFEST_COLUMNS = (
    "beta",
    "x",
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
# Default output locations. Each writer also emits an eta-tagged companion file,
# so scans at different fixed lattice spacings can coexist.
SUMMARY_PATH = Path("data/processed/production/qho_temperature_bare_summary.dat")
DELTA_PATH = Path("data/processed/production/qho_temperature_delta_summary.dat")
RECOMMENDATION_PATH = Path("data/processed/production/qho_temperature_bare_recommendation.md")
MANIFEST_PATH = Path("data/processed/production/qho_temperature_bare_manifest.dat")
RELATION_TABLE_PATH = Path("data/processed/production/qho_temperature_relation_table.dat")
PLOT_DIR = Path("plots/thermodynamics")


def parse_args() -> argparse.Namespace:
    """Parse the manifest path or an explicit list of raw files."""
    parser = argparse.ArgumentParser(
        description="Analyze fixed-eta temperature scan with U_b = V + Kdiv."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/production/qho_temperature_bare_manifest.dat"),
        help="Manifest written by scripts/run/run_temperature_bare_scan.sh",
    )
    parser.add_argument(
        "--raw-files",
        type=Path,
        nargs="*",
        help="Existing raw files to reprocess when a matching manifest is not available.",
    )
    return parser.parse_args()


def format_float(value: float) -> str:
    """Format floating-point values consistently in text tables."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.12g}"


def eta_tag(eta: float) -> str:
    """Return a filename-safe tag for a fixed lattice spacing."""
    text = f"{eta:.12g}".replace("-", "m").replace(".", "p")
    return f"eta{text}"


def tagged_path(path: Path, eta: float) -> Path:
    """Attach the eta tag before the file suffix."""
    return path.with_name(f"{path.stem}_{eta_tag(eta)}{path.suffix}")


def output_paths(path: Path, eta: float) -> tuple[Path, Path]:
    """Return the canonical output path and its eta-tagged companion."""
    return path, tagged_path(path, eta)


def parse_raw_metadata(path: Path) -> dict[str, str]:
    """Read key-value metadata from the commented header of a raw data file."""
    metadata: dict[str, str] = {}
    with path.open("r", encoding="ascii") as handle:
        # Metadata are stored as comment lines before the numeric table.
        for line in handle:
            if not line.startswith("#"):
                break
            fields = line[1:].strip().split(maxsplit=1)
            if len(fields) == 2:
                metadata[fields[0]] = fields[1]
    return metadata


def rows_from_raw_files(paths: list[Path]) -> list[dict[str, Any]]:
    """
    Build manifest-like rows directly from raw files.

    This fallback is useful when reprocessing completed runs whose manifest is
    unavailable. Required simulation parameters are recovered from the raw-file
    metadata written by the executable.
    """
    rows: list[dict[str, Any]] = []
    for path in paths:
        metadata = parse_raw_metadata(path)
        try:
            beta = float(metadata["beta"])
            eta = float(metadata["eta"])
            nt = int(metadata.get("Nt", metadata["nt"]))
            n_therm = int(metadata.get("n_therm", metadata["therm"]))
            n_sweeps = int(metadata.get("n_sweeps", metadata["sweeps"]))
            meas_stride = int(metadata.get("meas_stride", metadata["stride"]))
            seed = int(metadata["seed"])
            stream = int(metadata["stream"])
            update = metadata["update"]
            init = metadata["init"]
            n_over = int(metadata["n_over"])
            block_size_saved = int(metadata["block_size_saved"])
        except KeyError as exc:
            raise ValueError(f"{path}: missing required metadata field {exc}") from exc
        rows.append({
            "beta": beta,
            "x": 1.0 / beta,
            "eta": eta,
            "Nt": nt,
            "update": update,
            "init": init,
            "seed": seed,
            "stream": stream,
            "n_therm": n_therm,
            "n_sweeps": n_sweeps,
            "meas_stride": meas_stride,
            "n_over": n_over,
            "block_size_saved": block_size_saved,
            "raw_file": str(path),
            "runtime_seconds": math.nan,
            "seconds_per_sweep": math.nan,
            "seconds_per_site_sweep": math.nan,
        })
    if not rows:
        raise ValueError("no raw files supplied")
    return sorted(rows, key=lambda item: float(item["beta"]))


def parse_manifest(path: Path) -> list[dict[str, Any]]:
    """
    Read the production manifest and convert numerical columns to typed values.

    The consistency check beta = Nt * eta is important here because this scan is
    defined by holding eta fixed while beta changes.
    """
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
            row: dict[str, Any] = dict(zip(MANIFEST_COLUMNS, fields))
            row["beta"] = float(row["beta"])
            row["x"] = float(row["x"])
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
            # At fixed eta the intended inverse temperature is reconstructed as Nt*eta.
            actual_beta = row["Nt"] * row["eta"]
            if abs(actual_beta - row["beta"]) > 1.0e-10 * max(1.0, abs(row["beta"])):
                raise ValueError(
                    f"{path}:{line_number}: beta={row['beta']} inconsistent with "
                    f"Nt*eta={actual_beta}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no manifest rows found")
    return sorted(rows, key=lambda item: float(item["beta"]))


def write_manifest(rows: list[dict[str, Any]], eta: float) -> None:
    """Write the resolved manifest used by this analysis."""
    for path in output_paths(MANIFEST_PATH, eta):
        with path.open("w", encoding="ascii") as handle:
            handle.write("# QHO temperature scan with non-renormalized internal-energy estimator\n")
            handle.write("# beta x eta Nt update init seed stream n_therm n_sweeps meas_stride n_over block_size_saved raw_file runtime_seconds seconds_per_sweep seconds_per_site_sweep\n")
            for row in rows:
                handle.write(
                    f"{format_float(float(row['beta']))} {format_float(float(row['x']))} "
                    f"{format_float(float(row['eta']))} {int(row['Nt'])} {row['update']} {row['init']} "
                    f"{int(row['seed'])} {int(row['stream'])} {int(row['n_therm'])} "
                    f"{int(row['n_sweeps'])} {int(row['meas_stride'])} {int(row['n_over'])} "
                    f"{int(row['block_size_saved'])} {row['raw_file']} "
                    f"{format_float(float(row['runtime_seconds']))} "
                    f"{format_float(float(row['seconds_per_sweep']))} "
                    f"{format_float(float(row['seconds_per_site_sweep']))}\n"
                )
        print(f"Wrote {path}")


def project_root_from_manifest(path: Path) -> Path:
    """
    Infer the repository root from a standard processed-production manifest path.

    Relative raw-file paths in manifests are interpreted with respect to this
    root. If the manifest is elsewhere, the current working directory is used.
    """
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
    """Resolve absolute, current-directory, or repository-relative raw paths."""
    raw_path = Path(raw_file)
    if raw_path.is_absolute():
        return raw_path
    if raw_path.is_file():
        return raw_path
    return project_root_from_manifest(manifest) / raw_path


def load_measurements(path: Path) -> np.ndarray:
    """Load the numeric measurement table and keep the expected QHO columns."""
    data = np.loadtxt(path, comments="#")
    if data.size == 0:
        raise ValueError(f"{path}: no measurement rows found")
    data = np.atleast_2d(data)
    if data.shape[1] < len(COLUMNS):
        raise ValueError(f"{path}: expected at least {len(COLUMNS)} columns, found {data.shape[1]}")
    return data[:, : len(COLUMNS)]


def mean_error_from_blocks(block_values: np.ndarray) -> tuple[float, float, int]:
    """Return the block mean, standard error, and number of blocks."""
    n_blocks = int(block_values.size)
    if n_blocks <= 0:
        return math.nan, math.nan, 0
    mean = float(block_values.mean())
    if n_blocks >= 2:
        error = float(block_values.std(ddof=1) / math.sqrt(float(n_blocks)))
    else:
        error = math.nan
    return mean, error, n_blocks


def bare_energy_blocks(data: np.ndarray, eta: float, block_size_saved: int) -> np.ndarray:
    """
    Compute one U_b estimate per block of saved configurations.

    Each block first averages `y2_mean` and `dy2_mean`, then combines those
    block means into U_b. This keeps the nonlinear estimator aligned with the
    blocking procedure.
    """
    n_measurements = int(data.shape[0])
    # If a short run contains fewer saved measurements than requested, keep one
    # block rather than discarding the entire point.
    actual_block_size = min(block_size_saved, n_measurements)
    n_blocks = n_measurements // actual_block_size
    if n_blocks <= 0:
        return np.asarray([], dtype=float)
    # Drop only the incomplete tail so all retained blocks have equal weight.
    trimmed = data[: n_blocks * actual_block_size, :]
    blocks = trimmed.reshape(n_blocks, actual_block_size, data.shape[1])
    y2_blocks = blocks[:, :, COLUMNS.index("y2_mean")].mean(axis=1)
    dy2_blocks = blocks[:, :, COLUMNS.index("dy2_mean")].mean(axis=1)
    return 0.5 * y2_blocks - dy2_blocks / (2.0 * eta * eta)


def n_bose(beta: float) -> float:
    """Return the Bose occupation number for oscillator gap Delta E = 1."""
    return 1.0 / math.expm1(beta)


def analyze_chain(row: dict[str, Any], manifest: Path) -> dict[str, Any]:
    """Analyze one beta point and append derived block statistics to the row."""
    raw_path = resolve_raw_path(str(row["raw_file"]), manifest)
    data = load_measurements(raw_path)
    block_values = bare_energy_blocks(data, float(row["eta"]), int(row["block_size_saved"]))
    u_mean, u_err, n_blocks = mean_error_from_blocks(block_values)
    return {
        **row,
        "x": 1.0 / float(row["beta"]),
        "raw_file": str(row["raw_file"]),
        "resolved_raw_file": raw_path,
        "U_b": u_mean,
        "U_b_err": u_err,
        "n_blocks": n_blocks,
        "block_size_sweeps": int(row["block_size_saved"]) * int(row["meas_stride"]),
        "n_measurements": int(data.shape[0]),
    }


def build_delta_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Subtract the largest-beta point from all U_b estimates.

    The subtraction cancels the beta-independent bare offset at fixed eta and
    exposes the expected thermal dependence.
    """
    # The coldest available point is used as beta0, where thermal occupation is smallest.
    reference = max(rows, key=lambda item: float(item["beta"]))
    beta0 = float(reference["beta"])
    nb0 = n_bose(beta0)
    delta_rows: list[dict[str, Any]] = []
    for row in rows:
        delta = float(row["U_b"]) - float(reference["U_b"])
        if row is reference:
            # Delta U_b(beta0) is exactly zero because the point is subtracted from itself.
            delta_err = 0.0
        else:
            delta_err = math.sqrt(float(row["U_b_err"]) ** 2 + float(reference["U_b_err"]) ** 2)
        nb = n_bose(float(row["beta"]))
        delta_rows.append({
            **row,
            "beta0": beta0,
            "x0": 1.0 / beta0,
            "Nt0": int(reference["Nt"]),
            "U_b0": float(reference["U_b"]),
            "U_b0_err": float(reference["U_b_err"]),
            "Delta_U_b": delta,
            "Delta_U_b_err": delta_err,
            "nB": nb,
            "nB_minus_nB0": nb - nb0,
            "is_reference": row is reference,
        })
    return delta_rows, reference


def fit_amplitude(delta_rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """
    Fit Delta U_b = A * [n_B(beta) - n_B(beta0)] with weighted least squares.

    The reference point is excluded because its delta is exactly zero by
    construction and has zero assigned uncertainty.
    """
    selected = [
        row for row in delta_rows
        if not bool(row["is_reference"])
        and math.isfinite(float(row["Delta_U_b_err"]))
        and float(row["Delta_U_b_err"]) > 0.0
        and math.isfinite(float(row["nB_minus_nB0"]))
    ]
    n_points = len(selected)
    dof = n_points - 1
    if n_points < 2:
        return {"A": math.nan, "A_err": math.nan, "chi2": math.nan, "dof": dof, "chi2_red": math.nan}
    # The only fitted parameter is the overall amplitude multiplying the exact shape.
    f = np.asarray([float(row["nB_minus_nB0"]) for row in selected], dtype=float)
    y = np.asarray([float(row["Delta_U_b"]) for row in selected], dtype=float)
    sigma = np.asarray([float(row["Delta_U_b_err"]) for row in selected], dtype=float)
    weights = 1.0 / (sigma * sigma)
    denom = float(np.sum(weights * f * f))
    if denom <= 0.0:
        return {"A": math.nan, "A_err": math.nan, "chi2": math.nan, "dof": dof, "chi2_red": math.nan}
    amplitude = float(np.sum(weights * f * y) / denom)
    amplitude_err = math.sqrt(1.0 / denom)
    residuals = y - amplitude * f
    chi2 = float(np.sum(weights * residuals * residuals))
    chi2_red = chi2 / dof if dof > 0 else math.nan
    return {"A": amplitude, "A_err": amplitude_err, "chi2": chi2, "dof": dof, "chi2_red": chi2_red}


def configure_plot_style() -> None:
    """Set the plotting defaults used by the generated report figures."""
    if sns is not None:
        sns.set_theme(style="ticks", context="paper", font_scale=1.6)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "CMU Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
        "font.size": 20,
        "axes.labelsize": 24,
        "axes.titlesize": 20,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "black",
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def apply_report_style(ax: plt.Axes) -> None:
    """Apply common grid, tick, and spine settings to one axis."""
    ax.grid(True, which="major", axis="both", color="0.70", alpha=0.55, linestyle=":", linewidth=0.8)
    ax.tick_params(direction="in", which="major", top=True, right=True, width=1.0, length=4.8)
    ax.tick_params(direction="in", which="minor", top=True, right=True, width=0.8, length=3.2)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def report_legend(ax: plt.Axes, **kwargs: Any):
    """Create a compact white-background legend suitable for report figures."""
    kwargs.setdefault("fontsize", 18)
    kwargs.setdefault("borderpad", 0.30)
    kwargs.setdefault("labelspacing", 0.25)
    kwargs.setdefault("handlelength", 1.8)
    kwargs.setdefault("handletextpad", 0.55)
    kwargs.setdefault("borderaxespad", 0.35)
    kwargs.setdefault("markerscale", 0.85)
    legend = ax.legend(frameon=True, facecolor="white", framealpha=1.0, edgecolor="none", **kwargs)
    legend.get_frame().set_linewidth(0.0)
    return legend


def adapt_two_column_figure(fig: plt.Figure) -> None:
    """Keep the figure legible when included as a two-column report panel."""
    fig.set_size_inches(6.2, 4.8, forward=True)
    for ax in fig.axes:
        for line in ax.lines:
            line.set_linewidth(max(line.get_linewidth(), 1.5))
            if line.get_marker() not in (None, "None", "", " "):
                line.set_markersize(max(line.get_markersize(), 5.5))
        for collection in ax.collections:
            widths = collection.get_linewidths()
            if len(widths):
                collection.set_linewidth(max(float(np.max(widths)), 1.2))
        ax.tick_params(which="major", length=6.0, width=1.2)
        ax.tick_params(which="minor", length=3.8, width=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        for gridline in ax.get_xgridlines() + ax.get_ygridlines():
            gridline.set_color("0.75")
            gridline.set_alpha(0.40)
            gridline.set_linewidth(0.8)
        legend = ax.get_legend()
        if legend is not None:
            for handle in legend.legend_handles:
                if hasattr(handle, "set_linewidth"):
                    handle.set_linewidth(2.0)
                if hasattr(handle, "set_markersize"):
                    handle.set_markersize(5.5)


def save_figure(fig: plt.Figure, basenames: tuple[str, ...]) -> None:
    """Save each figure as PNG."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for basename in basenames:
        output = PLOT_DIR / f"{basename}.png"
        fig.savefig(output, bbox_inches="tight", dpi=300)
        print(f"Wrote {output}")
    plt.close(fig)


def plot_ub(rows: list[dict[str, Any]], reference: dict[str, Any], eta: float) -> None:
    """Plot the bare estimator U_b against x = 1 / beta."""
    x = np.asarray([float(row["x"]) for row in rows], dtype=float)
    y = np.asarray([float(row["U_b"]) for row in rows], dtype=float)
    err = np.asarray([float(row["U_b_err"]) for row in rows], dtype=float)
    # Plot the reference separately so it is visually identified as beta0.
    non_ref = np.asarray([row is not reference for row in rows], dtype=bool)
    order = np.argsort(x[non_ref])
    ref_x = float(reference["x"])
    ref_y = float(reference["U_b"])
    ref_err = float(reference["U_b_err"])
    blue = "#1f77b4"

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.errorbar(
        x[non_ref][order],
        y[non_ref][order],
        yerr=err[non_ref][order],
        fmt="o",
        ms=5.5,
        mfc="white",
        mec=blue,
        mew=0.9,
        capsize=0,
        elinewidth=0.9,
        color=blue,
        ecolor=blue,
        linestyle="none",
        label=r"$U_b$",
        zorder=3,
    )
    ax.errorbar(
        [ref_x],
        [ref_y],
        yerr=[ref_err],
        fmt="o",
        ms=5.5,
        mfc=blue,
        mec=blue,
        mew=0.9,
        capsize=0,
        elinewidth=0.9,
        color=blue,
        ecolor=blue,
        linestyle="none",
        label=r"$U_b(\beta_0)$",
        zorder=4,
    )
    ax.set_xlabel(r"$1/(N_t\eta)$")
    ax.set_ylabel(r"$U_b$")
    ax.set_xlim(0.0, max(float(np.max(x)) * 1.04, ref_x * 1.2))
    y_low = float(np.nanmin(y - err))
    y_high = float(np.nanmax(y + err))
    y_pad = max(0.02, 0.10 * (y_high - y_low))
    ax.set_ylim(y_low - y_pad, y_high + y_pad)
    apply_report_style(ax)
    report_legend(ax, loc="best", ncol=1)
    adapt_two_column_figure(fig)
    save_figure(fig, (f"fig_bare_energy_vs_invbeta_{eta_tag(eta)}",))


def plot_delta(delta_rows: list[dict[str, Any]], fit: dict[str, float | int], eta: float) -> None:
    """Plot Delta U_b together with the exact shape and fitted amplitude."""
    plot_rows = [row for row in delta_rows if not bool(row["is_reference"])]
    x = np.asarray([float(row["x"]) for row in plot_rows], dtype=float)
    y = np.asarray([float(row["Delta_U_b"]) for row in plot_rows], dtype=float)
    err = np.asarray([float(row["Delta_U_b_err"]) for row in plot_rows], dtype=float)
    order = np.argsort(x)
    beta0 = float(delta_rows[0]["beta0"])
    # The curve is parameterized by x=1/beta because this is the plotted variable.
    x_curve = np.linspace(min(float(np.min(x)), 1.0 / beta0), float(np.max(x)) * 1.01, 400)
    beta_curve = 1.0 / x_curve
    theory = np.asarray([n_bose(float(beta)) - n_bose(beta0) for beta in beta_curve], dtype=float)
    fit_curve = float(fit["A"]) * theory
    blue = "#1f77b4"

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.errorbar(
        x[order],
        y[order],
        yerr=err[order],
        fmt="o",
        ms=5.5,
        mfc=blue,
        mec=blue,
        mew=0.9,
        capsize=0,
        elinewidth=0.9,
        color=blue,
        ecolor=blue,
        linestyle="none",
        label=r"$\Delta U_b$",
        zorder=3,
    )
    ax.plot(x_curve, theory, color="black", lw=1.3, ls=":", label="Analytical", zorder=2)
    ax.plot(x_curve, fit_curve, color="#2ca02c", lw=1.3, ls="-.", label="Fit", zorder=2)
    ax.set_xlabel(r"$1/(N_t\eta)$")
    ax.set_ylabel(r"$\Delta U_b$")
    ax.set_xlim(0.0, float(np.max(x)) * 1.04)
    y_low = float(np.nanmin(np.concatenate((y - err, theory, fit_curve))))
    y_high = float(np.nanmax(np.concatenate((y + err, theory, fit_curve))))
    y_pad = max(0.02, 0.10 * (y_high - y_low))
    ax.set_ylim(y_low - y_pad, y_high + y_pad)
    apply_report_style(ax)
    report_legend(ax, loc="best", ncol=1)
    ax.text(
        0.97,
        0.04,
        rf"$\chi^2_{{\rm red}}={float(fit['chi2_red']):.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    adapt_two_column_figure(fig)
    save_figure(fig, (f"fig_subtracted_energy_vs_invbeta_{eta_tag(eta)}",))


def write_summary(rows: list[dict[str, Any]], eta: float) -> None:
    """Write one row per beta point with the measured U_b and block error."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    for path in output_paths(SUMMARY_PATH, eta):
        with path.open("w", encoding="ascii") as handle:
            handle.write("# QHO temperature scan with non-renormalized internal-energy estimator\n")
            handle.write("# estimator U_b = 0.5*y2_mean - dy2_mean/(2*eta^2)\n")
            handle.write("# columns beta x eta Nt U_b U_b_err n_blocks block_size_saved block_size_sweeps raw_file\n")
            for row in rows:
                handle.write(
                    f"{format_float(float(row['beta']))} {format_float(float(row['x']))} "
                    f"{format_float(float(row['eta']))} {int(row['Nt'])} "
                    f"{format_float(float(row['U_b']))} {format_float(float(row['U_b_err']))} "
                    f"{int(row['n_blocks'])} {int(row['block_size_saved'])} "
                    f"{int(row['block_size_sweeps'])} {row['raw_file']}\n"
                )
        print(f"Wrote {path}")


def write_delta(delta_rows: list[dict[str, Any]], eta: float) -> None:
    """Write the beta0-subtracted temperature-scan table."""
    for path in output_paths(DELTA_PATH, eta):
        with path.open("w", encoding="ascii") as handle:
            handle.write("# QHO temperature scan delta U_b summary\n")
            handle.write("# Delta_U_b(beta) = U_b(beta) - U_b(beta0), beta0=max(beta)\n")
            handle.write("# columns beta x eta Nt beta0 x0 Nt0 U_b U_b_err U_b0 U_b0_err Delta_U_b Delta_U_b_err nB nB_minus_nB0\n")
            for row in delta_rows:
                handle.write(
                    f"{format_float(float(row['beta']))} {format_float(float(row['x']))} "
                    f"{format_float(float(row['eta']))} {int(row['Nt'])} "
                    f"{format_float(float(row['beta0']))} {format_float(float(row['x0']))} {int(row['Nt0'])} "
                    f"{format_float(float(row['U_b']))} {format_float(float(row['U_b_err']))} "
                    f"{format_float(float(row['U_b0']))} {format_float(float(row['U_b0_err']))} "
                    f"{format_float(float(row['Delta_U_b']))} {format_float(float(row['Delta_U_b_err']))} "
                    f"{format_float(float(row['nB']))} {format_float(float(row['nB_minus_nB0']))}\n"
                )
        print(f"Wrote {path}")


def write_relation_table(delta_rows: list[dict[str, Any]], eta: float) -> None:
    """Write the compact table used by the final temperature-relation plot."""
    for path in output_paths(RELATION_TABLE_PATH, eta):
        with path.open("w", encoding="ascii") as handle:
            handle.write("# QHO temperature relation table for bare internal energy\n")
            handle.write("# columns beta x Nt Delta_U_b Delta_U_b_err theory\n")
            for row in delta_rows:
                if bool(row["is_reference"]):
                    continue
                handle.write(
                    f"{format_float(float(row['beta']))} {format_float(float(row['x']))} "
                    f"{int(row['Nt'])} {format_float(float(row['Delta_U_b']))} "
                    f"{format_float(float(row['Delta_U_b_err']))} "
                    f"{format_float(float(row['nB_minus_nB0']))}\n"
                )
        print(f"Wrote {path}")


def is_short_check(rows: list[dict[str, Any]]) -> bool:
    """Identify lightweight runs whose outputs should be labeled as checks."""
    return not (
        all(int(row["n_therm"]) >= 200000 for row in rows)
        and all(int(row["n_sweeps"]) >= 1000000 for row in rows)
        and all(int(row["meas_stride"]) == 10 for row in rows)
        and all(int(row["block_size_saved"]) >= 2000 for row in rows)
    )


def write_recommendation(
    rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    reference: dict[str, Any],
    fit: dict[str, float | int],
    eta: float,
) -> None:
    """
    Write a Markdown summary explaining the estimator, subtraction, and fit.

    This file is meant to be read directly when checking the temperature-scan
    result, so it repeats the central definitions and the fit diagnostics.
    """
    beta_values = [float(row["beta"]) for row in rows]
    nt_values = [int(row["Nt"]) for row in rows]
    x_values = [float(row["x"]) for row in rows]
    eta_values = sorted({float(row["eta"]) for row in rows})
    check = is_short_check(rows)
    compatibility = (float(fit["A"]) - 1.0) / float(fit["A_err"])
    for path in output_paths(RECOMMENDATION_PATH, eta):
        with path.open("w", encoding="ascii") as handle:
            handle.write("# Temperature dependence with bare internal energy\n\n")
            handle.write("## Summary for the relation\n\n")
            handle.write(f"- fixed lattice spacing: eta = {', '.join(format_float(value) for value in eta_values)}\n")
            handle.write(f"- beta grid: {' '.join(format_float(value) for value in beta_values)}\n")
            handle.write(f"- derived Nt grid: {' '.join(str(value) for value in nt_values)}\n")
            handle.write(f"- beta range: [{format_float(min(beta_values))}, {format_float(max(beta_values))}]\n")
            handle.write(f"- Nt range: [{min(nt_values)}, {max(nt_values)}]\n")
            handle.write(f"- x range: [{format_float(min(x_values))}, {format_float(max(x_values))}]\n")
            first = rows[0]
            handle.write(
                "- MC parameters: "
                f"therm={int(first['n_therm'])}, sweeps={int(first['n_sweeps'])}, "
                f"stride={int(first['meas_stride'])}, block_size_saved={int(first['block_size_saved'])}, "
                f"block_size_sweeps={int(first['block_size_sweeps'])}, "
                f"n_blocks={int(first['n_blocks'])}, update={first['update']}, "
                f"n_over={int(first['n_over'])}\n"
            )
            handle.write(
                f"- reference: beta0={format_float(float(reference['beta']))}, "
                f"Nt0={int(reference['Nt'])}, x0={format_float(float(reference['x']))}, "
                f"U_b(beta0)={format_float(float(reference['U_b']))} +/- "
                f"{format_float(float(reference['U_b_err']))}\n"
            )
            handle.write("- estimator: U_b = 0.5*<y^2> - <(Delta y)^2>/(2*eta^2)\n")
            handle.write("- delta definition: Delta U_b(beta) = U_b(beta) - U_b(beta0)\n")
            handle.write("- exact comparison: n_B(beta)-n_B(beta0), with n_B(beta)=1/(exp(beta)-1)\n")
            handle.write(
                "- errors: saved measurements are divided into blocks; each block computes U_b from "
                "the block means of y2_mean and dy2_mean, and U_b_err is the standard error of those "
                "block U_b values. Delta_U_b errors use independent-error quadrature against beta0.\n"
            )
            handle.write(
                "- delta plot: the beta0 reference point is omitted from the Delta U_b plot and from "
                "the weighted fit because Delta U_b(beta0)=0 by construction.\n"
            )
            handle.write(
                f"- run type: {'shortened check, not final production' if check else 'full production'}\n"
            )
            if check:
                handle.write("- warning: this is a shortened check run; plots are workflow checks only.\n")
            handle.write("\n## Fit\n\n")
            handle.write("Weighted one-parameter fit, excluding beta0:\n\n")
            handle.write("Delta U_b(x) = A * [1/(exp(1/x)-1) - 1/(exp(1/x0)-1)].\n\n")
            handle.write(f"- A: {format_float(float(fit['A']))}\n")
            handle.write(f"- A_err: {format_float(float(fit['A_err']))}\n")
            handle.write(f"- chi2: {format_float(float(fit['chi2']))}\n")
            handle.write(f"- dof: {int(fit['dof'])}\n")
            handle.write(f"- chi2_red: {format_float(float(fit['chi2_red']))}\n")
            handle.write(f"- (A - 1)/A_err: {format_float(compatibility)}\n")
            handle.write("\n## Compact Table\n\n")
            handle.write("| beta | x | Nt | Delta_U_b | Delta_U_b_err | theory |\n")
            handle.write("|---:|---:|---:|---:|---:|---:|\n")
            for row in delta_rows:
                if bool(row["is_reference"]):
                    continue
                handle.write(
                    f"| {format_float(float(row['beta']))} | {format_float(float(row['x']))} | "
                    f"{int(row['Nt'])} | {format_float(float(row['Delta_U_b']))} | "
                    f"{format_float(float(row['Delta_U_b_err']))} | "
                    f"{format_float(float(row['nB_minus_nB0']))} |\n"
                )
        print(f"Wrote {path}")


def main() -> None:
    """Run the complete fixed-eta temperature analysis."""
    args = parse_args()
    input_rows = rows_from_raw_files(args.raw_files) if args.raw_files else parse_manifest(args.manifest)
    rows = [analyze_chain(row, args.manifest) for row in input_rows]
    # This analysis is meaningful only for one fixed lattice spacing.
    eta_values = {round(float(row["eta"]), 15) for row in rows}
    if len(eta_values) != 1:
        raise ValueError("temperature scan analysis expects a single fixed eta")
    eta = float(rows[0]["eta"])
    delta_rows, reference = build_delta_rows(rows)
    fit = fit_amplitude(delta_rows)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, eta)
    write_summary(rows, eta)
    write_delta(delta_rows, eta)
    write_relation_table(delta_rows, eta)
    write_recommendation(rows, delta_rows, reference, fit, eta)
    configure_plot_style()
    plot_ub(rows, reference, eta)
    plot_delta(delta_rows, fit, eta)
    print(
        "Fit: A={A} A_err={A_err} chi2={chi2} dof={dof} chi2_red={chi2_red}".format(
            A=format_float(float(fit["A"])),
            A_err=format_float(float(fit["A_err"])),
            chi2=format_float(float(fit["chi2"])),
            dof=int(fit["dof"]),
            chi2_red=format_float(float(fit["chi2_red"])),
        )
    )


if __name__ == "__main__":
    main()
