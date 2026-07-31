#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/plotting/analyze_mc_diagnostics.py
# Purpose: Create Monte Carlo stationarity and autocorrelation check figures.
# Thermalization histories, normalized autocorrelations, and blocking plateaus
# show whether quoted path-observable errors account for Markov-chain memory.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: imports and configuration
# -----------------------------------------------------------------------------

"""Generate QHO Monte Carlo checks from existing data.

The slow mode is determined empirically from the primary path observables
``y_mean``, ``y2_mean``, and ``dy2_mean``. Derived observables such as
``energy_ren`` are included only as secondary checks.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

try:
    import qho_binary_io
except ImportError:
    qho_binary_io = None


RAW_ALG_DIR = Path("data/raw/checks/algorithm_comparison")
RAW_HB_DIR = Path("data/raw/checks/hbover")
RAW_PROD_DIR = Path("data/raw/production/thermo_beta5")
DIAG_PROCESSED_DIR = Path("data/processed/checks")
PROD_DIR = Path("data/processed/production")
PLOT_DIR = Path("plots/diagnostics")

OBS_AUDIT_PATH = PROD_DIR / "qho_mc_observable_autocorr_check.md"
AUDIT_PATH = PROD_DIR / "qho_mc_checks_check.md"
ALGORITHM_SUMMARY_PATH = PROD_DIR / "qho_mc_algorithm_checks.md"
TAUINT_TABLE_PATH = PROD_DIR / "qho_mc_tauint_primary_observables_beta5.dat"
BLOCKING_STABILITY_PATH = PROD_DIR / "qho_mc_blocking_stability_beta5_nt128.dat"
STATIONARITY_PATH = PROD_DIR / "qho_mc_stationarity_beta5_nt128.dat"
STATIONARITY_FIT_WINDOWS_PATH = PROD_DIR / "qho_mc_stationarity_fit_windows_beta5.dat"
STATIONARITY_FIT_WINDOWS_SUMMARY_PATH = PROD_DIR / "qho_mc_stationarity_fit_windows_beta5_summary.md"
THERM_LOGBINS_PATH = PROD_DIR / "qho_mc_thermalization_logbins_beta5_nt512.dat"
THERM_LOGBINS_DIRECT_PATH = DIAG_PROCESSED_DIR / "qho_thermalization_logbins_beta5_nt512.dat"
THERM_CUT_SCAN_PATH = PROD_DIR / "qho_mc_thermalization_cut_scan_beta5_nt512.dat"
THERM_CUT_SCAN_SUMMARY_PATH = PROD_DIR / "qho_mc_thermalization_cut_scan_beta5_nt512_summary.md"

ALGORITHM_FILES = {
    "metro": RAW_ALG_DIR / "qho_algorithm_comparison_grid_metro_nt400_init_zero.dat",
    "heatbath": RAW_ALG_DIR / "qho_algorithm_comparison_grid_heatbath_nt400_init_zero.dat",
    "hb-over": RAW_ALG_DIR / "qho_algorithm_comparison_grid_hb_over_nt400_init_zero.dat",
}
ALGORITHM_LABELS = {
    "metro": "Metropolis",
    "heatbath": "Heatbath",
    "hb-over": "HB+over",
}
ALGORITHM_LONG_LABELS = {
    "metro": "Metropolis",
    "heatbath": "Heatbath",
    "hb-over": "Heatbath + overrelaxation",
}
ALGORITHM_COLORS = {
    "metro": "#7b3294",
    "heatbath": "#1f77b4",
    "hb-over": "#2ca02c",
}

THERM_FILES = {
    "zero": RAW_HB_DIR / "qho_hbover_diag_beta5_nt512_sweeps10000000_stride1_init_zero.dat",
    "uniform": RAW_HB_DIR / "qho_hbover_diag_beta5_nt512_sweeps10000000_stride1_init_uniform.dat",
}
THERM_STAGE_SWEEPS = 3_000_000
THERM_STAGE_FILES = {
    "zero": DIAG_PROCESSED_DIR / "qho_thermalization_logbins_beta5_nt512_sweeps3000000_zero.dat",
    "uniform": DIAG_PROCESSED_DIR / "qho_thermalization_logbins_beta5_nt512_sweeps3000000_uniform.dat",
}
THERM_CUTS = (200000, 500000, 1000000, 2000000, 5000000, 10000000, 30000000, 100000000)
NT512_RAW_PATH = RAW_PROD_DIR / "qho_thermo_beta5_nt512.dat"
FINAL_RAW_PATH = RAW_PROD_DIR / "qho_thermo_beta5_nt128.dat"

PRIMARY_OBSERVABLES = ("y_mean", "y2_mean", "dy2_mean")
DERIVED_OBSERVABLES = ("energy_ren",)
ALL_AUTOCORR_OBSERVABLES = PRIMARY_OBSERVABLES + DERIVED_OBSERVABLES

N_THERM_CHOSEN = 400000
N_SWEEPS_PROD = 1000000
MEAS_STRIDE_PROD = 10
BLOCK_SIZE_SAVED = 2000
BLOCK_SIZE_SWEEPS = BLOCK_SIZE_SAVED * MEAS_STRIDE_PROD
N_BLOCKS = N_SWEEPS_PROD // BLOCK_SIZE_SWEEPS
MIN_RELIABLE_BLOCKS = 20
Y2_FIT_NT_MIN = 25
Y2_FIT_NT_MAX = 200
ENERGY_FIT_NT_MIN = 25
ENERGY_FIT_NT_MAX = 80
THERM_LOG_BIN_BASE = 1.35
EXACT_Y2_BETA5 = 0.50678365490630423109601990090301278422429144642918728442862089901745650460541947924441402265025547793848275957445677249299820175811025952269306848402838297584202792535447904021153593340776311348866982575120083379349522116932394953700083823923297405805454594
EXACT_YMEAN = 0.0
WINDOW_C = 6.0

def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.12g}"


def configure_style() -> None:
    if sns is not None:
        sns.set_theme(style="ticks", context="paper", font_scale=1.45)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "CMU Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
        "font.size": 14,
        "axes.labelsize": 17,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "black",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def apply_report_style(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="0.70", alpha=0.55, linestyle=":", linewidth=0.8)
    ax.tick_params(direction="in", which="major", top=True, right=True, width=1.0, length=4.8)
    ax.tick_params(direction="in", which="minor", top=True, right=True, width=0.8, length=3.2)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def report_legend(ax: plt.Axes, **kwargs: Any):
    kwargs.setdefault("fontsize", 9)
    kwargs.setdefault("borderpad", 0.28)
    kwargs.setdefault("labelspacing", 0.25)
    kwargs.setdefault("handlelength", 1.8)
    kwargs.setdefault("handletextpad", 0.55)
    kwargs.setdefault("borderaxespad", 0.35)
    legend = ax.legend(frameon=True, facecolor="white", framealpha=1.0, edgecolor="none", **kwargs)
    legend.get_frame().set_linewidth(0.0)
    return legend


def adapt_full_width_figure(fig: plt.Figure) -> None:
    """Adapt selected figures for inclusion at LaTeX text width."""
    fig.set_size_inches(8.0, 4.6, forward=True)
    for ax in fig.axes:
        ax.title.set_fontsize(12)
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.xaxis.get_offset_text().set_fontsize(10)
        ax.yaxis.get_offset_text().set_fontsize(10)
        ax.tick_params(which="major", labelsize=10, length=4.5, width=1.0)
        ax.tick_params(which="minor", length=3.0, width=0.8)
        for text in ax.texts:
            text.set_fontsize(10)
        for line in ax.lines:
            line.set_linewidth(max(line.get_linewidth(), 1.3))
            if line.get_marker() not in (None, "None", "", " "):
                line.set_markersize(4.5)
        for collection in ax.collections:
            widths = collection.get_linewidths()
            if len(widths):
                collection.set_linewidth(max(float(np.max(widths)), 1.0))
        for gridline in ax.get_xgridlines() + ax.get_ygridlines():
            gridline.set_color("0.75")
            gridline.set_alpha(0.35)
            gridline.set_linewidth(0.7)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(9)
            for handle in legend.legend_handles:
                if hasattr(handle, "set_linewidth"):
                    handle.set_linewidth(1.3)
                if hasattr(handle, "set_markersize"):
                    handle.set_markersize(4.5)


def adapt_three_quarter_width_figure(fig: plt.Figure) -> None:
    """Adapt a selected figure for inclusion at 0.75 LaTeX text width."""
    width, height = fig.get_size_inches()
    fig.set_size_inches(6.3, height * 6.3 / width, forward=True)
    for ax in fig.axes:
        ax.title.set_fontsize(15)
        ax.xaxis.label.set_fontsize(16)
        ax.yaxis.label.set_fontsize(16)
        ax.tick_params(which="major", labelsize=13, length=5.0, width=1.0)
        ax.tick_params(which="minor", length=3.2, width=0.8)
        for text in ax.texts:
            text.set_fontsize(13)
        for line in ax.lines:
            line.set_linewidth(max(line.get_linewidth(), 1.5))
            if line.get_marker() not in (None, "None", "", " "):
                line.set_markersize(6.0)
        for collection in ax.collections:
            widths = collection.get_linewidths()
            if len(widths):
                collection.set_linewidth(max(float(np.max(widths)), 1.2))
        for gridline in ax.get_xgridlines() + ax.get_ygridlines():
            gridline.set_color("0.75")
            gridline.set_alpha(0.35)
            gridline.set_linewidth(0.7)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(11)
            for handle in legend.legend_handles:
                if hasattr(handle, "set_linewidth"):
                    handle.set_linewidth(1.5)
                if hasattr(handle, "set_markersize"):
                    handle.set_markersize(6.0)


def adapt_half_width_figure(fig: plt.Figure) -> None:
    """Adapt selected figures for inclusion at 0.49 LaTeX text width."""
    width, height = fig.get_size_inches()
    fig.set_size_inches(6.3, height * 6.3 / width, forward=True)
    for ax in fig.axes:
        ax.title.set_fontsize(20)
        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)
        ax.tick_params(which="major", labelsize=17, length=5.5, width=1.1)
        ax.tick_params(which="minor", length=3.5, width=0.9)
        for text in ax.texts:
            text.set_fontsize(17)
        for line in ax.lines:
            line.set_linewidth(max(line.get_linewidth(), 1.8))
        for gridline in ax.get_xgridlines() + ax.get_ygridlines():
            gridline.set_color("0.75")
            gridline.set_alpha(0.35)
            gridline.set_linewidth(0.8)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(15)
            for handle in legend.legend_handles:
                if hasattr(handle, "set_linewidth"):
                    handle.set_linewidth(1.8)


def save_figure(fig: plt.Figure, basename: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output = PLOT_DIR / f"{basename}.png"
    fig.savefig(output, bbox_inches="tight", dpi=300)
    print(f"Wrote {output}")
    plt.close(fig)


def parse_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            fields = line[1:].strip().split(maxsplit=1)
            if len(fields) == 2:
                metadata[fields[0]] = fields[1]
    return metadata


def load_binary_table(path: Path) -> tuple[dict[str, str], np.ndarray, list[str]]:
    if qho_binary_io is None:
        raise RuntimeError("scripts/analysis/qho_binary_io.py is required to read QHO binary files")
    header = qho_binary_io.read_header(path)
    records = qho_binary_io.memmap_records(path)
    columns = ["sweep", "y_mean", "y2_mean", "dy2_mean", "energy_ren", "acc_rate"]
    metadata = {
        "nt": str(header["nt"]),
        "beta": str(header["beta"]),
        "eta": str(header["eta"]),
        "therm": str(header["n_therm"]),
        "sweeps": str(header["n_sweeps"]),
        "stride": str(header["meas_stride"]),
        "seed": str(header["seed"]),
        "stream": str(header["stream"]),
        "n_over": str(header["n_over"]),
        "format": "bin",
    }
    return metadata, records, columns


def load_raw_table(path: Path) -> tuple[dict[str, str], np.ndarray, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".bin" or (qho_binary_io is not None and qho_binary_io.is_binary_qho(path)):
        return load_binary_table(path)
    metadata = parse_metadata(path)
    columns: list[str] | None = None
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("# sweep "):
                columns = line[2:].split()
                break
    if columns is None:
        raise ValueError(f"{path}: missing column header")
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return metadata, data, columns


def column(data: np.ndarray, columns: list[str], name: str) -> np.ndarray:
    if data.dtype.names is not None:
        return data[name]
    return data[:, columns.index(name)]


def meta_float(metadata: dict[str, str], key: str, fallback: float = math.nan) -> float:
    value = metadata.get(key)
    if value is None:
        return fallback
    return float(value)


def meta_int(metadata: dict[str, str], key: str, fallback: int = 0) -> int:
    value = metadata.get(key)
    if value is None:
        return fallback
    return int(float(value))


def normalized_autocorr(series: np.ndarray, max_lag: int) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        raise ValueError("autocorrelation requires at least two values")
    max_lag = min(max_lag, n - 1)
    centered = values - np.mean(values)
    nfft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=nfft)
    raw = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft)[: max_lag + 1]
    norm = np.arange(n, n - max_lag - 1, -1, dtype=float)
    acov = raw / norm
    if acov[0] <= 0.0:
        raise ValueError("non-positive variance in autocorrelation input")
    return acov / acov[0]


def tau_int_madras_sokal(rho: np.ndarray, c: float = WINDOW_C) -> tuple[float, int, bool]:
    tau_curve = 0.5 + np.cumsum(rho[1:])
    for idx, tau in enumerate(tau_curve, start=1):
        if tau > 0.0 and idx >= c * tau:
            return float(tau), idx, False
    return float(tau_curve[-1]), int(len(rho) - 1), True


def tau_status(tau: float, window: int, saturated: bool, rho: np.ndarray) -> str:
    if saturated or tau <= 0.0 or not math.isfinite(tau):
        return "noisy"
    if window >= len(rho) - 2:
        return "noisy"
    return "stable"


def block_error(series: np.ndarray, block_size: int) -> tuple[float, int]:
    n_blocks = int(series.size // block_size)
    if n_blocks < 2:
        return math.nan, n_blocks
    trimmed = series[: n_blocks * block_size]
    block_means = trimmed.reshape(n_blocks, block_size).mean(axis=1)
    err = float(np.std(block_means, ddof=1) / math.sqrt(n_blocks))
    return err, n_blocks


def blocking_curve(series: np.ndarray) -> np.ndarray:
    max_block = series.size // MIN_RELIABLE_BLOCKS
    sizes = {2 ** k for k in range(int(math.log2(max_block)) + 1)}
    sizes.add(BLOCK_SIZE_SAVED)
    sizes.discard(2048)
    rows = []
    for size in sorted(size for size in sizes if 1 <= size <= max_block):
        err, n_blocks = block_error(series, size)
        if n_blocks >= MIN_RELIABLE_BLOCKS and math.isfinite(err):
            rows.append((float(size), float(n_blocks), err))
    return np.asarray(rows, dtype=float)


def block_mean_error(series: np.ndarray, block_size: int = BLOCK_SIZE_SAVED) -> tuple[float, float, int]:
    n_blocks = int(series.size // block_size)
    if n_blocks < 2:
        return float(np.mean(series)), math.nan, n_blocks
    trimmed = series[: n_blocks * block_size]
    means = trimmed.reshape(n_blocks, block_size).mean(axis=1)
    return float(np.mean(means)), float(np.std(means, ddof=1) / math.sqrt(n_blocks)), n_blocks


def stationarity_status(z_value: float) -> str:
    abs_z = abs(z_value)
    if abs_z <= 2.0:
        return "pass"
    if abs_z <= 3.0:
        return "warning"
    return "fail"


def available_observables(columns: list[str]) -> tuple[list[str], list[str]]:
    primary = [name for name in PRIMARY_OBSERVABLES if name in columns]
    derived = [name for name in DERIVED_OBSERVABLES if name in columns]
    return primary, derived


def analyze_observables(
    label: str,
    path: Path,
    update_label: str,
    max_lag: int,
    burn_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata, data, columns = load_raw_table(path)
    primary, derived = available_observables(columns)
    if primary != list(PRIMARY_OBSERVABLES):
        raise ValueError(f"{path}: expected primary columns {PRIMARY_OBSERVABLES}, found {primary}")

    burn = int(data.shape[0] * burn_fraction)
    stride = meta_int(metadata, "stride", MEAS_STRIDE_PROD)
    rows: dict[str, Any] = {}
    for obs in primary + derived:
        values = column(data, columns, obs)[burn:]
        rho = normalized_autocorr(values, min(max_lag, values.size - 1))
        tau, window, saturated = tau_int_madras_sokal(rho)
        rows[obs] = {
            "dataset": label,
            "path": path,
            "metadata": metadata,
            "columns": columns,
            "update": update_label,
            "observable": obs,
            "rho": rho,
            "tau_int_saved": tau,
            "tau_int_sweeps": tau * stride,
            "window": window,
            "method": f"Madras-Sokal c={format_float(WINDOW_C)}",
            "status": tau_status(tau, window, saturated, rho),
            "saturated": saturated,
            "burn_saved": burn,
            "stride": stride,
            "beta": meta_float(metadata, "beta"),
            "Nt": meta_int(metadata, "nt", meta_int(metadata, "Nt")),
            "eta": meta_float(metadata, "eta"),
        }
    loaded = {"metadata": metadata, "data": data, "columns": columns, "path": path}
    return rows, loaded


def choose_slowest_primary(rows: dict[str, Any]) -> str:
    return max(PRIMARY_OBSERVABLES, key=lambda obs: rows[obs]["tau_int_saved"])


def analyze_algorithm() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    loaded: dict[str, Any] = {}
    for update, path in ALGORITHM_FILES.items():
        rows, raw = analyze_observables(
            label=f"algorithm_{update}_beta5_nt400",
            path=path,
            update_label=update,
            max_lag=2000,
            burn_fraction=0.2,
        )
        raw["acceptance"] = (
            float(np.mean(column(raw["data"], raw["columns"], "acc_rate")))
            if "acc_rate" in raw["columns"]
            else math.nan
        )
        results[update] = rows
        loaded[update] = raw
    return results, loaded


def analyze_final() -> tuple[dict[str, Any], dict[str, Any]]:
    return analyze_observables(
        label="final_production_beta5_nt128",
        path=FINAL_RAW_PATH,
        update_label="hb-over",
        max_lag=5000,
        burn_fraction=0.0,
    )


def analyze_thermalization() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for init, path in THERM_FILES.items():
        metadata, data, columns = load_raw_table(path)
        result[init] = {
            "metadata": metadata,
            "data": data,
            "columns": columns,
            "path": path,
            "max_sweep": int(np.max(column(data, columns, "sweep"))),
            "has_post_therm": bool(np.max(column(data, columns, "sweep")) > N_THERM_CHOSEN),
        }
    return result


def analyze_blocking(raw: dict[str, Any]) -> dict[str, Any]:
    data = raw["data"]
    columns = raw["columns"]
    results: dict[str, Any] = {}
    for obs in ALL_AUTOCORR_OBSERVABLES:
        if obs not in columns:
            continue
        values = column(data, columns, obs)
        curve = blocking_curve(values)
        idx = int(np.argmin(np.abs(curve[:, 0] - BLOCK_SIZE_SAVED)))
        err_2000 = float(curve[idx, 2])
        larger = curve[curve[:, 0] > BLOCK_SIZE_SAVED, 2]
        median_larger = float(np.median(larger)) if larger.size else math.nan
        rel_diff = (
            abs(err_2000 - median_larger) / median_larger
            if math.isfinite(median_larger) and median_larger != 0.0
            else math.nan
        )
        results[obs] = {
            "curve": curve,
            "err_2000": err_2000,
            "median_larger": median_larger,
            "rel_diff": rel_diff,
            "plateau": bool(math.isfinite(rel_diff) and rel_diff <= 0.15),
        }
    return results


def analyze_stationarity(raw: dict[str, Any]) -> dict[str, Any]:
    data = raw["data"]
    columns = raw["columns"]
    results: dict[str, Any] = {}
    half = data.shape[0] // 2
    for obs in ALL_AUTOCORR_OBSERVABLES:
        values = column(data, columns, obs)
        first = values[:half]
        second = values[half:]
        mean_first, err_first, n_first = block_mean_error(first)
        mean_second, err_second, n_second = block_mean_error(second)
        denom = math.sqrt(err_first * err_first + err_second * err_second)
        z_diff = (mean_first - mean_second) / denom if denom > 0.0 and math.isfinite(denom) else math.nan
        quarters = []
        quarter_size = values.size // 4
        for idx in range(4):
            q_values = values[idx * quarter_size : (idx + 1) * quarter_size]
            q_mean, q_err, q_blocks = block_mean_error(q_values)
            quarters.append({"mean": q_mean, "err": q_err, "n_blocks": q_blocks})
        results[obs] = {
            "mean_first": mean_first,
            "err_first": err_first,
            "n_blocks_first": n_first,
            "mean_second": mean_second,
            "err_second": err_second,
            "n_blocks_second": n_second,
            "z_diff": z_diff,
            "status": stationarity_status(z_diff),
            "quarters": quarters,
        }
    return results


def production_raw_files_by_nt() -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(RAW_PROD_DIR.glob("qho_thermo_beta5_nt*.dat")):
        metadata = parse_metadata(path)
        nt = meta_int(metadata, "nt", meta_int(metadata, "Nt"))
        files[nt] = path
    for path in sorted(RAW_PROD_DIR.glob("qho_thermo_beta5_nt*.bin")):
        metadata, _, _ = load_raw_table(path)
        nt = meta_int(metadata, "nt", meta_int(metadata, "Nt"))
        files.setdefault(nt, path)
    return files


def analyze_stationarity_fit_windows() -> dict[str, Any]:
    files = production_raw_files_by_nt()
    windows = {
        "y2_fit": {"nt_values": [nt for nt in sorted(files) if Y2_FIT_NT_MIN <= nt <= Y2_FIT_NT_MAX], "plot_observable": "y2_mean"},
        "energy_fit": {"nt_values": [nt for nt in sorted(files) if ENERGY_FIT_NT_MIN <= nt <= ENERGY_FIT_NT_MAX], "plot_observable": "energy_ren"},
    }
    results: dict[str, Any] = {}
    for window_name, spec in windows.items():
        rows = []
        for nt in spec["nt_values"]:
            metadata, data, columns = load_raw_table(files[nt])
            raw = {"metadata": metadata, "data": data, "columns": columns, "path": files[nt]}
            stationarity = analyze_stationarity(raw)
            for obs in ALL_AUTOCORR_OBSERVABLES:
                row = stationarity[obs]
                rows.append({
                    "window": window_name,
                    "Nt": nt,
                    "beta": meta_float(metadata, "beta"),
                    "eta": meta_float(metadata, "eta"),
                    "observable": obs,
                    "raw_file": files[nt],
                    **row,
                })
        results[window_name] = {"rows": rows, **spec}
    return results


def log_bin_edges(max_sweep: int, base: float = THERM_LOG_BIN_BASE) -> list[int]:
    edges = [1]
    meas_time = 1
    while meas_time < max_sweep:
        meas_time = int(meas_time * base) + 1
        if meas_time <= edges[-1]:
            meas_time = edges[-1] + 1
        edges.append(min(meas_time, max_sweep))
    return edges


def analyze_thermalization_logbins(thermal: dict[str, Any]) -> list[dict[str, Any]]:
    max_sweep = max(thermal[init]["max_sweep"] for init in thermal)
    edges = log_bin_edges(max_sweep)
    rows: list[dict[str, Any]] = []
    for init in ("zero", "uniform"):
        data = thermal[init]["data"]
        columns = thermal[init]["columns"]
        sweep = column(data, columns, "sweep")
        for obs in ("y_mean", "y2_mean"):
            values = column(data, columns, obs)
            for start, end in zip(edges[:-1], edges[1:]):
                mask = (sweep >= start) & (sweep <= end)
                n_points = int(np.count_nonzero(mask))
                if n_points == 0:
                    continue
                selected = values[mask]
                err = float(np.std(selected, ddof=1) / math.sqrt(n_points)) if n_points >= 2 else math.nan
                rows.append({
                    "init": init,
                    "observable": obs,
                    "bin_start": int(start),
                    "bin_end": int(end),
                    "bin_center": math.sqrt(float(start) * float(end)),
                    "bin_mean": float(np.mean(selected)),
                    "bin_err": err,
                    "n_points": n_points,
                })
    return rows


def load_thermalization_logbins(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            init, observable, bin_start, bin_end, bin_center, bin_mean, bin_err, n_points = line.split()
            rows.append({
                "init": init,
                "observable": observable,
                "bin_start": int(bin_start),
                "bin_end": int(bin_end),
                "bin_center": float(bin_center),
                "bin_mean": float(bin_mean),
                "bin_err": float(bin_err),
                "n_points": int(n_points),
            })
    if not rows:
        raise ValueError(f"{path}: no log-bin rows found")
    return rows


def load_stage_logbins() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing = [path for path in THERM_STAGE_FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing staged log-bin files: " + ", ".join(str(path) for path in missing))
    for init, path in THERM_STAGE_FILES.items():
        init_rows = load_thermalization_logbins(path)
        rows.extend(init_rows)
    metadata = {
        "path": ", ".join(str(path) for path in THERM_STAGE_FILES.values()),
        "columns": ["init", "observable", "bin_start", "bin_end", "bin_center", "bin_mean", "bin_err", "n_points"],
        "max_sweep": max(row["bin_end"] for row in rows),
        "has_post_therm": max(row["bin_end"] for row in rows) > N_THERM_CHOSEN,
        "stage_sweeps": THERM_STAGE_SWEEPS,
        "inits": tuple(THERM_STAGE_FILES),
    }
    return rows, metadata


def add_reconstructed_energy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = list(rows)
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["init"], row["bin_start"], row["bin_end"])
        grouped.setdefault(key, {})[row["observable"]] = row
    eta = 5.0 / 512.0
    for (init, start, end), obs_rows in grouped.items():
        if "y2_mean" not in obs_rows or "dy2_mean" not in obs_rows:
            continue
        y2 = obs_rows["y2_mean"]
        dy2 = obs_rows["dy2_mean"]
        energy = 0.5 * y2["bin_mean"] + 1.0 / (2.0 * eta) - dy2["bin_mean"] / (2.0 * eta * eta)
        output.append({
            "init": init,
            "observable": "energy_ren",
            "bin_start": start,
            "bin_end": end,
            "bin_center": y2["bin_center"],
            "bin_mean": energy,
            "bin_err": math.nan,
            "n_points": min(y2["n_points"], dy2["n_points"]),
        })
    return output


def post_cut_logbin_mean(rows: list[dict[str, Any]], init: str, observable: str, cut: int) -> tuple[float, float, int]:
    values = np.asarray(
        [row["bin_mean"] for row in rows if row["init"] == init and row["observable"] == observable and row["bin_start"] >= cut],
        dtype=float,
    )
    if values.size < 2:
        return math.nan, math.nan, int(values.size)
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(values.size)), int(values.size)


def analyze_nt512_tau() -> tuple[dict[str, Any], dict[str, Any]]:
    return analyze_observables(
        label="final_production_beta5_nt512",
        path=NT512_RAW_PATH,
        update_label="hb-over",
        max_lag=5000,
        burn_fraction=0.0,
    )


def analyze_thermalization_cut_scan(rows: list[dict[str, Any]], nt512_rows: dict[str, Any]) -> dict[str, Any]:
    rows_with_energy = add_reconstructed_energy(rows)
    inits = tuple(sorted({row["init"] for row in rows}))
    observables = ("y_mean", "y2_mean", "dy2_mean", "energy_ren")
    scan_rows: list[dict[str, Any]] = []
    for cut in THERM_CUTS:
        if cut > THERM_STAGE_SWEEPS:
            continue
        for observable in observables:
            means = {init: post_cut_logbin_mean(rows_with_energy, init, observable, cut) for init in inits}
            for i, init_i in enumerate(inits):
                for init_j in inits[i + 1 :]:
                    mean_i, err_i, n_i = means[init_i]
                    mean_j, err_j, n_j = means[init_j]
                    denom = math.sqrt(err_i * err_i + err_j * err_j)
                    z_value = (mean_i - mean_j) / denom if denom > 0.0 and math.isfinite(denom) else math.nan
                    status = stationarity_status(z_value) if math.isfinite(z_value) else "fail"
                    scan_rows.append({
                        "cut": cut,
                        "observable": observable,
                        "init_i": init_i,
                        "init_j": init_j,
                        "mean_i": mean_i,
                        "err_i": err_i,
                        "n_i": n_i,
                        "mean_j": mean_j,
                        "err_j": err_j,
                        "n_j": n_j,
                        "z_init": z_value,
                        "status": status,
                    })
    n_therm0 = math.nan
    for cut in THERM_CUTS:
        primary_rows = [row for row in scan_rows if row["cut"] == cut and row["observable"] in PRIMARY_OBSERVABLES]
        if primary_rows and all(row["status"] == "pass" for row in primary_rows):
            n_therm0 = float(cut)
            break
    slow_obs = max(PRIMARY_OBSERVABLES, key=lambda obs: nt512_rows[obs]["tau_int_sweeps"])
    tau_int_slow = float(nt512_rows[slow_obs]["tau_int_sweeps"])
    n_therm_true = max(2.0 * n_therm0, 20.0 * tau_int_slow) if math.isfinite(n_therm0) else math.nan
    return {
        "rows": scan_rows,
        "inits": inits,
        "n_therm0": n_therm0,
        "slow_observable": slow_obs,
        "tau_int_slow": tau_int_slow,
        "n_therm_true": n_therm_true,
        "reference_therm_compatible": bool(math.isfinite(n_therm_true) and N_THERM_CHOSEN >= n_therm_true),
        "rerun_recommended": bool(math.isfinite(n_therm_true) and N_THERM_CHOSEN < n_therm_true),
    }


def plot_update_tau_ymean(algorithm: dict[str, dict[str, Any]]) -> None:
    updates = ("metro", "heatbath", "hb-over")
    x = np.arange(len(updates))
    tau_ymean = [algorithm[update]["y_mean"]["tau_int_sweeps"] for update in updates]

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    colors = [ALGORITHM_COLORS[update] for update in updates]
    bars = ax.bar(x, tau_ymean, color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([ALGORITHM_LABELS[update] for update in updates])
    ax.set_ylabel(r"$\tau_{\rm{int}, \langle y \rangle}\ (t[\mathrm{MCS}])$")
    ax.set_ylim(0.0, max(tau_ymean) * 1.18)
    for bar, tau in zip(bars, tau_ymean):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            tau,
            f"{tau:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    apply_report_style(ax)
    adapt_half_width_figure(fig)
    save_figure(fig, "fig_sampling_efficiency_ymean_beta5_nt400")


def plot_primary_autocorr(final_rows: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    colors = {
        "y_mean": "#1f77b4",
        "y2_mean": "#2ca02c",
        "dy2_mean": "#d62728",
        "energy_ren": "#7b3294",
    }
    labels = {
        "y_mean": r"$\langle y\rangle$",
        "y2_mean": r"$\langle y^2\rangle$",
        "dy2_mean": r"$\langle(\Delta y)^2\rangle$",
        "energy_ren": r"$H_{\rm ren}$",
    }
    stride = int(final_rows["y_mean"]["stride"])
    for obs in PRIMARY_OBSERVABLES:
        rho = final_rows[obs]["rho"]
        t_mcs = np.arange(rho.size) * stride
        ax.plot(t_mcs, rho, lw=1.2, color=colors[obs], label=labels[obs])
    if "energy_ren" in final_rows:
        rho = final_rows["energy_ren"]["rho"]
        t_mcs = np.arange(rho.size) * stride
        ax.plot(t_mcs, rho, lw=1.0, ls="--", color=colors["energy_ren"], label=labels["energy_ren"])
    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    ax.set_xlim(0, 500)
    ax.set_xlabel(r"$t\,[\mathrm{MCS}]$")
    ax.set_ylabel(r"$C_O(t)$")
    apply_report_style(ax)
    report_legend(ax, loc="upper right")
    adapt_half_width_figure(fig)
    save_figure(fig, "fig_autocorrelation_primary_beta5_nt128")


def plot_thermalization_logbins(rows: list[dict[str, Any]], cut_scan: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1), sharex=True)
    colors = {"zero": "#1f77b4", "uniform": "#d62728", "plus": "#2ca02c", "minus": "#9467bd"}
    labels = {"zero": r"$0$", "uniform": r"$U[-1,1]$", "plus": r"$+1$", "minus": r"$-1$"}
    specs = (("y_mean", r"$\langle y\rangle$", EXACT_YMEAN), ("y2_mean", r"$\langle y^2\rangle$", EXACT_Y2_BETA5))
    x_min = min(row["bin_center"] for row in rows)
    x_max = max(row["bin_center"] for row in rows)
    inits = tuple(init for init in ("zero", "uniform", "plus", "minus") if any(row["init"] == init for row in rows))
    for ax, (obs, ylabel, exact) in zip(axes, specs):
        for init in inits:
            selected = [row for row in rows if row["init"] == init and row["observable"] == obs]
            x = np.asarray([row["bin_center"] for row in selected], dtype=float)
            y = np.asarray([row["bin_mean"] for row in selected], dtype=float)
            yerr = np.asarray([row["bin_err"] for row in selected], dtype=float)
            ax.errorbar(x, y, yerr=yerr, color=colors[init], marker="o", ms=3.0, lw=0.9, elinewidth=0.6, capsize=0.0, label=labels[init])
        ax.axhline(exact, color="black", lw=1.0, ls="--", label="_nolegend_")
        if math.isfinite(cut_scan["n_therm_true"]):
            ax.axvline(cut_scan["n_therm_true"], color="0.25", lw=1.0, ls=":", label=r"$n_{\rm therm}$")
        else:
            for cut in (200000, 500000, 1000000):
                ax.axvline(cut, color="0.35", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(r"$t\,[\mathrm{MCS}]$")
        ax.set_ylabel(ylabel)
        apply_report_style(ax)
    axes[0].text(0.04, 0.92, "(a)", transform=axes[0].transAxes, fontsize=13, va="top")
    axes[1].text(0.04, 0.92, "(b)", transform=axes[1].transAxes, fontsize=13, va="top")
    for index, ax in enumerate(axes):
        report_legend(ax, loc="lower left" if index == 0 else "upper left")
    fig.tight_layout(w_pad=2.0)
    adapt_three_quarter_width_figure(fig)
    fig.subplots_adjust(wspace=0.32)
    for ax in axes:
        for line in ax.lines:
            line.set_linewidth(1.0)
            if line.get_marker() not in (None, "None", "", " "):
                line.set_markersize(3.5)
        legend = ax.get_legend()
        if legend is not None:
            for handle in legend.legend_handles:
                if hasattr(handle, "set_linewidth"):
                    handle.set_linewidth(1.0)
                if hasattr(handle, "set_markersize"):
                    handle.set_markersize(3.5)
    save_figure(fig, "fig_thermalization_logbins_beta5_nt512")


def write_tau_table(algorithm: dict[str, dict[str, Any]], final_rows: dict[str, Any]) -> None:
    with TAUINT_TABLE_PATH.open("w", encoding="ascii") as handle:
        handle.write("# beta Nt eta update observable tau_int_saved tau_int_sweeps W method status\n")
        for update in ("metro", "heatbath", "hb-over"):
            for obs in ALL_AUTOCORR_OBSERVABLES:
                row = algorithm[update][obs]
                handle.write(
                    f"{format_float(row['beta'])} {row['Nt']} {format_float(row['eta'])} {update} {obs} "
                    f"{format_float(row['tau_int_saved'])} {format_float(row['tau_int_sweeps'])} "
                    f"{row['window']} madras_sokal_c6 {row['status']}\n"
                )
        for obs in ALL_AUTOCORR_OBSERVABLES:
            row = final_rows[obs]
            handle.write(
                f"{format_float(row['beta'])} {row['Nt']} {format_float(row['eta'])} final_hb-over {obs} "
                f"{format_float(row['tau_int_saved'])} {format_float(row['tau_int_sweeps'])} "
                f"{row['window']} madras_sokal_c6 {row['status']}\n"
            )
    print(f"Wrote {TAUINT_TABLE_PATH}")


def write_blocking_stability(blocking: dict[str, Any]) -> None:
    with BLOCKING_STABILITY_PATH.open("w", encoding="ascii") as handle:
        handle.write("# observable block_size_saved min_reliable_blocks err_at_2000 median_err_larger_reliable rel_diff plateau_compatible\n")
        for obs in ALL_AUTOCORR_OBSERVABLES:
            row = blocking[obs]
            handle.write(
                f"{obs} {BLOCK_SIZE_SAVED} {MIN_RELIABLE_BLOCKS} {format_float(row['err_2000'])} "
                f"{format_float(row['median_larger'])} {format_float(row['rel_diff'])} {int(row['plateau'])}\n"
            )
    print(f"Wrote {BLOCKING_STABILITY_PATH}")


def write_stationarity(stationarity: dict[str, Any]) -> None:
    with STATIONARITY_PATH.open("w", encoding="ascii") as handle:
        handle.write("# observable mean_first err_first mean_second err_second z_diff status\n")
        for obs in ALL_AUTOCORR_OBSERVABLES:
            row = stationarity[obs]
            handle.write(
                f"{obs} {format_float(row['mean_first'])} {format_float(row['err_first'])} "
                f"{format_float(row['mean_second'])} {format_float(row['err_second'])} "
                f"{format_float(row['z_diff'])} {row['status']}\n"
            )
    print(f"Wrote {STATIONARITY_PATH}")


def write_stationarity_fit_windows(fit_windows: dict[str, Any]) -> None:
    with STATIONARITY_FIT_WINDOWS_PATH.open("w", encoding="ascii") as handle:
        handle.write("# window Nt beta eta observable mean_first err_first mean_second err_second z_diff status raw_file\n")
        for window_name in ("y2_fit", "energy_fit"):
            for row in fit_windows[window_name]["rows"]:
                handle.write(
                    f"{window_name} {row['Nt']} {format_float(row['beta'])} {format_float(row['eta'])} "
                    f"{row['observable']} {format_float(row['mean_first'])} {format_float(row['err_first'])} "
                    f"{format_float(row['mean_second'])} {format_float(row['err_second'])} "
                    f"{format_float(row['z_diff'])} {row['status']} {row['raw_file']}\n"
                )
    print(f"Wrote {STATIONARITY_FIT_WINDOWS_PATH}")


def write_stationarity_fit_windows_summary(fit_windows: dict[str, Any]) -> None:
    with STATIONARITY_FIT_WINDOWS_SUMMARY_PATH.open("w", encoding="ascii") as handle:
        handle.write("# Fit-window stationarity summary\n\n")
        for window_name, obs, label in (("y2_fit", "y2_mean", "y2_mean"), ("energy_fit", "energy_ren", "energy_ren")):
            rows = [row for row in fit_windows[window_name]["rows"] if row["observable"] == obs]
            max_abs = max(abs(row["z_diff"]) for row in rows)
            warnings = [row for row in rows if row["status"] == "warning"]
            failures = [row for row in rows if row["status"] == "fail"]
            all_pass = not warnings and not failures
            nt_values = " ".join(str(row["Nt"]) for row in rows)
            handle.write(f"## {label}\n\n")
            handle.write(f"- N_t values: {nt_values}\n")
            handle.write(f"- max |z_diff|: {format_float(max_abs)}\n")
            handle.write(f"- all points pass |z_diff| <= 2: {all_pass}\n")
            handle.write("- warnings: " + (", ".join(f"N_t={row['Nt']} z={format_float(row['z_diff'])}" for row in warnings) if warnings else "none") + "\n")
            handle.write("- failures: " + (", ".join(f"N_t={row['Nt']} z={format_float(row['z_diff'])}" for row in failures) if failures else "none") + "\n\n")
    print(f"Wrote {STATIONARITY_FIT_WINDOWS_SUMMARY_PATH}")


def write_thermalization_logbins(rows: list[dict[str, Any]]) -> None:
    with THERM_LOGBINS_PATH.open("w", encoding="ascii") as handle:
        handle.write(f"# Marinari-style log bins, base b={format_float(THERM_LOG_BIN_BASE)}\n")
        handle.write("# init observable bin_start bin_end bin_center bin_mean bin_err n_points\n")
        for row in rows:
            handle.write(
                f"{row['init']} {row['observable']} {row['bin_start']} {row['bin_end']} "
                f"{format_float(row['bin_center'])} {format_float(row['bin_mean'])} "
                f"{format_float(row['bin_err'])} {row['n_points']}\n"
            )
    print(f"Wrote {THERM_LOGBINS_PATH}")


def write_thermalization_cut_scan(cut_scan: dict[str, Any]) -> None:
    with THERM_CUT_SCAN_PATH.open("w", encoding="ascii") as handle:
        handle.write("# cut observable init_i init_j mean_i err_i n_i mean_j err_j n_j z_init status\n")
        for row in cut_scan["rows"]:
            handle.write(
                f"{row['cut']} {row['observable']} {row['init_i']} {row['init_j']} "
                f"{format_float(row['mean_i'])} {format_float(row['err_i'])} {row['n_i']} "
                f"{format_float(row['mean_j'])} {format_float(row['err_j'])} {row['n_j']} "
                f"{format_float(row['z_init'])} {row['status']}\n"
            )
    with THERM_CUT_SCAN_SUMMARY_PATH.open("w", encoding="ascii") as handle:
        handle.write("# Thermalization cut scan, beta=5, N_t=512\n\n")
        handle.write(f"- stages run: {THERM_STAGE_SWEEPS}\n")
        handle.write("- output mode: online log-binned, no full trace\n")
        handle.write(f"- log-bin base: {format_float(THERM_LOG_BIN_BASE)}\n")
        handle.write(f"- initializations included: {' '.join(cut_scan['inits'])}\n")
        handle.write("- plus/minus starts: not included; current executable supports zero, random, and uniform only.\n")
        handle.write(f"- n_therm0: {format_float(cut_scan['n_therm0'])}\n")
        handle.write(f"- tau_int_slow observable: {cut_scan['slow_observable']}\n")
        handle.write(f"- tau_int_slow [MCS]: {format_float(cut_scan['tau_int_slow'])}\n")
        handle.write(f"- n_therm_true = max(2*n_therm0, 20*tau_int_slow): {format_float(cut_scan['n_therm_true'])}\n")
        handle.write(f"- n_therm=200000 compatible with the conservative criterion: {cut_scan['reference_therm_compatible']}\n")
        handle.write(f"- production rerun recommended: {cut_scan['rerun_recommended']}\n")
        if cut_scan["rerun_recommended"]:
            handle.write(f"- suggested new n_therm: {int(math.ceil(cut_scan['n_therm_true']))}\n")
        handle.write("\n## Cut statuses\n\n")
        for cut in THERM_CUTS:
            rows = [row for row in cut_scan["rows"] if row["cut"] == cut and row["observable"] in PRIMARY_OBSERVABLES]
            if not rows:
                continue
            max_abs = max(abs(row["z_init"]) for row in rows if math.isfinite(row["z_init"]))
            statuses = sorted({row["status"] for row in rows})
            handle.write(f"- cut={cut}: statuses={','.join(statuses)}, max primary |z_init|={format_float(max_abs)}\n")
        handle.write("\nAgreement of different starts is treated as necessary but not sufficient; the final recommendation uses max(2*n_therm0, 20*tau_int_slow).\n")
    print(f"Wrote {THERM_CUT_SCAN_PATH}")
    print(f"Wrote {THERM_CUT_SCAN_SUMMARY_PATH}")


def write_observable_check(algorithm_raw: dict[str, Any], final_raw: dict[str, Any], thermal: dict[str, Any]) -> None:
    with OBS_AUDIT_PATH.open("w", encoding="ascii") as handle:
        handle.write("# Observable autocorrelation check\n\n")
        handle.write("## Raw columns\n\n")
        for update in ("metro", "heatbath", "hb-over"):
            raw = algorithm_raw[update]
            handle.write(f"- `{raw['path']}`: {' '.join(raw['columns'])}\n")
        handle.write(f"- `{final_raw['path']}`: {' '.join(final_raw['columns'])}\n")
        for init in ("zero", "uniform"):
            handle.write(f"- `{thermal[init]['path']}`: {' '.join(thermal[init]['columns'])}\n")
        handle.write("\n## Classification\n\n")
        handle.write("- Primary observables used to estimate the slow mode: y_mean, y2_mean, dy2_mean.\n")
        handle.write("- Derived observables: potential, kinetic_ren, energy_ren, acc_rate. The derived energy_ren is controlled because it is a final thermodynamic observable, but it is not used to define the primary slow mode.\n")
        handle.write("- The renormalized energy is derived from primary path observables through H_ren = 0.5*y2_mean + 1/(2 eta) - dy2_mean/(2 eta^2).\n")
        handle.write("- Column names match the raw headers, so no alias mapping was needed.\n")
        handle.write("- Blocking support is stored as a numerical stability table rather than a final plot.\n")
        handle.write("- Thermalization support uses Marinari-style log-binned interval averages plus production stationarity checks; two-start agreement alone is not treated as proof.\n")
        handle.write("- Final autocorrelation axes are expressed in MCS: lag times the `meas_stride` metadata.\n")
        handle.write("- Final relation-ready plots: fig_sampling_efficiency_ymean_beta5_nt400, fig_autocorrelation_primary_beta5_nt128, fig_thermalization_logbins_beta5_nt512.\n")
        handle.write("- Final beta=5 thermodynamic productions were rerun with n_therm=400000 after applying the conservative thermalization criterion.\n")
    print(f"Wrote {OBS_AUDIT_PATH}")


def write_algorithm_summary(algorithm: dict[str, dict[str, Any]], algorithm_raw: dict[str, Any]) -> None:
    with ALGORITHM_SUMMARY_PATH.open("w", encoding="ascii") as handle:
        handle.write("# Monte Carlo algorithm checks\n\n")
        handle.write("- check: beta=5, N_t=400, eta=0.0125, init=zero\n")
        handle.write("- final plot observable: y_mean, because it is the limiting primary observable in the final beta=5, N_t=128 production dataset\n")
        handle.write("- numerical slow mode definition kept for check: max tau_int over primary observables y_mean, y2_mean, dy2_mean\n")
        handle.write("- tau_int method: Madras-Sokal self-consistent window, W >= 6 tau_int(W)\n")
        handle.write("- production update: heatbath + overrelaxation with n_over=5\n")
        handle.write("- main motivation: the QHO lattice action is quadratic, so the single-site conditional distribution is Gaussian; the comparison is a qualitative check.\n\n")
        handle.write("| update | tau_int(y_mean) MCS | tau_int(y2_mean) MCS | tau_int(dy2_mean) MCS | slow primary observable | tau_slow MCS | acceptance |\n")
        handle.write("|---|---:|---:|---:|---|---:|---:|\n")
        for update in ("metro", "heatbath", "hb-over"):
            obs = choose_slowest_primary(algorithm[update])
            acceptance = algorithm_raw[update]["acceptance"]
            handle.write(
                f"| {ALGORITHM_LONG_LABELS[update]} | "
                f"{format_float(algorithm[update]['y_mean']['tau_int_sweeps'])} | "
                f"{format_float(algorithm[update]['y2_mean']['tau_int_sweeps'])} | "
                f"{format_float(algorithm[update]['dy2_mean']['tau_int_sweeps'])} | "
                f"{obs} | {format_float(algorithm[update][obs]['tau_int_sweeps'])} | "
                f"{format_float(acceptance)} |\n"
            )
    print(f"Wrote {ALGORITHM_SUMMARY_PATH}")


def write_check(
    algorithm: dict[str, dict[str, Any]],
    final_rows: dict[str, Any],
    thermal: dict[str, Any],
    blocking: dict[str, Any],
    stationarity: dict[str, Any],
    fit_windows: dict[str, Any],
    cut_scan: dict[str, Any],
) -> None:
    slow_final_primary = choose_slowest_primary(final_rows)
    slow_final_overall = max(ALL_AUTOCORR_OBSERVABLES, key=lambda obs: final_rows[obs]["tau_int_saved"])
    blocking_pass_all = all(blocking[obs]["plateau"] for obs in ALL_AUTOCORR_OBSERVABLES)
    max_sweep = max(thermal[init]["max_sweep"] for init in ("zero", "uniform"))
    reaches_therm = max_sweep >= N_THERM_CHOSEN
    extends_past_therm = max_sweep > N_THERM_CHOSEN
    with AUDIT_PATH.open("w", encoding="ascii") as handle:
        handle.write("# QHO Monte Carlo checks check\n\n")
        handle.write("## Source-faithful choices\n\n")
        handle.write("- Raw metadata and source conventions use the quadratic QHO action `S = sum_i [ eta/2 y_i^2 + 1/(2 eta) (y_{i+1}-y_i)^2 ]`.\n")
        handle.write("- The single-site conditional distribution is Gaussian, so heatbath is the natural local update.\n")
        handle.write("- The production convention is heatbath + overrelaxation with n_over=5, i.e. one heatbath sweep plus five microcanonical/overrelaxation sweeps.\n\n")
        handle.write("## Public figures\n\n")
        handle.write("- fig_sampling_efficiency_ymean_beta5_nt400.png\n")
        handle.write("- fig_autocorrelation_primary_beta5_nt128.png\n")
        handle.write("- fig_thermalization_logbins_beta5_nt512.png\n\n")
        handle.write("- Final MC plot inventory contains only: fig_sampling_efficiency_ymean_beta5_nt400, fig_autocorrelation_primary_beta5_nt128, fig_thermalization_logbins_beta5_nt512.\n")
        handle.write("- Blocking and stationarity are retained only as processed numerical checks.\n\n")
        handle.write("## Algorithm comparison\n\n")
        handle.write("- Observables scanned for the slow mode: y_mean, y2_mean, dy2_mean.\n")
        handle.write("- Final update-comparison plot uses only y_mean, because it is the limiting primary observable in the final beta=5, N_t=128 production dataset.\n")
        handle.write("- Derived control: energy_ren.\n")
        handle.write("\n| update | tau_int(y_mean) MCS | tau_int(y2_mean) MCS | tau_int(dy2_mean) MCS | slow primary observable | tau_slow MCS | status |\n")
        handle.write("|---|---:|---:|---:|---|---:|---|\n")
        for update in ("metro", "heatbath", "hb-over"):
            obs = choose_slowest_primary(algorithm[update])
            row = algorithm[update][obs]
            handle.write(
                f"| {ALGORITHM_LONG_LABELS[update]} | "
                f"{format_float(algorithm[update]['y_mean']['tau_int_sweeps'])} | "
                f"{format_float(algorithm[update]['y2_mean']['tau_int_sweeps'])} | "
                f"{format_float(algorithm[update]['dy2_mean']['tau_int_sweeps'])} | "
                f"{obs} | {format_float(row['tau_int_sweeps'])} | {row['status']} |\n"
            )
        handle.write("- This comparison is qualitative; the main motivation for heatbath is the Gaussian conditional distribution and the benchmark prescription.\n\n")
        handle.write("- For the main relation, use the y_mean-only update comparison plot. The worst-case tau_slow values are kept only as a numerical check table.\n\n")
        handle.write("## Thermalization\n\n")
        handle.write("- Files used: zero/uniform hb-over checks at beta=5, N_t=512.\n")
        handle.write(f"- Maximum sweep available: {max_sweep}; chosen n_therm={N_THERM_CHOSEN}.\n")
        handle.write(f"- Final visual check: Marinari-style log-binned interval averages with base b={format_float(THERM_LOG_BIN_BASE)}.\n")
        handle.write("- Thermalization is checked at N_t=512, the largest beta=5 thermodynamic lattice in the dataset.\n")
        if extends_past_therm:
            handle.write("- The available data extend beyond the chosen thermalization cut.\n")
        elif reaches_therm:
            handle.write("- The available data reach the chosen thermalization cut but do not show a post-cut region.\n")
        else:
            handle.write("- The available data do not extend beyond the chosen thermalization cut; this plot must not be used as final evidence of post-thermalization convergence.\n")
        handle.write("- Figure observables: log-binned y_mean and y2_mean interval averages.\n\n")
        handle.write(f"- Interpretation: agreement of zero/uniform starts is a necessary but not sufficient thermalization check. y2_mean approaches the exact benchmark value and the two starts become compatible after a transient; y_mean is slower and is treated as a conservative check. Final check of n_therm={N_THERM_CHOSEN} also uses stationarity tests on production data.\n\n")
        handle.write("- Conservative recommendation: n_therm_true = max(2*n_therm0, 20*tau_int_slow).\n")
        handle.write(f"- n_therm0 = {format_float(cut_scan['n_therm0'])}; tau_int_slow = {format_float(cut_scan['tau_int_slow'])} MCS from {cut_scan['slow_observable']}; n_therm_true = {format_float(cut_scan['n_therm_true'])}.\n")
        handle.write(f"- n_therm=200000 compatible with this criterion: {cut_scan['reference_therm_compatible']}.\n")
        handle.write(f"- Production rerun recommended: {cut_scan['rerun_recommended']}.\n\n")
        handle.write("- Final beta=5 thermodynamic productions have been rerun with n_therm=400000. The n_therm=200000 reference run passed the first-cut compatibility check but was not used for the final production because the conservative rule requires 2*n_therm0.\n\n")
        handle.write("## Final production beta=5, N_t=128\n\n")
        handle.write(f"- Slowest primary observable: {slow_final_primary} with tau_int={format_float(final_rows[slow_final_primary]['tau_int_saved'])} saved measurements = {format_float(final_rows[slow_final_primary]['tau_int_sweeps'])} MCS.\n")
        handle.write(f"- Slowest observable including derived controls: {slow_final_overall} with tau_int={format_float(final_rows[slow_final_overall]['tau_int_saved'])} saved measurements = {format_float(final_rows[slow_final_overall]['tau_int_sweeps'])} MCS.\n")
        handle.write("\n| observable | tau_int saved | tau_int sweeps | W | status |\n")
        handle.write("|---|---:|---:|---:|---|\n")
        for obs in ALL_AUTOCORR_OBSERVABLES:
            row = final_rows[obs]
            handle.write(
                f"| {obs} | {format_float(row['tau_int_saved'])} | {format_float(row['tau_int_sweeps'])} | "
                f"{row['window']} | {row['status']} |\n"
            )
        handle.write("\n| observable | err at k=2000 | median err for larger k | relative difference | plateau-compatible |\n")
        handle.write("|---|---:|---:|---:|---|\n")
        for obs in ALL_AUTOCORR_OBSERVABLES:
            row = blocking[obs]
            handle.write(
                f"| {obs} | {format_float(row['err_2000'])} | {format_float(row['median_larger'])} | "
                f"{format_float(row['rel_diff'])} | {row['plateau']} |\n"
            )
        if blocking_pass_all:
            handle.write("- Blocking result: all scanned observables pass the 15% stability criterion at k=2000. The blocking plot is not part of the final plot inventory; use only the quantitative stability table.\n")
        else:
            handle.write("- Blocking result: at least one scanned observable fails the 15% stability criterion at k=2000. Do not claim a blocking plateau; the final block choice needs review.\n")
        handle.write("\n## Production stationarity check\n\n")
        handle.write(f"- Dataset: final production beta=5, N_t=128, already after n_therm={N_THERM_CHOSEN}.\n")
        handle.write("- Method: split saved measurements into two halves and estimate each half error from block means with block_size_saved=2000.\n")
        handle.write("\n| observable | mean first | err first | mean second | err second | z_diff | status |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---|\n")
        for obs in ALL_AUTOCORR_OBSERVABLES:
            row = stationarity[obs]
            handle.write(
                f"| {obs} | {format_float(row['mean_first'])} | {format_float(row['err_first'])} | "
                f"{format_float(row['mean_second'])} | {format_float(row['err_second'])} | "
                f"{format_float(row['z_diff'])} | {row['status']} |\n"
            )
        final_obs_ok = all(stationarity[obs]["status"] == "pass" for obs in ("y2_mean", "dy2_mean", "energy_ren"))
        if final_obs_ok:
            handle.write(f"- Stationarity result: y2_mean, dy2_mean, and energy_ren pass the half-chain test, supporting n_therm={N_THERM_CHOSEN} for the thermodynamic analysis.\n")
        else:
            handle.write("- Stationarity result: at least one final thermodynamic observable does not pass; production parameters need review.\n")
        handle.write("- No stationarity plot is kept in the final inventory; stationarity is documented only through the numerical z_diff tables.\n\n")
        handle.write("## Fit-window stationarity\n\n")
        handle.write("- Quantitative output: data/processed/production/qho_mc_stationarity_fit_windows_beta5.dat.\n")
        for window_name, obs in (("y2_fit", "y2_mean"), ("energy_fit", "energy_ren")):
            rows = [row for row in fit_windows[window_name]["rows"] if row["observable"] == obs]
            max_abs = max(abs(row["z_diff"]) for row in rows)
            all_pass = all(row["status"] == "pass" for row in rows)
            warnings = [row for row in rows if row["status"] == "warning"]
            failures = [row for row in rows if row["status"] == "fail"]
            handle.write(f"- {obs}: max |z_diff| = {format_float(max_abs)}; all selected points pass |z_diff| <= 2: {all_pass}.\n")
            handle.write("- warnings: " + (", ".join(f"N_t={row['Nt']} z={format_float(row['z_diff'])}" for row in warnings) if warnings else "none") + ".\n")
            handle.write("- failures: " + (", ".join(f"N_t={row['Nt']} z={format_float(row['z_diff'])}" for row in failures) if failures else "none") + ".\n")
    print(f"Wrote {AUDIT_PATH}")


def main() -> None:
    configure_style()
    PROD_DIR.mkdir(parents=True, exist_ok=True)
    algorithm, algorithm_raw = analyze_algorithm()
    final_rows, final_raw = analyze_final()
    therm_logbins, thermal_info = load_stage_logbins()
    thermal = {
        init: {
            "path": THERM_STAGE_FILES[init],
            "columns": thermal_info["columns"],
            "max_sweep": thermal_info["max_sweep"],
            "has_post_therm": thermal_info["has_post_therm"],
        }
        for init in THERM_STAGE_FILES
    }
    blocking = analyze_blocking(final_raw)
    stationarity = analyze_stationarity(final_raw)
    fit_windows = analyze_stationarity_fit_windows()
    nt512_rows, _ = analyze_nt512_tau()
    cut_scan = analyze_thermalization_cut_scan(therm_logbins, nt512_rows)

    plot_update_tau_ymean(algorithm)
    plot_primary_autocorr(final_rows)
    plot_thermalization_logbins(therm_logbins, cut_scan)

    write_tau_table(algorithm, final_rows)
    write_blocking_stability(blocking)
    write_stationarity(stationarity)
    write_stationarity_fit_windows(fit_windows)
    write_stationarity_fit_windows_summary(fit_windows)
    write_thermalization_logbins(therm_logbins)
    write_thermalization_cut_scan(cut_scan)
    write_observable_check(algorithm_raw, final_raw, thermal)
    write_algorithm_summary(algorithm, algorithm_raw)
    write_check(algorithm, final_rows, thermal, blocking, stationarity, fit_windows, cut_scan)


if __name__ == "__main__":
    main()
