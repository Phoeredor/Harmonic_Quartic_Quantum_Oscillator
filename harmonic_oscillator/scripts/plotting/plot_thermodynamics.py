#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/plotting/plot_thermodynamics.py
# Purpose: Create final thermodynamic figures for the repository summary.
# Processed blocked estimates, exact thermal benchmarks, and eta^2 continuum fits
# are rendered together with representative periodic Euclidean paths.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: imports and configuration
# -----------------------------------------------------------------------------

"""Generate final report figures from existing QHO_PIMC data files."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

try:
    import seaborn as sns
except ImportError:
    sns = None


PLOT_DIR = Path("plots")
PATH_PLOT_DIR = PLOT_DIR / "euclidean_path"
THERMO_PLOT_DIR = PLOT_DIR / "thermodynamics"
BETA5_EXACT = 0.50678365490630423109601990090301278422429144642918728442862089901745650460541947924441402265025547793848275957445677249299820175811025952269306848402838297584202792535447904021153593340776311348866982575120083379349522116932394953700083823923297405805454594
BETA5_SUMMARY_PATH = Path("data/processed/production/qho_thermo_beta5_summary.dat")
BETA5_FITS_PATH = Path("data/processed/production/qho_thermo_beta5_fits.dat")
BETA5_RECOMMENDATION_PATH = Path("data/processed/production/qho_thermo_beta5_recommendation.md")
BETA5_VIRIAL_PATH = Path("data/processed/production/qho_thermo_beta5_virial.dat")
BETA5_KDIV_PATH = Path("data/processed/production/qho_thermo_beta5_kdiv.dat")
BETA5_Y2_ZSCORE_PATH = Path("data/processed/production/qho_thermo_beta5_y2_zscore.dat")
BETA5_OBSERVABLE_SPECS = {
    "y2_mean": {
        "label": r"$\langle y^2 \rangle$",
        "ylabel": r"$\langle y^2 \rangle$",
        "mean_col": 6,
        "error_col": 7,
        "basename": "fig_position_variance_continuum_beta5_eta2",
    },
    "energy_ren": {
        "label": r"$H_{\rm ren}$",
        "ylabel": r"$H_{\rm ren}$",
        "mean_col": 10,
        "error_col": 11,
        "basename": "fig_renormalized_energy_continuum_beta5_eta2",
    },
}


def warn(message: str) -> None:
    print(f"Warning: {message}")


def configure_report_style() -> None:
    if sns is not None:
        sns.set_theme(style="ticks", context="paper", font_scale=1.6)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "CMU Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
        "font.size": 15,
        "axes.labelsize": 18,
        "axes.titlesize": 15,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "black",
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def apply_selected_large_fonts(fig: plt.Figure) -> None:
    """Adapt selected figures for inclusion at 0.49\textwidth."""
    fig.set_size_inches(6.2, 4.8, forward=True)
    for ax in fig.axes:
        ax.title.set_fontsize(20)
        ax.xaxis.label.set_fontsize(24)
        ax.yaxis.label.set_fontsize(24)
        ax.xaxis.get_offset_text().set_fontsize(20)
        ax.yaxis.get_offset_text().set_fontsize(20)
        ax.tick_params(axis="both", which="both", labelsize=20)
        for text in ax.texts:
            text.set_fontsize(20)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(18)
            for handle in legend.legend_handles:
                if hasattr(handle, "set_linewidth"):
                    handle.set_linewidth(2.0)
                if hasattr(handle, "set_markersize"):
                    handle.set_markersize(5.5)
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
    for text in fig.texts:
        text.set_fontsize(20)


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
    for text in fig.texts:
        text.set_fontsize(13)


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


def read_numeric_table_with_comments(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    if data.size == 0:
        raise ValueError(f"{path}: no numeric rows found")
    return np.atleast_2d(data)


def save_figure(
    fig: plt.Figure,
    basename: str,
    generated: list[Path],
    output_dir: Path = PLOT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{basename}.png"
    fig.savefig(output, bbox_inches="tight", dpi=300)
    generated.append(output)
    print(f"Wrote {output}")
    plt.close(fig)


def apply_style(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in")


def apply_report_style(ax: plt.Axes) -> None:
    ax.grid(
        True,
        which="major",
        axis="both",
        color="0.70",
        alpha=0.55,
        linestyle=":",
        linewidth=0.8,
        zorder=0,
    )
    ax.tick_params(
        direction="in",
        which="major",
        top=True,
        right=True,
        width=1.0,
        length=4.8,
        colors="black",
        labelsize=15,
    )
    ax.tick_params(
        direction="in",
        which="minor",
        top=True,
        right=True,
        width=0.8,
        length=3.2,
        colors="black",
    )
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def report_legend(ax: plt.Axes, **kwargs):
    kwargs.setdefault("fontsize", 9)
    kwargs.setdefault("framealpha", 1.0)
    kwargs.setdefault("borderpad", 0.30)
    kwargs.setdefault("labelspacing", 0.25)
    kwargs.setdefault("handlelength", 1.8)
    kwargs.setdefault("handletextpad", 0.55)
    kwargs.setdefault("borderaxespad", 0.35)
    kwargs.setdefault("markerscale", 0.85)
    legend = ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        **kwargs,
    )
    legend.get_frame().set_linewidth(0.0)
    return legend


def read_beta5_summary(path: Path) -> np.ndarray:
    data = read_numeric_table_with_comments(path)
    if data.shape[1] < 17:
        raise ValueError(f"{path}: expected at least 17 columns, found {data.shape[1]}")
    return data


def read_beta5_scan_a(path: Path) -> dict[str, np.ndarray]:
    rows: dict[str, list[list[float]]] = {"y2_mean": [], "energy_ren": []}
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 20 or fields[0] != "scanA":
                continue
            observable = fields[1]
            if observable not in rows:
                continue
            rows[observable].append([
                float(fields[2]),
                float(fields[3]),
                float(fields[4]),
                float(fields[5]),
                float(fields[6]),
                float(fields[8]),
                float(fields[9]),
                float(fields[14]),
                float(fields[16]),
                float(fields[17]),
            ])
    return {key: np.asarray(value, dtype=float) for key, value in rows.items()}


def read_beta5_selected_fits(path: Path) -> dict[str, dict[str, float | int | str]]:
    selected: dict[str, dict[str, float | int | str]] = {}
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("# final_selected "):
                continue
            fields = stripped[1:].split()
            if len(fields) != 15:
                raise ValueError(
                    f"{path}: malformed final_selected row, expected 15 fields, found {len(fields)}"
                )
            observable = fields[1]
            if observable not in BETA5_OBSERVABLE_SPECS:
                continue
            eta_min = float(fields[5])
            eta_max = float(fields[6])
            nt_min = int(fields[3])
            nt_max = int(fields[4])
            selected[observable] = {
                "scan": fields[2],
                "nt_min": nt_min,
                "nt_max": nt_max,
                "eta_min": eta_min,
                "eta_max": eta_max,
                "eta2_min": eta_min * eta_min,
                "eta2_max": eta_max * eta_max,
                "n_points": int(fields[7]),
                "A": float(fields[8]),
                "A_error": float(fields[9]),
                "B": float(fields[10]),
                "B_error": float(fields[11]),
                "chi2_red": float(fields[12]),
                "exact": float(fields[13]),
                "A_minus_exact": float(fields[14]),
            }
    missing = [obs for obs in BETA5_OBSERVABLE_SPECS if obs not in selected]
    if missing:
        raise ValueError(
            f"{path}: missing final_selected entries for {', '.join(missing)}; "
            "rerun scripts/analysis/analyze_thermo_beta5.py before plotting"
        )
    return selected


def beta5_plot_specs(
    selected_fits: dict[str, dict[str, float | int | str]],
) -> dict[str, dict[str, object]]:
    specs: dict[str, dict[str, object]] = {}
    for observable, base_spec in BETA5_OBSERVABLE_SPECS.items():
        fit = selected_fits[observable]
        spec = dict(base_spec)
        spec.update(fit)
        spec["legend"] = (
            f"selected fit, Nt={int(fit['nt_min'])}..{int(fit['nt_max'])}, "
            f"chi2_red={float(fit['chi2_red']):.3f}"
        )
        specs[observable] = spec
    return specs


def plot_beta5_observable(
    generated: list[Path],
    data: np.ndarray,
    observable: str,
    spec: dict[str, object],
) -> None:
    eta2 = data[:, 2]
    nt = data[:, 3].astype(int)
    mean = data[:, int(spec["mean_col"])]
    error = data[:, int(spec["error_col"])]
    selected = (
        (eta2 >= float(spec["eta2_min"]))
        & (eta2 <= float(spec["eta2_max"]))
        & (nt >= int(spec["nt_min"]))
        & (nt <= int(spec["nt_max"]))
    )
    excluded = ~selected

    fit_x = np.linspace(float(spec["eta2_min"]), float(spec["eta2_max"]), 200)
    fit_y = float(spec["A"]) + float(spec["B"]) * fit_x
    y_low = float(np.min(mean - error))
    y_high = float(np.max(mean + error))
    y_pad = max(0.004, 0.08 * (y_high - y_low))
    point_color = "#2ca02c" if observable == "y2_mean" else "#1f77b4"
    excluded_color = "0.50"

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.errorbar(
        eta2[excluded],
        mean[excluded],
        yerr=error[excluded],
        fmt="o",
        ms=3.8,
        mfc="white",
        mec=excluded_color,
        mew=0.9,
        capsize=0,
        elinewidth=0.85,
        color=excluded_color,
        ecolor=excluded_color,
        alpha=0.95,
        label="_nolegend_",
        zorder=2,
    )
    ax.errorbar(
        eta2[selected],
        mean[selected],
        yerr=error[selected],
        fmt="o",
        ms=4.2,
        mfc=point_color,
        mec=point_color,
        mew=0.9,
        capsize=0,
        elinewidth=0.9,
        color=point_color,
        ecolor=point_color,
        label="_nolegend_",
        zorder=3,
    )
    ax.axhline(BETA5_EXACT, color="black", lw=1.2, ls=":", label="Analytical", zorder=1)
    ax.plot(
        fit_x,
        fit_y,
        color="red",
        lw=1.8,
        ls="-.",
        label=r"Fit: $a+b\eta^2$",
        zorder=4,
    )

    ax.set_xlim(-0.03, float(np.max(eta2)) * 1.04)
    ax.set_ylim(y_low - y_pad, y_high + y_pad)
    ax.set_xlabel(r"$\eta^2$", fontsize=18)
    ax.set_ylabel(str(spec["ylabel"]), fontsize=18)
    apply_report_style(ax)
    report_legend(ax, loc="best")
    chi2_precision = 3
    ax.text(
        0.03,
        0.04,
        rf"$\chi^2_{{\rm red}}={float(spec['chi2_red']):.{chi2_precision}f}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
    )

    inset_bounds = [0.45, 0.27, 0.52, 0.50]
    inset = ax.inset_axes(inset_bounds)
    inset.errorbar(
        eta2[selected],
        mean[selected],
        yerr=error[selected],
        fmt="o",
        ms=3.4,
        mfc=point_color,
        mec=point_color,
        mew=0.8,
        capsize=0,
        elinewidth=0.8,
        color=point_color,
        ecolor=point_color,
        zorder=3,
    )
    inset.axhline(BETA5_EXACT, color="black", lw=1.0, ls=":", zorder=1)
    inset.plot(fit_x, fit_y, color="red", lw=1.4, ls="-.", zorder=4)
    inset_y_values = np.concatenate((
        mean[selected] - error[selected],
        mean[selected] + error[selected],
        fit_y,
        np.asarray([BETA5_EXACT], dtype=float),
    ))
    inset_y_low = float(np.min(inset_y_values))
    inset_y_high = float(np.max(inset_y_values))
    inset_y_pad = max(0.0018, 0.16 * (inset_y_high - inset_y_low))
    inset_x_min = float(spec["eta2_min"])
    inset_x_max = float(spec["eta2_max"])
    inset_x_pad = 0.04 * (inset_x_max - inset_x_min)
    inset.set_xlim(inset_x_min - inset_x_pad, inset_x_max + inset_x_pad)
    inset.set_ylim(inset_y_low - inset_y_pad, inset_y_high + inset_y_pad)
    apply_report_style(inset)
    inset.tick_params(labelsize=9, length=3.0, width=0.8)
    for label in inset.get_xticklabels() + inset.get_yticklabels():
        label.set_fontsize(9)

    apply_selected_large_fonts(fig)
    inset.tick_params(axis="both", which="major", labelsize=17, length=4.0, width=1.0)
    inset.xaxis.set_major_locator(MaxNLocator(4))
    inset.yaxis.set_major_locator(MaxNLocator(4))
    inset.patch.set_facecolor("white")
    inset.patch.set_alpha(0.90)
    for line in inset.lines:
        line.set_linewidth(max(line.get_linewidth(), 1.3))
        if line.get_marker() not in (None, "None", "", " "):
            line.set_markersize(5.0)
    for collection in inset.collections:
        widths = collection.get_linewidths()
        if len(widths):
            collection.set_linewidth(max(float(np.max(widths)), 1.3))
    save_figure(fig, str(spec["basename"]), generated, THERMO_PLOT_DIR)


def plot_thermo_beta5_fit_windows(
    generated: list[Path],
    scan_a: dict[str, np.ndarray],
    selected_fits: dict[str, dict[str, float | int | str]],
) -> None:
    if not scan_a or any(scan_a.get(key, np.empty((0, 0))).size == 0 for key in BETA5_OBSERVABLE_SPECS):
        warn("missing beta=5 Scan A fit-window rows; skipping fit-window plot")
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    colors = {"y2_mean": "#2ca02c", "energy_ren": "#1f77b4"}
    labels = {
        "y2_mean": r"$\langle y^2 \rangle\ \mathrm{selected}$",
        "energy_ren": r"$H_{\rm ren}\ \mathrm{selected}$",
    }
    for observable in ("y2_mean", "energy_ren"):
        rows = scan_a[observable]
        order = np.argsort(rows[:, 3])
        eta_min = rows[order, 3]
        intercept = rows[order, 5]
        intercept_error = rows[order, 6]
        ax.errorbar(
            eta_min,
            intercept,
            yerr=intercept_error,
            fmt="o",
            ms=4.0,
            mfc="white",
            mec=colors[observable],
            mew=0.9,
            capsize=0,
            elinewidth=0.85,
            lw=1.0,
            color=colors[observable],
            ecolor=colors[observable],
            alpha=0.78,
            label="_nolegend_",
            zorder=3,
        )
        selected_fit = selected_fits[observable]
        selected_eta_min = float(selected_fit["eta_min"])
        matching = np.where(np.isclose(rows[:, 3], selected_eta_min, rtol=0.0, atol=1.0e-12))[0]
        if matching.size != 1:
            raise ValueError(
                f"{BETA5_FITS_PATH}: selected {observable} eta_min={selected_eta_min:.12g} "
                f"does not match exactly one scanA row"
            )
        ax.errorbar(
            selected_eta_min,
            float(selected_fit["A"]),
            yerr=float(selected_fit["A_error"]),
            fmt="o",
            ms=5.0,
            mfc=colors[observable],
            mec=colors[observable],
            mew=0.9,
            capsize=0,
            elinewidth=1.0,
            color=colors[observable],
            ecolor=colors[observable],
            label=labels[observable],
            zorder=5,
        )
    ax.axhline(BETA5_EXACT, color="black", lw=1.2, ls=":", label=r"$\mathrm{Analytical}$", zorder=1)
    ax.set_xlabel(r"$\eta_{\rm min}$", fontsize=18)
    ax.set_ylabel(r"$a$", fontsize=18)
    ax.set_xlim(0.0, 0.105)
    ax.set_ylim(0.500, 0.510)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: rf"${value:.2f}$"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: rf"${value:.3f}$"))
    apply_report_style(ax)
    report_legend(ax, loc="lower right", ncol=1)
    adapt_three_quarter_width_figure(fig)
    save_figure(fig, "fig_continuum_fit_window_stability_beta5", generated, THERMO_PLOT_DIR)


def plot_thermo_beta5_virial(generated: list[Path]) -> None:
    if not BETA5_VIRIAL_PATH.exists():
        warn(f"missing beta=5 virial table; skipping virial plot: {BETA5_VIRIAL_PATH}")
        return

    data = read_numeric_table_with_comments(BETA5_VIRIAL_PATH)
    if data.shape[1] < 16:
        raise ValueError(f"{BETA5_VIRIAL_PATH}: expected at least 16 columns, found {data.shape[1]}")

    beta = float(data[0, 0])
    eta = data[:, 1]
    v_mean = data[:, 4]
    v_err = data[:, 5]
    k_mean = data[:, 6]
    k_err = data[:, 7]
    analytical = 0.25 / math.tanh(0.5 * beta)
    order = np.argsort(eta)

    eta = eta[order]
    v_mean = v_mean[order]
    v_err = v_err[order]
    k_mean = k_mean[order]
    k_err = k_err[order]

    potential_color = "#7b3294"
    kinetic_color = "#2ca02c"
    y_low = float(np.min(np.concatenate((v_mean - v_err, k_mean - k_err, [analytical]))))
    y_high = float(np.max(np.concatenate((v_mean + v_err, k_mean + k_err, [analytical]))))
    y_pad = max(0.012, 0.10 * (y_high - y_low))

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.errorbar(
        eta,
        v_mean,
        yerr=v_err,
        fmt="o",
        ms=5.5,
        mfc=potential_color,
        mec=potential_color,
        mew=0.9,
        capsize=0,
        elinewidth=0.9,
        color=potential_color,
        ecolor=potential_color,
        linestyle="none",
        label=r"$V=\langle y^2\rangle/2$",
        zorder=3,
    )
    ax.errorbar(
        eta,
        k_mean,
        yerr=k_err,
        fmt="o",
        ms=5.5,
        mfc=kinetic_color,
        mec=kinetic_color,
        mew=0.9,
        capsize=0,
        elinewidth=0.9,
        color=kinetic_color,
        ecolor=kinetic_color,
        linestyle="none",
        label=r"$K_{\rm ren}$",
        zorder=2,
    )
    ax.axhline(
        analytical,
        color="black",
        lw=1.2,
        ls=":",
        label="Analytical",
        zorder=1,
    )
    ax.set_xlim(0.0, float(np.max(eta)) * 1.04)
    ax.set_ylim(y_low - y_pad, y_high + y_pad)
    ax.set_xlabel(r"$\eta$", fontsize=18)
    ax.set_ylabel(r"$V,\ K_{\rm ren}$", fontsize=18)
    apply_report_style(ax)
    report_legend(ax, loc="lower left", ncol=1, framealpha=0.7)

    inset = None
    inset_mask = eta <= 0.2 + 1.0e-12
    if np.count_nonzero(inset_mask) >= 3:
        inset = ax.inset_axes([0.50, 0.47, 0.47, 0.48])
        inset.errorbar(
            eta[inset_mask],
            v_mean[inset_mask],
            yerr=v_err[inset_mask],
            fmt="o",
            ms=3.2,
            mfc=potential_color,
            mec=potential_color,
            mew=0.8,
            capsize=0,
            elinewidth=0.75,
            color=potential_color,
            ecolor=potential_color,
            linestyle="none",
            zorder=3,
        )
        inset.errorbar(
            eta[inset_mask],
            k_mean[inset_mask],
            yerr=k_err[inset_mask],
            fmt="o",
            ms=3.0,
            mfc=kinetic_color,
            mec=kinetic_color,
            mew=0.8,
            capsize=0,
            elinewidth=0.75,
            color=kinetic_color,
            ecolor=kinetic_color,
            linestyle="none",
            zorder=2,
        )
        inset.axhline(analytical, color="black", lw=1.0, ls=":", zorder=1)
        inset_y_low = float(np.min(np.concatenate((
            v_mean[inset_mask] - v_err[inset_mask],
            k_mean[inset_mask] - k_err[inset_mask],
            [analytical],
        ))))
        inset_y_high = float(np.max(np.concatenate((
            v_mean[inset_mask] + v_err[inset_mask],
            k_mean[inset_mask] + k_err[inset_mask],
            [analytical],
        ))))
        inset_y_pad = max(0.006, 0.14 * (inset_y_high - inset_y_low))
        inset.set_xlim(0.0, 0.205)
        inset.set_ylim(inset_y_low - inset_y_pad, inset_y_high + inset_y_pad)
        apply_report_style(inset)
        inset.tick_params(labelsize=9, length=3.0, width=0.8)
        for label in inset.get_xticklabels() + inset.get_yticklabels():
            label.set_fontsize(9)

    apply_selected_large_fonts(fig)
    if inset is not None:
        inset.tick_params(axis="both", which="major", labelsize=17, length=4.0, width=1.0)
        inset.xaxis.set_major_locator(MaxNLocator(4))
        inset.yaxis.set_major_locator(MaxNLocator(4))
        inset.patch.set_facecolor("white")
        inset.patch.set_alpha(0.90)
        for line in inset.lines:
            line.set_linewidth(max(line.get_linewidth(), 1.3))
            if line.get_marker() not in (None, "None", "", " "):
                line.set_markersize(5.0)
        for collection in inset.collections:
            widths = collection.get_linewidths()
            if len(widths):
                collection.set_linewidth(max(float(np.max(widths)), 1.3))
    save_figure(fig, "fig_virial_estimator_check_beta5_eta", generated, THERMO_PLOT_DIR)


def plot_thermo_beta5_kdiv(generated: list[Path]) -> None:
    if not BETA5_KDIV_PATH.exists():
        warn(f"missing beta=5 Kdiv table; skipping Kdiv plot: {BETA5_KDIV_PATH}")
        return

    data = read_numeric_table_with_comments(BETA5_KDIV_PATH)
    if data.shape[1] < 9:
        raise ValueError(f"{BETA5_KDIV_PATH}: expected at least 9 columns, found {data.shape[1]}")

    eta = data[:, 1]
    kdiv_mean = data[:, 4]
    kdiv_err = data[:, 5]
    order = np.argsort(eta)
    eta = eta[order]
    kdiv_mean = kdiv_mean[order]
    kdiv_err = kdiv_err[order]

    kinetic_color = "#2ca02c"
    y_low = float(np.min(kdiv_mean - kdiv_err))
    y_high = float(max(0.0, np.max(kdiv_mean + kdiv_err)))
    y_pad = max(0.6, 0.08 * (y_high - y_low))

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.errorbar(
        eta,
        kdiv_mean,
        yerr=kdiv_err,
        fmt="o",
        ms=4.2,
        mfc=kinetic_color,
        mec=kinetic_color,
        mew=0.9,
        capsize=0,
        elinewidth=0.9,
        color=kinetic_color,
        ecolor=kinetic_color,
        linestyle="none",
        label=r"$K_{\rm div}$",
        zorder=3,
    )
    ax.axhline(0.0, color="black", lw=1.2, ls=":", label="_nolegend_", zorder=1)
    ax.set_xlim(0.0, float(np.max(eta)) * 1.04)
    ax.set_ylim(y_low - y_pad, y_high + y_pad)
    ax.set_xlabel(r"$\eta$", fontsize=18)
    ax.set_ylabel(r"$-\langle\Delta y^2\rangle/(2\eta^2)$", fontsize=18)
    apply_report_style(ax)
    report_legend(ax, loc="lower right", ncol=1)
    apply_selected_large_fonts(fig)
    save_figure(fig, "fig_divergent_kinetic_term_beta5_eta", generated, THERMO_PLOT_DIR)


def plot_thermo_beta5_y2_zscore(generated: list[Path]) -> None:
    if not BETA5_Y2_ZSCORE_PATH.exists():
        warn(f"missing beta=5 y2 z-score table; skipping z-score plot: {BETA5_Y2_ZSCORE_PATH}")
        return

    data = read_numeric_table_with_comments(BETA5_Y2_ZSCORE_PATH)
    if data.shape[1] < 18:
        raise ValueError(f"{BETA5_Y2_ZSCORE_PATH}: expected at least 18 columns, found {data.shape[1]}")

    eta = data[:, 1]
    abs_z_cont = data[:, 12]
    selected = data[:, 15].astype(int) == 1
    coarse = data[:, 16].astype(int) == 1
    fine = data[:, 17].astype(int) == 1
    point_color = "#7b3294"
    y_high = float(max(3.0, np.max(abs_z_cont)))
    y_pad = max(0.35, 0.08 * y_high)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for yref, style, linewidth, alpha, label in (
        (1.0, ":", 0.9, 0.55, "_nolegend_"),
        (2.0, ":", 0.9, 0.55, "_nolegend_"),
        (3.0, "--", 1.15, 0.85, r"$|z|=3$"),
    ):
        ax.axhline(yref, color="0.25", lw=linewidth, ls=style, alpha=alpha, label=label, zorder=1)

    ax.plot(
        eta[coarse],
        abs_z_cont[coarse],
        "o",
        ms=4.4,
        mfc="white",
        mec=point_color,
        mew=0.95,
        linestyle="none",
        label=r"$\eta>0.2$",
        zorder=3,
    )
    ax.plot(
        eta[fine],
        abs_z_cont[fine],
        "^",
        ms=4.4,
        mfc="white",
        mec=point_color,
        mew=0.95,
        linestyle="none",
        alpha=0.85,
        label=r"$\eta \ll 1$",
        zorder=3,
    )
    ax.plot(
        eta[selected],
        abs_z_cont[selected],
        "o",
        ms=4.6,
        mfc=point_color,
        mec=point_color,
        mew=0.95,
        linestyle="none",
        label="fit window",
        zorder=4,
    )
    ax.set_xlim(0.0, float(np.max(eta)) * 1.04)
    ax.set_ylim(0.0, y_high + y_pad)
    ax.set_xlabel(r"$\eta$", fontsize=18)
    ax.set_ylabel(r"$|z|$", fontsize=18)
    apply_report_style(ax)

    inset_mask = eta <= 0.3 + 1.0e-12
    inset = ax.inset_axes([0.09, 0.55, 0.42, 0.37])
    for yref, style, linewidth, alpha in (
        (1.0, ":", 0.75, 0.50),
        (2.0, ":", 0.75, 0.50),
        (3.0, "--", 0.95, 0.80),
    ):
        inset.axhline(yref, color="0.25", lw=linewidth, ls=style, alpha=alpha, zorder=1)
    inset.plot(
        eta[coarse & inset_mask],
        abs_z_cont[coarse & inset_mask],
        "o",
        ms=3.4,
        mfc="white",
        mec=point_color,
        mew=0.85,
        linestyle="none",
        zorder=3,
    )
    inset.plot(
        eta[fine & inset_mask],
        abs_z_cont[fine & inset_mask],
        "^",
        ms=3.4,
        mfc="white",
        mec=point_color,
        mew=0.85,
        linestyle="none",
        alpha=0.85,
        zorder=3,
    )
    inset.plot(
        eta[selected & inset_mask],
        abs_z_cont[selected & inset_mask],
        "o",
        ms=3.5,
        mfc=point_color,
        mec=point_color,
        mew=0.85,
        linestyle="none",
        zorder=4,
    )
    inset_y_high = float(max(3.0, np.max(abs_z_cont[inset_mask])))
    inset.set_xlim(0.0, 0.3)
    inset.set_ylim(0.0, inset_y_high + max(0.25, 0.10 * inset_y_high))
    apply_report_style(inset)
    inset.tick_params(labelsize=9, length=3.0, width=0.8)
    for label in inset.get_xticklabels() + inset.get_yticklabels():
        label.set_fontsize(9)

    report_legend(ax, loc="lower right", ncol=1)
    adapt_three_quarter_width_figure(fig)
    save_figure(fig, "fig_position_variance_pull_beta5_eta", generated, THERMO_PLOT_DIR)


def plot_thermo_beta5(generated: list[Path]) -> None:
    missing = [
        path
        for path in (BETA5_SUMMARY_PATH, BETA5_FITS_PATH, BETA5_RECOMMENDATION_PATH)
        if not path.exists()
    ]
    if missing:
        warn("missing beta=5 thermodynamic production inputs; skipping beta=5 report plots")
        for path in missing:
            warn(f"  missing {path}")
        return

    data = read_beta5_summary(BETA5_SUMMARY_PATH)
    scan_a = read_beta5_scan_a(BETA5_FITS_PATH)
    selected_fits = read_beta5_selected_fits(BETA5_FITS_PATH)
    specs = beta5_plot_specs(selected_fits)
    plot_beta5_observable(generated, data, "y2_mean", specs["y2_mean"])
    plot_beta5_observable(generated, data, "energy_ren", specs["energy_ren"])
    plot_thermo_beta5_fit_windows(generated, scan_a, selected_fits)
    plot_thermo_beta5_virial(generated)
    plot_thermo_beta5_kdiv(generated)
    plot_thermo_beta5_y2_zscore(generated)


def spectral_colormap():
    _ = sns
    return plt.get_cmap("cool")


def read_operator_summary(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 9:
                continue
            rows.append({
                "operator": fields[0],
                "nt": int(fields[1]),
                "eta": float(fields[2]),
                "eta2": float(fields[3]),
                "n_replica": int(fields[4]),
                "mean": float(fields[5]),
                "stderr": float(fields[6]),
                "exact": float(fields[7]),
                "diff": float(fields[8]),
            })
    return rows


def read_continuum_fit_section(path: Path) -> dict[str, dict[str, float]]:
    fits: dict[str, dict[str, float]] = {}
    in_fit_section = False
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("# continuum_fit"):
                in_fit_section = True
                continue
            if not in_fit_section or not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 6:
                continue
            fits[fields[0]] = {
                "exact": float(fields[1]),
                "gap0": float(fields[2]),
                "gap0_error": float(fields[3]),
                "slope": float(fields[4]),
                "diff": float(fields[5]),
            }
    return fits


def read_thermo_summary(path: Path) -> tuple[dict[str, str], np.ndarray, dict[str, dict[str, float]]]:
    metadata = parse_metadata(path)
    rows: list[list[float]] = []
    fits: dict[str, dict[str, float]] = {}
    in_fit_section = False
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("# continuum_fit"):
                in_fit_section = True
                continue
            if stripped.startswith("#"):
                continue
            fields = stripped.split()
            if in_fit_section:
                if len(fields) == 6:
                    fits[fields[0]] = {
                        "O0": float(fields[1]),
                        "O0_error": float(fields[2]),
                        "slope": float(fields[3]),
                        "exact": float(fields[4]),
                        "diff": float(fields[5]),
                    }
            elif len(fields) >= 14:
                rows.append([float(field) for field in fields[:14]])
    if not rows:
        raise ValueError(f"{path}: no thermodynamic continuum rows found")
    return metadata, np.asarray(rows, dtype=float), fits


def format_intercept_label(value: float, error: float) -> str:
    """Format the dense eta-scan intercept in compact report notation."""
    error_digits = int(round(error * 1.0e4))
    return f"A = {value:.5f}({error_digits:d})"


def plot_path_snapshots(generated: list[Path], beta: int = 5) -> None:
    inputs = [
        (r"$N_t=32$", Path(f"data/processed/qho_path_beta{beta}_nt32.dat")),
        (r"$N_t=256$", Path(f"data/processed/qho_path_beta{beta}_nt256.dat")),
        (r"$N_t=2048$", Path(f"data/processed/qho_path_beta{beta}_nt2048.dat")),
        (r"$N_t=4096$", Path(f"data/processed/qho_path_beta{beta}_nt4096.dat")),
    ]
    missing = [path for _, path in inputs if not path.exists()]
    if missing:
        warn("missing path snapshot inputs; skipping representative path plot")
        for path in missing:
            warn(f"  missing {path}")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    widths = [1.25, 1.15, 0.95, 0.85]
    cmap = plt.get_cmap("plasma")
    colors = [cmap(value) for value in np.linspace(0.08, 0.92, len(inputs))]
    for (label, path), width, color in zip(inputs, widths, colors):
        data = read_numeric_table_with_comments(path)
        ax.plot(data[:, 1], data[:, 2], lw=width, color=color, label=label)
    ax.set_xlabel(r"$\tau=j\eta$", fontsize=18)
    ax.set_ylabel(r"$y_j$", fontsize=18)
    apply_report_style(ax)
    legend = report_legend(ax, loc="upper left", fontsize=8, handlelength=1.5)
    for handle in legend.legend_handles:
        handle.set_linewidth(1.15)
    if beta == 5:
        adapt_three_quarter_width_figure(fig)
    save_figure(fig, f"fig_path_snapshots_beta{beta}", generated, PATH_PLOT_DIR)



def main() -> int:
    generated: list[Path] = []
    configure_report_style()

    plot_thermo_beta5(generated)
    plot_path_snapshots(generated, beta=5)

    print()
    print("Generated final plot files:")
    if generated:
        for path in generated:
            print(path)
    else:
        print("none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
