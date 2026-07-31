#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/plotting/plot_effective_gaps.py
# Purpose: Create spectrum-summary figures from processed correlator data.
# It presents jackknife effective gaps and eta^2 continuum extrapolations against
# the exact level spacings of the dimensionless harmonic oscillator.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: imports and configuration
# -----------------------------------------------------------------------------

"""Generate final QHO spectrum plots from processed production data."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


PROD_DIR = Path("data/processed/production")
PLOT_DIR = Path("plots/spectrum")

DELTAE_NT400_PATH = PROD_DIR / "qho_spectrum_beta40_highstat_nt400_deltaE_jackknife.dat"
GAP_ESTIMATES_PATH = PROD_DIR / "qho_spectrum_beta40_highstat_gap_estimates.dat"

PLOT_DELTAE_BASENAME = "fig_effective_gaps_beta40_nt400"
PLOT_CONTINUUM_BASENAME = "fig_gap_continuum_summary_beta40_eta2"


def configure_style() -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.45)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "CMU Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
        "font.size": 14,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "black",
        "axes.labelsize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 9,
    })


def apply_panel_style(ax: plt.Axes) -> None:
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
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)


def boxed_legend(ax: plt.Axes, **kwargs):
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("facecolor", "white")
    kwargs.setdefault("framealpha", 1.0)
    kwargs.setdefault("edgecolor", "none")
    kwargs.setdefault("borderpad", 0.28)
    kwargs.setdefault("labelspacing", 0.25)
    kwargs.setdefault("handlelength", 1.8)
    kwargs.setdefault("handletextpad", 0.55)
    kwargs.setdefault("borderaxespad", 0.35)
    legend = ax.legend(**kwargs)
    legend.get_frame().set_linewidth(0.0)
    return legend


def adapt_full_width_figure(fig: plt.Figure) -> None:
    """Adapt a selected figure for inclusion at LaTeX text width."""
    fig.set_size_inches(8.0, 4.6, forward=True)
    for ax in fig.axes:
        ax.title.set_fontsize(12)
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.xaxis.get_offset_text().set_fontsize(10)
        ax.yaxis.get_offset_text().set_fontsize(10)
        ax.tick_params(which="major", labelsize=10, length=4.5, width=1.0)
        ax.tick_params(which="minor", length=3.0, width=0.8)
        for line in ax.lines:
            line.set_linewidth(max(line.get_linewidth(), 1.3))
            if line.get_marker() not in (None, "None", "", " "):
                line.set_markersize(4.5)
        for collection in ax.collections:
            widths = collection.get_linewidths()
            if len(widths):
                collection.set_linewidth(max(float(np.max(widths)), 1.0))
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
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


def save_figure(fig: plt.Figure, basename: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / f"{basename}.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Wrote {out}")
    plt.close(fig)


def read_deltae_nt400() -> pd.DataFrame:
    cols = [
        "lag", "tau",
        "delta_y", "delta_y_err",
        "delta_y2", "delta_y2_err",
        "delta_y3", "delta_y3_err",
        "delta_A", "delta_A_err",
    ]
    return pd.read_csv(DELTAE_NT400_PATH, comment="#", sep=r"\s+", names=cols)


def read_gap_estimates() -> pd.DataFrame:
    return pd.read_csv(GAP_ESTIMATES_PATH, comment="#", sep=r"\s+")


def finite_errorbar(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    **kwargs,
) -> None:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    ax.errorbar(
        x[mask],
        y[mask],
        yerr=yerr[mask],
        capsize=0,
        **kwargs,
    )


def symmetric_ylim(ax: plt.Axes, values: np.ndarray, errors: np.ndarray, floor: float) -> None:
    finite = np.isfinite(values) & np.isfinite(errors)
    if not np.any(finite):
        ax.set_ylim(-floor, floor)
        return
    vmax = float(np.max(np.abs(values[finite]) + errors[finite]))
    limit = max(floor, 1.18 * vmax)
    ax.set_ylim(-limit, limit)


def plot_deltae_nt400() -> None:
    df = read_deltae_nt400()

    channels = [
        {
            "key": "y",
            "value": "delta_y",
            "error": "delta_y_err",
            "label": r"$y\rightarrow\Delta_1$",
            "exact": 1.0,
        },
        {
            "key": "y2",
            "value": "delta_y2",
            "error": "delta_y2_err",
            "label": r"$y_c^2\rightarrow\Delta_2$",
            "exact": 2.0,
        },
        {
            "key": "y3",
            "value": "delta_y3",
            "error": "delta_y3_err",
            "label": r"$y_c^3\rightarrow\Delta_1$",
            "exact": 1.0,
        },
        {
            "key": "A",
            "value": "delta_A",
            "error": "delta_A_err",
            "label": r"$A=y^3-\frac{3}{2}y\rightarrow\Delta_3$",
            "exact": 3.0,
        },
    ]

    colors = sns.color_palette("deep", n_colors=4)
    color_map = {ch["key"]: color for ch, color in zip(channels, colors)}

    fig = plt.figure(figsize=(12.4, 5.2), constrained_layout=True)
    grid = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[1.45, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.0,
        hspace=0.0,
    )

    ax_main = fig.add_subplot(grid[:, 0])
    ax_d3 = fig.add_subplot(grid[0, 1])
    ax_d2 = fig.add_subplot(grid[1, 1], sharex=ax_d3)
    ax_d1 = fig.add_subplot(grid[2, 1], sharex=ax_d3)

    for ch in channels:
        d = df[(df["tau"] >= 0.0) & (df["tau"] <= 2.5)]
        finite_errorbar(
            ax_main,
            d["tau"].to_numpy(float),
            d[ch["value"]].to_numpy(float),
            d[ch["error"]].to_numpy(float),
            fmt="o",
            ms=4.5,
            lw=1.0,
            elinewidth=0.85,
            color=color_map[ch["key"]],
            label=ch["label"],
            zorder=3,
        )

    for gap in (1.0, 2.0, 3.0):
        ax_main.axhline(gap, color="0.35", ls="--", lw=0.9, alpha=0.70, zorder=1)

    ax_main.set_xlim(0.0, 2.5)
    ax_main.set_ylim(0.0, 4.0)
    ax_main.set_xlabel(r"$n\eta$")
    ax_main.set_ylabel(r"$\Delta(n)$")
    ax_main.set_title(r"$\beta=40,\quad N_t=400$")
    apply_panel_style(ax_main)
    boxed_legend(ax_main, loc="lower right", fontsize=11)

    deviation_specs = [
        (
            ax_d3,
            "delta_A",
            "delta_A_err",
            3.0,
            r"$\Delta E_3(n)-3$",
            color_map["A"],
            0.070,
        ),
        (
            ax_d2,
            "delta_y2",
            "delta_y2_err",
            2.0,
            r"$\Delta E_2(n)-2$",
            color_map["y2"],
            0.016,
        ),
        (
            ax_d1,
            "delta_y",
            "delta_y_err",
            1.0,
            r"$\Delta E_1(n)-1$",
            color_map["y"],
            0.006,
        ),
    ]

    for ax, val, err, exact, label, color, floor in deviation_specs:
        d = df[(df["tau"] >= 0.0) & (df["tau"] <= 1.5)]
        x = d["tau"].to_numpy(float)
        y = d[val].to_numpy(float) - exact
        yerr = d[err].to_numpy(float)

        finite_errorbar(
            ax,
            x,
            y,
            yerr,
            fmt="o",
            ms=4.5,
            lw=1.0,
            elinewidth=0.80,
            color=color,
            label=label,
            zorder=3,
        )
        ax.axhline(0.0, color="0.35", ls="--", lw=0.9, alpha=0.75, zorder=1)
        ax.set_xlim(0.0, 1.5)
        symmetric_ylim(ax, y, yerr, floor)
        apply_panel_style(ax)
        boxed_legend(ax, loc="lower left", fontsize=11)

    ax_d3.tick_params(labelbottom=False)
    ax_d2.tick_params(labelbottom=False)
    ax_d1.set_xlabel(r"$n\eta$")

    adapt_full_width_figure(fig)
    save_figure(fig, PLOT_DELTAE_BASENAME)


def plot_continuum() -> None:
    df = read_gap_estimates()

    channels = [
        ("y", r"$\Delta_1$", -0.00035),
        ("y2", r"$\Delta_2$", 0.0),
        ("A", r"$\Delta_3$", 0.00035),
    ]

    colors = sns.color_palette("deep", n_colors=3)

    fig, ax = plt.subplots(figsize=(6.8, 4.7), constrained_layout=True)

    for (channel, label, offset), color in zip(channels, colors):
        subset = df[df["channel"] == channel].copy()
        subset = subset.sort_values("eta2")
        x = subset["eta2"].to_numpy(float) + offset
        y = subset["gap_minus_exact"].to_numpy(float)
        yerr = subset["gap_err"].to_numpy(float)

        finite_errorbar(
            ax,
            x,
            y,
            yerr,
            fmt="o",
            ms=4.2,
            lw=1.0,
            elinewidth=0.90,
            color=color,
            label=label,
            zorder=3,
        )

    ax.axhline(
        0.0,
        color="0.35",
        ls="--",
        lw=0.9,
        alpha=0.75,
        label=r"$\Delta_{\rm exact}$",
        zorder=1,
    )
    ax.set_xlabel(r"$\eta^2$")
    ax.set_ylabel(r"$\Delta(\eta)-\Delta_{\rm exact}$")
    ax.set_title(r"$\beta=40$")
    apply_panel_style(ax)
    boxed_legend(ax, loc="lower left", fontsize=11)

    adapt_three_quarter_width_figure(fig)
    save_figure(fig, PLOT_CONTINUUM_BASENAME)


def main() -> int:
    configure_style()
    plot_deltae_nt400()
    plot_continuum()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
