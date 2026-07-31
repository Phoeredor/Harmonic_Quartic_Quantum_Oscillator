#!/usr/bin/env python3
# Create compact relation plots linking lambda to continuum moments, position
# densities, and the lowest spectral gaps of the quartic oscillator.
"""Generate the minimal final anharmonic-oscillator relation plots."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


HIST_DIR = Path("data/raw/production/beta5_continuum_v2/histograms")
SPECTRUM_TABLE = Path("data/processed/spectrum/anharmonic_spectrum_scan_quick_gap_table.dat")
HIST_CHECK = Path("data/processed/production/anharmonic_beta5_v2_histogram_normalization_check.md")

DIST_OUT = Path("plots/distribution/fig_position_density_lambda_scan")
SPECTRUM_GAPS_OUT = Path("plots/spectrum/fig_excitation_gaps_vs_lambda")
SPECTRUM_RATIO_OUT = Path("plots/spectrum/fig_gap_ratio_vs_lambda")

HIST_TOL = 5.0e-3


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "CMU Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
        "font.size": 20,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "black",
        "axes.labelsize": 24,
        "axes.titlesize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
    })


def apply_panel_style(ax: plt.Axes) -> None:
    ax.grid(True, which="major", axis="both", color="0.70", alpha=0.55,
            linestyle=":", linewidth=0.8, zorder=0)
    ax.tick_params(direction="in", which="major", top=True, right=True,
                   width=1.0, length=4.8, colors="black")
    ax.tick_params(direction="in", which="minor", top=True, right=True,
                   width=0.8, length=3.2, colors="black")
    ax.minorticks_on()
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
                line.set_markersize(max(line.get_markersize(), 4.5))
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


def adapt_subfigure_032(fig: plt.Figure) -> None:
    """Adapt a single-panel plot for use as a 0.32 textwidth LaTeX subfigure."""
    fig.set_size_inches(3.2, 2.55, forward=True)
    for ax in fig.axes:
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.xaxis.get_offset_text().set_fontsize(8)
        ax.yaxis.get_offset_text().set_fontsize(8)
        ax.tick_params(which="major", labelsize=9, length=4.2, width=1.0)
        ax.tick_params(which="minor", length=2.8, width=0.8)
        for line in ax.lines:
            line.set_linewidth(max(line.get_linewidth(), 1.35))
            if line.get_marker() not in (None, "None", "", " "):
                line.set_markersize(max(line.get_markersize(), 4.8))
        for collection in ax.collections:
            widths = collection.get_linewidths()
            if len(widths):
                collection.set_linewidth(max(float(np.max(widths)), 1.05))
        for gridline in ax.get_xgridlines() + ax.get_ygridlines():
            gridline.set_color("0.75")
            gridline.set_alpha(0.35)
            gridline.set_linewidth(0.7)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(7.5)
            for handle in legend.legend_handles:
                if hasattr(handle, "set_linewidth"):
                    handle.set_linewidth(1.3)
                if hasattr(handle, "set_markersize"):
                    handle.set_markersize(4.8)


def save_figure(fig: plt.Figure, base: Path, generated: list[Path]) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    out = base.with_suffix(".png")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    generated.append(out)
    print(f"Wrote {out}")
    plt.close(fig)


def read_named_table(path: Path) -> np.ndarray:
    return np.genfromtxt(path, names=True, comments="#", dtype=None, encoding=None)


def lambda_from_hist(path: Path) -> float:
    match = re.search(r"lambda([0-9p]+)_nt512", path.name)
    if match is None:
        raise ValueError(f"cannot parse lambda from {path}")
    return float(match.group(1).replace("p", "."))


def load_histograms() -> list[dict[str, object]]:
    rows = []
    for path in sorted(HIST_DIR.glob("*_nt512_*_histogram.dat")):
        lam = lambda_from_hist(path)
        if lam <= 0.0:
            continue
        data = np.loadtxt(path, comments="#", ndmin=2)
        x = data[:, 0]
        density = data[:, 2] if data.shape[1] >= 3 else data[:, 1]
        if len(x) > 1:
            dy = float(np.mean(np.diff(x)))
        else:
            dy = np.nan
        integral = float(np.sum(density) * dy)
        rows.append({"lambda": lam, "path": path, "x": x, "density": density,
                     "dy": dy, "integral": integral})
    rows.sort(key=lambda row: float(row["lambda"]))
    return rows


def write_histogram_check(histograms: list[dict[str, object]]) -> None:
    HIST_CHECK.parent.mkdir(parents=True, exist_ok=True)
    bad = [row for row in histograms if abs(float(row["integral"]) - 1.0) > HIST_TOL]
    with HIST_CHECK.open("w", encoding="utf-8") as stream:
        stream.write("# Histogram normalization check\n\n")
        stream.write(f"Tolerance: `{HIST_TOL:g}` on `sum P(y) dy` for the plotted symmetrized density.\n\n")
        stream.write("| lambda | integral | deviation | status |\n")
        stream.write("|---:|---:|---:|:---|\n")
        for row in histograms:
            integral = float(row["integral"])
            deviation = integral - 1.0
            status = "OK" if abs(deviation) <= HIST_TOL else "WARNING"
            stream.write(f"| {float(row['lambda']):g} | {integral:.8f} | {deviation:.3e} | {status} |\n")
        stream.write("\n")
        if bad:
            stream.write(f"WARNING: {len(bad)} histogram(s) exceed the tolerance.\n")
        else:
            stream.write("All plotted histograms are normalized within tolerance.\n")


# Overlay normalized thermal position densities at selected interaction strengths.
def plot_distribution(generated: list[Path]) -> None:
    histograms = load_histograms()
    if len(histograms) != 12:
        print(f"Warning: expected 12 lambda>0 histograms, found {len(histograms)}")
    write_histogram_check(histograms)
    lambdas = np.array([float(row["lambda"]) for row in histograms])
    norm = Normalize(vmin=float(lambdas.min()), vmax=float(lambdas.max()))
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(3.2, 2.55), constrained_layout=True)
    for row in histograms:
        lam = float(row["lambda"])
        ax.plot(row["x"], row["density"], color=cmap(norm(lam)), lw=1.05, alpha=0.95)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$P(y)$")
    apply_panel_style(ax)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, pad=0.025)
    cbar.set_label(r"$\lambda$")
    cbar.ax.tick_params(direction="in", which="major", width=1.0, length=4.5,
                        labelsize=8, colors="black")
    adapt_subfigure_032(fig)
    cbar.set_label(r"$\lambda$", fontsize=11)
    for spine in cbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    save_figure(fig, DIST_OUT, generated)


# Present the first two excitation gaps and their ratio as lambda varies.
def plot_spectrum(generated: list[Path]) -> None:
    table = np.loadtxt(SPECTRUM_TABLE, comments="#", ndmin=2)
    table = table[table[:, 0] > 0.0]
    lam = table[:, 0]
    d1, e1, ed1 = table[:, 1], table[:, 2], table[:, 3]
    d2, e2, ed2 = table[:, 6], table[:, 7], table[:, 8]
    ratio = table[:, 11]
    ratio_ed = table[:, 12]
    ratio_err = ratio * np.sqrt((e2 / d2) ** 2 + (e1 / d1) ** 2)
    fig, ax = plt.subplots(figsize=(3.2, 2.55), constrained_layout=True)
    ax.errorbar(lam, d1, yerr=e1, fmt="o", ms=4.0, color="#1f77b4",
                ecolor="#1f77b4", capsize=0.0, label=r"$\Delta_1$ PIMC", zorder=4)
    ax.errorbar(lam, d2, yerr=e2, fmt="s", ms=4.0, color="#ff7f0e",
                ecolor="#ff7f0e", capsize=0.0, label=r"$\Delta_2$ PIMC", zorder=4)
    ax.plot(lam, ed1, ls=":", lw=1.1, color="#1f77b4", label=r"$\Delta_1$ ED")
    ax.plot(lam, ed2, ls=":", lw=1.1, color="#ff7f0e", label=r"$\Delta_2$ ED")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\Delta$")
    apply_panel_style(ax)
    boxed_legend(ax, loc="best", fontsize=7.2)
    adapt_subfigure_032(fig)
    save_figure(fig, SPECTRUM_GAPS_OUT, generated)

    fig, ax = plt.subplots(figsize=(3.2, 2.55), constrained_layout=True)
    ax.axhline(1.0, color="0.55", ls=":", lw=1.1, label=r"$R=1$")
    ax.errorbar(lam, ratio, yerr=ratio_err, fmt="o", ms=4.0, color="#2ca02c",
                ecolor="#2ca02c", capsize=0.0, label="PIMC", zorder=4)
    ax.plot(lam, ratio_ed, ls=":", lw=1.1, color="black", label="ED")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$R=\Delta_2/(2\Delta_1)$")
    apply_panel_style(ax)
    boxed_legend(ax, loc="best", fontsize=7.2)
    adapt_subfigure_032(fig)
    save_figure(fig, SPECTRUM_RATIO_OUT, generated)


def main() -> None:
    configure_style()
    generated: list[Path] = []
    plot_distribution(generated)
    plot_spectrum(generated)
    print(f"Wrote {HIST_CHECK}")
    print("Wrote subfigure-ready distribution and split spectrum plots")


if __name__ == "__main__":
    main()
