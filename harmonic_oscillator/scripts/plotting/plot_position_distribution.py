#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: scripts/plotting/plot_position_distribution.py
# Purpose: Create figures for the position probability density analysis.
# Blocked PIMC histograms and <y^2> estimates are compared with quantum thermal,
# finite-lattice, and classical harmonic-oscillator expectations.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: imports and configuration
# -----------------------------------------------------------------------------

"""Canonical plots for processed QHO position-distribution results."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import seaborn as sns
except ImportError:
    sns = None


def style() -> None:
    if sns is not None:
        sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "axes.grid": True,
        "grid.color": "0.80", "grid.linestyle": "--", "grid.linewidth": 0.55,
        "grid.alpha": 0.65, "axes.axisbelow": True,
        "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
        "xtick.top": True, "ytick.right": True, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


BETAS = (40.0, 1.0, 0.5, 0.25)


def beta_colors() -> dict[float, tuple[float, ...]]:
    colors = sns.color_palette("viridis", 4) if sns is not None else plt.cm.viridis(np.linspace(0.12, 0.88, 4))
    return {beta: colors[i] for i, beta in enumerate(BETAS)}


def load(path: Path, names: list[str]) -> np.ndarray:
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != len(names):
        raise ValueError(f"{path}: incompatible processed table")
    return data


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_y2(summary: np.ndarray, output_dir: Path) -> None:
    order = np.argsort(summary[:, 0])
    d = summary[order]
    beta = d[:, 0]
    dense = np.geomspace(beta.min(), beta.max(), 500)
    exact = np.array([0.5 + 1.0 / math.expm1(x) if x < 700 else 0.5 for x in dense])
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.8), sharex=True)
    ax = axes[0]
    ax.errorbar(beta, d[:, 13], yerr=d[:, 14], fmt="o", ms=5.5, elinewidth=1.0,
                color="black", capsize=0, label="PIMC", zorder=3)
    ax.plot(dense, exact, color="#2378b5", lw=1.5, label="quantum exact")
    ax.plot(dense, 1.0 / dense, color="0.35", lw=1.3, ls="--", label=r"$P_{cl}(y;\beta=0.25)$")
    ax.set(yscale="log", ylabel=r"$\langle y^2\rangle$")
    ax = axes[1]
    pimc_handle = ax.errorbar(beta, d[:, 19], yerr=d[:, 20], fmt="o", ms=5.5, elinewidth=1.0,
                              color="black", capsize=0, label="PIMC", zorder=3)
    quantum_handle, = ax.plot(dense, dense * exact, color="#2378b5", lw=1.5, label="quantum exact")
    classical_handle = ax.axhline(1.0, color="0.35", lw=1.3, ls="--", label="classical limit")
    ax.set(xscale="log", xlabel=r"$\beta$", ylabel=r"$\beta\langle y^2\rangle$")
    for ax in axes:
        ax.tick_params(which="major", direction="in", top=True, right=True, labelsize=20,
                       length=6.0, width=1.2)
        ax.tick_params(which="minor", direction="in", top=True, right=True,
                       length=3.8, width=1.0)
        ax.yaxis.label.set_size(24)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    axes[1].xaxis.label.set_size(24)
    fig.align_ylabels(axes)
    for ax in axes:
        for line in ax.lines:
            line.set_linewidth(max(line.get_linewidth(), 1.5))
            if line.get_marker() not in (None, "None", "", " "):
                line.set_markersize(max(line.get_markersize(), 5.5))
        for collection in ax.collections:
            widths = collection.get_linewidths()
            if len(widths):
                collection.set_linewidth(max(float(np.max(widths)), 1.2))
    axes[1].legend(
        handles=[quantum_handle, classical_handle, pimc_handle],
        labels=["Quantum Exact", "Classical Limit", "PIMC"],
        fontsize=18,
        handlelength=1.8,
        handletextpad=0.55,
        loc="upper left",
    )
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.15, top=0.975, hspace=0.10)
    save(fig, output_dir, "fig_position_variance_crossover")


def plot_overlay_histbars(data: np.ndarray, output_dir: Path) -> None:
    colors = beta_colors()
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for beta in reversed(BETAS):
        point = data[data[:, 0] == beta]
        color = colors[beta]
        width = point[:, 5] - point[:, 4]
        ax.bar(point[:, 6], point[:, 7], width=0.94 * width, align="center",
               facecolor=(*color[:3], 0.13), edgecolor=color, linewidth=0.9,
               zorder=2 + BETAS.index(beta))
    grid = np.linspace(-10.5, 10.5, 1000)
    ground_state = np.exp(-grid * grid) / math.sqrt(math.pi)
    beta_classical = 0.25
    classical = math.sqrt(beta_classical / (2.0 * math.pi)) \
        * np.exp(-0.5 * beta_classical * grid * grid)
    ax.plot(grid, ground_state, color="black", lw=2.0, zorder=10,
            label=r"Ground state $|\psi_0(y)|^2$")
    ax.plot(grid, classical, color="0.30", lw=2.0, ls="--", zorder=10,
            label=r"$P_{cl}(y;\beta=0.25)$")
    ax.set(xlim=(-5, 5), xlabel=r"$y$", ylabel=r"$P_\beta(y)$")
    ax.tick_params(which="major", direction="in", top=True, right=True, labelsize=20,
                   length=6.0, width=1.2)
    ax.tick_params(which="minor", direction="in", top=True, right=True,
                   length=3.8, width=1.0)
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    beta_handles = [Patch(facecolor=(*colors[b][:3], 0.18), edgecolor=colors[b],
                          label=rf"$\beta={b:g}$") for b in BETAS]
    theory_handles = [
        Line2D([], [], color="black", lw=2.0, label=r"$|\psi_0|^2$"),
        Line2D([], [], color="0.30", lw=2.0, ls="--",
               label=r"$P_{cl}(y;\beta=0.25)$"),
    ]
    beta_legend = ax.legend(handles=beta_handles, loc="upper left", ncol=1,
                            fontsize=18, handlelength=1.05, handletextpad=0.45,
                            borderpad=0.25, labelspacing=0.22, frameon=True,
                            facecolor="white", edgecolor="none", framealpha=0.78)
    ax.add_artist(beta_legend)
    ax.legend(handles=theory_handles, loc="upper right", ncol=1,
              fontsize=18, handlelength=1.55, handletextpad=0.45,
              borderpad=0.25, labelspacing=0.25, frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.78)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.15, top=0.975)
    save(fig, output_dir, "fig_position_distribution_beta_scan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histograms", type=Path, default=Path("data/processed/production/qho_position_distribution_histograms_final.dat"))
    parser.add_argument("--summary", type=Path, default=Path("data/processed/production/qho_position_distribution_summary_final.dat"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/distribution"))
    parser.add_argument("--variance", action="store_true", help="also regenerate the variance crossover")
    args = parser.parse_args()
    hist_names = "beta Nt eta bin_id bin_left bin_right bin_center P_mc P_mc_err P_cont_binavg P_lat_binavg residual_vs_cont pull_vs_cont residual_vs_lat pull_vs_lat n_blocks reblocking_factor".split()
    summary_names = "beta Nt eta seed1 seed2 n_therm n_sweeps stride n_blocks_base reblocking_factor n_blocks_effective y_mean y_mean_err y2 y2_err y4 y4_err y2_cont_exact y2_lat_exact beta_y2 beta_y2_err normalization normalization_err underflow_fraction overflow_fraction runtime_seconds".split()
    style()
    histogram_data = load(args.histograms, hist_names)
    if args.variance:
        plot_y2(load(args.summary, summary_names), args.output_dir)
    plot_overlay_histbars(histogram_data, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
