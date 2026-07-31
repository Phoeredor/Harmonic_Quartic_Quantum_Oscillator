#!/usr/bin/env python3
# Plot beta=5 continuum moments, virial estimators, and thermal position
# distributions across quartic couplings.  Figures distinguish finite-eta data,
# eta^2 intercepts, and exact reference values.
"""Make the final beta=5 anharmonic-oscillator plots."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from quartic_continuum_common import PROCESSED, is_excluded_eta_nt, read_named  # noqa: E402


POINTS = PROCESSED / "anharmonic_beta5_continuum_v2_points.dat"
FITS = PROCESSED / "anharmonic_beta5_continuum_v2_fits.dat"
FINAL = PROCESSED / "anharmonic_beta5_continuum_v2_final_table.dat"
DERIVED = PROCESSED / "anharmonic_beta5_derived_continuum_v2.dat"
SELECTED = (0.025, 0.25, 1.00)
Y2_COLOR = "#2ca02c"
Y4_COLOR = "#7b3294"
FIT_COLOR = "#d62728"
REF_COLOR = "black"
V_COLOR = "#1f77b4"
K_COLOR = "#ff7f0e"
UNUSED_COLOR = "0.45"


def configure_report_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "CMU Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
        "font.size": 14,
        "axes.labelsize": 17,
        "axes.titlesize": 14,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
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
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontsize(18)
            for handle in leg.legend_handles:
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
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontsize(9)
            for handle in leg.legend_handles:
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
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontsize(11)
            for handle in leg.legend_handles:
                if hasattr(handle, "set_linewidth"):
                    handle.set_linewidth(1.5)
                if hasattr(handle, "set_markersize"):
                    handle.set_markersize(6.0)


def apply_style(ax: plt.Axes) -> None:
    ax.grid(True, color="0.70", alpha=0.55, linestyle=":", linewidth=0.8)
    ax.tick_params(direction="in", which="major", top=True, right=True, width=1.0, length=4.8)
    ax.tick_params(direction="in", which="minor", top=True, right=True, width=0.8, length=3.2)
    ax.minorticks_on()


def legend(ax: plt.Axes, **kwargs: Any) -> None:
    kwargs.setdefault("fontsize", 8.5)
    kwargs.setdefault("frameon", True)
    leg = ax.legend(**kwargs)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(1.0)
    leg.get_frame().set_edgecolor("none")


def save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def pick(data: np.ndarray, lam: float, obs: str | None = None) -> np.ndarray:
    mask = np.isclose(data["lambda"], lam)
    if obs is not None:
        mask &= data["obs"] == obs
    return data[mask]


def eta_label(value: float) -> str:
    return f"{float(value):.6g}"


# Display the selected continuum window and O(eta)=A+B*eta^2 fit for one lambda.
def plot_eta2_zoom(
    points: np.ndarray,
    fits: np.ndarray,
    obs: str,
    color: str,
    base: Path,
    ylabel: str,
    note_fontsize: float = 8.5,
    legend_fontsize: float = 8.0,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.0), sharex=True)
    for ax, lam in zip(axes, SELECTED):
        sub = pick(points, lam)
        sub = sub[(sub["eta"] >= 0.015) & (sub["eta"] <= 0.25)]
        sub = sub[[not is_excluded_eta_nt(float(row["eta"]), int(row["Nt"])) for row in sub]]
        fit = pick(fits, lam, obs)[0]
        fit_mask = (sub["eta"] >= fit["eta_min"]) & (sub["eta"] <= fit["eta_max"])
        used = sub[fit_mask]
        unused = sub[~fit_mask]
        x = np.linspace(0.0, float(fit["eta_max"]) ** 2, 200)
        if len(unused) > 0:
            ax.errorbar(
                unused["eta2"],
                unused[obs],
                yerr=unused[f"{obs}_err"],
                fmt="o",
                ms=4.0,
                mfc="white",
                mec=UNUSED_COLOR,
                mew=1.0,
                color=UNUSED_COLOR,
                ecolor=UNUSED_COLOR,
                elinewidth=0.8,
                capsize=0.0,
                label="_nolegend_",
            )
        ax.errorbar(
            used["eta2"],
            used[obs],
            yerr=used[f"{obs}_err"],
            fmt="o",
            ms=4.0,
            mfc=color,
            mec=color,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=0.0,
            label="_nolegend_",
        )
        fit_line, = ax.plot(x, float(fit["A"]) + float(fit["B"]) * x, color=FIT_COLOR, ls="-.", lw=1.4, label=r"Fit: $a+b\eta^2$")
        ax.axhline(float(fit["exact"]), color=REF_COLOR, ls=":", lw=1.2, label="ED")
        ax.text(
            0.97,
            0.93,
            "\n".join((
                rf"$\eta_\mathrm{{min}}={eta_label(float(fit['eta_min']))}$",
                rf"$\eta_\mathrm{{max}}={eta_label(float(fit['eta_max']))}$",
                rf"$\chi^2_\mathrm{{red}}={float(fit['chi2_red']):.3f}$",
            )),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=note_fontsize,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 2.0},
        )
        ax.set_title(rf"$\lambda={lam:g}$")
        ax.set_xlabel(r"$\eta^2$")
        apply_style(ax)
        leg = ax.legend([ax.lines[-1], fit_line], ["ED", r"Fit: $a+b\eta^2$"], loc="lower left", fontsize=legend_fontsize, frameon=True)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(1.0)
        leg.get_frame().set_edgecolor("none")
    axes[0].set_ylabel(ylabel)
    fig.subplots_adjust(wspace=0.50)
    adapt_three_quarter_width_figure(fig)
    fig.set_size_inches(6.3, 3.2, forward=True)
    for ax in axes:
        ax.title.set_fontsize(13)
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.tick_params(axis="both", which="major", labelsize=10)
        for text in ax.texts:
            text.set_fontsize(9)
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontsize(9)
            for handle in leg.legend_handles:
                if hasattr(handle, "set_linewidth"):
                    handle.set_linewidth(1.3)
                if hasattr(handle, "set_markersize"):
                    handle.set_markersize(5.0)
    save(fig, base)


def total_errorbar(ax: plt.Axes, x, y, stat, sys, color: str, label: str) -> None:
    total = np.sqrt(stat * stat + sys * sys)
    ax.errorbar(x, y, yerr=total, fmt="o", ms=4.2, mfc=color, mec=color, color=color, ecolor=color, elinewidth=1.0, capsize=0.0, label=label)


def plot_continuum_vs_lambda(final: np.ndarray, obs: str, color: str, base: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.15))
    x = final["lambda"]
    total_errorbar(ax, x, final[f"{obs}_cont"], final[f"{obs}_stat"], final[f"{obs}_sys"], color, "PIMC")
    ax.plot(x, final[f"{obs}_exact"], color=REF_COLOR, marker="s", ms=3.6, lw=1.2, ls=":", label="ED")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(ylabel)
    apply_style(ax)
    legend(ax, loc="best")
    apply_selected_large_fonts(fig)
    save(fig, base)


# Compare continuum V, K_vir, and K_ren as functions of the interaction strength.
def plot_virial_components(derived: np.ndarray) -> None:
    v = derived[derived["obs"] == "V"]
    k = derived[derived["obs"] == "K_vir"]
    fig, ax = plt.subplots(figsize=(5.2, 4.15))
    ax.errorbar(
        v["lambda"],
        v["A"],
        yerr=v["A_err"],
        fmt="o",
        color=V_COLOR,
        ms=4.4,
        lw=1.1,
        capsize=0.0,
        label=r"$V_{\mathrm{cont}}$",
    )
    ax.errorbar(
        k["lambda"],
        k["A"],
        yerr=k["A_err"],
        fmt="o",
        color=K_COLOR,
        ms=4.4,
        lw=1.1,
        capsize=0.0,
        label=r"$K_{\mathrm{vir},\mathrm{cont}}$",
    )
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$V_{\mathrm{cont}},\ K_{\mathrm{vir},\mathrm{cont}}$")
    apply_style(ax)
    legend(ax, loc="upper left", fontsize=10.0)
    adapt_three_quarter_width_figure(fig)
    save(fig, Path("plots/thermodynamic/fig_virial_components_vs_lambda"))


def main() -> None:
    configure_report_style()
    points = read_named(POINTS)
    fits = read_named(FITS)
    final = read_named(FINAL)
    derived = read_named(DERIVED)
    plot_eta2_zoom(
        points,
        fits,
        "y4",
        Y4_COLOR,
        Path("plots/thermodynamic/fig_y4_continuum_fits"),
        r"$\langle y^4\rangle_\eta$",
        note_fontsize=14.0,
        legend_fontsize=14.0,
    )
    plot_continuum_vs_lambda(final, "y2", Y2_COLOR, Path("plots/thermodynamic/fig_y2_continuum_vs_lambda"), r"$\langle y^2\rangle_{\mathrm{cont}}$")
    plot_continuum_vs_lambda(final, "y4", Y4_COLOR, Path("plots/thermodynamic/fig_y4_continuum_vs_lambda"), r"$\langle y^4\rangle_{\mathrm{cont}}$")
    plot_virial_components(derived)
    print("wrote final beta=5 plots")


if __name__ == "__main__":
    main()
