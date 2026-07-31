#!/usr/bin/env python3
# Analyze block-resolved Euclidean-path histograms as estimates of the thermal
# position density P_beta(y), with blocking errors and exact lattice/continuum
# Gaussian benchmarks for the harmonic oscillator.
"""
Analyze block-resolved position histograms for the QHO PIMC project.

The C executable can write position histograms for each Monte Carlo block. This
script reads those block files, checks their metadata and normalization, combines
blocks when requested, and writes the final tables used for the position-density
plots and summary report.

Each saved path contributes all Euclidean-time coordinates y_j. The measured
histogram therefore estimates the thermal position density

    P_beta(y) = < delta(q - y) >_beta.

For the harmonic oscillator, the exact continuum density is Gaussian with
variance

    sigma_beta^2 = 0.5 coth(beta / 2),

while the finite-lattice Gaussian has a slightly different variance computed
from the lattice normal modes. Both benchmarks are written so that the plots can
separate thermal broadening from discretization effects.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


# The block-histogram writer in the C code currently emits this schema.
SCHEMA_VERSION = 1

# Column order expected in the input table. Keeping this explicit makes changes
# in the C writer fail loudly instead of silently corrupting the analysis.
COLUMNS = (
    "block_id block_measurements block_site_samples bin_id bin_left bin_right "
    "bin_center count probability_mass probability_density underflow_count "
    "overflow_count block_mean_y block_mean_y2 block_mean_y4"
).split()

# Metadata required to interpret the table physically and statistically.
REQUIRED_METADATA = {
    "schema_version", "beta", "nt", "eta", "measure_every", "n_bins",
    "y_min", "y_max", "bin_width", "use_all_timeslices", "therm", "sweeps",
    "stride", "seed", "stream", "init", "update", "n_over",
    "block_measurements_target", "block_sweeps", "n_blocks",
    "underflow_count", "overflow_count",
}


@dataclass
class BlockHistogram:
    """Block-resolved histogram data and bin geometry for one simulation run."""

    metadata: dict[str, str]
    measurements: np.ndarray
    site_samples: np.ndarray
    counts: np.ndarray
    underflow: np.ndarray
    overflow: np.ndarray
    moments: np.ndarray
    left: np.ndarray
    right: np.ndarray
    center: np.ndarray

    @property
    def n_blocks(self) -> int:
        """Number of completed Monte Carlo blocks."""
        return self.counts.shape[0]

    @property
    def n_bins(self) -> int:
        """Number of histogram bins per block."""
        return self.counts.shape[1]

    @property
    def width(self) -> np.ndarray:
        """Bin widths; normally all entries are equal."""
        return self.right - self.left


def parse_metadata(path: Path) -> dict[str, str]:
    """Read key-value metadata from comment lines at the top of a data file."""
    metadata: dict[str, str] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                continue

            fields = line[1:].strip().split(maxsplit=1)
            if len(fields) == 2:
                metadata[fields[0]] = fields[1]

    return metadata


def load_block_histogram(path: Path) -> BlockHistogram:
    """Load and validate one schema-v1 block histogram file."""
    metadata = parse_metadata(path)

    # A missing metadata entry usually means the file comes from an older writer
    # or from an interrupted run. Stop before interpreting any numeric columns.
    missing = sorted(REQUIRED_METADATA - metadata.keys())
    if missing:
        raise ValueError(f"{path}: missing metadata: {', '.join(missing)}")

    if int(metadata["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version")

    # The final density analysis assumes that every saved Euclidean path
    # contributes all Nt coordinates, not just one representative site.
    if metadata["use_all_timeslices"].lower() != "true":
        raise ValueError(f"{path}: use_all_timeslices must be true")

    if metadata.get("columns", "").split() != COLUMNS:
        raise ValueError(f"{path}: incompatible columns")

    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != len(COLUMNS) or not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: invalid or non-finite data")

    n_bins = int(metadata["n_bins"])
    beta, eta, nt = float(metadata["beta"]), float(metadata["eta"]), int(metadata["nt"])

    # beta = eta * Nt fixes the Euclidean lattice spacing convention.
    if beta <= 0.0 or eta <= 0.0 or nt <= 1 or not math.isclose(beta, eta * nt, rel_tol=2e-12, abs_tol=1e-13):
        raise ValueError(f"{path}: beta, eta, and nt are inconsistent")

    if int(metadata["measure_every"]) != int(metadata["stride"]):
        raise ValueError(f"{path}: measure_every and stride disagree")

    # Data are expected as consecutive blocks, each containing all bins once.
    block_ids = data[:, 0].astype(int)
    ids = np.unique(block_ids)
    if not np.array_equal(ids, np.arange(ids.size)):
        raise ValueError(f"{path}: block ids must be contiguous from zero")
    if data.shape[0] != ids.size * n_bins:
        raise ValueError(f"{path}: each block must contain every bin exactly once")

    rows = data.reshape(ids.size, n_bins, len(COLUMNS))
    expected_bins = np.arange(n_bins)
    if not np.all(rows[:, :, 3].astype(int) == expected_bins[None, :]):
        raise ValueError(f"{path}: inconsistent bin ids")

    # Bin geometry must be identical for every block so that block averages are
    # statistically meaningful bin by bin.
    left, right, center = rows[0, :, 4], rows[0, :, 5], rows[0, :, 6]
    width = right - left
    declared_width = float(metadata["bin_width"])
    if np.any(width <= 0.0) or not np.allclose(width, declared_width, rtol=1e-12, atol=1e-14):
        raise ValueError(f"{path}: bin width inconsistent with metadata")
    if not np.allclose(rows[:, :, 4:7], rows[0:1, :, 4:7], rtol=0.0, atol=1e-13):
        raise ValueError(f"{path}: bin geometry changes between blocks")
    if not math.isclose(left[0], float(metadata["y_min"]), rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError(f"{path}: y_min inconsistent with bins")
    if not math.isclose(right[-1], float(metadata["y_max"]), rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError(f"{path}: y_max inconsistent with bins")

    measurements = rows[:, 0, 1].astype(np.int64)
    site_samples = rows[:, 0, 2].astype(np.int64)

    # One saved path contributes Nt coordinate samples to the histogram.
    if np.any(measurements <= 0) or np.any(site_samples != measurements * nt):
        raise ValueError(f"{path}: block_site_samples != block_measurements * nt")

    counts = rows[:, :, 7].astype(np.int64)
    underflow = rows[:, 0, 10].astype(np.int64)
    overflow = rows[:, 0, 11].astype(np.int64)
    if np.any(counts < 0) or np.any(underflow < 0) or np.any(overflow < 0):
        raise ValueError(f"{path}: negative counts")

    # In-range counts plus tails must reproduce the number of sampled sites.
    if np.any(counts.sum(axis=1) + underflow + overflow != site_samples):
        raise ValueError(f"{path}: block normalization/count identity failed")

    if int(metadata["n_blocks"]) != ids.size:
        raise ValueError(f"{path}: n_blocks inconsistent with data")
    if int(metadata["underflow_count"]) != int(underflow.sum()) or int(metadata["overflow_count"]) != int(overflow.sum()):
        raise ValueError(f"{path}: aggregate tail counts inconsistent with blocks")

    # Check that the probabilities stored by the C writer match the integer
    # counts. The analysis itself uses counts, not precomputed probabilities.
    masses = counts / site_samples[:, None]
    if not np.allclose(rows[:, :, 8], masses, rtol=2e-12, atol=1e-15):
        raise ValueError(f"{path}: probability_mass inconsistent with counts")
    if not np.allclose(rows[:, :, 9], masses / width[None, :], rtol=2e-12, atol=1e-15):
        raise ValueError(f"{path}: probability_density inconsistent with counts")

    moments = rows[:, 0, 12:15]
    if not np.allclose(rows[:, :, 12:15], moments[:, None, :], rtol=0.0, atol=1e-13):
        raise ValueError(f"{path}: block moments change within a block")

    return BlockHistogram(metadata, measurements, site_samples, counts, underflow,
                          overflow, moments, left, right, center)


def reblock(hist: BlockHistogram, factor: int) -> BlockHistogram:
    """Merge consecutive Monte Carlo blocks into larger blocks."""
    if factor <= 0:
        raise ValueError("reblocking factor must be positive")

    n_out = hist.n_blocks // factor
    if n_out < 1:
        raise ValueError("reblocking factor leaves no complete block")

    # Drop leftover blocks so that every merged block has the same statistics.
    keep = n_out * factor
    groups = np.arange(keep).reshape(n_out, factor)

    measurements = hist.measurements[groups].sum(axis=1)
    site_samples = hist.site_samples[groups].sum(axis=1)
    counts = hist.counts[groups].sum(axis=1)
    underflow = hist.underflow[groups].sum(axis=1)
    overflow = hist.overflow[groups].sum(axis=1)

    # Direct block moments are averages over site samples, so merged moments are
    # weighted by the number of sampled coordinates in each original block.
    weights = hist.site_samples[groups]
    moments = (hist.moments[groups] * weights[:, :, None]).sum(axis=1) / weights.sum(axis=1)[:, None]

    metadata = dict(hist.metadata)
    metadata["reblocking_factor"] = str(factor)

    return BlockHistogram(metadata, measurements, site_samples, counts, underflow,
                          overflow, moments, hist.left, hist.right, hist.center)


def mean_and_error(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return block mean and standard error of the mean along axis zero."""
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)

    if values.shape[0] < 2:
        return mean, np.full_like(mean, np.nan, dtype=float)

    return mean, values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])


def sigma_cont_squared(beta: float) -> float:
    """Exact continuum variance 0.5*coth(beta/2)."""
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")

    # This form avoids overflow of coth(beta/2) at very large beta.
    return 0.5 if beta > 700.0 else 0.5 + 1.0 / math.expm1(beta)


def sigma_lat_squared(beta: float, eta: float, nt: int, form: int = 1) -> float:
    """Exact finite-lattice Gaussian variance for the discretized oscillator."""
    if beta <= 0.0 or eta <= 0.0 or nt <= 1 or not math.isfinite(beta + eta):
        raise ValueError("invalid lattice parameters")
    if not math.isclose(beta, eta * nt, rel_tol=2e-12, abs_tol=1e-13):
        raise ValueError("beta must equal eta * nt")

    # Normal-mode sum of the quadratic lattice action. The two algebraic forms
    # are equivalent; form=1 is kept as the default used in the final tables.
    p = np.arange(nt, dtype=float)
    s2 = np.sin(np.pi * p / nt) ** 2
    if form == 1:
        terms = 1.0 / (eta + 4.0 * s2 / eta)
    elif form == 2:
        terms = eta / (eta * eta + 4.0 * s2)
    else:
        raise ValueError("form must be 1 or 2")

    variance = float(terms.mean())
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("non-positive lattice variance")

    return variance


def gaussian_density(y: np.ndarray | float, variance: float) -> np.ndarray:
    """Gaussian probability density with zero mean and supplied variance."""
    y_array = np.asarray(y, dtype=float)
    return np.exp(-0.5 * y_array * y_array / variance) / math.sqrt(2.0 * math.pi * variance)


def gaussian_bin_probability(left: np.ndarray, right: np.ndarray, variance: float) -> np.ndarray:
    """Integrated Gaussian probability for each histogram bin."""
    scale = math.sqrt(2.0 * variance)
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (erf(np.asarray(right) / scale) - erf(np.asarray(left) / scale))


def histogram_moments(hist: BlockHistogram) -> np.ndarray:
    """Approximate moments from bin centers, used to quantify binning bias."""
    masses = hist.counts / hist.site_samples[:, None]
    return np.column_stack([masses @ hist.center, masses @ hist.center**2, masses @ hist.center**4])


def analyze(hist: BlockHistogram, factor: int) -> dict[str, np.ndarray | float | int]:
    """Analyze one run after optional reblocking."""
    reb = reblock(hist, factor)

    # The density estimate divides integer counts by both samples and bin width.
    density_blocks = reb.counts / reb.site_samples[:, None] / reb.width[None, :]
    density, density_err = mean_and_error(density_blocks)

    # Direct moments come from the C block writer and use all sampled sites.
    moments, moments_err = mean_and_error(reb.moments)

    # Bin-center moments are not used as final observables; they diagnose the
    # finite bin-width bias of the histogram representation.
    hist_mom, hist_mom_err = mean_and_error(histogram_moments(reb))

    normalization_blocks = (reb.counts.sum(axis=1) + reb.underflow + reb.overflow) / reb.site_samples
    normalization, normalization_err = mean_and_error(normalization_blocks)
    underflow, underflow_err = mean_and_error(reb.underflow / reb.site_samples)
    overflow, overflow_err = mean_and_error(reb.overflow / reb.site_samples)

    beta = float(reb.metadata["beta"])
    eta = float(reb.metadata["eta"])
    nt = int(reb.metadata["nt"])

    cont_var = sigma_cont_squared(beta)
    lat_var = sigma_lat_squared(beta, eta, nt)

    # Compare bin-averaged exact probabilities, not pointwise densities at bin
    # centers. This avoids a small but systematic finite-bin bias.
    cont_mass = gaussian_bin_probability(reb.left, reb.right, cont_var)
    lat_mass = gaussian_bin_probability(reb.left, reb.right, lat_var)

    return {
        "hist": reb, "density": density, "density_err": density_err,
        "moments": moments, "moments_err": moments_err,
        "hist_moments": hist_mom, "hist_moments_err": hist_mom_err,
        "normalization": float(normalization), "normalization_err": float(normalization_err),
        "underflow": float(underflow), "underflow_err": float(underflow_err),
        "overflow": float(overflow), "overflow_err": float(overflow_err),
        "cont_var": cont_var, "lat_var": lat_var,
        "cont_density": cont_mass / reb.width, "lat_density": lat_mass / reb.width,
        "factor": factor, "n_blocks": reb.n_blocks,
    }


def _pull(residual: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Return residual/error, leaving undefined pulls as NaN."""
    return np.divide(residual, error, out=np.full_like(residual, np.nan), where=error > 0.0)


def write_outputs(results: Sequence[dict[str, object]], output_dir: Path) -> None:
    """Write final histogram and moment-summary tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    histogram_path = output_dir / "qho_position_distribution_histograms_final.dat"
    with histogram_path.open("w", encoding="utf-8") as out:
        out.write("# beta Nt eta bin_id bin_left bin_right bin_center P_mc P_mc_err P_cont_binavg P_lat_binavg residual_vs_cont pull_vs_cont residual_vs_lat pull_vs_lat n_blocks reblocking_factor\n")

        for result in results:
            hist = result["hist"]
            assert isinstance(hist, BlockHistogram)

            density, error = np.asarray(result["density"]), np.asarray(result["density_err"])
            cont, lat = np.asarray(result["cont_density"]), np.asarray(result["lat_density"])
            beta, eta, nt = float(hist.metadata["beta"]), float(hist.metadata["eta"]), int(hist.metadata["nt"])
            n_blocks, factor = int(result["n_blocks"]), int(result["factor"])

            for i in range(hist.n_bins):
                rc, rl = density[i] - cont[i], density[i] - lat[i]
                pull_cont = _pull(np.array([rc]), np.array([error[i]]))[0]
                pull_lat = _pull(np.array([rl]), np.array([error[i]]))[0]
                out.write(f"{beta:.17g} {nt} {eta:.17g} {i} {hist.left[i]:.17g} {hist.right[i]:.17g} {hist.center[i]:.17g} {density[i]:.17g} {error[i]:.17g} {cont[i]:.17g} {lat[i]:.17g} {rc:.17g} {pull_cont:.17g} {rl:.17g} {pull_lat:.17g} {n_blocks} {factor}\n")

    summary_path = output_dir / "qho_position_distribution_summary_final.dat"
    with summary_path.open("w", encoding="utf-8") as out:
        out.write("# beta Nt eta seed1 seed2 n_therm n_sweeps stride n_blocks_base reblocking_factor n_blocks_effective y_mean y_mean_err y2 y2_err y4 y4_err y2_cont_exact y2_lat_exact beta_y2 beta_y2_err normalization normalization_err underflow_fraction overflow_fraction runtime_seconds\n")

        for result in results:
            hist = result["hist"]
            assert isinstance(hist, BlockHistogram)

            beta, eta, nt = float(hist.metadata["beta"]), float(hist.metadata["eta"]), int(hist.metadata["nt"])
            n_blocks = int(result["n_blocks"])
            moments, moment_errors = np.asarray(result["moments"]), np.asarray(result["moments_err"])
            runtime = float(hist.metadata.get("runtime_seconds", "nan"))
            n_blocks_base = int(hist.metadata["n_blocks"])
            factor = int(result["factor"])

            out.write(f"{beta:.17g} {nt} {eta:.17g} {hist.metadata['seed']} {hist.metadata['stream']} {hist.metadata['therm']} {hist.metadata['sweeps']} {hist.metadata['stride']} {n_blocks_base} {factor} {n_blocks} {moments[0]:.17g} {moment_errors[0]:.17g} {moments[1]:.17g} {moment_errors[1]:.17g} {moments[2]:.17g} {moment_errors[2]:.17g} {float(result['cont_var']):.17g} {float(result['lat_var']):.17g} {beta*moments[1]:.17g} {beta*moment_errors[1]:.17g} {float(result['normalization']):.17g} {float(result['normalization_err']):.17g} {float(result['underflow']):.17g} {float(result['overflow']):.17g} {runtime:.17g}\n")


def blocking_scan_rows(hist: BlockHistogram, factors: Sequence[int]) -> list[str]:
    """Build rows showing how moment errors change under reblocking."""
    rows = []

    for factor in factors:
        result = analyze(hist, factor)
        errors = np.asarray(result["moments_err"])
        rows.append(f"{hist.metadata['beta']} {hist.metadata['nt']} {hist.metadata['eta']} {factor} {result['n_blocks']} {errors[0]:.17g} {errors[1]:.17g} {errors[2]:.17g} {float(result['normalization_err']):.17g} {np.nanmean(np.asarray(result['density_err'])):.17g}\n")

    return rows


def main() -> int:
    """Command-line entry point for the position-distribution analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="+", help="schema-v1 block histogram files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/production"))
    parser.add_argument("--reblock", type=int, default=1)
    parser.add_argument("--scan-factors", default="1,2,4,8")
    parser.add_argument("--tail-warning", type=float, default=1.0e-3)
    parser.add_argument("--min-blocks", type=int, default=20)
    args = parser.parse_args()

    histograms = [load_block_histogram(path) for path in args.input]
    results = [analyze(hist, args.reblock) for hist in histograms]

    report_lines = [
        "# QHO position-distribution analysis", "",
        "The measured observable is $P_\\beta(y)=\\langle\\delta(q-y)\\rangle_\\beta$.", "",
        "Its spectral representation is $P_\\beta(y)=Z^{-1}\\sum_n e^{-\\beta E_n}|\\psi_n(y)|^2$; hence $P_\\beta(y)\\to|\\psi_0(y)|^2$ as $\\beta\\to\\infty$.", "",
        "For the harmonic oscillator, $P_\\beta(y)=\\sqrt{\\tanh(\\beta/2)/\\pi}\\,e^{-y^2\\tanh(\\beta/2)}$ and $\\langle y^2\\rangle=\\frac12\\coth(\\beta/2)$.", "",
        "In the classical limit, $P_\\beta(y)\\to\\sqrt{\\beta/(2\\pi)}e^{-\\beta y^2/2}$ and $\\beta\\langle y^2\\rangle\\to1$.", "",
        "All Euclidean time slices enter each saved measurement. Statistical errors are estimated from Monte Carlo blocks, not from Poisson bin errors. Exact bin comparisons use integrated Gaussian probabilities.", "",
        "## Runs and moments", "",
        "| beta | Nt | eta | seed | stream | blocks | reblock | <y> | <y2> | y2 lat | y2 cont | MC-lat | MC-cont | tails | runtime (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    run_rows: list[str] = []
    scan_rows: list[str] = []
    requested_factors = [int(value) for value in args.scan_factors.split(",") if value]

    for path, hist, result in zip(args.input, histograms, results):
        if int(result["n_blocks"]) < args.min_blocks:
            print(f"Warning: {path}: only {result['n_blocks']} blocks after reblocking")

        tail = float(result["underflow"]) + float(result["overflow"])
        if tail > args.tail_warning:
            print(f"Warning: {path}: tail fraction {tail:.6g} exceeds {args.tail_warning:.6g}")

        if np.all(hist.counts == 0, axis=0).any():
            print(f"Warning: {path}: at least one bin is empty in every block")

        # Direct moments and bin-center moments differ by finite-bin effects.
        direct, binned = np.asarray(result["moments"]), np.asarray(result["hist_moments"])
        print(f"{path}: direct minus bin-center moments:", *(direct - binned))

        # Only scan factors that leave at least two effective blocks, otherwise
        # a standard error cannot be estimated.
        factors = [factor for factor in requested_factors if hist.n_blocks // factor >= 2]
        scan_rows.extend(blocking_scan_rows(hist, factors))

        report_lines.extend([
            f"## beta={hist.metadata['beta']}, Nt={hist.metadata['nt']}", "",
            f"Input: `{path}`", "",
            f"Blocks after reblocking: {result['n_blocks']}", "",
            f"Seed source: `{hist.metadata.get('seed_source', 'not recorded')}`", "",
            f"MC parameters: therm={hist.metadata['therm']}, sweeps={hist.metadata['sweeps']}, stride={hist.metadata['stride']}, saved measurements/block={hist.metadata['block_measurements_target']}", "",
            f"Reblocking factor: {args.reblock}", "",
            f"Tail fraction: {tail:.8g}", "",
        ])

        moments = np.asarray(result["moments"])
        run_rows.append(
            f"| {hist.metadata['beta']} | {hist.metadata['nt']} | {hist.metadata['eta']} | {hist.metadata['seed']} | {hist.metadata['stream']} | {result['n_blocks']} | {args.reblock} | {moments[0]:.6g} | {moments[1]:.6g} | {float(result['lat_var']):.6g} | {float(result['cont_var']):.6g} | {moments[1]-float(result['lat_var']):+.3g} | {moments[1]-float(result['cont_var']):+.3g} | {tail:.3g} | {hist.metadata.get('runtime_seconds', 'n/a')} |"
        )

    # Insert the compact run table immediately below its Markdown header.
    report_lines[16:16] = run_rows + [""]

    write_outputs(results, args.output_dir)

    scan_path = args.output_dir / "qho_position_distribution_blocking_scan.dat"
    with scan_path.open("w", encoding="utf-8") as out:
        out.write("# beta Nt eta reblocking_factor n_blocks y_mean_err y2_err y4_err normalization_err mean_bin_density_err\n")
        out.writelines(scan_rows)

    report = args.output_dir / "qho_position_distribution_report.md"
    report_lines.extend([
        "## Interpretation", "",
        "The distributions broaden as beta decreases. The finite-lattice benchmark separates discretization from physical thermal broadening, while the continuous benchmark is the target as eta tends to zero.", "",
        "The approach of $\\beta\\langle y^2\\rangle$ to one quantifies the classical crossover. At beta=40 the exact density is numerically the ground-state density $|\\psi_0(y)|^2=\\pi^{-1/2}e^{-y^2}$.", "",
        "This report uses direct histogram comparisons only; it does not reconstruct excited-state densities by thermal subtraction.", "",
        "## Final products", "",
        "- `qho_position_distribution_histograms_final.dat`",
        "- `qho_position_distribution_summary_final.dat`",
        "- `qho_position_distribution_blocking_scan.dat`",
        "- `qho_position_distribution_report.md`",
        "- `plots/distribution/fig_position_distribution_beta_scan.png`",
        "- `plots/distribution/fig_position_variance_crossover.png`",
        "",
    ])

    report.write_text("\n".join(report_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
