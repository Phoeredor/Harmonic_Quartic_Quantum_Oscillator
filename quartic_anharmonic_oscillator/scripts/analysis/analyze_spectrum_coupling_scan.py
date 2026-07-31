#!/usr/bin/env python3
"""Robust quick-scan GEVP analysis for Delta1 and Delta2."""

from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_spectrum_gevp as gevp  # noqa: E402


RAW_DIR = Path("data/raw/spectrum/scan_quick")
ED_PATH = Path("data/processed/spectrum/anharmonic_spectrum_exact_reference_scan_quick.dat")
OUT_GAPS = Path("data/processed/spectrum/anharmonic_spectrum_scan_quick_effective_gaps.dat")
OUT_TABLE = Path("data/processed/spectrum/anharmonic_spectrum_scan_quick_gap_table.dat")
OUT_T0 = Path("data/processed/spectrum/anharmonic_spectrum_scan_quick_t0_stability.dat")
OUT_SUMMARY = Path("data/processed/spectrum/anharmonic_spectrum_scan_quick_summary.md")

T0_SCAN = (3, 5, 7)
DEFAULT_T0 = 5
PLATEAU = {
    "Delta1": ("odd", 0, 0.5, 1.5),
    "Delta2": ("even", 0, 0.5, 1.1),
}


def jk_error(samples, center):
    valid = np.isfinite(samples)
    n = int(valid.sum())
    if n < 2:
        return np.nan, n
    return float(np.sqrt((n - 1) / n * np.sum((samples[valid] - center) ** 2))), n


def weighted_average(values, weights, omit=None):
    return gevp.weighted_average(values, weights, omit)


def analyze_rows(rows, t0_dt):
    key = next(iter(gevp.group_records(rows)))
    lam, beta, nt_float, eta = key
    nt = int(round(nt_float))
    dts, weights, values = gevp.make_block_arrays(rows)
    matches = np.flatnonzero(dts == t0_dt)
    if len(matches) == 0:
        raise ValueError(f"missing t0_dt={t0_dt}")
    t0_index = int(matches[0])
    full_values = weighted_average(values, weights)
    result = {
        "meta": (lam, beta, nt, eta),
        "dts": dts,
        "t0_dt": t0_dt,
        "weights": weights,
        "sectors": {},
    }
    for sector in ("odd", "even"):
        matrices = gevp.connected_matrices(full_values, sector)
        rho, condition = gevp.solve_gevp(matrices, t0_index, 1.0e12)
        gaps = gevp.effective_gaps(rho, eta)
        jk_rho = np.full((len(weights), len(dts), 2), np.nan)
        jk_gaps = np.full((len(weights), len(dts) - 1, 2), np.nan)
        for omitted in range(len(weights)):
            sample_values = weighted_average(values, weights, omitted)
            sample_matrices = gevp.connected_matrices(sample_values, sector)
            try:
                jk_rho[omitted], _ = gevp.solve_gevp(sample_matrices, t0_index, 1.0e12)
                jk_gaps[omitted] = gevp.effective_gaps(jk_rho[omitted], eta)
            except np.linalg.LinAlgError:
                pass
        gap_err, gap_valid = gevp.jackknife_error(jk_gaps, gaps)
        rho_err, rho_valid = gevp.jackknife_error(jk_rho, rho)
        result["sectors"][sector] = {
            "rho": rho,
            "rho_err": rho_err,
            "rho_valid": rho_valid,
            "gaps": gaps,
            "gap_err": gap_err,
            "gap_valid": gap_valid,
            "jk_gaps": jk_gaps,
            "condition": condition,
        }
    return result


def plateau(result, label):
    sector, level, tau_min, tau_max = PLATEAU[label]
    eta = result["meta"][3]
    taus = result["dts"][:-1] * eta
    mask = ((result["dts"][:-1] >= result["t0_dt"]) &
            (taus >= tau_min - 1e-12) & (taus <= tau_max + 1e-12))
    if not np.any(mask):
        return np.nan, np.nan, 0, 0
    data = result["sectors"][sector]
    center = float(np.nanmean(data["gaps"][mask, level]))
    samples = np.nanmean(data["jk_gaps"][:, mask, level], axis=1)
    err, n_valid = jk_error(samples, center)
    return center, err, n_valid, int(mask.sum())


def load_ed():
    data = np.loadtxt(ED_PATH, comments="#", ndmin=2)
    return {float(row[0]): row for row in data}


def ed_row(ed, lam):
    for key, row in ed.items():
        if np.isclose(key, lam):
            return row
    raise KeyError(f"missing ED lambda={lam:g}")


def write_outputs(default_results, stability, table_rows):
    OUT_GAPS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_GAPS.open("w", encoding="utf-8") as stream:
        stream.write("# lambda beta Nt eta t0_dt sector level dt tau gap_eff gap_eff_err n_blocks n_jk_valid\n")
        for result in default_results:
            lam, beta, nt, eta = result["meta"]
            n_blocks = len(result["weights"])
            for sector, data in result["sectors"].items():
                for i, dt in enumerate(result["dts"][:-1]):
                    for level in range(2):
                        stream.write(
                            f"{lam:.17g} {beta:.17g} {nt:d} {eta:.17g} {DEFAULT_T0:d} "
                            f"{sector} {level:d} {dt:d} {dt * eta:.17g} "
                            f"{data['gaps'][i, level]:.17g} {data['gap_err'][i, level]:.17g} "
                            f"{n_blocks:d} {data['gap_valid'][i, level]:d}\n"
                        )
    with OUT_T0.open("w", encoding="utf-8") as stream:
        stream.write("# lambda t0_dt observable gap gap_err n_jk_valid n_tau_points\n")
        for row in stability:
            stream.write("%(lambda).17g %(t0_dt)d %(observable)s %(gap).17g %(err).17g "
                         "%(n_valid)d %(n_tau)d\n" % row)
    with OUT_TABLE.open("w", encoding="utf-8") as stream:
        stream.write("# lambda Delta1_pimc Delta1_err Delta1_ED Delta1_diff Delta1_z "
                     "Delta2_pimc Delta2_err Delta2_ED Delta2_diff Delta2_z "
                     "ratio_Delta2_over_2Delta1_pimc ratio_Delta2_over_2Delta1_ED\n")
        for row in table_rows:
            stream.write(" ".join(f"{value:.17g}" for value in row) + "\n")
    with OUT_SUMMARY.open("w", encoding="utf-8") as stream:
        stream.write("# Anharmonic spectrum quick scan summary\n\n")
        stream.write("Final reported gaps use GEVP `t0_dt = 5`, with fixed plateau windows "
                     "`0.5 <= tau <= 1.5` for Delta1 and `0.5 <= tau <= 1.1` for Delta2. "
                     "Errors are block jackknife errors of the window average.\n\n")
        stream.write("| lambda | blocks | cond odd | cond even | Delta1 PIMC | Delta1 ED | z1 | Delta2 PIMC | Delta2 ED | z2 |\n")
        stream.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for result, row in zip(default_results, table_rows):
            lam = row[0]
            stream.write(f"| {lam:g} | {len(result['weights'])} | "
                         f"{result['sectors']['odd']['condition']:.3e} | "
                         f"{result['sectors']['even']['condition']:.3e} | "
                         f"{row[1]:.6g} +- {row[2]:.2g} | {row[3]:.6g} | {row[5]:.2f} | "
                         f"{row[6]:.6g} +- {row[7]:.2g} | {row[8]:.6g} | {row[10]:.2f} |\n")
        max_z = np.nanmax(np.abs([[row[5], row[10]] for row in table_rows]))
        stream.write(f"\nMaximum |z| over Delta1 and Delta2: {max_z:.3g}.\n")
        stream.write("Delta3 and Delta4 are kept only as diagnostics in the effective-gap file.\n")


def main():
    paths = sorted(RAW_DIR.glob("*_corr.dat"))
    if not paths:
        raise SystemExit(f"no correlator files found in {RAW_DIR}")
    ed = load_ed()
    records = gevp.load_inputs(paths)
    groups = gevp.group_records(records)
    default_results = []
    stability = []
    table_rows = []
    for key, rows in sorted(groups.items()):
        lam = float(key[0])
        results_by_t0 = {}
        for t0_dt in T0_SCAN:
            try:
                result = analyze_rows(rows, t0_dt)
            except (ValueError, np.linalg.LinAlgError) as exc:
                print(f"[WARNING] lambda={lam:g} t0_dt={t0_dt}: {exc}", file=sys.stderr)
                continue
            results_by_t0[t0_dt] = result
            for observable in ("Delta1", "Delta2"):
                gap, err, n_valid, n_tau = plateau(result, observable)
                stability.append({
                    "lambda": lam, "t0_dt": t0_dt, "observable": observable,
                    "gap": gap, "err": err, "n_valid": n_valid, "n_tau": n_tau,
                })
        if DEFAULT_T0 not in results_by_t0:
            continue
        result = results_by_t0[DEFAULT_T0]
        default_results.append(result)
        d1, e1, _, _ = plateau(result, "Delta1")
        d2, e2, _, _ = plateau(result, "Delta2")
        exact = ed_row(ed, lam)
        ed1, ed2 = float(exact[6]), float(exact[7])
        diff1, diff2 = d1 - ed1, d2 - ed2
        z1 = diff1 / e1 if e1 > 0 else np.nan
        z2 = diff2 / e2 if e2 > 0 else np.nan
        ratio = d2 / (2.0 * d1)
        ratio_ed = ed2 / (2.0 * ed1)
        table_rows.append((lam, d1, e1, ed1, diff1, z1, d2, e2, ed2, diff2, z2, ratio, ratio_ed))
    if not table_rows:
        raise SystemExit("no valid scan groups")
    write_outputs(default_results, stability, table_rows)
    print(f"wrote {OUT_GAPS}")
    print(f"wrote {OUT_TABLE}")
    print(f"wrote {OUT_T0}")
    print(f"wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
