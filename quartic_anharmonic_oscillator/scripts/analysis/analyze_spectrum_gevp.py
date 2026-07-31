#!/usr/bin/env python3
# Extract anharmonic excitation gaps from block correlator matrices.  Connected
# odd (y,y^3) and even (y^2,y^4) sectors are analyzed with a generalized
# eigenvalue problem, with uncertainties from leave-one-block jackknife samples.
"""Block-jackknife GEVP analysis for anharmonic-oscillator correlator matrices."""

import argparse
import sys
from pathlib import Path

import numpy as np


COLUMNS = [
    "block_index", "beta", "eta", "Nt", "lambda", "n_meas", "dt", "tau",
    "mean_y", "mean_y2", "mean_y3", "mean_y4",
    "odd_00", "odd_01", "odd_10", "odd_11",
    "even_00", "even_01", "even_10", "even_11",
]
META = ("lambda", "beta", "Nt", "eta")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--t0-dt", type=int, required=True)
    parser.add_argument("--condition-max", type=float, default=1.0e12)
    parser.add_argument("--principal-out", type=Path, default=Path(
        "data/processed/spectrum/anharmonic_spectrum_gevp_principal.dat"))
    parser.add_argument("--gaps-out", type=Path, default=Path(
        "data/processed/spectrum/anharmonic_spectrum_gevp_effective_gaps.dat"))
    parser.add_argument("--summary-out", type=Path, default=Path(
        "data/processed/spectrum/anharmonic_spectrum_gevp_summary.md"))
    return parser.parse_args()


def load_inputs(paths):
    records = []
    for file_id, path in enumerate(paths):
        data = np.loadtxt(path, comments="#", ndmin=2)
        if data.shape[1] != len(COLUMNS):
            raise ValueError(f"{path}: expected {len(COLUMNS)} columns, got {data.shape[1]}")
        for row in data:
            record = dict(zip(COLUMNS, row))
            record["file_id"] = file_id
            records.append(record)
    if not records:
        raise ValueError("no correlator rows found")
    return records


def group_records(records):
    groups = {}
    for row in records:
        key = tuple(row[name] for name in META)
        groups.setdefault(key, []).append(row)
    return groups


# Arrange each statistically contiguous block on a common Euclidean-time grid.
def make_block_arrays(rows):
    dts = np.array(sorted({int(row["dt"]) for row in rows}), dtype=int)
    dt_to_index = {dt: i for i, dt in enumerate(dts)}
    block_keys = sorted({(int(row["file_id"]), int(row["block_index"])) for row in rows})
    block_to_index = {key: i for i, key in enumerate(block_keys)}
    values = np.full((len(block_keys), len(dts), 12), np.nan)
    weights = np.full(len(block_keys), np.nan)
    fields = COLUMNS[8:]
    for row in rows:
        b = block_to_index[(int(row["file_id"]), int(row["block_index"]))]
        t = dt_to_index[int(row["dt"])]
        values[b, t] = [row[name] for name in fields]
        if np.isnan(weights[b]):
            weights[b] = row["n_meas"]
        elif weights[b] != row["n_meas"]:
            raise ValueError("n_meas changes within one block")
    if np.isnan(values).any() or np.isnan(weights).any():
        raise ValueError("incomplete block/dt grid in correlator input")
    return dts, weights, values


# Subtract one-point products within each sample before symmetrizing the matrix.
def connected_matrices(weighted_values, sector):
    means = weighted_values[:, :4]
    if sector == "odd":
        raw = weighted_values[:, 4:8].reshape(-1, 2, 2)
        ops = means[:, [0, 2]]
    else:
        raw = weighted_values[:, 8:12].reshape(-1, 2, 2)
        ops = means[:, [1, 3]]
    matrices = raw - ops[:, :, None] * ops[:, None, :]
    return 0.5 * (matrices + np.swapaxes(matrices, 1, 2))


def weighted_average(values, weights, omit=None):
    mask = np.ones(len(weights), dtype=bool)
    if omit is not None:
        mask[omit] = False
    return np.tensordot(weights[mask], values[mask], axes=(0, 0)) / weights[mask].sum()


# Normalize by C(t0); principal correlators isolate states in one parity sector.
def solve_gevp(matrices, t0_index, condition_max):
    reference = matrices[t0_index]
    eig0 = np.linalg.eigvalsh(reference)
    if eig0[0] <= 0.0:
        raise np.linalg.LinAlgError(f"C(t0) is not positive definite (min eig={eig0[0]:.3e})")
    condition = eig0[-1] / eig0[0]
    if not np.isfinite(condition) or condition > condition_max:
        raise np.linalg.LinAlgError(f"C(t0) is ill conditioned (cond={condition:.3e})")
    chol = np.linalg.cholesky(reference)
    inv_chol = np.linalg.solve(chol, np.eye(2))
    rho = np.empty((len(matrices), 2))
    for t, matrix in enumerate(matrices):
        transformed = inv_chol @ matrix @ inv_chol.T
        transformed = 0.5 * (transformed + transformed.T)
        rho[t] = np.linalg.eigvalsh(transformed)[::-1]
    return rho, condition


# Convert leave-one-block fluctuations into the standard jackknife uncertainty.
def jackknife_error(samples, center):
    valid = np.isfinite(samples)
    count = valid.sum(axis=0)
    squared = np.nansum((samples - center) ** 2, axis=0)
    error = np.sqrt(np.maximum(count - 1, 0) / np.maximum(count, 1) * squared)
    error[count < 2] = np.nan
    return error, count


# Adjacent-time logarithmic ratios of principal correlators estimate energy gaps.
def effective_gaps(rho, eta):
    gaps = np.full((len(rho) - 1, 2), np.nan)
    valid = (rho[:-1] > 0.0) & (rho[1:] > 0.0)
    gaps[valid] = np.log(rho[:-1][valid] / rho[1:][valid]) / eta
    return gaps


def analyze_group(key, rows, t0_dt, condition_max):
    lam, beta, nt_float, eta = key
    nt = int(round(nt_float))
    dts, weights, values = make_block_arrays(rows)
    if len(weights) < 3:
        raise ValueError(f"need at least 3 blocks, found {len(weights)}")
    matches = np.flatnonzero(dts == t0_dt)
    if not len(matches):
        raise ValueError(f"t0 dt={t0_dt} absent from input")
    t0_index = int(matches[0])
    result = {"meta": (lam, beta, nt, eta), "dts": dts, "n_blocks": len(weights), "sectors": {}}
    full_values = weighted_average(values, weights)
    for sector in ("odd", "even"):
        full_matrices = connected_matrices(full_values, sector)
        try:
            full_rho, condition = solve_gevp(full_matrices, t0_index, condition_max)
        except np.linalg.LinAlgError as exc:
            print(f"[WARNING] lambda={lam:g} beta={beta:g} Nt={nt} {sector}: {exc}; skipped",
                  file=sys.stderr)
            continue
        jk_rho = np.full((len(weights), len(dts), 2), np.nan)
        for omitted in range(len(weights)):
            sample_values = weighted_average(values, weights, omitted)
            sample_matrices = connected_matrices(sample_values, sector)
            try:
                jk_rho[omitted], _ = solve_gevp(sample_matrices, t0_index, condition_max)
            except np.linalg.LinAlgError as exc:
                print(f"[WARNING] {sector} jackknife omit={omitted}: {exc}; sample skipped",
                      file=sys.stderr)
        rho_err, rho_valid = jackknife_error(jk_rho, full_rho)
        full_gaps = effective_gaps(full_rho, eta)
        jk_gaps = np.array([effective_gaps(sample, eta) for sample in jk_rho])
        gap_err, gap_valid = jackknife_error(jk_gaps, full_gaps)
        result["sectors"][sector] = {
            "rho": full_rho, "rho_err": rho_err, "rho_valid": rho_valid,
            "gaps": full_gaps, "gap_err": gap_err, "gap_valid": gap_valid,
            "condition": condition,
        }
    return result


def write_outputs(results, args):
    for path in (args.principal_out, args.gaps_out, args.summary_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.principal_out.open("w", encoding="utf-8") as stream:
        stream.write("# lambda beta Nt eta sector level dt tau rho rho_err n_blocks n_jk_valid\n")
        for result in results:
            lam, beta, nt, eta = result["meta"]
            for sector, data in result["sectors"].items():
                for t, dt in enumerate(result["dts"]):
                    for level in range(2):
                        stream.write(f"{lam:.17g} {beta:.17g} {nt:d} {eta:.17g} {sector} "
                                     f"{level:d} {dt:d} {dt * eta:.17g} "
                                     f"{data['rho'][t, level]:.17g} {data['rho_err'][t, level]:.17g} "
                                     f"{result['n_blocks']:d} {data['rho_valid'][t, level]:d}\n")
    with args.gaps_out.open("w", encoding="utf-8") as stream:
        stream.write("# lambda beta Nt eta sector level dt tau gap_eff gap_eff_err n_blocks n_jk_valid\n")
        for result in results:
            lam, beta, nt, eta = result["meta"]
            for sector, data in result["sectors"].items():
                for t, dt in enumerate(result["dts"][:-1]):
                    for level in range(2):
                        stream.write(f"{lam:.17g} {beta:.17g} {nt:d} {eta:.17g} {sector} "
                                     f"{level:d} {dt:d} {dt * eta:.17g} "
                                     f"{data['gaps'][t, level]:.17g} {data['gap_err'][t, level]:.17g} "
                                     f"{result['n_blocks']:d} {data['gap_valid'][t, level]:d}\n")
    with args.summary_out.open("w", encoding="utf-8") as stream:
        stream.write("# Anharmonic spectrum GEVP pilot summary\n\n")
        stream.write(f"GEVP reference separation: `t0_dt = {args.t0_dt}`. "
                     "Level assignments are candidates requiring plateau and ED validation.\n\n")
        stream.write("| lambda | beta | Nt | sector | blocks | cond C(t0) | positive rho | finite gaps |\n")
        stream.write("|---:|---:|---:|:---|---:|---:|---:|---:|\n")
        for result in results:
            lam, beta, nt, _ = result["meta"]
            for sector, data in result["sectors"].items():
                positive = int(np.sum(data["rho"] > 0.0))
                total = data["rho"].size
                finite = int(np.isfinite(data["gaps"]).sum())
                stream.write(f"| {lam:g} | {beta:g} | {nt:d} | {sector} | "
                             f"{result['n_blocks']:d} | {data['condition']:.3e} | "
                             f"{positive}/{total} | {finite}/{data['gaps'].size} |\n")
        stream.write("\nCandidate mapping: odd level 0 -> Delta1, odd level 1 -> Delta3; "
                     "even level 0 -> Delta2, even level 1 -> Delta4.\n")


def main():
    args = parse_args()
    if args.t0_dt < 0:
        raise SystemExit("--t0-dt must be non-negative")
    records = load_inputs(args.input)
    results = []
    for key, rows in sorted(group_records(records).items()):
        try:
            result = analyze_group(key, rows, args.t0_dt, args.condition_max)
        except ValueError as exc:
            print(f"[WARNING] group {key}: {exc}; skipped", file=sys.stderr)
            continue
        if result["sectors"]:
            results.append(result)
    if not results:
        raise SystemExit("no numerically valid GEVP groups")
    write_outputs(results, args)
    print(f"wrote {args.principal_out}")
    print(f"wrote {args.gaps_out}")
    print(f"wrote {args.summary_out}")


if __name__ == "__main__":
    main()
