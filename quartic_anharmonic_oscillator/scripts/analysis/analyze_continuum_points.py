#!/usr/bin/env python3
# Reduce beta=5 measurement chains to blocked estimates at each (lambda,Nt).
# The output contains moments, potential and virial kinetic terms, and the
# renormalized lattice kinetic estimator needed for continuum analysis.
"""Analyze final beta=5 production measurements point by point."""

from __future__ import annotations

from quartic_continuum_common import (
    MEAS_COLUMNS, PROCESSED, blocked_mean_error, is_excluded_eta_nt,
    is_required_measurement, load_measurement_columns, parse_measurement_metadata,
    raw_measurement_files, write_table,
)


OUT = PROCESSED / "anharmonic_beta5_continuum_v2_points.dat"
COLUMNS = (
    "lambda", "beta", "Nt", "eta", "eta2", "n_measurements",
    "block_size_rows", "n_blocks", "y2", "y2_err", "y4", "y4_err",
    "V_eta", "V_eta_err", "K_vir_eta", "K_vir_eta_err",
    "K_ren_eta", "K_ren_eta_err", "raw_file",
)


# Build one statistically blocked row for every retained lattice ensemble.
def main() -> None:
    col = {name: i for i, name in enumerate(MEAS_COLUMNS)}
    rows = []
    for raw in raw_measurement_files():
        item = parse_measurement_metadata(raw)
        if float(item["lambda"]) <= 0.0:
            continue
        if is_excluded_eta_nt(float(item["eta"]), int(item["Nt"])):
            continue
        if not is_required_measurement(float(item["lambda"]), int(item["Nt"])):
            continue
        data = load_measurement_columns(raw)
        block = int(item["block_size_rows"])
        row = {
            "lambda": float(item["lambda"]),
            "beta": float(item["beta"]),
            "Nt": int(item["Nt"]),
            "eta": float(item["eta"]),
            "eta2": float(item["eta"]) ** 2,
            "n_measurements": int(data.shape[0]),
            "block_size_rows": block,
            "raw_file": str(raw),
        }
        n_blocks_seen = 0
        for source, dest in (("y2_mean", "y2"), ("y4_mean", "y4")):
            mean, err, _, n_blocks = blocked_mean_error(data[:, col[source]], block)
            row[dest] = mean
            row[f"{dest}_err"] = err
            n_blocks_seen = max(n_blocks_seen, n_blocks)
        eta = float(item["eta"])
        lam = float(item["lambda"])
        y2_sample = data[:, col["y2_mean"]]
        y4_sample = data[:, col["y4_mean"]]
        dy2_sample = data[:, col["dy2_mean"]]
        v_eta_sample = 0.5 * y2_sample + lam * y4_sample
        k_vir_eta_sample = 0.5 * y2_sample + 2.0 * lam * y4_sample
        # The explicit 1/(2 eta) term cancels the short-distance path divergence.
        k_ren_eta_sample = -dy2_sample / (2.0 * eta * eta) + 1.0 / (2.0 * eta)
        for dest, sample in (
            ("V_eta", v_eta_sample),
            ("K_vir_eta", k_vir_eta_sample),
            ("K_ren_eta", k_ren_eta_sample),
        ):
            mean, err, _, n_blocks = blocked_mean_error(sample, block)
            row[dest] = mean
            row[f"{dest}_err"] = err
            n_blocks_seen = max(n_blocks_seen, n_blocks)
        row["n_blocks"] = n_blocks_seen
        rows.append(row)
    rows.sort(key=lambda r: (float(r["lambda"]), int(r["Nt"])))
    write_table(
        OUT, COLUMNS, rows,
        int_columns={"Nt", "n_measurements", "block_size_rows", "n_blocks"},
    )
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
