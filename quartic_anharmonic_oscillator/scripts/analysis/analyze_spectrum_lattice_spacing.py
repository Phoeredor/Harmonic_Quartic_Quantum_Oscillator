#!/usr/bin/env python3
"""Eta check at lambda=0.25 for the robust Delta1 and Delta2 gaps."""

from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_spectrum_coupling_scan as scan  # noqa: E402
import analyze_spectrum_gevp as gevp  # noqa: E402


RAW_DIR = Path("data/raw/spectrum/eta_check_lambda0p25")
ED_PATH = Path("data/processed/spectrum/anharmonic_spectrum_exact_reference_scan_quick.dat")
OUT_DATA = Path("data/processed/spectrum/anharmonic_spectrum_eta_check_lambda0p25.dat")
OUT_SUMMARY = Path("data/processed/spectrum/anharmonic_spectrum_eta_check_lambda0p25_summary.md")


def main():
    paths = sorted(RAW_DIR.glob("*_corr.dat"))
    if not paths:
        raise SystemExit(f"no correlator files found in {RAW_DIR}")
    records = gevp.load_inputs(paths)
    groups = gevp.group_records(records)
    ed = np.loadtxt(ED_PATH, comments="#", ndmin=2)
    ed025 = ed[np.isclose(ed[:, 0], 0.25)]
    if len(ed025) == 0:
        raise SystemExit("missing ED reference for lambda=0.25")
    ed1, ed2 = float(ed025[0, 6]), float(ed025[0, 7])
    rows = []
    for key, group_rows in sorted(groups.items(), key=lambda item: item[0][2]):
        lam, beta, nt_float, eta = key
        if not np.isclose(lam, 0.25):
            continue
        nt = int(round(nt_float))
        t0_dt = max(1, int(round(0.5 / eta)))
        result = scan.analyze_rows(group_rows, t0_dt)
        d1, e1, n1, _ = scan.plateau(result, "Delta1")
        d2, e2, n2, _ = scan.plateau(result, "Delta2")
        rows.append((nt, eta, t0_dt, len(result["weights"]), d1, e1, ed1, d1 - ed1,
                     (d1 - ed1) / e1 if e1 > 0 else np.nan, n1,
                     d2, e2, ed2, d2 - ed2, (d2 - ed2) / e2 if e2 > 0 else np.nan, n2))
    if not rows:
        raise SystemExit("no valid eta-check groups")
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DATA.open("w", encoding="utf-8") as stream:
        stream.write("# Nt eta t0_dt n_blocks Delta1_pimc Delta1_err Delta1_ED Delta1_diff Delta1_z "
                     "Delta1_n_jk_valid Delta2_pimc Delta2_err Delta2_ED Delta2_diff Delta2_z "
                     "Delta2_n_jk_valid\n")
        for row in rows:
            stream.write(" ".join(f"{value:.17g}" if isinstance(value, float) else str(value)
                                  for value in row) + "\n")
    nt400 = [row for row in rows if row[0] == 400]
    max_pair_z = np.nan
    if nt400:
        ref = nt400[0]
        comp = []
        for row in rows:
            if row[0] == 400:
                continue
            z1 = abs((ref[4] - row[4]) / np.sqrt(ref[5] ** 2 + row[5] ** 2))
            z2 = abs((ref[10] - row[10]) / np.sqrt(ref[11] ** 2 + row[11] ** 2))
            comp.extend([z1, z2])
        if comp:
            max_pair_z = float(np.nanmax(comp))
    with OUT_SUMMARY.open("w", encoding="utf-8") as stream:
        stream.write("# Eta check lambda=0.25\n\n")
        stream.write("| Nt | eta | blocks | Delta1 | Delta1 ED | z1 | Delta2 | Delta2 ED | z2 |\n")
        stream.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            stream.write(f"| {row[0]} | {row[1]:.3g} | {row[3]} | "
                         f"{row[4]:.6g} +- {row[5]:.2g} | {row[6]:.6g} | {row[8]:.2f} | "
                         f"{row[10]:.6g} +- {row[11]:.2g} | {row[12]:.6g} | {row[14]:.2f} |\n")
        stream.write("\n")
        if np.isfinite(max_pair_z) and max_pair_z <= 2.0:
            stream.write("Within the current statistical precision, Nt=400 is compatible with the "
                         "coarser/finer eta checks and is adequate for a qualitative quick-scan result.\n")
        else:
            stream.write("The eta check is not conclusive at the 2 sigma level; use Nt=400 as a "
                         "qualitative quick-scan point and avoid overclaiming continuum control.\n")
    print(f"wrote {OUT_DATA}")
    print(f"wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
