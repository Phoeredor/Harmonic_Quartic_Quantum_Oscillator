#!/usr/bin/env python3
# Assemble the final beta=5 continuum moments for every lambda, combining
# statistical block errors with fit-window systematics and exact references.
"""Build the final beta=5 continuum table for y2 and y4."""

from __future__ import annotations

import math

from quartic_continuum_common import PROCESSED, read_named, write_table


FITS = PROCESSED / "anharmonic_beta5_continuum_v2_fits.dat"
ADAPTIVE_REFS = PROCESSED / "anharmonic_beta5_exact_reference_v2_adaptive.dat"
OUT = PROCESSED / "anharmonic_beta5_continuum_v2_final_table.dat"
COLUMNS = (
    "lambda", "y2_cont", "y2_stat", "y2_sys", "y2_exact", "y2_diff", "y2_z_stat", "y2_z_tot",
    "y4_cont", "y4_stat", "y4_sys", "y4_exact", "y4_diff", "y4_z_stat", "y4_z_tot",
    "ed_cutoff_saturated",
)


# Pair the selected <y^2> and <y^4> intercepts and report their reference deviations.
def main() -> None:
    fits = read_named(FITS)
    adaptive = read_named(ADAPTIVE_REFS) if ADAPTIVE_REFS.is_file() else None
    saturated = {}
    if adaptive is not None:
        saturated = {float(r["lambda"]): str(r["ed_converged"]) for r in adaptive}
    rows = []
    for lam in sorted(float(x) for x in set(fits["lambda"])):
        row = {"lambda": lam}
        for obs in ("y2", "y4"):
            fit = next(f for f in fits if math.isclose(float(f["lambda"]), lam) and str(f["obs"]) == obs)
            cont = float(fit["A"])
            stat = float(fit["A_err"])
            sys = float(fit["sigma_sys"])
            exact = float(fit["exact"])
            diff = cont - exact
            tot = math.sqrt(stat * stat + sys * sys) if math.isfinite(sys) else math.nan
            row[f"{obs}_cont"] = cont
            row[f"{obs}_stat"] = stat
            row[f"{obs}_sys"] = sys
            row[f"{obs}_exact"] = exact
            row[f"{obs}_diff"] = diff
            row[f"{obs}_z_stat"] = diff / stat if stat > 0.0 else math.nan
            row[f"{obs}_z_tot"] = diff / tot if tot > 0.0 else math.nan
        row["ed_cutoff_saturated"] = saturated.get(lam, "not_checked")
        rows.append(row)
    write_table(OUT, COLUMNS, rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
