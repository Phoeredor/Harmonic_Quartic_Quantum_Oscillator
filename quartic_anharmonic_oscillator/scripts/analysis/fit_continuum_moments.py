#!/usr/bin/env python3
# Select eta ranges for beta=5 continuum fits of <y^2> and <y^4>.  Weighted
# O(eta)=A+B*eta^2 fits provide continuum intercepts and fit-window systematics
# for each quartic coupling lambda.
"""Select final beta=5 continuum fit windows for y2 and y4."""

from __future__ import annotations

import math

import numpy as np

from quartic_continuum_common import (
    DEFAULT_ETA_MAX, DEFAULT_ETA_MIN, PROCESSED, exact_map, read_named,
    weighted_linear_fit, write_table,
)


POINTS = PROCESSED / "anharmonic_beta5_continuum_v2_points.dat"
REFS = PROCESSED / "anharmonic_beta5_exact_reference_v2_adaptive.dat"
FITS = PROCESSED / "anharmonic_beta5_continuum_v2_fits.dat"
OBSERVABLES = (("y2", "y2_err"), ("y4", "y4_err"))
TARGET_LOW = 0.5
TARGET_HIGH = 1.0
TARGET_CENTER = 0.8
MIN_POINTS = 5
COLUMNS = (
    "lambda", "obs", "eta_min", "eta_max", "n_points",
    "A", "A_err", "B", "B_err", "chi2", "dof", "chi2_red",
    "selection_status", "sigma_sys", "n_target_windows", "exact",
    "A_minus_exact", "z_A_stat",
)


# Fit one observable over a specified lattice-spacing interval; A is its eta=0 limit.
def fit_subset(sub: np.ndarray, obs: str, err_col: str, exact: float, eta_min: float, eta_max: float) -> dict[str, float | str]:
    fit = weighted_linear_fit(np.asarray(sub["eta2"], dtype=float), np.asarray(sub[obs], dtype=float), np.asarray(sub[err_col], dtype=float))
    diff = fit["A"] - exact
    return {
        "lambda": float(sub["lambda"][0]),
        "obs": obs,
        "eta_min": eta_min,
        "eta_max": eta_max,
        "n_points": int(sub.size),
        "A": fit["A"],
        "A_err": fit["A_error"],
        "B": fit["B"],
        "B_err": fit["B_error"],
        "chi2": fit["chi2"],
        "dof": int(fit["dof"]),
        "chi2_red": fit["chi2_red"],
        "exact": exact,
        "A_minus_exact": diff,
        "z_A_stat": diff / fit["A_error"] if fit["A_error"] > 0.0 else math.nan,
    }


# Enumerate contiguous eta windows with enough points to constrain A and B.
def contiguous_windows(lam_points: np.ndarray, obs: str, err_col: str, exact: float) -> list[dict[str, float | str]]:
    windows: list[dict[str, float | str]] = []
    etas = sorted(float(x) for x in np.unique(lam_points["eta"]) if DEFAULT_ETA_MIN <= float(x) <= DEFAULT_ETA_MAX)
    for i, eta_min in enumerate(etas):
        for eta_max in etas[i:]:
            mask = (lam_points["eta"] >= eta_min - 1e-15) & (lam_points["eta"] <= eta_max + 1e-15)
            sub = lam_points[mask]
            if sub.size < MIN_POINTS:
                continue
            windows.append(fit_subset(sub, obs, err_col, exact, eta_min, eta_max))
    return windows


# Prefer broad statistically acceptable windows and use intercept spread as a systematic.
def select_window(windows: list[dict[str, float | str]]) -> dict[str, float | str]:
    target = [w for w in windows if TARGET_LOW <= float(w["chi2_red"]) < TARGET_HIGH]
    if target:
        selected = sorted(
            target,
            key=lambda w: (-int(w["n_points"]), abs(float(w["chi2_red"]) - TARGET_CENTER), float(w["eta_min"])),
        )[0]
        status = "target"
    else:
        selected = sorted(
            windows,
            key=lambda w: (abs(float(w["chi2_red"]) - TARGET_CENTER), -int(w["n_points"]), float(w["eta_min"])),
        )[0]
        status = "fallback"
    sys_pool = target if target else windows
    sigma = max((abs(float(w["A"]) - float(selected["A"])) for w in sys_pool), default=0.0)
    row = dict(selected)
    row["selection_status"] = status
    row["sigma_sys"] = sigma
    row["n_target_windows"] = len(target)
    return row


def select_fits(points: np.ndarray, refs: np.ndarray) -> list[dict[str, float | str]]:
    if np.any(np.isclose(points["eta"], 0.125)) or np.any(points["Nt"] == 40):
        raise RuntimeError("filtered points still contain eta=0.125 or Nt=40")
    ref_maps = {obs: exact_map(refs, obs) for obs, _ in OBSERVABLES}
    rows: list[dict[str, float | str]] = []
    for lam in sorted(float(x) for x in np.unique(points["lambda"])):
        lam_points = points[np.isclose(points["lambda"], lam)]
        for obs, err_col in OBSERVABLES:
            windows = contiguous_windows(lam_points, obs, err_col, ref_maps[obs][lam])
            if not windows:
                raise RuntimeError(f"no candidate windows for lambda={lam:g}, obs={obs}")
            rows.append(select_window(windows))
    return rows


def main() -> None:
    points = read_named(POINTS)
    refs = read_named(REFS)
    rows = select_fits(points, refs)
    write_table(FITS, COLUMNS, rows, int_columns={"n_points", "dof", "n_target_windows"})
    print(f"using ED reference {REFS}")
    print(f"wrote {FITS} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
